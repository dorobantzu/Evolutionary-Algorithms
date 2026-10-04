"""EC A2 template code - neuroevolution for targeted locomotion with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo*, not a solution. It spawns a robot, drives it with a neural network
whose weights are RANDOM, runs the simulation, and reports how close the robot
ended up to a target.

There is deliberately NO evolution in here. Building the EA (representation,
initialisation, parent selection, variation, survivor selection) is the assignment.
See "YOUR JOB" at the bottom of this file.

THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve the weights of a neural network controller so that a robot moves from
SPAWN_POS to TARGET_POSITION within the simulation time.

    fitness = distance between the robot's final position and TARGET_POSITION

HOW TO RUN
----------

Change MODE below to switch between an interactive viewer, a headless run,
a rendered video, or a single frame.
"""

# Standard library
import csv
import os
import random
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from typing import cast

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.ec import set_seed
from ariel.ec.generators import FloatMutator
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.targeted_locomotion import fitness_delta_distance
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel.ec import (
    EA,
    EAOperation,
    Individual,
    Population,
    config,
    Crossover
)

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
SEED = int(os.environ.get("SEED", 42))
RNG = np.random.default_rng(SEED)
random.seed(SEED)  # Population.shuffle() uses stdlib random; set_seed() doesn't cover it

# ariel.ec's own generators/mutators/crossover draw from a separate,
# package-level RNG. Reseed it too if you build your EA on ariel.ec,
# or every one of your "multiple seeds" runs the same variation operators.
set_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]  # where it should end up
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "simple"  # see run_experiment() for the options
SCENARIO = os.environ.get("SCENARIO", "random")

# Machine the run happens on, saved with each runtime. An explicit MACHINE
# value is useful for lab machines; otherwise use the current hostname.
MACHINE = os.environ.get("MACHINE", socket.gethostname())

# Keep these values fixed for every scenario in the final experiment. The
# scenario-specific tournament sizes are part of the experimental condition.
POPULATION_SIZE = 100
NUM_STEPS = 300
TOURNAMENT_K = {"ea1": 2, "ea2": 7}
# Each EA evaluates the initial population plus 50 offspring per generation.
EVALUATION_BUDGET = POPULATION_SIZE + NUM_STEPS * (POPULATION_SIZE // 2)


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
def build_world() -> SimpleFlatWorld:
    """Create the environment the robot lives in.

    YOU MAY CHANGE THIS. Options include: SimpleFlatWorld, RuggedTerrainWorld,
    CraterTerrainWorld, AmphitheatreTerrainWorld, OlympicArena, ...
    (SimpleTiltedWorld is not supported for this task.)

    Whatever you pick, keep it FIXED for all runs you compare against each
    other, and say in your report which one you used. A controller evolved on
    flat ground and one evolved on rugged terrain are not comparable numbers.
    """
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    """Create the robot body.

    YOU MAY CHANGE THIS. Options include the prebuilt bodies in
    `ariel.body_phenotypes.robogen_lite.prebuilt_robots` (gecko, spider, ...).

    Two consequences of this choice, and they matter:
      * The body determines `model.nu` (the number of hinges you must send
        commands to) - that is the OUTPUT size of your controller.
      * The body determines the size of `data.qpos` - if you feed qpos to your
        network, that is (part of) your INPUT size.
    Change the body and your genotype length changes with it. Keep the body
    FIXED within an experiment.
    """
    return gecko()


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
#
# MuJoCo calls the controller every physics step with (model, data); its job
# is to write into data.ctrl.
#
#   INPUTS   : whatever you read from `data` (qpos, qvel, time, ...), plus any
#              task info you already know, e.g. the vector to TARGET_POSITION.
#              INPUT SIZE is your choice, but must stay CONSTANT.
#   OUTPUTS  : exactly `model.nu` values, one per actuated hinge.
#   RANGE    : hinges accept [-pi/2, +pi/2] radians. A tanh output gives
#              [-1, 1] - rescale: actions * (np.pi / 2).
#   WRITING  : DIRECT (data.ctrl[:] = actions) commands the angle straight -
#              fast, but can destabilise the sim on large jumps. DELTA
#              (data.ctrl[:] += actions * alpha, alpha ~ 0.05, then clip) is
#              smoother but accumulates, so clipping is required. Pick one,
#              justify it, use it everywhere.
#   NaN      : blown-up weights silently write NaN into data.ctrl. Assert
#              against it while developing.
#
# ============================================================================ #

# Controller architecture - decide before writing your EA.
HIDDEN_SIZE: int = 6


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Map robot state to hinge commands: in -> hidden -> actions.

    In this demo `weights` is drawn at RANDOM. In your assignment, `weights`
    is what the evolutionary algorithm produces: an individual's genotype,
    reshaped into these matrices. You are free to change the architecture
    itself (layers, activations, ...) - just keep input/output sizes correct.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model. Use `model.nu` for the number of hinges.
    data : mj.MjData
        The MuJoCo data. This is where you read the robot's state from.
    weights : list of ndarray
        [w1, w2] - the layer weight matrices.

    Returns
    -------
    npt.NDArray[np.float64]
        `model.nu` action values, already scaled to [-pi/2, pi/2].
    """
    w1, w2 = weights

    # --- INPUTS ---------------------------------------------------------- #
    # Bare qpos - the simplest choice, not necessarily a good one. See
    # YOUR JOB below.
    inputs = data.qpos

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]

def make_individual(weights: list[npt.NDArray[np.float64]]) -> Individual:
    """Create an Individual from a list of weight matrices.

    This is a convenience function for your EA. It is not used in the demo.
    """
    ind = Individual()
    ind.genotype = np.concatenate([w.ravel() for w in weights]).tolist()
    return ind

def decode_genotype(
    genotype: list[float],
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Reshape a flat genotype back into the [w1, w2] matrices of `nn_controller`."""
    flat = np.asarray(genotype, dtype=np.float64)
    split = input_size * HIDDEN_SIZE
    expected = split + HIDDEN_SIZE * output_size
    if flat.size != expected:
        raise ValueError(f"genotype length {flat.size} != expected {expected}")
    w1 = flat[:split].reshape(input_size, HIDDEN_SIZE)
    w2 = flat[split:].reshape(HIDDEN_SIZE, output_size)
    return [w1, w2]

def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    Example output:
    [
            array([[ 0.29, -0.13,  0.55, ...],   # shape (8, 6)
                   [-0.42,  0.11,  0.03, ...],
                   ...
                   [ 0.18, -0.07,  0.24, ...]]),  # 8 rows × 6 cols

            array([[ 0.31, -0.22, -0.09,  0.14],  # shape (6, 4)
                   [-0.05,  0.38,  0.17, -0.29],
                   ...
                   [ 0.09, -0.16,  0.21,  0.04]]),  # 6 rows × 4 cols
        ]
    """
        
    return [
        RNG.normal(scale=0.5, size=(input_size, HIDDEN_SIZE)),
        RNG.normal(scale=0.5, size=(HIDDEN_SIZE, output_size)),
    ]


# ============================================================================ #
#  3. POSITION AND FITNESS
# ============================================================================ #
#
# The robot is spawned with a free joint, so data.qpos[0:3] IS the core's
# (x, y, z) world position. Read it before and after stepping - no tracker or
# bookkeeping needed. (`data.geom("robot1_core").xpos` works too.)
#
# ============================================================================ #


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    The plain version: how far is the robot from the target when time runs out?

    `initial_position` is unused here on purpose - it is passed in because the
    moment you want a less naive fitness you will need it. Some things worth
    thinking about (and, ideally, comparing in your report):
      * Distance *reduced* rather than distance remaining, so a robot that
        starts closer is not rewarded for standing still.
      * Penalising a robot that falls over or leaves the arena.
      * Whether the z-axis should count at all - a robot that jumps is not
        closer to the target in any way you care about.
    See `ariel.simulation.tasks.targeted_locomotion` for some worked variants.
    """
    target: npt.NDArray[Any] = np.asarray(TARGET_POSITION)
    #CHANGED FITNESS FUNCTION TO REDUCED DISTANCE
    return fitness_delta_distance(initial_position, final_position, target)


# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #




def run_experiment(mode: ViewerTypes = MODE, genotype: list[float] | None = None) -> float:
    """Set up the world, run one simulation, and return the fitness.

    This is the function your EA calls once per individual, with `mode` set
    to "simple" (headless).

    Returns
    -------
    float
        The fitness of this run. Lower is better.
    """
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    # --- World and robot --------------------------------------------------- #
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size: int = len(data.qpos)
    output_size: int = model.nu

    if genotype is None: #random scenario doesn't pass genotype, so generate random weights for that case
        weights = make_random_weights(input_size, output_size)
    else:
        weights = decode_genotype(genotype, input_size, output_size)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights)

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
    initial_position = get_core_position(data)

    # --- Run ---------------------------------------------------------------- #
    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "launcher":
            # Interactive window. Great for seeing what your robot does,
            # useless inside an evolutionary loop.
            viewer.launch(model=model, data=data)
        case "simple":
            # Headless. THIS is the one your EA uses.
            simple_runner(model, data, duration=SIM_DURATION)
        case "video":
            # Render to an mp4 - for the figures in your report.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "frame":
            # A single image of the scene. Useful to check your spawn position
            # and that the robot is not clipping through the floor.
            single_frame_renderer(model, data, steps=1, show=True)
        case "no_control":
            # No controller attached: drag the hinges around by hand.
            viewer.launch(model=model, data=data)

    # Detach the callback again so the next run starts clean.
    mj.set_mjcb_control(None)

    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    fitness = fitness_function(initial_position, final_position)

    console.log(f"start  : {np.round(initial_position, 3)}")
    console.log(f"end    : {np.round(final_position, 3)}")
    console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness

def parent_selection(population: Population, k: int = 2) -> Population:
    '''Implements tournament selection to select parents for crossover.
        Parent count is population size / 2.
    '''
    for ind in population:
        ind.tags = {"selected": False}
    selected_count = 0
    while selected_count < config.target_population_size / 2:
        shuffled = population.where(lambda ind: not bool(ind.tags.get("selected", False))).shuffle()
        contenders = shuffled[:k]
        winner = contenders.best(sort="min", attribute="fitness_", n=1)[0]
        winner.tags = {"selected": True}
        selected_count += 1
    shuffled = population.shuffle()
    return shuffled
    
def crossover(population: Population) -> Population:
    '''Implements uniform crossover to generate offspring from selected parents.'''
    selected_parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))
    assert len(selected_parents) % 2 == 0, "Number of selected parents must be even for crossover."
    assert len(selected_parents) == config.target_population_size / 2, "Number of selected parents must be equal to half the target population size for crossover."
    for idx in range(0, len(selected_parents) - 1, 2):
        parent_a = selected_parents[idx]
        parent_b = selected_parents[idx + 1]
        g_a = parent_a.genotype
        g_b = parent_b.genotype
        child_a = Individual()
        child_b = Individual()
        child_a.genotype, child_b.genotype = Crossover.uniform(
            cast("list[float]", g_a),
            cast("list[float]", g_b),
            swap_probability=0.5 #TODO change if needed, justfiy from literature
        )
        child_a.tags = {"mutate": True} #ONLY CHILDREN GET MUTATED?
        child_b.tags = {"mutate": True}
        population.extend([child_a, child_b])
    return population

def mutate(population: Population) -> Population:
    '''Implements gaussian mutation to introduce variation in the offspring.'''
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
        #TODO why gaussian? justify from literature
        ind.genotype = FloatMutator.gaussian(
            individual=cast("list[float]", ind.genotype),
            std=0.2, #TODO why std 0.2? justify from literature
            mutation_probability=0.2 # TODO why mutation probability 0.2? justify from literature
        )
        ind.requires_eval = True
        ind.tags = {"mutate": False} #reset mutate tag after mutation
    return population

evaluation_count: int = 0  # simulations run so far, saved with the runtime

def evaluate(population: Population) -> Population:
    '''Evaluates the fitness of individuals in the population.'''
    global evaluation_count
    for ind in population.unevaluated:
        evaluation_count += 1
        assert isinstance(ind.genotype, list) and all(isinstance(g, float) for g in ind.genotype), "Genotype must be a list of floats."
        fitness = run_experiment(MODE, ind.genotype)
        ind.fitness = fitness
    return population

def survivor_selection_elitism(population: Population) -> Population:
    '''Selects the best individuals to survive to the next generation.'''
    sorted_population = population.alive.best(sort="min", attribute="fitness_", n=population.size)
    for ind in sorted_population[config.target_population_size:]:
        ind.alive = False
    assert sorted_population.alive.size == config.target_population_size
    return sorted_population

def survivor_selection_total_replace(population: Population) -> Population:
    '''Replace children with parents regardless of fitness.'''
    parents = population.where(lambda ind: ind.tags.get("selected", False))
    for p in parents:
        p.alive = False
    assert population.alive.size == config.target_population_size
    return population

def save_runtime(start: datetime, end: datetime, duration_s: float) -> None:
    '''Write this run's timing to __data__/runtimes.csv, replacing an earlier row for the same scenario, seed and machine.'''
    path = config.output_folder / "runtimes.csv"
    row = {
        "scenario": SCENARIO,
        "seed": str(SEED),
        "machine": MACHINE,
        "population_size": str(config.target_population_size),
        "generations": str(0 if SCENARIO == "random" else config.num_steps),
        "evaluations": str(evaluation_count),
        "start": start.isoformat(timespec="seconds"),
        "end": end.isoformat(timespec="seconds"),
        "duration_s": f"{duration_s:.3f}",
    }
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="") as fh:
            rows = [
                r for r in csv.DictReader(fh)
                if (r["scenario"], r["seed"], r["machine"]) != (row["scenario"], row["seed"], row["machine"])
            ]
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    console.log(f"runtime: {duration_s:.1f} s ({start:%H:%M:%S} -> {end:%H:%M:%S}) saved to {path}")

# IMPORTANT: DO NOT RUN SCENARIOS IN PARALLEL WHEN MEASURING RUNTIME. PARALLEL RUNS
# SHARE THE CPU AND SLOW EACH OTHER DOWN, SO THEIR RUNTIMES ARE NOT COMPARABLE.
# RUN THEM ONE AFTER ANOTHER, OR STATE IN THE REPORT WHICH RUNS WERE PARALLEL.
def main() -> None:
    """Run a single demo evaluation with a randomly-weighted controller."""
    start = datetime.now()
    t0 = time.perf_counter()

    # A quick look at the size of the problem you are about to search.
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(data.qpos)
    output_size = model.nu
    num_weights = (
        input_size * HIDDEN_SIZE
        + HIDDEN_SIZE * output_size
    )
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")
    console.log(f"genotype length (total weights)    : {num_weights}")

    config.target_population_size = POPULATION_SIZE
    config.num_steps = NUM_STEPS
    config.is_maximisation = False
    tournament_k = TOURNAMENT_K.get(SCENARIO)
    k_label = str(tournament_k) if tournament_k is not None else "na"
    generations = 0 if SCENARIO == "random" else config.num_steps
    config.db_file_name = (
        f"{SCENARIO}_pop{config.target_population_size}_steps{generations}"
        f"_k{k_label}_eval{EVALUATION_BUDGET}_seed{SEED}.db"
    )
    match SCENARIO:
        case "random":
            # Match the EA budget exactly: 100 initial evaluations + 300 × 50
            # independently sampled controllers. No selection or variation is used.
            population: Population = Population([
                make_individual(make_random_weights(input_size, output_size))
                for _ in range(EVALUATION_BUDGET)
            ])
            initial: Population = evaluate(population)
            # Zero steps: the EA only commits the evaluated random individuals to the database.
            EA(initial, [], num_steps=0)

        case "ea1":
            population: Population = Population([make_individual(make_random_weights(input_size, output_size)) for _ in range(config.target_population_size)])
            #init eval
            initial: Population = evaluate(population)
            ops: list[EAOperation] = [
                EAOperation(parent_selection)(k=TOURNAMENT_K["ea1"]),
                EAOperation(crossover),
                EAOperation(mutate),
                EAOperation(evaluate),
                EAOperation(survivor_selection_elitism),
            ]
            ea = EA(initial, ops, num_steps=config.num_steps) #justfiy num_steps from literature
            ea.run()
        case "ea2":
            population: Population = Population([make_individual(make_random_weights(input_size, output_size)) for _ in range(config.target_population_size)])
            #init eval
            initial: Population = evaluate(population)
            ops: list[EAOperation] = [
                EAOperation(parent_selection)(k=TOURNAMENT_K["ea2"]),
                EAOperation(crossover),
                EAOperation(mutate),
                EAOperation(evaluate),
                EAOperation(survivor_selection_elitism),
            ]
            ea = EA(initial, ops, num_steps=config.num_steps) #justfiy num_steps from literature
            ea.run()

        case _:
            raise ValueError(f"invalid SCENARIO: {SCENARIO!r}. Valid options: {'random', 'ea1', 'ea2'}")

    save_runtime(start, datetime.now(), time.perf_counter() - t0)

    # MACHINE=atilla_linux SCENARIO=random uv run assignments/assignment_2/A2_template_2026.py
    # MACHINE=atilla_windows SCENARIO=random uv run assignments/assignment_2/A2_template_2026.py
    # MACHINE=atilla_linux SCENARIO=ea1 uv run assignments/assignment_2/A2_template_2026.py
    # MACHINE=atilla_linux SCENARIO=ea2 uv run assignments/assignment_2/A2_template_2026.py
    



if __name__ == "__main__":
    main()


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above runs one robot with random weights. It will score badly, and
# it will score badly in a slightly different way every time you change SEED.
# Your task is to replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing. Repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline. Random search with the same evaluation
#     budget is a simple, but reasonable choice; and it is nearly free to run.
#   Keep body, world, SIM_DURATION and fitness function identical across
#     everything you compare. Change one thing at a time.
#
# ============================================================================ #
