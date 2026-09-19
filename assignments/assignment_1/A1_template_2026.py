"""EC A1 template code - evolving robot morphologies with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo* file for starting you out with assignment 1. 
It samples one body at random, decodes it, scores it
against a set of target bodies, and shows you the result. 

*Your Job* section at the bottom of this file summarises the programming task. Full assignment description can be found in the pdf file on Canvas.


THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve a robot BODY that is as structurally close as possible to a whole set
of given target bodies at once.

    fitness = mean tree edit distance to every body in TARGET_DIR,
              plus one standard deviation across those per-target distances
"""


# Standard library
import random
from pathlib import Path
from typing import Literal
from typing import cast

# Third-party libraries
import mujoco as mj
import networkx as nx
import numpy as np
import torch
from mujoco import viewer

# Local scripts
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    Individual,
    FloatMutator,
    Population,
    config,
)
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type GenotypeTypes = Literal["nde", "tree"]
type ViewerTypes = Literal["launcher", "video", "frame", "none"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
# NOTE: the tree operators use the `random` module, the NDE uses numpy for its
# own genotype vectors AND is a torch.nn.Module for its internal network - that
# network's weight initialisation uses torch's own RNG, entirely separate from
# numpy/random. If you're using "nde", seed all THREE or your runs will not be
# reproducible across separate script runs, even with the same seed value.
SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
TARGET_DIR: Path = HERE / "target_bodies"  # the bodies you must approach
NUM_OF_MODULES: int = 20  # module budget per evolved body
GENOTYPE: GenotypeTypes = "nde"  # "nde" | "tree" 
MODE: ViewerTypes = "frame"  # see show_body() for the options
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]


# ============================================================================ #
#  1. THE TARGET BODIES
# ============================================================================ #
#
# The targets are plain nx.DiGraph JSON files.
# They vary in size on purpose. A body that just matches the average module
# count will not score well against all of them.
#
# ============================================================================ #


def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory.

    Returns
    -------
    list of nx.DiGraph
        One graph per JSON file, sorted by filename.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


# ============================================================================ #
#  2. THE GENOTYPE CONTRACT
# ============================================================================ #
#
# You may use EITHER of ARIEL's two body encodings below. You may NOT invent
# your own, and CPPN is not offered for this assignment.
# Whichever you pick, the contract is the same and it is very short:
#
#       your genotype  --(its decoder)-->  nx.DiGraph  -->  fitness
#
# That DiGraph is the phenotype, and it is all the fitness function ever sees:
#
#       nodes carry   type      : "CORE" | "BRICK" | "HINGE"
#                     rotation  : "DEG_0" | "DEG_45" | "DEG_90"
#       edges carry   face      : "FRONT" | "BACK" | "RIGHT" | "LEFT"
#                                 | "TOP" | "BOTTOM"
#
# THE TWO ENCODINGS
#
#   "nde"   NeuralDevelopmentalEncoding + HighProbabilityDecoder
#           Genotype: three fixed-length float vectors (type / connection /
#           rotation genes). An INDIRECT encoding - a small vector is expanded
#           by a fixed neural network into probability matrices, which are
#           then decoded greedily into a body.
#           -> Fixed-length real vector. Standard real-valued operators work
#              out of the box. But the genotype-phenotype map is wildly
#              non-linear: a small mutation can rebuild the robot entirely.
#           -> IMPORTANT: `NeuralDevelopmentalEncoding`'s internal network is
#              randomly (re-)initialised every time you construct it, and NOT
#              derived from the genotype you pass in. If your EA's decode step
#              builds a fresh `NeuralDevelopmentalEncoding(...)` per individual
#              (the natural way to write it - see `random_nde_body` below),
#              the SAME genotype decodes to a DIFFERENT random body every call,
#              and fitness stops reflecting the genotype at all. Construct it
#              ONCE for your whole run and reuse that one instance's
#              `.forward()` for every genotype you decode.
#           -> ALSO IMPORTANT: `NeuralDevelopmentalEncoding` is a
#              `torch.nn.Module`. Its weight initialisation uses torch's own
#              RNG, entirely separate from numpy/random. `np.random.seed(...)`
#              and `random.seed(...)` do NOT control it - you also need
#              `torch.manual_seed(...)`, or your results will not reproduce
#              across separate runs even with "the same" seed.
#
#   "tree"  TreeGenome + its operators
#           Genotype: the tree itself, nodes and edges.
#           A DIRECT encoding - genotype and phenotype are the same shape.
#           -> ariel.ec.genotypes.tree.operators already gives you
#              random_tree, add_node, remove_subtree, subtree_swap,
#              crossover_subtree, mutate_hoist, mutate_shrink,
#              mutate_replace_node, mutate_subtree_replacement.
#              Variable-length genotype, so watch for bloat.
#
#
# Below, each encoding gets ONE random genotype, decoded to a graph. That is
# your starting point, not your solution: your EA has to search this space,
# not sample it once.
#
# ============================================================================ #

# NDE settings
GENOTYPE_SIZE: int = 64  # length of each of the three NDE gene vectors


# Constructed ONCE, at import time, and reused for every decode call below and
# in your own EA. See the "IMPORTANT" note on "nde" in THE GENOTYPE CONTRACT
# above: rebuilding this per individual silently breaks the genotype -> body
# mapping, because its internal network randomises on construction.
_NDE = NeuralDevelopmentalEncoding(
    number_of_modules=NUM_OF_MODULES,
    genotype_size=GENOTYPE_SIZE,
)


def random_nde_body(num_modules: int = NUM_OF_MODULES) -> nx.DiGraph:
    """Sample a random NDE genotype and decode it into a body graph.

    THIS IS THE FUNCTION YOUR EA REPLACES. The three vectors below are the
    genotype: that is what you mutate, recombine and select on. Note this
    function does NOT construct its own `NeuralDevelopmentalEncoding` - it
    reuses the module-level `_NDE` instance. Do the same in your EA.

    `num_modules` must match the value `_NDE` was built with (NUM_OF_MODULES).
    """
    genotype = random_nde_genotype()
    type_p, conn_p, rot_p = _NDE.forward(genotype)

    decoder = HighProbabilityDecoder(num_modules)
    return decoder.probability_matrices_to_graph(type_p, conn_p, rot_p)

def random_nde_genotype():
    """Sample a random NDE genotype and decode it into a body graph.

    THIS IS THE FUNCTION YOUR EA REPLACES. The three vectors below are the
    genotype: that is what you mutate, recombine and select on. Note this
    function does NOT construct its own `NeuralDevelopmentalEncoding` - it
    reuses the module-level `_NDE` instance. Do the same in your EA.

    `num_modules` must match the value `_NDE` was built with (NUM_OF_MODULES).
    """
    genotype = [
        RNG.uniform(-1.0, 1.0, GENOTYPE_SIZE).astype(np.float32)  # module types
        for _ in range(3)  # types, connections, rotations
    ]
    return genotype


def random_tree_body(num_modules: int = NUM_OF_MODULES) -> nx.DiGraph:
    """Sample a random tree genotype and convert it into a body graph.

    THIS IS THE FUNCTION YOUR EA REPLACES. Here the genotype IS the tree, so
    `TreeGenome` is what your population holds - call `.to_networkx()` only
    when it is time to compute fitness.
    """
    genome = random_tree(max_modules=num_modules)
    return genome.to_networkx()

def random_body(
    genotype: GenotypeTypes = GENOTYPE,
    num_modules: int = NUM_OF_MODULES,
) -> nx.DiGraph:
    """Sample one random body using the chosen encoding."""
    match genotype:
        case "nde":
            return random_nde_body(num_modules)
        case "tree":
            return random_tree_body(num_modules)

def make_individual() -> Individual:
    ind = Individual()
    ind.genotype = random_nde_genotype()
    return ind
# ============================================================================ #
#  3. FITNESS
# ============================================================================ #
#
# Fitness is the MEAN tree edit distance to every target body, PLUS one
# standard deviation across those per-target distances. LOWER IS BETTER, and
# 0.0 would mean your body is identical to all of them at once - which, since
# the targets differ from each other, is impossible. There is a floor above
# zero here and you will not reach it. Work out roughly where it is: a body
# cannot be closer to a set than the set is to itself.
#
# The distance itself lives in tree_edit_distance.py.
# Read that file - you cannot reason about your EA's behaviour without knowing what it is climbing.
#
# ============================================================================ #


def fitness_function(
    body: nx.DiGraph,
    targets: list[nx.DiGraph],
) -> float:
    """Score one body against the whole target set. LOWER IS BETTER.

    Some things worth thinking about:
      * The std term charges for unevenness - body that is mediocre against every target
        and one that is excellent on most but bad on one can still land close
        in fitness, but the latter is penalized a bit more.
      * Nothing here rewards small bodies. Does your EA bloat? Should a size
        penalty be part of fitness, or is that the encoding's job?
    """
    return mean_plus_std_tree_edit_distance(body, targets)

def evaluate(
    population: Population
    ) -> Population:
    targets = load_targets()
    for ind in population.unevaluated:
        phenotype = ind.genotype
        ind.fitness = fitness_function(ind.genotype, targets)
    return population
# ============================================================================ #
#  4. LOOKING AT A BODY
# ============================================================================ #

def parent_selection(population: Population) -> Population:
    sorted: Population = population.best(n=population.size)
    parent_count: int = population.size // 2
    for p in sorted[:parent_count]:
        p.tags = {"selected": True}

    return sorted

def crossover(population: Population) -> Population:
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))
    # Sliding window: (p0,p1), (p1,p2), ..., (p_last,p0)
    for idx in range(0, len(parents) - 1, 2):
            p_a = parents[idx]
            p_b = parents[idx + 1]
            og_shape, p_a_flat, p_b_flat = Crossover._load(p_a.genotype, p_b.genotype)
            g_a, g_b = Crossover.one_point(
                p_a_flat,
                p_b_flat,
            )
            child_a = Individual()
            child_a.genotype = g_a
            child_a.tags = {"mutate": True}
    
            child_b = Individual()
            child_b.genotype = g_b
            child_b.tags = {"mutate": True}

            population.extend([child_a, child_b])
    return population

def mutate(population: Population) -> Population:
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
        ind.genotype = FloatMutator.uniform_reset(
            individual=cast("list[float]", ind.genotype),
            low=-1.0,
            high=1.0,
            mutation_probability=0.1,
        )
        ind.requires_eval = True
    return population

def survivor_selection(population: Population) -> Population:
    """Select the survivors of the current generation."""
    sorted: Population = population.best(n=population.size)
    return sorted[:config.target_population_size] #only the best survive

def show_body(
    body: nx.DiGraph,
    mode: ViewerTypes = MODE,
    file_name: str = "body",
) -> None:
    """Build a body graph in MuJoCo and look at it.

    There is no controller and no physics worth speaking of - this exists so
    you can SEE what your fitness function is actually rewarding. Do this
    early and often. A number going down is not evidence that the bodies look
    anything like the targets.
    """
    if mode == "none":
        return

    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(body)
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    match mode:
        case "launcher":
            # Interactive window. Drag the modules around; nothing drives them.
            viewer.launch(model=model, data=data)
        case "frame":
            # A still image - the cheapest way to eyeball a body.
            save_path = str(DATA / f"{file_name}.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
            console.log(f"saved {save_path}")
        case "video":
            # Mostly useful for showing a body slumping under gravity.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(model, data, duration=5.0, video_recorder=recorder)


# ============================================================================ #
#  5. ENTRY POINT
# ============================================================================ #


def main() -> None:
    """Score one randomly-sampled body against the target set."""
    targets = load_targets()

    console.log(f"encoding      : {GENOTYPE}")
    console.log(f"module budget : {NUM_OF_MODULES}")
    console.log(f"targets       : {len(targets)} bodies from {TARGET_DIR.name}")
    console.log(
        "target sizes  : "
        + ", ".join(str(t.number_of_nodes()) for t in targets),
    )

    # How far apart are the targets from each other? Your fitness cannot go
    # below the best possible compromise, and this is the clue to where that is.
    spread = [
        tree_edit_distance(a, b)
        for i, a in enumerate(targets)
        for b in targets[i + 1 :]
    ]
    console.log(f"target spread : mean pairwise distance {np.mean(spread):.2f}")

    # --- One random body --------------------------------------------------- #
    config.target_population_size = 20
    config.num_steps = 50
    
    initial = Population([make_individual() for _ in range(config.target_population_size)])
    initial: Population = evaluate(initial)

    ops: list[EAOperation] = [
            EAOperation(parent_selection),
            EAOperation(crossover),
            EAOperation(mutate),
            EAOperation(evaluate),
            EAOperation(survivor_selection),
        ]

    ea = EA(initial, ops, num_steps=config.num_steps)
    ea.run()
    # uv run assignments\assignment_1\A1_template_2026.py


    #console.log("")
    #console.log(f"random body   : {body.number_of_nodes()} modules")
    #console.log(
    #    "per-target    : "
    #    + ", ".join(f"{d:.1f}" for d in distances_to_targets(body, targets)),
    #)
    #console.log(f"fitness       : {fitness:.4f}   (lower is better)")
#
    #show_body(body, MODE, file_name=f"random_{GENOTYPE}")


if __name__ == "__main__":
    main()


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above samples ONE body at random and scores it. Your task is to
# replace "random" with "evolved".
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
# For morphology-specific evolution with the tree encoding, read:
#
#     examples/c_genotypes/1_body_evolution_tree.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- GENOTYPE - DEPENDENT "GOTCHA"S -------------------------
#
#   TREE: VARIABLE LENGTH - Tree genotypes grow; without pressure against it they
#     will grow forever, and every extra module costs an edit.
#   NDE: REPRODUCIBILITY   If you're using "nde": construct
#     `NeuralDevelopmentalEncoding` ONCE for your whole run, never per
#     individual or per generation, AND call `torch.manual_seed(...)` in
#     addition to the numpy/random seeds. See the two "IMPORTANT" notes under
#     "nde" in THE GENOTYPE CONTRACT above - getting either wrong means your
#     your runs won't reproduce cleanly.
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing - repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline, a good standard is at least a random search.
#   Keep the encoding, module budget and target set identical across
#     everything you compare, change one thing at a time.
#
# ============================================================================ #
