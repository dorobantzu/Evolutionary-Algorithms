"""Tree-encoded body evolution with a choice of parent selection.

Research question: does lexicase parent selection (one target at a time)
instead of tournament selection (on mean + std) change population diversity,
the bodies found, and the final fitness?

Everything except parent selection is fixed:
  subtree crossover (p = 0.7) -> one random ariel tree mutation per child
  -> discard children above 21 nodes -> evaluate
  -> generational replacement with one elite.

Usage (from the repository root):
    uv run python assignments/assignment_1/ea_tree.py --selection tournament --k 2 --seed 0
    uv run python assignments/assignment_1/ea_tree.py --selection lexicase --seed 0

Output: results/<condition>/seed_XX/{config.json, history.csv, database.db,
        best_body.json, specialists.json}

NOTE: no `from __future__ import annotations` here - `EAOperation` checks that
each step's first argument is annotated with the actual `Population` class.
"""

# Standard library
import argparse
import copy
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Third-party libraries
import numpy as np

# Local libraries (ARIEL)
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

# Local scripts
from common import (
    MAX_NODES,
    RESULTS_DIR,
    TARGET_SIZES,
    HistoryWriter,
    diversity,
    ensure_fresh,
    evaluate_genome,
    generation_row,
    random_genome,
    save_json,
    seed_everything,
)
from selection import Selector, make_selector


def mutate_subtree(genome: TreeGenome) -> None:
    """Replace a random subtree with a new random branch of 1-3 modules."""
    mutate_subtree_replacement(genome, max_modules=3)


MUTATIONS = [mutate_replace_node, mutate_shrink, mutate_hoist, mutate_subtree]


# ============================================================================ #
#  Configuration and run state
# ============================================================================ #


@dataclass
class RunConfig:
    selection: str = "tournament"  # "tournament" | "lexicase"
    k: int = 2  # tournament size (ignored for lexicase)
    seed: int = 0
    pop_size: int = 100
    generations: int = 100
    p_crossover: float = 0.7
    init_size: str = "uniform"  # "uniform" | "full"
    diversity_sample: int = 20
    out_dir: Path | None = None
    overwrite: bool = False  # replace an existing run in the output directory

    @property
    def condition(self) -> str:
        return "lexicase" if self.selection == "lexicase" else f"tournament_k{self.k}"

    def output_dir(self) -> Path:
        return self.out_dir or RESULTS_DIR / self.condition / f"seed_{self.seed:02d}"


@dataclass
class RunState:
    config: RunConfig
    select: Selector
    history: HistoryWriter
    diversity_rng: random.Random
    generation: int = 0
    distinct_parents: int = 0
    parent_picks: int = 0
    timings: dict[str, float] = field(default_factory=dict)


# ============================================================================ #
#  Helpers
# ============================================================================ #


def make_individual(genome: TreeGenome) -> Individual:
    ind = Individual()
    ind.genotype = genome.to_dict()  # the database stores the genotype as JSON
    return ind


def genome_of(ind: Individual) -> TreeGenome:
    """A private, mutable copy of an individual's genome.

    `TreeGenome.from_dict` reuses the stored edge list, so without the deep
    copy a mutation would silently change the parent's stored genotype.
    """
    return TreeGenome.from_dict(copy.deepcopy(ind.genotype))


def record(individuals: list[Individual], run: RunState) -> None:
    """Write one CSV row for the current population."""
    cfg = run.config
    genomes = [TreeGenome.from_dict(ind.genotype) for ind in individuals]
    run.history.write(
        generation_row(
            generation=run.generation,
            evaluations=(run.generation + 1) * cfg.pop_size,
            fitnesses=[ind.fitness for ind in individuals],
            dists=[ind.tags["dists"] for ind in individuals],
            sizes=[ind.tags["size"] for ind in individuals],
            diversity_value=diversity(genomes, run.diversity_rng, cfg.diversity_sample),
            distinct_parents=run.distinct_parents if run.generation > 0 else "",
            parent_picks=run.parent_picks if run.generation > 0 else "",
        ),
    )


# ============================================================================ #
#  EA steps (run in this order every generation)
# ============================================================================ #


def reproduce(population: Population, run: RunState) -> Population:
    """Select parents, recombine, mutate; add `pop_size` new children."""
    cfg = run.config
    parents = population.alive.to_list()
    children: list[Individual] = []
    picks: list[int] = []  # every parent pick, in order

    while len(children) < cfg.pop_size:
        parent_a, parent_b = run.select(parents), run.select(parents)
        picks += [id(parent_a), id(parent_b)]
        a, b = genome_of(parent_a), genome_of(parent_b)
        if random.random() < cfg.p_crossover:
            a, b = crossover_subtree(a, b)
        for genome in (a, b):
            random.choice(MUTATIONS)(genome)
            if len(genome.nodes) <= MAX_NODES:  # over-budget children are discarded
                children.append(make_individual(genome))

    population.extend(children[: cfg.pop_size])
    run.generation += 1
    # Selection strength = distinct bodies among the FIRST pop_size picks. Each
    # pick pair yields at most two kept children, so at least pop_size picks are
    # always made; fixing the count keeps the measure independent of how many
    # over-budget children were discarded, and identical to calibrate_k.py.
    run.distinct_parents = len(set(picks[: cfg.pop_size]))
    run.parent_picks = len(picks)
    return population


def evaluate(population: Population, run: RunState) -> Population:  # noqa: ARG001
    for ind in population.unevaluated:
        result = evaluate_genome(TreeGenome.from_dict(ind.genotype))
        ind.fitness = result["fitness"]
        ind.tags = {"dists": result["dists"], "size": result["size"]}
    return population


def survivor_selection(population: Population, run: RunState) -> Population:
    """Generational replacement with one elite.

    The EA engine stamps `time_of_birth` when it saves a generation, so the
    children created in this step still have `time_of_birth == -1`.
    """
    n = run.config.pop_size
    alive = population.alive.to_list()
    parents = [ind for ind in alive if ind.time_of_birth != -1]
    children = [ind for ind in alive if ind.time_of_birth == -1]
    if len(parents) != n or len(children) != n:
        msg = (
            f"expected {n} parents and {n} children, got {len(parents)} and "
            f"{len(children)} - has the EA engine's time_of_birth convention changed?"
        )
        raise RuntimeError(msg)

    elite = min(parents, key=lambda ind: ind.fitness)
    for parent in parents:
        if parent is not elite:
            parent.alive = False
    max(children, key=lambda ind: ind.fitness).alive = False  # the elite takes its place
    return population


def log_generation(population: Population, run: RunState) -> Population:
    record(population.alive.to_list(), run)
    return population


# ============================================================================ #
#  Running one experiment
# ============================================================================ #


def save_final(individuals: list[Individual], out: Path) -> None:
    """Save the best body and, per target, the closest body in the population."""

    def describe(ind: Individual) -> dict[str, Any]:
        return {
            "fitness": ind.fitness,
            "size": ind.tags["size"],
            "dists": dict(zip(TARGET_SIZES, ind.tags["dists"], strict=True)),
            "genotype": ind.genotype,
        }

    best = min(individuals, key=lambda ind: ind.fitness)
    save_json(out / "best_body.json", describe(best))
    specialists = {
        f"target_{size}": describe(min(individuals, key=lambda ind, j=j: ind.tags["dists"][j]))
        for j, size in enumerate(TARGET_SIZES)
    }
    save_json(out / "specialists.json", specialists)


def run_experiment(cfg: RunConfig, extra_ops: Sequence[EAOperation] = ()) -> Path:
    """Run one EA and return its output directory.

    `extra_ops` run after survivor selection (used by calibrate_k.py).
    """
    start = time.perf_counter()
    out = cfg.output_dir()
    ensure_fresh(out, cfg.overwrite)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", {**cfg.__dict__, "condition": cfg.condition})

    seed_everything(cfg.seed)
    run = RunState(
        config=cfg,
        select=make_selector(cfg.selection, cfg.k),
        history=HistoryWriter(out / "history.csv"),
        diversity_rng=random.Random(10_000 + cfg.seed),
    )

    initial = Population(
        [make_individual(random_genome(cfg.init_size)) for _ in range(cfg.pop_size)],
    )
    initial = evaluate(initial, run)
    record(initial.to_list(), run)  # generation 0

    ops = [
        EAOperation(reproduce, run=run),
        EAOperation(evaluate, run=run),
        EAOperation(survivor_selection, run=run),
        *extra_ops,
        EAOperation(log_generation, run=run),
    ]
    ea = EA(
        initial,
        ops,
        num_steps=cfg.generations,
        is_maximisation=False,  # lower fitness is better
        quiet=True,
        db_file_path=out / "database.db",
        db_handling="delete",
    )
    ea.run()
    run.history.close()

    ea.fetch_population()  # reload the final generation from the database
    final = ea.population.to_list()
    save_final(final, out)

    best = min(ind.fitness for ind in final)
    print(
        f"{cfg.condition} seed {cfg.seed}: best fitness {best:.3f} "
        f"({time.perf_counter() - start:.0f}s) -> {out}",
    )
    return out


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selection", choices=["tournament", "lexicase"], default="tournament")
    parser.add_argument("--k", type=int, default=2, help="tournament size (ignored for lexicase)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--p-crossover", type=float, default=0.7)
    parser.add_argument("--init-size", choices=["uniform", "full"], default="uniform")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="use for test runs so they never replace final results")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run")
    args = parser.parse_args()
    return RunConfig(
        selection=args.selection,
        k=args.k,
        seed=args.seed,
        pop_size=args.pop_size,
        generations=args.generations,
        p_crossover=args.p_crossover,
        init_size=args.init_size,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    run_experiment(parse_args())
