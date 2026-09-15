"""Shared pieces for the Assignment 1 experiments.

Targets, the fitness evaluation, seeding, body sampling and the per-generation
CSV log. Every script (EA, random search, calibration, analysis) imports from
here, so all conditions are scored and logged in exactly the same way.
"""

# Standard library
import csv
import json
import random
from pathlib import Path
from typing import Any

# Third-party libraries
import networkx as nx
import numpy as np

# Local libraries (ARIEL)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

# Local scripts
from tree_edit_distance import tree_edit_distance

# --- PATHS --- #
HERE = Path(__file__).parent
TARGET_DIR = HERE / "target_bodies"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

# --- TARGETS --- #
TARGETS: list[nx.DiGraph] = [
    load_graph_from_json(p) for p in sorted(TARGET_DIR.glob("*.json"))
]
TARGET_SIZES: list[int] = [t.number_of_nodes() for t in TARGETS]  # 7, 11, 15, 19, 25
N_TARGETS: int = len(TARGETS)

# --- BODY BUDGET --- #
MAX_MODULES: int = 20  # modules per body, excluding the core
MAX_NODES: int = MAX_MODULES + 1

# --- LOG FORMAT --- #
HISTORY_COLUMNS: list[str] = [
    "generation",
    "evaluations",
    "population_size",
    "best_fitness",
    "mean_fitness",
    "std_fitness",
    "best_size",
    "mean_size",
    "diversity",
    "distinct_parents",
    *(f"closest_{s}" for s in TARGET_SIZES),  # lowest distance to each target in the population
    *(f"best_dist_{s}" for s in TARGET_SIZES),  # the best body's distance to each target
]


def seed_everything(seed: int) -> None:
    """Seed every RNG the tree encoding uses (tree operators use `random`)."""
    random.seed(seed)
    np.random.seed(seed)


def random_genome(init_size: str) -> TreeGenome:
    """Sample a random tree body.

    `random_tree(n)` keeps adding modules until it reaches `n`, so "full"
    always gives 20-module bodies; "uniform" first draws n from 1..20.
    """
    n = MAX_MODULES if init_size == "full" else random.randint(1, MAX_MODULES)
    return random_tree(max_modules=n)


def evaluate_genome(genome: TreeGenome) -> dict[str, Any]:
    """Score one body against all targets. LOWER IS BETTER.

    Identical to `mean_plus_std_tree_edit_distance`, but also returns the
    per-target distances (lexicase selection and the analysis need them).
    """
    body = genome.to_networkx()
    dists = [tree_edit_distance(body, target) for target in TARGETS]
    return {
        "fitness": float(np.mean(dists) + np.std(dists)),  # population std, as in the assignment
        "dists": dists,
        "size": len(genome.nodes),
    }


def diversity(genomes: list[TreeGenome], rng: random.Random, sample: int = 20) -> float:
    """Mean pairwise tree edit distance within a random sample of bodies.

    Uses its own `rng` so that measuring diversity never changes the run.
    """
    chosen = rng.sample(genomes, min(sample, len(genomes)))
    graphs = [g.to_networkx() for g in chosen]
    pairs = [(a, b) for i, a in enumerate(graphs) for b in graphs[i + 1 :]]
    return float(np.mean([tree_edit_distance(a, b) for a, b in pairs]))


def generation_row(
    *,
    generation: int,
    evaluations: int,
    fitnesses: list[float],
    dists: list[list[float]],
    sizes: list[int],
    diversity_value: float,
    distinct_parents: int | str = "",
) -> dict[str, Any]:
    """Build one CSV row describing a population."""
    fit = np.asarray(fitnesses)
    dist = np.asarray(dists)
    best = int(np.argmin(fit))
    row: dict[str, Any] = {
        "generation": generation,
        "evaluations": evaluations,
        "population_size": len(fit),
        "best_fitness": fit[best],
        "mean_fitness": fit.mean(),
        "std_fitness": fit.std(),
        "best_size": sizes[best],
        "mean_size": float(np.mean(sizes)),
        "diversity": diversity_value,
        "distinct_parents": distinct_parents,
    }
    for j, s in enumerate(TARGET_SIZES):
        row[f"closest_{s}"] = dist[:, j].min()
        row[f"best_dist_{s}"] = dist[best, j]
    return row


class HistoryWriter:
    """Append-only per-generation CSV log."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=HISTORY_COLUMNS)
        self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
