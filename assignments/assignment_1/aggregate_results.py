"""Aggregate the random-search baseline and both EA variants into one figure.

Reads results already on disk - runs nothing, overwrites nothing:

    randomsearch_seed{S}.csv   random search baseline (one row per individual)
    seed{S}_mut{P}.db          EA run, ariel.ec's SQLite log

Writes `comparison_curve.png` and `comparison_stats.csv`.

    uv run assignments\\assignment_1\\aggregate_results.py
"""

import csv
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path.cwd() / "__data__" / "A1_template_2026"
SEEDS = [1, 2, 3, 4, 5]
NUM_GENERATIONS = 101


def _best_so_far(best_per_gen: dict[int, float]) -> list[float]:
    """Running minimum over generations, carrying through generations with no births."""
    curve: list[float] = []
    best = float("inf")
    for gen in range(NUM_GENERATIONS):
        best = min(best, best_per_gen.get(gen, float("inf")))
        curve.append(best)
    return curve


def random_search_curve(seed: int) -> list[float]:
    with (DATA / f"randomsearch_seed{seed}.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    best_per_gen: dict[int, float] = {}
    for row in rows:
        gen, fit = int(row["generation"]), float(row["fitness"])
        best_per_gen[gen] = min(best_per_gen.get(gen, float("inf")), fit)
    return _best_so_far(best_per_gen)


def ea_curve(seed: int, mutation_prob: str) -> list[float]:
    con = sqlite3.connect(DATA / f"seed{seed}_mut{mutation_prob}.db")
    best_per_gen = dict(
        con.execute(
            "SELECT time_of_birth, MIN(fitness_) FROM individual GROUP BY time_of_birth",
        ).fetchall(),
    )
    con.close()
    return _best_so_far(best_per_gen)


def main() -> None:
    variants = {
        "random search": ([random_search_curve(s) for s in SEEDS], "tab:blue"),
        "EA (mut=0.1)": ([ea_curve(s, "0.1") for s in SEEDS], "tab:orange"),
        "EA (mut=0.4)": ([ea_curve(s, "0.4") for s in SEEDS], "tab:green"),
    }

    gens = np.arange(NUM_GENERATIONS)
    plt.figure(figsize=(8, 5))
    rows = []
    for name, (curves, colour) in variants.items():
        curves = np.array(curves)
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        plt.plot(gens, mean, color=colour, label=name)
        plt.fill_between(gens, mean - std, mean + std, color=colour, alpha=0.2)

        finals = curves[:, -1]
        rows.append([
            name,
            len(finals),
            f"{finals.mean():.4f}",
            f"{finals.std():.4f}",
            f"{finals.min():.4f}",
            f"{finals.max():.4f}",
        ])

    plt.xlabel("generation")
    plt.ylabel("fitness (lower is better)")
    plt.title(f"Best-so-far fitness, mean ± std over {len(SEEDS)} seeds")
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = DATA / "comparison_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")

    header = ["variant", "runs", "mean", "std", "min", "max"]
    stats_path = DATA / "comparison_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([header, *rows])

    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt.format(*map(str, row)))
    print(f"\nplot  : {plot_path}\nstats : {stats_path}")


if __name__ == "__main__":
    main()
