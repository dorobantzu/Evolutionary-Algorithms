"""Random-search baseline at the same evaluation budget as the EA.

Samples bodies in blocks of `pop_size` (one block per "generation") for
`generations + 1` blocks, i.e. exactly as many evaluations as the EA, and
writes the same CSV columns so the analysis treats it like any other condition.
`best_fitness` is the best body found so far; the other columns describe the
current block.

Usage (from the repository root):
    uv run python assignments/assignment_1/random_search.py --seed 0
"""

# Standard library
import argparse
import random
import time
from pathlib import Path

# Local scripts
from common import (
    RESULTS_DIR,
    TARGET_SIZES,
    HistoryWriter,
    diversity,
    evaluate_genome,
    generation_row,
    random_genome,
    save_json,
    seed_everything,
)


def run_random_search(
    seed: int,
    pop_size: int = 100,
    generations: int = 100,
    init_size: str = "uniform",
    out_dir: Path | None = None,
) -> Path:
    start = time.perf_counter()
    out = out_dir or RESULTS_DIR / "random" / f"seed_{seed:02d}"
    save_json(
        out / "config.json",
        {"condition": "random", "seed": seed, "pop_size": pop_size,
         "generations": generations, "init_size": init_size},
    )
    seed_everything(seed)
    diversity_rng = random.Random(10_000 + seed)
    history = HistoryWriter(out / "history.csv")

    best: dict | None = None
    for generation in range(generations + 1):
        genomes = [random_genome(init_size) for _ in range(pop_size)]
        results = [evaluate_genome(g) for g in genomes]
        for genome, result in zip(genomes, results, strict=True):
            if best is None or result["fitness"] < best["fitness"]:
                best = {**result, "genotype": genome.to_dict()}

        row = generation_row(
            generation=generation,
            evaluations=(generation + 1) * pop_size,
            fitnesses=[r["fitness"] for r in results],
            dists=[r["dists"] for r in results],
            sizes=[r["size"] for r in results],
            diversity_value=diversity(genomes, diversity_rng),
        )
        # best-so-far instead of best-in-block
        row["best_fitness"] = best["fitness"]
        row["best_size"] = best["size"]
        for j, s in enumerate(TARGET_SIZES):
            row[f"best_dist_{s}"] = best["dists"][j]
        history.write(row)

    history.close()
    best["dists"] = dict(zip(TARGET_SIZES, best["dists"], strict=True))
    save_json(out / "best_body.json", best)
    print(f"random seed {seed}: best fitness {best['fitness']:.3f} "
          f"({time.perf_counter() - start:.0f}s) -> {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--init-size", choices=["uniform", "full"], default="uniform")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_random_search(args.seed, args.pop_size, args.generations, args.init_size, args.out_dir)
