"""Choose the tournament size for the matched-strength control condition.

Selection strength is measured as the number of DISTINCT bodies among
pop_size parent picks - the same measure ea_tree.py logs as `distinct_parents`.
Fewer distinct parents = stronger selection.

To compare selectors fairly, all of them are applied to the SAME populations:
we run lexicase EAs and, after survivor selection in every generation, draw
parents from that population with lexicase and with tournaments of several
sizes. The RNG state is saved and restored around the measurement, so the
runs themselves are unaffected.

Usage (from the repository root):
    uv run python assignments/assignment_1/calibrate_k.py
"""

# Standard library
import argparse
import csv
import random

# Third-party libraries
import numpy as np

# Local libraries (ARIEL)
from ariel.ec import EAOperation, Population

# Local scripts
from common import RESULTS_DIR, save_json
from ea_tree import RunConfig, run_experiment
from selection import Selector, lexicase, make_selector

# Lexicase over only 5 targets selects very strongly (in test runs, about as
# strongly as a size-30 tournament), so the range must reach far beyond small k.
CANDIDATE_K = [2, 5, 10, 15, 20, 25, 30, 40, 50]


def measure_selection(population: Population, selectors: dict[str, Selector],
                      counts: dict[str, list[int]]) -> Population:
    parents = population.alive.to_list()
    state = random.getstate()
    for name, select in selectors.items():
        picked = {id(select(parents)) for _ in range(len(parents))}  # pop_size picks
        counts[name].append(len(picked))
    random.setstate(state)
    return population


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102],
                        help="use seeds different from the final experiments")
    parser.add_argument("--generations", type=int, default=50)
    args = parser.parse_args()

    selectors: dict[str, Selector] = {"lexicase": lexicase}
    selectors |= {f"tournament_k{k}": make_selector("tournament", k) for k in CANDIDATE_K}
    counts: dict[str, list[int]] = {name: [] for name in selectors}

    out_root = RESULTS_DIR / "_calibration"
    for seed in args.seeds:
        cfg = RunConfig(selection="lexicase", seed=seed, generations=args.generations,
                        out_dir=out_root / f"seed_{seed}", overwrite=True)  # calibration is meant to be re-run
        run_experiment(cfg, extra_ops=[EAOperation(measure_selection, selectors=selectors, counts=counts)])

    means = {name: float(np.mean(v)) for name, v in counts.items()}
    target = means["lexicase"]
    best_k = min(CANDIDATE_K, key=lambda k: abs(means[f"tournament_k{k}"] - target))

    print("\nmean distinct parents per generation (out of pop_size picks):")
    for name, value in means.items():
        marker = "  <- matches lexicase" if name == f"tournament_k{best_k}" else ""
        print(f"  {name:15s} {value:6.1f}{marker}")
    print(f"\nchosen control: tournament with k = {best_k}")
    print(f"run the final experiments with: K_CONTROL={best_k} bash assignments/assignment_1/run_all.sh")

    with (out_root / "calibration.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["selector", "mean_distinct_parents", "std_distinct_parents"])
        for name, v in counts.items():
            writer.writerow([name, np.mean(v), np.std(v)])
    save_json(out_root / "calibration.json",
              {"seeds": args.seeds, "generations": args.generations,
               "mean_distinct_parents": means, "chosen_k": best_k})


if __name__ == "__main__":
    main()
