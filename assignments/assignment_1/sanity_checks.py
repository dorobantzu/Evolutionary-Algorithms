"""Quick checks that the experiment code does what the report will claim.

Run this after every change to the EA code (takes well under a minute):
    uv run python assignments/assignment_1/sanity_checks.py
"""

# Standard library
import csv
import random
import sys
import tempfile
from pathlib import Path

# Local libraries (ARIEL)
from ariel.ec import Individual

# Local scripts
from common import TARGETS, evaluate_genome, random_genome
from ea_tree import RunConfig, run_experiment
from random_search import run_random_search
from selection import lexicase, tournament
from tree_edit_distance import mean_plus_std_tree_edit_distance

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{f' - {detail}' if detail and not ok else ''}")
    if not ok:
        FAILURES.append(name)


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fake(fitness: float, dists: list[float]) -> Individual:
    ind = Individual()
    ind.fitness = fitness
    ind.tags = {"dists": dists}
    return ind


def main() -> None:
    random.seed(0)

    print("1. fitness")
    mismatches = 0
    for init in ("uniform", "full"):
        for _ in range(15):
            genome = random_genome(init)
            ours = evaluate_genome(genome)["fitness"]
            official = mean_plus_std_tree_edit_distance(genome.to_networkx(), TARGETS)
            mismatches += abs(ours - official) > 1e-9
    check("evaluate_genome matches mean_plus_std_tree_edit_distance on 30 bodies", mismatches == 0,
          f"{mismatches} mismatches")

    print("2. selection (worked example from the proposal)")
    a = fake(12.41, [9, 10, 11, 12, 13])
    b = fake(16.27, [5, 8, 12, 15, 18])
    c = fake(11.97, [12, 12, 11, 10, 9])
    d = fake(15.53, [5, 9, 12, 14, 17])
    check("lexicase, order T7 -> T11 picks B", lexicase([a, b, c, d], order=[0, 1, 2, 3, 4]) is b)
    check("lexicase, order T25 first picks C", lexicase([a, b, c, d], order=[4, 0, 1, 2, 3]) is c)
    check("tournament between A and C picks C", tournament([a, c], k=2) is c)

    with tempfile.TemporaryDirectory() as tmp:
        pop, gens = 20, 5
        for selection in ("tournament", "lexicase"):
            print(f"3. short EA run ({selection}, pop {pop} x {gens} generations)")
            out = run_experiment(RunConfig(selection=selection, seed=0, pop_size=pop,
                                           generations=gens, out_dir=Path(tmp) / selection))
            rows = read_history(out / "history.csv")
            best = [float(r["best_fitness"]) for r in rows]
            check("one CSV row per generation (incl. generation 0)", len(rows) == gens + 1, f"{len(rows)} rows")
            check("population size constant", all(int(r["population_size"]) == pop for r in rows),
                  str([r["population_size"] for r in rows]))
            check("best fitness never gets worse (elitism)", all(x >= y - 1e-12 for x, y in zip(best, best[1:])),
                  str(best))
            check("evaluation count = (generations + 1) x pop", int(rows[-1]["evaluations"]) == (gens + 1) * pop,
                  rows[-1]["evaluations"])
            check("database, best body and specialists saved",
                  all((out / f).exists() for f in ("database.db", "best_body.json", "specialists.json")))

        print(f"4. random search (pop {pop} x {gens} generations)")
        out = run_random_search(seed=0, pop_size=pop, generations=gens, out_dir=Path(tmp) / "random")
        rows = read_history(out / "history.csv")
        best = [float(r["best_fitness"]) for r in rows]
        check("same evaluation budget as the EA", int(rows[-1]["evaluations"]) == (gens + 1) * pop)
        check("best-so-far never gets worse", all(x >= y - 1e-12 for x, y in zip(best, best[1:])))

        print("5. reproducibility")
        runs = [run_experiment(RunConfig(selection="lexicase", seed=3, pop_size=pop, generations=3,
                                         out_dir=Path(tmp) / f"repeat_{i}")) for i in range(2)]
        same = (runs[0] / "history.csv").read_text() == (runs[1] / "history.csv").read_text()
        check("same seed gives an identical history.csv", same)

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} CHECK(S) FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
