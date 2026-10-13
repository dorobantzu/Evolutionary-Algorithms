"""Extract metrics from the A2 scenario databases and plot fitness across generations.

Reads every database named
`[std_<X>_<Y>_]<scenario>_pop<N>_steps<N>_k<K>_eval<N>_seed<N>.db` in `__data__/` of the current
directory, the same folder the A2 template writes to. For example,
`std_0_2_ea1_pop52_steps500_k2_eval13052_seed1.db` is an EA1 run with mutation std 0.2.
Each scenario + mutation-std combination is analysed and plotted as its own series.
Run from the repo root:

    SCENARIO=all    uv run assignments/assignment_2/analyze.py   # default
    SCENARIO=ea1    uv run assignments/assignment_2/analyze.py
"""

import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import pdist
from scipy.stats import wilcoxon
from sqlmodel import Session, create_engine, select

from ariel.ec import Individual, Population

# Resolve relative to this file, not the shell's working directory.  This lets
# `python analyze.py` work when launched from assignments/assignment_2 as well
# as from the repository root.
DATA = Path(__file__).resolve().parents[2] / "__data__"
SCENARIOS = ("random", "ea1", "ea2")
SCENARIO = os.environ.get("SCENARIO", "all")
if SCENARIO not in (*SCENARIOS, "all"):
    raise ValueError(f"invalid SCENARIO: {SCENARIO!r}. Valid options: {SCENARIOS + ('all',)}")
SELECTED = SCENARIOS if SCENARIO == "all" else (SCENARIO,)

# Colour follows the scenario, so filtering never repaints a series.
STYLE = {
    "random": {"color": "#1baf7a", "linestyle": "--", "label": "random"},
    "ea1": {"color": "#2a78d6", "linestyle": "-", "label": "ea1"},
    "ea2": {"color": "#eb6834", "linestyle": "-.", "label": "ea2"},
}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3dd", "#fcfcfb"

# Used only when databases carry a mutation-std prefix, so one scenario has several series.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
LINESTYLES = ("-", "--", "-.", ":")
DB_NAME = re.compile(
    r"(?:std_(?P<std>\d+(?:_\d+)+)_)?"
    r"(?P<scenario>random|ea1|ea2)"
    r"_pop(?P<population>\d+)"
    r"_steps(?P<steps>\d+)"
    r"_k(?P<k>[^_]+)"
    r"_eval(?P<evaluations>\d+)"
    r"_seed(?P<seed>\d+)"
)


def discover() -> dict[str, dict]:
    """Group every database in DATA into series: one per scenario + mutation std."""
    series: dict[tuple[int, float], dict] = {}
    for db in sorted(DATA.glob("*.db")):  # sorted: the same file wins a duplicate on every machine
        m = DB_NAME.fullmatch(db.stem)
        if m is None:
            continue
        scenario = m["scenario"]
        mutation_std = m["std"].replace("_", ".") if m["std"] else None
        key = (SCENARIOS.index(scenario), float(mutation_std or -1))
        entry = series.setdefault(key, {"scenario": scenario, "mutation_std": mutation_std, "dbs": {}})
        if int(m["seed"]) in entry["dbs"]:
            print(f"[skip] {db.name}: same scenario, std and seed as {entry['dbs'][int(m['seed'])].name}")
            continue
        entry["dbs"][int(m["seed"])] = db

    ordered = [series[k] for k in sorted(series)]
    has_std = any(s["mutation_std"] for s in ordered)
    result: dict[str, dict] = {}
    for i, s in enumerate(ordered):
        base = STYLE[s["scenario"]]
        if s["mutation_std"] is None:
            name, label = s["scenario"], base["label"]
        else:
            name = f"{s['scenario']} std={s['mutation_std']}"
            label = f"{base['label']}, std={s['mutation_std']}"
        # Colours are assigned over all databases found, so SCENARIO filtering never repaints a series.
        s["color"] = PALETTE[i % len(PALETTE)] if has_std else base["color"]
        variant = sum(1 for other in ordered[:i] if other["scenario"] == s["scenario"])
        s["linestyle"] = LINESTYLES[variant % len(LINESTYLES)] if has_std else base["linestyle"]
        s["label"] = label
        result[name] = s
    return result


SERIES = discover()


def load(db: Path) -> Population:
    with Session(create_engine(f"sqlite:///{db}")) as session:
        return Population(list(session.exec(select(Individual)).all()))


def members(pop: Population, gen: int) -> Population:
    """Individuals present at the end of generation `gen` (after survivor selection)."""
    return pop.where(
        lambda ind: ind.time_of_birth <= gen
        and (ind.time_of_death > gen or (ind.time_of_death == gen and ind.alive))
    )


def std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def diversity(genotypes: np.ndarray) -> float:
    """Mean pairwise Euclidean distance between genotypes (rows)."""
    return float(pdist(genotypes).mean()) if len(genotypes) > 1 else 0.0


assert np.isclose(diversity(np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])), 10 / 3)

# Random search has no generations. Its individuals are split, in evaluation order, into groups of
# the EA population size to measure diversity per "generation". Leftover individuals are dropped.
RANDOM_GROUP_SIZE = 52
TARGET_FITNESS = -2.0
FITNESS_TOLERANCE = 0.1
TARGET_FITNESS_RANGE = (TARGET_FITNESS - FITNESS_TOLERANCE, TARGET_FITNESS + FITNESS_TOLERANCE)


def random_generations(pop: Population) -> list[dict]:
    """Per-group diversity for a random-search run; the fitness columns describe the whole run."""
    inds = sorted(pop, key=lambda ind: ind.id)
    f = np.array([ind.fitness for ind in inds])
    whole_run = {"best": float(f.min()), "mean": float(f.mean()), "worst": float(f.max()), "std": std(f)}
    genotypes = np.array([ind.genotype for ind in inds])
    # A group sits at the generation where an EA has used as many evaluations: an EA starts with
    # RANDOM_GROUP_SIZE individuals and adds half that many per generation, so groups land on 0, 2, 4, ...
    return [
        {"diversity": diversity(genotypes[end - RANDOM_GROUP_SIZE:end]),
         "generation": (end - RANDOM_GROUP_SIZE) // (RANDOM_GROUP_SIZE // 2),
         "evaluations": end, **whole_run}
        for end in range(RANDOM_GROUP_SIZE, len(genotypes) + 1, RANDOM_GROUP_SIZE)
    ]


def per_generation(pop: Population) -> list[dict]:
    rows = []
    last_gen = max(ind.time_of_death for ind in pop)
    if last_gen == 0:  # a zero-step run: random search
        return random_generations(pop)
    for gen in range(last_gen + 1):
        present = members(pop, gen)
        f = np.array([ind.fitness for ind in present])
        rows.append({
            "diversity": diversity(np.array([ind.genotype for ind in present])),
            "generation": gen,
            "evaluations": sum(1 for ind in pop if ind.time_of_birth <= gen),
            "best": float(f.min()),
            "mean": float(f.mean()),
            "worst": float(f.max()),
            "std": std(f),
        })
    return rows


def first_target_hit(pop: Population) -> tuple[int, int, float] | None:
    """Return evaluation count, generation and fitness of the first target-range evaluation.

    Individuals are ordered by their birth generation and then their ID, which is the order in
    which the run evaluated them. The target range is inclusive: -2.10 <= fitness <= -1.90.
    """
    lower, upper = TARGET_FITNESS_RANGE
    for evaluations, ind in enumerate(sorted(pop, key=lambda ind: (ind.time_of_birth, ind.id)), start=1):
        if lower <= ind.fitness <= upper:
            return evaluations, ind.time_of_birth, ind.fitness
    return None


def takeover_time(pop: Population) -> int | None:
    """Takeover time τ*: first generation in which every individual carries the genotype
    of the best individual of generation 0. None if that never happens within the run."""
    best_genotype = min(members(pop, 0), key=lambda ind: ind.fitness).genotype
    last_gen = max(ind.time_of_death for ind in pop)
    for gen in range(1, last_gen + 1):
        if all(ind.genotype == best_genotype for ind in members(pop, gen)):
            return gen
    return None


def main() -> None:
    runs: dict[str, dict[int, list[dict]]] = {}
    csv_rows = []

    for scenario in SELECTED:
        if not any(s["scenario"] == scenario for s in SERIES.values()):
            print(f"[{scenario}] no databases found, skipping")

    for name, series in SERIES.items():
        scenario = series["scenario"]
        if scenario not in SELECTED:
            continue
        dbs = dict(sorted(series["dbs"].items()))
        runs[name] = {}
        print(f"\n=== {name} ({len(dbs)} run(s)) ===")
        print(f"{'seed':>6} {'evals':>6} {'gens':>5} {'best id':>8} {'best':>9} {'mean':>9} {'std':>9}"
              f" {'hit eval':>9} {'hit gen':>8} {'hit fitness':>12}")

        for seed, db in dbs.items():
            try:  # an unreadable, empty or unfinished database must not stop the whole analysis
                pop = load(db)
                fitness = np.array([ind.fitness for ind in pop])
                best = pop.best(sort="min", attribute="fitness_", n=1)[0]
                gens = per_generation(pop)
            except Exception as error:  # noqa: BLE001
                print(f"[skip] {db.name}: {type(error).__name__}: {str(error).splitlines()[0]}")
                continue
            runs[name][seed] = gens
            csv_rows += [
                {"scenario": scenario, "mutation_std": series["mutation_std"] or "", "seed": seed, **row}
                for row in gens
            ]
            print(f"{seed:>6} {pop.size:>6} {max(ind.time_of_death for ind in pop):>5} {best.id:>8} "
                  f"{best.fitness:>9.4f} {fitness.mean():>9.4f} {std(fitness):>9.4f}", end="")
            hit = first_target_hit(pop)
            if hit is None:
                print(f" {'--':>9} {'--':>8} {'--':>12}")
                print("       first fitness within 0.1 of -2.00: not reached")
            else:
                evaluations, generation, hit_fitness = hit
                print(f" {evaluations:>9} {generation:>8} {hit_fitness:>12.4f}")
                print("       first fitness within 0.1 of -2.00: "
                      f"evaluation {evaluations}, generation {generation}, fitness {hit_fitness:.4f}")
            top10 = pop.best(sort="min", attribute="fitness_", n=10)
            print("       top10 (id: fitness): "
                  + ", ".join(f"{ind.id}: {ind.fitness:.4f}" for ind in top10))
            if scenario != "random":  # no selection in random, so no takeover
                tau = takeover_time(pop)
                print(f"       takeover time: {'not reached' if tau is None else f'{tau} generations'}")

        if not runs[name]:
            del runs[name]
            print("  no usable runs in this series")
            continue
        final_best = np.array([gens[-1]["best"] for gens in runs[name].values()])
        overall_best = np.array([min(r["best"] for r in gens) for gens in runs[name].values()])
        print(f"  across seeds: final-generation best {final_best.mean():.4f} ± {std(final_best):.4f}, "
              f"best ever {overall_best.mean():.4f} ± {std(overall_best):.4f}")

    compare_ea1_ea2(runs)

    if runs:
        csv_path = DATA / f"metrics_{SCENARIO}.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nper-generation metrics -> {csv_path}")
        fitness_label = "fitness (lower is better)"
        plot(runs, "fitness", [("best", "Best fitness in population", fitness_label),
                               ("mean", "Mean fitness in population", fitness_label)])
        ea_runs = {name: seeds for name, seeds in runs.items() if SERIES[name]["scenario"] != "random"}
        if ea_runs:  # the diversity plot shows the EAs only
            plot(ea_runs, "diversity", [("diversity", "Population diversity", "mean pairwise Euclidean distance")])


DIVERSITY_TEST_GENERATIONS = (0, 100, 200, 300, 400, 500)


def signed_rank(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired samples: (W, p)."""
    if np.all(x == y):  # scipy raises when every pair is identical, e.g. the shared initial populations
        return 0.0, 1.0
    result = wilcoxon(x, y, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def compare_ea1_ea2(runs: dict[str, dict[int, list[dict]]]) -> None:
    """Compare EA1 and EA2 runs, paired by seed, on fitness and on population diversity at fixed generations."""
    names = {s: [n for n in runs if SERIES[n]["scenario"] == s] for s in ("ea1", "ea2")}
    if any(len(found) != 1 for found in names.values()):
        print("\n[statistics] exactly one EA1 series and one EA2 series are required; skipping comparison "
              f"(found EA1: {names['ea1'] or 'none'}, EA2: {names['ea2'] or 'none'})")
        return
    name1, name2 = names["ea1"][0], names["ea2"][0]

    # EA1 and EA2 start from the same initial population for a given seed, so runs are paired by seed.
    seeds = sorted(set(runs[name1]) & set(runs[name2]))
    if not seeds:
        print("\n[statistics] EA1 and EA2 share no seeds; skipping the paired comparison")
        return
    ea1 = np.array([min(row["best"] for row in runs[name1][s]) for s in seeds])
    ea2 = np.array([min(row["best"] for row in runs[name2][s]) for s in seeds])
    statistic, p_value = signed_rank(ea1, ea2)
    print("\n=== EA1 vs EA2: best-ever fitness (two-sided Wilcoxon signed-rank, paired by seed) ===")
    print(f"{name1}: n={ea1.size}, mean={ea1.mean():.4f} ± {std(ea1):.4f}")
    print(f"{name2}: n={ea2.size}, mean={ea2.mean():.4f} ± {std(ea2):.4f}")
    print(f"W={statistic:.1f}, p={p_value:.4g}")

    # At each tested generation every seed contributes one pair of diversity values. Several tests
    # are made, so each p-value is judged against a Bonferroni threshold.
    threshold = 0.05 / len(DIVERSITY_TEST_GENERATIONS)
    print("\n=== EA1 vs EA2: population diversity per generation (two-sided Wilcoxon signed-rank, paired by seed) ===")
    print(f"{len(DIVERSITY_TEST_GENERATIONS)} tests, Bonferroni threshold p < {threshold:.4f}")
    print(f"{'gen':>4}  {name1 + ' mean ± std':>24}  {name2 + ' mean ± std':>24}  {'W':>7}  {'p':>9}  significant")
    for gen in DIVERSITY_TEST_GENERATIONS:
        # Row i of an EA run is generation i; a seed is left out if either run ended before `gen`.
        paired = [s for s in seeds if len(runs[name1][s]) > gen and len(runs[name2][s]) > gen]
        if not paired:
            print(f"{gen:>4}  no seed reaches this generation in both EAs, skipped")
            continue
        diversity1 = np.array([runs[name1][s][gen]["diversity"] for s in paired])
        diversity2 = np.array([runs[name2][s][gen]["diversity"] for s in paired])
        statistic, p_value = signed_rank(diversity1, diversity2)
        print(f"{gen:>4}  {f'{diversity1.mean():.4f} ± {std(diversity1):.4f} (n={diversity1.size})':>24}  "
              f"{f'{diversity2.mean():.4f} ± {std(diversity2):.4f} (n={diversity2.size})':>24}  "
              f"{statistic:>7.1f}  {p_value:>9.3g}  {'yes' if p_value < threshold else 'no'}")
    if min(ea1.size, ea2.size) < 5:
        print("Warning: the assignment requires at least five independent runs per condition.")


def plot(runs: dict[str, dict[int, list[dict]]], stem: str, panels: list[tuple[str, str, str]]) -> None:
    """One panel per (metric, title, y-label), saved as plots/<stem>_<SCENARIO>.png."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.2), facecolor=SURFACE, squeeze=False)
    axes = axes[0]

    for ax, (metric, title, ylabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for name, seeds in runs.items():
            style = SERIES[name]
            n_gens = min(len(g) for g in seeds.values())  # align seeds on shared generations
            values = np.array([[row[metric] for row in g[:n_gens]] for g in seeds.values()])
            mean = values.mean(axis=0)
            spread = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)

            x = np.array([row["generation"] for row in next(iter(seeds.values()))[:n_gens]])

            label = f"{style['label']} (n={len(seeds)})"
            ax.plot(x, mean, color=style["color"], linestyle=style["linestyle"], linewidth=2, label=label)
            ax.fill_between(x, mean - spread, mean + spread, color=style["color"], alpha=0.15, linewidth=0)
            ax.annotate(name, (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)

        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("generation", color=MUTED)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(ylabel, color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.margins(x=0.08)

    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.01, "Lines: mean across seeds. Bands: ±1 std across seeds.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out_dir = DATA / "plots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{stem}_{SCENARIO}.png"
    fig.savefig(out, dpi=200)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
