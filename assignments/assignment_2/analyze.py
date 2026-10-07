"""Extract metrics from the A2 scenario databases and plot fitness across generations.

Reads every `[std_<X>_<Y>_]<scenario>_pop<N>_steps<N>_k<N>_eval<N>_seed<N>.db` in `__data__/` of the
current directory, the same folder the A2 template writes to. The optional `std_<X>_<Y>_` prefix is the
mutation std (`std_0_2_` = 0.2); each scenario + std combination is analysed and plotted as its own series.
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
from scipy.stats import mannwhitneyu
from sqlmodel import Session, create_engine, select

from ariel.ec import Individual, Population, config

DATA = config.output_folder
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
    r"^(?:std_(?P<std>\d+(?:_\d+)?)_)?(?P<scenario>random|ea1|ea2)"
    r"_pop\d+_steps\d+_k\w+_eval\d+_seed(?P<seed>\d+)$"
)


def discover() -> dict[str, dict]:
    """Group every database in DATA into series: one per scenario + mutation std."""
    series: dict[tuple[int, float], dict] = {}
    for db in DATA.glob("*.db"):
        m = DB_NAME.match(db.stem)
        if m is None:
            continue
        scenario = m["scenario"]
        mutation_std = m["std"].replace("_", ".") if m["std"] else None
        key = (SCENARIOS.index(scenario), float(mutation_std or -1))
        entry = series.setdefault(key, {"scenario": scenario, "mutation_std": mutation_std, "dbs": {}})
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


def per_generation(pop: Population) -> list[dict]:
    rows = []
    last_gen = max(ind.time_of_death for ind in pop)
    for gen in range(last_gen + 1):
        f = np.array([ind.fitness for ind in members(pop, gen)])
        rows.append({
            "generation": gen,
            "evaluations": sum(1 for ind in pop if ind.time_of_birth <= gen),
            "best": float(f.min()),
            "mean": float(f.mean()),
            "worst": float(f.max()),
            "std": std(f),
        })
    return rows


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
        print(f"{'seed':>6} {'evals':>6} {'gens':>5} {'best id':>8} {'best':>9} {'mean':>9} {'std':>9}")

        for seed, db in dbs.items():
            pop = load(db)
            fitness = np.array([ind.fitness for ind in pop])
            best = pop.best(sort="min", attribute="fitness_", n=1)[0]
            gens = per_generation(pop)
            runs[name][seed] = gens
            csv_rows += [
                {"scenario": scenario, "mutation_std": series["mutation_std"] or "", "seed": seed, **row}
                for row in gens
            ]
            print(f"{seed:>6} {pop.size:>6} {len(gens) - 1:>5} {best.id:>8} "
                  f"{best.fitness:>9.4f} {fitness.mean():>9.4f} {std(fitness):>9.4f}")
            top10 = pop.best(sort="min", attribute="fitness_", n=10)
            print("       top10 (id: fitness): "
                  + ", ".join(f"{ind.id}: {ind.fitness:.4f}" for ind in top10))
            if scenario != "random":  # no selection in random, so no takeover
                tau = takeover_time(pop)
                print(f"       takeover time: {'not reached' if tau is None else f'{tau} generations'}")

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
        plot(runs)

    runtimes()


def compare_ea1_ea2(runs: dict[str, dict[int, list[dict]]]) -> None:
    """Compare independent EA1 and EA2 runs on best-ever fitness."""
    names = {s: [n for n in runs if SERIES[n]["scenario"] == s] for s in ("ea1", "ea2")}
    if any(len(found) != 1 for found in names.values()):
        print("\n[statistics] exactly one EA1 series and one EA2 series are required; skipping comparison "
              f"(found EA1: {names['ea1'] or 'none'}, EA2: {names['ea2'] or 'none'})")
        return
    name1, name2 = names["ea1"][0], names["ea2"][0]

    ea1 = np.array([min(row["best"] for row in run) for run in runs[name1].values()])
    ea2 = np.array([min(row["best"] for row in run) for run in runs[name2].values()])
    statistic, p_value = mannwhitneyu(ea1, ea2, alternative="two-sided", method="auto")
    print("\n=== EA1 vs EA2: best-ever fitness (two-sided Mann–Whitney U) ===")
    print(f"{name1}: n={ea1.size}, mean={ea1.mean():.4f} ± {std(ea1):.4f}")
    print(f"{name2}: n={ea2.size}, mean={ea2.mean():.4f} ± {std(ea2):.4f}")
    print(f"U={statistic:.1f}, p={p_value:.4g}")
    if min(ea1.size, ea2.size) < 5:
        print("Warning: the assignment requires at least five independent runs per condition.")


def runtimes() -> None:
    """Summarise __data__/runtimes.csv and draw a boxplot of run durations per scenario."""
    path = DATA / "runtimes.csv"
    if not path.exists():
        print(f"\n[runtime] {path} not found, skipping")
        return
    with path.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["scenario"] in SELECTED]
    by_scenario = {s: [r for r in rows if r["scenario"] == s] for s in SELECTED}
    by_scenario = {s: rs for s, rs in by_scenario.items() if rs}
    if not by_scenario:
        print("\n[runtime] no runtimes for the selected scenario(s), skipping")
        return

    print("\n=== runtime (seconds) ===")
    print(f"{'scenario':>9} {'runs':>5} {'mean':>9} {'std':>9} {'min':>9} {'max':>9}  machines")
    for scenario, rs in by_scenario.items():
        d = np.array([float(r["duration_s"]) for r in rs])
        machines = sorted({r["machine"] for r in rs})
        print(f"{scenario:>9} {d.size:>5} {d.mean():>9.1f} {std(d):>9.1f} {d.min():>9.1f} {d.max():>9.1f}  {', '.join(machines)}")

    machines = sorted({r["machine"] for r in rows})
    markers = dict(zip(machines, "osD^v<>p"))
    rng = np.random.default_rng(0)  # fixed jitter so the plot is reproducible

    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(by_scenario), 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    names = list(by_scenario)
    data = [[float(r["duration_s"]) for r in by_scenario[s]] for s in names]
    box = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                     medianprops={"color": INK, "linewidth": 2},
                     whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for patch, scenario in zip(box["boxes"], names):
        patch.set_facecolor(STYLE[scenario]["color"] + "33")
        patch.set_edgecolor(STYLE[scenario]["color"])

    # Every run as a dot, so a box drawn from only a few runs is not mistaken for more data.
    for i, scenario in enumerate(names, start=1):
        for r in by_scenario[scenario]:
            ax.scatter(i + rng.uniform(-0.12, 0.12), float(r["duration_s"]), s=36,
                       marker=markers[r["machine"]], color=STYLE[scenario]["color"],
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for machine in machines:
        ax.scatter([], [], marker=markers[machine], color=MUTED, label=machine)

    ax.set_xticks(range(1, len(names) + 1), [f"{s}\n(n={len(by_scenario[s])})" for s in names])
    ax.set_title("Run duration", loc="left", fontsize=11, color=INK)
    ax.set_ylabel("wall-clock time (s)", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(title="machine", frameon=False, fontsize=8, title_fontsize=8, labelcolor=INK)
    fig.tight_layout()

    out_dir = DATA / "plots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"runtime_{SCENARIO}.png"
    fig.savefig(out, dpi=200)
    print(f"plot -> {out}")


def plot(runs: dict[str, dict[int, list[dict]]]) -> None:
    max_gen = max(len(g) for seeds in runs.values() for g in seeds.values()) - 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    for ax, metric, title in zip(axes, ("best", "mean"), ("Best fitness in population", "Mean fitness in population")):
        ax.set_facecolor(SURFACE)
        for name, seeds in runs.items():
            style = SERIES[name]
            n_gens = min(len(g) for g in seeds.values())  # align seeds on shared generations
            values = np.array([[row[metric] for row in g[:n_gens]] for g in seeds.values()])
            mean = values.mean(axis=0)
            spread = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)

            if n_gens == 1:  # random: no generations, draw as a flat baseline
                x = np.array([0, max(max_gen, 1)])
                mean, spread = np.repeat(mean, 2), np.repeat(spread, 2)
            else:
                x = np.arange(n_gens)

            label = f"{style['label']} (n={len(seeds)})"
            ax.plot(x, mean, color=style["color"], linestyle=style["linestyle"], linewidth=2, label=label)
            ax.fill_between(x, mean - spread, mean + spread, color=style["color"], alpha=0.15, linewidth=0)
            ax.annotate(name, (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)

        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("generation", color=MUTED)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("fitness (lower is better)", color=MUTED)
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
    out = out_dir / f"fitness_{SCENARIO}.png"
    fig.savefig(out, dpi=200)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
