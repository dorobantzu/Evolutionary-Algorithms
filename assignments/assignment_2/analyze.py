"""Extract metrics from the A2 scenario databases and plot fitness across generations.

Reads every `<scenario>_seed<N>.db` in `__data__/` of the current directory, the
same folder the A2 template writes to. Run from the repo root:

    SCENARIO=all    uv run assignments/assignment_2/analyze.py   # default
    SCENARIO=ea1    uv run assignments/assignment_2/analyze.py
"""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
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
    "ea1": {"color": "#2a78d6", "linestyle": "-", "label": "ea1 (elitism)"},
    "ea2": {"color": "#eb6834", "linestyle": "-.", "label": "ea2 (generational)"},
}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3dd", "#fcfcfb"


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


def seed_of(db: Path) -> int:
    return int(db.stem.rsplit("_seed", 1)[1])


def main() -> None:
    runs: dict[str, dict[int, list[dict]]] = {}
    csv_rows = []

    for scenario in SELECTED:
        dbs = sorted(DATA.glob(f"{scenario}_seed*.db"), key=seed_of)
        if not dbs:
            print(f"[{scenario}] no databases found, skipping")
            continue
        runs[scenario] = {}
        print(f"\n=== {scenario} ({len(dbs)} run(s)) ===")
        print(f"{'seed':>6} {'evals':>6} {'gens':>5} {'best id':>8} {'best':>9} {'mean':>9} {'std':>9}")

        for db in dbs:
            seed = seed_of(db)
            pop = load(db)
            fitness = np.array([ind.fitness for ind in pop])
            best = pop.best(sort="min", attribute="fitness_", n=1)[0]
            gens = per_generation(pop)
            runs[scenario][seed] = gens
            csv_rows += [{"scenario": scenario, "seed": seed, **row} for row in gens]
            print(f"{seed:>6} {pop.size:>6} {len(gens) - 1:>5} {best.id:>8} "
                  f"{best.fitness:>9.4f} {fitness.mean():>9.4f} {std(fitness):>9.4f}")
            top10 = pop.best(sort="min", attribute="fitness_", n=10)
            print(f"       top10 (id: fitness): "
                  + ", ".join(f"{ind.id}: {ind.fitness:.4f}" for ind in top10))

        final_best = np.array([gens[-1]["best"] for gens in runs[scenario].values()])
        overall_best = np.array([min(r["best"] for r in gens) for gens in runs[scenario].values()])
        print(f"  across seeds: final-generation best {final_best.mean():.4f} ± {std(final_best):.4f}, "
              f"best ever {overall_best.mean():.4f} ± {std(overall_best):.4f}")

    if runs:
        csv_path = DATA / f"metrics_{SCENARIO}.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nper-generation metrics -> {csv_path}")
        plot(runs)

    runtimes()


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
        for scenario, seeds in runs.items():
            style = STYLE[scenario]
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
            ax.annotate(scenario, (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
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
