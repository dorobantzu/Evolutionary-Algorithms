"""Figures and statistics for the Assignment 1 report.

Reads every results/<condition>/seed_XX/history.csv and writes, into figures/:

  fitness_curve.pdf         best fitness vs evaluations, mean +/- std over seeds (required plot)
  diversity_curve.pdf       population diversity per generation (EAs)
  selection_strength.pdf    distinct parents per generation (EAs) - checks the control's k
  closest_per_target.pdf    lowest distance to each target in the final population (EAs)
  final_fitness.pdf         final best fitness per condition, one dot per seed
  summary.csv               final-generation mean and std per condition
  stats.csv                 Mann-Whitney U tests with Holm correction and A12 effect sizes

Each figure is also saved as PNG for quick viewing.

Usage (from the repository root):
    uv run python assignments/assignment_1/analysis.py
    uv run python assignments/assignment_1/analysis.py --results path/to/results --figures path/to/figures
"""

# Standard library
import argparse
import re
from dataclasses import dataclass
from pathlib import Path

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Local scripts
from common import FIGURES_DIR, RESULTS_DIR, TARGET_SIZES

# --- STYLE --- #
# Colours from a colour-blind-validated categorical palette; the baseline is
# neutral grey. Every series also has its own marker so figures survive
# grayscale printing.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
COLUMN_WIDTH = 3.33  # inches, one GECCO column


@dataclass(frozen=True)
class Style:
    label: str
    short: str  # two-line label for category axes
    color: str
    marker: str
    rank: int


def style_of(condition: str) -> Style:
    if condition == "tournament_k2":
        return Style("Tournament (k = 2)", "Tournament\nk = 2", "#2a78d6", "o", 0)
    if condition == "lexicase":
        return Style("Lexicase", "Lexicase", "#eb6834", "s", 1)
    if match := re.fullmatch(r"tournament_k(\d+)", condition):
        k = match.group(1)
        return Style(f"Tournament (k = {k}, matched)", f"Tournament\nk = {k}", "#1baf7a", "^", 2)
    if condition == "random":
        # dark neutral: a lighter grey is confusable with the aqua control under deuteranopia
        return Style("Random search", "Random\nsearch", INK_2, "D", 3)
    return Style(condition, condition, MUTED, "v", 4)


def is_control(condition: str) -> bool:
    return bool(re.fullmatch(r"tournament_k\d+", condition)) and condition != "tournament_k2"


plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed fonts as TrueType (required by some PDF checkers)
})


# ============================================================================ #
#  Loading
# ============================================================================ #


def load_histories(results: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results.glob("*/seed_*/history.csv")):
        condition = path.parent.parent.name
        if condition.startswith("_"):  # e.g. _calibration
            continue
        frame = pd.read_csv(path)
        frame["condition"] = condition
        frame["seed"] = path.parent.name
        frames.append(frame)
    if not frames:
        msg = f"no history.csv files found under {results}"
        raise SystemExit(msg)
    return pd.concat(frames, ignore_index=True)


def ordered(conditions) -> list[str]:
    return sorted(set(conditions), key=lambda c: (style_of(c).rank, c))


def final_generation(data: pd.DataFrame) -> pd.DataFrame:
    last = data.groupby(["condition", "seed"])["generation"].transform("max")
    return data[data["generation"] == last]


# ============================================================================ #
#  Figures
# ============================================================================ #


def mean_std_curve(ax, data: pd.DataFrame, x: str, y: str, conditions: list[str]) -> None:
    for condition in conditions:
        s = style_of(condition)
        grouped = data[data["condition"] == condition].groupby(x)[y]
        mean = grouped.mean()
        std = grouped.std().fillna(0.0)  # sample std over seeds; 0 when only one seed
        ax.fill_between(mean.index, mean - std, mean + std, color=s.color, alpha=0.12, linewidth=0)
        ax.plot(mean.index, mean.to_numpy(), color=s.color, label=s.label)


def legend_above(ax) -> None:
    """Legend in a row above the axes, so it never covers data."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, borderaxespad=0,
              handlelength=1.6, columnspacing=1.2)


def save(fig, figures: Path, name: str) -> None:
    fig.savefig(figures / f"{name}.pdf")
    fig.savefig(figures / f"{name}.png", dpi=200)
    plt.close(fig)


def plot_fitness_curve(data: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))
    mean_std_curve(ax, data, "evaluations", "best_fitness", ordered(data["condition"]))
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best fitness (lower is better)")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    legend_above(ax)
    save(fig, figures, "fitness_curve")


def plot_ea_curve(data: pd.DataFrame, figures: Path, column: str, ylabel: str, name: str) -> None:
    ea = data[data["condition"] != "random"].dropna(subset=[column])
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.1))
    mean_std_curve(ax, ea, "generation", column, ordered(ea["condition"]))
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    legend_above(ax)
    save(fig, figures, name)


def plot_closest_per_target(final: pd.DataFrame, figures: Path) -> None:
    ea = final[final["condition"] != "random"]
    conditions = ordered(ea["condition"])
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
    rows = np.arange(len(TARGET_SIZES))
    offsets = np.linspace(-0.22, 0.22, len(conditions)) if len(conditions) > 1 else [0.0]
    for offset, condition in zip(offsets, conditions, strict=True):
        s = style_of(condition)
        values = ea[ea["condition"] == condition][[f"closest_{t}" for t in TARGET_SIZES]]
        ax.errorbar(
            values.mean().to_numpy(), rows + offset, xerr=values.std().fillna(0.0).to_numpy(),
            fmt=s.marker, color=s.color, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8,
            elinewidth=1, capsize=0, label=s.label,
        )
    ax.set_yticks(rows, [f"{t}-node target" for t in TARGET_SIZES])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(left=0)
    ax.set_xlabel("closest body in final population (edit distance)")
    legend_above(ax)
    save(fig, figures, "closest_per_target")


def plot_final_fitness(final: pd.DataFrame, figures: Path) -> None:
    conditions = ordered(final["condition"])
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))
    rng = np.random.default_rng(0)  # jitter only
    for i, condition in enumerate(conditions):
        s = style_of(condition)
        values = final[final["condition"] == condition]["best_fitness"].to_numpy()
        ax.boxplot(
            values, positions=[i], widths=0.5, showfliers=False,
            medianprops={"color": INK, "linewidth": 1},
            boxprops={"color": AXIS}, whiskerprops={"color": AXIS}, capprops={"color": AXIS},
        )
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, s=18, color=s.color, marker=s.marker,
                   edgecolors=SURFACE, linewidths=0.8, zorder=3)
    ax.set_xticks(range(len(conditions)), [style_of(c).short for c in conditions])
    ax.set_ylabel("final best fitness (lower is better)")
    save(fig, figures, "final_fitness")


# ============================================================================ #
#  Statistics
# ============================================================================ #


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A12: probability that a run of x has a HIGHER value than a run of y."""
    greater = (x[:, None] > y[None, :]).sum()
    equal = (x[:, None] == y[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(x) * len(y)))


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def compare(final: pd.DataFrame, hypothesis: str, metric: str, a: str, b: str) -> dict | None:
    x = final[final["condition"] == a][metric].to_numpy(dtype=float)
    y = final[final["condition"] == b][metric].to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    return {
        "hypothesis": hypothesis, "metric": metric, "a": a, "b": b,
        "n_a": len(x), "n_b": len(y),
        "mean_a": x.mean(), "std_a": x.std(ddof=1), "mean_b": y.mean(), "std_b": y.std(ddof=1),
        "U": u, "p": p, "a12": a12(x, y),
    }


def run_statistics(final: pd.DataFrame) -> pd.DataFrame:
    conditions = set(final["condition"])
    controls = sorted(c for c in conditions if is_control(c))
    eas = [c for c in ordered(conditions) if c != "random"]
    closest = [f"closest_{t}" for t in TARGET_SIZES]

    planned: list[tuple[str, str, str, str]] = []
    planned += [("H0 EA vs random", "best_fitness", ea, "random") for ea in eas]
    planned += [("H1 diversity", "diversity", "lexicase", other) for other in ["tournament_k2", *controls]]
    planned += [("H2 specialists", m, "lexicase", other) for other in ["tournament_k2", *controls] for m in closest]
    planned += [("H3 combined fitness", "best_fitness", "lexicase", other) for other in ["tournament_k2", *controls]]
    planned += [("H4 control diversity", "diversity", control, "tournament_k2") for control in controls]

    rows = [r for args in planned if (r := compare(final, *args)) is not None]
    stats = pd.DataFrame(rows)
    if stats.empty:
        return stats
    stats["p_holm"] = stats.groupby("hypothesis")["p"].transform(lambda p: holm(p.tolist()))
    return stats


def summarise(data: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    columns = ["best_fitness", "best_size", "mean_size", "diversity", *[f"closest_{t}" for t in TARGET_SIZES]]
    summary = final.groupby("condition")[columns].agg(["mean", "std"])
    summary[("distinct_parents_over_run", "mean")] = data.groupby("condition")["distinct_parents"].mean()
    summary[("seeds", "n")] = final.groupby("condition")["seed"].nunique()
    return summary.loc[ordered(summary.index)]


# ============================================================================ #
#  Entry point
# ============================================================================ #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)

    data = load_histories(args.results)
    final = final_generation(data)
    seeds = final.groupby("condition")["seed"].nunique()
    print("runs found:", ", ".join(f"{c} ({seeds[c]} seeds)" for c in ordered(seeds.index)))
    if (seeds < 5).any():
        print("warning: fewer than 5 seeds for some conditions - the assignment requires at least 5")

    plot_fitness_curve(data, args.figures)
    plot_ea_curve(data, args.figures, "diversity", "diversity (mean pairwise dist.)", "diversity_curve")
    plot_ea_curve(data, args.figures, "distinct_parents", "distinct parents per generation", "selection_strength")
    plot_closest_per_target(final, args.figures)
    plot_final_fitness(final, args.figures)

    summary = summarise(data, final)
    summary.to_csv(args.figures / "summary.csv")
    stats = run_statistics(final)
    stats.to_csv(args.figures / "stats.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print("\nfinal generation, mean and std over seeds:\n", summary)
        if not stats.empty:
            print("\nMann-Whitney U (two-sided), Holm-corrected within each hypothesis; "
                  "a12 = P(a > b), 0.5 = no difference:")
            print(stats[["hypothesis", "metric", "a", "b", "mean_a", "mean_b", "p", "p_holm", "a12"]]
                  .to_string(index=False))
    print(f"\nwrote figures and tables to {args.figures}")


if __name__ == "__main__":
    main()
