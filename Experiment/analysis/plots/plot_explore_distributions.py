"""Per-(benchmark, method) distribution of model-call counts per question.

Single source of truth for explore-count histograms. Produces ONE figure:

    explore_distribution_all.pdf / .png
        2 rows x 4 benchmarks. Top row = ATTS, bottom row = Self-Refine.
        Referenced by Figure 4 in main.tex.

Budget Forcing is intentionally not plotted because its distribution is
degenerate at T=8 by construction (see App E in main.tex); its mean is
still printed in the console summary for reference.

ALGORITHM (independent of code)
    For each (benchmark, method) cell:
      1. Load the per-question records from that method's eval log
      2. Apply the method's "calls-per-question" extractor to each record
      3. Histogram the resulting counts (x = number of calls, y = number of
         questions)
    Plot grid; annotate each bar with the exact question count; report n
    and mean per panel.

The "extractor" is per-method because future methods may store the count
under a different field. Today all three methods (ATTS / Self-Refine /
Budget Forcing) populate the `num_explores` field, so the same extractor
works for all of them. To add a new method, register an extractor + add
a NAMESPACE_FOR_METHOD entry; do not touch the plotting code.

No arguments; all paths and method registry hardcoded.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # dr-claw/Explain (was parents[2]; pre-existing bug -- file lives under Experiment/analysis/plots, so parents[3] is Explain)
RUN_DIR = PROJECT_ROOT / "Experiment/analysis/run"
FIGURES_DIR = PROJECT_ROOT / "Publication/paper/figures"

# x-axis upper bound for "explore calls per question". T=8 is the budget cap
# for ATTS / Self-Refine / Budget Forcing on every benchmark we report.
XMAX = 8


# ----- per-method extractors -------------------------------------------------
# Each extractor maps one results.jsonl record to "calls per question" (int).
# All three current methods populate `num_explores` so they share an extractor;
# the registry pattern is here so a future method that does not can plug in
# its own extractor without changing the plotting code.

def _from_num_explores(record: dict) -> int:
    return int(record["num_explores"])


METHOD_EXTRACTORS = {
    "ATTS":           _from_num_explores,
    "Self-Refine":    _from_num_explores,
    "Budget Forcing": _from_num_explores,
    "Socratic Self-Refine": _from_num_explores,
}

METHOD_COLORS = {
    "ATTS":           "#4C78A8",  # blue   -- default ATTS (no integrator)
    "Self-Refine":    "#54A24B",  # green  -- iterative critic-revise
    "Budget Forcing": "#E45756",  # red    -- forced N rounds with "Wait"
    "Socratic Self-Refine": "#2E7D32",  # dark green -- match scatter plot
}

# Cache namespace -> method label. The cache namespace is the name under
# Experiment/analysis/run/<bench>/<NAMESPACE>/run_*. Listed once so future
# additions stay consistent.
NAMESPACE_FOR_METHOD = {
    "ATTS":           "sonnet_no_integrate",
    "Self-Refine":    "sonnet_self_refine",
    "Budget Forcing": "sonnet_budget_forcing",
    "Socratic Self-Refine": "sonnet_socratic_self_refine",
}

# Per-(method, benchmark) panels that should NOT be plotted because the
# run is incomplete. Listed explicitly so a partial-run histogram is never
# silently passed off as a finished distribution.
INCOMPLETE_PANELS: set[tuple[str, str]] = {
    ("Socratic Self-Refine", "lcb"),
    ("Socratic Self-Refine", "gpqa"),
    ("Socratic Self-Refine", "babyvision"),
}

BENCHMARKS = [
    ("HLE-Verified",  "hle"),
    ("LiveCodeBench", "lcb"),
    ("GPQA-Diamond",  "gpqa"),
    ("BabyVision",    "babyvision"),
]


def _latest_results_jsonl(bench_dir_name: str, namespace: str) -> Path:
    method_dir = RUN_DIR / bench_dir_name / namespace
    candidates = sorted(method_dir.glob("run_*/results.jsonl"))
    assert candidates, f"no results.jsonl under {method_dir}"
    return candidates[-1]


def load_counts(bench_dir_name: str, method_label: str) -> list[int]:
    namespace = NAMESPACE_FOR_METHOD[method_label]
    extractor = METHOD_EXTRACTORS[method_label]
    path = _latest_results_jsonl(bench_dir_name, namespace)
    with path.open() as f:
        return [extractor(json.loads(line)) for line in f if line.strip()]


def _draw_histogram(
    ax: plt.Axes,
    counts: list[int],
    title: str,
    color: str,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    font_scale: float = 1.0,
) -> None:
    n = len(counts)
    counter = Counter(counts)
    xs = list(range(0, XMAX + 1))
    ys = [counter.get(x, 0) for x in xs]
    mean_val = sum(x * counter[x] for x in counter) / n if n else 0.0
    ymax = max(ys) if ys else 1

    ax.bar(xs, ys, width=0.72, color=color, edgecolor="black", linewidth=0.5)
    for x, y in zip(xs, ys):
        if y > 0:
            ax.text(x, y + ymax * 0.02, str(y), ha="center", va="bottom",
                    fontsize=8 * font_scale)

    if show_xlabel:
        ax.set_xlabel("Number of explores", fontsize=10 * font_scale)
    if show_ylabel:
        ax.set_ylabel("Number of questions", fontsize=10 * font_scale)
    ax.set_xticks(xs)
    ax.tick_params(axis="both", labelsize=10 * font_scale)
    ax.set_xlim(-0.6, XMAX + 0.6)
    ax.set_ylim(0, ymax * 1.20 if ymax > 0 else 1)
    ax.set_title(f"{title}\n(n={n}, mean = {mean_val:.2f})",
                 fontsize=10 * font_scale)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def _fill_method_row(axes_row, method: str) -> None:
    """Draw histograms for one method across all benchmarks into a given row of axes."""
    for ax, (bench_label, bench_dir) in zip(axes_row, BENCHMARKS):
        counts = load_counts(bench_dir, method)
        title = f"{method} -- {bench_label}"
        _draw_histogram(ax, counts, title, METHOD_COLORS[method], font_scale=2.0)


def _draw_blank(ax: plt.Axes, title: str, font_scale: float = 1.0) -> None:
    """Render an empty placeholder panel for an unfinished run."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.5, 0.5, "run not finished", ha="center", va="center",
            transform=ax.transAxes, fontsize=10 * font_scale, color="#888888",
            style="italic")
    ax.set_title(title, fontsize=10 * font_scale)


def _fill_method_row_with_blanks(axes_row, method: str) -> None:
    """Like _fill_method_row, but skips panels listed in INCOMPLETE_PANELS."""
    for ax, (bench_label, bench_dir) in zip(axes_row, BENCHMARKS):
        title = f"{method} -- {bench_label}"
        if (method, bench_dir) in INCOMPLETE_PANELS:
            _draw_blank(ax, title, font_scale=2.0)
            continue
        counts = load_counts(bench_dir, method)
        _draw_histogram(ax, counts, title, METHOD_COLORS[method], font_scale=2.0)


def plot_atts_sr_stacked(out_stem: Path) -> None:
    """2 rows x 4 benchmarks: row 0 = ATTS, row 1 = Self-Refine. Single shared figure."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10.4))
    _fill_method_row(axes[0], "ATTS")
    _fill_method_row(axes[1], "Self-Refine")
    fig.tight_layout()
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png", dpi=200)
    plt.close(fig)


def plot_atts_sr_ssc_stacked(out_stem: Path) -> None:
    """3 rows x 4 benchmarks: ATTS / Self-Refine / Socratic Self-Refine.

    Socratic Self-Refine row only has the HLE panel filled today; the other
    three benchmarks are still running and are rendered as blank placeholders
    so the row is structurally present but does not falsely report data.
    """
    fig, axes = plt.subplots(3, 4, figsize=(22, 15.6))
    _fill_method_row(axes[0], "ATTS")
    _fill_method_row(axes[1], "Self-Refine")
    _fill_method_row_with_blanks(axes[2], "Socratic Self-Refine")
    fig.tight_layout()
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Single combined 2x4 figure: top row ATTS, bottom row Self-Refine.
    # Output filename kept as `explore_distribution_all` so main.tex Figure 4
    # reference does not need to change. Budget Forcing intentionally omitted:
    # its distribution is degenerate at T=8 by construction (see Section App E).
    out = FIGURES_DIR / "explore_distribution_all"
    plot_atts_sr_ssc_stacked(out)
    print(f"Saved: {out}.pdf  (3x4 stacked: ATTS / Self-Refine / Socratic Self-Refine)")

    print("\nDistribution summary (mean explore-count per question):")
    methods = list(METHOD_EXTRACTORS.keys())
    print(f"  {'benchmark':16s}  " + "  ".join(f"{m:>22s}" for m in methods))
    for bench_label, bench_dir in BENCHMARKS:
        means = []
        for method in methods:
            if (method, bench_dir) in INCOMPLETE_PANELS:
                means.append(f"{'(unfinished)':>22s}")
                continue
            counts = load_counts(bench_dir, method)
            means.append(f"{sum(counts)/len(counts):>22.2f}")
        print(f"  {bench_label:16s}  " + "  ".join(means))


if __name__ == "__main__":
    main()
