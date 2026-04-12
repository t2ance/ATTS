"""Plot explore-count distributions for all four main paper benchmarks
(HLE-Verified, LiveCodeBench, GPQA-Diamond, BabyVision) for the default
ATTS variant (sonnet_no_integrate, v3).

Produces a single 2x2 figure saved to Publication/paper/figures/.
No arguments; all paths hardcoded.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # dr-claw/Explain
RUN_DIR = PROJECT_ROOT / "Experiment/analysis/run"
FIGURES_DIR = PROJECT_ROOT / "Publication/paper/figures"
OUT_STEM = FIGURES_DIR / "explore_distribution_all"

BENCHMARKS = [
    ("HLE-Verified",  "hle/sonnet_no_integrate/run_20260319_003712/results.jsonl"),
    ("LiveCodeBench", "lcb/sonnet_no_integrate/run_20260317_213909/results.jsonl"),
    ("GPQA-Diamond",  "gpqa/sonnet_no_integrate/run_20260317_181859/results.jsonl"),
    ("BabyVision",    "babyvision/sonnet_no_integrate/run_20260319_021914/results.jsonl"),
]

BAR_COLOR = "#4C78A8"
XMAX = 8  # exploration budget upper bound

fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
axes_flat = axes.flatten()

for ax, (name, rel_path) in zip(axes_flat, BENCHMARKS):
    rows = [json.loads(l) for l in (RUN_DIR / rel_path).open()]
    n = len(rows)
    counter = Counter(r["num_explores"] for r in rows)
    xs = list(range(0, XMAX + 1))
    ys = [counter.get(x, 0) for x in xs]
    mean_explores = sum(x * counter[x] for x in counter) / n

    ax.bar(xs, ys, width=0.72, color=BAR_COLOR, edgecolor="black", linewidth=0.5)
    ymax = max(ys)
    for x, y in zip(xs, ys):
        if y > 0:
            ax.text(x, y + ymax * 0.02, str(y),
                    ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Number of explores", fontsize=10)
    ax.set_ylabel("Number of questions", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xlim(-0.6, XMAX + 0.6)
    ax.set_ylim(0, ymax * 1.20 if ymax > 0 else 1)
    ax.set_title(f"{name}  (n={n}, mean = {mean_explores:.2f})",
                 fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

fig.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT_STEM}.pdf")
fig.savefig(f"{OUT_STEM}.png", dpi=200)
print(f"Saved: {OUT_STEM}.pdf / .png")

# Print summary for sanity
print("\nDistribution summary:")
for name, rel_path in BENCHMARKS:
    rows = [json.loads(l) for l in (RUN_DIR / rel_path).open()]
    n = len(rows)
    counter = Counter(r["num_explores"] for r in rows)
    mean_explores = sum(x * counter[x] for x in counter) / n
    dist = {k: counter[k] for k in sorted(counter.keys())}
    print(f"  {name}: n={n}, mean={mean_explores:.2f}, dist={dist}")
