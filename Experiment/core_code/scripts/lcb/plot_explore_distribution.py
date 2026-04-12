"""Plot distribution of questions by number of explores used on LiveCodeBench
for the no_integrate (v3, default ATTS) variant.

No arguments; all paths hardcoded.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # dr-claw/Explain
RESULTS_PATH = (
    PROJECT_ROOT
    / "Experiment/analysis/run/lcb/sonnet_no_integrate/run_20260317_213909/results.jsonl"
)
FIGURES_DIR = PROJECT_ROOT / "Publication/paper/figures"
OUT_STEM = FIGURES_DIR / "lcb_explore_distribution"

rows = [json.loads(l) for l in RESULTS_PATH.open()]
total = len(rows)
counter = Counter(r["num_explores"] for r in rows)
xs = sorted(counter.keys())
ys = [counter[x] for x in xs]
mean_explores = sum(x * counter[x] for x in xs) / total

fig, ax = plt.subplots(figsize=(5.2, 3.2))
bars = ax.bar(xs, ys, width=0.72, color="#4C78A8", edgecolor="black", linewidth=0.6)

for x, y in zip(xs, ys):
    ax.text(x, y + 1.2, str(y), ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Number of explores")
ax.set_ylabel("Number of questions")
ax.set_xticks(xs)
ax.set_ylim(0, max(ys) * 1.18)
ax.set_title(
    f"LiveCodeBench: Explore-count distribution (ATTS, no integrate)\n"
    f"n={total} questions, mean = {mean_explores:.2f} explores/question",
    fontsize=10,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)

fig.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT_STEM}.pdf")
fig.savefig(f"{OUT_STEM}.png", dpi=200)
print(f"Saved: {OUT_STEM}.pdf / .png")
print(f"Distribution: {dict(sorted(counter.items()))}")
print(f"Mean explores: {mean_explores:.3f}")
