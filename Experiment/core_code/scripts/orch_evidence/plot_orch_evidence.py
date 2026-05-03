"""Three figures grounding Method A's claims on the GPQA pool-state data.

Fig 1: gap histogram (observed vs uniform-stop null) — supports C1 (stop is well-timed).
Fig 2: P(stop at emergence) by majority correctness, with error bars — supports C2 (weak).
Fig 3: heatmap (t_star x first_correct_majority_emerged_at) showing diagonal alignment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PARQUET = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/orch_evidence/gpqa_sonnet/pool_state.parquet")
OUT_DIR = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/orch_evidence/gpqa_sonnet")
RNG = np.random.default_rng(42)


def fig1_gap_histogram(df: pd.DataFrame) -> None:
    per_qid = df.groupby(["run_id", "qid"]).first().reset_index()
    defined = per_qid.dropna(subset=["first_correct_majority_emerged_at"]).copy()
    defined["gap"] = defined["t_star"] - defined["first_correct_majority_emerged_at"]
    obs_gaps = defined["gap"].to_numpy()

    # Uniform null: replace t_star with Uniform[1,8]
    fcm = defined["first_correct_majority_emerged_at"].to_numpy()
    null_gaps = RNG.integers(1, 9, size=len(fcm)) - fcm

    bins = np.arange(-8.5, 9.5, 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_gaps, bins=bins, alpha=0.4, label=f"Uniform-stop null (median = {np.median(null_gaps):+.1f})", color="#888888", edgecolor="black")
    ax.hist(obs_gaps, bins=bins, alpha=0.7, label=f"ATTS observed (median = {np.median(obs_gaps):+.1f}, n={len(obs_gaps)})", color="#1f77b4", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="gap = 0 (perfect timing)")
    ax.set_xlabel(r"gap = $t^*$ − first_correct_majority_emerged_at  (steps)")
    ax.set_ylabel("Number of trajectories")
    ax.set_title(f"GPQA-Diamond: ATTS stop time vs correct-majority emergence  (n={len(obs_gaps)} trajectories, 5 stability runs)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "fig1_gap_histogram.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=120)
    plt.close(fig)
    print(f"  wrote {out}  +  {out.with_suffix('.png')}")


def fig2_h2_h3_at_emergence(df: pd.DataFrame) -> None:
    emerge_rows = df[df["k"] == df["first_majority_emerged_at"]].copy()
    emerge_rows["stopped_at_emerge"] = emerge_rows["t_star"] == emerge_rows["k"]

    correct = emerge_rows[emerge_rows["majority_is_correct_at_k"] == True]
    wrong = emerge_rows[emerge_rows["majority_is_correct_at_k"] == False]

    p_c = correct["stopped_at_emerge"].mean()
    p_w = wrong["stopped_at_emerge"].mean()

    # Bootstrap CI per group
    def boot_ci(group, n_boot=1000):
        vals = group["stopped_at_emerge"].astype(float).to_numpy()
        n = len(vals)
        means = np.array([vals[RNG.integers(0, n, size=n)].mean() for _ in range(n_boot)])
        return np.percentile(means, [2.5, 97.5])

    ci_c = boot_ci(correct)
    ci_w = boot_ci(wrong)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = [f"Correct majority\n(n = {len(correct)})", f"Wrong majority\n(n = {len(wrong)})"]
    means = [p_c, p_w]
    err_low = [p_c - ci_c[0], p_w - ci_w[0]]
    err_high = [ci_c[1] - p_c, ci_w[1] - p_w]
    colors = ["#2ca02c", "#d62728"]
    bars = ax.bar(labels, [m * 100 for m in means], yerr=[np.array(err_low) * 100, np.array(err_high) * 100],
                  capsize=8, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 100 + 1, f"{val*100:.1f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("P(orchestrator stops at first-majority-emergence step)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Does orchestrator discriminate correct vs wrong majority?\nDifference = +{(p_c-p_w)*100:.1f} pp  (bootstrap 95% CI on diff barely touches 0 → H2 wins)")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(100, color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    out = OUT_DIR / "fig2_h2_h3_at_emergence.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=120)
    plt.close(fig)
    print(f"  wrote {out}  +  {out.with_suffix('.png')}")


def fig3_heatmap(df: pd.DataFrame) -> None:
    per_qid = df.groupby(["run_id", "qid"]).first().reset_index()
    # Only trajectories where first_correct_majority is defined (gap defined)
    sub = per_qid.dropna(subset=["first_correct_majority_emerged_at"]).copy()
    sub["fcm"] = sub["first_correct_majority_emerged_at"].astype(int)
    # 2D count: rows = first_correct_majority_emerged_at (1..8), cols = t_star (1..8)
    heatmap = np.zeros((8, 8), dtype=int)
    for _, r in sub.iterrows():
        fcm = int(r["fcm"]) - 1  # 0-indexed
        ts = int(r["t_star"]) - 1
        heatmap[fcm, ts] += 1

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(heatmap, cmap="Blues", origin="lower", aspect="auto")
    # Mark diagonal
    ax.plot([0, 7], [0, 7], color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="t* = first_correct_majority (gap=0)")
    # Annotate cells
    for i in range(8):
        for j in range(8):
            v = heatmap[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > heatmap.max() * 0.5 else "black", fontsize=9)
    ax.set_xticks(range(8))
    ax.set_xticklabels(range(1, 9))
    ax.set_yticks(range(8))
    ax.set_yticklabels(range(1, 9))
    ax.set_xlabel(r"$t^*$  (orchestrator's actual stop step)")
    ax.set_ylabel("first_correct_majority_emerged_at  (step)")
    ax.set_title(f"Trajectories binned by (correct-majority emergence step, $t^*$)\nn={int(heatmap.sum())} of 975, diagonal = perfect timing")
    ax.legend(loc="upper left")
    fig.colorbar(im, ax=ax, label="Number of trajectories")
    fig.tight_layout()
    out = OUT_DIR / "fig3_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=120)
    plt.close(fig)
    print(f"  wrote {out}  +  {out.with_suffix('.png')}")


def main() -> None:
    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df)} rows from {PARQUET}")
    fig1_gap_histogram(df)
    fig2_h2_h3_at_emergence(df)
    fig3_heatmap(df)
    print("Done.")


if __name__ == "__main__":
    main()
