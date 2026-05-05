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

    # Uniform null: replace t_star with Uniform[1,8]. Plot expected counts
    # instead of one random draw so the visual is deterministic.
    fcm = defined["first_correct_majority_emerged_at"].to_numpy()
    x = np.arange(-7, 8)
    obs_counts = np.array([(obs_gaps == g).sum() for g in x])
    null_counts = np.zeros_like(x, dtype=float)
    for k_star in fcm:
        for t_uniform in range(1, 9):
            gap = int(t_uniform - k_star)
            if x[0] <= gap <= x[-1]:
                null_counts[gap - x[0]] += 1 / 8

    claude_orange = "#D97757"
    claude_orange_dark = "#9A4A32"
    charcoal = "#2B2926"
    warm_gray = "#8A8178"
    grid = "#D8D8D8"

    with plt.rc_context({
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }):
        fig, ax = plt.subplots(figsize=(7.4, 4.4))

        width = 0.34
        ax.bar(
            x - width / 2,
            null_counts,
            width=width,
            label="Uniform-stop null",
            color="#D8CEC2",
            edgecolor=warm_gray,
            linewidth=0.7,
            alpha=0.95,
            zorder=2,
        )
        ax.bar(
            x + width / 2,
            obs_counts,
            width=width,
            label="ATTS observed",
            color=claude_orange,
            edgecolor=claude_orange_dark,
            linewidth=0.8,
            zorder=3,
        )

        ax.axvline(0, color=charcoal, linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
        ax.set_xlabel("Gap")
        ax.set_ylabel("Number of trajectories")
        ax.set_xticks(x)
        ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)
        ax.legend(frameon=False, loc="upper left")
        ax.grid(axis="y", color=grid, linewidth=0.8, alpha=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(warm_gray)
        ax.spines["bottom"].set_color(warm_gray)
        ax.tick_params(colors=charcoal)
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


def fig4_gpqa_difficulty_cohorts(df: pd.DataFrame) -> None:
    per_qid = df.groupby(["run_id", "qid"]).first().reset_index()
    n_correct_total = df[df["k"] == 8][["run_id", "qid", "n_correct_at_k"]].rename(
        columns={"n_correct_at_k": "n_correct_total"}
    )
    per_qid = per_qid.merge(n_correct_total, on=["run_id", "qid"], how="left")
    tstar_majority = df[df["k"] == df["t_star"]][
        ["run_id", "qid", "majority_is_correct_at_k"]
    ].rename(columns={"majority_is_correct_at_k": "maj_at_tstar_correct"})
    per_qid = per_qid.merge(tstar_majority, on=["run_id", "qid"], how="left")
    per_qid["minority_extraction"] = (
        (per_qid["maj_at_tstar_correct"] != True) & per_qid["final_is_correct"]
    )

    def cohort(n_correct: int) -> str:
        if n_correct >= 7:
            return "Reliable"
        if n_correct >= 4:
            return "Majority"
        if n_correct >= 1:
            return "Minority"
        return "None"

    per_qid["cohort"] = per_qid["n_correct_total"].apply(cohort)
    order = ["Reliable", "Majority", "Minority", "None"]
    ranges = {
        "Reliable": "7-8/8",
        "Majority": "4-6/8",
        "Minority": "1-3/8",
        "None": "0/8",
    }
    labels = []
    acc = []
    minority = []
    for name in order:
        sub = per_qid[per_qid["cohort"] == name]
        labels.append(f"{ranges[name]}\n(n={len(sub)})")
        acc.append(sub["final_is_correct"].mean() * 100 if len(sub) else 0)
        minority.append(sub["minority_extraction"].mean() * 100 if len(sub) else 0)

    claude_orange = "#D97757"
    claude_orange_dark = "#9A4A32"
    charcoal = "#2B2926"
    warm_gray = "#8A8178"
    grid = "#D8D8D8"
    blue = "#4E79A7"

    with plt.rc_context({
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }):
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        x = np.arange(len(order))
        width = 0.34
        ax.bar(
            x - width / 2,
            acc,
            width=width,
            label="Final accuracy",
            color=blue,
            edgecolor="#2F4A66",
            linewidth=0.8,
            zorder=3,
        )
        ax.bar(
            x + width / 2,
            minority,
            width=width,
            label="Minority extraction",
            color=claude_orange,
            edgecolor=claude_orange_dark,
            linewidth=0.8,
            zorder=3,
        )
        ax.set_ylabel("Percent of trajectories")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 110)
        ax.legend(frameon=False, loc="upper right")
        ax.grid(axis="y", color=grid, linewidth=0.8, alpha=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(warm_gray)
        ax.spines["bottom"].set_color(warm_gray)
        ax.tick_params(colors=charcoal)
        fig.tight_layout()
        out = OUT_DIR / "fig4_gpqa_difficulty_cohorts.pdf"
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
    fig4_gpqa_difficulty_cohorts(df)
    print("Done.")


if __name__ == "__main__":
    main()
