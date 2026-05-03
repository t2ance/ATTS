"""Difficulty stratification using n_correct out of 8 (objective) instead of fcm.

For each trajectory, classify by how many of the 8 cached explores produced the
gold answer (i.e. is_correct=True). This is purely objective: independent of stop
time, independent of agreement structure.

Buckets:
  - reliable (7-8/8 correct):    model's confidence interval is tight on correct
  - majority correct (4-6/8):    correct is the modal answer but with errors
  - minority correct (1-3/8):    correct is a minority of explores
  - never correct (0/8):         capability ceiling

Outputs:
  - JSON stats for each (benchmark, bucket): n, final_acc, median_gap, frac_gap_zero, minority_extraction_rate
  - Cross-tab against fcm bucketing on GPQA, showing reclassification of trajectories
  - 4-panel figure analogous to fig4 but with n_correct buckets
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/orch_evidence")
BENCHES = [("gpqa", "GPQA-Diamond"), ("hle", "HLE-Verified"), ("lcb", "LiveCodeBench"), ("babyvision", "BabyVision")]


def n_correct_bucket(n: int) -> str:
    if n >= 7: return "reliable (7-8/8)"
    if n >= 4: return "majority correct (4-6/8)"
    if n >= 1: return "minority correct (1-3/8)"
    return "never correct (0/8)"


def fcm_bucket(fcm) -> str:
    if pd.isna(fcm): return "undefined"
    if fcm == 2: return "easy"
    if fcm in (3, 4): return "medium"
    return "hard"


def get_per_qid(bench: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"{bench}_sonnet" / "pool_state.parquet")
    pq = df.groupby(["run_id", "qid"]).first().reset_index()
    nc8 = df[df["k"] == 8][["run_id", "qid", "n_correct_at_k"]].rename(
        columns={"n_correct_at_k": "n_correct_total"}
    )
    pq = pq.merge(nc8, on=["run_id", "qid"], how="left")
    ts = df[df["k"] == df["t_star"]][["run_id", "qid", "majority_is_correct_at_k"]].rename(
        columns={"majority_is_correct_at_k": "maj_at_tstar_correct"}
    )
    pq = pq.merge(ts, on=["run_id", "qid"], how="left")
    pq["minority_extraction"] = (pq["maj_at_tstar_correct"] != True) & (pq["final_is_correct"])
    pq["gap"] = pq["t_star"] - pq["first_correct_majority_emerged_at"]
    pq["nc_bucket"] = pq["n_correct_total"].apply(n_correct_bucket)
    pq["fcm_bucket"] = pq["first_correct_majority_emerged_at"].apply(fcm_bucket)
    return pq


def main() -> None:
    bucket_order = ["reliable (7-8/8)", "majority correct (4-6/8)", "minority correct (1-3/8)", "never correct (0/8)"]
    stats = {}
    for bench, _ in BENCHES:
        pq = get_per_qid(bench)
        per_bucket = {}
        for buck in bucket_order:
            sub = pq[pq["nc_bucket"] == buck]
            if len(sub) == 0:
                per_bucket[buck] = {"n": 0}
                continue
            gp = sub["gap"].dropna()
            per_bucket[buck] = {
                "n": int(len(sub)),
                "final_accuracy_pct": round(sub["final_is_correct"].mean() * 100, 2),
                "median_gap": float(gp.median()) if len(gp) else None,
                "frac_gap_zero_pct": round((gp == 0).mean() * 100, 2) if len(gp) else None,
                "minority_extraction_pct": round(sub["minority_extraction"].mean() * 100, 2),
            }
        stats[bench] = per_bucket

    # GPQA cross-tab fcm vs n_correct
    pq_gpqa = get_per_qid("gpqa")
    ct = pd.crosstab(pq_gpqa["fcm_bucket"], pq_gpqa["nc_bucket"])
    stats["_gpqa_crosstab_fcm_x_n_correct"] = ct.to_dict()

    out_path = OUT / "stats_n_correct_bucketing.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")

    # Figure: 4-panel, n_correct bucket on x-axis
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=False)
    for ax, (bench, title) in zip(axes, BENCHES):
        pq = get_per_qid(bench)
        accs, mes, ns = [], [], []
        for buck in bucket_order:
            sub = pq[pq["nc_bucket"] == buck]
            ns.append(len(sub))
            accs.append(sub["final_is_correct"].mean() * 100 if len(sub) else 0)
            mes.append(sub["minority_extraction"].mean() * 100 if len(sub) else 0)
        x = np.arange(4)
        ax.bar(x - 0.2, accs, 0.4, color="#1f77b4", label="Final accuracy", edgecolor="black")
        ax.bar(x + 0.2, mes, 0.4, color="#d62728", label="Minority extraction %", edgecolor="black", alpha=0.85)
        for i, n in enumerate(ns):
            ax.text(i, max(accs[i], mes[i]) + 3, f"n={n}", ha="center", fontsize=8)
            if accs[i] > 0:
                ax.text(i - 0.2, accs[i] + 0.5, f"{accs[i]:.0f}", ha="center", fontsize=8, color="#1f77b4", fontweight="bold")
            if mes[i] > 0:
                ax.text(i + 0.2, mes[i] + 0.5, f"{mes[i]:.0f}", ha="center", fontsize=8, color="#d62728", fontweight="bold")
        labels = ["7-8/8\nreliable", "4-6/8\nmajority", "1-3/8\nminority", "0/8\nnever"]
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Percent of trajectories (%)")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("ATTS behavior stratified by objective question difficulty (n_correct out of 8 cached explores)\n4 benchmarks, Sonnet 4.6 single-model ATTS", fontsize=12, y=1.02)
    fig.tight_layout()
    fig_path = OUT / "fig6_n_correct_difficulty.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=120, bbox_inches="tight")
    print(f"Wrote {fig_path}.{{pdf,png}}")


if __name__ == "__main__":
    main()
