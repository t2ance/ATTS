"""Compute gap distribution + H2-vs-H3 contrast stats from pool_state.parquet.

Outputs to `analysis/orch_evidence/gpqa_sonnet/stats.json`:
  gap_distribution:
    per-run + pooled
    - n_total, n_undefined (no correct majority ever)
    - median_gap, mean_gap, frac_gap_zero, frac_gap_within_pm1
    - bootstrap_ci_median_gap (qid-level, 1000 resamples) for both ATTS and uniform-null
  h2_h3:
    per step k=2..8 + aggregated
    - P_stop_at_k_given_correct_majority
    - P_stop_at_k_given_wrong_majority
    - log_ratio = log(P_correct / P_wrong)
    - bootstrap_ci_log_ratio (qid-level, 1000 resamples)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/orch_evidence/gpqa_sonnet/pool_state.parquet")
OUT = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/orch_evidence/gpqa_sonnet/stats.json")
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)


def per_qid_view(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (run_id, qid) keeping per-qid columns."""
    return df.groupby(["run_id", "qid"], as_index=False).first()[
        [
            "run_id",
            "qid",
            "first_majority_emerged_at",
            "first_correct_majority_emerged_at",
            "t_star",
            "final_is_correct",
        ]
    ]


def gap_summary(per_qid: pd.DataFrame) -> dict:
    defined = per_qid.dropna(subset=["first_correct_majority_emerged_at"]).copy()
    defined["gap"] = defined["t_star"] - defined["first_correct_majority_emerged_at"]
    n_undef = len(per_qid) - len(defined)
    return {
        "n_total_clean": int(len(per_qid)),
        "n_correct_majority_undefined": int(n_undef),
        "frac_undefined": round(n_undef / len(per_qid), 4),
        "n_defined": int(len(defined)),
        "median_gap": float(defined["gap"].median()),
        "mean_gap": float(defined["gap"].mean()),
        "std_gap": float(defined["gap"].std()),
        "frac_gap_zero": float((defined["gap"] == 0).mean()),
        "frac_gap_within_pm1": float((defined["gap"].abs() <= 1).mean()),
        "frac_gap_negative": float((defined["gap"] < 0).mean()),
        "frac_gap_positive": float((defined["gap"] > 0).mean()),
    }


def bootstrap_median_gap(per_qid_defined: pd.DataFrame, n_boot: int = N_BOOTSTRAP) -> dict:
    """Qid-level bootstrap: resample (run_id, qid) pairs with replacement."""
    gaps = per_qid_defined["gap"].to_numpy()
    n = len(gaps)
    medians = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        medians[i] = np.median(gaps[idx])
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return {"median": float(np.median(gaps)), "ci_low": float(lo), "ci_high": float(hi)}


def uniform_null_gap(per_qid: pd.DataFrame, n_boot: int = N_BOOTSTRAP) -> dict:
    """Null distribution: replace t* with uniform[1,8] sampling, recompute gap."""
    defined = per_qid.dropna(subset=["first_correct_majority_emerged_at"]).copy()
    fcm = defined["first_correct_majority_emerged_at"].to_numpy()
    n = len(defined)
    medians = np.empty(n_boot)
    for i in range(n_boot):
        t_uniform = RNG.integers(1, 9, size=n)  # uniform on {1,...,8}
        gaps = t_uniform - fcm
        medians[i] = np.median(gaps)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return {"median": float(np.median(medians)), "ci_low": float(lo), "ci_high": float(hi)}


def h2_h3_contrast(df: pd.DataFrame, n_boot: int = N_BOOTSTRAP) -> dict:
    """P(stop at k | majority correct) vs P(stop at k | majority wrong).

    Vectorized qid-level bootstrap: pre-aggregate per (run_id, qid) into 4 counters
      n_stop_correct, n_correct, n_stop_wrong, n_wrong
    Then resample qid indices and sum — O(N_qids * n_boot) integer ops, no pandas.
    """
    cells = df[df["majority_is_correct_at_k"].notna()].copy()
    cells["stopped_here"] = cells["k"] == cells["t_star"]
    cells["is_correct_majority"] = cells["majority_is_correct_at_k"].astype(bool)

    # Per (run_id, qid): four counters
    grp = cells.groupby(["run_id", "qid"])
    n_stop_correct = grp.apply(lambda g: int(((g["is_correct_majority"]) & (g["stopped_here"])).sum())).to_numpy()
    n_correct = grp.apply(lambda g: int((g["is_correct_majority"]).sum())).to_numpy()
    n_stop_wrong = grp.apply(lambda g: int((~g["is_correct_majority"] & g["stopped_here"]).sum())).to_numpy()
    n_wrong = grp.apply(lambda g: int((~g["is_correct_majority"]).sum())).to_numpy()
    n_qids = len(n_correct)

    # Observed
    p_stop_correct = n_stop_correct.sum() / n_correct.sum() if n_correct.sum() > 0 else float("nan")
    p_stop_wrong = n_stop_wrong.sum() / n_wrong.sum() if n_wrong.sum() > 0 else float("nan")
    log_ratio_obs = math.log(p_stop_correct / p_stop_wrong) if p_stop_wrong > 0 and p_stop_correct > 0 else float("inf")

    # Vectorized bootstrap: sample n_qids indices, sum the 4 counters
    idx = RNG.integers(0, n_qids, size=(n_boot, n_qids))
    sc = n_stop_correct[idx].sum(axis=1)
    nc = n_correct[idx].sum(axis=1)
    sw = n_stop_wrong[idx].sum(axis=1)
    nw = n_wrong[idx].sum(axis=1)
    valid = (nc > 0) & (nw > 0) & (sc > 0) & (sw > 0)
    pc = np.where(valid, sc / np.where(nc > 0, nc, 1), np.nan)
    pw = np.where(valid, sw / np.where(nw > 0, nw, 1), np.nan)
    lr = np.where(valid, np.log(np.where(valid, pc, 1)) - np.log(np.where(valid, pw, 1)), np.nan)
    lr_kept = lr[np.isfinite(lr)]
    if len(lr_kept) >= 100:
        ci_lo, ci_hi = np.percentile(lr_kept, [2.5, 97.5])
    else:
        ci_lo, ci_hi = float("nan"), float("nan")

    return {
        "n_correct_majority_cells": int(n_correct.sum()),
        "n_wrong_majority_cells": int(n_wrong.sum()),
        "p_stop_given_correct_majority": float(p_stop_correct),
        "p_stop_given_wrong_majority": float(p_stop_wrong),
        "log_ratio_observed": float(log_ratio_obs) if math.isfinite(log_ratio_obs) else "inf",
        "log_ratio_ci_low": float(ci_lo),
        "log_ratio_ci_high": float(ci_hi),
        "bootstrap_n_resamples_kept": int(len(lr_kept)),
    }


def per_step_h2_h3(df: pd.DataFrame) -> dict:
    """Step-by-step P(stop at k | majority correct) vs P(stop at k | majority wrong)."""
    cells = df[df["majority_is_correct_at_k"].notna()].copy()
    cells["stopped_here"] = cells["k"] == cells["t_star"]
    out = {}
    for k in range(2, 9):
        slice_k = cells[cells["k"] == k]
        if len(slice_k) == 0:
            continue
        c = slice_k[slice_k["majority_is_correct_at_k"] == True]
        w = slice_k[slice_k["majority_is_correct_at_k"] == False]
        out[f"k={k}"] = {
            "n_correct_majority": int(len(c)),
            "n_wrong_majority": int(len(w)),
            "p_stop_given_correct": float(c["stopped_here"].mean()) if len(c) else None,
            "p_stop_given_wrong": float(w["stopped_here"].mean()) if len(w) else None,
        }
    return out


def main(analysis: str) -> None:
    df = pd.read_parquet(PARQUET)
    per_qid = per_qid_view(df)
    out = {}

    if analysis in ("gap", "all"):
        out["gap_pooled"] = gap_summary(per_qid)
        defined = per_qid.dropna(subset=["first_correct_majority_emerged_at"]).copy()
        defined["gap"] = defined["t_star"] - defined["first_correct_majority_emerged_at"]
        out["gap_pooled"]["bootstrap_ci_median"] = bootstrap_median_gap(defined)
        out["gap_pooled"]["uniform_null_median_ci"] = uniform_null_gap(per_qid)
        # Per-run
        out["gap_per_run"] = {}
        for run in sorted(per_qid["run_id"].unique()):
            sub = per_qid[per_qid["run_id"] == run]
            out["gap_per_run"][run] = gap_summary(sub)

    if analysis in ("h2_h3", "all"):
        out["h2_h3_pooled"] = h2_h3_contrast(df)
        out["h2_h3_per_step"] = per_step_h2_h3(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", choices=["gap", "h2_h3", "all"], default="all")
    args = p.parse_args()
    main(args.analysis)
