"""Build pool_state.parquet for HLE / LCB / BabyVision in addition to GPQA.

Key difference from GPQA: non-multiple-choice benchmarks have free-form answers,
so majority is computed by string equality of `normalized_answer` (lowercased,
stripped) rather than letter extraction. is_correct is taken as-stored
(grader-validated already).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from benchmarks.grader import _extract_mc_letter, normalize_answer  # noqa: E402

OUT_BASE = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/orch_evidence")

# Per-benchmark canonical run dirs (Sonnet ATTS no-integrate)
BENCH_RUNS = {
    "gpqa": [Path(f"/data3/peijia/dr-claw/Explain/Experiment/analysis/run/gpqa/sonnet_no_integrate_run{i}/run_20260328_060344/results.jsonl") for i in range(1, 6)],
    "hle": [Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_no_integrate/run_20260319_003712/results.jsonl")],
    "lcb": [Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet/run_20260308_230222/results.jsonl")],
    "babyvision": [Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/run/babyvision/sonnet_no_integrate/run_20260319_021914/results.jsonl")],
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_key(answer: str | None, benchmark: str) -> str | None:
    """Convert normalized_answer to a comparable key for majority counting.
    GPQA: extract letter. Others: lowercased+stripped raw normalized_answer.
    """
    if answer is None:
        return None
    s = normalize_answer(answer)
    if benchmark == "gpqa":
        return _extract_mc_letter(s)
    if not s:
        return None
    return s


def compute_features(keys: list[str | None], correct: list[bool]) -> list[dict]:
    """Benchmark-agnostic: majority_is_correct uses the stored is_correct of any explore
    in the majority cluster (same key → same is_correct since the grader is deterministic)."""
    out = []
    for k in range(1, len(keys) + 1):
        prefix_indexed = [(i, x) for i, x in enumerate(keys[:k]) if x is not None]
        n_correct = sum(correct[:k])
        counts = Counter(x for _, x in prefix_indexed)
        most = counts.most_common(2)
        if most and most[0][1] >= 2 and (len(most) == 1 or most[0][1] > most[1][1]):
            maj_answer = most[0][0]
            maj_count = most[0][1]
            # Any explore index with this key — look up its is_correct
            sample_idx = next(i for i, x in prefix_indexed if x == maj_answer)
            maj_correct = bool(correct[sample_idx])
        else:
            maj_answer, maj_count, maj_correct = None, (most[0][1] if most else 0), None
        out.append({"k": k, "n_correct_at_k": n_correct,
                    "majority_answer_at_k": maj_answer, "majority_count_at_k": maj_count,
                    "majority_is_correct_at_k": maj_correct})
    return out


def main() -> None:
    for bench, paths in BENCH_RUNS.items():
        print(f"\n=== {bench} ===")
        rows_out = []
        n_anom = 0
        for path in paths:
            # use the method-folder name as run_label so multiple stability runs don't collide
            run_label = f"{path.parent.parent.name}/{path.parent.name}"
            traj_rows = load_jsonl(path)
            for r in traj_rows:
                ec = r.get("explore_candidates") or []
                if len(ec) != 8:
                    n_anom += 1
                    continue
                # Skip rows where any candidate has is_correct=None (precache failure)
                if any(c.get("is_correct") is None for c in ec):
                    n_anom += 1
                    continue
                keys = [to_key(c.get("normalized_answer"), bench) for c in ec]
                correct = [bool(c["is_correct"]) for c in ec]
                feats = compute_features(keys, correct)
                first_majority = next((f["k"] for f in feats if f["majority_count_at_k"] >= 2), None)
                first_correct_majority = next((f["k"] for f in feats if f["majority_is_correct_at_k"] is True), None)
                for f in feats:
                    rows_out.append({
                        "benchmark": bench, "run_id": run_label, "qid": r["id"],
                        **f,
                        "first_majority_emerged_at": first_majority,
                        "first_correct_majority_emerged_at": first_correct_majority,
                        "t_star": r["num_explores"], "final_is_correct": bool(r["is_correct"]),
                        "first_candidate_correct": bool(r["first_candidate_correct"]),
                    })
        df = pd.DataFrame(rows_out)
        out_dir = OUT_BASE / f"{bench}_sonnet"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pool_state.parquet"
        df.to_parquet(out_path, index=False)
        n_traj = df.groupby(["run_id", "qid"]).ngroups
        n_qids = df["qid"].nunique()
        print(f"  wrote {out_path}: {len(df)} rows, {n_qids} qids, {n_traj} trajectories, {n_anom} anomalies excluded")


if __name__ == "__main__":
    main()
