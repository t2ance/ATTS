"""Build per-(run, qid, k) pool-state feature table from 5 GPQA ATTS stability runs.

Input: 5 results.jsonl from `analysis/run/gpqa/sonnet_no_integrate_run{1..5}/run_20260328_060344/`.
Output: `analysis/orch_evidence/gpqa_sonnet/pool_state.parquet` with one row per (run, qid, k).

Algorithm (no model calls, pure offline reconstruction):
  For each (run, qid), walk explores in order k=1..8.
  At each k, define pool C_k = {explores 1..k} and compute:
    - n_correct_at_k:           sum of is_correct over C_k
    - majority_answer_at_k:     mode of normalized_answer in C_k (None on tie or count<2)
    - majority_count_at_k:      count of the modal answer
    - majority_is_correct_at_k: True if mode equals gold (case-insensitive); None if no majority
  Per-qid summary:
    - first_majority_emerged_at:         smallest k where majority_count_at_k >= 2
    - first_correct_majority_emerged_at: smallest k where majority_is_correct_at_k == True
  Broadcast per-qid summaries onto every k row of that qid for easy join later.

Excluded: 3 anomalous qids with precache failures (`recK9F5aqdaybl8bb`,
`recRgabRzMaEoBRcM`, `recZ13cwgDQf9jRd9`) → 195 qids × 5 runs × 8 k = 7800 rows.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Reuse grader's MC letter extractor (handles "(a)", "b)", "answer is c", etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from benchmarks.grader import _extract_mc_letter, normalize_answer  # noqa: E402

RUNS = {
    f"run{i}": Path(
        f"/data3/peijia/dr-claw/Explain/Experiment/analysis/run/gpqa/sonnet_no_integrate_run{i}/run_20260328_060344/results.jsonl"
    )
    for i in range(1, 6)
}
ANOMALOUS_QIDS = {"recK9F5aqdaybl8bb", "recRgabRzMaEoBRcM", "recZ13cwgDQf9jRd9"}
OUT = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/orch_evidence/gpqa_sonnet/pool_state.parquet")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_letter(normalized_answer: str | None) -> str | None:
    """GPQA-specific: collapse 'b) text...' / '(B)' / 'b' all to 'b'.
    Uses the grader's _extract_mc_letter so majority computation uses the same
    letter-level semantics as the stored is_correct flag.
    """
    if normalized_answer is None:
        return None
    s = normalize_answer(normalized_answer)
    return _extract_mc_letter(s)


def compute_pool_features(letters: list[str | None], correct_flags: list[bool], gold_letter: str) -> list[dict]:
    """For each prefix length k=1..8, compute pool-state features at letter level."""
    out = []
    for k in range(1, len(letters) + 1):
        prefix_letters = [l for l in letters[:k] if l is not None]
        prefix_correct = correct_flags[:k]
        n_correct = sum(prefix_correct)
        counts = Counter(prefix_letters)
        most = counts.most_common(2)
        if most and most[0][1] >= 2 and (len(most) == 1 or most[0][1] > most[1][1]):
            maj_answer = most[0][0]
            maj_count = most[0][1]
            maj_correct = (maj_answer == gold_letter)
        else:
            maj_answer = None
            maj_count = most[0][1] if most else 0
            maj_correct = None
        out.append({
            "k": k,
            "n_correct_at_k": n_correct,
            "majority_answer_at_k": maj_answer,
            "majority_count_at_k": maj_count,
            "majority_is_correct_at_k": maj_correct,
        })
    return out


def main() -> None:
    rows = []
    for run_label, jsonl_path in RUNS.items():
        traj_rows = load_jsonl(jsonl_path)
        for r in traj_rows:
            qid = r["id"]
            if qid in ANOMALOUS_QIDS:
                continue
            ec = r["explore_candidates"]
            assert len(ec) == 8, f"clean cohort invariant: {run_label}/{qid} has len(ec)={len(ec)}"
            letters = [to_letter(c["normalized_answer"]) for c in ec]
            correct_flags = [bool(c["is_correct"]) for c in ec]
            gold_letter = to_letter(r["gold_answer"])
            assert gold_letter in {"a", "b", "c", "d", "e"}, f"gold extraction failed: {r['gold_answer']!r}"

            features = compute_pool_features(letters, correct_flags, gold_letter)

            # Per-qid summary
            first_majority = next(
                (f["k"] for f in features if f["majority_count_at_k"] >= 2),
                None,
            )
            first_correct_majority = next(
                (f["k"] for f in features if f["majority_is_correct_at_k"] is True),
                None,
            )

            for f in features:
                rows.append({
                    "run_id": run_label,
                    "qid": qid,
                    **f,
                    "first_majority_emerged_at": first_majority,
                    "first_correct_majority_emerged_at": first_correct_majority,
                    "t_star": r["num_explores"],
                    "final_is_correct": bool(r["is_correct"]),
                    "first_candidate_correct": bool(r["first_candidate_correct"]),
                })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    # Self-check gates
    assert len(df) == 7800, f"expected 7800 rows, got {len(df)}"
    assert df["k"].between(1, 8).all()
    assert df["t_star"].between(1, 8).all()
    g4_violations = df[
        df["first_correct_majority_emerged_at"].notna()
        & df["first_majority_emerged_at"].notna()
        & (df["first_correct_majority_emerged_at"] < df["first_majority_emerged_at"])
    ]
    assert len(g4_violations) == 0, f"invariant violation: {len(g4_violations)} rows where first_correct_majority < first_majority"

    n_qids = df["qid"].nunique()
    n_traj = df.groupby(["run_id", "qid"]).ngroups
    print(f"Wrote {OUT}  ({len(df)} rows, {n_qids} unique qids, {n_traj} trajectories)")
    print(f"Schema: {list(df.columns)}")
    print(f"\nSample first 3 rows:\n{df.head(3).to_string()}")
    print(f"\nPer-run trajectory counts:\n{df.groupby('run_id')['qid'].nunique()}")


if __name__ == "__main__":
    main()
