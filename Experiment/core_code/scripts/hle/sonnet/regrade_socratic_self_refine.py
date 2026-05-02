"""Re-grade sonnet_socratic_self_refine results with Haiku judge.

The original run was launched before benchmarks/hle.py was patched to restore
judge_model="claude-haiku-4-5-20251001". As a result, all 100 grade.json files
under run_20260427_071039/grading/<qid>/grade.json were written with
judge_model="none", and is_correct fell back to str(predicted) == str(gold)
exact match. HLE answers are LaTeX/text and need semantic-equivalence judging,
so accuracy was understated by ~8 pp on this run alone.

This script:
  1. Reads each row from results.jsonl (final integrated answer).
  2. Calls the canonical grader with Haiku as judge_model.
  3. Overwrites grade.json under grading/<qid>/.
  4. Rewrites results.jsonl with corrected is_correct + judge_cost_usd fields.
  5. Updates progress.json acc.

It does NOT re-run the explore/refine pipeline. Trajectories and final
predictions are preserved as-is; only the grade is recomputed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# repo bootstrap (mirrors eval.py)
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from benchmarks import get_benchmark
from logger import setup_console_logging


RUN_DIR = REPO.parent / "analysis" / "run" / "hle" / "sonnet_socratic_self_refine" / "run_20260427_071039"
RESULTS_PATH = RUN_DIR / "results.jsonl"
PROGRESS_PATH = RUN_DIR / "progress.json"
GRADING_DIR = RUN_DIR / "grading"
JUDGE_BACKEND = "claude"
NUM_WORKERS = 4


async def regrade_one(bench, row: dict, sem: asyncio.Semaphore) -> dict:
    qid = row["id"]
    out_dir = GRADING_DIR / qid
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = str(row["predicted_answer"]).strip()
    gold = str(row["gold_answer"]).strip()
    # Fast path: byte-identical predicted == gold means the LLM judge would
    # also say correct. Skip the API call (saves ~45/100 quota on this run).
    if pred == gold:
        grade = {
            "judge_model": "exact_match_skip",
            "is_correct": True,
            "predicted": row["predicted_answer"],
            "gold": row["gold_answer"],
            "judge_cost_usd": 0.0,
        }
        (out_dir / "grade.json").write_text(json.dumps(grade, indent=2))
        return {"qid": qid, "is_correct": True, "judge_cost_usd": 0.0, "skipped": True}
    async with sem:
        synth_row = {"id": qid, "question": row["question"], "answer": row["gold_answer"]}
        is_correct, judge_cost = await bench.grade(
            predicted=row["predicted_answer"],
            gold=row["gold_answer"],
            question=row["question"],
            row=synth_row,
            backend=JUDGE_BACKEND,
            out_dir=out_dir,
        )
        return {
            "qid": qid,
            "is_correct": bool(is_correct),
            "judge_cost_usd": float(judge_cost),
            "skipped": False,
        }


async def main() -> None:
    setup_console_logging()
    bench = get_benchmark("hle")
    logger.info(f"Benchmark: {bench.name}, judge_model={bench.judge_model}")
    assert bench.judge_model == "claude-haiku-4-5-20251001", \
        f"hle.py patch not applied: judge_model={bench.judge_model!r}"

    rows = [json.loads(line) for line in RESULTS_PATH.read_text().splitlines() if line.strip()]
    logger.info(f"Loaded {len(rows)} rows from {RESULTS_PATH}")

    # Pre-snapshot current is_correct for diff
    old_correct = sum(1 for r in rows if r["is_correct"])
    logger.info(f"Pre-regrade acc: {old_correct}/{len(rows)} = {old_correct/len(rows)*100:.1f}%")

    sem = asyncio.Semaphore(NUM_WORKERS)
    tasks = [regrade_one(bench, r, sem) for r in rows]
    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
        r = await coro
        results.append(r)
        marker = "OK " if r["is_correct"] else "wrn"
        logger.info(f"  [{i:>3}/{len(rows)}] {r['qid'][:24]} -> {marker} (judge_cost=${r['judge_cost_usd']:.4f})")

    # Update results.jsonl in-place
    by_qid = {r["qid"]: r for r in results}
    for row in rows:
        u = by_qid[row["id"]]
        row["is_correct"] = u["is_correct"]
        # accumulate previous + delta
        row["judge_cost_usd"] = u["judge_cost_usd"]
    with RESULTS_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    new_correct = sum(1 for r in rows if r["is_correct"])
    total_judge_cost = sum(r["judge_cost_usd"] for r in rows)
    logger.info(f"\n=== regrade complete ===")
    logger.info(f"Old acc: {old_correct}/{len(rows)} = {old_correct/len(rows)*100:.1f}%")
    logger.info(f"New acc: {new_correct}/{len(rows)} = {new_correct/len(rows)*100:.1f}%  (delta {new_correct - old_correct:+d})")
    logger.info(f"Judge cost: ${total_judge_cost:.4f}")

    # Patch progress.json
    if PROGRESS_PATH.exists():
        prog = json.loads(PROGRESS_PATH.read_text())
        prog["correct"] = new_correct
        prog["accuracy_pct"] = 100.0 * new_correct / len(rows)
        prog["summary"]["correct"] = new_correct
        prog["summary"]["judge_cost_usd"] = total_judge_cost
        PROGRESS_PATH.write_text(json.dumps(prog, indent=2))
        logger.info(f"Patched {PROGRESS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
