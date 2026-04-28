"""Full restoration of run_20260427_071039 to a state indistinguishable from
a clean run where hle.py had judge_model='claude-haiku-4-5-20251001' from
launch.

Walks every grade.json under grading/, grading/<qid>/explore_*/,
grading/<qid>/rollout_*/, grading/<qid>/rollout_*/explore_*/. For each whose
judge_model is anything other than 'claude-haiku-4-5-20251001':

  - Calls grader.judge_answer with Haiku (no fast-path skip; always API call).
  - Passes out_dir so a per-call judge trajectory is saved (parity with eval.py).
  - Overwrites grade.json with the canonical schema:
        {judge_model, is_correct, predicted, gold, judge_cost_usd}.

After grade.json restoration, rebuilds:
  - results.jsonl: only is_correct + judge_cost_usd fields (other fields preserved).
  - progress.json: correct + accuracy_pct + summary.judge_cost_usd.

Idempotent: re-running picks up only files still tagged with the wrong judge_model.

Cost estimate: ~748 calls x $0.002 ~= $1.5 on the first full run.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from benchmarks import get_benchmark
from benchmarks.grader import judge_answer

RUN_DIR = REPO.parent / "analysis" / "run" / "hle" / "sonnet_socratic_self_refine" / "run_20260427_071039"
GRADING_DIR = RUN_DIR / "grading"
RESULTS_PATH = RUN_DIR / "results.jsonl"
PROGRESS_PATH = RUN_DIR / "progress.json"
JUDGE_BACKEND = "claude"
JUDGE_MODEL = "claude-haiku-4-5-20251001"
NUM_WORKERS = 4


async def fix_one(qid_to_question: dict[str, str], path: Path, sem: asyncio.Semaphore) -> dict:
    rel = path.relative_to(RUN_DIR)
    qid = rel.parts[1]  # grading/<qid>/...
    question = qid_to_question.get(qid)
    assert question is not None, f"qid {qid!r} not found in HLE dataset (path={rel})"

    d = json.loads(path.read_text())
    predicted = d.get("predicted", "")
    gold = d.get("gold", "")

    judge_out = path.parent / "judge"
    judge_out.mkdir(parents=True, exist_ok=True)
    async with sem:
        is_correct, cost = await judge_answer(
            predicted, gold, question, JUDGE_MODEL,
            backend=JUDGE_BACKEND, out_dir=judge_out,
        )
    new = {
        "judge_model": JUDGE_MODEL,
        "is_correct": bool(is_correct),
        "predicted": predicted,
        "gold": gold,
        "judge_cost_usd": float(cost),
    }
    path.write_text(json.dumps(new, indent=2, ensure_ascii=False))
    return {"path": str(rel), "is_correct": bool(is_correct), "cost": float(cost)}


def rebuild_results_and_progress() -> None:
    rows = [json.loads(l) for l in RESULTS_PATH.read_text().splitlines() if l.strip()]
    total_judge_cost = 0.0
    correct = 0
    for row in rows:
        qid = row["id"]
        gp = GRADING_DIR / qid / "grade.json"
        d = json.loads(gp.read_text())
        assert d["judge_model"] == JUDGE_MODEL, f"final grade.json for {qid} still wrong: {d['judge_model']!r}"
        row["is_correct"] = d["is_correct"]
        row["judge_cost_usd"] = d["judge_cost_usd"]
        total_judge_cost += d["judge_cost_usd"]
        if d["is_correct"]:
            correct += 1
    with RESULTS_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if PROGRESS_PATH.exists():
        prog = json.loads(PROGRESS_PATH.read_text())
        prog["correct"] = correct
        prog["accuracy_pct"] = 100.0 * correct / len(rows)
        prog.setdefault("summary", {})
        prog["summary"]["correct"] = correct
        prog["summary"]["judge_cost_usd"] = total_judge_cost
        PROGRESS_PATH.write_text(json.dumps(prog, indent=2))

    print(f"\n=== results.jsonl + progress.json rebuilt ===")
    print(f"correct: {correct}/{len(rows)} = {correct/len(rows)*100:.1f}%")
    print(f"total judge_cost_usd: ${total_judge_cost:.4f}")


async def main() -> None:
    bench = get_benchmark("hle")
    assert bench.judge_model == JUDGE_MODEL, f"hle.py judge_model={bench.judge_model!r}, expected {JUDGE_MODEL!r}"

    rows = bench.load_dataset()
    qid_to_question = {bench.get_id(r): bench.get_question(r) for r in rows}
    print(f"Loaded {len(qid_to_question)} HLE rows")

    stale: list[Path] = []
    for f in GRADING_DIR.rglob("grade.json"):
        d = json.loads(f.read_text())
        if d.get("judge_model") != JUDGE_MODEL:
            stale.append(f)
    print(f"Found {len(stale)} grade.json with judge_model != {JUDGE_MODEL!r}")
    if not stale:
        print("(nothing to do at the file level)")
    else:
        sem = asyncio.Semaphore(NUM_WORKERS)
        tasks = [fix_one(qid_to_question, p, sem) for p in stale]
        total_cost = 0.0
        correct = 0
        done = 0
        for coro in asyncio.as_completed(tasks):
            r = await coro
            done += 1
            total_cost += r["cost"]
            correct += int(r["is_correct"])
            if done % 25 == 0 or done == len(stale):
                pct = 100.0 * correct / done
                print(f"  [{done:>4}/{len(stale)}] correct_so_far={correct} ({pct:.1f}%) cost=${total_cost:.4f}", flush=True)
        print(f"\nfile-level fix complete: {len(stale)} re-judged, total cost ${total_cost:.4f}")

    rebuild_results_and_progress()


if __name__ == "__main__":
    asyncio.run(main())
