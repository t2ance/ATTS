"""Re-grade an entire explore cache with a new judge, writing fresh bundles.

Use case: cache/<benchmark>/<model>/<subset>/<qid>/explore_N/ has result.json
files but no judge bundle (legacy grade.json was archive-only'd, or this is a
fresh cache that has never been judged). This script reads judge_spec +
cache_dir from a YAML, walks every <qid>/explore_N/ that contains a
result.json, calls _grade_with_cache, and writes a bundle to
explore_N/judges/<label>/.

Differs from eval.py: skips orchestrator decision-making, integration, and
final/first-explore re-judging. Pure cached-explore re-grading. Cache pre-flight
is bypassed because we walk the cache directly instead of the dataset.

Idempotent: find_cached_judge inside _grade_with_cache skips already-judged
explores (config.json match), so re-runs are safe.

Usage:
    python scripts/regrade_cache.py --config scripts/hle/haiku/hle_haiku_regrade_qwen36judge.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval import EvalConfig, _grade_with_cache, load_config
from benchmarks import get_benchmark
from benchmarks.base import judge_label


# Default concurrency. Override via YAML's num_workers if you want.
DEFAULT_NUM_WORKERS = 64


async def regrade_one(
    bench, qid: str, row: dict, explore_dir: Path, judge_label_str: str,
    backend: str, sem: asyncio.Semaphore,
) -> dict:
    """Re-grade one (qid, explore_N). Cache-hit on bundle = no judge call."""
    async with sem:
        result = json.loads((explore_dir / "result.json").read_text(encoding="utf-8"))
        if result.get("timed_out"):
            return {"qid": qid, "explore": explore_dir.name, "status": "timed_out"}
        predicted = bench.get_answer_from_explore(result)
        gold = str(row.get("answer", ""))
        question = str(row.get("question", ""))
        # Probe bundle existence BEFORE the call: vllm judges return cost=0
        # whether they hit cache or actually judged, so we can't infer the
        # status from the return value alone.
        bundle_existed = (explore_dir / "judges" / judge_label_str / "grade.json").exists()
        is_correct, judge_cost = await _grade_with_cache(
            bench, predicted, gold, question, row,
            backend=backend, grade_dir=explore_dir, quiet=True,
        )
        return {
            "qid": qid,
            "explore": explore_dir.name,
            "is_correct": bool(is_correct),
            "judge_cost_usd": float(judge_cost),
            "status": "cached" if bundle_existed else "judged",
        }


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, required=True,
                   help="YAML with benchmark.judge + method.cache_dir.")
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                   help=f"asyncio Semaphore concurrency (default {DEFAULT_NUM_WORKERS}).")
    args = p.parse_args()

    cfg = load_config(config_path=args.config, schema=EvalConfig)
    judge_spec = cfg.benchmark.judge.model_dump()
    cache_dir = Path(cfg.method.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (REPO / cache_dir).resolve()
    backend = judge_spec["name"]

    bench = get_benchmark(cfg.benchmark.name, judge_spec=judge_spec)
    label = judge_label(judge_spec)
    print(f"=== regrade_cache ===")
    print(f"config:     {args.config}")
    print(f"judge_spec: {json.dumps(judge_spec, ensure_ascii=False)}")
    print(f"cache_dir:  {cache_dir}")
    print(f"backend:    {backend}")
    print(f"label:      {label}")
    print(f"workers:    {args.num_workers}")

    print(f"\nLoading {bench.name.upper()} dataset...")
    all_rows = bench.load_dataset()
    bench_filters = cfg.benchmark.model_dump(exclude={"name", "judge"}, exclude_defaults=True)
    filtered = bench.filter_dataset(all_rows, **bench_filters)
    row_by_id = {bench.get_id(r): r for r in filtered}
    print(f"Dataset rows after filter: {len(filtered)}")

    # Walk cache_dir, queue (qid, explore_N) jobs for every result.json present.
    tasks = []
    sem = asyncio.Semaphore(args.num_workers)
    qids_skipped_no_row = 0
    for qid_dir in sorted(cache_dir.iterdir()):
        if not qid_dir.is_dir():
            continue
        qid = qid_dir.name
        if qid not in row_by_id:
            qids_skipped_no_row += 1
            continue
        row = row_by_id[qid]
        for explore_dir in sorted(qid_dir.iterdir()):
            if not (explore_dir.is_dir() and explore_dir.name.startswith("explore_")):
                continue
            if not (explore_dir / "result.json").exists():
                continue
            tasks.append(regrade_one(bench, qid, row, explore_dir, label, backend, sem))

    print(f"qid in cache but not in dataset (skipped): {qids_skipped_no_row}")
    print(f"Total (qid, explore) re-grade jobs: {len(tasks)}\n")
    if not tasks:
        print("Nothing to do.")
        return

    correct = wrong = timed_out = 0
    judged = cached = 0
    total_cost = 0.0
    t0 = time.time()
    for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
        r = await coro
        status = r["status"]
        if status == "timed_out":
            timed_out += 1
        else:
            if r["is_correct"]:
                correct += 1
            else:
                wrong += 1
            if status == "judged":
                judged += 1
                total_cost += r["judge_cost_usd"]
            else:
                cached += 1
        if i % 100 == 0 or i == len(tasks):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            eta_s = (len(tasks) - i) / max(rate, 1e-6)
            print(
                f"  [{i:>4}/{len(tasks)}] correct={correct} wrong={wrong} "
                f"timed_out={timed_out} | judged={judged} cached={cached} "
                f"cost=${total_cost:.4f} | rate={rate:.1f}/s eta={eta_s/60:.1f}min"
            )

    print(f"\n=== regrade_cache complete ===")
    print(f"Correct:    {correct}")
    print(f"Wrong:      {wrong}")
    print(f"Timed-out:  {timed_out}")
    print(f"Judge calls (cache miss): {judged}")
    print(f"Cache hits (skipped):     {cached}")
    print(f"Total judge cost: ${total_cost:.4f}")
    print(f"Wall time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
