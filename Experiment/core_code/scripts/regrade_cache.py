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
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval import EvalConfig, _grade_with_cache, load_config
from benchmarks import get_benchmark
from benchmarks.base import judge_label
from logger import setup_console_logging


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
            backend=backend, grade_dir=explore_dir,
        )
        return {
            "qid": qid,
            "explore": explore_dir.name,
            "is_correct": bool(is_correct),
            "judge_cost_usd": float(judge_cost),
            "status": "cached" if bundle_existed else "judged",
        }


async def main() -> None:
    setup_console_logging()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, required=True,
                   help="YAML with benchmark.judge + method.cache_dir.")
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                   help=f"asyncio Semaphore concurrency (default {DEFAULT_NUM_WORKERS}).")
    args = p.parse_args()

    cfg = load_config(config_path=args.config, schema=EvalConfig)
    judge_spec = cfg.benchmark.judge.model_dump()
    # TODO(modelconfig-refactor 2026-05-04): cfg.method.cache_dir no longer
    # exists on the unified TTSAgentSpec; the cache lives at
    # spec.explore[0].cache_dir (single-variant) or spec.explore[i].cache_dir
    # (multi-variant). For self-refine etc.: spec.explore.cache_dir. Walk the
    # variants instead of treating cache_dir as a single Path. Until this is
    # reworked, this script is broken on tts-agent yamls. Rerank yamls still
    # have a top-level cache_dir field and should still work.
    if hasattr(cfg.method, "cache_dir"):
        cache_dir = Path(cfg.method.cache_dir)
    else:
        raise NotImplementedError(
            "regrade_cache.py needs an update for the unified TTSAgentSpec; "
            "see TODO in source. As of 2026-05-04 it works only for rerank yamls."
        )
    if not cache_dir.is_absolute():
        cache_dir = (REPO / cache_dir).resolve()
    backend = judge_spec["backend"]

    bench = get_benchmark(cfg.benchmark.name, judge_spec=judge_spec)
    label = judge_label(judge_spec)
    logger.info(f"=== regrade_cache ===")
    logger.info(f"config:     {args.config}")
    logger.info(f"judge_spec: {json.dumps(judge_spec, ensure_ascii=False)}")
    logger.info(f"cache_dir:  {cache_dir}")
    logger.info(f"backend:    {backend}")
    logger.info(f"label:      {label}")
    logger.info(f"workers:    {args.num_workers}")

    logger.info(f"\nLoading {bench.name.upper()} dataset...")
    all_rows = bench.load_dataset()
    bench_filters = cfg.benchmark.model_dump(exclude={"name", "judge"}, exclude_defaults=True)
    filtered = bench.filter_dataset(all_rows, **bench_filters)
    row_by_id = {bench.get_id(r): r for r in filtered}
    logger.info(f"Dataset rows after filter: {len(filtered)}")

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

    logger.info(f"qid in cache but not in dataset (skipped): {qids_skipped_no_row}")
    logger.info(f"Total (qid, explore) re-grade jobs: {len(tasks)}\n")
    if not tasks:
        logger.info("Nothing to do.")
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
            logger.info(
                f"  [{i:>4}/{len(tasks)}] correct={correct} wrong={wrong} "
                f"timed_out={timed_out} | judged={judged} cached={cached} "
                f"cost=${total_cost:.4f} | rate={rate:.1f}/s eta={eta_s/60:.1f}min"
            )

    logger.info(f"\n=== regrade_cache complete ===")
    logger.info(f"Correct:    {correct}")
    logger.info(f"Wrong:      {wrong}")
    logger.info(f"Timed-out:  {timed_out}")
    logger.info(f"Judge calls (cache miss): {judged}")
    logger.info(f"Cache hits (skipped):     {cached}")
    logger.info(f"Total judge cost: ${total_cost:.4f}")
    logger.info(f"Wall time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
