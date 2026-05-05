"""Pre-cache explore results for the delegated TTS agent.

Runs N independent explorer sub-models per question in parallel (worker pool),
saving structured results + trajectories to disk. These can then be replayed
by the orchestrator via the ExploreVariant.cache_dir contract.

Post-modelconfig-refactor (2026-05-04): the precache yaml mirrors the
new method-spec shape — `explore: ExploreVariant` carries the model config
(backend / model / budget / effort / timeout / vllm_sampling /
openrouter_provider_*) plus the cache_dir and num_explores. No more flat
backend / explore_model / sampling / provider_order at the top level.

Usage:
    python precache_explores.py --config scripts/hle/sonnet/hle_sonnet_precache.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

os.environ.pop("CLAUDECODE", None)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks import get_benchmark
from benchmarks.base import BenchmarkConfig
from benchmarks.specs import BenchmarkSpec
from methods.base import make_sub_model_caller
from methods.specs import ExploreVariant
from logger import setup_console_logging, PrecacheLogger


class PrecacheConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": False}

    benchmark: BenchmarkSpec
    # The explore-call invocation parameters + cache target + how many
    # candidates to draw per question. Mirrors method.explore in
    # SelfRefine/Socratic/BudgetForcing yamls and method.explore[i] in
    # tts-agent yamls. The same ExploreVariant produced here is what the
    # downstream eval will read from cache.
    explore: ExploreVariant

    # Top-level concerns that are NOT part of the explore call itself
    # (parallelism + dataset slicing). Kept flat.
    num_workers: int = 1
    num: int | None = None
    skip: int = 0
    seed: int = 42
    shuffle: bool = False


async def precache(
    benchmark: BenchmarkConfig,
    rows: list[dict],
    variant: ExploreVariant,
    num_workers: int,
    num: int | None = None,
) -> None:
    """Pre-cache explore results for the given rows using one ExploreVariant."""
    backend = variant.model.backend
    cache_dir = variant.cache_dir
    num_explores = variant.num_explores
    sampling_dump = (
        variant.model.vllm_sampling.model_dump()
        if variant.model.vllm_sampling is not None
        else None
    )

    explorer_prompt = benchmark.get_explorer_system_prompt(backend)
    explore_schema = benchmark.get_explore_schema()
    if num is not None:
        rows = rows[:num]

    # Build qid list (matches the worker enqueue loop below). Used by the
    # progress logger to scope its disk scan; stale qids no longer in this
    # filter never inflate the cumulative counts.
    qids = [benchmark.get_id(row) for row in rows]

    # Init progress.json. Reconstructs cumulative state from any result.json
    # files already on disk in cache_dir; safe on a fresh dir too.
    progress_logger = PrecacheLogger(
        cache_dir=cache_dir,
        qids=qids,
        num_explores=num_explores,
    )
    logger.info(
        f"Progress: {cache_dir / 'progress.json'} "
        f"({progress_logger._initial_record_count}/{len(qids) * num_explores} already on disk)"
    )

    tasks: list[tuple[str, dict, int]] = []
    skipped = 0
    for row in rows:
        qid = benchmark.get_id(row)
        for i in range(1, num_explores + 1):
            if (cache_dir / qid / f"explore_{i}" / "result.json").exists():
                skipped += 1
            else:
                tasks.append((qid, row, i))

    total = len(tasks)
    logger.info(f"Tasks: {total} to run, {skipped} already cached")
    if total == 0:
        progress_logger.finalize()
        logger.info("Nothing to do.")
        return

    completed = 0
    sem = asyncio.Semaphore(num_workers)

    async def worker(qid: str, row: dict, explore_idx: int) -> None:
        nonlocal completed
        async with sem:
            logger.info(f"  [{qid} explore_{explore_idx}] started")
            question_cache_dir = cache_dir / qid
            sub_model_fn = make_sub_model_caller(
                backend, cache_dir=question_cache_dir, cache_only=False,
                traj_dir=question_cache_dir, timeout=variant.model.timeout,
            )

            image_data_url = benchmark.get_image(row)
            input_text = benchmark.build_explorer_message(benchmark.get_question(row))

            result, traj, cost_usd, usage, duration = await sub_model_fn(
                system_prompt=explorer_prompt,
                user_message=input_text,
                image_data_url=image_data_url,
                model=variant.model.model,
                output_schema=explore_schema,
                cache_key=f"explore_{explore_idx}",
                budget_tokens=variant.model.budget_tokens,
                effort=variant.model.effort,
                sampling=sampling_dump,
                provider_order=variant.model.openrouter_provider_order,
                provider_allow_fallbacks=variant.model.openrouter_provider_allow_fallbacks,
            )

            import shutil
            result_dir = question_cache_dir / f"explore_{explore_idx}"

            if result.get("timed_out"):
                progress_logger.record_task(
                    qid=qid, explore_idx=explore_idx,
                    result=result, usage=usage,
                    duration_seconds=duration, cost_usd=cost_usd,
                )
                completed += 1
                logger.info(f"  [{completed}/{total}] {qid} explore_{explore_idx}: TIMED OUT after {duration:.0f}s")
                return

            try:
                answer = benchmark.get_answer_from_explore(result)
            except KeyError as e:
                shutil.rmtree(result_dir, ignore_errors=True)
                logger.warning(f"  [{qid} explore_{explore_idx}] MALFORMED (missing {e}), retrying...")
                result, traj, cost_usd, usage, duration = await sub_model_fn(
                    system_prompt=explorer_prompt,
                    user_message=input_text,
                    image_data_url=image_data_url,
                    model=variant.model.model,
                    output_schema=explore_schema,
                    cache_key=f"explore_{explore_idx}",
                    budget_tokens=variant.model.budget_tokens,
                    effort=variant.model.effort,
                    sampling=sampling_dump,
                    provider_order=variant.model.openrouter_provider_order,
                    provider_allow_fallbacks=variant.model.openrouter_provider_allow_fallbacks,
                )
                if result.get("timed_out"):
                    progress_logger.record_task(
                        qid=qid, explore_idx=explore_idx,
                        result=result, usage=usage,
                        duration_seconds=duration, cost_usd=cost_usd,
                    )
                    completed += 1
                    logger.info(f"  [{completed}/{total}] {qid} explore_{explore_idx}: TIMED OUT on retry")
                    return
                answer = benchmark.get_answer_from_explore(result)

            progress_logger.record_task(
                qid=qid, explore_idx=explore_idx,
                result=result, usage=usage,
                duration_seconds=duration, cost_usd=cost_usd,
            )
            completed += 1
            answer_short = answer.replace("\n", " ")[:80]
            logger.info(f"  [{completed}/{total}] {qid} explore_{explore_idx}: answer={answer_short}, confidence={result.get('confidence', 'N/A')}")

    await asyncio.gather(*(worker(qid, row, idx) for qid, row, idx in tasks))

    progress_logger.finalize()
    logger.info(
        f"Done. {completed} cached, {skipped} skipped (already existed). "
        f"Progress: {cache_dir / 'progress.json'}"
    )


def main() -> None:
    setup_console_logging()
    from eval import load_config

    parser = argparse.ArgumentParser(description="Pre-cache explore results")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(
        config_path=args.config,
        schema=PrecacheConfig,
    )
    bench_dump = cfg.benchmark.model_dump()
    judge_spec = bench_dump.pop("judge", None)
    benchmark = get_benchmark(cfg.benchmark.name, judge_spec=judge_spec)
    bench_filters = cfg.benchmark.model_dump(exclude={"name", "judge"}, exclude_defaults=True)

    logger.info(f"Loading {benchmark.name.upper()} dataset...")
    all_rows = benchmark.load_dataset()

    filtered = benchmark.filter_dataset(all_rows, **bench_filters)
    logger.info(f"Filtered to {len(filtered)} questions")

    if cfg.shuffle:
        import random
        random.seed(cfg.seed)
        random.shuffle(filtered)

    cfg.explore.cache_dir.mkdir(parents=True, exist_ok=True)

    # No openrouter.set_provider call here — provider routing rides on
    # cfg.explore.model.openrouter_provider_order, threaded per-call into
    # backends.openrouter.call_sub_model via make_sub_model_caller.

    asyncio.run(precache(
        benchmark=benchmark,
        rows=filtered,
        variant=cfg.explore,
        num_workers=cfg.num_workers,
        num=cfg.num,
    ))


if __name__ == "__main__":
    main()
