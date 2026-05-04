"""Pre-cache explore results for the delegated TTS agent.

Runs N independent explorer sub-models per question in parallel (worker pool),
saving structured results + trajectories to disk. These can then be replayed
by the orchestrator via --cache-dir.

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
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

os.environ.pop("CLAUDECODE", None)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks import get_benchmark
from benchmarks.base import BenchmarkConfig
from benchmarks.specs import BenchmarkSpec
from eval import SamplingConfig
from methods.base import make_sub_model_caller
from logger import now_str, setup_console_logging


class PrecacheConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": False}

    benchmark: BenchmarkSpec
    # "openrouter" added 2026-05-03: see methods/specs.py BackendConfig comment.
    backend: Literal["codex", "claude", "vllm", "openrouter"]
    explore_model: str
    cache_dir: Path

    num_explores: int = 8
    num_workers: int = 1
    num: int | None = None
    skip: int = 0
    seed: int = 42
    shuffle: bool = False
    budget_tokens: int = 32000
    effort: Literal["low", "medium", "high", "max"] = "low"
    explore_timeout: float = 1200.0
    # OpenRouter-only: pin upstream provider routing. See
    # backends/openrouter.py:set_provider for rationale (2026-05-04
    # deepseek-v4-flash incident: ~33% of routes 400'd on forced tool_choice).
    # When set, precache_explores.py calls backends.openrouter.set_provider(...)
    # at startup; both call_sub_model and run_tool_conversation inject the
    # block into extra_body.provider on every request. Silently ignored by
    # non-openrouter backends.
    provider_order: list[str] | None = None
    provider_allow_fallbacks: bool = True
    # vLLM-only sampling block. MUST match the orchestrator's sampling block in
    # the eval yaml that consumes this cache -- explorer cache distribution and
    # on-the-fly orchestrator-explore distribution must be identical, otherwise
    # cache_only assertion in make_sub_model_caller would mask a behavior gap.
    # See backends/vllm.py:call_sub_model for the per-key default fallback.
    sampling: SamplingConfig | None = None


async def precache(
    benchmark: BenchmarkConfig,
    rows: list[dict],
    cache_dir: Path,
    num_explores: int,
    num_workers: int,
    backend: str,
    model: str,
    num: int | None = None,
    budget_tokens: int = 32000,
    effort: str | None = None,
    explore_timeout: float = 600,
    sampling: dict | None = None,
) -> None:
    """Pre-cache explore results for the given rows."""
    explorer_prompt = benchmark.get_explorer_system_prompt(backend)
    explore_schema = benchmark.get_explore_schema()
    if num is not None:
        rows = rows[:num]

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
                traj_dir=question_cache_dir, timeout=explore_timeout,
            )

            image_data_url = benchmark.get_image(row)
            input_text = benchmark.build_explorer_message(benchmark.get_question(row))

            result, traj, cost_usd, usage, duration = await sub_model_fn(
                system_prompt=explorer_prompt,
                user_message=input_text,
                image_data_url=image_data_url,
                model=model,
                output_schema=explore_schema,
                cache_key=f"explore_{explore_idx}",
                budget_tokens=budget_tokens,
                effort=effort,
                sampling=sampling,
            )

            import shutil
            result_dir = question_cache_dir / f"explore_{explore_idx}"

            if result.get("timed_out"):
                completed += 1
                logger.info(f"  [{completed}/{total}] {qid} explore_{explore_idx}: TIMED OUT after {duration:.0f}s")
                return

            # Validate structured output; retry if malformed
            try:
                answer = benchmark.get_answer_from_explore(result)
            except KeyError as e:
                shutil.rmtree(result_dir, ignore_errors=True)
                logger.warning(f"  [{qid} explore_{explore_idx}] MALFORMED (missing {e}), retrying...")
                result, traj, cost_usd, usage, duration = await sub_model_fn(
                    system_prompt=explorer_prompt,
                    user_message=input_text,
                    image_data_url=image_data_url,
                    model=model,
                    output_schema=explore_schema,
                    cache_key=f"explore_{explore_idx}",
                    budget_tokens=budget_tokens,
                    effort=effort,
                )
                if result.get("timed_out"):
                    completed += 1
                    logger.info(f"  [{completed}/{total}] {qid} explore_{explore_idx}: TIMED OUT on retry")
                    return
                answer = benchmark.get_answer_from_explore(result)

            completed += 1
            answer_short = answer.replace("\n", " ")[:80]
            logger.info(f"  [{completed}/{total}] {qid} explore_{explore_idx}: answer={answer_short}, confidence={result.get('confidence', 'N/A')}")

    await asyncio.gather(*(worker(qid, row, idx) for qid, row, idx in tasks))

    logger.info(f"\nDone. {completed} cached, {skipped} skipped (already existed).")


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
    # Same pattern as eval.py: pull `judge` out of the benchmark spec dump.
    # Precache itself does not grade, but we pass judge_spec through so the
    # constructed benchmark stays a faithful representation of the YAML.
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

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    # If the yaml pinned an OpenRouter provider, configure the backend module
    # before any explore call fires. No-op for other backends (the import
    # below is guarded — only attempted when backend=='openrouter').
    if cfg.backend == "openrouter" and cfg.provider_order is not None:
        from backends import openrouter as _openrouter
        _openrouter.set_provider(cfg.provider_order, cfg.provider_allow_fallbacks)

    asyncio.run(precache(
        benchmark=benchmark,
        rows=filtered,
        cache_dir=cfg.cache_dir,
        num_explores=cfg.num_explores,
        num_workers=cfg.num_workers,
        backend=cfg.backend,
        model=cfg.explore_model,
        num=cfg.num,
        budget_tokens=cfg.budget_tokens,
        effort=cfg.effort,
        explore_timeout=cfg.explore_timeout,
        sampling=cfg.sampling.model_dump() if cfg.sampling else None,
    ))


if __name__ == "__main__":
    main()
