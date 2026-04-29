"""Pre-cache explore results for the delegated TTS agent.

Runs N independent explorer sub-models per question in parallel (worker pool),
saving structured results + trajectories to disk. These can then be replayed
by the orchestrator via --cache-dir.

Usage:
    python precache_explores.py --config configs/precache_hle.yaml \
        -o num_explores=4 -o num=10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

os.environ.pop("CLAUDECODE", None)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks import get_benchmark
from benchmarks.base import BenchmarkConfig
from benchmarks.specs import BenchmarkSpec
from methods.base import make_sub_model_caller


class PrecacheConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": False}

    benchmark: BenchmarkSpec
    backend: Literal["codex", "claude", "vllm"]
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
    print(f"Tasks: {total} to run, {skipped} already cached")
    if total == 0:
        print("Nothing to do.")
        return

    completed = 0
    sem = asyncio.Semaphore(num_workers)

    async def worker(qid: str, row: dict, explore_idx: int) -> None:
        nonlocal completed
        async with sem:
            print(f"  [{qid} explore_{explore_idx}] started")
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
            )

            import shutil
            result_dir = question_cache_dir / f"explore_{explore_idx}"

            if result.get("timed_out"):
                completed += 1
                print(f"  [{completed}/{total}] {qid} explore_{explore_idx}: TIMED OUT after {duration:.0f}s")
                return

            # Validate structured output; retry if malformed
            try:
                answer = benchmark.get_answer_from_explore(result)
            except KeyError as e:
                shutil.rmtree(result_dir, ignore_errors=True)
                print(f"  [{qid} explore_{explore_idx}] MALFORMED (missing {e}), retrying...")
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
                    print(f"  [{completed}/{total}] {qid} explore_{explore_idx}: TIMED OUT on retry")
                    return
                answer = benchmark.get_answer_from_explore(result)

            completed += 1
            answer_short = answer.replace("\n", " ")[:80]
            print(f"  [{completed}/{total}] {qid} explore_{explore_idx}: answer={answer_short}, confidence={result.get('confidence', 'N/A')}")

    await asyncio.gather(*(worker(qid, row, idx) for qid, row, idx in tasks))

    print(f"\nDone. {completed} cached, {skipped} skipped (already existed).")


def parse_cli() -> "PrecacheConfig":
    """Build PrecacheConfig from --config + -o overrides only."""
    from eval import load_config

    parser = argparse.ArgumentParser(description="Pre-cache explore results")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="Dot-path override, e.g. -o num_explores=4")
    args = parser.parse_args()

    return load_config(
        config_path=args.config,
        dot_overrides=list(args.override),
        schema=PrecacheConfig,
    )


def main() -> None:
    cfg = parse_cli()
    benchmark = get_benchmark(cfg.benchmark.name)
    bench_filters = cfg.benchmark.model_dump(exclude={"name"}, exclude_defaults=True)

    print(f"Loading {benchmark.name.upper()} dataset...")
    all_rows = benchmark.load_dataset()

    filtered = benchmark.filter_dataset(all_rows, **bench_filters)
    print(f"Filtered to {len(filtered)} questions")

    if cfg.shuffle:
        import random
        random.seed(cfg.seed)
        random.shuffle(filtered)

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

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
    ))


if __name__ == "__main__":
    main()
