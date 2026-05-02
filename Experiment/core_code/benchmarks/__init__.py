"""Benchmark registry."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the parent dir (EXPLaIN/) is on sys.path for multimodal_input etc.
# Must run before importing the benchmark modules below, since some of them
# transitively reach for top-level project utilities.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.base import BenchmarkConfig
from benchmarks.hle import HLEBenchmark
from benchmarks.lcb import LCBBenchmark
from benchmarks.babyvision import BabyVisionBenchmark
from benchmarks.aime import AIMEBenchmark, AIME2025Benchmark, AIME2026Benchmark
from benchmarks.rbenchv import RBenchVBenchmark
from benchmarks.gpqa import GPQABenchmark

BENCHMARKS: dict[str, type[BenchmarkConfig]] = {
    "hle": HLEBenchmark,
    "lcb": LCBBenchmark,
    "babyvision": BabyVisionBenchmark,
    "aime": AIMEBenchmark,
    "aime2025": AIME2025Benchmark,
    "aime2026": AIME2026Benchmark,
    "rbenchv": RBenchVBenchmark,
    "gpqa": GPQABenchmark,
}


def get_benchmark(
    name: str,
    judge_spec: dict | None = None,
    judge_max_retries: int = 3,
) -> BenchmarkConfig:
    assert name in BENCHMARKS, f"Unknown benchmark: {name!r}. Available: {list(BENCHMARKS)}"
    return BENCHMARKS[name](judge_spec=judge_spec, judge_max_retries=judge_max_retries)
