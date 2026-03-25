"""Benchmark registry."""

from __future__ import annotations

import sys
from pathlib import Path

from benchmarks.base import BenchmarkConfig

# Ensure the parent dir (EXPLaIN/) is on sys.path for multimodal_input etc.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_REGISTRY: dict[str, type[BenchmarkConfig]] = {}


def register(name: str, cls: type[BenchmarkConfig]) -> None:
    _REGISTRY[name] = cls


def get_benchmark(name: str) -> BenchmarkConfig:
    if not _REGISTRY:
        _register_builtins()
    assert name in _REGISTRY, f"Unknown benchmark: {name!r}. Available: {list(_REGISTRY)}"
    return _REGISTRY[name]()


def _register_builtins() -> None:
    from benchmarks.hle import HLEBenchmark
    register("hle", HLEBenchmark)

    from benchmarks.lcb import LCBBenchmark
    register("lcb", LCBBenchmark)

    from benchmarks.babyvision import BabyVisionBenchmark
    register("babyvision", BabyVisionBenchmark)

    from benchmarks.aime import AIMEBenchmark, AIME2025Benchmark, AIME2026Benchmark
    register("aime", AIMEBenchmark)
    register("aime2025", AIME2025Benchmark)
    register("aime2026", AIME2026Benchmark)

    from benchmarks.rbenchv import RBenchVBenchmark
    register("rbenchv", RBenchVBenchmark)

    from benchmarks.gpqa import GPQABenchmark
    register("gpqa", GPQABenchmark)

# Note: imports above resolve to benchmarks/hle.py, benchmarks/lcb.py, etc.
# (single-file modules, not packages)
