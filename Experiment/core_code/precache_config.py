"""Pydantic schema for precache_explores.py configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from benchmarks.specs import BenchmarkSpec


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
