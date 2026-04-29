"""Pydantic schema for precache_explores.py configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class PrecacheConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": False}

    benchmark: str
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

    filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_filters(self):
        from benchmarks import get_benchmark
        bench = get_benchmark(self.benchmark)
        filter_model = bench.make_filter_model()
        validated = filter_model.model_validate(self.filters)
        self.filters = validated.model_dump(exclude_defaults=True)
        return self
