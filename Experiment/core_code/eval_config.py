"""Pydantic schema and loader for eval.py configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


METHODS = Literal[
    "tts-agent",
    "tts-agent-multi",
    "tts-agent-effort",
    "self-refine",
    "socratic-self-refine",
    "budget-forcing",
    "rerank",
    "standalone-integrator",
]


class EvalConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": False}

    # Required identity
    benchmark: str
    backend: Literal["codex", "claude", "vllm"]
    explore_model: str

    # Method selection
    method: METHODS = "tts-agent"
    orchestrator_model: str | None = None
    integrate_model: str | None = None
    reward_model: str | None = None

    # Cache: separated by method shape
    cache_dir: Path | None = None
    cache_dirs: dict[str, Path] = Field(default_factory=dict)

    # Multi/effort budgets
    model_budgets: dict[str, int] = Field(default_factory=dict)
    effort_budgets: dict[str, int] = Field(default_factory=dict)
    exploration_effort: Literal["low", "medium", "high"] | None = None

    # Dataset
    num: int | None = None
    skip: int = 0
    seed: int = 42
    shuffle: bool = False
    filters: dict[str, Any] = Field(default_factory=dict)

    # Model knobs
    budget_tokens: int = 32000
    effort: Literal["low", "medium", "high", "max"] = "low"
    num_explores: int = 8
    num_workers: int = 1
    explore_timeout: float = 1200.0
    max_output_chars: int | None = None

    # Run
    verbose: bool = False
    resume: str | None = None
    log_dir: str = "logs"
    no_cache_only: bool = False
    timeout: float = 1200.0
    no_integrate: bool = False
    num_rollouts: int = 1

    @model_validator(mode="after")
    def _validate_method_constraints(self):
        m = self.method
        if m == "tts-agent-multi":
            assert self.orchestrator_model, "tts-agent-multi requires orchestrator_model"
            assert self.cache_dirs, "tts-agent-multi requires cache_dirs (dict[model_alias, path])"
            assert self.model_budgets, "tts-agent-multi requires model_budgets"
        elif m == "tts-agent-effort":
            assert self.orchestrator_model, "tts-agent-effort requires orchestrator_model"
            assert self.cache_dirs, "tts-agent-effort requires cache_dirs (dict[level, path])"
            assert self.effort_budgets, "tts-agent-effort requires effort_budgets"
        elif m == "tts-agent":
            assert self.orchestrator_model, "tts-agent requires orchestrator_model"
            assert self.integrate_model, "tts-agent requires integrate_model"
        elif m == "rerank":
            assert self.reward_model, "rerank requires reward_model"
        elif m == "standalone-integrator":
            assert self.integrate_model, "standalone-integrator requires integrate_model"

        is_multi = m in ("tts-agent-multi", "tts-agent-effort")
        if is_multi:
            assert not self.cache_dir, (
                f"{m} uses cache_dirs (dict), not cache_dir (single)"
            )
        else:
            assert not self.cache_dirs, (
                f"{m} uses cache_dir (single Path), not cache_dirs (dict). "
                f"got cache_dirs={dict(self.cache_dirs)}"
            )

        if self.num_rollouts > 1:
            assert self.method == "tts-agent", (
                f"num_rollouts > 1 only supported for method=tts-agent, got {self.method}"
            )
            assert self.backend == "vllm", (
                f"num_rollouts > 1 only supported for backend=vllm, got {self.backend}"
            )
        assert self.num_rollouts >= 1, f"num_rollouts must be >= 1, got {self.num_rollouts}"

        return self
