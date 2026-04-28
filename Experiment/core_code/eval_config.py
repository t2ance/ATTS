"""Pydantic schema for eval.py configuration. Loader added in Task 2."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
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


def _coerce_scalar(s: str) -> Any:
    """Best-effort coerce a CLI-string scalar to int/float/bool/str."""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _set_dotpath(d: dict, path: str, value: str) -> None:
    """Set d[a][b][c] = value for path 'a.b.c'. Creates nested dicts as needed."""
    parts = path.split(".")
    assert parts and all(parts), f"empty segment in path {path!r}"
    target = d
    for p in parts[:-1]:
        nxt = target.get(p)
        if nxt is None:
            nxt = {}
            target[p] = nxt
        else:
            assert isinstance(nxt, dict), (
                f"cannot descend into non-dict at {p!r} while resolving override {path!r} "
                f"(found {type(nxt).__name__}: {nxt!r})"
            )
        target = nxt
    target[parts[-1]] = _coerce_scalar(value)


def load_config(
    *,
    config_path: Path | str | None,
    flat_overrides: dict[str, Any],
    dot_overrides: list[str],
) -> EvalConfig:
    """Merge config sources and validate via pydantic.

    Order of precedence (later wins):
      1. YAML file (if config_path is given)
      2. Flat overrides (kwargs from argparse, key matches field name)
      3. Dot-path overrides (e.g. "model_budgets.haiku=2")
    """
    merged: dict[str, Any] = {}
    if config_path is not None:
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}
        assert isinstance(yaml_data, dict), (
            f"top level of {config_path} must be a mapping, got {type(yaml_data).__name__}"
        )
        merged.update(yaml_data)

    for key, val in flat_overrides.items():
        merged[key] = val

    for ov in dot_overrides:
        k, sep, v = ov.partition("=")
        assert sep == "=", f"override must be key=value, got {ov!r}"
        _set_dotpath(merged, k.strip(), v.strip())

    return EvalConfig.model_validate(merged)
