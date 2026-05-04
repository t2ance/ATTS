"""Method behavior registry: per-method runtime configuration + dispatch.

Pairs with methods/specs.py (data layer). Each method name maps to one
MethodConfig subclass that declares its runtime properties (cache_only,
pre_flight_check, etc.) and provides build_solve_fn / filter_rows /
preflight hooks that eval.py calls polymorphically.
"""
from __future__ import annotations

import functools
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

from methods.specs import (
    MethodSpec, TTSAgentSpec,
    SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
    RerankSpec, StandaloneIntegratorSpec,
)


class MethodConfig(ABC):
    name: str
    cache_only: bool                              # explore cache miss -> AssertionError?
    pre_flight_check: bool = False                # banner-time cache completeness?
    pre_filter_by_cache: bool = False             # drop rows with no cache before run?
    supports_num_rollouts: bool = False           # rejection-sampling K>1?
    consumes_sampling_block: bool = False         # YAML sampling block transparently passed?

    @abstractmethod
    def build_solve_fn(self, spec: MethodSpec) -> Callable:
        """Return the partialed solve function for this method, bound to spec fields."""

    def derive_evaluate_args(self, spec: MethodSpec) -> dict:
        """Return kwargs for evaluate(): orchestrator_model / explore_model /
        integrate_model / cache_dirs_multi. Each method picks what makes sense."""
        return {
            "orchestrator_model": getattr(spec, "orchestrator_model", "") or "",
            "explore_model": getattr(spec, "explore_model", "") or "",
            "integrate_model": getattr(spec, "integrate_model", "") or "",
            "cache_dirs_multi": getattr(spec, "cache_dirs", None),
        }

    def filter_rows(self, rows: list[dict], cache_dir: Path | None, benchmark) -> list[dict]:
        """Default: pass-through. Override in pre_filter_by_cache methods."""
        if not self.pre_filter_by_cache or cache_dir is None:
            return rows
        cached_ids = {
            p.name for p in cache_dir.iterdir()
            if p.is_dir() and (p / "explore_1" / "result.json").exists()
        }
        before = len(rows)
        out = [r for r in rows if benchmark.get_id(r) in cached_ids]
        logger.info(f"{self.name}: {len(out)} questions with cache (from {before})")
        return out

    def preflight(
        self,
        rows: list[dict],
        cache_dir: Path | None,
        num_explores: int,
        num: int | None,
        benchmark,
    ) -> None:
        """Default: no-op. Override in pre_flight_check methods."""
        if not self.pre_flight_check or cache_dir is None:
            return
        rows_to_check = rows if num is None else rows[:num]
        qids = [benchmark.get_id(r) for r in rows_to_check]
        missing: list[tuple[str, int]] = []
        for qid in qids:
            for idx in range(1, num_explores + 1):
                if not (cache_dir / qid / f"explore_{idx}" / "result.json").exists():
                    missing.append((qid, idx))
        if missing:
            sample = ", ".join(f"({q}, explore_{i})" for q, i in missing[:10])
            raise AssertionError(
                f"Cache pre-flight FAILED: {len(missing)} missing entries "
                f"(of {len(qids) * num_explores} expected) under {cache_dir}. "
                f"First 10: {sample}"
            )
        logger.info(
            f"Cache pre-flight OK: {len(qids)} qids x {num_explores} explores "
            f"= {len(qids) * num_explores} cache files present at {cache_dir}"
        )


class TTSAgentMethod(MethodConfig):
    name = "tts-agent"
    cache_only = True
    pre_flight_check = True
    supports_num_rollouts = True
    consumes_sampling_block = True

    def build_solve_fn(self, spec: TTSAgentSpec):
        from methods.tts_agent import solve
        return solve


class SelfRefineMethod(MethodConfig):
    name = "self-refine"
    cache_only = False                  # generates new refines on top of cached drafts
    pre_flight_check = True

    def build_solve_fn(self, spec: SelfRefineSpec):
        from methods.self_refine import solve
        return solve


class SocraticSelfRefineMethod(MethodConfig):
    name = "socratic-self-refine"
    cache_only = False
    pre_flight_check = True

    def build_solve_fn(self, spec: SocraticSelfRefineSpec):
        from methods.socratic_self_refine import solve
        return solve


class BudgetForcingMethod(MethodConfig):
    name = "budget-forcing"
    cache_only = False
    pre_flight_check = True

    def build_solve_fn(self, spec: BudgetForcingSpec):
        from methods.budget_forcing import solve
        return solve


class RerankMethod(MethodConfig):
    name = "rerank"
    cache_only = True
    pre_filter_by_cache = True

    def build_solve_fn(self, spec: RerankSpec):
        from methods.reward_rerank import solve
        return functools.partial(solve, reward_model_name=spec.reward_model)


class StandaloneIntegratorMethod(MethodConfig):
    name = "standalone-integrator"
    cache_only = True
    pre_filter_by_cache = True

    def build_solve_fn(self, spec: StandaloneIntegratorSpec):
        from methods.standalone_integrator import solve
        return functools.partial(
            solve,
            integrate_model=spec.integrate_model,
            num_explores=spec.num_explores,
        )


METHODS: dict[str, type[MethodConfig]] = {
    "tts-agent": TTSAgentMethod,
    "self-refine": SelfRefineMethod,
    "socratic-self-refine": SocraticSelfRefineMethod,
    "budget-forcing": BudgetForcingMethod,
    "rerank": RerankMethod,
    "standalone-integrator": StandaloneIntegratorMethod,
}


def get_method(name: str) -> MethodConfig:
    assert name in METHODS, f"Unknown method: {name!r}. Available: {list(METHODS)}"
    return METHODS[name]()
