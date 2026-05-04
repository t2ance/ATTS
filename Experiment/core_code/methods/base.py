"""Shared infrastructure for all solving methods.

Data structures, sub-model dispatch, caching, and solve context factory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from methods.tool_state import ExploreStepState
from trajectory import CostTracker, RoundLog, SolveResult, TrajectoryWriter
from logger import RunLogger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InfraConfig:
    """Per-run infrastructure shared by all solving methods.

    Per-role backend/model/effort/budget/timeout previously shared at this
    level moved into per-role ModelConfig (see methods/specs.py). InfraConfig
    now carries only cross-role context: the cache_dir for cache pre-flight
    and explore-cache reads (used by solvers like rerank / standalone_integrator
    that walk a single cache directory), max_iterations, the benchmark, the
    logger, and the integrate-enabled flag.
    """
    max_iterations: int
    cache_dir: Path | None
    cache_only: bool
    benchmark: Any
    logger: RunLogger | None
    enable_integrate: bool = True


@dataclass
class Candidate:
    """A single candidate answer."""
    answer: str
    reasoning: str
    approach: str
    confidence: float
    cost_usd: float = 0.0


@dataclass
class SolvingState:
    """State of the solving process."""
    problem: str
    explore: ExploreStepState
    candidates: list[Candidate] = field(default_factory=list)
    final_answer: str | None = None
    final_reasoning: str | None = None
    final_analysis: str | None = None


@dataclass
class SolveContext:
    """Common state for all solve methods, created by create_solve_context().

    Per-role invocation parameters are NO LONGER cached here. Solvers pass
    a `model_cfg: ModelConfig` to `call_sub_model`; the underlying
    _sub_model_fn was bound at create_solve_context time to one specific
    backend (the role that owns the default caller). For cross-backend roles
    (e.g. multi-variant explore), solvers build their own per-variant
    `make_sub_model_caller` instances inline.
    """
    state: SolvingState
    cost: CostTracker
    rounds: list[RoundLog]
    _sub_model_fn: Any
    writer: TrajectoryWriter
    traj_dir: Path | None
    question_cache_dir: Path | None
    image_data_url: str | None
    benchmark: Any  # BenchmarkConfig
    logger: RunLogger | None
    question_id: str | None
    cache_only: bool
    rollout_idx: int | None = None

    async def call_sub_model(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model_cfg,  # methods.specs.ModelConfig (typed at call site)
        output_schema: dict[str, Any],
        cache_key: str = "",
        writer: TrajectoryWriter | None = None,
    ) -> tuple[dict[str, Any], str, float, dict[str, Any], float]:
        """Dispatch a single backend call using a ModelConfig.

        All backend selection / budget / effort / timeout / sampling /
        provider routing comes from model_cfg. The infra-level _sub_model_fn
        was constructed at create_solve_context() time and is bound to a
        single backend module — so model_cfg.backend MUST match what
        _sub_model_fn was built against. Solvers that need to dispatch
        across backends (cross-backend explore variants) keep one
        _sub_model_fn per backend; see methods/tts_agent.py per-variant
        callers.
        """
        return await self._sub_model_fn(
            system_prompt, user_message, self.image_data_url,
            model_cfg.model, output_schema,
            cache_key=cache_key,
            writer=writer or TrajectoryWriter.noop(),
            budget_tokens=model_cfg.budget_tokens,
            effort=model_cfg.effort,
            sampling=(model_cfg.vllm_sampling.model_dump()
                      if model_cfg.vllm_sampling is not None else None),
            provider_order=model_cfg.openrouter_provider_order,
            provider_allow_fallbacks=model_cfg.openrouter_provider_allow_fallbacks,
        )

    def result(self, answer: str) -> SolveResult:
        return SolveResult(answer=answer, cost=self.cost, rounds=self.rounds, writer=self.writer)


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cached_candidates(
    cache_dir: Path,
    question_id: str,
    benchmark: Any,
) -> tuple[list[Candidate], float]:
    """Load all non-timed-out cached explore results for a question.

    Returns (candidates, total_explore_cost_usd).
    """
    question_cache = cache_dir / question_id
    candidates: list[Candidate] = []
    total_cost = 0.0
    idx = 1
    while True:
        result_path = question_cache / f"explore_{idx}" / "result.json"
        if not result_path.exists():
            break
        d = json.loads(result_path.read_text(encoding="utf-8"))
        if not d.get("timed_out"):
            answer = benchmark.get_answer_from_explore(d)
            cost = d.get("cost_usd", 0.0)
            candidates.append(Candidate(
                answer=answer,
                reasoning=d.get("reasoning", ""),
                approach=d.get("approach", ""),
                confidence=d.get("confidence", 0.0),
                cost_usd=cost,
            ))
            total_cost += cost
        idx += 1
    return candidates, total_cost


# ---------------------------------------------------------------------------
# Sub-model save / load
# ---------------------------------------------------------------------------

def save_sub_model_input(
    out_dir: Path,
    input_text: str,
    system_prompt: str,
    image_data_url: str | None = None,
) -> None:
    """Save sub-model input to out_dir/{input.md, input.png} before query."""
    import base64
    import re
    out_dir.mkdir(parents=True, exist_ok=True)
    full_input = f"## System Prompt\n\n{system_prompt}\n\n## User Message\n\n{input_text}"
    (out_dir / "input.md").write_text(full_input, encoding="utf-8")
    if image_data_url:
        m = re.match(r"data:image/(\w+);base64,(.+)", image_data_url, re.DOTALL)
        assert m, f"Unexpected image_data_url format: {image_data_url[:80]}"
        ext = m.group(1).replace("jpeg", "jpg")
        (out_dir / f"input.{ext}").write_bytes(base64.b64decode(m.group(2)))


def save_sub_model_result(
    out_dir: Path,
    result: dict[str, Any],
    trajectory_text: str,
    cost_usd: float,
    usage: dict[str, Any],
    duration_seconds: float,
    model: str,
) -> None:
    """Save sub-model result to out_dir/result.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **result,
        "trajectory": trajectory_text,
        "cost_usd": cost_usd,
        "usage": usage,
        "duration_seconds": duration_seconds,
        "model": model,
    }
    # errors="surrogatepass" tolerates lone UTF-16 surrogate codepoints
    # (e.g. \udcba) that some backends emit through token decode quirks.
    # Without this, write_text raises `UnicodeEncodeError: 'utf-8' codec
    # can't encode character '\udcba' ... surrogates not allowed` and aborts
    # mid-precache. Observed 2026-05-03 on gpt-oss-20b HLE precache after 88
    # successful writes; the lone surrogate came from a tool_call argument
    # string in the trajectory. WTF-8 serialization (PEP 383) round-trips
    # the byte exactly so any later reader sees the same string.
    (out_dir / "result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
        errors="surrogatepass",
    )


# ---------------------------------------------------------------------------
# Sub-model call (dispatches to backend module)
# ---------------------------------------------------------------------------

async def call_sub_model(
    backend: str,
    system_prompt: str,
    user_message: str,
    image_data_url: str | None,
    model: str,
    output_schema: dict[str, Any],
    writer: TrajectoryWriter = TrajectoryWriter.noop(),
    budget_tokens: int = 32000,
    effort: str | None = None,
    sampling: dict | None = None,
    provider_order: list[str] | None = None,
    provider_allow_fallbacks: bool = True,
) -> tuple[dict[str, Any], str, float, dict[str, Any]]:
    """Spawn a fresh sub-model call (no caching). Used by grader etc.

    Returns (structured_output, trajectory_text, cost_usd, usage).

    provider_order / provider_allow_fallbacks are openrouter-only routing
    knobs; non-openrouter backends accept and ignore them.
    """
    backend_mod = import_module(f"backends.{backend}")
    return await backend_mod.call_sub_model(
        system_prompt, user_message, image_data_url, model, output_schema,
        writer, budget_tokens=budget_tokens, effort=effort, sampling=sampling,
        provider_order=provider_order,
        provider_allow_fallbacks=provider_allow_fallbacks,
    )


# ---------------------------------------------------------------------------
# Sub-model caller factory (with optional transparent caching)
# ---------------------------------------------------------------------------

def make_sub_model_caller(
    backend: str,
    cache_dir: Path | None = None,
    cache_only: bool = True,
    traj_dir: Path | None = None,
    timeout: float | None = None,
):
    """Return a sub-model callable with integrated caching and trajectory I/O.

    Used by create_solve_context() and precache_explores.py.
    """
    backend_mod = import_module(f"backends.{backend}")

    def _out_dirs(cache_key: str) -> list[Path]:
        """Return non-None output directories for a given cache_key."""
        dirs = []
        if traj_dir and cache_key:
            dirs.append(traj_dir / cache_key)
        if cache_dir and cache_key:
            dirs.append(cache_dir / cache_key)
        return dirs

    async def call(
        system_prompt: str,
        user_message: str,
        image_data_url: str | None,
        model: str,
        output_schema: dict[str, Any],
        *,
        cache_key: str = "",
        writer: TrajectoryWriter = TrajectoryWriter.noop(),
        budget_tokens: int = 32000,
        effort: str | None = None,
        sampling: dict | None = None,
        provider_order: list[str] | None = None,
        provider_allow_fallbacks: bool = True,
    ) -> tuple[dict[str, Any], str, float, dict[str, Any], float]:
        # Cache read
        if cache_dir and cache_key:
            result_path = cache_dir / cache_key / "result.json"
            if result_path.exists():
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                traj = cached.pop("trajectory", "")
                cost = cached.pop("cost_usd", 0.0)
                usage = cached.pop("usage", {})
                dur = cached.pop("duration_seconds", 0.0)
                model_name = cached.pop("model", None)
                for d in _out_dirs(cache_key):
                    save_sub_model_input(d, user_message, system_prompt, image_data_url)
                    save_sub_model_result(d, cached, traj, cost, usage, dur, model_name or "")
                    (d / "output.md").write_text(traj, encoding="utf-8")
                return cached, traj, cost, usage, dur
            if cache_only and not cache_key.startswith("integrate_"):
                raise AssertionError(f"cache_only mode: cache miss at {result_path}")

        # Save input before call
        for d in _out_dirs(cache_key):
            save_sub_model_input(d, user_message, system_prompt, image_data_url)

        # API call with optional wall-clock timeout
        t0 = time.time()
        api_coro = backend_mod.call_sub_model(
            system_prompt, user_message, image_data_url, model, output_schema,
            writer, budget_tokens=budget_tokens, effort=effort, sampling=sampling,
            provider_order=provider_order,
            provider_allow_fallbacks=provider_allow_fallbacks,
        )
        if timeout is not None:
            try:
                result, traj, cost, usage = await asyncio.wait_for(api_coro, timeout=timeout)
            except asyncio.TimeoutError:
                duration = time.time() - t0
                payload = json.dumps({"timed_out": True, "timeout_seconds": timeout, "duration_seconds": duration}, indent=2)
                for d in _out_dirs(cache_key):
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "result.json").write_text(payload, encoding="utf-8")
                logger.info(f"  [sub-model] {cache_key}: WALL-CLOCK TIMEOUT after {duration:.0f}s (limit: {timeout}s)")
                return {"timed_out": True}, "", 0.0, {}, duration
        else:
            result, traj, cost, usage = await api_coro
        duration = time.time() - t0

        # Save result + output
        for d in _out_dirs(cache_key):
            save_sub_model_result(d, result, traj, cost, usage, duration, model)
            (d / "output.md").write_text(traj, encoding="utf-8")

        return result, traj, cost, usage, duration

    return call


# ---------------------------------------------------------------------------
# Solve context factory
# ---------------------------------------------------------------------------

def create_solve_context(
    *,
    infra: InfraConfig,
    backend: str,
    timeout: float,
    problem: str,
    image_data_url: str | None = None,
    question_id: str | None = None,
    writer_system_prompt: str,
    writer_user_message: str,
    writer_header_lines: list[str],
    writer_title_suffix: str,
    rollout_idx: int | None = None,
) -> SolveContext:
    """Create common solve infrastructure shared by all methods.

    `backend` and `timeout` are passed in by the solver because they belong
    on the role's ModelConfig now, not on infra. The default _sub_model_fn
    is built against `backend`; solvers needing cross-backend dispatch (e.g.
    multi-variant explore in tts_agent.py) build additional _sub_model_fn
    instances per variant inline.

    When rollout_idx is None (default, K=1 path), trajectory is written to
    trajectories/<qid>/ (flat, old behavior). When rollout_idx is set (K>1
    path), trajectory is written to trajectories/<qid>/rollout_<k>/ (nested).
    """
    state = SolvingState(
        problem=problem,
        explore=ExploreStepState(max_explores=infra.max_iterations),
    )
    cost = CostTracker()
    rounds: list[RoundLog] = []

    traj_dir = None
    if infra.logger and question_id:
        if rollout_idx is None:
            traj_dir = infra.logger.run_dir / "trajectories" / question_id
        else:
            traj_dir = infra.logger.run_dir / "trajectories" / question_id / f"rollout_{rollout_idx}"
        traj_dir.mkdir(parents=True, exist_ok=True)

    question_cache_dir = None
    if infra.cache_dir is not None:
        assert question_id is not None, "question_id required when using cache_dir"
        question_cache_dir = infra.cache_dir / question_id
    sub_model_fn = make_sub_model_caller(
        backend, question_cache_dir, infra.cache_only,
        traj_dir=traj_dir, timeout=timeout,
    )

    if traj_dir is not None:
        writer = TrajectoryWriter.create(
            traj_dir=traj_dir,
            question_id=question_id,
            system_prompt=writer_system_prompt,
            user_message=writer_user_message,
            header_lines=writer_header_lines,
            title_suffix=writer_title_suffix,
            image_data_url=image_data_url,
        )
    else:
        writer = TrajectoryWriter.noop()

    return SolveContext(
        state=state,
        cost=cost,
        rounds=rounds,
        _sub_model_fn=sub_model_fn,
        writer=writer,
        traj_dir=traj_dir,
        question_cache_dir=question_cache_dir,
        image_data_url=image_data_url,
        benchmark=infra.benchmark,
        logger=infra.logger,
        question_id=question_id,
        cache_only=infra.cache_only,
        rollout_idx=rollout_idx,
    )
