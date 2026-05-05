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

    Sub-model dispatch is no longer wrapped here. Solvers either build their
    own ExploreVariant.get_exploration calls (explore role) or call backend
    modules directly (integrate / feedback). cache_dir / cache_only are kept
    on the context for solvers that still want to inspect them (e.g. for
    cache-only enforcement in generate_fn closures).
    """
    state: SolvingState
    cost: CostTracker
    rounds: list[RoundLog]
    writer: TrajectoryWriter
    traj_dir: Path | None
    question_cache_dir: Path | None
    image_data_url: str | None
    benchmark: Any  # BenchmarkConfig
    logger: RunLogger | None
    question_id: str | None
    cache_only: bool
    rollout_idx: int | None = None

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

# make_sub_model_caller deleted in 2026-05-05 explore-cache-owner refactor.
# Explore caching is owned by ExploreVariant.get_exploration; integrate calls
# go directly through backend.call_sub_model with no caching; auxiliary
# feedback calls (self_refine / socratic_self_refine) likewise call the
# backend directly. The cache_only / integrate_* exemption disappears with
# this deletion: cache_only is enforced at the generate_fn closure level
# inside ExploreVariant.get_exploration callers.


# ---------------------------------------------------------------------------
# Solve context factory
# ---------------------------------------------------------------------------

def create_solve_context(
    *,
    infra: InfraConfig,
    backend: str,
    timeout: float | None,
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

    Sub-model callers no longer cached on the context: solvers either build
    ExploreVariant.get_exploration calls (explore role) or call backend modules
    directly (integrate / feedback). `backend` and `timeout` are kept in the
    signature for API stability with existing callers but are not consumed
    by this factory anymore.

    When rollout_idx is None (default, K=1 path), trajectory is written to
    trajectories/<qid>/ (flat). When rollout_idx is set (K>1), trajectory is
    written to trajectories/<qid>/rollout_<k>/ (nested).
    """
    del backend, timeout  # historical args; sub-model dispatch moved to callers
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
