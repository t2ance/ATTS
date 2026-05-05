"""TTS (Test-Time Scaling) agent: unified orchestrator-driven explore/integrate loop.

Post-modelconfig-refactor (2026-05-04): one solver covers single-variant ATTS,
multi-model (haiku/sonnet/opus), and multi-effort (low/medium/high) modes.
Operating mode is encoded by `len(spec.explore)` and `spec.orchestrator_prompt`.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

logger = logging.getLogger(__name__)

import asyncio
import time

from cache_types import Exploration
from methods.base import (
    Candidate,
    InfraConfig,
    SolveContext,
    create_solve_context,
)
from methods.tool_io import CandidateRecord, FullRenderer
from methods.tool_state import advance
from trajectory import RoundLog, SolveResult, TrajectoryWriter
from prompts import build_user_message, select_orchestrator_prompt


# ---------------------------------------------------------------------------
# Orchestrator tool definitions (backend-agnostic)
# ---------------------------------------------------------------------------

def _build_explore_tool(variants) -> dict[str, Any]:
    """Build the explore tool schema.

    Length-1 (single-variant): no parameter, byte-identical with the pre-
    refactor EXPLORE_TOOL constant.
    Length-N (multi-variant): exposes `variant: enum[<labels>]` so the
    orchestrator picks which variant to dispatch each call. Mirrors the
    old EXPLORE_TOOL_MULTI / EXPLORE_TOOL_EFFORT shape.
    """
    if len(variants) == 1:
        return {
            "name": "explore",
            "description": (
                "Dispatch a fresh, independent solver to generate a new candidate answer. "
                "Takes no parameters -- a separate model will solve the problem from scratch."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    labels = [v.label for v in variants]
    return {
        "name": "explore",
        "description": (
            "Dispatch a fresh, independent solver to generate a new candidate answer. "
            "You must specify which variant to use. Each variant has its own budget; "
            "do not call a variant whose budget is exhausted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "variant": {
                    "type": "string",
                    "enum": labels,
                    "description": "Which variant to dispatch for this explore call.",
                },
            },
            "required": ["variant"],
            "additionalProperties": False,
        },
    }


INTEGRATE_TOOL: dict[str, Any] = {
    "name": "integrate",
    "description": (
        "Dispatch a synthesizer to produce the final answer from all candidates. "
        "Takes no parameters -- a separate model will analyze all candidates."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Shared tool handler logic
# ---------------------------------------------------------------------------

def process_explore_result(
    ctx: SolveContext,
    exp: Exploration,
    *,
    label: str,
    model_label: str = "",
    extra_budget_text: str = "",
) -> str:
    """Process an Exploration: update state, return tool result text.

    The returned text is produced by `methods.tool_io.FullRenderer` -- the
    single source of truth for candidate-text rendering, also consumed by
    the GRPO rollout tool and the SFT data builder, so train and eval
    observe byte-identical strings.
    """
    state = ctx.state
    state.explore = advance(state.explore, label=label) if label else advance(state.explore)
    used = state.explore.used
    label_suffix = f" ({model_label})" if model_label else ""
    extra = exp.extra or {}
    explore_cost = exp.cost_usd
    explore_usage = extra.get("usage", {})

    if exp.timed_out:
        logger.info(f"  [sub-model] explore #{used}{label_suffix}: TIMED OUT -- recording empty candidate")
        state.candidates.append(
            Candidate(answer="", reasoning="timed out", approach="", confidence=0.0, cost_usd=0.0)
        )
        ctx.writer.write_explore_timeout()
        return FullRenderer().render(CandidateRecord(
            idx=used,
            answer="",
            confidence=0.0,
            approach="",
            reasoning="",
            cost_usd=0.0,
            used=used,
            max_explores=state.explore.max_explores,
            model_label=model_label,
            extra_budget_text=extra_budget_text,
            timed_out=True,
        ))

    ctx.cost.add(explore_cost, explore_usage, component="explorer")
    answer = exp.answer

    state.candidates.append(
        Candidate(
            answer=answer,
            reasoning=extra.get("reasoning", ""),
            approach=extra.get("approach", ""),
            confidence=extra.get("confidence", 0.0),
            cost_usd=explore_cost,
        )
    )

    text = FullRenderer().render(CandidateRecord(
        idx=used,
        answer=answer,
        confidence=float(extra.get("confidence", 0.0)),
        approach=extra.get("approach", ""),
        reasoning=extra.get("reasoning", ""),
        cost_usd=explore_cost,
        used=used,
        max_explores=state.explore.max_explores,
        model_label=model_label,
        extra_budget_text=extra_budget_text,
    ))
    logger.info(f"  [sub-model] explore candidate #{len(state.candidates)}{label_suffix}: answer={answer}, confidence={extra.get('confidence', 'N/A')}")
    return text


def make_structured_output_handler(ctx: SolveContext):
    """Create the on_structured_output callback. Shared by single and multi-variant."""
    def on_structured_output(result: dict) -> None:
        ctx.state.final_answer = ctx.benchmark.get_answer_from_explore(result)
        ctx.state.final_reasoning = result.get("reasoning", "")
        logger.info(f"[structured_output] answer={ctx.state.final_answer}")
        _log_round(ctx, RoundLog(
            round_num=ctx.state.explore.call_count + 1,
            action="submit_answer",
            tool_input=result,
        ))
    return on_structured_output


# ---------------------------------------------------------------------------
# Per-variant explore + cached integrate
# ---------------------------------------------------------------------------

def _budget_status_text(spec, ctx: SolveContext) -> str:
    """Format per-variant budget status for orchestrator (length>1 only)."""
    if len(spec.explore) == 1:
        return ""
    parts = []
    for v in spec.explore:
        used = ctx.state.explore.variant_call_counts.get(v.label, 0)
        cap = v.num_explores
        remaining = cap - used
        status = "EXHAUSTED" if remaining <= 0 else f"{remaining} remaining"
        parts.append(f"{v.label}: {used}/{cap} used ({status})")
    return "\nPer-variant budget: " + ", ".join(parts) + "."


async def run_explore(ctx: SolveContext, spec, variants_by_label: dict, label: str) -> str:
    """Run an explore call against the variant identified by `label`.

    Cache + persistence is owned by ExploreVariant.get_exploration. The
    generate_fn closure here only handles the API call + Exploration
    construction; the variant decides cache hit vs miss and writes the
    bundle into <cache_dir>/<qid>/[rollout_<r>/]explore_<idx>/.
    """
    multi = len(spec.explore) > 1
    if ctx.state.explore.is_exhausted:
        return (
            f"Explore quota exhausted ({ctx.state.explore.max_explores} explores already used). "
            f"You must call submit_answer with the best candidate from prior explores now."
        )
    if multi and ctx.state.explore.variant_exhausted(label):
        cap = ctx.state.explore.variant_caps.get(label, 0)
        return (
            f"Variant {label!r} budget exhausted ({cap} used). "
            f"Call explore with a different variant or submit_answer."
        )
    variant = variants_by_label[label]
    in_idx = ctx.state.explore.variant_call_counts.get(label, 0) + 1

    user_msg = ctx.benchmark.build_explorer_message(ctx.state.problem)
    explorer_system_prompt = ctx.benchmark.get_explorer_system_prompt(variant.model.backend)
    explore_schema = ctx.benchmark.get_explore_schema()
    cache_only = ctx.cache_only
    timeout = variant.model.timeout

    async def generate_fn() -> Exploration:
        if cache_only:
            raise AssertionError(
                f"cache_only mode: explore cache miss at "
                f"{variant._explore_dir(ctx.question_id, in_idx, ctx.rollout_idx)}"
            )
        backend_mod = import_module(f"backends.{variant.model.backend}")
        t0 = time.time()
        api_coro = backend_mod.call_sub_model(
            explorer_system_prompt, user_msg, ctx.image_data_url,
            variant.model.model, explore_schema,
            TrajectoryWriter.noop(),
            budget_tokens=variant.model.budget_tokens,
            effort=variant.model.effort,
            sampling=(variant.model.vllm_sampling.model_dump()
                      if variant.model.vllm_sampling is not None else None),
            provider_order=variant.model.openrouter_provider_order,
            provider_allow_fallbacks=variant.model.openrouter_provider_allow_fallbacks,
        )
        if timeout is not None:
            try:
                result, traj, cost, usage = await asyncio.wait_for(api_coro, timeout=timeout)
            except asyncio.TimeoutError:
                duration = time.time() - t0
                logger.info(
                    f"  [sub-model] explore_{in_idx}: WALL-CLOCK TIMEOUT after {duration:.0f}s "
                    f"(limit: {timeout}s)"
                )
                return Exploration(
                    qid=ctx.question_id or "", idx=in_idx, rollout_idx=ctx.rollout_idx,
                    answer="", trajectory="", cost_usd=0.0, model=variant.model.model,
                    timed_out=True,
                    extra={"timeout_seconds": timeout, "duration_seconds": duration},
                    system_prompt=explorer_system_prompt,
                    user_message=user_msg,
                )
        else:
            result, traj, cost, usage = await api_coro

        answer = ctx.benchmark.get_answer_from_explore(result)
        return Exploration(
            qid=ctx.question_id or "", idx=in_idx, rollout_idx=ctx.rollout_idx,
            answer=answer,
            trajectory=traj,
            cost_usd=cost,
            model=variant.model.model,
            timed_out=bool(result.get("timed_out", False)),
            extra={"usage": usage,
                   **{k: v for k, v in result.items() if k not in {"answer"}}},
            system_prompt=explorer_system_prompt,
            user_message=user_msg,
        )

    exp = await variant.get_exploration(
        ctx.question_id or "", in_idx,
        rollout_idx=ctx.rollout_idx,
        generate_fn=generate_fn,
    )

    # Mirror to traj_dir for this run's working record.
    if ctx.traj_dir is not None:
        exp.persist(ctx.traj_dir / f"explore_{in_idx}")

    return process_explore_result(
        ctx, exp,
        label=label,
        model_label=label if multi else "",
        extra_budget_text=_budget_status_text(spec, ctx),
    )


async def run_integrate(ctx: SolveContext, spec) -> str:
    """Run the integrate call against spec.integrate (RoleSlot).

    No cache: integrate input is candidate content (already cached at the
    explore layer); caching by (qid, count) was content-blind and unsafe.
    Trajectory still written to run_dir/trajectories via ctx.writer.
    """
    assert spec.integrate is not None, "integrate called when spec.integrate is None"
    state = ctx.state
    assert state.candidates, "integrate called with no candidates"

    integrator_system_prompt = ctx.benchmark.get_integrator_system_prompt(spec.integrate.model.backend)
    integrate_schema = ctx.benchmark.get_integrate_schema()
    user_msg = ctx.benchmark.build_integrator_message(state.problem, state.candidates)

    backend_mod = import_module(f"backends.{spec.integrate.model.backend}")
    timeout = spec.integrate.model.timeout
    api_coro = backend_mod.call_sub_model(
        integrator_system_prompt, user_msg, ctx.image_data_url,
        spec.integrate.model.model, integrate_schema,
        ctx.writer,
        budget_tokens=spec.integrate.model.budget_tokens,
        effort=spec.integrate.model.effort,
        sampling=(spec.integrate.model.vllm_sampling.model_dump()
                  if spec.integrate.model.vllm_sampling is not None else None),
        provider_order=spec.integrate.model.openrouter_provider_order,
        provider_allow_fallbacks=spec.integrate.model.openrouter_provider_allow_fallbacks,
    )
    if timeout is not None:
        try:
            result, _traj, cost_usd, usage = await asyncio.wait_for(api_coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.info(f"  [integrate] WALL-CLOCK TIMEOUT after {timeout}s")
            ctx.cost.add(0.0, {}, component="integrator")
            state.final_answer = ""
            return "Integrate timed out."
    else:
        result, _traj, cost_usd, usage = await api_coro
    ctx.cost.add(cost_usd, usage, component="integrator")

    final_answer = ctx.benchmark.get_answer_from_integrate(result)
    state.final_answer = final_answer
    state.final_reasoning = result.get("reasoning")
    state.final_analysis = result.get("analysis")

    logger.info(f"  [sub-model] integrate: final_answer={final_answer}")
    return "Final answer recorded."


# ---------------------------------------------------------------------------
# Orchestrator loop (algorithm logic, backend-agnostic)
# ---------------------------------------------------------------------------

def _log_round(ctx: SolveContext, round_log: RoundLog) -> None:
    """Append a round log and persist via logger if available."""
    ctx.rounds.append(round_log)
    if ctx.logger and ctx.question_id:
        ctx.logger.log_round(
            question_id=ctx.question_id,
            round_num=round_log.round_num,
            action=round_log.action,
            tool_input=round_log.tool_input,
            rollout_idx=ctx.rollout_idx,
        )


async def _run_orchestrator(
    ctx: SolveContext,
    spec,
    variants_by_label: dict,
    user_message_text: str,
    system_prompt: str,
    temperature: float | None = None,
) -> None:
    """Run the orchestrator loop via the backend's run_tool_conversation."""
    backend_mod = import_module(f"backends.{spec.orchestrator.backend}")
    multi = len(spec.explore) > 1

    async def tool_handler(name: str, args: dict) -> tuple[str, bool]:
        if name == "explore":
            if not multi:
                # No `variant` parameter exposed to orchestrator in this case.
                label = spec.explore[0].label
            else:
                # Length>1: `variant` is required by the tool schema. Fail
                # loud rather than silently defaulting to spec.explore[0] --
                # a missing field means the orchestrator/backend dropped it
                # and we want the operator to see the gap, not silently
                # bias every multi-variant run toward the first variant.
                assert "variant" in args, (
                    f"explore tool called without `variant` param under "
                    f"length>1 spec; args={args!r}"
                )
                label = args["variant"]
            n_before = len(ctx.state.candidates)
            result_text = await run_explore(ctx, spec, variants_by_label, label)
            if len(ctx.state.candidates) > n_before:
                cand = ctx.state.candidates[-1]
                _log_round(ctx, RoundLog(
                    round_num=ctx.state.explore.used,
                    action="explore",
                    tool_input={
                        "variant": label,
                        "answer": cand.answer,
                        "reasoning": cand.reasoning,
                        "approach": cand.approach,
                        "confidence": cand.confidence,
                        "cost_usd": cand.cost_usd,
                    },
                ))
            return result_text, False
        elif name == "integrate":
            result_text = await run_integrate(ctx, spec)
            _log_round(ctx, RoundLog(
                round_num=ctx.state.explore.call_count + 1,
                action="integrate",
                tool_input={
                    "final_answer": ctx.state.final_answer,
                    "reasoning": ctx.state.final_reasoning,
                    "analysis": ctx.state.final_analysis,
                },
            ))
            return result_text, True
        else:
            assert False, f"Unknown tool: {name}"

    explore_tool = _build_explore_tool(spec.explore)
    if spec.integrate is not None:
        tools = [explore_tool, INTEGRATE_TOOL]
        output_format = None
    else:
        tools = [explore_tool]
        output_format = {"type": "json_schema", "schema": ctx.benchmark.get_explore_schema()}

    cost, usage, exit_reason = await backend_mod.run_tool_conversation(
        system_prompt=system_prompt,
        user_message=user_message_text,
        image_data_url=ctx.image_data_url,
        model=spec.orchestrator.model,
        tools=tools,
        max_turns=ctx.state.explore.max_explores + 2,
        tool_handler=tool_handler,
        effort=spec.orchestrator.effort,
        output_format=output_format,
        writer=ctx.writer,
        on_structured_output=make_structured_output_handler(ctx),
        max_output_tokens=spec.orchestrator.max_output_tokens,
        temperature=temperature,
        sampling=(spec.orchestrator.vllm_sampling.model_dump()
                  if spec.orchestrator.vllm_sampling is not None else None),
        provider_order=spec.orchestrator.openrouter_provider_order,
        provider_allow_fallbacks=spec.orchestrator.openrouter_provider_allow_fallbacks,
    )
    ctx._exit_reason = exit_reason
    ctx.cost.add(cost, usage, component="orchestrator")

    ctx.writer.write_session_summary(ctx.cost.total_cost_usd, {
        "input_tokens": ctx.cost.total_input_tokens,
        "output_tokens": ctx.cost.total_output_tokens,
    })


# ---------------------------------------------------------------------------
# Main solve entry point
# ---------------------------------------------------------------------------

async def solve(
    infra: InfraConfig,
    problem: str,
    *,
    spec,  # methods.specs.TTSAgentSpec
    image_data_url: str | None = None,
    question_id: str | None = None,
    rollout_idx: int | None = None,
    temperature: float | None = None,
    **_extra,
) -> SolveResult:
    """Solve a problem using the unified TTS agent.

    spec.explore length 1 == single-variant ATTS; length > 1 == multi-model
    or effort runs depending on label set + orchestrator_prompt.

    temperature: orchestrator sampling temperature pin for the rejection-
        sampling K>1 path (eval.py expands rows with _temperature). When
        spec.orchestrator.vllm_sampling already carries a temperature, this
        kwarg is ignored by the backend.
    rollout_idx: K>1 trajectories live under trajectories/<qid>/rollout_<k>/.
    """
    from methods.specs import TTSAgentSpec  # circular-import guard
    assert isinstance(spec, TTSAgentSpec), type(spec)

    max_iterations = sum(v.num_explores for v in spec.explore)
    assert infra.max_iterations == max_iterations, (
        f"infra.max_iterations={infra.max_iterations} does not match "
        f"sum(num_explores)={max_iterations}"
    )

    user_message_text = build_user_message(
        problem,
        max_iterations,
        variant_budgets=({v.label: v.num_explores for v in spec.explore}
                         if len(spec.explore) > 1 else None),
    )
    system_prompt = select_orchestrator_prompt(spec)

    ctx = create_solve_context(
        infra=infra,
        backend=spec.orchestrator.backend,
        timeout=spec.orchestrator.timeout,
        problem=problem,
        image_data_url=image_data_url,
        question_id=question_id,
        writer_system_prompt=system_prompt,
        writer_user_message=user_message_text,
        writer_header_lines=[
            f"**Orchestrator**: {spec.orchestrator.backend}/{spec.orchestrator.model}",
            *(f"**Variant {v.label}**: {v.model.backend}/{v.model.model} (n={v.num_explores})"
              for v in spec.explore),
            *([f"**Integrate**: {spec.integrate.model.backend}/{spec.integrate.model.model}"]
              if spec.integrate else []),
            f"**Max iterations**: {max_iterations}",
        ],
        writer_title_suffix="(unified)",
        rollout_idx=rollout_idx,
    )
    # Populate per-variant caps for budget guard (length-1 leaves the dict
    # empty; variant_exhausted is never consulted on that path).
    if len(spec.explore) > 1:
        ctx.state.explore = type(ctx.state.explore)(
            max_explores=ctx.state.explore.max_explores,
            call_count=ctx.state.explore.call_count,
            variant_call_counts=ctx.state.explore.variant_call_counts,
            variant_caps={v.label: v.num_explores for v in spec.explore},
        )

    # Per-variant ExploreVariant lookup, keyed by label. Each ExploreVariant
    # owns its cache_dir + persistence + judge cache; the orchestrator passes
    # generate_fn closures into get_exploration.
    variants_by_label: dict[str, Any] = {v.label: v for v in spec.explore}

    logger.info(
        f"\nTTS Agent [{spec.orchestrator.backend}] -- "
        f"{len(spec.explore)} variant(s), up to {max_iterations} explores"
    )
    if image_data_url:
        logger.info("Image: included")
    logger.info("")

    await _run_orchestrator(
        ctx, spec, variants_by_label, user_message_text, system_prompt,
        temperature=temperature,
    )

    if ctx.state.final_answer is None:
        ctx.state.final_answer = ""

    logger.info(
        f"\nTotal cost: ${ctx.cost.total_cost_usd}"
        f" (input: {ctx.cost.total_input_tokens}, output: {ctx.cost.total_output_tokens})"
    )
    result = ctx.result(ctx.state.final_answer)
    result.exit_reason = getattr(ctx, "_exit_reason", "incomplete")
    return result
