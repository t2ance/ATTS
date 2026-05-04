"""Standalone Integrator baseline (LLM Selection).

Reads pre-generated candidates from a cache directory, feeds ALL candidates
to an LLM integrator in a single call, and returns its synthesized answer.
No orchestrator, no adaptive stopping, no multi-turn context -- isolates
the value of LLM-based aggregation from sequential adaptive exploration.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from methods.base import InfraConfig, create_solve_context, load_cached_candidates
from trajectory import CostTracker, RoundLog, SolveResult


async def solve(
    infra: InfraConfig,
    problem: str,
    *,
    spec,  # methods.specs.StandaloneIntegratorSpec
    image_data_url: str | None = None,
    question_id: str | None = None,
    rollout_idx: int | None = None,
    **_extra,
) -> SolveResult:
    """Synthesize from up to `num_explores` pre-cached candidates in one LLM call."""
    integrate_role = spec.integrate  # RoleSlot
    num_explores = spec.num_explores
    backend = integrate_role.model.backend
    assert infra.cache_dir is not None, "cache_dir is required for standalone-integrator"
    assert question_id is not None, "question_id is required for standalone-integrator"
    assert num_explores >= 1, f"num_explores must be >= 1, got {num_explores}"

    ctx = create_solve_context(
        infra=infra,
        backend=backend,
        # Single-shot integrator: keep old "no wall-clock timeout" behavior
        # (paper main runs use this path; long-tail synthesis on hard rows
        # can exceed the per-explore timeout). Pass None so make_sub_model_caller
        # disables the asyncio.wait_for guard.
        timeout=None,
        problem=problem,
        image_data_url=image_data_url,
        question_id=question_id,
        writer_system_prompt="(standalone integrator -- single-call synthesis)",
        writer_user_message=problem,
        writer_header_lines=[
            f"**Integrate Model**: {backend}/{integrate_role.model.model}",
            f"**Method**: standalone-integrator",
            f"**num_explores**: {num_explores}",
        ],
        writer_title_suffix="(standalone-integrator)",
        rollout_idx=rollout_idx,
    )

    candidates, _ = load_cached_candidates(
        infra.cache_dir, question_id, ctx.benchmark,
    )
    # Truncate to first N cached candidates; recompute cost from kept entries
    # only so paper-style $/q reporting reflects the candidates actually
    # integrated. Default num_explores=8 keeps original "consume all" behavior.
    candidates = candidates[:num_explores]
    explore_cost_total = sum(c.cost_usd for c in candidates)

    if len(candidates) == 0:
        logger.info(f"  [standalone-integrator] No valid candidates for {question_id}")
        return SolveResult(answer="", cost=CostTracker(), rounds=[], writer=ctx.writer)

    ctx.cost.add(explore_cost_total, {}, component="explorer")

    result, traj, cost_usd, usage, dur = await ctx.call_sub_model(
        system_prompt=ctx.benchmark.get_integrator_system_prompt(backend),
        user_message=ctx.benchmark.build_integrator_message(problem, candidates),
        model_cfg=integrate_role.model,
        output_schema=ctx.benchmark.get_integrate_schema(),
        cache_key=f"integrate_standalone_{len(candidates)}",
    )
    assert not result.get("timed_out"), f"integrate call timed out for {question_id}"
    ctx.cost.add(cost_usd, usage, component="integrator")

    final_answer = ctx.benchmark.get_answer_from_integrate(result)

    for i, cand in enumerate(candidates, 1):
        ctx.rounds.append(RoundLog(
            round_num=i, action="explore",
            tool_input={"answer": cand.answer, "cost_usd": cand.cost_usd},
        ))
    ctx.rounds.append(RoundLog(
        round_num=len(candidates) + 1, action="integrate",
        tool_input={"answer": final_answer, "cost_usd": cost_usd},
    ))

    logger.info(f"  [standalone-integrator] {len(candidates)} candidates -> "
          f"answer={final_answer}, cost=${ctx.cost.total_cost_usd:.4f}")

    return ctx.result(final_answer)
