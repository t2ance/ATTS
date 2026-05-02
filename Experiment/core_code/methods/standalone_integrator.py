"""Standalone Integrator baseline (LLM Selection).

Reads pre-generated candidates from a cache directory, feeds ALL candidates
to an LLM integrator in a single call, and returns its synthesized answer.
No orchestrator, no adaptive stopping, no multi-turn context -- isolates
the value of LLM-based aggregation from sequential adaptive exploration.
"""

from __future__ import annotations

from dataclasses import replace
from methods.base import InfraConfig, create_solve_context, load_cached_candidates
from trajectory import CostTracker, RoundLog, SolveResult


async def solve(
    infra: InfraConfig,
    problem: str,
    image_data_url: str | None = None,
    question_id: str | None = None,
    integrate_model: str = "claude-sonnet-4-6",
    **_extra,
) -> SolveResult:
    """Synthesize from all pre-cached candidates in one LLM call."""
    assert infra.cache_dir is not None, "cache_dir is required for standalone-integrator"
    assert question_id is not None, "question_id is required for standalone-integrator"

    ctx = create_solve_context(
        infra=replace(infra, timeout=None),
        problem=problem, image_data_url=image_data_url,
        question_id=question_id,
        writer_system_prompt="(standalone integrator -- single-call synthesis)",
        writer_user_message=problem,
        writer_header_lines=[
            f"**Integrate Model**: {integrate_model}",
            f"**Method**: standalone-integrator",
        ],
        writer_title_suffix="(standalone-integrator)",
    )

    candidates, explore_cost_total = load_cached_candidates(
        infra.cache_dir, question_id, ctx.benchmark,
    )

    if len(candidates) == 0:
        print(f"  [standalone-integrator] No valid candidates for {question_id}")
        return SolveResult(answer="", cost=CostTracker(), rounds=[], writer=ctx.writer)

    ctx.cost.add(explore_cost_total, {}, component="explorer")

    result, traj, cost_usd, usage, dur = await ctx.call_sub_model(
        system_prompt=ctx.benchmark.get_integrator_system_prompt(ctx.backend),
        user_message=ctx.benchmark.build_integrator_message(problem, candidates),
        model=integrate_model,
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

    print(f"  [standalone-integrator] {len(candidates)} candidates -> "
          f"answer={final_answer}, cost=${ctx.cost.total_cost_usd:.4f}")

    return ctx.result(final_answer)
