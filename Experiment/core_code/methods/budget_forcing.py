"""Budget Forcing baseline (Muennighoff et al., 2025).

Simulates the s1 compute-control strategy at the API level: after each
generation, the model's previous reasoning is fed back with "Wait" appended,
forcing the model to continue deliberating before producing a final answer.

All N rounds run (no early stopping) since budget forcing is about forcing
more compute, not self-evaluation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from methods.base import Candidate, InfraConfig, create_solve_context
from methods.tool_state import advance
from trajectory import RoundLog, SolveResult


async def solve(
    infra: InfraConfig,
    problem: str,
    *,
    spec,  # methods.specs.BudgetForcingSpec
    image_data_url: str | None = None,
    question_id: str | None = None,
    rollout_idx: int | None = None,
    **_extra,
) -> SolveResult:
    """Solve via Budget Forcing: Generate -> (Wait -> Regenerate)*."""
    variant = spec.explore  # ExploreVariant
    ctx = create_solve_context(
        infra=infra,
        backend=variant.model.backend,
        timeout=variant.model.timeout,
        problem=problem,
        image_data_url=image_data_url,
        question_id=question_id,
        writer_system_prompt=infra.benchmark.get_explorer_system_prompt(variant.model.backend),
        writer_user_message=infra.benchmark.build_explorer_message(problem),
        writer_header_lines=[
            f"**Backend**: {variant.model.backend}",
            f"**Model**: {variant.model.model}",
            f"**Max iterations**: {infra.max_iterations}",
            f"**Method**: budget-forcing",
        ],
        writer_title_suffix="(budget-forcing)",
        rollout_idx=rollout_idx,
    )

    # get_explorer_system_prompt(backend) already includes Claude structured
    # suffix when backend == "claude", so no manual suffix append needed.
    system_prompt = ctx.benchmark.get_explorer_system_prompt(variant.model.backend)
    explore_schema = ctx.benchmark.get_explore_schema()
    user_msg = ctx.benchmark.build_explorer_message(problem)

    prev_trajectory = ""

    for i in range(1, infra.max_iterations + 1):
        msg = user_msg if i == 1 else f"{user_msg}\n\n{prev_trajectory}\n\nWait"

        result, trajectory_text, r_cost, usage, duration = await ctx.call_sub_model(
            system_prompt=system_prompt,
            user_message=msg,
            model_cfg=variant.model,
            output_schema=explore_schema,
            cache_key=f"explore_{i}",
            writer=ctx.writer,
        )

        ctx.cost.add(r_cost, usage, component="explorer")
        ctx.state.explore = advance(ctx.state.explore)

        if result.get("timed_out"):
            logger.info(f"  [budget-forcing] Round {i}: TIMED OUT")
            ctx.writer.write_text(f"## Round {i}: TIMED OUT")
            break

        answer = ctx.benchmark.get_answer_from_explore(result)
        ctx.state.candidates.append(Candidate(
            answer=answer,
            reasoning=result.get("reasoning", ""),
            approach=result.get("approach", ""),
            confidence=result.get("confidence", 0.0),
            cost_usd=r_cost,
        ))

        ctx.rounds.append(RoundLog(
            round_num=i,
            action="explore",
            tool_input={"answer": answer, "cost_usd": r_cost},
        ))

        label = "Initial" if i == 1 else "Budget Forcing"
        ctx.writer.write_text(
            f"## Round {i} ({label})\n\n"
            f"- **Approach**: {result.get('approach', '')}\n"
            f"- **Answer**: {answer}\n"
            f"- **Confidence**: {result.get('confidence', 'N/A')}\n"
            f"- **Cost**: ${r_cost}"
        )

        logger.info(f"  [budget-forcing] Round {i}: answer={answer}, confidence={result.get('confidence', 'N/A')}")

        prev_trajectory = trajectory_text

    final_answer = ctx.state.candidates[-1].answer if ctx.state.candidates else ""

    logger.info(f"  [budget-forcing] final answer: {final_answer} after {len(ctx.rounds)} rounds")
    logger.info(f"  [budget-forcing] total cost: ${ctx.cost.total_cost_usd}")

    return ctx.result(final_answer)
