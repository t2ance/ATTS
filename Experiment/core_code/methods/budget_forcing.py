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

import asyncio
import time
from importlib import import_module

from cache_types import Exploration
from methods.base import Candidate, InfraConfig, create_solve_context
from methods.tool_state import advance
from trajectory import RoundLog, SolveResult, TrajectoryWriter


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
    qid = question_id or ""

    def _make_explore_gen_fn(idx: int, system_prompt: str, user_message: str):
        async def gen() -> Exploration:
            if infra.cache_only:
                raise AssertionError(
                    f"cache_only mode: explore cache miss at {variant._explore_dir(qid, idx, rollout_idx)}"
                )
            backend_mod = import_module(f"backends.{variant.model.backend}")
            t0 = time.time()
            api_coro = backend_mod.call_sub_model(
                system_prompt, user_message, image_data_url,
                variant.model.model, explore_schema,
                TrajectoryWriter.noop(),
                budget_tokens=variant.model.budget_tokens,
                effort=variant.model.effort,
                sampling=(variant.model.vllm_sampling.model_dump()
                          if variant.model.vllm_sampling is not None else None),
                provider_order=variant.model.openrouter_provider_order,
                provider_allow_fallbacks=variant.model.openrouter_provider_allow_fallbacks,
            )
            timeout = variant.model.timeout
            if timeout is not None:
                try:
                    result, traj, cost, usage = await asyncio.wait_for(api_coro, timeout=timeout)
                except asyncio.TimeoutError:
                    duration = time.time() - t0
                    return Exploration(
                        qid=qid, idx=idx, rollout_idx=rollout_idx,
                        answer="", trajectory="", cost_usd=0.0, model=variant.model.model,
                        timed_out=True,
                        extra={"timeout_seconds": timeout, "duration_seconds": duration},
                        system_prompt=system_prompt, user_message=user_message,
                    )
            else:
                result, traj, cost, usage = await api_coro
            answer = ctx.benchmark.get_answer_from_explore(result)
            return Exploration(
                qid=qid, idx=idx, rollout_idx=rollout_idx,
                answer=answer, trajectory=traj, cost_usd=cost, model=variant.model.model,
                timed_out=bool(result.get("timed_out", False)),
                extra={"usage": usage, **{k: v for k, v in result.items() if k != "answer"}},
                system_prompt=system_prompt, user_message=user_message,
            )
        return gen

    for i in range(1, infra.max_iterations + 1):
        msg = user_msg if i == 1 else f"{user_msg}\n\n{prev_trajectory}\n\nWait"

        exp_i = await variant.get_exploration(
            qid, i, rollout_idx=rollout_idx,
            generate_fn=_make_explore_gen_fn(i, system_prompt, msg),
        )
        if ctx.traj_dir is not None:
            exp_i.persist(ctx.traj_dir / f"explore_{i}")

        r_cost = exp_i.cost_usd
        extra = exp_i.extra
        ctx.cost.add(r_cost, extra.get("usage", {}), component="explorer")
        ctx.state.explore = advance(ctx.state.explore)

        if exp_i.timed_out:
            logger.info(f"  [budget-forcing] Round {i}: TIMED OUT")
            ctx.writer.write_text(f"## Round {i}: TIMED OUT")
            break

        answer = exp_i.answer
        ctx.state.candidates.append(Candidate(
            answer=answer,
            reasoning=extra.get("reasoning", ""),
            approach=extra.get("approach", ""),
            confidence=extra.get("confidence", 0.0),
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
            f"- **Approach**: {extra.get('approach', '')}\n"
            f"- **Answer**: {answer}\n"
            f"- **Confidence**: {extra.get('confidence', 'N/A')}\n"
            f"- **Cost**: ${r_cost}"
        )

        logger.info(f"  [budget-forcing] Round {i}: answer={answer}, confidence={extra.get('confidence', 'N/A')}")

        prev_trajectory = exp_i.trajectory

    final_answer = ctx.state.candidates[-1].answer if ctx.state.candidates else ""

    logger.info(f"  [budget-forcing] final answer: {final_answer} after {len(ctx.rounds)} rounds")
    logger.info(f"  [budget-forcing] total cost: ${ctx.cost.total_cost_usd}")

    return ctx.result(final_answer)
