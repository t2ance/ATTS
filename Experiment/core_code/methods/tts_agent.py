"""TTS (Test-Time Scaling) agent: orchestrator-driven explore/integrate loop."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from methods.base import (
    Candidate,
    InfraConfig,
    SolveContext,
    create_solve_context,
)
from trajectory import RoundLog, SolveResult
from prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_NO_INTEGRATE_SYSTEM_PROMPT,
    build_user_message,
)


# ---------------------------------------------------------------------------
# Orchestrator tool definitions (backend-agnostic)
# ---------------------------------------------------------------------------

EXPLORE_TOOL: dict[str, Any] = {
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
    result: dict,
    explore_cost: float,
    explore_usage: dict,
    model_label: str = "",
    extra_budget_text: str = "",
) -> str:
    """Process an explore result: update state, return tool result text.

    Shared by single-model and multi-model explore.
    """
    state = ctx.state
    state.current_iteration += 1
    used = state.current_iteration
    remaining = state.max_iterations - used
    label = f" Model: {model_label}." if model_label else ""

    if result.get("timed_out"):
        if not ctx.quiet:
            print(f"  [sub-model] explore #{used}{(' (' + model_label + ')') if model_label else ''}: TIMED OUT -- recording empty candidate")
        state.candidates.append(
            Candidate(answer="", reasoning="timed out", approach="", confidence=0.0, cost_usd=0.0)
        )
        ctx.writer.write_explore_timeout()
        return (
            f"Candidate #{used} recorded (timed out, empty answer).{label}\n"
            f"Explore budget: {used}/{state.max_iterations} used, {remaining} remaining."
            f"{extra_budget_text}"
        )

    ctx.cost.add(explore_cost, explore_usage, component="explorer")
    answer = ctx.benchmark.get_answer_from_explore(result)

    state.candidates.append(
        Candidate(
            answer=answer,
            reasoning=result.get("reasoning", ""),
            approach=result.get("approach", ""),
            confidence=result.get("confidence", 0.0),
            cost_usd=explore_cost,
        )
    )

    text = (
        f"Candidate #{used} recorded.{label}\n"
        f"- Answer: {answer}\n"
        f"- Confidence: {result.get('confidence', 'N/A')}\n"
        f"- Approach: {result.get('approach', 'N/A')}\n"
        f"- Reasoning: {result.get('reasoning', 'N/A')}\n"
        f"- Cost: ${explore_cost:.2f}\n\n"
        f"Explore budget: {used}/{state.max_iterations} used, {remaining} remaining."
        f"{extra_budget_text}"
    )
    if not ctx.quiet:
        print(f"  [sub-model] explore candidate #{len(state.candidates)}{(' (' + model_label + ')') if model_label else ''}: answer={answer}, confidence={result.get('confidence', 'N/A')}")
    return text


def make_structured_output_handler(ctx: SolveContext):
    """Create the on_structured_output callback. Shared by single and multi-model."""
    def on_structured_output(result: dict) -> None:
        ctx.state.final_answer = ctx.benchmark.get_answer_from_explore(result)
        ctx.state.final_reasoning = result.get("reasoning", "")
        if not ctx.quiet:
            print(f"[structured_output] answer={ctx.state.final_answer}")
        _log_round(ctx, RoundLog(
            round_num=ctx.state.current_iteration + 1,
            action="submit_answer",
            tool_input=result,
        ))
    return on_structured_output


async def run_explore(ctx: SolveContext, explore_model: str) -> str:
    """Run an explore sub-model call. Returns tool result text."""
    explore_idx = ctx.state.current_iteration + 1
    user_msg = ctx.benchmark.build_explorer_message(ctx.state.problem)
    explorer_system_prompt = ctx.benchmark.get_explorer_system_prompt(ctx.backend)
    explore_schema = ctx.benchmark.get_explore_schema()

    result, trajectory_text, explore_cost, explore_usage, duration = await ctx.call_sub_model(
        system_prompt=explorer_system_prompt,
        user_message=user_msg,
        model=explore_model,
        output_schema=explore_schema,
        cache_key=f"explore_{explore_idx}",
    )

    return process_explore_result(ctx, result, explore_cost, explore_usage)


async def run_integrate(ctx: SolveContext, integrate_model: str) -> str:
    """Run an integrate sub-model call. Returns tool result text."""
    state = ctx.state
    assert state.candidates, "integrate called with no candidates"

    integrator_system_prompt = ctx.benchmark.get_integrator_system_prompt(ctx.backend)
    integrate_schema = ctx.benchmark.get_integrate_schema()
    user_msg = ctx.benchmark.build_integrator_message(state.problem, state.candidates)

    result, trajectory_text, cost_usd, usage, duration = await ctx.call_sub_model(
        system_prompt=integrator_system_prompt,
        user_message=user_msg,
        model=integrate_model,
        output_schema=integrate_schema,
        cache_key=f"integrate_{state.current_iteration + 1}",
    )
    ctx.cost.add(cost_usd, usage, component="integrator")

    final_answer = ctx.benchmark.get_answer_from_integrate(result)
    state.final_answer = final_answer
    state.final_reasoning = result.get("reasoning")
    state.final_analysis = result.get("analysis")

    if not ctx.quiet:
        print(f"  [sub-model] integrate: final_answer={final_answer}")
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
        )


async def _run_orchestrator(
    ctx: SolveContext,
    orchestrator_model: str,
    explore_model: str,
    integrate_model: str,
    user_message_text: str,
    enable_integrate: bool = True,
) -> None:
    """Run the orchestrator loop via the backend's run_tool_conversation."""
    backend_mod = import_module(f"backends.{ctx.backend}")

    async def tool_handler(name: str, args: dict) -> tuple[str, bool]:
        if name == "explore":
            n_before = len(ctx.state.candidates)
            result_text = await run_explore(ctx, explore_model)
            if len(ctx.state.candidates) > n_before:
                cand = ctx.state.candidates[-1]
                _log_round(ctx, RoundLog(
                    round_num=ctx.state.current_iteration,
                    action="explore",
                    tool_input={
                        "answer": cand.answer,
                        "reasoning": cand.reasoning,
                        "approach": cand.approach,
                        "confidence": cand.confidence,
                        "cost_usd": cand.cost_usd,
                    },
                ))
            return result_text, False
        elif name == "integrate":
            result_text = await run_integrate(ctx, integrate_model)
            _log_round(ctx, RoundLog(
                round_num=ctx.state.current_iteration + 1,
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

    if enable_integrate:
        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT
        tools = [EXPLORE_TOOL, INTEGRATE_TOOL]
        output_format = None
    else:
        system_prompt = ORCHESTRATOR_NO_INTEGRATE_SYSTEM_PROMPT
        tools = [EXPLORE_TOOL]
        output_format = {"type": "json_schema", "schema": ctx.benchmark.get_explore_schema()}

    cost, usage = await backend_mod.run_tool_conversation(
        system_prompt=system_prompt,
        user_message=user_message_text,
        image_data_url=ctx.image_data_url,
        model=orchestrator_model,
        tools=tools,
        max_turns=ctx.state.max_iterations + 2,
        tool_handler=tool_handler,
        effort=ctx.effort,
        output_format=output_format,
        writer=ctx.writer,
        quiet=ctx.quiet,
        on_structured_output=make_structured_output_handler(ctx),
    )
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
    image_data_url: str | None = None,
    question_id: str | None = None,
    orchestrator_model: str = "gpt-5.2",
    explore_model: str = "gpt-5.2",
    integrate_model: str = "gpt-5.2",
    **_extra,
) -> SolveResult:
    """Solve a problem using delegated test-time scaling."""
    user_message_text = build_user_message(problem, infra.max_iterations)
    writer_prompt = ORCHESTRATOR_NO_INTEGRATE_SYSTEM_PROMPT if not infra.enable_integrate else ORCHESTRATOR_SYSTEM_PROMPT
    ctx = create_solve_context(
        infra=infra, problem=problem, image_data_url=image_data_url,
        question_id=question_id,
        writer_system_prompt=writer_prompt,
        writer_user_message=user_message_text,
        writer_header_lines=[
            f"**Backend**: {infra.backend}",
            f"**Orchestrator**: {orchestrator_model}",
            f"**Explorer**: {explore_model}",
            f"**Integrator**: {integrate_model}",
            f"**Max iterations**: {infra.max_iterations}",
        ],
        writer_title_suffix="(delegated)",
    )

    if not ctx.quiet:
        print(f"\nDelegated TTS Agent [{infra.backend}] -- solving with up to {infra.max_iterations} rounds")
        print(f"Problem: {problem}")
        if image_data_url:
            print("Image: included")
        print()

    await _run_orchestrator(
        ctx, orchestrator_model, explore_model, integrate_model, user_message_text,
        enable_integrate=infra.enable_integrate,
    )

    if ctx.state.final_answer is None:
        ctx.state.final_answer = ""

    if not ctx.quiet:
        print(f"\nTotal cost: ${ctx.cost.total_cost_usd}"
              f" (input: {ctx.cost.total_input_tokens}, output: {ctx.cost.total_output_tokens})")

    return ctx.result(ctx.state.final_answer)
