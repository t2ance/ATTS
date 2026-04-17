"""TTS agent with effort selection: orchestrator chooses reasoning effort per explore call."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from methods.base import (
    Candidate,
    InfraConfig,
    SolveContext,
    SolvingState,
    CostTracker,
    make_sub_model_caller,
)
from methods.tts_agent import _log_round, process_explore_result, make_structured_output_handler
from trajectory import RoundLog, SolveResult, TrajectoryWriter
from prompts import (
    ORCHESTRATOR_EFFORT_SYSTEM_PROMPT,
    build_user_message,
)
from logger import RunLogger


# ---------------------------------------------------------------------------
# Effort levels
# ---------------------------------------------------------------------------

EFFORT_LEVELS = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Explore tool with effort parameter
# ---------------------------------------------------------------------------

EXPLORE_TOOL_EFFORT: dict[str, Any] = {
    "name": "explore",
    "description": (
        "Dispatch a fresh, independent solver to generate a new candidate answer. "
        "You must specify the reasoning effort level. "
        "Low is cheapest but uses minimal chain-of-thought. "
        "Medium uses standard extended thinking. "
        "High uses deep analysis with extensive chain-of-thought. "
        "Each effort level has its own budget limit."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "effort": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Reasoning effort for this explore call.",
            },
        },
        "required": ["effort"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Effort solve context
# ---------------------------------------------------------------------------

@dataclass
class EffortSolveContext:
    """Solve context with per-effort sub_model callers."""
    ctx: SolveContext
    effort_callers: dict[str, Any]  # {effort_level: sub_model_fn}
    effort_budgets: dict[str, int]  # {effort_level: max explores}
    effort_explore_counts: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.effort_explore_counts = {e: 0 for e in self.effort_budgets}
        self.effort_costs: dict[str, float] = {e: 0.0 for e in self.effort_budgets}

    @property
    def state(self) -> SolvingState:
        return self.ctx.state

    def record_cost(self, effort_level: str, cost: float) -> None:
        self.effort_costs[effort_level] += cost

    def budget_status_text(self) -> str:
        parts = []
        for level in self.effort_budgets:
            used = self.effort_explore_counts.get(level, 0)
            limit = self.effort_budgets[level]
            remaining = limit - used
            spent = self.effort_costs.get(level, 0.0)
            status = "EXHAUSTED" if remaining <= 0 else f"{remaining} remaining"
            parts.append(f"{level}: {used}/{limit} used, ${spent:.2f} spent ({status})")
        total_spent = sum(self.effort_costs.values())
        return "Per-effort budget: " + ", ".join(parts) + f". Total explore cost: ${total_spent:.2f}."

    async def call_explore(self, effort_level: str, model: str, **kwargs):
        assert effort_level in self.effort_callers, f"Unknown effort: {effort_level}"
        self.effort_explore_counts[effort_level] += 1
        explore_idx = self.effort_explore_counts[effort_level]
        caller = self.effort_callers[effort_level]

        return await caller(
            kwargs["system_prompt"],
            kwargs["user_message"],
            self.ctx.image_data_url,
            model,
            kwargs["output_schema"],
            cache_key=f"explore_{explore_idx}",
            writer=kwargs.get("writer", TrajectoryWriter.noop()),
            budget_tokens=self.ctx.budget_tokens,
            effort=effort_level,
        )


# ---------------------------------------------------------------------------
# Effort-adaptive explore
# ---------------------------------------------------------------------------

async def run_explore_effort(ectx: EffortSolveContext, effort_level: str, explore_model: str) -> str:
    used = ectx.effort_explore_counts.get(effort_level, 0)
    limit = ectx.effort_budgets[effort_level]
    assert used <= limit, (
        f"{effort_level} budget exhausted: {used}/{limit} used."
    )

    ctx = ectx.ctx
    user_msg = ctx.benchmark.build_explorer_message(ctx.state.problem)
    explorer_system_prompt = ctx.benchmark.get_explorer_system_prompt(ctx.backend)
    explore_schema = ctx.benchmark.get_explore_schema()

    result, trajectory_text, explore_cost, explore_usage, duration = await ectx.call_explore(
        effort_level,
        explore_model,
        system_prompt=explorer_system_prompt,
        user_message=user_msg,
        output_schema=explore_schema,
    )

    ectx.record_cost(effort_level, explore_cost)

    return process_explore_result(
        ctx, result, explore_cost, explore_usage,
        model_label=f"effort={effort_level}",
        extra_budget_text=f"\n{ectx.budget_status_text()}",
    )


# ---------------------------------------------------------------------------
# Orchestrator loop
# ---------------------------------------------------------------------------

async def _run_orchestrator_effort(
    ectx: EffortSolveContext,
    orchestrator_model: str,
    explore_model: str,
    user_message_text: str,
) -> None:
    ctx = ectx.ctx
    backend_mod = import_module(f"backends.{ctx.backend}")

    async def tool_handler(name: str, args: dict) -> tuple[str, bool]:
        if name == "explore":
            effort_level = args.get("effort", "medium")
            n_before = len(ctx.state.candidates)
            result_text = await run_explore_effort(ectx, effort_level, explore_model)
            if len(ctx.state.candidates) > n_before:
                cand = ctx.state.candidates[-1]
                _log_round(ctx, RoundLog(
                    round_num=ctx.state.explore.used,
                    action="explore",
                    tool_input={
                        "effort": effort_level,
                        "answer": cand.answer,
                        "reasoning": cand.reasoning,
                        "approach": cand.approach,
                        "confidence": cand.confidence,
                        "cost_usd": cand.cost_usd,
                    },
                ))
            return result_text, False
        else:
            assert False, f"Unknown tool: {name}"

    system_prompt = ORCHESTRATOR_EFFORT_SYSTEM_PROMPT
    tools = [EXPLORE_TOOL_EFFORT]
    output_format = {"type": "json_schema", "schema": ctx.benchmark.get_explore_schema()}

    cost, usage, _ = await backend_mod.run_tool_conversation(
        system_prompt=system_prompt,
        user_message=user_message_text,
        image_data_url=ctx.image_data_url,
        model=orchestrator_model,
        tools=tools,
        max_turns=ctx.state.explore.max_explores + 2,
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
    orchestrator_model: str = "claude-sonnet-4-6",
    explore_model: str = "claude-sonnet-4-6",
    integrate_model: str = "claude-sonnet-4-6",  # unused
    cache_dirs: dict[str, Path] | None = None,
    effort_budgets: dict[str, int] | None = None,
    **_extra,
) -> SolveResult:
    """Solve using effort-adaptive exploration.

    cache_dirs: {effort_level: cache_base_path} e.g. {"low": Path("cache/low"), "medium": ...}
    effort_budgets: {effort_level: max_explores} e.g. {"low": 8, "medium": 6, "high": 4}
    """
    assert cache_dirs is not None, "cache_dirs required for effort-adaptive solve"
    assert effort_budgets is not None, "effort_budgets required for effort-adaptive solve"

    max_iterations = sum(effort_budgets.values())
    infra = InfraConfig(
        benchmark=infra.benchmark,
        backend=infra.backend,
        cache_dir=infra.cache_dir,
        cache_only=infra.cache_only,
        quiet=infra.quiet,
        max_iterations=max_iterations,
        timeout=infra.timeout,
        budget_tokens=infra.budget_tokens,
        effort=infra.effort,
        enable_integrate=False,
    )

    user_message_text = build_user_message(problem, max_iterations, effort_budgets=effort_budgets)

    from methods.base import create_solve_context
    ctx = create_solve_context(
        infra=infra, problem=problem, image_data_url=image_data_url,
        question_id=question_id,
        writer_system_prompt=ORCHESTRATOR_EFFORT_SYSTEM_PROMPT,
        writer_user_message=user_message_text,
        writer_header_lines=[
            f"**Backend**: {infra.backend}",
            f"**Orchestrator**: {orchestrator_model}",
            f"**Explorer**: {explore_model}",
            f"**Effort budgets**: {effort_budgets}",
            f"**Max iterations**: {max_iterations}",
        ],
        writer_title_suffix="(effort-adaptive)",
    )

    # Create per-effort sub_model callers
    effort_callers = {}
    for level, cache_base in cache_dirs.items():
        question_cache_dir = cache_base / question_id if question_id else None
        effort_callers[level] = make_sub_model_caller(
            infra.backend, question_cache_dir, infra.cache_only,
            traj_dir=ctx.traj_dir, timeout=infra.timeout,
        )

    ectx = EffortSolveContext(ctx=ctx, effort_callers=effort_callers, effort_budgets=effort_budgets)

    if not ctx.quiet:
        print(f"\nEffort-adaptive TTS Agent [{infra.backend}] -- solving with up to {max_iterations} rounds")
        print(f"Effort levels: {list(cache_dirs.keys())}")
        print(f"Problem: {problem[:100]}")
        print()

    await _run_orchestrator_effort(ectx, orchestrator_model, explore_model, user_message_text)

    assert ctx.state.final_answer is not None, "Orchestrator did not submit a final answer"

    if not ctx.quiet:
        print(f"\nTotal cost: ${ctx.cost.total_cost_usd}"
              f" (input: {ctx.cost.total_input_tokens}, output: {ctx.cost.total_output_tokens})")

    return ctx.result(ctx.state.final_answer)
