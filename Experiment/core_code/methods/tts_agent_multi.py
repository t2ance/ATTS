"""TTS agent with multi-model explore: orchestrator chooses which model to use per explore call."""

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
from methods.tts_agent import run_integrate, _log_round, process_explore_result, make_structured_output_handler
from trajectory import RoundLog, SolveResult, TrajectoryWriter
from prompts import (
    ORCHESTRATOR_MULTI_MODEL_SYSTEM_PROMPT,
    build_user_message,
)
from logger import RunLogger


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
}


# ---------------------------------------------------------------------------
# Explore tool with model parameter
# ---------------------------------------------------------------------------

EXPLORE_TOOL_MULTI: dict[str, Any] = {
    "name": "explore",
    "description": (
        "Dispatch a fresh, independent solver to generate a new candidate answer. "
        "You must specify which model to use. "
        "Haiku is the cheapest and fastest but weakest. "
        "Sonnet is mid-range. "
        "Opus is the most expensive but strongest. "
        "Each model has its own budget limit. Do NOT call a model whose budget is exhausted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "enum": ["haiku", "sonnet", "opus"],
                "description": "Which model to use for this explore call.",
            },
        },
        "required": ["model"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Multi-model solve context
# ---------------------------------------------------------------------------

@dataclass
class MultiModelSolveContext:
    """Solve context with per-model sub_model callers."""
    ctx: SolveContext
    model_callers: dict[str, Any]  # {alias: sub_model_fn}
    model_budgets: dict[str, int]  # {alias: max explores for this model}
    model_explore_counts: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.model_explore_counts = {m: 0 for m in self.model_budgets}
        self.model_costs: dict[str, float] = {m: 0.0 for m in self.model_budgets}

    @property
    def state(self) -> SolvingState:
        return self.ctx.state

    def record_cost(self, model_alias: str, cost: float) -> None:
        self.model_costs[model_alias] += cost

    def budget_status_text(self) -> str:
        """Format per-model budget status for tool return."""
        parts = []
        for alias in self.model_budgets:
            used = self.model_explore_counts.get(alias, 0)
            limit = self.model_budgets[alias]
            remaining = limit - used
            spent = self.model_costs.get(alias, 0.0)
            status = "EXHAUSTED" if remaining <= 0 else f"{remaining} remaining"
            parts.append(f"{alias}: {used}/{limit} used, ${spent:.2f} spent ({status})")
        total_spent = sum(self.model_costs.values())
        return "Per-model budget: " + ", ".join(parts) + f". Total explore cost: ${total_spent:.2f}."

    async def call_explore(self, model_alias: str, **kwargs):
        """Call sub-model using the correct model's cache."""
        assert model_alias in self.model_callers, f"Unknown model: {model_alias}"
        self.model_explore_counts[model_alias] += 1
        explore_idx = self.model_explore_counts[model_alias]
        caller = self.model_callers[model_alias]
        model_id = MODEL_ALIASES[model_alias]

        return await caller(
            kwargs["system_prompt"],
            kwargs["user_message"],
            self.ctx.image_data_url,
            model_id,
            kwargs["output_schema"],
            cache_key=f"explore_{explore_idx}",
            writer=kwargs.get("writer", TrajectoryWriter.noop()),
            budget_tokens=self.ctx.budget_tokens,
            effort=self.ctx.effort,
        )


# ---------------------------------------------------------------------------
# Multi-model explore
# ---------------------------------------------------------------------------

async def run_explore_multi(mctx: MultiModelSolveContext, model_alias: str) -> str:
    """Run an explore with a specific model. Returns tool result text."""
    used = mctx.model_explore_counts.get(model_alias, 0)
    limit = mctx.model_budgets[model_alias]
    assert used < limit, (
        f"{model_alias} budget exhausted: {used}/{limit} used. "
        f"The orchestrator should not have called explore with this model."
    )

    ctx = mctx.ctx
    user_msg = ctx.benchmark.build_explorer_message(ctx.state.problem)
    explorer_system_prompt = ctx.benchmark.get_explorer_system_prompt(ctx.backend)
    explore_schema = ctx.benchmark.get_explore_schema()

    result, trajectory_text, explore_cost, explore_usage, duration = await mctx.call_explore(
        model_alias,
        system_prompt=explorer_system_prompt,
        user_message=user_msg,
        output_schema=explore_schema,
    )

    mctx.record_cost(model_alias, explore_cost)

    return process_explore_result(
        ctx, result, explore_cost, explore_usage,
        model_label=model_alias,
        extra_budget_text=f"\n{mctx.budget_status_text()}",
    )


# ---------------------------------------------------------------------------
# Orchestrator loop
# ---------------------------------------------------------------------------

async def _run_orchestrator_multi(
    mctx: MultiModelSolveContext,
    orchestrator_model: str,
    user_message_text: str,
) -> None:
    """Run the multi-model orchestrator loop."""
    ctx = mctx.ctx
    backend_mod = import_module(f"backends.{ctx.backend}")

    async def tool_handler(name: str, args: dict) -> tuple[str, bool]:
        if name == "explore":
            model_alias = args.get("model", "sonnet")
            n_before = len(ctx.state.candidates)
            result_text = await run_explore_multi(mctx, model_alias)
            if len(ctx.state.candidates) > n_before:
                cand = ctx.state.candidates[-1]
                _log_round(ctx, RoundLog(
                    round_num=ctx.state.explore.used,
                    action="explore",
                    tool_input={
                        "model": model_alias,
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

    system_prompt = ORCHESTRATOR_MULTI_MODEL_SYSTEM_PROMPT
    tools = [EXPLORE_TOOL_MULTI]
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
    explore_model: str = "claude-sonnet-4-6",  # unused, kept for interface compat
    integrate_model: str = "claude-sonnet-4-6",  # unused
    cache_dirs: dict[str, Path] | None = None,
    model_budgets: dict[str, int] | None = None,
    exploration_effort: str | None = None,
    **_extra,
) -> SolveResult:
    """Solve a problem using multi-model delegated test-time scaling.

    cache_dirs: {model_alias: cache_base_path} e.g. {"haiku": Path("cache/haiku"), "sonnet": ...}
    model_budgets: {model_alias: max_explores} e.g. {"haiku": 4, "sonnet": 4, "opus": 4}
    exploration_effort: "low", "medium", or "high" — controls stopping aggressiveness
    """
    assert cache_dirs is not None, "cache_dirs required for multi-model solve"
    assert model_budgets is not None, "model_budgets required for multi-model solve"

    user_message_text = build_user_message(problem, infra.max_iterations, model_budgets, exploration_effort)

    # Create base solve context (uses first available cache_dir for trajectory)
    from methods.base import create_solve_context
    ctx = create_solve_context(
        infra=infra, problem=problem, image_data_url=image_data_url,
        question_id=question_id,
        writer_system_prompt=ORCHESTRATOR_MULTI_MODEL_SYSTEM_PROMPT,
        writer_user_message=user_message_text,
        writer_header_lines=[
            f"**Backend**: {infra.backend}",
            f"**Orchestrator**: {orchestrator_model}",
            f"**Explorer models**: {', '.join(cache_dirs.keys())}",
            f"**Max iterations**: {infra.max_iterations}",
        ],
        writer_title_suffix="(multi-model)",
    )

    # Create per-model sub_model callers
    model_callers = {}
    for alias, cache_base in cache_dirs.items():
        question_cache_dir = cache_base / question_id if question_id else None
        model_callers[alias] = make_sub_model_caller(
            infra.backend, question_cache_dir, infra.cache_only,
            traj_dir=ctx.traj_dir, timeout=infra.timeout,
        )

    mctx = MultiModelSolveContext(ctx=ctx, model_callers=model_callers, model_budgets=model_budgets)

    if not ctx.quiet:
        print(f"\nMulti-model TTS Agent [{infra.backend}] -- solving with up to {infra.max_iterations} rounds")
        print(f"Available models: {list(cache_dirs.keys())}")
        print(f"Problem: {problem[:100]}")
        print()

    await _run_orchestrator_multi(mctx, orchestrator_model, user_message_text)

    assert ctx.state.final_answer is not None, "Orchestrator did not submit a final answer"

    if not ctx.quiet:
        print(f"\nTotal cost: ${ctx.cost.total_cost_usd}"
              f" (input: {ctx.cost.total_input_tokens}, output: {ctx.cost.total_output_tokens})")

    return ctx.result(ctx.state.final_answer)
