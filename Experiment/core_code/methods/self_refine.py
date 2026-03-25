"""Self-Refine baseline (Madaan et al., 2023).

Three separate prompt "masks" on the same LLM, with accumulating context:
  1. Generator  -- produce initial Draft 0 (reuses benchmark explorer prompt)
  2. Feedback   -- critique the latest draft (separate LLM call)
  3. Refiner    -- revise based on feedback (separate LLM call)

The full iteration history (all drafts + all feedbacks) is included in every
subsequent prompt, so the model sees its complete "error notebook".
Stopping criterion: Feedback says is_correct=True, or hard limit reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from methods.base import Candidate, InfraConfig, create_solve_context
from methods.tts_agent import run_explore
from trajectory import RoundLog, SolveResult
from prompts import format_claude_structured_suffix
from benchmarks.base import ANSWER_FORMAT_RULES


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

FEEDBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "feedback": {
            "type": "string",
            "description": (
                "Detailed critique: identify logical errors, arithmetic mistakes, "
                "missing cases, or flawed assumptions in the current solution"
            ),
        },
        "is_correct": {
            "type": "boolean",
            "description": "True if the current solution is correct and needs no revision",
        },
    },
    "required": ["feedback", "is_correct"],
    "additionalProperties": False,
}

# Refiner uses the same schema as the benchmark's explore schema,
# so get_answer_from_explore() works directly on the result.

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

FEEDBACK_SYSTEM_PROMPT = """\
You are a critical evaluator. You will receive a problem and a solution attempt.

Your job:
1. Carefully verify the LATEST solution step by step.
2. Check for logical errors, arithmetic mistakes, missing edge cases, and flawed assumptions.
3. If the solution is correct, set is_correct to true and explain why it is correct in the feedback field.
4. If the solution has errors, set is_correct to false and describe the specific errors in the feedback field.

Be rigorous: do not accept a solution just because it looks plausible. Verify each step independently.
"""

REFINER_SYSTEM_PROMPT = f"""\
You are an expert problem solver. You will receive a problem, the full history \
of previous solution attempts and their critiques, and the latest feedback.

Your job:
1. Read the full iteration history to understand what has been tried and what went wrong.
2. Address every issue raised in the latest feedback.
3. Produce a corrected, complete solution from scratch (do not just patch the old one).

{ANSWER_FORMAT_RULES}
"""


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------

@dataclass
class Draft:
    reasoning: str
    answer: str
    approach: str
    confidence: float


@dataclass
class IterationHistory:
    """Accumulates the full draft/feedback history for snowball context."""
    drafts: list[Draft] = field(default_factory=list)
    feedbacks: list[str] = field(default_factory=list)  # feedbacks[i] critiques drafts[i]

    def build_feedback_message(self, problem: str) -> str:
        """Build user message for the Feedback call: only problem + current draft (per Algorithm 1 line 3)."""
        latest = self.drafts[-1]
        return (
            f"## Problem\n\n{problem}\n\n"
            f"## Current Solution\n\n"
            f"**Approach:** {latest.approach}\n"
            f"**Reasoning:** {latest.reasoning}\n"
            f"**Answer:** {latest.answer}\n"
            f"**Confidence:** {latest.confidence}\n\n"
            f"Critically evaluate this solution."
        )

    def build_refiner_message(self, problem: str) -> str:
        """Build user message for the Refiner call, including full history + latest feedback."""
        parts = [f"## Problem\n\n{problem}\n\n## Iteration History\n"]
        for i, draft in enumerate(self.drafts):
            parts.append(
                f"### Draft {i}\n"
                f"**Approach:** {draft.approach}\n"
                f"**Reasoning:** {draft.reasoning}\n"
                f"**Answer:** {draft.answer}\n"
                f"**Confidence:** {draft.confidence}\n"
            )
            if i < len(self.feedbacks):
                parts.append(f"### Feedback {i + 1}\n{self.feedbacks[i]}\n")

        parts.append(
            f"\nBased on the latest feedback (Feedback {len(self.feedbacks)}) "
            f"and the full iteration history, produce an improved solution."
        )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

async def solve(
    infra: InfraConfig,
    problem: str,
    image_data_url: str | None = None,
    question_id: str | None = None,
    explore_model: str = "gpt-5.2",
    **_extra,
) -> SolveResult:
    """Solve via Self-Refine: Generate -> (Feedback -> Refine)*."""
    ctx = create_solve_context(
        infra=infra, problem=problem, image_data_url=image_data_url,
        question_id=question_id,
        writer_system_prompt=infra.benchmark.get_explorer_system_prompt(infra.backend),
        writer_user_message=infra.benchmark.build_explorer_message(problem),
        writer_header_lines=[
            f"**Backend**: {infra.backend}",
            f"**Model**: {explore_model}",
            f"**Max iterations**: {infra.max_iterations}",
            f"**Method**: self-refine",
        ],
        writer_title_suffix="(self-refine)",
    )

    history = IterationHistory()

    # -- Step 1: Generator (Draft 0) via run_explore --
    await run_explore(ctx, explore_model)

    if not ctx.state.candidates:
        if not ctx.quiet:
            print("  [self-refine] Draft 0 TIMED OUT, no answer")
        return ctx.result("")

    draft0 = ctx.state.candidates[-1]
    history.drafts.append(Draft(
        reasoning=draft0.reasoning,
        answer=draft0.answer,
        approach=draft0.approach,
        confidence=draft0.confidence,
    ))

    ctx.rounds.append(RoundLog(
        round_num=1,
        action="explore",
        tool_input={"answer": draft0.answer, "cost_usd": draft0.cost_usd},
    ))

    ctx.writer.write_text(
        f"## Draft 0 (Generator)\n\n"
        f"- **Approach**: {draft0.approach}\n"
        f"- **Answer**: {draft0.answer}\n"
        f"- **Confidence**: {draft0.confidence}\n"
        f"- **Cost**: ${draft0.cost_usd}"
    )

    if not ctx.quiet:
        print(f"  [self-refine] Draft 0 (generator): answer={draft0.answer}")

    # -- Steps 2..max_iterations: Feedback -> Refine loop --
    # These are custom prompts (not from benchmark), so we manually add the
    # Claude structured suffix when needed.
    feedback_prompt = FEEDBACK_SYSTEM_PROMPT
    if ctx.backend == "claude":
        feedback_prompt += format_claude_structured_suffix(FEEDBACK_SCHEMA)

    explore_schema = ctx.benchmark.get_explore_schema()
    refiner_prompt = REFINER_SYSTEM_PROMPT
    if ctx.backend == "claude":
        refiner_prompt += format_claude_structured_suffix(explore_schema)

    for i in range(2, infra.max_iterations + 1):
        # -- Feedback call --
        fb_msg = history.build_feedback_message(problem)

        fb_result, fb_traj, fb_cost, fb_usage, fb_dur = await ctx.call_sub_model(
            system_prompt=feedback_prompt,
            user_message=fb_msg,
            model=explore_model,
            output_schema=FEEDBACK_SCHEMA,
            cache_key=f"feedback_{i}",
            writer=ctx.writer,
        )

        ctx.cost.add(fb_cost, fb_usage, component="feedback")

        if fb_result.get("timed_out"):
            if not ctx.quiet:
                print(f"  [self-refine] Feedback {i}: TIMED OUT")
            ctx.writer.write_text(f"## Feedback {i}: TIMED OUT")
            break

        feedback_text = fb_result["feedback"]
        is_correct = fb_result["is_correct"]
        history.feedbacks.append(feedback_text)

        status = "CORRECT" if is_correct else "HAS ERRORS"
        ctx.writer.write_text(
            f"## Feedback {i} ({status})\n\n"
            f"{feedback_text}\n\n"
            f"- **Cost**: ${fb_cost}"
        )

        if not ctx.quiet:
            print(f"  [self-refine] Feedback {i}: {status}")

        # If feedback says the current solution is correct, stop
        if is_correct:
            break

        # -- Refiner call --
        ref_msg = history.build_refiner_message(problem)

        ref_result, ref_traj, ref_cost, ref_usage, ref_dur = await ctx.call_sub_model(
            system_prompt=refiner_prompt,
            user_message=ref_msg,
            model=explore_model,
            output_schema=explore_schema,
            cache_key=f"explore_{i}",
            writer=ctx.writer,
        )

        ctx.cost.add(ref_cost, ref_usage, component="explorer")

        if ref_result.get("timed_out"):
            if not ctx.quiet:
                print(f"  [self-refine] Refiner {i}: TIMED OUT")
            ctx.writer.write_text(f"## Draft {i - 1} (Refiner): TIMED OUT")
            break

        answer = ctx.benchmark.get_answer_from_explore(ref_result)
        history.drafts.append(Draft(
            reasoning=ref_result.get("reasoning", ""),
            answer=answer,
            approach=ref_result.get("approach", ""),
            confidence=ref_result.get("confidence", 0.0),
        ))

        ctx.state.candidates.append(Candidate(
            answer=answer,
            reasoning=ref_result.get("reasoning", ""),
            approach=ref_result.get("approach", ""),
            confidence=ref_result.get("confidence", 0.0),
            cost_usd=ref_cost,
        ))
        ctx.state.current_iteration += 1

        ctx.rounds.append(RoundLog(
            round_num=i,
            action="explore",
            tool_input={"answer": answer, "cost_usd": ref_cost},
        ))

        ctx.writer.write_text(
            f"## Draft {i - 1} (Refiner)\n\n"
            f"- **Approach**: {ref_result.get('approach', '')}\n"
            f"- **Answer**: {answer}\n"
            f"- **Confidence**: {ref_result.get('confidence', 'N/A')}\n"
            f"- **Cost**: ${ref_cost}"
        )

        if not ctx.quiet:
            print(f"  [self-refine] Draft {i - 1} (refiner): answer={answer}, confidence={ref_result.get('confidence', 'N/A')}")

    final_answer = history.drafts[-1].answer

    if not ctx.quiet:
        print(f"  [self-refine] final answer: {final_answer} after {len(ctx.rounds)} drafts")
        print(f"  [self-refine] total cost: ${ctx.cost.total_cost_usd}")

    return ctx.result(final_answer)
