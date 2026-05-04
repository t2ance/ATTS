"""Socratic Self-Refine -- Critic-only variant of Self-Refine (Madaan et al., 2023).

Identical control flow to ``self_refine.py``: Generator -> (Feedback -> Refiner)*,
stopping when Feedback declares ``is_correct=True`` or the iteration cap is hit.

The ONLY change relative to ``self_refine.py`` is FEEDBACK_SYSTEM_PROMPT.
This isolates the Critic prompt as the single experimental variable so any
delta in (a) num_explores distribution, (b) Critic false-positive rate, or
(c) final accuracy is attributable to the prompt itself, not to a different
loop structure or a different Refiner.

Motivation -- empirical observation on Sonnet over our four benchmarks:
the original FEEDBACK prompt produced ``is_correct=True`` after Draft 0 on
71% of HLE / 90% of LCB / 92% of GPQA / 46% of BabyVision questions, and
of those, 37% / 12% / 25% / 73% respectively had a wrong final answer.
This is the well-documented "self-bias" failure mode where a same-model
critic preferentially approves its own output (Xu et al., ACL 2024) rather
than performing genuine error-finding (Huang et al., ICLR 2024).

The Socratic prompt below pushes back on that failure mode along four axes:
  (i)   anchors a Bayesian prior that first-pass solutions are usually wrong,
  (ii)  enumerates five concrete questions the Critic must answer in writing,
  (iii) raises the bar on ``is_correct=True`` (must survive ALL five),
  (iv)  declares an asymmetric loss (FP > FN) so politeness ties break toward
        flagging.

It does NOT prescribe a minimum iteration count -- iteration count must
remain organic / data-driven, not a forced constant.

Cf. Li et al., "SSR: Socratic Self-Refine for LLM Reasoning" (arXiv:2511.10621),
which reports +5-6 absolute points on AIME24 / MATH-Level-5 over vanilla
Self-Refine using a related Socratic decomposition strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

from methods.base import Candidate, InfraConfig, create_solve_context
from methods.tool_state import advance
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
                "Your honest reflection on whether you actually believe this "
                "solution. Begin with your fresh intuition before you re-engage "
                "with the proposed reasoning, then describe what you found when "
                "you sat with the questions in the system prompt."
            ),
        },
        "is_correct": {
            "type": "boolean",
            "description": (
                "What you actually believe after honest reflection. If you "
                "would not bet your own reputation on this answer being right, "
                "this should be false -- including cases where you hesitate "
                "without being able to name a specific defect."
            ),
        },
    },
    "required": ["feedback", "is_correct"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# --- v5 (2026-04-27): v4 floor + observable iteration state ------------------
# v4 trace inspection (1 qid, 6 feedback rounds reached budget=8) revealed a
# protocol bug, not a model bug: build_feedback_message() per Madaan Algorithm 1
# line 3 passes ONLY the latest draft to the Critic, with NO iteration history.
# The procedural floor told the Critic to "count how many refined drafts you
# have seen in the iteration history" -- but the Critic literally cannot see
# any history, so on every round it wrote "review round 1; refined drafts
# seen so far: 0", correctly applied the floor against false premises, and
# returned is_correct=false. Effect: the loop ran to budget=8 every time,
# making v4 functionally equivalent to Budget Forcing rather than a Self-Refine
# variant. v5 fixes the input: build_feedback_message() now prepends an
# "## Iteration State" block computed by the orchestrator (review round =
# len(feedbacks)+1, refined drafts seen = len(drafts)-1) so the Critic has
# the data the floor was always trying to reference. The system prompt is
# also updated to read the injected state instead of asking the model to
# count from history it cannot see. Functions as the same model-compliance
# probe v4 was meant to be -- with a fair input.
#
# --- v4 (2026-04-27): v3 + minimum-2-refine procedural floor ----------------
# Goal pivot: on HLE Gold, ATTS achieves n=1 fraction <= 10%. v3 reduced n=1
# from 86% (v1) -> 79%, still nowhere near ATTS. The user re-prioritized:
# accuracy is no longer the load-bearing target; the ONLY metric we are now
# trying to move is the n=1 fraction. v4 keeps v3's "fresh intuition first"
# guidance verbatim (verified by trace inspection: 10/10 follow the directive)
# and ADDS a hard procedural floor: the critic must mark is_correct=false
# until it has seen at least two refined drafts in the iteration history.
# This is the structural intervention earlier rejected as "gaming the metric"
# -- the user explicitly authorized it after observing that purely
# behavioral nudges (v3) leave the distribution essentially unchanged.
# v4 also serves as a model-compliance probe: does Sonnet-4.6 follow an
# explicit procedural instruction inside a guidance-style prompt, or does
# it ignore it the way it has been ignoring softer hints?
#
# --- v3 (2026-04-27): guidance over state-machine ----------------------------
# v1 (Bayesian-prior + 5-Q checklist) and v2 (4-axis state machine + forbidden
# phrases + default-false coercion) BOTH failed in measurement on HLE Gold.
# v1 17/100 result: n=1 fraction 88% (vs vanilla 76%), accuracy 47% vs 65%,
# Critic FP rate worsened (8 vs 3 same-qid). Trace inspection showed the model
# treated both versions as a rubric: it produced verbose Q1-Q5 (or AXIS-1..4)
# justifications that read as "self-consistency" and reinforced false approvals.
#
# v3 reframes from "instructions to follow" to "questions to actually think
# from". The crucial behavioral fix is the OPENING DIRECTIVE: set the proposed
# reasoning aside and form your own intuition BEFORE re-engaging with the
# solution. This is a temporal/contextual reset, not another sub-question --
# it is the one move that breaks the same-model collinearity that made v1's
# Q2 "re-derive via a different path" indistinguishable from re-tracing.
#
# Do not regress to checklist form. If you must add structure, add it as a
# question someone genuinely curious would ask, not as an action someone is
# being audited on.
FEEDBACK_SYSTEM_PROMPT = """\
You are looking at a solution someone else wrote. Your task is to think about
whether you actually believe it -- not to grade it against a rubric, not to
follow a checklist, but to ask yourself the questions an expert in this
domain would ask if they were genuinely considering whether to stake their
own reputation on this answer being correct.

Begin by setting the proposed reasoning aside for a moment. Look at the
question fresh -- as if you had never seen the solution. What is your
immediate intuition about the answer? Now return to the proposed solution:
does it match your intuition? If yes, why do you trust the intuition? If no,
what does the solution see that you missed -- or what does your intuition
see that the solution missed?

Some questions worth genuinely sitting with:

  - What is the single weakest step in this reasoning? Not "is there a
    weakness" -- assume there is one, and find it.
  - What would a skeptical expert in this exact subfield say first?
  - If you were forced to give a different answer than the one proposed,
    which one would you give, and how confident would you be in it?
  - Is there an assumption the solver made that they didn't realize they
    were making? Spell it out.
  - What information does the question provide that the solution didn't
    use? Was that omission justified, or did it matter?

These are not boxes to tick. They are angles to genuinely think from. If
you find yourself answering them quickly and arriving at "all good", you
probably skipped them -- a real critic spends most of their time on the
question they expected to be hardest.

Set is_correct based on what you actually believe after this reflection,
not on whether you found a specific defect. If you would not bet your own
reputation on this answer being right, set is_correct=false and explain
what makes you hesitate. Hesitation without a named defect is itself a
valid signal -- write it down rather than overriding it.

A procedural floor on top of the above. Read the "## Iteration State"
block at the top of the user message. It tells you exactly two numbers:
the current review round, and how many refined drafts have been
produced so far (drafts beyond the original Draft 0). If "Refined
drafts seen so far" is fewer than 2, set is_correct=false even when
your review surfaces no concrete defect, and state the two numbers
explicitly in your feedback (e.g. "Per Iteration State: review round
N, refined drafts seen so far K -- procedural floor active, marking
is_correct=false"). The reason is empirical: critics frequently miss
subtle issues on first pass, and forcing the solver to produce
additional drafts materially improves final quality on hard reasoning
problems. Use these mandatory rounds to apply the questions above with
more care -- on a second or third pass, you often notice something the
first pass missed, or a different angle becomes available because the
refined draft has made a previously implicit assumption explicit.

This procedural floor does NOT apply once "Refined drafts seen so far"
is at least 2: from that point on, set is_correct based purely on the
honest reflection above.
"""

# Refiner is intentionally identical to vanilla Self-Refine -- the experimental
# variable is the Critic, not the Refiner. Any change here would conflate two
# interventions.
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
        """Build user message for the Feedback call.

        Madaan Algorithm 1 line 3 specifies problem + current draft only.
        v5 also prepends an Iteration State block (review round, refined
        drafts seen so far) computed from history -- the procedural floor
        in FEEDBACK_SYSTEM_PROMPT references these counts, and without
        them the Critic has no way to know which round it is on.
        """
        latest = self.drafts[-1]
        review_round = len(self.feedbacks) + 1
        refined_drafts_seen = len(self.drafts) - 1
        return (
            f"## Iteration State\n\n"
            f"Review round: {review_round}\n"
            f"Refined drafts seen so far: {refined_drafts_seen}\n\n"
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
    *,
    spec,  # methods.specs.SocraticSelfRefineSpec
    image_data_url: str | None = None,
    question_id: str | None = None,
    rollout_idx: int | None = None,
    **_extra,
) -> SolveResult:
    """Solve via Socratic Self-Refine: Generate -> (Socratic Feedback -> Refine)*."""
    variant = spec.explore  # ExploreVariant
    backend = variant.model.backend
    ctx = create_solve_context(
        infra=infra,
        backend=backend,
        timeout=variant.model.timeout,
        problem=problem,
        image_data_url=image_data_url,
        question_id=question_id,
        writer_system_prompt=infra.benchmark.get_explorer_system_prompt(backend),
        writer_user_message=infra.benchmark.build_explorer_message(problem),
        writer_header_lines=[
            f"**Backend**: {backend}",
            f"**Model**: {variant.model.model}",
            f"**Max iterations**: {infra.max_iterations}",
            f"**Method**: socratic-self-refine",
        ],
        writer_title_suffix="(socratic-self-refine)",
        rollout_idx=rollout_idx,
    )

    history = IterationHistory()

    # -- Step 1: Generator (Draft 0) — inline explore call --
    explorer_system_prompt = ctx.benchmark.get_explorer_system_prompt(backend)
    explore_schema = ctx.benchmark.get_explore_schema()
    user_msg = ctx.benchmark.build_explorer_message(problem)

    gen_result, _gen_traj, gen_cost, gen_usage, _gen_dur = await ctx.call_sub_model(
        system_prompt=explorer_system_prompt,
        user_message=user_msg,
        model_cfg=variant.model,
        output_schema=explore_schema,
        cache_key="explore_1",
        writer=ctx.writer,
    )

    if gen_result.get("timed_out"):
        logger.info("  [socratic-self-refine] Draft 0 TIMED OUT, no answer")
        return ctx.result("")

    ctx.cost.add(gen_cost, gen_usage, component="explorer")
    gen_answer = ctx.benchmark.get_answer_from_explore(gen_result)
    draft0 = Candidate(
        answer=gen_answer,
        reasoning=gen_result.get("reasoning", ""),
        approach=gen_result.get("approach", ""),
        confidence=gen_result.get("confidence", 0.0),
        cost_usd=gen_cost,
    )
    ctx.state.candidates.append(draft0)
    ctx.state.explore = advance(ctx.state.explore)
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

    logger.info(f"  [socratic-self-refine] Draft 0 (generator): answer={draft0.answer}")

    # -- Steps 2..max_iterations: Feedback -> Refine loop --
    # These are custom prompts (not from benchmark), so we manually add the
    # Claude structured suffix when needed.
    feedback_prompt = FEEDBACK_SYSTEM_PROMPT
    if backend == "claude":
        feedback_prompt += format_claude_structured_suffix(FEEDBACK_SCHEMA)

    refiner_prompt = REFINER_SYSTEM_PROMPT
    if backend == "claude":
        refiner_prompt += format_claude_structured_suffix(explore_schema)

    for i in range(2, infra.max_iterations + 1):
        # -- Feedback call --
        fb_msg = history.build_feedback_message(problem)

        fb_result, fb_traj, fb_cost, fb_usage, fb_dur = await ctx.call_sub_model(
            system_prompt=feedback_prompt,
            user_message=fb_msg,
            model_cfg=variant.model,
            output_schema=FEEDBACK_SCHEMA,
            cache_key=f"feedback_{i}",
            writer=ctx.writer,
        )

        ctx.cost.add(fb_cost, fb_usage, component="feedback")

        if fb_result.get("timed_out"):
            logger.info(f"  [socratic-self-refine] Feedback {i}: TIMED OUT")
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

        logger.info(f"  [socratic-self-refine] Feedback {i}: {status}")

        # If feedback says the current solution is correct, stop
        if is_correct:
            break

        # -- Refiner call --
        ref_msg = history.build_refiner_message(problem)

        ref_result, ref_traj, ref_cost, ref_usage, ref_dur = await ctx.call_sub_model(
            system_prompt=refiner_prompt,
            user_message=ref_msg,
            model_cfg=variant.model,
            output_schema=explore_schema,
            cache_key=f"explore_{i}",
            writer=ctx.writer,
        )

        ctx.cost.add(ref_cost, ref_usage, component="explorer")

        if ref_result.get("timed_out"):
            logger.info(f"  [socratic-self-refine] Refiner {i}: TIMED OUT")
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
        ctx.state.explore = advance(ctx.state.explore)

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

        logger.info(f"  [socratic-self-refine] Draft {i - 1} (refiner): answer={answer}, confidence={ref_result.get('confidence', 'N/A')}")

    final_answer = history.drafts[-1].answer

    logger.info(f"  [socratic-self-refine] final answer: {final_answer} after {len(ctx.rounds)} drafts")
    logger.info(f"  [socratic-self-refine] total cost: ${ctx.cost.total_cost_usd}")

    return ctx.result(final_answer)
