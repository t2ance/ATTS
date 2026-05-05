"""Answer normalization, matching, and LLM-based judging for benchmarks."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from cache_types import JudgeOutcome
from methods.base import call_sub_model
from prompts import format_claude_structured_suffix
from trajectory import TrajectoryWriter


def normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison."""
    s = str(text).strip()
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"^\\boxed\{(.+)\}$", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def _extract_mc_letter(text: str) -> str | None:
    """Extract a multiple-choice letter from text."""
    # Priority 1: answer starts with option letter (e.g., "d: formula", "(a) value")
    m = re.match(r"\(?([a-e])[):\s.,]", text)
    if m:
        return m.group(1)
    # Priority 2: explicit "answer is X" pattern
    m = re.search(r"(?:answer\s*(?:is|are|=|:)\s*\(?([a-e])\)?)", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:option|choice)\s*\(?([a-e])\)?", text)
    if m:
        return m.group(1)
    # Priority 3: last standalone letter a-e (handles "analyzing a through d, answer is c")
    matches = re.findall(r"\b([a-e])\b", text)
    if matches:
        return matches[-1]
    return None


def check_answer(predicted: str, gold: str, answer_type: str) -> bool:
    """Check if the predicted answer matches the gold answer (string-based)."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if answer_type == "multipleChoice":
        pred_letter = _extract_mc_letter(pred_norm)
        gold_letter = _extract_mc_letter(gold_norm)
        if pred_letter and gold_letter:
            return pred_letter == gold_letter
        return False

    return pred_norm == gold_norm


_JUDGE_BASE = """\
You are an answer-comparator, NOT a problem-solver. The [correct_answer] is GIVEN to you as ground truth. Your only job is to compare [response] against [correct_answer].

ABSOLUTE RULES (violating any of these is failure):
1. DO NOT solve the problem yourself. Do NOT re-derive, recompute, or reason about the problem domain.
2. DO NOT enumerate cases, run through subproblems, or expand the [correct_answer]'s logic. The [correct_answer] is correct by definition; treat it as a fixed string/number/expression to compare against.
3. Your reasoning must be a direct comparison: take [extracted_final_answer], take [correct_answer], state whether they are equivalent (allowing notation / formatting / mathematical-equivalence differences). Nothing else.
4. Reasoning MUST be at most 2 short sentences. If you find yourself writing more than 2 sentences, you are violating rule 1.
5. If [response] has no extractable final answer, set extracted_final_answer="None" and correct=false.
6. You MUST mark an extracted_final_answer of None, and mark it as incorrect, if an empty answer is provided. DO NOT SOLVE THE PROBLEM IN THIS CASE.

GOOD reasoning examples (do this):
- "Extracted '1/2' equals correct_answer '0.5' (mathematical equivalence). Match."
- "Extracted 'A' does not equal correct_answer 'C'. No match."
- "Extracted '5' does not equal correct_answer '4'. No match."

BAD reasoning examples (NEVER do this):
- "Let's verify: torus is homogeneous so 1 class, sphere is homogeneous so 1 class..." (re-deriving)
- "Actually let me reconsider whether 1+1=2 in this ring..." (recomputing)
- Any chain-of-thought that does NOT explicitly cite [correct_answer]'s value in the first sentence."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "extracted_final_answer": {
            "type": "string",
            "description": "The final exact answer extracted from the response. 'None' if no exact answer found.",
        },
        "reasoning": {
            "type": "string",
            "description": "Why the extracted answer matches or does not match the correct answer.",
        },
        "correct": {
            "type": "boolean",
            "description": "Whether the extracted answer matches the correct answer.",
        },
    },
    "required": ["extracted_final_answer", "reasoning", "correct"],
    "additionalProperties": False,
}

def _get_judge_system_prompt(backend: str) -> str:
    if backend == "claude":
        return _JUDGE_BASE + format_claude_structured_suffix(JUDGE_SCHEMA)
    return _JUDGE_BASE


async def judge_answer(
    predicted: str, gold: str, question: str, judge_spec: dict,
    *,
    max_retries: int,
) -> JudgeOutcome:
    """Use an LLM to judge if predicted answer matches gold. Returns JudgeOutcome.

    Pure function: no disk I/O. Caller (ExploreVariant or eval.py) is responsible
    for persisting the bundle via outcome.persist(target_dir).

    Retries up to `max_retries` times if the judge returns an invalid verdict
    (timed_out or parse_failed). All attempts' costs are summed into the
    returned outcome. After all retries exhaust, raises RuntimeError -- never
    silently judges as incorrect.

    judge_spec carries `backend` (claude/codex/vllm), `model`, and optional
    knobs (vllm_sampling, effort, budget_tokens).
    """
    backend = judge_spec["backend"]
    model = judge_spec["model"]
    sampling = judge_spec.get("vllm_sampling")
    # Optional thinking-budget knobs (claude backend only). Fall back to
    # call_sub_model's defaults (budget_tokens=32000, effort=None) when the
    # judge_spec doesn't set them, preserving the original cache build-time
    # behavior for any bundle that lacks these keys.
    effort = judge_spec.get("effort")
    budget_tokens = judge_spec.get("budget_tokens")
    judge_prompt = _get_judge_system_prompt(backend)
    user_message = (
        f"[question]: {question}\n"
        f"[response]: {predicted}\n"
        f"[correct_answer]: {gold}"
    )
    total_cost = 0.0
    last_result: dict = {}
    last_trajectory = ""
    last_usage: dict = {}
    sub_model_kwargs: dict = {"sampling": sampling}
    if effort is not None:
        sub_model_kwargs["effort"] = effort
    if budget_tokens is not None:
        sub_model_kwargs["budget_tokens"] = budget_tokens
    for attempt in range(1, max_retries + 1):
        result, trajectory_text, cost_usd, usage = await call_sub_model(
            backend=backend,
            system_prompt=judge_prompt,
            user_message=user_message,
            image_data_url=None,
            model=model,
            output_schema=JUDGE_SCHEMA,
            writer=TrajectoryWriter.noop(),
            **sub_model_kwargs,
        )
        total_cost += cost_usd
        last_result, last_trajectory, last_usage = result, trajectory_text, usage
        if not (result.get("timed_out") or result.get("parse_failed")):
            break
        logger.warning(
            f"  [judge] attempt {attempt}/{max_retries} invalid "
            f"(timed_out={result.get('timed_out')}, parse_failed={result.get('parse_failed')}, "
            f"finish_reason={result.get('finish_reason')})"
            + (" -- retrying" if attempt < max_retries else " -- giving up")
        )
    if last_result.get("timed_out") or last_result.get("parse_failed"):
        raise RuntimeError(
            f"judge_answer: {max_retries} attempts all returned invalid verdict "
            f"(timed_out={last_result.get('timed_out')}, parse_failed={last_result.get('parse_failed')}, "
            f"finish_reason={last_result.get('finish_reason')}). "
            f"Backend={backend} model={model}. Refuse to silently judge as incorrect -- "
            f"tighten judge prompt or sampling (e.g. shrink max_tokens, disable thinking) and retry."
        )
    return JudgeOutcome(
        is_correct=last_result["correct"],
        cost_usd=total_cost,
        judge_spec_snapshot=dict(judge_spec),
        input_md=f"## System Prompt\n\n{judge_prompt}\n\n## User Message\n\n{user_message}",
        output_md=last_trajectory,
        result_dict={**last_result, "usage": last_usage, "model": model},
    )


async def grade_answer(
    predicted: str, gold: str, question: str, answer_type: str,
    judge_spec: dict | None = None,
    *,
    max_retries: int = 3,
) -> JudgeOutcome:
    """Grade an answer. Returns JudgeOutcome.

    Routing:
      - multipleChoice answer_type   -> check_answer (string match), rule-based outcome
      - judge_spec is None           -> check_answer (no LLM judge available), rule-based outcome
      - otherwise                    -> judge_answer with judge_spec
    """
    if answer_type == "multipleChoice" or judge_spec is None:
        is_correct = check_answer(predicted, gold, answer_type)
        return JudgeOutcome(
            is_correct=is_correct,
            cost_usd=0.0,
            judge_spec_snapshot=None,
            input_md="",
            output_md="",
            result_dict={"correct": is_correct},
        )
    return await judge_answer(predicted, gold, question, judge_spec, max_retries=max_retries)


# ---------------------------------------------------------------------------
# LCB code execution grading (via official lcb_runner)
# ---------------------------------------------------------------------------

def normalize_code(text: str) -> str:
    """Normalize code for comparison -- strip whitespace."""
    return text.strip()


def _row_to_lcb_problem(row: dict):
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    return CodeGenerationProblem(
        question_id=row["question_id"],
        question_title=row.get("question_title", ""),
        question_content=row.get("question_content", row.get("question", "")),
        platform=row.get("platform", ""),
        contest_id=row.get("contest_id", ""),
        contest_date=row.get("contest_date", ""),
        difficulty=row.get("difficulty", ""),
        public_test_cases=row.get("public_test_cases", "[]"),
        private_test_cases=row.get("private_test_cases", "[]"),
        starter_code=row.get("starter_code", ""),
        metadata=row.get("metadata", "{}"),
    )


async def grade_code(
    predicted_code: str,
    row: dict,
    timeout: int = 30,
) -> tuple[bool, float]:
    """Grade a code solution using lcb_runner's check_correctness.

    Returns (passed_all_tests, cost_usd=0.0).
    """
    import os
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

    problem = _row_to_lcb_problem(row)
    sample = problem.get_evaluation_sample()

    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    try:
        results, _metadata = check_correctness(
            sample=sample,
            generation=predicted_code,
            timeout=timeout,
            debug=False,
        )
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull_fd)

    passed = all(r == True or r == 1 for r in results)
    return passed, 0.0
