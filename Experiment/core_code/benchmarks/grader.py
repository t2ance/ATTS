"""Answer normalization, matching, and LLM-based judging for benchmarks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from methods.base import call_sub_model, save_sub_model_result
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
    m = re.search(r"(?:answer\s*(?:is|are|=|:)\s*\(?([a-e])\)?)", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:option|choice)\s*\(?([a-e])\)?", text)
    if m:
        return m.group(1)
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
Judge whether the following [response] to [question] is correct based on the [correct_answer].

Your judgement must follow these criteria:
- extracted_final_answer: The final exact answer extracted from the [response]. Put "None" if there is no exact final answer to extract.
- reasoning: Explain why the extracted answer is correct or incorrect based on [correct_answer]. Focus only on whether there are meaningful differences. Do not comment on background, do not attempt to solve the problem, do not argue for any answer different than [correct_answer].
- correct: true if extracted_final_answer matches [correct_answer], or is within a small margin of error for numerical problems. Consider mathematical equivalence (1/2 = 0.5), notation differences (LaTeX vs Unicode), and formatting differences that don't change meaning. false otherwise."""

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
    predicted: str, gold: str, question: str, model: str,
    backend: str = "codex",
    out_dir: Path | None = None,
) -> tuple[bool, float]:
    """Use an LLM to judge if predicted answer matches gold. Returns (correct, cost_usd)."""
    judge_prompt = _get_judge_system_prompt(backend)
    user_message = (
        f"[question]: {question}\n"
        f"[response]: {predicted}\n"
        f"[correct_answer]: {gold}"
    )
    writer = TrajectoryWriter.create_simple(out_dir / "output.md") if out_dir else TrajectoryWriter.noop()
    result, trajectory_text, cost_usd, usage = await call_sub_model(
        backend=backend,
        system_prompt=judge_prompt,
        user_message=user_message,
        image_data_url=None,
        model=model,
        output_schema=JUDGE_SCHEMA,
        writer=writer,
    )
    if out_dir is not None:
        save_sub_model_result(
            out_dir=out_dir,
            result=result,
            trajectory_text=trajectory_text,
            cost_usd=cost_usd,
            usage=usage,
            duration_seconds=0.0,
            model=model,
        )
    if result.get("timed_out"):
        print(f"  [judge] timed out -- treating as incorrect")
        return False, 0.0
    return result["correct"], cost_usd


async def grade_answer(
    predicted: str, gold: str, question: str, answer_type: str,
    judge_model: str | None = None,
    backend: str = "codex",
    out_dir: Path | None = None,
) -> tuple[bool, float]:
    """Grade an answer. Returns (correct, judge_cost_usd)."""
    if answer_type == "multipleChoice":
        return check_answer(predicted, gold, answer_type), 0.0
    if judge_model is None:
        return check_answer(predicted, gold, answer_type), 0.0
    return await judge_answer(predicted, gold, question, judge_model, backend=backend, out_dir=out_dir)


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
