"""Trajectory-level reward function for ATTS GRPO.

R = correctness(0.0 or 1.0) - 0.05 * num_explore_calls

Called by verl via reward.custom_reward_function config.
"""

from __future__ import annotations

import re


def _normalize_answer(text: str) -> str:
    """Lowercase, strip whitespace and punctuation for fuzzy matching."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _extract_answer_from_response(solution_str: str) -> str:
    """Extract the answer field from the StructuredOutput tool call in the response."""
    # Look for the last StructuredOutput tool call
    pattern = r'"name"\s*:\s*"StructuredOutput".*?"answer"\s*:\s*"([^"]*)"'
    matches = re.findall(pattern, solution_str, re.DOTALL)
    if matches:
        return matches[-1]
    return ""


def _count_explore_calls(solution_str: str) -> int:
    """Count explore tool calls in the response."""
    return len(re.findall(r'"name"\s*:\s*"explore"', solution_str))


def _grade(predicted: str, ground_truth: str) -> float:
    """Grade answer correctness. Returns 1.0 or 0.0."""
    if _normalize_answer(predicted) == _normalize_answer(ground_truth):
        return 1.0
    # Single-letter answer matching (for GPQA)
    pred_letter = re.search(r"\b([A-Ea-e])\b", predicted)
    gt_letter = re.search(r"\b([A-Ea-e])\b", ground_truth)
    if pred_letter and gt_letter:
        if pred_letter.group(1).upper() == gt_letter.group(1).upper():
            return 1.0
    return 0.0


EXPLORE_PENALTY = 0.05


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Compute trajectory-level reward for ATTS GRPO.

    R = correctness - 0.05 * num_explores
    """
    predicted = _extract_answer_from_response(solution_str)
    correctness = _grade(predicted, ground_truth)
    num_explores = _count_explore_calls(solution_str)
    return correctness - EXPLORE_PENALTY * num_explores
