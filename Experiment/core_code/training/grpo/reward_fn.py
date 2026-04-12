"""Trajectory-level reward function for ATTS GRPO.

GRPO reward: R = correctness(0.0 or 1.0) - 0.05 * num_explore_calls

Returns a dict so verl logs extra metrics (raw accuracy, explore count,
answer emission rate) separately from the shaped GRPO reward. verl 0.7.1
special-cases the key "acc" as the core validation variable (see
verl/trainer/ppo/ray_trainer.py:631), so it is elevated to val-core/<data_source>/acc/
in W&B, while other keys are logged under val-aux/.

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
    """Count explore tool calls in the response.

    Only counts complete <tool_call>...</tool_call> blocks containing explore,
    not substrings in tool response text or hallucinated repetitions.
    Capped at MAX_EXPLORE_CALLS to bound reward even if model hallucinates
    many tool_call blocks in a single turn.
    """
    MAX_EXPLORE_CALLS = 9  # max_assistant_turns - 1 (last turn is StructuredOutput)
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', solution_str, re.DOTALL)
    count = sum(1 for b in blocks if '"name"' in b and '"explore"' in b)
    return min(count, MAX_EXPLORE_CALLS)


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
) -> dict:
    """Compute trajectory-level reward and auxiliary metrics for ATTS GRPO.

    Returns a dict:
        score:        the GRPO reward = correctness - 0.05 * num_explores  (used for gradient)
        acc:          raw correctness 0.0 or 1.0                          (val-core metric)
        num_explores: explore tool call count                              (val-aux metric)
        has_answer:   1.0 if the model emitted StructuredOutput, else 0.0  (val-aux metric)
        penalty:      EXPLORE_PENALTY * num_explores                       (val-aux metric)
    """
    predicted = _extract_answer_from_response(solution_str)
    correctness = _grade(predicted, ground_truth)
    num_explores = _count_explore_calls(solution_str)
    penalty = EXPLORE_PENALTY * num_explores
    reward = correctness - penalty
    return {
        "score": reward,
        "acc": correctness,
        "num_explores": float(num_explores),
        "has_answer": 1.0 if predicted else 0.0,
        "penalty": penalty,
    }
