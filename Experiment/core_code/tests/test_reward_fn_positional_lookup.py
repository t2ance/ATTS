from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from training.grpo import reward_fn


def _candidate_block(idx: int, used: int, max_explores: int, answer: str) -> str:
    return (
        f"<tool_response>\nCandidate #{idx} recorded.\n"
        f"- Answer: {answer}\n"
        f"- Confidence: 0.5\n"
        f"- Approach: approach\n"
        f"- Reasoning: reasoning\n"
        f"- Cost: $0.02\n\n"
        f"Explore budget: {used}/{max_explores} used, {max_explores - used} remaining."
        f"</tool_response>"
    )


def _explore_call(query: str = "test") -> str:
    return f'<tool_call>{{"name": "explore", "arguments": {{"query": "{query}"}}}}</tool_call>'


def _struct_output(answer: str) -> str:
    return (
        '<tool_call>{"name": "StructuredOutput", "arguments": {'
        f'"answer": "{answer}", "approach": "a", "reasoning": "r", "confidence": 0.8'
        "}}</tool_call>"
    )


def test_positional_lookup_hits_correct():
    cached = [
        {"cache_id": "x0", "answer": "A", "approach": "a", "reasoning": "r", "is_correct": False},
        {"cache_id": "x1", "answer": "B", "approach": "b", "reasoning": "r", "is_correct": True},
        {"cache_id": "x2", "answer": "C", "approach": "c", "reasoning": "r", "is_correct": False},
        {"cache_id": "x3", "answer": "D", "approach": "d", "reasoning": "r", "is_correct": False},
        {"cache_id": "x4", "answer": "E", "approach": "e", "reasoning": "r", "is_correct": False},
        {"cache_id": "x5", "answer": "F", "approach": "f", "reasoning": "r", "is_correct": False},
        {"cache_id": "x6", "answer": "G", "approach": "g", "reasoning": "r", "is_correct": False},
        {"cache_id": "x7", "answer": "H", "approach": "h", "reasoning": "r", "is_correct": False},
    ]
    solution_str = (
        _explore_call("q1") + _candidate_block(1, 1, 8, "A")
        + _explore_call("q2") + _candidate_block(2, 2, 8, "B")
        + _struct_output("B")
    )
    extra_info = {
        "question": "dummy",
        "tools_kwargs": {"explore": {"create_kwargs": {"cached_explores": cached, "max_explores": 8}}},
    }
    with patch.object(reward_fn, "_judge_remote", return_value=1.0):
        out = reward_fn.compute_score("atts_hle", solution_str, "B", extra_info)
    assert out["discovery"] == 1.0, out
    assert out["acc"] == 1.0, out
    assert out["num_explores"] == 2.0, out


def test_out_of_range_idx_raises():
    cached = [{"cache_id": "x", "answer": "A", "approach": "a", "reasoning": "r", "is_correct": False}]
    solution_str = _candidate_block(5, 1, 1, "A") + _struct_output("A")
    extra_info = {"tools_kwargs": {"explore": {"create_kwargs": {"cached_explores": cached, "max_explores": 1}}}}
    with patch.object(reward_fn, "_judge_remote", return_value=0.0):
        try:
            reward_fn.compute_score("atts_hle", solution_str, "A", extra_info)
        except ValueError as e:
            assert "out of range" in str(e)
            return
    raise AssertionError("expected ValueError on out-of-range idx")
