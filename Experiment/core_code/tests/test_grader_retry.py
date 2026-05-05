"""Unit tests for judge_answer retry behavior.

Covers: (a) success on first attempt, (b) success after one retry,
(c) raise after all retries exhaust, (d) max_retries kwarg is required.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from benchmarks.grader import judge_answer


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


JUDGE_SPEC = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}


def test_judge_answer_succeeds_first_attempt():
    """Single successful call -> returns immediately, no retries."""
    successful = (
        {"correct": True, "extracted_final_answer": "42", "reasoning": "match"},
        "trajectory text",
        0.001,
        {"input_tokens": 100, "output_tokens": 50},
    )
    with patch("benchmarks.grader.call_sub_model", new=AsyncMock(return_value=successful)) as m:
        outcome = _run(judge_answer(
            "42", "42", "what is the answer?", JUDGE_SPEC,
            max_retries=3,
        ))
    assert outcome.is_correct is True
    assert outcome.cost_usd == pytest.approx(0.001)
    assert m.await_count == 1


def test_judge_answer_succeeds_after_one_retry():
    """First attempt parse_failed, second succeeds -> success returned, cost summed."""
    failure = (
        {"timed_out": True, "parse_failed": True, "finish_reason": "length"},
        "truncated text",
        0.0008,
        {"input_tokens": 100, "output_tokens": 4096},
    )
    success = (
        {"correct": False, "extracted_final_answer": "x", "reasoning": "mismatch"},
        "good text",
        0.0005,
        {"input_tokens": 100, "output_tokens": 50},
    )
    with patch("benchmarks.grader.call_sub_model", new=AsyncMock(side_effect=[failure, success])) as m:
        outcome = _run(judge_answer(
            "x", "y", "q?", JUDGE_SPEC,
            max_retries=3,
        ))
    assert outcome.is_correct is False
    assert outcome.cost_usd == pytest.approx(0.0008 + 0.0005)
    assert m.await_count == 2


def test_judge_answer_raises_after_all_retries():
    """All N attempts fail -> RuntimeError; no silent False."""
    failure = (
        {"timed_out": True, "parse_failed": True, "finish_reason": "length"},
        "always bad",
        0.0001,
        {"input_tokens": 100, "output_tokens": 4096},
    )
    with patch("benchmarks.grader.call_sub_model", new=AsyncMock(return_value=failure)) as m:
        with pytest.raises(RuntimeError, match="2 attempts all returned invalid"):
            _run(judge_answer("x", "y", "q?", JUDGE_SPEC, max_retries=2))
    assert m.await_count == 2


def test_judge_answer_max_retries_is_required():
    """max_retries has no default — caller must pass it explicitly."""
    with pytest.raises(TypeError, match="max_retries"):
        _run(judge_answer("x", "y", "q?", JUDGE_SPEC))
