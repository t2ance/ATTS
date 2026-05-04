"""Small-scope unit tests for grader.py's new judge_spec-based API (T5).

These tests exercise grade_answer / judge_answer at the module level, mocking
call_sub_model so no real network calls happen.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from benchmarks.grader import grade_answer, judge_answer


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---- grade_answer: judge_spec=None bypasses the LLM judge ----


def test_grade_answer_judge_spec_none_uses_string_match():
    is_correct, cost = _run(grade_answer(
        "42", "42", "q", "exactMatch", judge_spec=None,
    ))
    assert is_correct is True
    assert cost == 0.0


def test_grade_answer_multiplechoice_short_circuits_even_with_judge_spec():
    """Multiple-choice short-circuits to string match regardless of judge_spec."""
    is_correct, cost = _run(grade_answer(
        "C", "C", "q", "multipleChoice",
        judge_spec={"backend": "claude", "model": "claude-haiku-4-5-20251001"},
    ))
    assert is_correct is True
    assert cost == 0.0


# ---- grade_answer: dispatches to judge_answer with judge_spec dict ----


def test_grade_answer_with_judge_spec_invokes_judge_answer():
    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    with patch("benchmarks.grader.judge_answer", new=AsyncMock(return_value=(True, 0.012))) as m:
        is_correct, cost = _run(grade_answer(
            "x", "y", "q", "exactMatch", judge_spec=spec,
        ))
    assert is_correct is True
    assert cost == 0.012
    args, kwargs = m.call_args
    # New signature: (predicted, gold, question, judge_spec, out_dir=...)
    assert args[0] == "x"
    assert args[1] == "y"
    assert args[2] == "q"
    assert args[3] == spec  # the dict, not just model string


# ---- judge_answer: writes config.json into out_dir ----


def test_judge_answer_writes_config_json(tmp_path):
    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    fake_result = {"correct": True, "extracted_final_answer": "x", "reasoning": ""}
    fake_call_sub_model = AsyncMock(return_value=(fake_result, "trajectory text", 0.001, {}))
    with patch("benchmarks.grader.call_sub_model", new=fake_call_sub_model):
        is_correct, cost = _run(judge_answer(
            "x", "y", "q", spec, max_retries=3, out_dir=tmp_path,
        ))
    assert is_correct is True
    config_path = tmp_path / "config.json"
    assert config_path.exists(), "judge_answer must write config.json into out_dir"
    stored = json.loads(config_path.read_text(encoding="utf-8"))
    assert stored == spec


def test_judge_answer_routes_backend_from_spec_name(tmp_path):
    """name field in judge_spec selects the backend; model is the model arg."""
    spec = {"backend": "vllm", "model": "qwen36-35b-a3b-fp8",
            "vllm_sampling": {"temperature": 0.6}}
    fake_result = {"correct": False, "extracted_final_answer": "", "reasoning": ""}
    fake_call_sub_model = AsyncMock(return_value=(fake_result, "tt", 0.0, {}))
    with patch("benchmarks.grader.call_sub_model", new=fake_call_sub_model):
        _run(judge_answer("x", "y", "q", spec, max_retries=3, out_dir=tmp_path))
    _, call_kwargs = fake_call_sub_model.call_args
    assert call_kwargs["backend"] == "vllm"
    assert call_kwargs["model"] == "qwen36-35b-a3b-fp8"
    assert call_kwargs["sampling"] == {"temperature": 0.6}
