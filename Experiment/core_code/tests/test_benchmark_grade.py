from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from benchmarks.gpqa import GPQABenchmark
from benchmarks.hle import HLEBenchmark
from benchmarks.babyvision import BabyVisionBenchmark
from benchmarks.rbenchv import RBenchVBenchmark
from benchmarks.lcb import LCBBenchmark
from benchmarks.aime import AIMEBenchmark


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---- GPQA: pure string match on multipleChoice letter ----

def test_gpqa_grade_multiplechoice_correct():
    bench = GPQABenchmark()
    is_correct, cost = _run(bench.grade("c", "c", "q", {}, backend="claude"))
    assert is_correct is True
    assert cost == 0.0


def test_gpqa_grade_multiplechoice_wrong():
    bench = GPQABenchmark()
    is_correct, cost = _run(bench.grade("a", "c", "q", {}, backend="claude"))
    assert is_correct is False
    assert cost == 0.0


# ---- BabyVision: hybrid (choice -> string match, blank -> LLM judge) ----

def test_babyvision_choice_row_uses_string_match():
    bench = BabyVisionBenchmark()
    row = {"ansType": "choice"}
    is_correct, cost = _run(bench.grade("b", "b", "q", row, backend="claude"))
    assert is_correct is True
    assert cost == 0.0


def test_babyvision_blank_row_uses_judge_answer():
    bench = BabyVisionBenchmark()
    row = {"ansType": "blank"}
    with patch("benchmarks.babyvision.judge_answer", new=AsyncMock(return_value=(True, 0.001))) as m:
        is_correct, cost = _run(bench.grade("foo", "bar", "q", row, backend="claude"))
    assert is_correct is True
    assert cost == 0.001
    args, kwargs = m.call_args
    assert args[0] == "foo"
    assert args[1] == "bar"
    assert args[3] == "claude-haiku-4-5-20251001"
    assert kwargs["backend"] == "claude"


# ---- HLE: judge with codex bridge inline ----

def test_hle_grade_claude_backend_uses_haiku():
    bench = HLEBenchmark()
    row = {"answer_type": "exactMatch"}
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=(True, 0.002))) as m:
        is_correct, cost = _run(bench.grade("x", "y", "q", row, backend="claude"))
    args, _ = m.call_args
    assert args[3] == "claude-haiku-4-5-20251001"
    assert is_correct is True


def test_hle_grade_codex_backend_remaps_to_gpt5_codex_mini():
    bench = HLEBenchmark()
    row = {"answer_type": "exactMatch"}
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=(False, 0.003))) as m:
        is_correct, cost = _run(bench.grade("x", "y", "q", row, backend="codex"))
    args, kwargs = m.call_args
    assert args[3] == "gpt-5-codex-mini"
    assert kwargs["backend"] == "codex"


def test_hle_grade_vllm_backend_routes_to_claude():
    bench = HLEBenchmark()
    row = {"answer_type": "exactMatch"}
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=(True, 0.0))) as m:
        _run(bench.grade("x", "y", "q", row, backend="vllm"))
    args, kwargs = m.call_args
    assert kwargs["backend"] == "claude"
    assert args[3] == "claude-haiku-4-5-20251001"


def test_hle_grade_multiplechoice_row_uses_string_match():
    bench = HLEBenchmark()
    row = {"answer_type": "multipleChoice"}
    is_correct, cost = _run(bench.grade("c", "c", "q", row, backend="claude"))
    assert is_correct is True
    assert cost == 0.0


# ---- RBenchV: always LLM judge ----

def test_rbenchv_uses_judge_answer():
    bench = RBenchVBenchmark()
    with patch("benchmarks.rbenchv.judge_answer", new=AsyncMock(return_value=(True, 0.004))) as m:
        is_correct, cost = _run(bench.grade("p", "g", "q", {}, backend="claude"))
    args, kwargs = m.call_args
    assert args[3] == "claude-haiku-4-5-20251001"
    assert kwargs["backend"] == "claude"
    assert is_correct is True


# ---- grading_summary class attr present on all 6 benchmarks ----

def test_all_benchmarks_have_grading_summary():
    for cls in (LCBBenchmark, AIMEBenchmark, GPQABenchmark, HLEBenchmark,
                BabyVisionBenchmark, RBenchVBenchmark):
        assert hasattr(cls, "grading_summary"), f"{cls.__name__} missing grading_summary"
        assert isinstance(cls.grading_summary, str)
        assert len(cls.grading_summary) > 10


# ---- judge_model class attr preserved (cache invalidation) ----

def test_judge_model_preserved_per_benchmark():
    assert LCBBenchmark.judge_model is None
    assert AIMEBenchmark.judge_model is None
    assert GPQABenchmark.judge_model is None
    assert HLEBenchmark.judge_model == "claude-haiku-4-5-20251001"
    assert BabyVisionBenchmark.judge_model == "claude-haiku-4-5-20251001"
    assert RBenchVBenchmark.judge_model == "claude-haiku-4-5-20251001"
