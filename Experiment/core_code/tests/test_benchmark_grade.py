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


# Standard claude-haiku judge_spec for tests that need an LLM judge.
_HAIKU_SPEC = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
_CODEX_SPEC = {"name": "codex", "model": "gpt-5-codex-mini"}


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
    bench = BabyVisionBenchmark(judge_spec=_HAIKU_SPEC)
    row = {"ansType": "choice"}
    is_correct, cost = _run(bench.grade("b", "b", "q", row, backend="claude"))
    assert is_correct is True
    assert cost == 0.0


def test_babyvision_blank_row_uses_judge_answer():
    bench = BabyVisionBenchmark(judge_spec=_HAIKU_SPEC)
    row = {"ansType": "blank"}
    with patch("benchmarks.babyvision.judge_answer", new=AsyncMock(return_value=(True, 0.001))) as m:
        is_correct, cost = _run(bench.grade("foo", "bar", "q", row, backend="claude"))
    assert is_correct is True
    assert cost == 0.001
    args, _ = m.call_args
    assert args[0] == "foo"
    assert args[1] == "bar"
    assert args[3] == _HAIKU_SPEC  # full spec dict, not just model string


# ---- HLE: judge identity from YAML, no bespoke backend routing ----

def test_hle_grade_uses_yaml_supplied_judge():
    bench = HLEBenchmark(judge_spec=_HAIKU_SPEC)
    row = {"answer_type": "exactMatch"}
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=(True, 0.002))) as m:
        is_correct, _ = _run(bench.grade("x", "y", "q", row, backend="claude"))
    args, _ = m.call_args
    assert args[3] == _HAIKU_SPEC
    assert is_correct is True


def test_hle_grade_with_codex_judge_spec():
    """Old behavior 'codex backend remaps to gpt-5-codex-mini' is now the
    user's responsibility: they put name=codex+model=gpt-5-codex-mini in YAML."""
    bench = HLEBenchmark(judge_spec=_CODEX_SPEC)
    row = {"answer_type": "exactMatch"}
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=(False, 0.003))) as m:
        _run(bench.grade("x", "y", "q", row, backend="codex"))
    args, _ = m.call_args
    assert args[3] == _CODEX_SPEC


def test_hle_grade_orchestrator_backend_does_not_affect_judge():
    """orchestrator backend (here vllm) is independent of judge selection."""
    bench = HLEBenchmark(judge_spec=_HAIKU_SPEC)
    row = {"answer_type": "exactMatch"}
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=(True, 0.0))) as m:
        _run(bench.grade("x", "y", "q", row, backend="vllm"))
    args, _ = m.call_args
    assert args[3] == _HAIKU_SPEC


def test_hle_grade_multiplechoice_row_uses_string_match():
    bench = HLEBenchmark(judge_spec=_HAIKU_SPEC)
    row = {"answer_type": "multipleChoice"}
    is_correct, cost = _run(bench.grade("c", "c", "q", row, backend="claude"))
    assert is_correct is True
    assert cost == 0.0


# ---- RBenchV: always LLM judge from YAML spec ----

def test_rbenchv_uses_judge_answer():
    bench = RBenchVBenchmark(judge_spec=_HAIKU_SPEC)
    with patch("benchmarks.rbenchv.judge_answer", new=AsyncMock(return_value=(True, 0.004))) as m:
        is_correct, _ = _run(bench.grade("p", "g", "q", {}, backend="claude"))
    args, _ = m.call_args
    assert args[3] == _HAIKU_SPEC
    assert is_correct is True


# ---- grading_summary class attr present on all 6 benchmarks ----

def test_all_benchmarks_have_grading_summary():
    for cls in (LCBBenchmark, AIMEBenchmark, GPQABenchmark, HLEBenchmark,
                BabyVisionBenchmark, RBenchVBenchmark):
        assert hasattr(cls, "grading_summary"), f"{cls.__name__} missing grading_summary"
        assert isinstance(cls.grading_summary, str)
        assert len(cls.grading_summary) > 10


# ---- judge_spec is now an instance attribute carried from __init__ ----

def test_judge_spec_carried_per_instance():
    """Replaces test_judge_model_preserved_per_benchmark. judge_model class
    attribute was removed 2026-05-01; identity now comes from YAML."""
    assert LCBBenchmark().judge_spec is None
    assert AIMEBenchmark().judge_spec is None
    assert GPQABenchmark().judge_spec is None
    bench = HLEBenchmark(judge_spec=_HAIKU_SPEC)
    assert bench.judge_spec == _HAIKU_SPEC


# ---- judge_max_retries: operational knob, defaulted, not in judge_spec ----

def test_judge_max_retries_default_is_3():
    """All BenchmarkConfig subclasses default to judge_max_retries=3."""
    for cls in (LCBBenchmark, AIMEBenchmark, GPQABenchmark,
                HLEBenchmark, BabyVisionBenchmark, RBenchVBenchmark):
        assert cls().judge_max_retries == 3, f"{cls.__name__} default != 3"


def test_judge_max_retries_overridable_at_construction():
    bench = HLEBenchmark(judge_spec=_HAIKU_SPEC, judge_max_retries=7)
    assert bench.judge_max_retries == 7


def test_hle_grade_forwards_judge_max_retries_to_judge_answer():
    """HLEBenchmark.grade() passes self.judge_max_retries as max_retries kwarg."""
    bench = HLEBenchmark(judge_spec=_HAIKU_SPEC, judge_max_retries=7)
    fake_row = {"answer_type": "exactMatch"}
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=(True, 0.001))) as m:
        _run(bench.grade("predicted", "gold", "q?", fake_row, backend="claude"))
    _, kwargs = m.call_args
    assert kwargs.get("max_retries") == 7


def test_babyvision_grade_forwards_judge_max_retries_to_judge_answer():
    bench = BabyVisionBenchmark(judge_spec=_HAIKU_SPEC, judge_max_retries=5)
    fake_row = {"ansType": "blank"}
    with patch("benchmarks.babyvision.judge_answer", new=AsyncMock(return_value=(False, 0.002))) as m:
        _run(bench.grade("p", "g", "q?", fake_row, backend="claude"))
    _, kwargs = m.call_args
    assert kwargs.get("max_retries") == 5


def test_rbenchv_grade_forwards_judge_max_retries_to_judge_answer():
    bench = RBenchVBenchmark(judge_spec=_HAIKU_SPEC, judge_max_retries=4)
    fake_row = {}
    with patch("benchmarks.rbenchv.judge_answer", new=AsyncMock(return_value=(True, 0.003))) as m:
        _run(bench.grade("p", "g", "q?", fake_row, backend="claude"))
    _, kwargs = m.call_args
    assert kwargs.get("max_retries") == 4
