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
from cache_types import JudgeOutcome


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_judge_answer_returns_judge_outcome(monkeypatch):
    """judge_answer is pure: returns JudgeOutcome, writes nothing."""
    from benchmarks.grader import judge_answer

    async def fake_call_sub_model(*args, **kwargs):
        return (
            {"correct": True, "extracted_final_answer": "42", "reasoning": "ok"},
            "trajectory text",
            0.001,
            {"prompt_tokens": 10, "completion_tokens": 5},
        )

    monkeypatch.setattr("benchmarks.grader.call_sub_model", fake_call_sub_model)

    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    outcome = _run(judge_answer(
        predicted="42", gold="42", question="what is 6*7?",
        judge_spec=spec, max_retries=1,
    ))
    assert isinstance(outcome, JudgeOutcome)
    assert outcome.is_correct is True
    assert outcome.cost_usd == 0.001
    assert outcome.judge_spec_snapshot == spec
    assert outcome.label == "claude__claude-haiku-4-5-20251001"
    assert "trajectory text" in outcome.output_md


# Standard claude-haiku judge_spec for tests that need an LLM judge.
_HAIKU_SPEC = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
_CODEX_SPEC = {"backend": "codex", "model": "gpt-5-codex-mini"}


def _make_outcome(is_correct: bool, cost_usd: float, spec: dict) -> JudgeOutcome:
    return JudgeOutcome(
        is_correct=is_correct, cost_usd=cost_usd, judge_spec_snapshot=spec,
        input_md="i", output_md="o", result_dict={"correct": is_correct},
    )


# ---- GPQA: pure string match on multipleChoice letter ----

def test_gpqa_grade_multiplechoice_correct():
    bench = GPQABenchmark()
    outcome = _run(bench.grade("c", "c", "q", {}, backend="claude"))
    assert outcome.is_correct is True
    assert outcome.cost_usd == 0.0


def test_gpqa_grade_multiplechoice_wrong():
    bench = GPQABenchmark()
    outcome = _run(bench.grade("a", "c", "q", {}, backend="claude"))
    assert outcome.is_correct is False
    assert outcome.cost_usd == 0.0


# ---- BabyVision: hybrid (choice -> string match, blank -> LLM judge) ----

def test_babyvision_choice_row_uses_string_match():
    bench = BabyVisionBenchmark(judge_spec=_HAIKU_SPEC)
    row = {"ansType": "choice"}
    outcome = _run(bench.grade("b", "b", "q", row, backend="claude"))
    assert outcome.is_correct is True
    assert outcome.cost_usd == 0.0


def test_babyvision_blank_row_uses_judge_answer():
    bench = BabyVisionBenchmark(judge_spec=_HAIKU_SPEC)
    row = {"ansType": "blank"}
    fake_outcome = _make_outcome(True, 0.001, _HAIKU_SPEC)
    with patch("benchmarks.babyvision.judge_answer", new=AsyncMock(return_value=fake_outcome)) as m:
        outcome = _run(bench.grade("foo", "bar", "q", row, backend="claude"))
    assert outcome.is_correct is True
    assert outcome.cost_usd == 0.001
    args, _ = m.call_args
    assert args[0] == "foo"
    assert args[1] == "bar"
    assert args[3] == _HAIKU_SPEC


# ---- HLE: judge identity from YAML, no bespoke backend routing ----

def test_hle_grade_uses_yaml_supplied_judge():
    bench = HLEBenchmark(judge_spec=_HAIKU_SPEC)
    row = {"answer_type": "exactMatch"}
    fake_outcome = _make_outcome(True, 0.002, _HAIKU_SPEC)
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=fake_outcome)) as m:
        outcome = _run(bench.grade("x", "y", "q", row, backend="claude"))
    args, _ = m.call_args
    assert args[3] == _HAIKU_SPEC
    assert outcome.is_correct is True


def test_hle_grade_with_codex_judge_spec():
    """Old behavior 'codex backend remaps to gpt-5-codex-mini' is now the
    user's responsibility: they put name=codex+model=gpt-5-codex-mini in YAML."""
    bench = HLEBenchmark(judge_spec=_CODEX_SPEC)
    row = {"answer_type": "exactMatch"}
    fake_outcome = _make_outcome(False, 0.003, _CODEX_SPEC)
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=fake_outcome)) as m:
        _run(bench.grade("x", "y", "q", row, backend="codex"))
    args, _ = m.call_args
    assert args[3] == _CODEX_SPEC


def test_hle_grade_orchestrator_backend_does_not_affect_judge():
    """orchestrator backend (here vllm) is independent of judge selection."""
    bench = HLEBenchmark(judge_spec=_HAIKU_SPEC)
    row = {"answer_type": "exactMatch"}
    fake_outcome = _make_outcome(True, 0.0, _HAIKU_SPEC)
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=fake_outcome)) as m:
        _run(bench.grade("x", "y", "q", row, backend="vllm"))
    args, _ = m.call_args
    assert args[3] == _HAIKU_SPEC


def test_hle_grade_multiplechoice_row_uses_string_match():
    bench = HLEBenchmark(judge_spec=_HAIKU_SPEC)
    row = {"answer_type": "multipleChoice"}
    outcome = _run(bench.grade("c", "c", "q", row, backend="claude"))
    assert outcome.is_correct is True
    assert outcome.cost_usd == 0.0


# ---- RBenchV: always LLM judge from YAML spec ----

def test_rbenchv_uses_judge_answer():
    bench = RBenchVBenchmark(judge_spec=_HAIKU_SPEC)
    fake_outcome = _make_outcome(True, 0.004, _HAIKU_SPEC)
    with patch("benchmarks.rbenchv.judge_answer", new=AsyncMock(return_value=fake_outcome)) as m:
        outcome = _run(bench.grade("p", "g", "q", {}, backend="claude"))
    args, _ = m.call_args
    assert args[3] == _HAIKU_SPEC
    assert outcome.is_correct is True


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
    fake_outcome = _make_outcome(True, 0.001, _HAIKU_SPEC)
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=fake_outcome)) as m:
        _run(bench.grade("predicted", "gold", "q?", fake_row, backend="claude"))
    _, kwargs = m.call_args
    assert kwargs.get("max_retries") == 7


def test_babyvision_grade_forwards_judge_max_retries_to_judge_answer():
    bench = BabyVisionBenchmark(judge_spec=_HAIKU_SPEC, judge_max_retries=5)
    fake_row = {"ansType": "blank"}
    fake_outcome = _make_outcome(False, 0.002, _HAIKU_SPEC)
    with patch("benchmarks.babyvision.judge_answer", new=AsyncMock(return_value=fake_outcome)) as m:
        _run(bench.grade("p", "g", "q?", fake_row, backend="claude"))
    _, kwargs = m.call_args
    assert kwargs.get("max_retries") == 5


def test_rbenchv_grade_forwards_judge_max_retries_to_judge_answer():
    bench = RBenchVBenchmark(judge_spec=_HAIKU_SPEC, judge_max_retries=4)
    fake_row = {}
    fake_outcome = _make_outcome(True, 0.003, _HAIKU_SPEC)
    with patch("benchmarks.rbenchv.judge_answer", new=AsyncMock(return_value=fake_outcome)) as m:
        _run(bench.grade("p", "g", "q?", fake_row, backend="claude"))
    _, kwargs = m.call_args
    assert kwargs.get("max_retries") == 4
