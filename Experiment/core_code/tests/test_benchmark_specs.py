from __future__ import annotations
import pytest
from pydantic import BaseModel, ValidationError
from benchmarks.specs import (
    BenchmarkSpec, HLESpec, GPQASpec, LCBSpec, BabyVisionSpec,
    RBenchVSpec, AIME2025Spec, AIME2026Spec,
)


class _Holder(BaseModel):
    benchmark: BenchmarkSpec


# Shared judge fixture for hle/babyvision/rbenchv specs that now require it.
_JUDGE = {"name": "claude", "model": "claude-haiku-4-5-20251001"}


def test_hle_full():
    h = _Holder.model_validate({
        "benchmark": {"name": "hle", "subset": "gold", "text_only": True, "judge": _JUDGE}
    })
    assert isinstance(h.benchmark, HLESpec)
    assert h.benchmark.subset == "gold"
    assert h.benchmark.text_only is True


def test_hle_minimal():
    h = _Holder.model_validate({"benchmark": {"name": "hle", "judge": _JUDGE}})
    assert isinstance(h.benchmark, HLESpec)
    assert h.benchmark.subset is None
    assert h.benchmark.text_only is False


def test_gpqa_domain():
    h = _Holder.model_validate({"benchmark": {"name": "gpqa", "domain": "physics"}})
    assert isinstance(h.benchmark, GPQASpec)
    assert h.benchmark.domain == "physics"


def test_lcb_difficulty():
    h = _Holder.model_validate({"benchmark": {"name": "lcb", "difficulty": "medium"}})
    assert isinstance(h.benchmark, LCBSpec)
    assert h.benchmark.difficulty == "medium"


def test_babyvision_type_subtype():
    h = _Holder.model_validate({
        "benchmark": {"name": "babyvision", "type": "ansType", "subtype": "choice", "judge": _JUDGE}
    })
    assert isinstance(h.benchmark, BabyVisionSpec)


def test_rbenchv_category():
    h = _Holder.model_validate({
        "benchmark": {"name": "rbenchv", "category": "Physics", "judge": _JUDGE}
    })
    assert isinstance(h.benchmark, RBenchVSpec)


def test_aime2025_year():
    h = _Holder.model_validate({"benchmark": {"name": "aime2025", "year": 2025}})
    assert isinstance(h.benchmark, AIME2025Spec)


def test_aime2026_year():
    h = _Holder.model_validate({"benchmark": {"name": "aime2026", "year": 2026}})
    assert isinstance(h.benchmark, AIME2026Spec)


def test_unknown_name_rejected():
    with pytest.raises(ValidationError):
        _Holder.model_validate({"benchmark": {"name": "nosuch"}})


def test_extra_filter_key_rejected():
    with pytest.raises(ValidationError):
        _Holder.model_validate({"benchmark": {"name": "hle", "domain": "x"}})


def test_missing_name_rejected():
    with pytest.raises(ValidationError):
        _Holder.model_validate({"benchmark": {"subset": "gold"}})
