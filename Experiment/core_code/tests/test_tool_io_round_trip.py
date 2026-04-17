from __future__ import annotations

import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from methods.tool_io import CandidateRecord, FullRenderer


def test_round_trip_success_record_drops_cache_id():
    renderer = FullRenderer()
    original = CandidateRecord(
        idx=3, answer="D", confidence=0.75,
        approach="geographic verification", reasoning="Big Bend coordinates match",
        cost_usd=0.04, used=3, max_explores=8, cache_id="explore_5",
    )
    text = renderer.render(original)
    assert "Cache ID" not in text, f"Cache ID leaked into render: {text!r}"
    parsed = renderer.parse(text)
    assert parsed.cache_id == "", f"expected cache_id='', got {parsed.cache_id!r}"
    assert parsed.idx == original.idx
    assert parsed.answer == original.answer
    assert parsed.confidence == original.confidence
    assert parsed.approach == original.approach
    assert parsed.reasoning == original.reasoning
    assert parsed.cost_usd == original.cost_usd
    assert parsed.used == original.used
    assert parsed.max_explores == original.max_explores


def test_round_trip_timeout_record_drops_cache_id():
    renderer = FullRenderer()
    original = CandidateRecord(
        idx=2, answer="", confidence=0.0, approach="", reasoning="",
        cost_usd=0.0, used=2, max_explores=8, cache_id="explore_7", timed_out=True,
    )
    text = renderer.render(original)
    assert "Cache ID" not in text
    parsed = renderer.parse(text)
    assert parsed.timed_out is True
    assert parsed.cache_id == ""
    assert parsed.idx == original.idx
    assert parsed.used == original.used
    assert parsed.max_explores == original.max_explores
