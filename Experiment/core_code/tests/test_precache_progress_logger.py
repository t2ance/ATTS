"""Unit tests for the precache progress logger and its helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

import pytest

from logger import _summarize_distribution


def test_summarize_distribution_empty():
    assert _summarize_distribution([]) == {
        "min": 0.0, "p50": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0, "sum": 0.0
    }


def test_summarize_distribution_single_value():
    out = _summarize_distribution([42.0])
    assert out["min"] == 42.0
    assert out["p50"] == 42.0
    assert out["p95"] == 42.0
    assert out["max"] == 42.0
    assert out["sum"] == 42.0
    assert out["mean"] == 42.0


def test_summarize_distribution_mixed_values():
    # 1..100 inclusive
    xs = list(range(1, 101))
    out = _summarize_distribution(xs)
    assert out["min"] == 1
    assert out["max"] == 100
    assert out["sum"] == 5050
    assert out["mean"] == 50.5
    # p50 over 100 sorted ints — using statistics.median == 50.5
    assert out["p50"] == 50.5
    # p95 by sorted-index nearest-rank: index = ceil(0.95 * 100) - 1 = 94 → value 95
    assert out["p95"] == 95


# ---------------------------------------------------------------------------
# _classify_result_json
# ---------------------------------------------------------------------------

from logger import _classify_result_json


def test_classify_success():
    payload = {"answer": "A", "confidence": 0.9, "cost_usd": 0.01}
    assert _classify_result_json(payload) == ("success", None)


def test_classify_soft_fail_no_tool_call():
    payload = {"timed_out": True, "reason": "no_tool_call", "finish_reason": "stop"}
    assert _classify_result_json(payload) == ("soft_fail", "no_tool_call")


def test_classify_soft_fail_invalid_json():
    payload = {"timed_out": True, "reason": "invalid_json_in_tool_args", "json_error": "Expecting value"}
    assert _classify_result_json(payload) == ("soft_fail", "invalid_json_in_tool_args")


def test_classify_soft_fail_empty_choices():
    payload = {"timed_out": True, "reason": "empty_choices", "error_type": "EmptyChoices"}
    assert _classify_result_json(payload) == ("soft_fail", "empty_choices")


def test_classify_wall_timeout():
    # Shape produced by methods/base.py:335-338 — no `reason` key.
    payload = {"timed_out": True, "timeout_seconds": 1200, "duration_seconds": 1201.4}
    assert _classify_result_json(payload) == ("wall_timeout", "wall_timeout")


def test_classify_unknown_timed_out():
    payload = {"timed_out": True, "weird_field": 1}
    assert _classify_result_json(payload) == ("soft_fail", "other")
