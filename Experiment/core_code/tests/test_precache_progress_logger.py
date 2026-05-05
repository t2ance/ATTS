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
