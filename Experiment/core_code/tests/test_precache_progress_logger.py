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


# ---------------------------------------------------------------------------
# _scan_cache_dir
# ---------------------------------------------------------------------------

from logger import _TaskRecord, _scan_cache_dir


def _write_result(cache_dir: Path, qid: str, idx: int, payload: dict) -> None:
    d = cache_dir / qid / f"explore_{idx}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_scan_empty_cache_dir(tmp_path):
    out = _scan_cache_dir(tmp_path, qids=["q1", "q2"], num_explores=4)
    assert out == {}


def test_scan_mixed_payloads(tmp_path):
    # q1: 4 explores all successful (with usage + duration + cost)
    for i in range(1, 5):
        _write_result(tmp_path, "q1", i, {
            "answer": "A", "confidence": 0.8,
            "usage": {"input_tokens": 1000 + i, "output_tokens": 500 + i},
            "duration_seconds": 10.0 + i,
            "cost_usd": 0.01 * i,
        })
    # q2: 2 successes, 1 soft_fail (no_tool_call), 1 wall_timeout
    _write_result(tmp_path, "q2", 1, {
        "answer": "B", "usage": {"input_tokens": 800, "output_tokens": 200},
        "duration_seconds": 8.0, "cost_usd": 0.005,
    })
    _write_result(tmp_path, "q2", 2, {
        "answer": "B", "usage": {"input_tokens": 900, "output_tokens": 250},
        "duration_seconds": 9.0, "cost_usd": 0.006,
    })
    _write_result(tmp_path, "q2", 3, {
        "timed_out": True, "reason": "no_tool_call",
        "usage": {"input_tokens": 700, "output_tokens": 0},
        "duration_seconds": 4.5, "cost_usd": 0.002,
    })
    _write_result(tmp_path, "q2", 4, {
        "timed_out": True, "timeout_seconds": 1200, "duration_seconds": 1201.0,
        # wall timeouts have no usage / cost
    })

    out = _scan_cache_dir(tmp_path, qids=["q1", "q2"], num_explores=4)
    assert len(out) == 8

    r = out[("q1", 1)]
    assert isinstance(r, _TaskRecord)
    assert r.bucket == "success"
    assert r.reason is None
    assert r.input_tokens == 1001
    assert r.output_tokens == 501
    assert r.duration_seconds == 11.0
    assert r.cost_usd == 0.01

    assert out[("q2", 3)].bucket == "soft_fail"
    assert out[("q2", 3)].reason == "no_tool_call"
    assert out[("q2", 3)].cost_usd == 0.002

    assert out[("q2", 4)].bucket == "wall_timeout"
    assert out[("q2", 4)].reason == "wall_timeout"
    assert out[("q2", 4)].input_tokens == 0
    assert out[("q2", 4)].output_tokens == 0
    assert out[("q2", 4)].cost_usd == 0.0
    assert out[("q2", 4)].duration_seconds == 1201.0


def test_scan_skips_qids_not_listed(tmp_path):
    # A leftover qid on disk that's no longer in the dataset filter.
    _write_result(tmp_path, "leftover", 1, {"answer": "stale"})
    _write_result(tmp_path, "q1", 1, {"answer": "fresh"})
    out = _scan_cache_dir(tmp_path, qids=["q1"], num_explores=4)
    assert ("leftover", 1) not in out
    assert ("q1", 1) in out


# ---------------------------------------------------------------------------
# _atomic_write_json + PrecacheLogger.__init__
# ---------------------------------------------------------------------------

from logger import _atomic_write_json, PrecacheLogger


def test_atomic_write_json_creates_file(tmp_path):
    path = tmp_path / "x.json"
    _atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}


def test_atomic_write_json_overwrites_existing(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"old": true}')
    _atomic_write_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"new": True}
    # No leftover .tmp file.
    assert not (tmp_path / "x.json.tmp").exists()


def test_precache_logger_init_empty_cache(tmp_path):
    log = PrecacheLogger(
        cache_dir=tmp_path,
        qids=["q1", "q2"],
        num_explores=4,
    )
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["mode"] == "precache"
    assert progress["status"] == "running"
    assert progress["tasks_total"] == 8
    assert progress["tasks_skipped_cached"] == 0
    assert progress["tasks_done_this_session"] == 0
    assert progress["tasks_remaining"] == 8
    assert progress["questions_total"] == 2
    assert progress["explores_per_question"] == 4
    assert progress["total_cost_usd"] == 0.0
    assert progress["soft_failures"]["total"] == 0
    assert progress["per_question_completion"] == {"0": 2}
    assert progress["timed_out_explores_per_question"] == {"0": 2}


def test_precache_logger_init_loads_disk_state(tmp_path):
    # 2 successful explores for q1 already on disk.
    for i in (1, 2):
        _write_result(tmp_path, "q1", i, {
            "answer": "A",
            "usage": {"input_tokens": 1000, "output_tokens": 500},
            "duration_seconds": 10.0,
            "cost_usd": 0.01,
        })
    # 1 soft-fail for q2.
    _write_result(tmp_path, "q2", 1, {
        "timed_out": True, "reason": "no_tool_call",
        "usage": {"input_tokens": 800, "output_tokens": 0},
        "duration_seconds": 5.0, "cost_usd": 0.005,
    })

    log = PrecacheLogger(cache_dir=tmp_path, qids=["q1", "q2"], num_explores=4)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["tasks_total"] == 8
    assert progress["tasks_skipped_cached"] == 3
    assert progress["tasks_done_this_session"] == 0
    assert progress["tasks_remaining"] == 5
    assert progress["total_cost_usd"] == pytest.approx(0.025)
    assert progress["tokens"]["explorer"]["calls"] == 3
    assert progress["tokens"]["explorer"]["input_tokens"]["sum"] == 2800
    assert progress["soft_failures"]["by_reason"]["no_tool_call"] == 1
    assert progress["soft_failures"]["total"] == 1
    # q1 has 2 successes, q2 has 0; histograms count qids per bucket size.
    assert progress["per_question_completion"] == {"0": 1, "2": 1}
    assert progress["timed_out_explores_per_question"] == {"0": 1, "1": 1}


# ---------------------------------------------------------------------------
# record_task + finalize
# ---------------------------------------------------------------------------

def test_precache_logger_record_task_increments_counters(tmp_path):
    log = PrecacheLogger(cache_dir=tmp_path, qids=["q1"], num_explores=4)

    log.record_task(
        qid="q1", explore_idx=1,
        result={"answer": "A"},
        usage={"input_tokens": 1000, "output_tokens": 500},
        duration_seconds=10.0,
        cost_usd=0.01,
    )

    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["tasks_done_this_session"] == 1
    assert progress["tasks_remaining"] == 3
    assert progress["tokens"]["explorer"]["calls"] == 1
    assert progress["tokens"]["explorer"]["input_tokens"]["sum"] == 1000


def test_precache_logger_record_task_soft_fail(tmp_path):
    log = PrecacheLogger(cache_dir=tmp_path, qids=["q1"], num_explores=4)

    log.record_task(
        qid="q1", explore_idx=1,
        result={"timed_out": True, "reason": "no_tool_call"},
        usage={"input_tokens": 700, "output_tokens": 0},
        duration_seconds=4.0,
        cost_usd=0.002,
    )

    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["soft_failures"]["by_reason"]["no_tool_call"] == 1
    assert progress["soft_failures"]["total"] == 1
    assert progress["timed_out_explores_per_question"]["1"] == 1


def test_precache_logger_record_task_wall_timeout(tmp_path):
    log = PrecacheLogger(cache_dir=tmp_path, qids=["q1"], num_explores=4)

    log.record_task(
        qid="q1", explore_idx=1,
        result={"timed_out": True},  # short-form returned by methods/base.py:340
        usage={},
        duration_seconds=1201.0,
        cost_usd=0.0,
    )

    progress = json.loads((tmp_path / "progress.json").read_text())
    # record_task synthesizes timeout_seconds when the caller only sees
    # the `{"timed_out": True}` short-form, so this lands in wall_timeout.
    assert progress["soft_failures"]["by_reason"]["wall_timeout"] == 1


def test_precache_logger_record_task_idempotent_on_replay(tmp_path):
    """Replaying the same (qid, idx) does not double-count."""
    log = PrecacheLogger(cache_dir=tmp_path, qids=["q1"], num_explores=4)
    log.record_task(qid="q1", explore_idx=1, result={"answer": "A"},
                    usage={"input_tokens": 100, "output_tokens": 50},
                    duration_seconds=5.0, cost_usd=0.001)
    log.record_task(qid="q1", explore_idx=1, result={"answer": "A"},
                    usage={"input_tokens": 100, "output_tokens": 50},
                    duration_seconds=5.0, cost_usd=0.001)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["tasks_done_this_session"] == 1


def test_precache_logger_finalize_marks_completed(tmp_path):
    log = PrecacheLogger(cache_dir=tmp_path, qids=["q1"], num_explores=2)
    log.record_task(qid="q1", explore_idx=1, result={"answer": "A"},
                    usage={"input_tokens": 100, "output_tokens": 50},
                    duration_seconds=5.0, cost_usd=0.001)
    log.record_task(qid="q1", explore_idx=2, result={"answer": "A"},
                    usage={"input_tokens": 100, "output_tokens": 50},
                    duration_seconds=5.0, cost_usd=0.001)
    log.finalize()
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["status"] == "completed"
    assert progress["tasks_remaining"] == 0
