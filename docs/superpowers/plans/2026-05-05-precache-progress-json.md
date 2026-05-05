# Precache progress.json (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `precache_explores.py` the same kind of live status file that `eval.py` already drops at `RUN_DIR/progress.json`, written to `<cache_dir>/progress.json` so it co-locates with the explore cache and auto-resumes whenever the cache_dir already has content.

**Architecture:**
- A new `PrecacheLogger` class in `logger.py`. It treats `<cache_dir>/<qid>/explore_<i>/result.json` as the source of truth and reconstructs all cumulative fields (counts, token sums, duration distribution, soft-failure breakdown) by scanning those files at startup, then keeps an in-memory `dict[(qid,idx) -> _TaskRecord]` updated as new tasks finish. On every update it atomically rewrites `<cache_dir>/progress.json`.
- `precache_explores.py` instantiates one `PrecacheLogger` before the worker pool starts. After every worker task completes (success, soft-fail, wall timeout), the worker calls `logger.record_task(...)` with the same `result / usage / duration / cost_usd` values it already has. `logger.finalize()` runs once at the end of `precache()`.
- Truncation rate is **out of scope** for Phase 1 because surfacing `finish_reason="length"` requires changing every backend's return signature; that's a separate plan.

**Tech stack:** Python 3.11 (`explain` conda env), pydantic v2 (already in use elsewhere — not used directly here), pytest + pytest's `tmp_path` for unit tests. No new third-party deps.

**Scope:**
- Modify: `Experiment/core_code/logger.py` (add `PrecacheLogger` and shared helpers; do NOT touch `RunLogger`).
- Modify: `Experiment/core_code/precache_explores.py` (instantiate logger, call `record_task` and `finalize`).
- Create: `Experiment/core_code/tests/test_precache_progress_logger.py`.

**Out of scope (Phase 2 / later):**
- `eval.py` progress.json enhancements (token distributions per role, `explore_count_distribution` with correctness, etc.). Requires touching shared code while the live grok-4.1-fast HLE eval may still be running.
- Any backend change to surface `finish_reason` for non-failed calls. That blocks the `truncation` field.
- A migration utility for old precache caches without `progress.json`. Not needed: scan-from-disk works on any cache_dir, including pre-existing ones.

---

## File Structure

| Path | Role |
|---|---|
| `Experiment/core_code/logger.py` | Adds: `_atomic_write_json(path, payload)`, `_summarize_distribution(values)`, `_TaskRecord` dataclass, `_classify_result_json(payload) -> ("success" \| "soft_fail" \| "wall_timeout", reason: str \| None)`, `_scan_cache_dir(cache_dir, qids, num_explores) -> dict[(qid,idx), _TaskRecord]`, and the `PrecacheLogger` class. Keeps `RunLogger` untouched. |
| `Experiment/core_code/precache_explores.py` | Instantiates `PrecacheLogger` before the worker pool; threads `logger` into `worker(...)`; calls `logger.record_task(...)` after the task finalizes (both the success path and the timed-out paths) and `logger.finalize()` after `asyncio.gather`. |
| `Experiment/core_code/tests/test_precache_progress_logger.py` | Unit tests for `_classify_result_json`, `_summarize_distribution`, `_scan_cache_dir`, `PrecacheLogger.{__init__, record_task, finalize}`, and atomic write semantics. Uses `tmp_path` to build synthetic cache_dirs; no real backend calls. |

---

## On-disk Schema (locked)

`<cache_dir>/progress.json`:

```json
{
  "mode": "precache",
  "status": "running",
  "updated_at": "2026-05-05T01:12:33.401200",
  "elapsed_seconds": 412.3,

  "tasks_total": 800,
  "tasks_skipped_cached": 88,
  "tasks_done_this_session": 312,
  "tasks_remaining": 400,

  "questions_total": 100,
  "explores_per_question": 8,

  "total_cost_usd": 1.7842,
  "avg_cost_per_task": 0.0057,

  "throughput": {
    "tasks_per_min_overall": 45.4,
    "tasks_per_min_last_5": 47.2,
    "tasks_per_min_last_10": 44.0
  },

  "wall_per_task_sec": {
    "min": 6.2, "p50": 22.1, "mean": 24.7, "p95": 58.0, "max": 121.4
  },

  "tokens": {
    "explorer": {
      "calls": 312,
      "input_tokens":  {"min": 380, "p50": 1480, "mean": 1612, "p95": 3210, "max": 4801, "sum": 503000},
      "output_tokens": {"min": 24,  "p50": 580,  "mean": 642,  "p95": 1400, "max": 2812, "sum": 200300}
    }
  },

  "soft_failures": {
    "by_reason": {
      "no_tool_call": 1,
      "invalid_json_in_tool_args": 0,
      "empty_choices": 0,
      "wall_timeout": 0,
      "other": 0
    },
    "total": 1,
    "rate_pct": 0.32
  },

  "per_question_completion": {"8": 92, "7": 5, "6": 2, "5": 1},
  "timed_out_explores_per_question": {"0": 92, "1": 5, "2": 2, "3": 1}
}
```

**Field provenance recap (cumulative from disk vs session-only):**

| field | cumulative? | source |
|---|---|---|
| `tasks_total` | n/a (constant) | `len(qids) * num_explores` |
| `tasks_skipped_cached` | yes | `result.json` count at startup |
| `tasks_done_this_session` | no | in-memory counter for this run |
| `tasks_remaining` | yes | `tasks_total - len(records)` |
| `total_cost_usd`, `avg_cost_per_task` | yes | sum of `cost_usd` across records |
| `throughput.*` | no | rolling-window over `_session_completion_times` |
| `wall_per_task_sec`, `tokens.explorer.*` | yes | aggregated over records |
| `soft_failures.*` | yes | aggregated over records |
| `per_question_completion`, `timed_out_explores_per_question` | yes | aggregated over records |

**Classification rule** (from `methods/base.py:335-340` + `:346-347`):

| `result.json` payload | bucket | `reason` |
|---|---|---|
| `{"timed_out": true, "reason": "no_tool_call", ...}` | soft_fail | `"no_tool_call"` |
| `{"timed_out": true, "reason": "invalid_json_in_tool_args", ...}` | soft_fail | `"invalid_json_in_tool_args"` |
| `{"timed_out": true, "reason": "empty_choices", ...}` | soft_fail | `"empty_choices"` |
| `{"timed_out": true, "timeout_seconds": <int>, "duration_seconds": <float>}` (no `reason`) | wall_timeout | `"wall_timeout"` |
| `{"timed_out": true, ...}` (anything else) | soft_fail | `"other"` |
| `{...success fields...}` (no `timed_out`, or `timed_out: false`) | success | `None` |

---

## Task 1: Pure helper — `_summarize_distribution`

**Files:**
- Modify: `Experiment/core_code/logger.py`
- Test: `Experiment/core_code/tests/test_precache_progress_logger.py` (create)

This task adds one stateless helper that returns `{min, p50, mean, p95, max, sum}` for any list of numbers. p50 = median; p95 via sorted-index. No new dependency. The empty-list case returns all zeros so callers don't need to special-case it.

- [ ] **Step 1: Write the failing tests for `_summarize_distribution`**

Create `Experiment/core_code/tests/test_precache_progress_logger.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/test_precache_progress_logger.py::test_summarize_distribution_empty -v
```

Expected: FAIL with `ImportError: cannot import name '_summarize_distribution' from 'logger'`.

- [ ] **Step 3: Implement `_summarize_distribution`**

Append to `Experiment/core_code/logger.py` (after the existing `now_str()` helper, before `_LOGGING_CONFIGURED`):

```python
import math
import statistics


def _summarize_distribution(values: list[float]) -> dict[str, float]:
    """Return {min, p50, mean, p95, max, sum} for a numeric list.

    Empty list returns all zeros so callers don't have to branch.
    p95 uses nearest-rank on the sorted list (no interpolation), which
    keeps the value an exact element of the input set — easier to reason
    about than numpy's default linear interpolation.
    """
    if not values:
        return {"min": 0.0, "p50": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0, "sum": 0.0}
    xs = sorted(values)
    n = len(xs)
    p95_idx = max(0, math.ceil(0.95 * n) - 1)
    return {
        "min": xs[0],
        "p50": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "p95": xs[p95_idx],
        "max": xs[-1],
        "sum": sum(xs),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/test_precache_progress_logger.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add logger.py tests/test_precache_progress_logger.py
git commit -m "feat(logger): add _summarize_distribution helper for progress stats"
```

---

## Task 2: Result-classifier helper — `_classify_result_json`

**Files:**
- Modify: `Experiment/core_code/logger.py`
- Test: `Experiment/core_code/tests/test_precache_progress_logger.py`

Maps a parsed `result.json` payload onto one of `(success, None)`, `(soft_fail, "<reason>")`, `(wall_timeout, "wall_timeout")`. This is the single place that encodes the rule above so the logger's bucket counters stay consistent. No I/O — pure dict-in / tuple-out.

- [ ] **Step 1: Add classification tests**

Append to `tests/test_precache_progress_logger.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/test_precache_progress_logger.py::test_classify_success -v
```

Expected: FAIL on import.

- [ ] **Step 3: Implement `_classify_result_json`**

Append to `Experiment/core_code/logger.py` (under `_summarize_distribution`):

```python
def _classify_result_json(payload: dict) -> tuple[str, str | None]:
    """Classify a parsed result.json into (bucket, reason).

    bucket is one of: "success", "soft_fail", "wall_timeout".
    reason is None for success; the soft-fail subcategory or "wall_timeout"
    otherwise. The rule mirrors the writers in methods/base.py:335-340
    (wall-clock timeout — no `reason` key) vs :346-347 (backend soft-failures
    that come back from the backend with a populated `reason`).
    """
    if not payload.get("timed_out"):
        return ("success", None)
    reason = payload.get("reason")
    if reason in {"no_tool_call", "invalid_json_in_tool_args", "empty_choices"}:
        return ("soft_fail", reason)
    if reason is None and "timeout_seconds" in payload:
        return ("wall_timeout", "wall_timeout")
    return ("soft_fail", "other")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/test_precache_progress_logger.py -v
```

Expected: 9 passed (3 from Task 1 + 6 new).

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add logger.py tests/test_precache_progress_logger.py
git commit -m "feat(logger): add _classify_result_json mapping result.json shape to bucket/reason"
```

---

## Task 3: `_TaskRecord` dataclass + `_scan_cache_dir`

**Files:**
- Modify: `Experiment/core_code/logger.py`
- Test: `Experiment/core_code/tests/test_precache_progress_logger.py`

`_TaskRecord` is the in-memory unit per `(qid, explore_idx)`. `_scan_cache_dir` walks the listed `qids`, reads each present `result.json`, and returns a `dict[(qid, idx)] -> _TaskRecord`. Missing files are simply absent from the dict — that's how the logger knows "still to run".

- [ ] **Step 1: Add scanner tests**

Append to `tests/test_precache_progress_logger.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/test_precache_progress_logger.py -v
```

Expected: 3 new tests fail on import.

- [ ] **Step 3: Implement `_TaskRecord` and `_scan_cache_dir`**

Append to `Experiment/core_code/logger.py` (under `_classify_result_json`):

```python
@dataclass(frozen=True)
class _TaskRecord:
    """One precache call's contribution to the progress aggregate.

    Built from a parsed result.json payload. Frozen because `record_task`
    semantics is "first writer wins per (qid, idx)" — once a task is
    classified, re-recording it is a logger-misuse bug.
    """
    qid: str
    explore_idx: int
    bucket: str            # "success" | "soft_fail" | "wall_timeout"
    reason: str | None     # None for success; sub-bucket otherwise
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    cost_usd: float


def _record_from_payload(qid: str, explore_idx: int, payload: dict) -> _TaskRecord:
    bucket, reason = _classify_result_json(payload)
    usage = payload.get("usage") or {}
    return _TaskRecord(
        qid=qid,
        explore_idx=explore_idx,
        bucket=bucket,
        reason=reason,
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
        cost_usd=float(payload.get("cost_usd", 0.0) or 0.0),
    )


def _scan_cache_dir(
    cache_dir: Path,
    qids: list[str],
    num_explores: int,
) -> dict[tuple[str, int], _TaskRecord]:
    """Walk the listed qids and load every result.json that exists.

    qids are taken from the filtered dataset, not from the cache_dir
    directory listing — that way stale leftover qids (filtered out of
    this run) never inflate the counts.
    """
    out: dict[tuple[str, int], _TaskRecord] = {}
    for qid in qids:
        for idx in range(1, num_explores + 1):
            rp = cache_dir / qid / f"explore_{idx}" / "result.json"
            if not rp.exists():
                continue
            try:
                payload = json.loads(rp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Corrupt cache file; treat same as missing so the worker
                # rewrites it on the next pass. Don't crash the logger.
                continue
            out[(qid, idx)] = _record_from_payload(qid, idx, payload)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/test_precache_progress_logger.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add logger.py tests/test_precache_progress_logger.py
git commit -m "feat(logger): add _TaskRecord + _scan_cache_dir for cumulative state"
```

---

## Task 4: `_atomic_write_json` and `PrecacheLogger.__init__`

**Files:**
- Modify: `Experiment/core_code/logger.py`
- Test: `Experiment/core_code/tests/test_precache_progress_logger.py`

Two pieces. `_atomic_write_json(path, payload)` does the tmp+rename dance the existing `RunLogger._write_progress` already does — pulled out so both classes share it. `PrecacheLogger.__init__` calls `_scan_cache_dir`, populates `self.records`, and writes the first `progress.json` synchronously before any worker fires.

- [ ] **Step 1: Tests for atomic write + initial progress.json**

Append to `tests/test_precache_progress_logger.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/test_precache_progress_logger.py -v
```

Expected: 4 new tests fail on import.

- [ ] **Step 3: Implement `_atomic_write_json` and `PrecacheLogger.__init__`**

Add to `Experiment/core_code/logger.py`. First, replace the body of `RunLogger._write_progress`'s tmp+rename block with a call to the shared helper, AND add the new class. Place this section after `_scan_cache_dir`:

```python
def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write `payload` to `path` atomically (tmp + rename).

    The existing pattern duplicated in RunLogger._write_progress; pulled
    out so PrecacheLogger uses the same dance and we have one place to
    fix any future race conditions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    tmp.rename(path)


class PrecacheLogger:
    """Writes <cache_dir>/progress.json for precache_explores.py.

    Cumulative fields are reconstructed from the result.json files already
    on disk at __init__ time. record_task() / finalize() update an in-memory
    record map and rewrite the progress file atomically.
    """

    PROGRESS_FILENAME = "progress.json"

    def __init__(
        self,
        cache_dir: Path,
        qids: list[str],
        num_explores: int,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.qids = list(qids)
        self.num_explores = int(num_explores)

        self._start_time = time.time()
        # Records: source of truth for all cumulative fields.
        self.records: dict[tuple[str, int], _TaskRecord] = _scan_cache_dir(
            self.cache_dir, self.qids, self.num_explores,
        )
        # `tasks_skipped_cached` is frozen at the count we observed on disk
        # at startup. Anything we add via record_task() afterward counts as
        # "this session" instead.
        self._initial_record_count = len(self.records)
        # Rolling window of completion timestamps for throughput.
        self._session_completion_times: list[float] = []
        self._write_progress(status="running")

    # ------------------------------------------------------------------ #
    # Snapshot / write
    # ------------------------------------------------------------------ #

    def _build_payload(self, status: str) -> dict:
        records = list(self.records.values())
        successes = [r for r in records if r.bucket == "success"]
        soft = [r for r in records if r.bucket == "soft_fail"]
        wall = [r for r in records if r.bucket == "wall_timeout"]

        # Per-qid bucket counts.
        succ_per_qid: dict[str, int] = {q: 0 for q in self.qids}
        timed_per_qid: dict[str, int] = {q: 0 for q in self.qids}
        for r in successes:
            succ_per_qid[r.qid] = succ_per_qid.get(r.qid, 0) + 1
        for r in soft + wall:
            timed_per_qid[r.qid] = timed_per_qid.get(r.qid, 0) + 1

        per_q_hist: dict[str, int] = {}
        for c in succ_per_qid.values():
            per_q_hist[str(c)] = per_q_hist.get(str(c), 0) + 1
        timed_hist: dict[str, int] = {}
        for c in timed_per_qid.values():
            timed_hist[str(c)] = timed_hist.get(str(c), 0) + 1

        # Soft-fail breakdown (wall_timeout counted in its own bucket).
        sf_keys = ["no_tool_call", "invalid_json_in_tool_args", "empty_choices", "wall_timeout", "other"]
        sf_counts: dict[str, int] = {k: 0 for k in sf_keys}
        for r in soft:
            sf_counts[r.reason or "other"] = sf_counts.get(r.reason or "other", 0) + 1
        for r in wall:
            sf_counts["wall_timeout"] += 1
        sf_total = sum(sf_counts.values())

        tasks_total = len(self.qids) * self.num_explores
        tasks_done_session = len(self.records) - self._initial_record_count
        tasks_remaining = max(0, tasks_total - len(self.records))
        total_cost = sum(r.cost_usd for r in records)
        durations = [r.duration_seconds for r in records if r.duration_seconds > 0]
        in_toks = [r.input_tokens for r in records if r.input_tokens > 0]
        out_toks = [r.output_tokens for r in records if r.output_tokens > 0]

        elapsed = time.time() - self._start_time
        throughput = self._compute_throughput(elapsed)

        return {
            "mode": "precache",
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,

            "tasks_total": tasks_total,
            "tasks_skipped_cached": self._initial_record_count,
            "tasks_done_this_session": tasks_done_session,
            "tasks_remaining": tasks_remaining,

            "questions_total": len(self.qids),
            "explores_per_question": self.num_explores,

            "total_cost_usd": total_cost,
            "avg_cost_per_task": (total_cost / len(records)) if records else 0.0,

            "throughput": throughput,
            "wall_per_task_sec": _summarize_distribution(durations),

            "tokens": {
                "explorer": {
                    "calls": len(records),
                    "input_tokens": _summarize_distribution(in_toks),
                    "output_tokens": _summarize_distribution(out_toks),
                }
            },

            "soft_failures": {
                "by_reason": sf_counts,
                "total": sf_total,
                "rate_pct": (100.0 * sf_total / len(records)) if records else 0.0,
            },

            "per_question_completion": per_q_hist,
            "timed_out_explores_per_question": timed_hist,
        }

    def _compute_throughput(self, elapsed: float) -> dict[str, float]:
        n_session = len(self._session_completion_times)
        overall = (60.0 * n_session / elapsed) if elapsed > 0 else 0.0

        def _rate_over_last(k: int) -> float:
            if n_session < k:
                return 0.0
            window = self._session_completion_times[-k:]
            span = window[-1] - window[0]
            if span <= 0:
                return 0.0
            return 60.0 * (k - 1) / span

        return {
            "tasks_per_min_overall": overall,
            "tasks_per_min_last_5": _rate_over_last(5),
            "tasks_per_min_last_10": _rate_over_last(10),
        }

    def _write_progress(self, status: str) -> None:
        payload = self._build_payload(status)
        _atomic_write_json(self.cache_dir / self.PROGRESS_FILENAME, payload)
```

Note: this adds three new imports (`dataclass`, `Path`, `Any`, `datetime`) — all already at the top of `logger.py` from the existing `RunLogger` block, except `dataclass` is `@dataclass` from `dataclasses` (also imported). Confirm by checking the file's existing top-of-file imports before adding anything new; if `Any` or `datetime` is not already imported, add them.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/test_precache_progress_logger.py -v
```

Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add logger.py tests/test_precache_progress_logger.py
git commit -m "feat(logger): add PrecacheLogger.__init__ + atomic write"
```

---

## Task 5: `PrecacheLogger.record_task` + `finalize`

**Files:**
- Modify: `Experiment/core_code/logger.py`
- Test: `Experiment/core_code/tests/test_precache_progress_logger.py`

`record_task` accepts the four pieces every backend already returns (`result, usage, duration, cost_usd`), classifies, stores into `self.records[(qid, idx)]`, appends a completion timestamp to the session window, and rewrites `progress.json`. `finalize` writes one last snapshot with `status="completed"`.

- [ ] **Step 1: Tests for record + finalize**

Append to `tests/test_precache_progress_logger.py`:

```python
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
        result={"timed_out": True},  # post-hoc: precache_explores.py only sees this
        usage={},
        duration_seconds=1201.0,
        cost_usd=0.0,
    )

    progress = json.loads((tmp_path / "progress.json").read_text())
    # Without timeout_seconds in result we'd classify as "other" — record_task
    # synthesizes timeout_seconds = duration_seconds when result has no `reason`
    # so it lands in the wall_timeout bucket. See impl.
    assert progress["soft_failures"]["by_reason"]["wall_timeout"] == 1


def test_precache_logger_record_task_idempotent_on_replay(tmp_path):
    """Replaying the same (qid, idx) does not double-count.

    record_task uses dict-set semantics — second call overwrites the first
    entry for the same key.
    """
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/test_precache_progress_logger.py -v
```

Expected: 5 new tests fail with `AttributeError: 'PrecacheLogger' object has no attribute 'record_task'`.

- [ ] **Step 3: Implement `record_task` and `finalize`**

Append to the `PrecacheLogger` class in `Experiment/core_code/logger.py`:

```python
    # ------------------------------------------------------------------ #
    # Record / finalize
    # ------------------------------------------------------------------ #

    def record_task(
        self,
        qid: str,
        explore_idx: int,
        result: dict,
        usage: dict,
        duration_seconds: float,
        cost_usd: float,
    ) -> None:
        """Record one finished precache call and rewrite progress.json.

        Idempotent on (qid, explore_idx) — overwrites any prior record for
        the same key, so replaying a worker is safe. Synthesizes
        `timeout_seconds` into the payload when the caller only sees the
        `{"timed_out": True}` short-form returned by methods/base.py:340
        (the wall-clock-timeout return path strips most fields). Without
        the synthesis, `_classify_result_json` would mark this as
        "soft_fail / other" which is wrong.
        """
        payload = dict(result)
        if (
            payload.get("timed_out")
            and "reason" not in payload
            and "timeout_seconds" not in payload
        ):
            payload["timeout_seconds"] = duration_seconds
        rec = _record_from_payload(qid, explore_idx, {
            **payload,
            "usage": usage,
            "duration_seconds": duration_seconds,
            "cost_usd": cost_usd,
        })
        prev = self.records.get((qid, explore_idx))
        self.records[(qid, explore_idx)] = rec
        if prev is None:
            self._session_completion_times.append(time.time())
        self._write_progress(status="running")

    def finalize(self) -> None:
        self._write_progress(status="completed")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/test_precache_progress_logger.py -v
```

Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add logger.py tests/test_precache_progress_logger.py
git commit -m "feat(logger): PrecacheLogger.record_task + finalize"
```

---

## Task 6: Wire `PrecacheLogger` into `precache_explores.py`

**Files:**
- Modify: `Experiment/core_code/precache_explores.py`

This task adds the live wiring. No new test — Task 7 covers the smoke check end-to-end.

- [ ] **Step 1: Read `precache_explores.py` so the diff is precise**

```bash
cat /data3/peijia/dr-claw/Explain/Experiment/core_code/precache_explores.py | head -170 | tail -130
```

Confirm worker structure: `worker(qid, row, explore_idx)` returns from THREE control-flow points (timed-out path, malformed-retry path, the bottom `completed += 1` path).

- [ ] **Step 2: Update the import block + `precache()` signature**

Edit `Experiment/core_code/precache_explores.py`:

```python
# OLD (line ~41):
from logger import setup_console_logging

# NEW:
from logger import setup_console_logging, PrecacheLogger
```

- [ ] **Step 3: Build qid list + instantiate logger inside `precache()`**

Replace the body block of `precache()` from line `tasks: list[tuple[str, dict, int]] = []` down to `if total == 0: return`:

```python
    # Build qid list (for the logger; matches the worker enqueue loop below).
    qids = [benchmark.get_id(row) for row in rows]

    # Init progress.json. Reconstructs cumulative state from any result.json
    # files already on disk in cache_dir; safe on a fresh dir too.
    progress_logger = PrecacheLogger(
        cache_dir=cache_dir,
        qids=qids,
        num_explores=num_explores,
    )
    logger.info(
        f"Progress: {cache_dir / 'progress.json'} "
        f"({progress_logger._initial_record_count}/{len(qids) * num_explores} already on disk)"
    )

    tasks: list[tuple[str, dict, int]] = []
    skipped = 0
    for row in rows:
        qid = benchmark.get_id(row)
        for i in range(1, num_explores + 1):
            if (cache_dir / qid / f"explore_{i}" / "result.json").exists():
                skipped += 1
            else:
                tasks.append((qid, row, i))

    total = len(tasks)
    logger.info(f"Tasks: {total} to run, {skipped} already cached")
    if total == 0:
        progress_logger.finalize()
        logger.info("Nothing to do.")
        return
```

- [ ] **Step 4: Pipe results into `record_task` from each worker exit point**

Update the worker function so every return path that has a result calls `record_task`. Replace the worker body from `async with sem:` down to the function's last line with:

```python
    async def worker(qid: str, row: dict, explore_idx: int) -> None:
        nonlocal completed
        async with sem:
            logger.info(f"  [{qid} explore_{explore_idx}] started")
            question_cache_dir = cache_dir / qid
            sub_model_fn = make_sub_model_caller(
                backend, cache_dir=question_cache_dir, cache_only=False,
                traj_dir=question_cache_dir, timeout=variant.model.timeout,
            )

            image_data_url = benchmark.get_image(row)
            input_text = benchmark.build_explorer_message(benchmark.get_question(row))

            result, traj, cost_usd, usage, duration = await sub_model_fn(
                system_prompt=explorer_prompt,
                user_message=input_text,
                image_data_url=image_data_url,
                model=variant.model.model,
                output_schema=explore_schema,
                cache_key=f"explore_{explore_idx}",
                budget_tokens=variant.model.budget_tokens,
                effort=variant.model.effort,
                sampling=sampling_dump,
                provider_order=variant.model.openrouter_provider_order,
                provider_allow_fallbacks=variant.model.openrouter_provider_allow_fallbacks,
            )

            import shutil
            result_dir = question_cache_dir / f"explore_{explore_idx}"

            if result.get("timed_out"):
                progress_logger.record_task(
                    qid=qid, explore_idx=explore_idx,
                    result=result, usage=usage,
                    duration_seconds=duration, cost_usd=cost_usd,
                )
                completed += 1
                logger.info(f"  [{completed}/{total}] {qid} explore_{explore_idx}: TIMED OUT after {duration:.0f}s")
                return

            try:
                answer = benchmark.get_answer_from_explore(result)
            except KeyError as e:
                shutil.rmtree(result_dir, ignore_errors=True)
                logger.warning(f"  [{qid} explore_{explore_idx}] MALFORMED (missing {e}), retrying...")
                result, traj, cost_usd, usage, duration = await sub_model_fn(
                    system_prompt=explorer_prompt,
                    user_message=input_text,
                    image_data_url=image_data_url,
                    model=variant.model.model,
                    output_schema=explore_schema,
                    cache_key=f"explore_{explore_idx}",
                    budget_tokens=variant.model.budget_tokens,
                    effort=variant.model.effort,
                    sampling=sampling_dump,
                    provider_order=variant.model.openrouter_provider_order,
                    provider_allow_fallbacks=variant.model.openrouter_provider_allow_fallbacks,
                )
                if result.get("timed_out"):
                    progress_logger.record_task(
                        qid=qid, explore_idx=explore_idx,
                        result=result, usage=usage,
                        duration_seconds=duration, cost_usd=cost_usd,
                    )
                    completed += 1
                    logger.info(f"  [{completed}/{total}] {qid} explore_{explore_idx}: TIMED OUT on retry")
                    return
                answer = benchmark.get_answer_from_explore(result)

            progress_logger.record_task(
                qid=qid, explore_idx=explore_idx,
                result=result, usage=usage,
                duration_seconds=duration, cost_usd=cost_usd,
            )
            completed += 1
            answer_short = answer.replace("\n", " ")[:80]
            logger.info(f"  [{completed}/{total}] {qid} explore_{explore_idx}: answer={answer_short}, confidence={result.get('confidence', 'N/A')}")
```

- [ ] **Step 5: Finalize after the gather**

Replace the last two lines of `precache()`:

```python
# OLD:
    await asyncio.gather(*(worker(qid, row, idx) for qid, row, idx in tasks))

    logger.info(f"\nDone. {completed} cached, {skipped} skipped (already existed).")

# NEW:
    await asyncio.gather(*(worker(qid, row, idx) for qid, row, idx in tasks))

    progress_logger.finalize()
    logger.info(
        f"Done. {completed} cached, {skipped} skipped (already existed). "
        f"Progress: {cache_dir / 'progress.json'}"
    )
```

- [ ] **Step 6: Run the existing test suite to confirm we didn't regress anything**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/ -v 2>&1 | tail -40
```

Expected: all previously-passing tests still pass; 21 new logger tests still pass.

- [ ] **Step 7: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add precache_explores.py
git commit -m "feat(precache): wire PrecacheLogger; write progress.json to <cache_dir>"
```

---

## Task 7: End-to-end smoke against rbenchv smoke yaml

**Files:**
- No code changes; uses existing `Experiment/core_code/scripts/rbenchv/openrouter/rbenchv_gemini-3-flash-preview_precache_physics_smoke4.yaml`.

Smoke that the wiring lands a real `progress.json` with sensible numbers.

- [ ] **Step 1: Inspect the smoke yaml so we know what to expect**

```bash
cat /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/rbenchv/openrouter/rbenchv_gemini-3-flash-preview_precache_physics_smoke4.yaml
```

Note `cache_dir`, `num_explores`, and the `num` filter — these set `tasks_total`.

- [ ] **Step 2: Run the smoke yaml in the foreground (~2 min, paid model — short)**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python precache_explores.py \
    --config scripts/rbenchv/openrouter/rbenchv_gemini-3-flash-preview_precache_physics_smoke4.yaml \
    2>&1 | tee /tmp/smoke_precache_progress.log
```

- [ ] **Step 3: Verify `progress.json` exists and is well-formed**

Expected: a banner line in the log "Progress: <cache_dir>/progress.json (N/M already on disk)" near the top, and "Progress: <cache_dir>/progress.json" near the bottom. Then:

```bash
SMOKE_CACHE=$(grep -oE 'Progress: [^ ]+/progress.json' /tmp/smoke_precache_progress.log | head -1 | sed 's/^Progress: //')
ls -l "$SMOKE_CACHE"
conda run -n explain --no-capture-output python -c "
import json
p = json.loads(open('$SMOKE_CACHE').read())
assert p['mode'] == 'precache'
assert p['status'] == 'completed'
assert p['tasks_remaining'] == 0
assert p['tokens']['explorer']['calls'] >= 1
print(json.dumps(p, indent=2))
"
```

Expected: prints a populated payload with all the agreed fields. `tasks_remaining == 0`, `status == "completed"`, non-zero token sums.

- [ ] **Step 4: Re-run the same yaml to confirm "non-empty resume" works**

```bash
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python precache_explores.py \
    --config scripts/rbenchv/openrouter/rbenchv_gemini-3-flash-preview_precache_physics_smoke4.yaml \
    2>&1 | tee /tmp/smoke_precache_resume.log
grep -E 'already on disk|Tasks: ' /tmp/smoke_precache_resume.log
```

Expected: banner shows "tasks_skipped_cached == tasks_total" and "Tasks: 0 to run, M already cached". `progress.json` after the rerun still has `status == "completed"` and the token sums are identical to the first run (records loaded from disk, no new records added).

- [ ] **Step 5: Final commit (none — tests already committed; this task is verification only)**

If steps 2-4 produced unexpected output, fix the bug in `logger.py` or `precache_explores.py` and reset to Task 4-6 to add a regression test. Otherwise no commit; close out the plan.

---

## Self-Review

**Spec coverage:**
- "precache 没有 progress.json，要统一写日志" → Task 4-6.
- "复用 cache_dir / 自动 resume 如果非空" → covered in Task 4 (`__init__` calls `_scan_cache_dir`); Task 7 step 4 verifies.
- Agreed schema fields — `mode/status/updated_at/elapsed_seconds`, `tasks_total/skipped_cached/done_this_session/remaining`, `questions_total/explores_per_question`, `total_cost_usd/avg_cost_per_task`, `throughput.{overall,last_5,last_10}`, `wall_per_task_sec`, `tokens.explorer.{input,output}_tokens`, `soft_failures.{by_reason,total,rate_pct}`, `per_question_completion`, `timed_out_explores_per_question` → Task 4 `_build_payload`.
- Truncation rate → explicitly out of scope (Phase 2).
- yaml `num_explores: 4` vs comment `8` discrepancy → not in scope (waiting on user).

**Placeholder scan:** No "TBD" / "fill in details". Every step has either complete code or an exact command. The one "Read the file first" step (Task 6 step 1) is concrete (`cat | head | tail`).

**Type consistency:**
- `_TaskRecord` field names used in tests match the dataclass declaration: `qid, explore_idx, bucket, reason, input_tokens, output_tokens, duration_seconds, cost_usd`.
- `_classify_result_json` returns `tuple[str, str | None]` consistently across Task 2 / Task 3 / Task 4.
- `PrecacheLogger.record_task(qid, explore_idx, result, usage, duration_seconds, cost_usd)` signature is identical in Task 5 impl, Task 5 tests, and Task 6 wiring.
- `PROGRESS_FILENAME = "progress.json"` used both by `_write_progress` and Task 7 verification.

No issues found. Plan ready.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-05-precache-progress-json.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
