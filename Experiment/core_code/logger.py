"""Real-time logging for TTS agent runs.

Creates a timestamped run directory under logs/ with:
  - progress.json  — updated after every question (for live monitoring)
  - rounds.jsonl   — one line per round, written in real time
  - results.jsonl  — one line per completed question
  - run_config.json — snapshot of run configuration
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _json_dump(obj: Any, fp: Any) -> None:
    json.dump(obj, fp, indent=2, ensure_ascii=False, default=str)


def now_str() -> str:
    """Wall-clock timestamp string for stdout heartbeat lines.

    Single source of truth for the human-readable timestamp format used
    across eval.py and precache_explores.py worker progress prints.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


@dataclass(frozen=True)
class _TaskRecord:
    """One precache call's contribution to the progress aggregate.

    Built from a parsed result.json payload. Frozen because record_task
    semantics is "first writer wins per (qid, idx)" — once a task is
    classified, re-recording it overwrites by replacing the dict entry,
    not by mutating the record in place.
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
        the synthesis, _classify_result_json would mark this as
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


_LOGGING_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_console_logging(level: int = logging.INFO) -> None:
    """Configure the root logger to stream to stdout in unified format.

    Idempotent: safe to call from multiple entry points or twice in the
    same process; the second call is a no-op. Stdout (not stderr) so that
    `nohup ... > log 2>&1` and `tail -f log` continue to surface messages
    the same way bare print did before this migration.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    _LOGGING_CONFIGURED = True


@dataclass
class RunLogger:
    """Manages a single evaluation run's log directory."""

    run_dir: Path
    _start_time: float = field(default_factory=time.time, init=False)
    _question_count: int = field(default=0, init=False)
    _correct_count: int = field(default=0, init=False)
    _error_count: int = field(default=0, init=False)
    _total_cost: float = field(default=0.0, init=False)
    _latest_summary: dict[str, Any] | None = field(default=None, init=False)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        base_dir: str | Path = "logs",
        config: dict[str, Any] | None = None,
    ) -> "RunLogger":
        """Create a new run directory with a timestamp name.

        Args:
            base_dir: Parent directory for all runs.
            config: Optional configuration dict to persist.

        Returns:
            A RunLogger ready to use.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base_dir) / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger = cls(run_dir=run_dir)

        # Write initial progress
        logger._write_progress()

        # Persist config
        if config:
            with open(run_dir / "run_config.json", "w") as f:
                _json_dump(config, f)
                f.write("\n")

        return logger

    @classmethod
    def resume(cls, run_dir: str | Path) -> "RunLogger":
        """Resume logging into an existing run directory.

        Reads results.jsonl to restore counters so that progress.json
        stays accurate across the resumed session.
        """
        run_dir = Path(run_dir)
        assert run_dir.exists(), f"Run directory does not exist: {run_dir}"

        logger = cls(run_dir=run_dir)

        results_path = run_dir / "results.jsonl"
        if results_path.exists():
            with open(results_path) as f:
                for line in f:
                    rec = json.loads(line)
                    logger._question_count += 1
                    if rec.get("is_correct"):
                        logger._correct_count += 1
                    if str(rec.get("predicted_answer", "")).startswith("ERROR"):
                        logger._error_count += 1
                    logger._total_cost += rec.get("cost_usd", 0.0)

        # Restore summary from previous progress.json if available
        progress_path = run_dir / "progress.json"
        if progress_path.exists():
            prev = json.loads(progress_path.read_text())
            if "summary" in prev:
                logger._latest_summary = prev["summary"]

        logger._write_progress()
        return logger

    # ------------------------------------------------------------------ #
    # Real-time round logging
    # ------------------------------------------------------------------ #

    def log_round(
        self,
        question_id: str,
        round_num: int,
        action: str,
        tool_input: dict[str, Any],
        cost_usd: float = 0.0,
        rollout_idx: int | None = None,
    ) -> None:
        """Append a single round entry to rounds.jsonl (real-time).

        When rollout_idx is None (K=1 old behavior), the entry schema is
        unchanged. When rollout_idx is set, it is added as a field so rounds
        from different rollouts of the same question can be disambiguated.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question_id": question_id,
            "round": round_num,
            "action": action,
            "cost_usd": cost_usd,
            **tool_input,
        }
        if rollout_idx is not None:
            entry["rollout_idx"] = rollout_idx
        with open(self.run_dir / "rounds.jsonl", "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    # Per-question result
    # ------------------------------------------------------------------ #

    def log_question(self, record: dict[str, Any], summary: dict[str, Any] | None = None) -> None:
        """Append a completed question record and update progress."""
        # Append to results.jsonl
        with open(self.run_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Update counters
        self._question_count += 1
        if record.get("is_correct"):
            self._correct_count += 1
        if str(record.get("predicted_answer", "")).startswith("ERROR"):
            self._error_count += 1
        self._total_cost += record.get("cost_usd", 0.0)

        if summary is not None:
            self._latest_summary = summary

        # Refresh progress.json
        self._write_progress()

    # ------------------------------------------------------------------ #
    # Progress file (overwritten each time for easy polling)
    # ------------------------------------------------------------------ #

    def _write_progress(self) -> None:
        elapsed = time.time() - self._start_time
        progress = {
            "status": "running",
            "updated_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "questions_completed": self._question_count,
            "correct": self._correct_count,
            "errors": self._error_count,
            "accuracy_pct": self._correct_count / self._question_count * 100
            if self._question_count > 0
            else 0.0,
            "total_cost_usd": self._total_cost,
            "avg_cost_per_question": self._total_cost / self._question_count
            if self._question_count > 0
            else 0.0,
        }
        if self._latest_summary is not None:
            progress["summary"] = self._latest_summary
        # Atomic-ish write via tmp + rename
        tmp = self.run_dir / "progress.json.tmp"
        with open(tmp, "w") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp.rename(self.run_dir / "progress.json")

    # ------------------------------------------------------------------ #
    # Finalize
    # ------------------------------------------------------------------ #

    def finalize(self, summary: dict[str, Any] | None = None) -> None:
        """Mark the run as completed."""
        elapsed = time.time() - self._start_time
        progress = {
            "status": "completed",
            "updated_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "questions_completed": self._question_count,
            "correct": self._correct_count,
            "errors": self._error_count,
            "accuracy_pct": self._correct_count / self._question_count * 100
            if self._question_count > 0
            else 0.0,
            "total_cost_usd": self._total_cost,
            "avg_cost_per_question": self._total_cost / self._question_count
            if self._question_count > 0
            else 0.0,
        }
        final_summary = summary or self._latest_summary
        if final_summary:
            progress["summary"] = final_summary
        with open(self.run_dir / "progress.json", "w") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
            f.write("\n")
