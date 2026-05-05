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
