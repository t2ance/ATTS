"""Owner-blind dataclasses: each carries its own persistence schema.

JudgeOutcome  -- one verdict + its trace artifacts. Pure data. .persist(dir)
                 writes the on-disk bundle. .label_for(spec) is the canonical
                 judge-bundle directory naming function.

Exploration   -- one explore call's result + its trace. Optional verdict.
                 .persist(dir) writes input.md / output.md / result.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JudgeOutcome:
    is_correct: bool
    cost_usd: float
    judge_spec_snapshot: dict | None
    input_md: str
    output_md: str
    result_dict: dict[str, Any]

    @staticmethod
    def label_for(judge_spec: dict | None) -> str | None:
        """Canonical bundle directory name for a judge_spec.

        Returns None for rule-based grading (LCB/GPQA/AIME) -- caller checks
        `if outcome.label is not None` before persisting.
        """
        if judge_spec is None:
            return None
        return f"{judge_spec['backend']}__{judge_spec['model']}"

    @property
    def label(self) -> str | None:
        return self.label_for(self.judge_spec_snapshot)

    def persist(self, target_dir: Path) -> None:
        """Write 5-file judge bundle to target_dir.

        Caller must NOT call this when self.label is None (rule-based grading
        has no LLM trace to archive). Asserts if invoked in that state.
        """
        assert self.label is not None, (
            "JudgeOutcome.persist called for rule-based grading "
            "(judge_spec_snapshot is None). Caller must guard with "
            "`if outcome.label is not None`."
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "config.json").write_text(
            json.dumps(self.judge_spec_snapshot, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        (target_dir / "input.md").write_text(self.input_md, encoding="utf-8")
        (target_dir / "output.md").write_text(self.output_md, encoding="utf-8")
        (target_dir / "result.json").write_text(
            json.dumps({**self.result_dict, "cost_usd": self.cost_usd},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (target_dir / "grade.json").write_text(
            json.dumps({
                "judge_spec": self.judge_spec_snapshot,
                "is_correct": self.is_correct,
                "cost_usd": self.cost_usd,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# Process-level counters mirroring the original benchmarks/base.py policy.
# Banner-aggregated to avoid 800 per-call warning lines per HLE run.
_JUDGE_CACHE_STATS: dict = {
    "exact_hits": 0,
    "best_effort_hits": 0,
    "best_effort_extras": set(),
}


def reset_judge_cache_stats() -> None:
    _JUDGE_CACHE_STATS["exact_hits"] = 0
    _JUDGE_CACHE_STATS["best_effort_hits"] = 0
    _JUDGE_CACHE_STATS["best_effort_extras"] = set()


def summarize_judge_cache() -> dict:
    return {
        "exact_hits": _JUDGE_CACHE_STATS["exact_hits"],
        "best_effort_hits": _JUDGE_CACHE_STATS["best_effort_hits"],
        "best_effort_extras": sorted(_JUDGE_CACHE_STATS["best_effort_extras"]),
    }


@dataclass
class Exploration:
    """One explore call's full record. Self-describing schema.

    On-disk layout under target_dir:
      input.md         -- the prompt sent (rendered markdown)
      output.md        -- the model's raw trajectory text
      result.json      -- structured: answer, cost_usd, model, timed_out, ...
    """
    qid: str
    idx: int
    rollout_idx: int | None
    answer: str
    trajectory: str
    cost_usd: float
    model: str
    timed_out: bool = False
    # Optional grading layer attached by ExploreVariant when grader is provided.
    verdict: JudgeOutcome | None = None
    # Free-form fields preserved from backend response (usage, finish_reason, ...)
    extra: dict[str, Any] = field(default_factory=dict)
    # Inputs needed to write input.md (caller fills these in before persist).
    system_prompt: str = ""
    user_message: str = ""

    def persist(self, target_dir: Path) -> None:
        """Write input.md / output.md / result.json into target_dir.

        Used by ExploreVariant for cache_dir persistence and by callers
        for run_dir/trajectories/ mirror writes -- same schema, different
        target.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "input.md").write_text(
            f"## System Prompt\n\n{self.system_prompt}\n\n## User Message\n\n{self.user_message}",
            encoding="utf-8",
        )
        (target_dir / "output.md").write_text(self.trajectory, encoding="utf-8")
        result_payload = {
            "answer": self.answer,
            "cost_usd": self.cost_usd,
            "model": self.model,
            "timed_out": self.timed_out,
            **self.extra,
        }
        (target_dir / "result.json").write_text(
            json.dumps(result_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
