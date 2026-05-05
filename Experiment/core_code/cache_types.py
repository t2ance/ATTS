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
