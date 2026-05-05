"""GPQA-Diamond benchmark configuration."""

from __future__ import annotations

import hashlib
import random

from benchmarks.base import BenchmarkConfig, ANSWER_FORMAT_RULES
from benchmarks.grader import check_answer
from cache_types import JudgeOutcome


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _shuffle_choices(row: dict) -> dict:
    """Shuffle the 4 choices deterministically and store the result."""
    correct = row["Correct Answer"]
    choices = [
        correct,
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    seed = int(hashlib.md5(row["Record ID"].encode()).hexdigest(), 16)
    rng = random.Random(seed)
    rng.shuffle(choices)
    row["_choices"] = choices
    row["_correct_letter"] = chr(65 + choices.index(correct))
    return row


def _load_gpqa_dataset() -> list[dict]:
    """Load all rows from the GPQA Diamond dataset."""
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    split = "train" if "train" in ds else list(ds.keys())[0]
    return [_shuffle_choices(dict(r)) for r in ds[split]]


def _filter_dataset(
    rows: list[dict],
    domain: str | None = None,
) -> list[dict]:
    """Filter dataset by domain."""
    if domain is None:
        return rows
    return [r for r in rows if r.get("High-level domain", "").lower() == domain.lower()]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class GPQABenchmark(BenchmarkConfig):
    name = "gpqa"
    grading_summary = "string match (multipleChoice letter A-E)"
    explorer_base_prompt = f"""\
You are an expert scientist solving graduate-level science questions.
Solve the given problem step by step.
If you cannot solve it exactly, give your best estimate and set confidence accordingly.

{ANSWER_FORMAT_RULES}
"""
    integrator_base_prompt = f"""\
You are an expert answer synthesizer for graduate-level science questions.

Your job:
1. Analyze each candidate's reasoning individually
2. Identify common conclusions and points of disagreement
3. For disagreements, judge which side has more rigorous reasoning
4. If you find logical errors in any candidate, point them out
5. Produce a verified final answer

{ANSWER_FORMAT_RULES}
"""

    def load_dataset(self) -> list[dict]:
        return _load_gpqa_dataset()

    def filter_dataset(self, rows: list[dict], **kwargs) -> list[dict]:
        return _filter_dataset(rows, **kwargs)

    def get_question(self, row: dict) -> str:
        q = row["Question"]
        for i, choice in enumerate(row["_choices"]):
            q += f"\n({chr(65 + i)}) {choice}"
        return q

    def get_answer(self, row: dict) -> str:
        return row["_correct_letter"]

    def get_id(self, row: dict) -> str:
        return str(row.get("Record ID", row.get("Unnamed: 0", id(row))))

    def get_image(self, row: dict) -> str | None:
        return None

    def classify_subset(self, row: dict) -> str:
        return row.get("High-level domain", "unknown")

    async def grade(self, predicted, gold, question, row, backend) -> JudgeOutcome:
        is_correct = check_answer(predicted, gold, "multipleChoice")
        return JudgeOutcome(
            is_correct=is_correct,
            cost_usd=0.0,
            judge_spec_snapshot=None,
            input_md="",
            output_md="",
            result_dict={"correct": is_correct, "kind": "rule_based_mc"},
        )

