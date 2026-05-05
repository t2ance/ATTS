"""AIME benchmark configuration."""

from __future__ import annotations

import re

from benchmarks.base import BenchmarkConfig, ANSWER_FORMAT_RULES
from cache_types import JudgeOutcome


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_SOURCES = {
    2025: "yentinglin/aime_2025",
    2026: "MathArena/aime_2026",
}


def _normalize_row(row: dict, year: int) -> dict:
    """Normalize row to a common schema: id, problem, answer, year."""
    return {
        "id": f"{year}_{row.get('id', row.get('problem_idx'))}",
        "problem": row["problem"],
        "answer": str(row["answer"]),
        "year": year,
    }


def _load_aime_dataset() -> list[dict]:
    """Load all rows from AIME 2025 and 2026 datasets."""
    from datasets import load_dataset
    all_rows = []
    for year, path in _SOURCES.items():
        ds = load_dataset(path)
        split = "train" if "train" in ds else list(ds.keys())[0]
        for row in ds[split]:
            all_rows.append(_normalize_row(dict(row), year))
    return all_rows


def _filter_dataset(
    rows: list[dict],
    year: int | None = None,
) -> list[dict]:
    """Filter dataset by year."""
    if year is None:
        return rows
    return [r for r in rows if r.get("year") == year]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _normalize_aime_answer(text: str) -> str:
    """Normalize an AIME answer to an integer string."""
    s = str(text).strip()
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"^\\boxed\{(.+)\}$", r"\1", s)
    s = s.strip()
    m = re.search(r"(\d+)", s)
    if m:
        return str(int(m.group(1)))
    return s.lower()


class AIMEBenchmark(BenchmarkConfig):
    name = "aime"
    grading_summary = "string match (integer normalize, modulo 1000)"
    _year: int | None = None
    explorer_base_prompt = f"""\
You are an expert mathematician solving AIME competition problems.
The answer is always a non-negative integer between 000 and 999 inclusive.
Solve the given problem step by step.

{ANSWER_FORMAT_RULES}
"""
    integrator_base_prompt = f"""\
You are an expert answer synthesizer for AIME math competition problems.
The answer is always a non-negative integer between 000 and 999 inclusive.

Your job:
1. Analyze each candidate's reasoning individually
2. Identify common conclusions and points of disagreement
3. For disagreements, judge which side has more rigorous reasoning
4. If you find logical errors in any candidate, point them out
5. Produce a verified final answer

{ANSWER_FORMAT_RULES}
"""

    def load_dataset(self) -> list[dict]:
        rows = _load_aime_dataset()
        if self._year is not None:
            rows = _filter_dataset(rows, year=self._year)
        return rows

    def filter_dataset(self, rows: list[dict], **kwargs) -> list[dict]:
        return _filter_dataset(rows, **kwargs)

    def get_question(self, row: dict) -> str:
        return row["problem"]

    def get_answer(self, row: dict) -> str:
        return str(row["answer"])

    def get_id(self, row: dict) -> str:
        return str(row["id"])

    def get_image(self, row: dict) -> str | None:
        return None

    def classify_subset(self, row: dict) -> str:
        return str(row.get("year", "unknown"))

    async def grade(self, predicted, gold, question, row, backend) -> JudgeOutcome:
        pred_norm = _normalize_aime_answer(predicted)
        gold_norm = _normalize_aime_answer(gold)
        is_correct = pred_norm == gold_norm
        return JudgeOutcome(
            is_correct=is_correct,
            cost_usd=0.0,
            judge_spec_snapshot=None,
            input_md="",
            output_md="",
            result_dict={
                "correct": is_correct,
                "kind": "rule_based_exact",
                "pred_norm": pred_norm,
                "gold_norm": gold_norm,
            },
        )

    def normalize_answer(self, text: str) -> str:
        return _normalize_aime_answer(text)



class AIME2025Benchmark(AIMEBenchmark):
    name = "aime2025"
    _year = 2025


class AIME2026Benchmark(AIMEBenchmark):
    name = "aime2026"
    _year = 2026
