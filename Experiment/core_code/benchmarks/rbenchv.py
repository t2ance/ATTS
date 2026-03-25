"""R-Bench-V benchmark configuration."""

from __future__ import annotations

import argparse

from benchmarks.base import BenchmarkConfig, ANSWER_FORMAT_RULES, image_to_data_url


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_rbenchv_dataset() -> list[dict]:
    """Load all rows from the R-Bench-V dataset."""
    from datasets import load_dataset
    ds = load_dataset("R-Bench/R-Bench-V")
    split = "full" if "full" in ds else list(ds.keys())[0]
    return list(ds[split])


def _filter_dataset(
    rows: list[dict],
    category: str | None = None,
) -> list[dict]:
    """Filter dataset by category."""
    if category is None:
        return rows
    return [r for r in rows if r.get("catagory", "").lower() == category.lower()]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class RBenchVBenchmark(BenchmarkConfig):
    name = "rbenchv"
    filter_keys = ("category",)
    majority_vote_compatible = False
    judge_model = "claude-haiku-4-5-20251001"
    explorer_base_prompt = f"""\
You are an expert problem solver specializing in visual reasoning tasks.
Solve the given problem step by step.
If you cannot solve it exactly, give your best estimate and set confidence accordingly.

{ANSWER_FORMAT_RULES}
"""

    def load_dataset(self) -> list[dict]:
        rows = _load_rbenchv_dataset()
        for i, r in enumerate(rows):
            r["_index"] = i
        return rows

    def filter_dataset(self, rows: list[dict], **kwargs) -> list[dict]:
        return _filter_dataset(rows, **kwargs)

    def get_question(self, row: dict) -> str:
        return row["question"]

    def get_answer(self, row: dict) -> str:
        return str(row["answer"])

    def get_id(self, row: dict) -> str:
        return str(row["_index"])

    def get_image(self, row: dict) -> str | None:
        image = row.get("image")
        if image is None:
            return None
        return image_to_data_url(image)

    def classify_subset(self, row: dict) -> str:
        return row.get("catagory", "unknown")

    def add_dataset_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--category", type=str, default=None, help="Filter by category")
        super().add_dataset_args(parser)
