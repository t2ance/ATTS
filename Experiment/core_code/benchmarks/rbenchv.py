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
        # qid design: f"{category}_{within_category_idx}" (e.g. "physics_22").
        # Reason: HF dataset has no native id field, only a 'catagory' (sic, upstream
        # typo) string. A bare global index would change meaning when only one subset
        # is filtered (physics-only run vs full run), breaking cache reuse. Encoding
        # the subset into the qid keeps cache paths stable across full/subset filters
        # and makes `ls cache/rbenchv/<orch>/` show per-subset progress at a glance.
        # The qid format is opaque to all downstream consumers (eval.py, audit.py,
        # precache_explores.py treat it as a string path segment / dict key only).
        rows = _load_rbenchv_dataset()
        from collections import defaultdict
        within_category_idx: dict[str, int] = defaultdict(int)
        for r in rows:
            cat = r.get("catagory", "unknown").lower()
            r["_qid"] = f"{cat}_{within_category_idx[cat]}"
            within_category_idx[cat] += 1
        return rows

    def filter_dataset(self, rows: list[dict], **kwargs) -> list[dict]:
        return _filter_dataset(rows, **kwargs)

    def get_question(self, row: dict) -> str:
        return row["question"]

    def get_answer(self, row: dict) -> str:
        return str(row["answer"])

    def get_id(self, row: dict) -> str:
        return row["_qid"]

    def get_image(self, row: dict) -> str | None:
        image = row.get("image")
        if image is None:
            return None
        return image_to_data_url(image)

    def classify_subset(self, row: dict) -> str:
        # Lowercase to align with HLE ('gold'/'uncertain') convention. Upstream
        # field 'catagory' (sic) returns "Physics"/"Game"/etc; we normalize at
        # the boundary so per_subset keys in results.jsonl + audit are consistent.
        return row.get("catagory", "unknown").lower()

    def add_dataset_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--category", type=str, default=None, help="Filter by category")
        super().add_dataset_args(parser)
