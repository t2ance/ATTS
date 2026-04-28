"""LiveCodeBench benchmark configuration."""

from __future__ import annotations

import argparse
from typing import Any

from benchmarks.base import BenchmarkConfig
from benchmarks.grader import grade_code, normalize_code


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_lcb_dataset(version: str = "v6") -> list[dict]:
    """Load LiveCodeBench code_generation dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("livecodebench/code_generation_lite", version, trust_remote_code=True)
    split = "test" if "test" in ds else list(ds.keys())[0]
    return list(ds[split])


def _filter_dataset(
    rows: list[dict],
    difficulty: str | None = None,
) -> list[dict]:
    """Filter LCB dataset by difficulty."""
    if difficulty is None:
        return rows
    return [r for r in rows if r.get("difficulty", "").lower() == difficulty.lower()]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_EXPLORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approach": {
            "type": "string",
            "description": "Brief description of your algorithmic approach (one sentence)",
        },
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning about the problem, edge cases, and complexity",
        },
        "code": {
            "type": "string",
            "description": "Complete Python solution code.",
        },
        "confidence": {
            "type": "number",
            "description": "Your confidence in this solution (0.0 - 1.0)",
        },
    },
    "required": ["approach", "reasoning", "code", "confidence"],
    "additionalProperties": False,
}

_INTEGRATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": "Analysis of all candidate solutions (correctness, edge cases, complexity)",
        },
        "final_code": {
            "type": "string",
            "description": "The best complete Python solution.",
        },
        "reasoning": {
            "type": "string",
            "description": "Why this solution is the best among the candidates",
        },
    },
    "required": ["analysis", "final_code", "reasoning"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class LCBBenchmark(BenchmarkConfig):
    name = "lcb"
    filter_keys = ("difficulty",)
    majority_vote_compatible = False
    judge_model = None
    grading_summary = "code execution (lcb_runner test cases)"
    explore_schema = _EXPLORE_SCHEMA
    integrate_schema = _INTEGRATION_SCHEMA
    explorer_base_prompt = """\
You are an expert Python programmer. You will be given a question (problem specification) \
and will generate a correct Python program that matches the specification and passes all tests.

- If starter code is provided, complete the given function.
- Otherwise, write a program that reads from stdin and writes to stdout.
- Handle all edge cases and be efficient enough to pass within time limits.

Put your complete solution code in the `code` field.
"""
    integrator_base_prompt = """\
You are an expert code reviewer and competitive programmer.
You will receive a problem and multiple candidate solutions.

Your job:
1. Analyze each candidate's approach and correctness
2. Check for edge cases, off-by-one errors, and complexity issues
3. For incorrect solutions, identify the bugs
4. Produce the best correct solution

Put your final solution code in the `final_code` field.
"""

    def load_dataset(self) -> list[dict]:
        return _load_lcb_dataset()

    def filter_dataset(self, rows: list[dict], **kwargs) -> list[dict]:
        return _filter_dataset(rows, **kwargs)

    def get_question(self, row: dict) -> str:
        q = row.get("question_content", row.get("question", ""))
        starter = row.get("starter_code", "").strip()
        if starter:
            q += f"\n\nStarter Code:\n```python\n{starter}\n```\n\nComplete the function above."
        return q

    def get_answer(self, row: dict) -> str:
        return ""

    def get_id(self, row: dict) -> str:
        return row.get("question_id", row.get("id", ""))

    def get_image(self, row: dict) -> str | None:
        return None

    def classify_subset(self, row: dict) -> str:
        return row.get("difficulty", "unknown").lower()

    async def grade(self, predicted, gold, question, row, backend, out_dir=None):
        return await grade_code(predicted, row)

    def normalize_answer(self, text: str) -> str:
        return normalize_code(text)

    def build_explorer_message(self, problem: str) -> str:
        return f"## Problem\n\n{problem}\n\nWrite a complete Python solution."

    def build_integrator_message(self, problem: str, candidates: list) -> str:
        parts = [f"## Problem\n\n{problem}\n\n## Candidate Solutions\n"]
        for i, c in enumerate(candidates, 1):
            parts.append(
                f"### Candidate {i}\n"
                f"- **Approach**: {c.approach}\n"
                f"- **Confidence**: {c.confidence}\n"
                f"- **Code**:\n```python\n{c.answer}\n```\n"
                f"- **Reasoning**: {c.reasoning}\n"
            )
        parts.append("\nAnalyze all candidates and produce the best solution.")
        return "\n".join(parts)

    def get_answer_from_explore(self, result: dict) -> str:
        # Empty dict means orchestrator submitted "no answer" (protocol-defined
        # action when solvers fail). Returns empty string -> grade_code marks wrong.
        return result.get("code", "")

    def get_answer_from_integrate(self, result: dict) -> str:
        return result.get("final_code", "")

    def make_filter_model(self) -> type:
        from pydantic import BaseModel
        from typing import Literal
        class LCBFilters(BaseModel):
            model_config = {"extra": "forbid"}
            difficulty: Literal["easy", "medium", "hard"] | None = None
        return LCBFilters

    def add_dataset_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
        super().add_dataset_args(parser)
