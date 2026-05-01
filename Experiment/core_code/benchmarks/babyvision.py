"""BabyVision benchmark configuration."""

from __future__ import annotations


from benchmarks.base import BenchmarkConfig, ANSWER_FORMAT_RULES, image_to_data_url
from benchmarks.grader import check_answer, judge_answer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_babyvision_dataset() -> list[dict]:
    """Load all rows from the BabyVision dataset."""
    from datasets import load_dataset
    ds = load_dataset("UnipatAI/BabyVision")
    split = "train" if "train" in ds else list(ds.keys())[0]
    return list(ds[split])


def _filter_dataset(
    rows: list[dict],
    type: str | None = None,
    subtype: str | None = None,
) -> list[dict]:
    """Filter dataset by type and/or subtype."""
    filtered = []
    for r in rows:
        if type and r.get("type", "").lower() != type.lower():
            continue
        if subtype and r.get("subtype", "").lower() != subtype.lower():
            continue
        filtered.append(r)
    return filtered


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _format_choices(row: dict) -> str:
    choices = row.get("options", [])
    return " ".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))


class BabyVisionBenchmark(BenchmarkConfig):
    name = "babyvision"
    # 2026-05-01: judge_model class attribute removed; identity now in YAML.
    grading_summary = (
        "hybrid: string match for ansType=choice rows; "
        "LLM judge per YAML benchmark.judge block for ansType=blank rows"
    )
    explorer_base_prompt = f"""\
You are an expert problem solver specializing in visual reasoning and cognitive tasks.
Solve the given problem step by step.
If you cannot solve it exactly, give your best estimate and set confidence accordingly.

{ANSWER_FORMAT_RULES}
"""

    def load_dataset(self) -> list[dict]:
        return _load_babyvision_dataset()

    def filter_dataset(self, rows: list[dict], **kwargs) -> list[dict]:
        return _filter_dataset(rows, **kwargs)

    def get_question(self, row: dict) -> str:
        q = row["question"]
        if row.get("ansType") == "choice":
            q += "\n" + _format_choices(row)
        # 2026-05-01: dropped the "\\boxed{Answer}" instruction that previously
        # appeared here. It collided with the orchestrator's StructuredOutput
        # tool requirement: in run_20260501_064415, 142/388 BabyVision questions
        # (36.6%) had the orchestrator write a \\boxed{X} answer in the trajectory
        # but skip the StructuredOutput tool call, leaving predicted_answer empty
        # and the question graded wrong. The 4-benchmark control confirms this is
        # BabyVision-specific (HLE/GPQA/LCB had 0 SO-skip, 0% \\boxed in
        # benchmark prompt). Replacing with an explicit SO instruction.
        q += '\n\nThink about the question, then submit your final answer by calling the StructuredOutput tool. Do not write the answer as \\boxed{...} in free-form text; the tool call is the only accepted submission path.'
        return q

    def get_answer(self, row: dict) -> str:
        if row.get("ansType") == "choice":
            return chr(65 + row["choiceAns"])
        return str(row["blankAns"])

    def get_id(self, row: dict) -> str:
        return str(row["taskId"])

    def get_image(self, row: dict) -> str | None:
        image = row.get("image")
        if image is None:
            return None
        return image_to_data_url(image)

    def classify_subset(self, row: dict) -> str:
        return row.get("type", "unknown")

    async def grade(self, predicted, gold, question, row, backend, out_dir=None):
        if row.get("ansType") == "choice":
            return check_answer(predicted, gold, "multipleChoice"), 0.0
        return await judge_answer(
            predicted, gold, question, self.judge_spec, out_dir=out_dir,
        )

