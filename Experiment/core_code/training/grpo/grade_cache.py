"""Batch-grade HLE q101-300 explore caches using the deployed model judge.

For each question in the training pool (q101-300) and each of the 8 cached
explores, calls the judge server at JUDGE_URL to determine correctness and
writes grade.json next to each result.json.

Skips explores that already have grade.json.

Usage:
    python -m training.grpo.grade_cache
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from openai import AsyncOpenAI


CORE_CODE_DIR = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = CORE_CODE_DIR.parent
sys.path.insert(0, str(CORE_CODE_DIR))

from benchmarks.hle import _filter_dataset, _load_hle_dataset


JUDGE_URL = "http://127.0.0.1:8000/v1"
JUDGE_MODEL = "Qwen/Qwen3-8B"
JUDGE_TIMEOUT_S = 300.0
NUM_WORKERS = 32

SKIP = 100
NUM_TRAIN_POOL = 200
MAX_EXPLORES = 8
HLE_HAIKU_CACHE = EXPERIMENT_DIR / "analysis" / "cache" / "hle" / "haiku" / "gold"

_JUDGE_SYSTEM = """\
Judge whether the following [response] to [question] is correct based on the [correct_answer].

Your judgement must follow these criteria:
- extracted_final_answer: The final exact answer extracted from the [response]. Put "None" if there is no exact final answer to extract.
- reasoning: Explain why the extracted answer is correct or incorrect based on [correct_answer]. Focus only on whether there are meaningful differences. Do not comment on background, do not attempt to solve the problem, do not argue for any answer different than [correct_answer].
- correct: true if extracted_final_answer matches [correct_answer], or is within a small margin of error for numerical problems. Consider mathematical equivalence (1/2 = 0.5), notation differences (LaTeX vs Unicode), and formatting differences that don't change meaning. false otherwise."""

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "extracted_final_answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "correct": {"type": "boolean"},
    },
    "required": ["extracted_final_answer", "reasoning", "correct"],
    "additionalProperties": False,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
_LOGGER = logging.getLogger(__name__)


async def grade_one(client: AsyncOpenAI, question: str, predicted: str, gold: str) -> dict:
    user = f"[question]: {question}\n[response]: {predicted}\n[correct_answer]: {gold}"
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=6144,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_result",
                "schema": _JUDGE_SCHEMA,
                "strict": True,
            },
        },
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "repetition_penalty": 1.1,
        },
    )
    content = resp.choices[0].message.content or ""
    finish_reason = resp.choices[0].finish_reason
    assert content.strip(), f"judge returned empty content, finish={finish_reason}"
    assert finish_reason == "stop", (
        f"judge truncated (finish={finish_reason}). content[:200]={content[:200]!r}"
    )
    result = json.loads(content)
    assert isinstance(result.get("correct"), bool), f"judge bad correct field: {result}"
    return {
        "judge_model": JUDGE_MODEL,
        "is_correct": result["correct"],
        "predicted": predicted,
        "gold": gold,
        "judge_cost_usd": 0.0,
    }


async def run(pool: list[dict]) -> None:
    client = AsyncOpenAI(base_url=JUDGE_URL, api_key="dummy", timeout=JUDGE_TIMEOUT_S)
    sem = asyncio.Semaphore(NUM_WORKERS)

    counters = {"done": 0, "skipped": 0, "errors": 0}
    total = len(pool) * MAX_EXPLORES

    async def process(q_idx: int, row: dict, i: int) -> None:
        qid = row["id"]
        question = row["question"]
        gold = str(row["answer"])
        explore_dir = HLE_HAIKU_CACHE / qid / f"explore_{i}"
        grade_path = explore_dir / "grade.json"
        result_path = explore_dir / "result.json"

        if grade_path.exists():
            counters["skipped"] += 1
            return

        assert result_path.exists(), f"result.json missing: {result_path}"
        data = json.loads(result_path.read_text(encoding="utf-8"))
        predicted = data.get("answer", "")

        async with sem:
            try:
                grade = await grade_one(client, question, predicted, gold)
            except Exception as exc:
                counters["errors"] += 1
                _LOGGER.error("FAILED q%d %s explore_%d: %s", SKIP + q_idx + 1, qid, i, exc)
                return

        grade_path.write_text(
            json.dumps(grade, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        counters["done"] += 1
        total_seen = counters["done"] + counters["skipped"] + counters["errors"]
        if total_seen % 100 == 0 or total_seen == total:
            _LOGGER.info(
                "Progress: %d/%d done, %d skipped, %d errors",
                counters["done"], total, counters["skipped"], counters["errors"],
            )

    tasks = [
        process(q_idx, row, i)
        for q_idx, row in enumerate(pool)
        for i in range(1, MAX_EXPLORES + 1)
    ]
    await asyncio.gather(*tasks)
    _LOGGER.info(
        "Done. graded=%d skipped=%d errors=%d",
        counters["done"], counters["skipped"], counters["errors"],
    )
    if counters["errors"] > 0:
        _LOGGER.warning("%d explores failed — re-run to retry", counters["errors"])


def main() -> None:
    _LOGGER.info("Loading HLE dataset...")
    all_rows = _load_hle_dataset()
    gold_text = _filter_dataset(all_rows, subset="gold", text_only=True)
    pool = gold_text[SKIP:SKIP + NUM_TRAIN_POOL]
    _LOGGER.info("Training pool: %d questions (q%d-q%d)", len(pool), SKIP + 1, SKIP + NUM_TRAIN_POOL)
    for row in pool:
        assert (HLE_HAIKU_CACHE / row["id"]).exists(), f"cache dir missing: {HLE_HAIKU_CACHE / row['id']}"
    asyncio.run(run(pool))


if __name__ == "__main__":
    main()
