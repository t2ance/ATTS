"""Build verl GRPO parquet data source from HLE q101-300 with cached haiku explores.

Replaces the old prepare_data.py pipeline (which read sft_all.jsonl that contained
only 56 HLE episodes, all from outside the q101-300 range). This script constructs
the training data directly from the HLE dataset + cached haiku explores for the
clean q101-300 training pool used in the RFT/GRPO experiments (2026-04-11).

Output schema matches prepare_data.py exactly so verl's tool_agent_loop continues
to consume the rows without changes.

Usage:
    python -m training.grpo.prepare_data_hle
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


CORE_CODE_DIR = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = CORE_CODE_DIR.parent
sys.path.insert(0, str(CORE_CODE_DIR))

from benchmarks.hle import _filter_dataset, _load_hle_dataset
from prompts import ORCHESTRATOR_NO_INTEGRATE_SYSTEM_PROMPT, build_user_message
from training.grpo.sync_tool_config import verify_tool_config_matches_canonical


MAX_EXPLORES = 8
SKIP = 100         # first 100 HLE gold text-only questions are the held-out eval set
NUM_TRAIN_POOL = 200  # q101-300 (after skip)
VAL_FRACTION = 0.10   # first 10% of the pool becomes val (20 questions)
HLE_HAIKU_CACHE = EXPERIMENT_DIR / "analysis" / "cache" / "hle" / "haiku" / "gold"
OUT_DIR = CORE_CODE_DIR / "training_data" / "grpo"


def load_cached_explores(qid: str) -> list[dict]:
    """Load 8 cached haiku explore results for one HLE qid. Crashes on incomplete."""
    qdir = HLE_HAIKU_CACHE / qid
    assert qdir.exists(), f"cache dir missing: {qdir}"
    explores: list[dict] = []
    for i in range(1, MAX_EXPLORES + 1):
        p = qdir / f"explore_{i}" / "result.json"
        assert p.exists(), f"missing explore {i} for {qid}: {p}"
        with open(p) as f:
            data = json.load(f)
        explores.append({
            "answer": data.get("answer", ""),
            "confidence": data.get("confidence", 0.0),
            "approach": data.get("approach", ""),
            "reasoning": data.get("reasoning", ""),
            "cost_usd": data.get("cost_usd", 0.0),
        })
    assert len(explores) == MAX_EXPLORES, f"{qid}: got {len(explores)} explores"
    return explores


def build_row(row: dict) -> dict:
    """Convert one HLE question row to verl GRPO parquet row."""
    qid = row["id"]
    problem = row["question"]
    gold = str(row["answer"])
    cached_explores = load_cached_explores(qid)

    prompt_messages = [
        {"role": "system", "content": ORCHESTRATOR_NO_INTEGRATE_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(problem, MAX_EXPLORES)},
    ]

    return {
        "data_source": "atts_hle",
        "agent_name": "tool_agent",
        "prompt": prompt_messages,
        "ability": "orchestration",
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": {
            "question_id": qid,
            "question": problem,
            "benchmark": "hle",
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "explore": {
                    "create_kwargs": {
                        "cached_explores": cached_explores,
                        "max_explores": MAX_EXPLORES,
                    },
                },
                "StructuredOutput": {
                    "create_kwargs": {
                        "ground_truth": gold,
                    },
                },
            },
        },
    }


def main() -> None:
    # Fail loud if tool_config.yaml has drifted from the canonical
    # EXPLORE_SCHEMA before building any training rows.
    verify_tool_config_matches_canonical()
    print("tool_config.yaml StructuredOutput schema matches EXPLORE_SCHEMA.")

    print(f"Loading HLE dataset from {EXPERIMENT_DIR}...")
    all_rows = _load_hle_dataset()
    gold_text = _filter_dataset(all_rows, subset="gold", text_only=True)
    print(f"Gold text-only: {len(gold_text)} questions")

    assert len(gold_text) >= SKIP + NUM_TRAIN_POOL, (
        f"Need at least {SKIP + NUM_TRAIN_POOL} gold text-only questions, got {len(gold_text)}"
    )

    pool = gold_text[SKIP:SKIP + NUM_TRAIN_POOL]
    print(f"Training pool [{SKIP}:{SKIP + NUM_TRAIN_POOL}]: {len(pool)} questions (q{SKIP + 1}-q{SKIP + NUM_TRAIN_POOL})")

    rows: list[dict] = []
    for r in pool:
        rows.append(build_row(r))
    assert len(rows) == NUM_TRAIN_POOL

    val_size = max(1, int(len(rows) * VAL_FRACTION))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]
    print(f"Split: train={len(train_rows)}, val={len(val_rows)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train_rows), ("val", val_rows)]:
        df = pd.DataFrame(data)
        out = OUT_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"Wrote {out} ({len(df)} rows)")

    # Sanity check: reload and verify schema matches expected fields
    for name in ("train", "val"):
        p = OUT_DIR / f"{name}.parquet"
        df2 = pd.read_parquet(p)
        cols = set(df2.columns)
        expected = {"data_source", "agent_name", "prompt", "ability", "reward_model", "extra_info"}
        assert cols == expected, f"{name}: cols mismatch {cols} vs {expected}"
        sample = df2.iloc[0].to_dict()
        assert sample["data_source"] == "atts_hle"
        assert sample["agent_name"] == "tool_agent"
        assert len(sample["prompt"]) == 2
        assert sample["prompt"][0]["role"] == "system"
        assert sample["prompt"][1]["role"] == "user"
        ei = sample["extra_info"]
        assert ei["benchmark"] == "hle"
        assert ei["need_tools_kwargs"] is True
        assert len(ei["tools_kwargs"]["explore"]["create_kwargs"]["cached_explores"]) == MAX_EXPLORES
    print("Schema sanity check: OK")


if __name__ == "__main__":
    main()
