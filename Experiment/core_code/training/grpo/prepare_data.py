"""Convert ATTS episodes + cached explores into verl GRPO parquet format.

Reads sft_all.jsonl (for prompts and metadata) and cached explore results.
Outputs train.parquet and val.parquet for verl training.

Usage:
    python -m training.grpo.prepare_data
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CACHE_DIRS = {
    "gpqa": "analysis/cache/gpqa/haiku",
    "hle": "analysis/cache/hle/haiku/gold",
}

MAX_EXPLORES = 8


def load_cached_explores(cache_dir: Path, qid: str) -> list[dict]:
    """Load cached explore results for a question."""
    qid_dir = cache_dir / qid
    if not qid_dir.exists():
        return []
    explores = []
    for i in range(1, MAX_EXPLORES + 1):
        result_path = qid_dir / f"explore_{i}" / "result.json"
        if not result_path.exists():
            break
        with open(result_path) as f:
            data = json.load(f)
        explores.append({
            "answer": data.get("answer", ""),
            "confidence": data.get("confidence", 0.0),
            "approach": data.get("approach", ""),
            "reasoning": data.get("reasoning", ""),
        })
    return explores


def main():
    core_code = Path(__file__).resolve().parent.parent.parent  # core_code/
    base = core_code.parent  # Experiment/
    sft_data_path = core_code / "training_data" / "sft_all.jsonl"

    assert sft_data_path.exists(), f"SFT data not found: {sft_data_path}"

    # Load SFT episodes
    episodes = []
    with open(sft_data_path) as f:
        for line in f:
            episodes.append(json.loads(line))

    print(f"Loaded {len(episodes)} SFT episodes")

    # Build verl-format rows
    rows = []
    skipped = 0
    for ep in episodes:
        meta = ep["metadata"]
        benchmark = meta["benchmark"]
        qid = meta["qid"]
        gold = meta.get("gold", "")

        # Resolve cache dir
        cache_key = benchmark if benchmark in CACHE_DIRS else None
        if cache_key is None:
            skipped += 1
            continue

        cache_dir = base / CACHE_DIRS[cache_key]
        cached_explores = load_cached_explores(cache_dir, qid)
        if not cached_explores:
            skipped += 1
            continue

        # Extract prompt (system + user only)
        messages = ep["messages"]
        prompt = []
        for m in messages:
            if m["role"] in ("system", "user"):
                prompt.append(m)
            else:
                break  # Stop at first assistant message

        assert len(prompt) >= 2, f"Expected system+user, got {len(prompt)} messages for {qid}"

        row = {
            "data_source": f"atts_{benchmark}",
            "agent_name": "atts_agent",
            "prompt": prompt,
            "ability": "orchestration",
            "reward_model": {"style": "rule", "ground_truth": gold},
            "extra_info": {
                "question_id": qid,
                "benchmark": benchmark,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "explore": {
                        "create_kwargs": {
                            "cached_explores": cached_explores,
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
        rows.append(row)

    print(f"Prepared {len(rows)} GRPO rows, skipped {skipped}")

    # Split: use first 10% as val
    val_size = max(1, len(rows) // 10)
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

    # Write parquet
    out_dir = core_code / "training_data" / "grpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train_rows), ("val", val_rows)]:
        # Convert to DataFrame -- store complex fields as JSON strings
        df_rows = []
        for r in data:
            df_rows.append({
                "data_source": r["data_source"],
                "agent_name": r["agent_name"],
                "prompt": json.dumps(r["prompt"], ensure_ascii=False),
                "ability": r["ability"],
                "reward_model": json.dumps(r["reward_model"], ensure_ascii=False),
                "extra_info": json.dumps(r["extra_info"], ensure_ascii=False),
            })
        df = pd.DataFrame(df_rows)
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
