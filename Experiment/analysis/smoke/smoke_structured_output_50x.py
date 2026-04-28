"""50-row end-state stability test for 14B + StructuredOutput.

Each trial = one distinct HLE row, end-state with N=7 cached explores already
done, temp=0.0 deterministic generation. Tests cross-row generalization rather
than within-row stochasticity. Batched into a single llm.chat() call.

Outputs a success/fail tally + list of failing qids for inspection.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

PROJECT_DIR = Path("/data3/peijia/dr-claw/Explain/Experiment/core_code")
import sys
sys.path.insert(0, str(PROJECT_DIR))
from methods.tool_io import CandidateRecord, FullRenderer

MAX_EXPLORES = 8  # matches training.grpo.prepare_data_hle.MAX_EXPLORES
_RENDERER = FullRenderer()

VAL_PATH = PROJECT_DIR / "training" / "training_data" / "grpo" / "val.parquet"
TRAIN_PATH = PROJECT_DIR / "training" / "training_data" / "grpo" / "train.parquet"
TOOL_CONFIG_PATH = PROJECT_DIR / "training" / "grpo" / "tool_config.yaml"

NUM_VAL = 10
NUM_TRAIN = 40
N_EXPLORES = 7

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("VLLM_USE_V1", "1")

from vllm import LLM, SamplingParams


def load_tools() -> list[dict]:
    with open(TOOL_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return [t["tool_schema"] for t in cfg["tools"]]


def build_end_state_messages(row: dict, num_explores: int) -> list[dict]:
    messages: list[dict] = list(row["prompt"])
    cached = list(row["extra_info"]["tools_kwargs"]["explore"]["create_kwargs"]["cached_explores"])
    assert num_explores <= len(cached), f"only {len(cached)} cached explores available"
    for i in range(num_explores):
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "explore", "arguments": "{}"},
            }],
        })
        explore = cached[i]
        tool_text = _RENDERER.render(CandidateRecord(
            idx=i + 1,
            answer=explore["answer"],
            confidence=float(explore["confidence"]),
            approach=explore["approach"],
            reasoning=explore["reasoning"],
            cost_usd=float(explore.get("cost_usd", 0.0)),
            used=i + 1,
            max_explores=MAX_EXPLORES,
        ))
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": tool_text,
        })
    return messages


def classify(text: str) -> str:
    has_struct_call = ("StructuredOutput" in text) and ("<tool_call>" in text or '"name"' in text)
    has_explore_call = ('"explore"' in text or "name>explore" in text) and ("<tool_call>" in text)
    if has_struct_call:
        return "PASS"
    if has_explore_call:
        return "FAIL-still-explore"
    return "FAIL-no-tool-call"


def main() -> None:
    val_df = pd.read_parquet(VAL_PATH)
    train_df = pd.read_parquet(TRAIN_PATH)
    print(f"val={len(val_df)} rows, train={len(train_df)} rows")

    rows = []
    for i in range(NUM_VAL):
        rows.append(("val", val_df.iloc[i].to_dict()))
    for i in range(NUM_TRAIN):
        rows.append(("train", train_df.iloc[i].to_dict()))
    print(f"total trials: {len(rows)}")

    tools = load_tools()

    model_name = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-32B")
    tp = int(os.environ.get("SMOKE_TP", "2"))
    print(f"\nLoading {model_name} (TP={tp})...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.6,
        max_model_len=16384,
        dtype="bfloat16",
        enforce_eager=False,
        trust_remote_code=False,
    )
    sampling = SamplingParams(temperature=0.7, max_tokens=1024, n=1, seed=42)

    all_messages = [build_end_state_messages(r, N_EXPLORES) for _, r in rows]
    print(f"\nBatched generation of {len(all_messages)} conversations at N={N_EXPLORES} explores...")

    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling,
        tools=tools,
        chat_template_kwargs={"enable_thinking": False},
    )

    print("\n=== per-trial results ===")
    counts = {"PASS": 0, "FAIL-still-explore": 0, "FAIL-no-tool-call": 0}
    failures = []
    for (split, row), out in zip(rows, outputs):
        text = out.outputs[0].text
        verdict = classify(text)
        counts[verdict] += 1
        qid = row["extra_info"]["question_id"]
        marker = "OK" if verdict == "PASS" else "XX"
        print(f"  [{marker}] {split:5s} qid={qid} -> {verdict}")
        if verdict != "PASS":
            failures.append((split, qid, verdict, text[:200]))

    print(f"\n=== summary (N={N_EXPLORES} explores, temp=0.7, model={model_name}) ===")
    total = len(rows)
    for k, v in counts.items():
        print(f"  {k}: {v}/{total} ({100*v/total:.1f}%)")

    if failures:
        print(f"\n=== {len(failures)} failure samples (first 200 chars) ===")
        for split, qid, verdict, snippet in failures:
            print(f"\n[{verdict}] {split} qid={qid}")
            print(f"  {snippet!r}")


if __name__ == "__main__":
    main()
