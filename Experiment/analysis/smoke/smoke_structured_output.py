"""End-state smoke test: after 7 cached explores, does Qwen3-8B base call StructuredOutput?

Discriminating test for the prompt fix. The original failure mode is NOT that
the model fails to call any tool -- it correctly calls explore at turn 1. The
failure is that after several explores it never transitions to StructuredOutput
and just keeps calling explore until the 10-turn cap.

This test simulates the END state by manually constructing a conversation:
  system + user
  + 7 turns of (assistant tool_call to explore) (tool response with cached explore)
  -> generate next assistant turn
and checks whether the next turn is a tool_call to StructuredOutput.

PASS criterion: model emits a <tool_call> with name=StructuredOutput.
FAIL: model emits another explore call, plain prose, or nothing.

Compares two prompt variants in one run by passing system_prompt as a parameter.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_DIR = Path("/data3/peijia/dr-claw/Explain/Experiment/core_code")
sys.path.insert(0, str(PROJECT_DIR))
from methods.tool_io import CandidateRecord, FullRenderer

MAX_EXPLORES = 8  # matches training.grpo.prepare_data_hle.MAX_EXPLORES
_RENDERER = FullRenderer()

VAL_PATH = PROJECT_DIR / "training" / "training_data" / "grpo" / "val.parquet"
TOOL_CONFIG_PATH = PROJECT_DIR / "training" / "grpo" / "tool_config.yaml"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_V1", "1")

from vllm import LLM, SamplingParams


def load_tools() -> list[dict]:
    with open(TOOL_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return [t["tool_schema"] for t in cfg["tools"]]


def build_end_state_messages(row: dict, num_explores: int) -> list[dict]:
    """Build a conversation: system + user + N rounds of (explore call, tool response)."""
    messages: list[dict] = list(row["prompt"])
    cached = row["extra_info"]["tools_kwargs"]["explore"]["create_kwargs"]["cached_explores"]
    cached = list(cached)
    assert num_explores <= len(cached), f"only {len(cached)} cached explores available"

    for i in range(num_explores):
        # Assistant tool call (no content; just the call). Use OpenAI tool_calls schema.
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "explore",
                        "arguments": "{}",
                    },
                }
            ],
        })
        explore = cached[i]
        candidate_num = i + 1
        tool_text = _RENDERER.render(CandidateRecord(
            idx=candidate_num,
            answer=explore["answer"],
            confidence=float(explore["confidence"]),
            approach=explore["approach"],
            reasoning=explore["reasoning"],
            cost_usd=float(explore.get("cost_usd", 0.0)),
            used=candidate_num,
            max_explores=MAX_EXPLORES,
        ))
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": tool_text,
        })
    return messages


def classify(text: str) -> str:
    has_struct_call = (
        ("StructuredOutput" in text)
        and ("<tool_call>" in text or '"name"' in text)
    )
    has_explore_call = (
        ('"explore"' in text or "name>explore" in text)
        and ("<tool_call>" in text)
    )
    if has_struct_call:
        return "PASS-StructuredOutput"
    if has_explore_call:
        return "FAIL-still-explore"
    return "FAIL-no-tool-call-prose-only"


def main() -> None:
    df = pd.read_parquet(VAL_PATH)
    print(f"val.parquet: {len(df)} rows")
    row = df.iloc[0].to_dict()
    qid = row["extra_info"]["question_id"]
    print(f"row 0 qid={qid}")

    tools = load_tools()
    print(f"tools available: {[t['function']['name'] for t in tools]}")

    model_name = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-14B")
    print(f"\nLoading {model_name} via vllm (single GPU)...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.65,
        max_model_len=16384,
        dtype="bfloat16",
        enforce_eager=False,
        trust_remote_code=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=1024, n=1)

    for n_explores in (3, 7):
        print(f"\n========== END-STATE: {n_explores} explores already done ==========")
        messages = build_end_state_messages(row, n_explores)
        print(f"conversation length: {len(messages)} messages")
        # Show last few messages to confirm structure
        last_tool_msg = messages[-1]
        print(f"last message role={last_tool_msg['role']}, "
              f"content[:200]={last_tool_msg.get('content','')[:200]!r}")

        out = llm.chat(
            messages=messages,
            sampling_params=sampling,
            tools=tools,
            chat_template_kwargs={"enable_thinking": False},
        )
        text = out[0].outputs[0].text
        print("\n--- model continuation ---")
        print(text)
        print("--- end ---")
        print(f"VERDICT@{n_explores}_explores: {classify(text)}")


if __name__ == "__main__":
    main()
