"""Convert sonnet_training HLE q101-300 trajectories to Qwen3-Thinking SFT jsonl.

Difference from build_sft_hle_q101_300.py: parses trajectory.md to extract the
orchestrator's deliberation content (Sonnet `<thinking>` blocks AND any plain
prose between tool calls) and emits it as Qwen3 `<think>...</think>` content
preceding each assistant tool_call.

Source : analysis/run/hle/sonnet_training/run_20260409_222818/
            rounds.jsonl (round-by-round tool args)
            trajectories/<qid>/trajectory.md (orchestrator deliberation)
         + benchmarks.hle (problem text)
Target : training_data/sft_hle_q101_300_thinking.jsonl

Format (per row):
  {
    "messages": [
      {"role": "system",    "content": ORCHESTRATOR_NO_INTEGRATE_SYSTEM_PROMPT},
      {"role": "user",      "content": build_user_message(problem, MAX_EXPLORES)},
      {"role": "assistant", "content": "<think>...sonnet deliberation...</think>\n<tool_call>{...explore...}</tool_call>"},
      {"role": "tool",      "content": "Candidate #1 recorded.\n..."},
      ...
      {"role": "assistant", "content": "<think>...final synthesis...</think>\n<tool_call>{...StructuredOutput...}</tool_call>"}
    ],
    "tools": [<explore schema>, <StructuredOutput schema>],
    "metadata": {"benchmark": "hle", "qid": ...}
  }

Key parsing rules for trajectory.md -> per-turn synthesis content:
  - Walk tool call markers (`**Tool call**: explore` and StructuredOutput
    `<details><summary><b>StructuredOutput</b></summary>`) in document order.
  - For each marker, take the text segment from the END of the previous tool
    result's `</details>` (or, for the first marker, from the end of the file
    metadata header `---` separator) up to the START of the current marker.
  - From that segment: strip `<thinking>` / `</thinking>` tags but KEEP their
    inner text, drop bare `---` lines, collapse whitespace.
  - The result is the synthesis content for that tool call. May be empty if
    Sonnet emitted nothing between calls.

Trajectories without a final submit_answer round are dropped.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CORE_CODE_DIR = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = CORE_CODE_DIR.parent
sys.path.insert(0, str(CORE_CODE_DIR))

from prompts import ORCHESTRATOR_NO_INTEGRATE_SYSTEM_PROMPT, build_user_message
from benchmarks.hle import _filter_dataset, _load_hle_dataset
from methods.tool_io import CandidateRecord, FullRenderer

RUN_DIR = (
    EXPERIMENT_DIR
    / "analysis"
    / "run"
    / "hle"
    / "sonnet_training"
    / "run_20260409_222818"
)
ROUNDS_PATH = RUN_DIR / "rounds.jsonl"
TRAJ_DIR = RUN_DIR / "trajectories"
OUT_PATH = CORE_CODE_DIR / "training_data" / "sft_hle_q101_300_thinking.jsonl"
TOOL_CONFIG_PATH = CORE_CODE_DIR / "training" / "grpo" / "tool_config.yaml"

MAX_EXPLORES = 8

_RENDERER = FullRenderer()


def load_tools_from_config() -> list[dict]:
    import yaml

    with open(TOOL_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return [t["tool_schema"] for t in cfg["tools"]]


def candidate_text(idx: int, answer, confidence, approach, reasoning, cost_usd) -> str:
    """Render one explore tool-return using the canonical FullRenderer.

    Matches the string emitted by the production Claude orchestrator
    (`methods.tts_agent.process_explore_result`) and the GRPO rollout tool
    (`training.grpo.explore_tool.ExploreTool`), so SFT data, RL rollout,
    and eval observe byte-identical tool-return text for `explore`.
    """
    record = CandidateRecord(
        idx=idx,
        answer=answer if answer is not None else "",
        confidence=float(confidence) if confidence is not None else 0.0,
        approach=approach if approach is not None else "",
        reasoning=reasoning if reasoning is not None else "",
        cost_usd=float(cost_usd) if cost_usd is not None else 0.0,
        used=idx,
        max_explores=MAX_EXPLORES,
    )
    return _RENDERER.render(record)


# ---------- trajectory.md parser ----------

EXPLORE_MARKER_RE = re.compile(r"\*\*Tool call\*\*: explore")
STRUCT_MARKER_RE = re.compile(r"<details>\s*<summary><b>StructuredOutput</b></summary>")
DETAILS_BLOCK_RE = re.compile(r"<details>.*?</details>", re.DOTALL)
THINKING_OPEN_RE = re.compile(r"<thinking>\s*")
THINKING_CLOSE_RE = re.compile(r"\s*</thinking>")
HEADER_END_RE = re.compile(r"^# Trajectory:.*?\n---\n", re.DOTALL)


def find_tool_markers(text: str) -> list[tuple[int, int, str]]:
    """Return ordered (start, end, kind) for every tool call marker."""
    markers: list[tuple[int, int, str]] = []
    for m in EXPLORE_MARKER_RE.finditer(text):
        markers.append((m.start(), m.end(), "explore"))
    for m in STRUCT_MARKER_RE.finditer(text):
        markers.append((m.start(), m.end(), "StructuredOutput"))
    markers.sort(key=lambda x: x[0])
    return markers


def clean_segment(segment: str) -> str:
    """Strip <details> blocks, <thinking> tags, separator dashes; return tidy text."""
    segment = DETAILS_BLOCK_RE.sub("", segment)
    segment = THINKING_OPEN_RE.sub("", segment)
    segment = THINKING_CLOSE_RE.sub("", segment)
    # drop bare --- separator lines
    segment = re.sub(r"^---\s*$", "", segment, flags=re.MULTILINE)
    # collapse 3+ blank lines into one
    segment = re.sub(r"\n{3,}", "\n\n", segment)
    return segment.strip()


def extract_think_per_call(traj_text: str, num_markers: int) -> list[str]:
    """For each tool call marker (in order), return the synthesis content that precedes it.

    The first marker's segment starts right after the file metadata header.
    Subsequent markers' segments start at the end of the previous marker's match
    (which lands inside the tool-result `<details>` block; clean_segment will
    discard the rest of that block via the `<details>...</details>` regex).
    """
    markers = find_tool_markers(traj_text)
    assert len(markers) == num_markers, (
        f"trajectory marker count ({len(markers)}) != rounds count ({num_markers})"
    )

    # Strip header for first segment computation
    header_match = HEADER_END_RE.match(traj_text)
    body_start = header_match.end() if header_match else 0

    thinks: list[str] = []
    prev_end = body_start
    for start, end, _kind in markers:
        segment = traj_text[prev_end:start]
        thinks.append(clean_segment(segment))
        prev_end = end
    return thinks


# ---------- assistant turn builders ----------


def assistant_turn(think_content: str, tool_call_json: dict) -> str:
    tool_call_text = (
        "<tool_call>\n"
        + json.dumps(tool_call_json, ensure_ascii=False)
        + "\n</tool_call>"
    )
    if think_content:
        return f"<think>\n{think_content}\n</think>\n\n{tool_call_text}"
    # Empty thinking is still wrapped, so the chat template stays consistent.
    return f"<think>\n\n</think>\n\n{tool_call_text}"


def build_messages(
    problem: str,
    rounds: list[dict],
    thinks: list[str],
) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": ORCHESTRATOR_NO_INTEGRATE_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(problem, MAX_EXPLORES)},
    ]

    explore_idx = 0
    submitted = False
    for r, think in zip(rounds, thinks):
        action = r.get("action")
        if action == "explore":
            explore_idx += 1
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_turn(think, {"name": "explore", "arguments": {}}),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "content": candidate_text(
                        explore_idx,
                        r.get("answer"),
                        r.get("confidence"),
                        r.get("approach"),
                        r.get("reasoning"),
                        r.get("cost_usd"),
                    ),
                }
            )
        elif action == "submit_answer":
            args = {
                "approach": r.get("approach") or "",
                "reasoning": r.get("reasoning") or "",
                "answer": r.get("answer") or "",
                "confidence": r.get("confidence") if r.get("confidence") is not None else 0.0,
            }
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_turn(
                        think, {"name": "StructuredOutput", "arguments": args}
                    ),
                }
            )
            submitted = True
            break
    assert submitted
    return messages


# ---------- main ----------


def main() -> None:
    print("Loading HLE dataset for problem text...")
    all_rows = _load_hle_dataset()
    gold_text = _filter_dataset(all_rows, subset="gold", text_only=True)
    qid_to_problem: dict[str, str] = {r["id"]: r["question"] for r in gold_text}
    print(f"  loaded {len(qid_to_problem)} HLE gold text-only questions")

    tools = load_tools_from_config()
    print(f"  tools schema: {[t['function']['name'] for t in tools]}")

    print(f"\nReading rounds: {ROUNDS_PATH}")
    rounds_by_qid: dict[str, list[dict]] = defaultdict(list)
    with open(ROUNDS_PATH) as f:
        for line in f:
            d = json.loads(line)
            rounds_by_qid[d["question_id"]].append(d)
    for qid in rounds_by_qid:
        rounds_by_qid[qid].sort(key=lambda r: r["round"])
    print(f"  unique qids in rounds.jsonl: {len(rounds_by_qid)}")

    rows_out: list[dict] = []
    skipped_no_submit = 0
    skipped_no_problem = 0
    skipped_no_traj = 0
    empty_think_counts: list[int] = []
    nonempty_think_counts: list[int] = []
    for qid, rounds in rounds_by_qid.items():
        if not any(r.get("action") == "submit_answer" for r in rounds):
            skipped_no_submit += 1
            continue
        problem = qid_to_problem.get(qid)
        if problem is None:
            skipped_no_problem += 1
            continue
        traj_path = TRAJ_DIR / qid / "trajectory.md"
        if not traj_path.exists():
            skipped_no_traj += 1
            continue

        # Truncate rounds to up-to-and-including submit_answer (matches what
        # build_messages will consume).
        rounds_used = []
        for r in rounds:
            rounds_used.append(r)
            if r.get("action") == "submit_answer":
                break

        traj_text = traj_path.read_text()
        thinks = extract_think_per_call(traj_text, num_markers=len(rounds_used))

        n_empty = sum(1 for t in thinks if not t)
        empty_think_counts.append(n_empty)
        nonempty_think_counts.append(len(thinks) - n_empty)

        messages = build_messages(problem, rounds_used, thinks)
        rows_out.append(
            {
                "messages": messages,
                "tools": tools,
                "metadata": {"benchmark": "hle", "qid": qid},
            }
        )

    print(f"\nBuilt {len(rows_out)} SFT rows (Qwen3-Thinking format)")
    print(f"  skipped (no submit_answer): {skipped_no_submit}")
    print(f"  skipped (no problem text):  {skipped_no_problem}")
    print(f"  skipped (no trajectory.md): {skipped_no_traj}")
    total_calls = sum(empty_think_counts) + sum(nonempty_think_counts)
    print(
        f"  total assistant turns: {total_calls}, "
        f"with non-empty <think>: {sum(nonempty_think_counts)} "
        f"({100 * sum(nonempty_think_counts) / max(total_calls, 1):.1f}%), "
        f"empty <think>: {sum(empty_think_counts)}"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUT_PATH}")

    # Sanity check first row -- print full content of all assistant turns
    print("\n--- first row sanity check ---")
    sample = rows_out[0]
    print(f"  qid: {sample['metadata']['qid']}")
    print(f"  num messages: {len(sample['messages'])}")
    for j, m in enumerate(sample["messages"]):
        role = m["role"]
        c = m["content"]
        if role == "assistant":
            print(f"\n  [{j:2d} {role}] full content:")
            print("    " + c.replace("\n", "\n    "))
        else:
            preview = c[:80].replace("\n", " ")
            print(f"  [{j:2d} {role:9s}] {preview!r}")


if __name__ == "__main__":
    main()
