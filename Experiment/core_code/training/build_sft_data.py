"""Parse ATTS trajectory files into SFT training data (JSONL).

Reads trajectory.md and input.md from ATTS runs, converts them into
multi-turn chat format with Qwen2.5 tool-calling conventions.

Usage:
    python -m training.build_sft_data

Output: training_data/sft_all.jsonl (one JSON object per episode)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CORE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_CODE_DIR))

from benchmarks.base import EXPLORE_SCHEMA, make_structured_output_function_schema  # noqa: E402


# Tool definitions for the orchestrator. The `explore` tool takes no parameters;
# the `StructuredOutput` tool's parameter schema is derived from the canonical
# EXPLORE_SCHEMA, so any field change in benchmarks/base.py propagates here.
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "explore",
            "description": (
                "Dispatch a fresh, independent solver on the original problem. "
                "Returns a structured candidate with answer, reasoning, approach, and confidence."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    make_structured_output_function_schema(EXPLORE_SCHEMA),
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _split_by_divider(text: str) -> list[str]:
    """Split text by ``---`` dividers that are NOT inside fenced code blocks."""
    lines = text.split("\n")
    segments: list[str] = []
    current: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            current.append(line)
        elif stripped == "---" and not in_code_block:
            segments.append("\n".join(current))
            current = []
        else:
            current.append(line)

    if current:
        segments.append("\n".join(current))
    return segments


def _extract_all_tool_results(segment: str) -> list[str]:
    """Extract text from ALL ``<details><summary>Tool result...`` blocks."""
    results = []
    for m in re.finditer(
        r"<details>\s*<summary>Tool result.*?</summary>\s*```\n?(.*?)```\s*</details>",
        segment,
        re.DOTALL,
    ):
        content = m.group(1).strip()
        if content:
            results.append(content)
    return results


def _text_after_last_details(segment: str) -> str:
    """Return text after the LAST ``</details>`` tag in a segment."""
    idx = segment.rfind("</details>")
    if idx == -1:
        return ""
    rest = segment[idx + len("</details>") :]
    rest = re.sub(r"\*\*Tool result\*\*:.*", "", rest)
    return rest.strip()


def _extract_structured_output_json(segment: str) -> dict | None:
    """Extract the JSON object from a StructuredOutput ``<details>`` block.

    Uses ``\\n``` `` (newline before closing fence) to avoid matching
    backticks that appear inside JSON string values as literal ``\\n``` ``.
    """
    match = re.search(r"```json\s*\n(.*?)\n```", segment, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    return None


def _strip_thinking_tags(text: str) -> str:
    """Convert ``<thinking>...</thinking>`` blocks to plain text."""
    return re.sub(r"</?thinking>", "", text)


def _make_tool_call(name: str, arguments: dict) -> str:
    """Format a tool call in Qwen2.5 style."""
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


# ---------------------------------------------------------------------------
# Main parsers
# ---------------------------------------------------------------------------


def parse_input_md(path: Path) -> tuple[str, str]:
    """Extract (system_prompt, user_message) from ``input.md``."""
    text = path.read_text()
    sys_match = re.search(
        r"## System Prompt\n(.*?)(?=\n## User Message|\Z)", text, re.DOTALL
    )
    usr_match = re.search(r"## User Message\n(.*?)$", text, re.DOTALL)
    system_prompt = sys_match.group(1).strip() if sys_match else ""
    user_message = usr_match.group(1).strip() if usr_match else ""
    return system_prompt, user_message


def parse_trajectory_md(path: Path) -> tuple[list[dict], dict]:
    """Parse ``trajectory.md`` into ``(messages, metadata)``.

    Returns
    -------
    messages : list[dict]
        List of ``{"role": ..., "content": ...}`` dicts.
    metadata : dict
        ``{"qid", "correct", "gold", "rounds", "num_explores"}``.
    """
    text = path.read_text()

    # -- metadata from grading section --
    metadata: dict = {}
    if m := re.search(r"# Trajectory: (\S+)", text):
        metadata["qid"] = m.group(1)
    if m := re.search(r"\*\*Result\*\*:\s*(\w+)", text):
        metadata["correct"] = m.group(1) == "CORRECT"
    if m := re.search(r"\*\*Gold\*\*:\s*(.+)", text):
        metadata["gold"] = m.group(1).strip()
    if m := re.search(r"\*\*Rounds\*\*:\s*(\d+)", text):
        metadata["rounds"] = int(m.group(1))
    if m := re.search(r"\*\*Cost\*\*:\s*\$([0-9.]+)", text):
        metadata["cost_usd"] = float(m.group(1))

    # -- cut off summary / grading --
    body = re.split(r"\n---\n## Session Summary", text)[0]

    # -- skip header (everything before first ---) --
    first_div = body.find("\n---\n")
    assert first_div != -1, f"No --- divider found in {path}"
    body = body[first_div + 5:]

    # -- split into segments --
    segments = _split_by_divider(body)

    messages: list[dict] = []
    num_explores = 0
    accumulated_text = ""

    for segment in segments:
        seg = segment.strip()
        if not seg:
            continue

        seg = _strip_thinking_tags(seg)

        if seg.startswith("**Tool call**: explore"):
            tool_results = _extract_all_tool_results(seg)

            if not tool_results:
                # Parallel explore call with no result in this segment.
                # The result will appear in a later segment's <details> block.
                # Skip -- don't emit a dangling tool call.
                continue

            # Emit one (assistant → tool) pair per candidate result.
            # This converts parallel explores into sequential for SFT.
            for i, tool_content in enumerate(tool_results):
                assistant_content = accumulated_text.strip() if i == 0 else ""
                tool_call = _make_tool_call("explore", {})
                content = (
                    f"{assistant_content}\n{tool_call}" if assistant_content else tool_call
                )
                messages.append({"role": "assistant", "content": content.strip()})
                messages.append({"role": "tool", "content": tool_content})
                num_explores += 1

            accumulated_text = _text_after_last_details(seg)

        elif "<summary><b>StructuredOutput</b></summary>" in seg:
            so_json = _extract_structured_output_json(seg)
            assert so_json is not None, f"Failed to parse StructuredOutput JSON in {path}"
            assistant_content = accumulated_text.strip()
            tool_call = _make_tool_call("StructuredOutput", so_json)
            content = (
                f"{assistant_content}\n{tool_call}" if assistant_content else tool_call
            )
            messages.append({"role": "assistant", "content": content.strip()})
            accumulated_text = ""

        else:
            # Pure assistant text -- accumulate
            accumulated_text += ("\n" + seg) if accumulated_text else seg

    metadata["num_explores"] = num_explores
    return messages, metadata


def parse_episode(trajectory_dir: Path, benchmark: str) -> dict | None:
    """Parse a single episode directory into an SFT-ready dict.

    Returns None if parsing fails (skips bad episodes).
    """
    traj_path = trajectory_dir / "trajectory.md"
    input_path = trajectory_dir / "input.md"
    if not traj_path.exists() or not input_path.exists():
        return None

    system_prompt, user_message = parse_input_md(input_path)
    conversation, metadata = parse_trajectory_md(traj_path)

    if not conversation:
        return None

    # Build full message list: system + user + conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
        *conversation,
    ]

    metadata["benchmark"] = benchmark
    return {
        "messages": messages,
        "tools": TOOL_DEFS,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Discovery and output
# ---------------------------------------------------------------------------


def discover_trajectories(run_dir: Path) -> list[Path]:
    """Find all trajectory subdirectories in a run directory."""
    traj_root = run_dir / "trajectories"
    if not traj_root.exists():
        return []
    return sorted(
        d for d in traj_root.iterdir()
        if d.is_dir() and (d / "trajectory.md").exists()
    )


# -- Source configuration --

SOURCES = {
    "gpqa": [
        "analysis/run/gpqa/sonnet_no_integrate/run_20260317_181859",
    ],
    # HLE eval run REMOVED — contaminated (overlaps with eval set).
    # HLE training data comes from HLE_TRAINING_DIR below.
}

# HLE training trajectories: auto-discovered from hle/sonnet_training/.
# These are Sonnet orchestrator runs on Gold training pool questions (101-668),
# using haiku cached explores in cache_only mode.
HLE_TRAINING_DIR = "analysis/run/hle/sonnet_training"

# First 100 Gold text-only questions are the eval set. Exclude from training.
HLE_EVAL_QIDS: set[str] | None = None


def _get_hle_eval_qids() -> set[str]:
    """Load and cache HLE eval set QIDs (first 100 Gold text-only)."""
    global HLE_EVAL_QIDS
    if HLE_EVAL_QIDS is not None:
        return HLE_EVAL_QIDS
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmarks.hle import _load_hle_dataset, _classify_subset, _filter_dataset
    all_rows = _load_hle_dataset()
    gold_text = _filter_dataset(all_rows, subset="gold", text_only=True)
    HLE_EVAL_QIDS = {r["id"] for r in gold_text[:100]}
    return HLE_EVAL_QIDS


def main():
    base = Path(__file__).resolve().parent.parent  # core_code/
    experiment_dir = base.parent  # Experiment/
    out_dir = base / "training_data"
    out_dir.mkdir(exist_ok=True)

    all_episodes = []
    stats: dict[str, dict] = {}

    # Collect from fixed sources
    for benchmark, run_paths in SOURCES.items():
        bm_episodes = []
        for rp in run_paths:
            run_dir = experiment_dir / rp
            if not run_dir.exists():
                print(f"  SKIP {run_dir} (not found)")
                continue
            traj_dirs = discover_trajectories(run_dir)
            for td in traj_dirs:
                ep = parse_episode(td, benchmark)
                if ep is not None:
                    bm_episodes.append(ep)

        correct_episodes = [e for e in bm_episodes if e["metadata"].get("correct")]
        MAX_COST_USD = 0.5
        before_cost = len(correct_episodes)
        correct_episodes = [e for e in correct_episodes if e["metadata"].get("cost_usd", 0) <= MAX_COST_USD]
        if before_cost > len(correct_episodes):
            print(f"  Cost filter: removed {before_cost - len(correct_episodes)} episodes > ${MAX_COST_USD}")
        avg_explores = (
            sum(e["metadata"]["num_explores"] for e in correct_episodes) / len(correct_episodes)
            if correct_episodes
            else 0
        )
        stats[benchmark] = {
            "total": len(bm_episodes),
            "correct": len(correct_episodes),
            "rejected": len(bm_episodes) - len(correct_episodes),
            "avg_explores": round(avg_explores, 2),
        }
        all_episodes.extend(correct_episodes)

    # Auto-discover HLE training runs (exclude eval QIDs)
    hle_train_dir = experiment_dir / HLE_TRAINING_DIR
    if hle_train_dir.exists():
        eval_qids = _get_hle_eval_qids()
        hle_train_episodes = []
        hle_eval_filtered = 0
        for run_dir in sorted(hle_train_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            traj_dirs = discover_trajectories(run_dir)
            for td in traj_dirs:
                ep = parse_episode(td, "hle")
                if ep is None:
                    continue
                if ep["metadata"]["qid"] in eval_qids:
                    hle_eval_filtered += 1
                    continue
                hle_train_episodes.append(ep)
        if hle_eval_filtered:
            print(f"  HLE training: filtered out {hle_eval_filtered} eval-set episodes")
        if hle_train_episodes:
            hle_train_episodes = [e for e in hle_train_episodes if e["metadata"].get("correct")]
            before_cost = len(hle_train_episodes)
            hle_train_episodes = [e for e in hle_train_episodes if e["metadata"].get("cost_usd", 0) <= MAX_COST_USD]
            if before_cost > len(hle_train_episodes):
                print(f"  HLE cost filter: removed {before_cost - len(hle_train_episodes)} episodes > ${MAX_COST_USD}")
            correct = len(hle_train_episodes)
            avg_explores = sum(e["metadata"]["num_explores"] for e in hle_train_episodes) / len(hle_train_episodes)
            stats["hle"] = {
                "total": len(hle_train_episodes),
                "correct": correct,
                "avg_explores": round(avg_explores, 2),
            }
            all_episodes.extend(hle_train_episodes)

    # Write all episodes to single file (splitting is done downstream)
    out_path = out_dir / "sft_all.jsonl"
    with open(out_path, "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_episodes)} episodes to {out_path}")
    print("\nPer-benchmark stats:")
    for bm, s in stats.items():
        print(f"  {bm}: {s['total']} episodes, {s['correct']} correct, avg {s['avg_explores']} explores")


if __name__ == "__main__":
    main()
