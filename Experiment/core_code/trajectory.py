"""Trajectory I/O: dataclasses and TrajectoryWriter."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO, Literal

from logger import now_str


@dataclass
class CostTracker:
    """Track cumulative cost and token usage."""
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    by_component: dict[str, float] = field(default_factory=dict)

    def add(self, cost: float | None, usage: dict[str, Any] | None, component: str = "") -> None:
        self.total_cost_usd += cost or 0.0
        if usage:
            self.total_input_tokens += usage.get("input_tokens", 0)
            self.total_output_tokens += usage.get("output_tokens", 0)
        if component:
            self.by_component[component] = self.by_component.get(component, 0.0) + (cost or 0.0)


@dataclass
class RoundLog:
    """Log of a single round."""
    round_num: int
    action: str  # "explore" or "integrate"
    tool_input: dict[str, Any] = field(default_factory=dict)


class TrajectoryWriter:
    """Centralized writer for trajectory markdown files."""

    def __init__(self, f: IO[str] | None):
        self._f = f

    @classmethod
    def create(
        cls,
        traj_dir: Path,
        question_id: str,
        system_prompt: str,
        user_message: str,
        header_lines: list[str],
        title_suffix: str = "",
        image_data_url: str | None = None,
    ) -> TrajectoryWriter:
        traj_dir.mkdir(parents=True, exist_ok=True)
        full_input = f"## System Prompt\n\n{system_prompt}\n\n## User Message\n\n{user_message}"
        (traj_dir / "input.md").write_text(full_input, encoding="utf-8")
        if image_data_url:
            m = re.match(r"data:image/(\w+);base64,(.+)", image_data_url, re.DOTALL)
            assert m, f"Unexpected image_data_url format: {image_data_url[:80]}"
            ext = m.group(1).replace("jpeg", "jpg")
            (traj_dir / f"input.{ext}").write_bytes(base64.b64decode(m.group(2)))
        f = open(traj_dir / "trajectory.md", "w", encoding="utf-8")
        suffix = f" {title_suffix}" if title_suffix else ""
        header = (
            f"# Trajectory: {question_id}{suffix}\n\n"
            f"- **Started**: {now_str()}\n"
        )
        for line in header_lines:
            header += f"- {line}\n"
        header += f"- **Input**: [input.md](./input.md)\n\n---\n\n"
        f.write(header)
        f.flush()
        return cls(f)

    @classmethod
    def create_simple(cls, path: Path) -> TrajectoryWriter:
        """Open a file at path (creating parent dirs), no header."""
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(path, "w", encoding="utf-8")
        return cls(f)

    @classmethod
    def noop(cls) -> TrajectoryWriter:
        return cls(None)

    def _write(self, text: str) -> None:
        if self._f is None:
            return
        self._f.write(text)
        self._f.flush()

    def write_chunk(self, text: str) -> None:
        """Write raw text and flush immediately (no trailing newlines)."""
        self._write(text)

    def write_text(self, text: str) -> None:
        self._write(f"{text}\n\n")

    def write_tool_use(self, name: str, args: dict[str, Any]) -> None:
        if args:
            input_json = json.dumps(args, indent=2, ensure_ascii=False)
            self._write(
                f"\n---\n<details>\n<summary><b>{name}</b></summary>\n\n"
                f"```json\n{input_json}\n```\n\n</details>\n\n"
            )
        else:
            self._write(f"\n---\n**Tool call**: {name}\n\n")

    def write_tool_result(self, result_text: str) -> None:
        text = result_text.strip()
        if not text:
            return
        if "\n" in text:
            n_lines = text.count("\n") + 1
            self._write(
                f"<details>\n<summary>Tool result ({n_lines} lines)</summary>\n\n"
                f"```\n{text}\n```\n\n</details>\n\n"
            )
        else:
            self._write(f"**Tool result**: {text}\n\n")

    def write_session_summary(self, cost_usd: float | None, usage: dict[str, Any] | None) -> None:
        usage = usage or {}
        self._write(
            f"\n---\n## Session Summary\n\n"
            f"- Cost: ${cost_usd or 0}\n"
            f"- Input tokens: {usage.get('input_tokens', 0)}\n"
            f"- Output tokens: {usage.get('output_tokens', 0)}\n\n"
        )

    def write_explore_timeout(self) -> None:
        self._write("\n---\n**Explore timed out** (no answer produced)\n\n")

    def write_grading(
        self,
        is_correct: bool,
        predicted: str,
        gold: str,
        elapsed: float,
        cost: float,
        num_rounds: int,
    ) -> None:
        correct_str = "CORRECT" if is_correct else "INCORRECT"
        self._write(
            f"\n---\n## Grading\n\n"
            f"- **Result**: {correct_str}\n"
            f"- **Predicted**: {predicted}\n"
            f"- **Gold**: {gold}\n"
            f"- **Time**: {elapsed}s\n"
            f"- **Cost**: ${cost}\n"
            f"- **Rounds**: {num_rounds}\n"
        )

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


@dataclass
class SolveResult:
    """Result from a solve() call, including answer, cost, and full trace."""
    answer: str
    cost: CostTracker
    rounds: list[RoundLog] = field(default_factory=list)
    writer: TrajectoryWriter = field(default_factory=TrajectoryWriter.noop)
    # Exit state from backend's run_tool_conversation main loop:
    #   - "committed": orchestrator emitted StructuredOutput, OR a tool returned should_stop=True
    #   - "cap_exceeded": cumulative output_tokens exceeded user-set max_output_tokens cap
    #   - "incomplete": orchestrator gave up emitting tool_call (text-only response), OR
    #     for-loop walked all max_turns without commitment
    exit_reason: Literal["committed", "cap_exceeded", "incomplete"] = "incomplete"
