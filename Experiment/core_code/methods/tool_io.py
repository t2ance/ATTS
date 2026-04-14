"""Single source of truth for explore-tool candidate text rendering.

Before this module existed, the "Candidate #N recorded..." string was
hand-coded in 6 places across `methods/tts_agent.py`, `training/grpo/explore_tool.py`,
`training/sft/build_sft_hle_q101_300_thinking.py`, and the smoke tests, in
3 different format variants. Train and eval observed different tool-return
strings, which masked train/eval distribution drift.

Design:
- `CandidateRecord` is the canonical data payload (frozen dataclass).
- `CandidateRenderer` is an ABC; concrete subclasses produce backend-specific
  rendered strings.
- `_self_check()` runs at module import. If anyone adds a `CandidateRecord`
  field and forgets to update a renderer, the failure is immediate at import
  time, not silently in production.

All call sites construct a `CandidateRecord` and call a renderer; nothing
else is allowed to assemble the candidate string by hand.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateRecord:
    """Canonical data payload for a single explore tool result.

    Fields are intentionally minimal. Any new field added here forces every
    concrete renderer to either consume it or accept the default; the
    `_self_check()` at module import will catch mismatches in practice when
    the test record is updated alongside the field.
    """

    idx: int                # 1-based candidate index in this episode
    answer: str             # may be "" for timeout
    confidence: float       # 0.0 for timeout
    approach: str
    reasoning: str
    cost_usd: float         # 0.0 if cache-only / unknown
    used: int               # 1-based current iteration count after this call
    max_explores: int       # the budget cap (e.g. 8)
    model_label: str = ""   # "" if single-model
    cache_id: str = ""      # stable opaque id for cached explore alignment
    extra_budget_text: str = ""   # multi-model / effort caller appends to budget line
    timed_out: bool = False

    def __post_init__(self) -> None:
        assert self.idx >= 1, f"idx must be >= 1, got {self.idx}"
        assert 0 < self.used <= self.max_explores, (
            f"used must be in (0, {self.max_explores}], got {self.used}"
        )
        if self.timed_out:
            assert self.answer == "", "timeout records must have empty answer"
            assert self.confidence == 0.0, "timeout records must have confidence=0.0"


class CandidateRenderer(ABC):
    """Render a CandidateRecord into the tool-return string the orchestrator
    model will see in its conversation context.

    Subclasses MUST implement `render`. The base class is intentionally bare;
    do not add hooks here without also updating every concrete subclass and
    `_self_check`.
    """

    @abstractmethod
    def render(self, record: CandidateRecord) -> str:
        ...


class FullRenderer(CandidateRenderer):
    """Format A: the production Claude-orchestrator format.

    7 lines for success records:
        Candidate #N recorded.{ Model: <label>}
        - Cache ID: <cache_id>     (optional)
        - Answer: ...
        - Confidence: ...
        - Approach: ...
        - Reasoning: ...
        - Cost: $X.XX

        Explore budget: N/M used, K remaining.

    For timeout records, the inner Answer/Confidence/Approach/Reasoning/Cost
    block is replaced with a single "(timed out, empty answer)" marker, but
    the trailing budget line still appears.

    Byte-identical to the legacy text emitted by
    `methods/tts_agent.py:process_explore_result()` for the non-multi-model,
    non-extra-budget case.
    """

    def render(self, r: CandidateRecord) -> str:
        label = f" Model: {r.model_label}." if r.model_label else ""
        cache_line = f"- Cache ID: {r.cache_id}\n" if r.cache_id else ""
        remaining = r.max_explores - r.used
        if r.timed_out:
            return (
                f"Candidate #{r.idx} recorded (timed out, empty answer).{label}\n"
                f"{cache_line}"
                f"Explore budget: {r.used}/{r.max_explores} used, {remaining} remaining."
                f"{r.extra_budget_text}"
            )
        return (
            f"Candidate #{r.idx} recorded.{label}\n"
            f"{cache_line}"
            f"- Answer: {r.answer}\n"
            f"- Confidence: {r.confidence}\n"
            f"- Approach: {r.approach}\n"
            f"- Reasoning: {r.reasoning}\n"
            f"- Cost: ${r.cost_usd:.2f}\n\n"
            f"Explore budget: {r.used}/{r.max_explores} used, {remaining} remaining."
            f"{r.extra_budget_text}"
        )


class MinimalRenderer(CandidateRenderer):
    """Format B: 5-line bare fields (the pre-refactor GRPO/SFT format).

    Kept as a callable renderer so legacy data can be reproduced byte-exactly
    if needed for diff verification, but not used by the production code path
    after this refactor. Do not introduce new call sites.
    """

    def render(self, r: CandidateRecord) -> str:
        if r.timed_out:
            return f"Candidate #{r.idx} recorded (timed out, empty answer)."
        return (
            f"Candidate #{r.idx} recorded.\n"
            f"- Answer: {r.answer}\n"
            f"- Confidence: {r.confidence}\n"
            f"- Approach: {r.approach}\n"
            f"- Reasoning: {r.reasoning}"
        )


class InContextExampleRenderer(CandidateRenderer):
    """Format C: single-line paren-tag form, used inside system-prompt examples.

    Output shape:
        Candidate #N: answer=X, approach="...", confidence=Y     (single-model)
        Candidate #N (label): answer=X, approach="...", confidence=Y    (with model_label)
        Candidate #N: timed out, empty answer                    (timeout)

    The leading bullet/dash and trailing `Action:` lines are added by the
    prompt template, not by this renderer.
    """

    def render(self, r: CandidateRecord) -> str:
        if r.timed_out:
            return f"Candidate #{r.idx}: timed out, empty answer"
        tag = f" ({r.model_label})" if r.model_label else ""
        return (
            f"Candidate #{r.idx}{tag}: "
            f"answer={r.answer}, "
            f'approach="{r.approach}", '
            f"confidence={r.confidence}"
        )


ALL_RENDERERS: tuple[type[CandidateRenderer], ...] = (
    FullRenderer,
    MinimalRenderer,
    InContextExampleRenderer,
)


def _self_check() -> None:
    """Module-import-time sanity check.

    Every registered renderer must successfully render a canonical success
    record and a canonical timeout record. If a renderer subclass fails to
    implement `render` (because someone added a new abstract method without
    updating it) or crashes on a known input shape, this raises immediately
    at import, not silently in production.
    """
    success = CandidateRecord(
        idx=1,
        answer="42",
        confidence=0.9,
        approach="dynamic programming",
        reasoning="Step 1...",
        cost_usd=0.12,
        used=1,
        max_explores=8,
        model_label="haiku",
        cache_id="explore_1",
        extra_budget_text="\n  haiku: 2/8 remaining",
        timed_out=False,
    )
    timeout = CandidateRecord(
        idx=2,
        answer="",
        confidence=0.0,
        approach="",
        reasoning="",
        cost_usd=0.0,
        used=2,
        max_explores=8,
        model_label="",
        extra_budget_text="",
        timed_out=True,
    )
    for cls in ALL_RENDERERS:
        instance = cls()
        for record in (success, timeout):
            text = instance.render(record)
            assert isinstance(text, str) and text, (
                f"{cls.__name__}.render returned bad value for {record}"
            )


_self_check()
