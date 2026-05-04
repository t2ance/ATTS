"""Single source of truth for explore-loop state accumulation.

Before this module existed, the explore-step counter was hand-written in 7
places across eval (`methods/tts_agent*.py`, `methods/self_refine.py`,
`methods/budget_forcing.py`), GRPO training rollout (`training/grpo/
explore_tool.py`), and SFT data construction (`training/sft/build_sft_hle_
q101_300_thinking.py`). Three of the call sites used `state.current_iteration
+= 1`, one used `state.current_iteration = i`, one used `state["call_count"]
+= 1`, and one used `enumerate`. The GRPO path had a real bug: verl's
`tool_agent_loop.py:_call_tool()` calls `tool.create()` around every tool
invocation, wiping our per-instance counter and rendering `Explore budget:
1/8 used, 7 remaining` on every call regardless of actual progress.

This file defines the canonical state and its ONLY legal transition. By
making the state frozen (`@dataclass(frozen=True)`), any hand-written
`state.call_count += 1` becomes a `FrozenInstanceError` — the bug class is
physically impossible to re-introduce.

Design is symmetric to `methods.tool_io`:
- `tool_io.FullRenderer`  : single source for string rendering
- `tool_state.advance`    : single source for state transition
- `CandidateRecord`       : glue between the two layers

Both the displayed budget line (UI) and the candidate selected by the
caller (behavior) must derive from the same post-advance `ExploreStepState`;
any divergence is caught by the import-time `_self_check()` below and by
`tests/test_ui_behavior_parity.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ExploreStepState:
    """Canonical explore-loop state. Immutable by construction.

    Invariants (enforced by `__post_init__`):
    - `max_explores >= 1`
    - `0 <= call_count <= max_explores`

    The ONLY legal transition is `advance(state, label=...)`. `frozen=True`
    makes hand-written `state.call_count += 1` raise `FrozenInstanceError`,
    which structurally prevents the UI/behavior divergence bug class
    where rendered text and candidate-picking read from different counters.

    Fields:
    - `max_explores`: hard cap on explore calls per rollout/question.
    - `call_count`: number of advances completed so far.
    - `variant_call_counts`: per-label in-variant counter for the unified
      TTSAgentSpec multi/effort paths (label -> int). Empty dict for
      length-1 (single-variant) runs; in that case the existing
      `call_count` indexing of cache_keys still applies.
    - `variant_caps`: per-label budget cap, populated at solver init from
      each ExploreVariant.num_explores. Used by variant_exhausted().

    Mutability convention: although Python's `frozen=True` does not stop
    in-place dict mutation, the project convention is that BOTH dicts are
    only ever replaced (not mutated) by `advance()`. Treat them as
    structurally immutable. The same convention that protects call_count
    against hand-bumping protects the per-variant counters.
    """

    max_explores: int
    call_count: int = 0
    variant_call_counts: dict[str, int] = field(default_factory=dict)
    variant_caps: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.max_explores >= 1, (
            f"max_explores must be >= 1, got {self.max_explores}"
        )
        assert 0 <= self.call_count <= self.max_explores, (
            f"call_count must be in [0, {self.max_explores}], "
            f"got {self.call_count}"
        )

    @property
    def used(self) -> int:
        """1-based count of completed advances. Read AFTER `advance()`.

        Before any advance, `used == 0` (zero explores done). After the
        N-th advance, `used == N`, matching the `N` in the rendered
        "Explore budget: N/M used" line.
        """
        return self.call_count

    @property
    def remaining(self) -> int:
        return self.max_explores - self.call_count

    @property
    def is_exhausted(self) -> bool:
        return self.call_count >= self.max_explores

    def variant_exhausted(self, label: str) -> bool:
        """Per-variant budget guard for unified multi/effort runs."""
        used = self.variant_call_counts.get(label, 0)
        cap = self.variant_caps.get(label, 0)
        return used >= cap


def advance(state: ExploreStepState, label: str | None = None) -> ExploreStepState:
    """Advance the explore-step counter by exactly one.

    Single source of truth for explore-state transition. All paths
    (eval runtime, GRPO rollout tool, SFT data builder) MUST call this
    function; none may hand-increment or hand-assign any counter.

    Postconditions:
    - returned.call_count == state.call_count + 1
    - returned.max_explores == state.max_explores
    - if label is None: variant_call_counts unchanged
    - if label is set: variant_call_counts[label] == prev + 1, others unchanged

    Precondition:
    - state.is_exhausted is False (asserts otherwise)
    """
    assert not state.is_exhausted, (
        f"explore budget exhausted: "
        f"call_count={state.call_count}, max={state.max_explores}"
    )
    if label is None:
        return replace(state, call_count=state.call_count + 1)
    new_counts = {
        **state.variant_call_counts,
        label: state.variant_call_counts.get(label, 0) + 1,
    }
    return replace(
        state,
        call_count=state.call_count + 1,
        variant_call_counts=new_counts,
    )


def _self_check() -> None:
    """Module-import-time sanity check.

    Verifies the transition semantics, frozen guarantee, and boundary
    behavior. If any of these break, import fails immediately rather than
    silently producing wrong training signals.
    """
    # Basic advance + derived properties
    s0 = ExploreStepState(max_explores=8)
    assert (s0.used, s0.remaining, s0.is_exhausted) == (0, 8, False)
    s1 = advance(s0)
    assert (s1.used, s1.remaining, s1.is_exhausted) == (1, 7, False)
    s2 = advance(s1)
    assert (s2.used, s2.remaining, s2.is_exhausted) == (2, 6, False)

    # Frozen: hand-mutation MUST fail at runtime
    try:
        s2.call_count = 999  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("ExploreStepState must be frozen (call_count)")
    try:
        s2.max_explores = 16  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("ExploreStepState must be frozen (max_explores)")

    # Advance-to-exhaustion
    s = s2
    for _ in range(6):
        s = advance(s)
    assert s.is_exhausted
    assert s.used == 8
    assert s.remaining == 0

    # Advancing past the cap must fail
    try:
        advance(s)
    except AssertionError:
        pass
    else:
        raise AssertionError("advance past max_explores must assert")

    # Constructor invariants
    try:
        ExploreStepState(max_explores=0)
    except AssertionError:
        pass
    else:
        raise AssertionError("max_explores=0 must be rejected")
    try:
        ExploreStepState(max_explores=8, call_count=9)
    except AssertionError:
        pass
    else:
        raise AssertionError("call_count > max_explores must be rejected")
    try:
        ExploreStepState(max_explores=8, call_count=-1)
    except AssertionError:
        pass
    else:
        raise AssertionError("negative call_count must be rejected")

    # Per-variant counters (unified TTSAgentSpec path)
    sm = ExploreStepState(
        max_explores=24,
        variant_caps={"haiku": 8, "sonnet": 8, "opus": 8},
    )
    assert not sm.variant_exhausted("haiku")
    assert sm.variant_call_counts == {}
    sm = advance(sm, label="haiku")
    assert sm.call_count == 1
    assert sm.variant_call_counts == {"haiku": 1}
    sm = advance(sm, label="haiku")
    sm = advance(sm, label="sonnet")
    assert sm.variant_call_counts == {"haiku": 2, "sonnet": 1}
    assert sm.call_count == 3
    # Exhaust haiku and verify variant_exhausted flips
    for _ in range(6):
        sm = advance(sm, label="haiku")
    assert sm.variant_exhausted("haiku")
    assert not sm.variant_exhausted("sonnet")


_self_check()
