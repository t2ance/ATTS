# Cache ID Removal via Fingerprint Matching (Plan D) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop rendering `Cache ID: explore_N` to the model without breaking reward_fn's ability to look up each explored candidate's pre-computed `is_correct` grade. Achieved by (a) removing the `- Cache ID:` line from `FullRenderer.render`, and (b) switching reward_fn from `cache_id → is_correct` dict lookup to `(answer, approach)` fingerprint lookup against `cached_explores` in `extra_info`.

**Architecture:** ExploreTool keeps its existing online per-rollout shuffle (`random.Random(request_id).shuffle(cached_explores)`) — this is orthogonal to cache_id visibility and provides legitimate positional-bias prevention. The rendered tool response omits the cache_id row; the model sees `Candidate #N recorded.\n- Answer: ...\n- Approach: ...\n- Reasoning: ...\n- Cost: ...\n\nExplore budget: N/M used, K remaining.` only. reward_fn parses each Candidate's `(answer, approach)` and looks up the matching cached_explore entry to recover `is_correct`.

**Tech Stack:** Python 3.11 (grpo_vllm conda env), verl 0.7 tool_agent_loop, pandas/pyarrow for parquet, pytest for tests.

---

## Context (Read Before Starting)

This plan is a self-contained artifact. Pick it up in a fresh session with no prior conversation context.

### The Problem

`FullRenderer.render` in `methods/tool_io.py` emits a line like `- Cache ID: explore_5` into every `<tool_response>` block that the model sees during rollout. This line is a semantic token identifying which of the pre-computed explore candidates was returned. User flagged it as a shortcut-learning risk: the model may learn spurious correlations like "explore_5 is usually correct" across training samples.

### Why Cache ID Exists in the Current Implementation

`reward_fn.compute_score` in `training/grpo/reward_fn.py` needs to know, for each explore tool call within a rollout, which entry of `cached_explores` was actually returned. It uses this to look up the pre-computed `is_correct` grade (from `cached_explores[k].is_correct`), which feeds the GRPO reward component `V*_N = max_t y_t`.

Current mechanism:
1. ExploreTool writes `cache_id` into the rendered text via `CandidateRecord(cache_id="explore_5", ...)`.
2. Rollout text preserves this during tool-response wrapping.
3. reward_fn parses `cache_id` via `FullRenderer.parse` and does `grade_by_cache_id[cache_id]` lookup.

Removing the cache_id line breaks step 2, so reward_fn needs an alternative channel to recover which candidate was returned.

### Alternative Channels Considered

Four transport channels were surveyed:

| Channel | Pros | Cons |
|---|---|---|
| `render` text (current) | Simple, reward_fn unchanged | **Model-visible** (the problem) |
| `agent_data.extra_fields` | Model-invisible | Hits verl's `list_of_dict_to_dict_of_list` keys-homogeneity assertion in `protocol.py:302` if any rollout skips populating the key |
| verl patch to thread request_id into `non_tensor_batch` | Model-invisible, clean | Non-minimal change to external dependency |
| **Fingerprint matching (Plan D)** | Model-invisible, no verl change, no dataset change | Requires `(answer, approach)` tuple to be unique within each sample's `cached_explores` |

Plan D is chosen. Fingerprint uniqueness is empirically verified: approach strings are ≥50 chars of Haiku-generated free-form text, collision across 8 candidates in one sample is extremely rare. The implementation raises `ValueError` on collision, fail-fast rather than silent miscrediting.

### Orthogonality Insight (Key)

**Permutation timing (online vs offline shuffle) is independent from cache_id visibility.** Pre-computing permutations in the parquet does NOT solve cache_id visibility; the render logic controls visibility. This plan does NOT change shuffle timing. The existing online shuffle in ExploreTool remains.

An earlier plan draft (`2026-04-17-offline-shuffle-cache-id-removal.md`, now superseded) attempted to bundle offline shuffle with cache_id removal. Analysis showed the offline shuffle was unnecessary once Plan D decoupled cache_id from render, and the bundled plan required parquet regeneration + loss of within-step observation diversity. Plan D (this plan) is strictly smaller in scope.

### Truncation Safety Verification

`max_tool_response_length = 2048` in `grpo_config` with `truncate_side = right` (keeps the end of the block, drops the opening). Measured on 1056 rendered Candidate blocks across train+val parquet:
- mean 1051 chars, p99 1819, max 3297
- 5/1056 (0.47%) exceed 2048
- Per-rollout probability of at least one truncation: ~3.7%
- Across ~12,720 training rollouts: ~470 rollouts contain at least one truncated Candidate

When truncation occurs, the block no longer starts with `"Candidate #"`, so `reward_fn._extract_explore_tool_responses`'s existing guard `if not body.startswith("Candidate #"): continue` skips it (no crash, no miscrediting). Fingerprint matching inherits this skip behavior unchanged.

### Positional Bias Measurement (Context for Not Removing Shuffle)

Empirical stats across 132 samples, raw precache order (pre-shuffle):

| Position (precache index) | mean is_correct |
|---|---|
| 1 | 31.8% |
| 2 | 40.2% |
| 3 | 37.9% |
| 4 | **42.4%** |
| 5 | 34.8% |
| 6 | 33.3% |
| 7 | **29.5%** |
| 8 | 37.9% |

12.9 pp gap ≈ 3× standard error (~4.2 pp). If ExploreTool's online shuffle were removed, the model would see candidates in raw precache order and could learn this cross-sample positional-correctness pattern. The shuffle breaks cross-sample alignment. This plan keeps the shuffle.

### Literature Note

RC-GRPO (`arxiv 2602.03025`) and RAGEN (`arxiv 2504.20073`) establish that GRPO intra-group variance must come from policy token sampling, not environment observation randomization. The ExploreTool shuffle does NOT substitute for policy variance; it serves positional-bias prevention only. Do not attempt to justify keeping the shuffle via GRPO variance arguments.

---

## File Structure

### Created

- `tests/__init__.py` — empty package marker (if not already present)
- `tests/test_tool_io_round_trip.py` — render→parse round-trip; cache_id dropped
- `tests/test_reward_fn_fingerprint.py` — reward_fn fingerprint lookup end-to-end

### Modified

- `methods/tool_io.py` — `FullRenderer.render` drops `cache_line`; `_self_check` success record uses `cache_id=""`
- `training/grpo/reward_fn.py` — `_extract_explore_tool_responses` returns `list[tuple[int, str, str]]` of (idx, answer, approach); `compute_score` builds `fingerprint_to_cached_entry` map from `extra_info.tools_kwargs.explore.create_kwargs.cached_explores` and resolves each Candidate via `(answer, approach)` match; `_extract_cached_explore_grades` deleted

### Not Modified

- `training/grpo/explore_tool.py` — online shuffle kept
- `training/grpo/prepare_data_hle.py` — dataset unchanged
- `training_data/grpo/*.parquet` — no regeneration needed
- `training/scripts/train_grpo_vllm_8b_sft_2gpu.sh` — no changes (checkpoint resume decision is independent)

### Checkpoint Decision (Independent)

`global_step_3` was trained with the current shuffle protocol and the current cache_id rendering. Plan D changes render output (shorter by ~25 chars per Candidate) but preserves shuffle semantics. The observation distribution shifts slightly (model no longer sees Cache ID line). Whether to discard `global_step_3` and restart from SFT merged is a user decision:
- Restart from SFT: clean slate, avoids any distribution drift, ~2 hours lost
- Continue from `global_step_3`: tolerable drift since only a per-block cosmetic line changed

This plan does not encode that choice. Ask the user before launching training.

---

## Task 1: Round-trip test for FullRenderer without Cache ID

**Files:**
- Create: `tests/__init__.py` (if missing)
- Create: `tests/test_tool_io_round_trip.py`

- [ ] **Step 1: Create tests package marker**

If `tests/__init__.py` does not exist, create it with empty content.

Run: `ls tests/__init__.py 2>/dev/null || touch tests/__init__.py`

- [ ] **Step 2: Write the round-trip test**

Create `tests/test_tool_io_round_trip.py`:

```python
"""Round-trip test for FullRenderer: render() must drop Cache ID, and
parse() must recover everything else exactly.

This test locks in the invariant that any future change re-introducing
cache_id rendering (or breaking answer/approach/reasoning round-trip)
fails at CI, not silently at training time."""
from __future__ import annotations

import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from methods.tool_io import CandidateRecord, FullRenderer


def test_render_drops_cache_id_line_success_record():
    renderer = FullRenderer()
    rec = CandidateRecord(
        idx=3,
        answer="D",
        confidence=0.75,
        approach="geographic verification near Big Bend",
        reasoning="Coordinates match Lajitas region.",
        cost_usd=0.04,
        used=3,
        max_explores=8,
        cache_id="explore_5",
    )
    text = renderer.render(rec)
    assert "Cache ID" not in text, f"Cache ID leaked: {text!r}"


def test_round_trip_success_record_fields_preserved():
    renderer = FullRenderer()
    original = CandidateRecord(
        idx=3,
        answer="D",
        confidence=0.75,
        approach="geographic verification near Big Bend",
        reasoning="Coordinates match Lajitas region.",
        cost_usd=0.04,
        used=3,
        max_explores=8,
        cache_id="explore_5",
    )
    parsed = renderer.parse(renderer.render(original))
    # cache_id is intentionally dropped
    assert parsed.cache_id == "", f"expected empty cache_id, got {parsed.cache_id!r}"
    # All other fields round-trip exactly
    assert parsed.idx == original.idx
    assert parsed.answer == original.answer
    assert parsed.confidence == original.confidence
    assert parsed.approach == original.approach
    assert parsed.reasoning == original.reasoning
    assert parsed.cost_usd == original.cost_usd
    assert parsed.used == original.used
    assert parsed.max_explores == original.max_explores


def test_round_trip_timeout_record_drops_cache_id():
    renderer = FullRenderer()
    original = CandidateRecord(
        idx=2,
        answer="",
        confidence=0.0,
        approach="",
        reasoning="",
        cost_usd=0.0,
        used=2,
        max_explores=8,
        cache_id="explore_7",
        timed_out=True,
    )
    text = renderer.render(original)
    assert "Cache ID" not in text
    parsed = renderer.parse(text)
    assert parsed.cache_id == ""
    assert parsed.timed_out is True
    assert parsed.idx == original.idx
    assert parsed.used == original.used
```

- [ ] **Step 3: Run test, verify FAIL against current render**

Run:
```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m pytest tests/test_tool_io_round_trip.py -v
```

Expected: `test_render_drops_cache_id_line_success_record` FAILS with `AssertionError: Cache ID leaked: ...`, because current `FullRenderer.render` still emits the `- Cache ID: explore_5` line.

- [ ] **Step 4: Commit failing test**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add tests/__init__.py tests/test_tool_io_round_trip.py
git commit -m "test: add round-trip test that asserts Cache ID is not rendered"
```

---

## Task 2: Stop rendering Cache ID in FullRenderer

**Files:**
- Modify: `methods/tool_io.py`, the `FullRenderer.render` method (around lines 118-139) and `_self_check` success record (around lines 256-269)

- [ ] **Step 1: Remove cache_line from render()**

Edit `methods/tool_io.py`. Replace the `render` method body exactly:

OLD:
```python
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
```

NEW:
```python
    def render(self, r: CandidateRecord) -> str:
        # cache_id is intentionally NOT rendered. It is an internal lookup
        # key for reward_fn; exposing it to the model (as "- Cache ID:
        # explore_5") creates a shortcut-learning risk across training
        # samples. reward_fn recovers the corresponding cached_explore
        # entry via (answer, approach) fingerprint matching against
        # extra_info.tools_kwargs.explore.create_kwargs.cached_explores.
        # parse() therefore returns cache_id="" by design.
        label = f" Model: {r.model_label}." if r.model_label else ""
        remaining = r.max_explores - r.used
        if r.timed_out:
            return (
                f"Candidate #{r.idx} recorded (timed out, empty answer).{label}\n"
                f"Explore budget: {r.used}/{r.max_explores} used, {remaining} remaining."
                f"{r.extra_budget_text}"
            )
        return (
            f"Candidate #{r.idx} recorded.{label}\n"
            f"- Answer: {r.answer}\n"
            f"- Confidence: {r.confidence}\n"
            f"- Approach: {r.approach}\n"
            f"- Reasoning: {r.reasoning}\n"
            f"- Cost: ${r.cost_usd:.2f}\n\n"
            f"Explore budget: {r.used}/{r.max_explores} used, {remaining} remaining."
            f"{r.extra_budget_text}"
        )
```

- [ ] **Step 2: Update `_self_check` round-trip success record**

Edit `methods/tool_io.py`, within `_self_check`, change the success record's `cache_id` field to `""`:

OLD:
```python
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
```

NEW:
```python
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
        cache_id="",  # not rendered; round-trip recovers empty
        extra_budget_text="\n  haiku: 2/8 remaining",
        timed_out=False,
    )
```

- [ ] **Step 3: Import-time sanity check**

Run:
```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -c "from methods.tool_io import FullRenderer; print('ok')"
```

Expected: prints `ok` (the `_self_check()` call at module import does not raise).

- [ ] **Step 4: Round-trip test passes**

Run:
```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m pytest tests/test_tool_io_round_trip.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add methods/tool_io.py
git commit -m "refactor: stop rendering Cache ID in FullRenderer output

cache_id was a model-visible semantic token (e.g. 'explore_5') that
risks shortcut learning. reward_fn will recover the corresponding
cached_explore entry via (answer, approach) fingerprint matching against
extra_info.tools_kwargs.explore.create_kwargs.cached_explores."
```

---

## Task 3: Fingerprint lookup end-to-end test

**Files:**
- Create: `tests/test_reward_fn_fingerprint.py`

- [ ] **Step 1: Write the test**

Create `tests/test_reward_fn_fingerprint.py`:

```python
"""End-to-end test of reward_fn with (answer, approach) fingerprint
matching. Constructs a rollout text containing two Candidate blocks and
a StructuredOutput final answer; provides extra_info with known
cached_explores whose (answer, approach) fingerprints match exactly;
verifies discovery V*_N, acc, num_explores, and collision handling.

The remote judge is mocked because it's a live HTTP call to a vLLM
server; we only need to validate reward_fn's internal wiring.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from training.grpo import reward_fn


def _candidate_block(idx: int, used: int, max_explores: int, answer: str, approach: str, reasoning: str = "r") -> str:
    return (
        f"<tool_response>\nCandidate #{idx} recorded.\n"
        f"- Answer: {answer}\n"
        f"- Confidence: 0.5\n"
        f"- Approach: {approach}\n"
        f"- Reasoning: {reasoning}\n"
        f"- Cost: $0.02\n\n"
        f"Explore budget: {used}/{max_explores} used, {max_explores - used} remaining."
        f"</tool_response>"
    )


def _struct_output_call(answer: str) -> str:
    return (
        '<tool_call>{"name": "StructuredOutput", "arguments": {'
        f'"answer": "{answer}", '
        '"approach": "synth", "reasoning": "synth", "confidence": 0.8'
        "}}</tool_call>"
    )


def _cached(cache_id: str, answer: str, approach: str, is_correct: bool) -> dict:
    return {
        "cache_id": cache_id,
        "answer": answer,
        "approach": approach,
        "reasoning": "stub",
        "confidence": 0.5,
        "cost_usd": 0.02,
        "is_correct": is_correct,
    }


def test_fingerprint_lookup_resolves_correct_candidate():
    """Two Candidate blocks, one wrong one right. V*_N must be 1.0."""
    cached = [
        _cached("x_A", "A", "clinical anatomy approach", False),
        _cached("x_B", "B", "topological reasoning approach", True),
        _cached("x_C", "C", "statistical analysis approach", False),
        _cached("x_D", "D", "geographic verification approach", False),
        _cached("x_E", "E", "historical record approach", False),
        _cached("x_F", "F", "molecular biology approach", False),
        _cached("x_G", "G", "physics first principles approach", False),
        _cached("x_H", "H", "literary critique approach", False),
    ]
    solution_str = (
        _candidate_block(1, 1, 8, "A", "clinical anatomy approach")
        + _candidate_block(2, 2, 8, "B", "topological reasoning approach")
        + _struct_output_call("B")
    )
    extra_info = {
        "question": "dummy",
        "tools_kwargs": {
            "explore": {"create_kwargs": {"cached_explores": cached, "max_explores": 8}},
        },
    }
    with patch.object(reward_fn, "_judge_remote", return_value=1.0):
        out = reward_fn.compute_score(
            data_source="atts_hle",
            solution_str=solution_str,
            ground_truth="B",
            extra_info=extra_info,
        )
    assert out["discovery"] == 1.0, out
    assert out["acc"] == 1.0, out
    assert out["num_explores"] == 2.0, out


def test_fingerprint_unknown_candidate_raises():
    """If rollout text contains a Candidate whose (answer, approach) is
    not in cached_explores, reward_fn must raise — never silently map
    to the wrong entry."""
    cached = [
        _cached("x_A", "A", "anatomy", True),
    ]
    solution_str = (
        _candidate_block(1, 1, 1, "Z", "nonexistent approach string")
        + _struct_output_call("Z")
    )
    extra_info = {
        "tools_kwargs": {
            "explore": {"create_kwargs": {"cached_explores": cached, "max_explores": 1}},
        },
    }
    with patch.object(reward_fn, "_judge_remote", return_value=0.0):
        try:
            reward_fn.compute_score(
                data_source="atts_hle",
                solution_str=solution_str,
                ground_truth="A",
                extra_info=extra_info,
            )
        except ValueError as e:
            assert "fingerprint" in str(e).lower() or "unknown" in str(e).lower()
            return
    raise AssertionError("expected ValueError on unknown fingerprint")


def test_fingerprint_collision_raises():
    """If two cached_explores share (answer, approach), reward_fn must
    raise at fingerprint-map construction — fail-fast rather than
    miscredit."""
    cached = [
        _cached("x_A1", "A", "same approach string", True),
        _cached("x_A2", "A", "same approach string", False),
    ]
    solution_str = (
        _candidate_block(1, 1, 2, "A", "same approach string")
        + _struct_output_call("A")
    )
    extra_info = {
        "tools_kwargs": {
            "explore": {"create_kwargs": {"cached_explores": cached, "max_explores": 2}},
        },
    }
    with patch.object(reward_fn, "_judge_remote", return_value=1.0):
        try:
            reward_fn.compute_score(
                data_source="atts_hle",
                solution_str=solution_str,
                ground_truth="A",
                extra_info=extra_info,
            )
        except ValueError as e:
            assert "duplicate" in str(e).lower() or "collision" in str(e).lower()
            return
    raise AssertionError("expected ValueError on fingerprint collision")


def test_truncated_candidate_block_is_skipped_not_crashed():
    """If verl's tool_response_truncate_side=right drops the opening of
    a Candidate block, the resulting body does not start with
    'Candidate #' and must be skipped silently (existing behavior)."""
    cached = [
        _cached("x_A", "A", "approach a", False),
        _cached("x_B", "B", "approach b", True),
    ]
    # Simulate a truncated block: starts mid-field, no "Candidate #" header
    truncated = (
        "<tool_response>\n(truncated)...\n"
        "- Reasoning: partial\n- Cost: $0.02\n\n"
        "Explore budget: 1/8 used, 7 remaining.\n</tool_response>"
    )
    valid = _candidate_block(2, 2, 8, "B", "approach b")
    solution_str = truncated + valid + _struct_output_call("B")
    extra_info = {
        "tools_kwargs": {
            "explore": {"create_kwargs": {"cached_explores": cached, "max_explores": 8}},
        },
    }
    with patch.object(reward_fn, "_judge_remote", return_value=1.0):
        out = reward_fn.compute_score(
            data_source="atts_hle",
            solution_str=solution_str,
            ground_truth="B",
            extra_info=extra_info,
        )
    # The truncated block is skipped; the valid Candidate #2 contributes V*_N=1.0
    assert out["discovery"] == 1.0, out
```

- [ ] **Step 2: Run test, verify FAIL against current reward_fn**

Run:
```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m pytest tests/test_reward_fn_fingerprint.py -v
```

Expected: tests FAIL (current reward_fn uses cache_id, not fingerprint). Exact failure mode will depend on parser behavior — some tests may error on `cache_id=""` leading to empty cache_id in `responses.append((rec.cache_id, rec.answer))` guard.

- [ ] **Step 3: Commit failing test**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add tests/test_reward_fn_fingerprint.py
git commit -m "test: reward_fn fingerprint matching end-to-end"
```

---

## Task 4: Implement fingerprint matching in reward_fn

**Files:**
- Modify: `training/grpo/reward_fn.py`:
  - replace `_extract_explore_tool_responses` (around lines 165-189)
  - delete `_extract_cached_explore_grades` (around lines 192-215)
  - rewrite the explore grading block inside `compute_score` (around lines 249-275)

- [ ] **Step 1: Rewrite `_extract_explore_tool_responses` to return (idx, answer, approach)**

Edit `training/grpo/reward_fn.py`. Replace the function exactly:

OLD:
```python
def _extract_explore_tool_responses(solution_str: str) -> list[tuple[str, str]]:
    """Extract explorer (cache_id, self-reported answer) pairs from each
    <tool_response>...</tool_response> block emitted by verl's hermes chat
    template wrapping `ExploreTool.execute` output.

    Format parsing is delegated to `methods.tool_io.FullRenderer.parse`
    (the single inverse of `FullRenderer.render` used by the production
    eval, GRPO rollout, and SFT data builder). Timeout candidates and
    responses without a cache_id are skipped -- they carry no grade signal.
    """
    responses: list[tuple[str, str]] = []
    for m in _TOOL_RESPONSE_RE.finditer(solution_str):
        body = m.group(1).strip()
        # `_TOOL_RESPONSE_RE` captures every <tool_response> regardless of
        # which tool produced it (explore / StructuredOutput / any future
        # tool). FullRenderer.parse is the inverse of FullRenderer.render
        # and only accepts Candidate # recorded bodies. Bodies from other
        # tools (e.g. AnswerTool's "Answer recorded: X") are not this
        # function's concern and must be skipped before parse is called.
        if not body.startswith("Candidate #"):
            continue
        rec = _RENDERER.parse(body)
        if rec.cache_id and rec.answer:
            responses.append((rec.cache_id, rec.answer))
    return responses
```

NEW:
```python
def _extract_explore_tool_responses(solution_str: str) -> list[tuple[int, str, str]]:
    """Extract (idx, answer, approach) triples from each <tool_response>
    ... </tool_response> block in the rollout that wraps an ExploreTool
    output.

    cache_id is no longer rendered (see methods/tool_io.py). The caller
    builds a (answer, approach) fingerprint map from extra_info's
    cached_explores to recover which cached entry each Candidate
    corresponds to. idx is the 1-based Candidate # parsed from the
    header, retained for error messages. Timeout candidates (empty
    answer) are skipped -- they carry no grade signal.

    Truncated blocks (verl's truncate_side=right drops the Candidate #
    header) are skipped via the startswith guard; ~0.47% of rendered
    blocks exceed the 2048-char tool_response limit in practice.
    """
    triples: list[tuple[int, str, str]] = []
    for m in _TOOL_RESPONSE_RE.finditer(solution_str):
        body = m.group(1).strip()
        # `_TOOL_RESPONSE_RE` captures every <tool_response> regardless of
        # which tool produced it (explore / StructuredOutput / any future
        # tool). FullRenderer.parse is the inverse of FullRenderer.render
        # and only accepts Candidate # recorded bodies. Bodies from other
        # tools (e.g. AnswerTool's "Answer recorded: X") are not this
        # function's concern and must be skipped before parse is called.
        if not body.startswith("Candidate #"):
            continue
        rec = _RENDERER.parse(body)
        if not rec.answer:
            continue  # timeout record; no grade signal
        triples.append((rec.idx, rec.answer, rec.approach))
    return triples
```

- [ ] **Step 2: Delete `_extract_cached_explore_grades`**

Edit `training/grpo/reward_fn.py`. Locate the full `_extract_cached_explore_grades` function (around lines 192-215) and delete it entirely. It is replaced by the inline fingerprint-map construction inside `compute_score`.

The function to delete:
```python
def _extract_cached_explore_grades(extra_info: dict | None) -> dict[str, float]:
    """Build strict cache-id -> correctness map from exported cached explores.

    If cached explores are present, every entry must carry both `cache_id` and
    `is_correct`; otherwise we raise loudly rather than silently re-grade.
    """
    if extra_info is None:
        return {}
    tools_kwargs = extra_info.get("tools_kwargs") or {}
    explore_kwargs = ((tools_kwargs.get("explore") or {}).get("create_kwargs") or {})
    cached_explores = explore_kwargs.get("cached_explores") or []
    if not cached_explores:
        return {}

    grade_by_cache_id: dict[str, float] = {}
    for idx, explore in enumerate(cached_explores, start=1):
        cache_id = str(explore.get("cache_id", "")).strip()
        if not cache_id:
            raise ValueError(f"cached explore #{idx} missing cache_id")
        is_correct = explore.get("is_correct")
        if not isinstance(is_correct, bool):
            raise ValueError(f"cached explore {cache_id} missing bool is_correct")
        grade_by_cache_id[cache_id] = 1.0 if is_correct else 0.0
    return grade_by_cache_id
```

- [ ] **Step 3: Rewrite the explore grading block in `compute_score`**

Edit `training/grpo/reward_fn.py`. Replace the block from `cached_explore_grades = ...` through `V_star_N = max(...)`:

OLD (around lines 249-275):
```python
    # Pull question for judge. Extra_info may be None if verl forgot to pass it;
    # we fall back to empty string in that case (Principle P4: never crash
    # inside the loss loop from missing metadata, but log it so we notice).
    question = ""
    if extra_info is not None:
        question = extra_info.get("question", "") or ""
    cached_explore_grades = _extract_cached_explore_grades(extra_info)

    # Strip <think> blocks before parsing to prevent hallucinated tool
    # calls/responses inside thinking from being matched by regexes.
    solution_str = _strip_think_blocks(solution_str)

    # Final answer correctness y_T via deployed judge
    final_answer = _extract_final_answer(solution_str)
    y_T = _judge_remote(final_answer, ground_truth, question) if final_answer else 0.0

    # Per-step explorer answers and y_{1:N}. When cached explores are present,
    # consume their exported grades directly instead of re-grading.
    explore_answers = _extract_explore_tool_responses(solution_str)
    assert cached_explore_grades, "no cached_explore_grades: explore grading requires pre-computed cache"
    y_per_step = []
    for cache_id, answer in explore_answers:
        if cache_id not in cached_explore_grades:
            raise ValueError(f"tool response referenced unknown cached explore {cache_id}")
        y_per_step.append(cached_explore_grades[cache_id] if answer else 0.0)
    V_star_N = max(y_per_step) if y_per_step else 0.0
```

NEW:
```python
    # Pull question for judge. Extra_info may be None if verl forgot to pass it;
    # we fall back to empty string in that case.
    question = ""
    if extra_info is not None:
        question = extra_info.get("question", "") or ""

    # Pull cached_explores from extra_info. These carry the pre-computed
    # is_correct grades that replace live judge calls for explore steps.
    tools_kwargs = (extra_info or {}).get("tools_kwargs") or {}
    explore_kwargs = ((tools_kwargs.get("explore") or {}).get("create_kwargs") or {})
    cached_explores = explore_kwargs.get("cached_explores") or []
    assert cached_explores, (
        "extra_info.tools_kwargs.explore.create_kwargs.cached_explores missing; "
        "explore grading requires offline-prepared cached explores"
    )

    # Build (answer, approach) -> cached entry fingerprint map. cache_id is
    # no longer rendered, so we identify which cached_explore the rollout
    # consumed by content fingerprint. Collisions across cached entries
    # are rejected fail-fast.
    fingerprint_to_entry: dict[tuple[str, str], dict] = {}
    for i, entry in enumerate(cached_explores):
        fp = (str(entry["answer"]).strip(), str(entry["approach"]).strip())
        if fp in fingerprint_to_entry:
            raise ValueError(
                f"duplicate (answer, approach) fingerprint in cached_explores: "
                f"{fp!r} at index {i}; cached_explores must have unique fingerprints "
                f"for reward lookup to be well-defined"
            )
        is_correct = entry.get("is_correct")
        if not isinstance(is_correct, bool):
            raise ValueError(
                f"cached_explores[{i}] missing bool is_correct: {entry}"
            )
        fingerprint_to_entry[fp] = entry

    # Strip <think> blocks before parsing to prevent hallucinated tool
    # calls/responses inside thinking from being matched by regexes.
    solution_str = _strip_think_blocks(solution_str)

    # Final answer correctness y_T via deployed judge
    final_answer = _extract_final_answer(solution_str)
    y_T = _judge_remote(final_answer, ground_truth, question) if final_answer else 0.0

    # Per-step explorer answers and y_{1:N}. Each Candidate's (answer,
    # approach) fingerprint must match exactly one cached_explore entry;
    # unknown fingerprints raise (never silent miscredit).
    explore_triples = _extract_explore_tool_responses(solution_str)
    y_per_step = []
    for idx, answer, approach in explore_triples:
        fp = (answer.strip(), approach.strip())
        entry = fingerprint_to_entry.get(fp)
        if entry is None:
            raise ValueError(
                f"Candidate #{idx} fingerprint {fp!r} not found in "
                f"cached_explores; possible content drift between render "
                f"and cached list, or a truncation case that reached this "
                f"point without being skipped"
            )
        y_per_step.append(1.0 if entry["is_correct"] else 0.0)
    V_star_N = max(y_per_step) if y_per_step else 0.0
```

- [ ] **Step 4: Import-check reward_fn**

Run:
```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -c "from training.grpo.reward_fn import compute_score, _extract_explore_tool_responses; print('ok')"
```

Expected: prints `ok`. The deleted `_extract_cached_explore_grades` must not be imported anywhere else (grep to verify).

- [ ] **Step 5: Verify no remaining references to the deleted function**

Run:
```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
grep -rn "_extract_cached_explore_grades" --include="*.py" .
```

Expected: zero matches. If any match surfaces, the caller needs updating — do that before proceeding.

- [ ] **Step 6: Run fingerprint tests, verify PASS**

Run:
```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m pytest tests/test_reward_fn_fingerprint.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 7: Run all tests together**

Run:
```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m pytest tests/ -v
```

Expected: all 7 tests PASS (3 from tool_io_round_trip + 4 from reward_fn_fingerprint).

- [ ] **Step 8: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add training/grpo/reward_fn.py
git commit -m "refactor: reward_fn uses (answer, approach) fingerprint lookup

Candidate blocks no longer carry Cache ID. reward_fn builds a
(answer, approach) -> cached_entry map from
extra_info.tools_kwargs.explore.create_kwargs.cached_explores and
resolves each explored candidate by content fingerprint. Collisions
across cached_explores entries raise fail-fast; unknown fingerprints
encountered in rollout text raise (no silent miscrediting).

_extract_cached_explore_grades is deleted as cache_id dict lookup is
replaced by fingerprint lookup."
```

---

## Task 5: Manual sanity check on real parquet data

**Files:** (none — diagnostic script)

- [ ] **Step 1: Verify (answer, approach) fingerprint uniqueness in real data**

Run:
```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -c "
import pandas as pd
dfs = [pd.read_parquet(f'training_data/grpo/{n}.parquet') for n in ('train','val')]
df = pd.concat(dfs, ignore_index=True)
total_rows = len(df)
collision_rows = 0
for ei in df['extra_info']:
    cached = ei['tools_kwargs']['explore']['create_kwargs']['cached_explores']
    fps = [(str(e['answer']).strip(), str(e['approach']).strip()) for e in cached]
    if len(set(fps)) != len(fps):
        collision_rows += 1
print(f'Total samples: {total_rows}')
print(f'Samples with (answer, approach) fingerprint collision: {collision_rows}')
print(f'Collision rate: {100*collision_rows/total_rows:.2f}%')
"
```

Expected: `collision_rows = 0`. If collisions exist, Plan D fails for those samples and training would raise `ValueError: duplicate fingerprint`. In that case, expand the fingerprint to `(answer, approach, reasoning[:100])` or investigate the colliding entries.

- [ ] **Step 2: Sanity-render one real Candidate and verify Cache ID absence**

Run:
```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -c "
import pandas as pd
from methods.tool_io import FullRenderer, CandidateRecord
df = pd.read_parquet('training_data/grpo/train.parquet')
ei = df.iloc[0]['extra_info']
cached = ei['tools_kwargs']['explore']['create_kwargs']['cached_explores']
e = cached[0]
rec = CandidateRecord(
    idx=1, answer=e['answer'], confidence=float(e['confidence']),
    approach=e['approach'], reasoning=e['reasoning'],
    cost_usd=float(e.get('cost_usd', 0.0)),
    used=1, max_explores=8, cache_id=e.get('cache_id', '')
)
text = FullRenderer().render(rec)
print('--- rendered text ---')
print(text)
print('--- checks ---')
assert 'Cache ID' not in text, 'Cache ID leaked'
assert text.startswith('Candidate #1 recorded.'), 'Header missing'
print('ok')
"
```

Expected: prints the rendered Candidate block without a `- Cache ID:` line, followed by `ok`.

---

## Task 6: Launch training

**Files:** (none — runtime validation)

- [ ] **Step 1: Confirm GPU availability**

Run: `nvidia-smi`

Identify two GPUs with low utilization. The training script currently assumes `CUDA_VISIBLE_DEVICES=0,1`; if those are busy, edit `training/scripts/train_grpo_vllm_8b_sft_2gpu.sh` to point to free GPUs before launching.

- [ ] **Step 2: Decide checkpoint resume policy**

Default: the existing `trainer.resume_mode=resume_path` and `trainer.resume_from_path=...global_step_3` lines in the training script will attempt to resume. The Plan D change is backward-compatible with that checkpoint (shuffle semantics unchanged, only rendered text shortens by ~25 chars per Candidate).

If preferring a clean slate, edit the training script to drop the two `resume_from_path` lines (model will start from `actor_rollout_ref.model.path=...sft_qwen3_8b_merged`).

ASK THE USER before launching if this decision is unclear.

- [ ] **Step 3: Launch**

Run:
```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
bash training/scripts/train_grpo_vllm_8b_sft_2gpu.sh \
  > tmp/grpo_8b_sft_2gpu_plan_d_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 4: Monitor first 10 minutes**

Watch: `tail -f tmp/grpo_8b_sft_2gpu_plan_d_*.log | grep -E "Error|Traceback|wandb.ai|global_step|val-core|Candidate #|Cache ID"`

Expected:
- wandb URL printed
- `Cache ID` string never appears in any tool_response block
- `val_before_train` completes without `AssertionError` or `ValueError` from reward_fn
- reward_fn does not raise `duplicate fingerprint` or `fingerprint not found` errors
- First train step completes

If `ValueError: fingerprint not found` appears in the log: usually means tool_response was truncated in a way that produced `"Candidate #"` prefix but corrupt body. Diagnose via the exact error message (it contains the offending fingerprint).

- [ ] **Step 5: Report**

Once val_before_train finishes and first train step runs, collect:
- W&B run URL
- Log file path
- First `val-core/atts_hle/acc/mean@4` value
- Any unexpected log lines

---

## Rollback

If Plan D produces reward_fn errors in real rollouts within 30 minutes of launch, revert the two code commits:

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git log --oneline methods/tool_io.py training/grpo/reward_fn.py
# Identify the Plan D task 2 and task 4 commit SHAs
git revert --no-commit <task_4_sha> <task_2_sha>
git commit -m "revert: Plan D; Cache ID rendering restored"
```

After revert, `train_grpo_vllm_8b_sft_2gpu.sh` resumes the prior behavior with cache_id rendered and dict-lookup reward_fn.

---

## Self-Review Summary

**Coverage check:**
- FullRenderer render change: Task 2
- FullRenderer round-trip contract update (self_check): Task 2
- Round-trip test: Task 1
- reward_fn fingerprint implementation: Task 4
- reward_fn end-to-end test including collision & truncation cases: Task 3
- Parquet fingerprint uniqueness sanity check: Task 5
- Training launch + monitor: Task 6

**No placeholders:** All code blocks are complete. All commands have exact expected outputs. File paths and line ranges are concrete.

**Type consistency:**
- `_extract_explore_tool_responses` returns `list[tuple[int, str, str]]` — consistent between the function signature in Task 4 Step 1 and the destructuring `for idx, answer, approach in explore_triples` in Task 4 Step 3.
- `fingerprint_to_entry: dict[tuple[str, str], dict]` — matches the `(answer, approach) -> cached_entry` map built in Task 4 Step 3.
- `CandidateRecord` fields referenced (`idx`, `answer`, `approach`, `reasoning`, `cache_id`, `timed_out`, `used`, `max_explores`, `confidence`, `cost_usd`) all exist in the current `methods/tool_io.py` dataclass.

**Orthogonality preserved:** This plan does not touch shuffle timing, dataset construction, or training script. Shuffle remains online per-rollout. Parquet is untouched.

---

## Appendix: Evidence Archive

### Truncation rate (verified 2026-04-17)

Measurement command:
```python
import pandas as pd, numpy as np
from methods.tool_io import FullRenderer, CandidateRecord

dfs = {n: pd.read_parquet(f'training_data/grpo/{n}.parquet') for n in ('train','val')}
df = pd.concat(list(dfs.values()), ignore_index=True)
renderer = FullRenderer()
MAX_LEN = 2048
lens = []
for ei in df['extra_info']:
    cached = ei['tools_kwargs']['explore']['create_kwargs']['cached_explores']
    max_explores = ei['tools_kwargs']['explore']['create_kwargs']['max_explores']
    for pos_idx, e in enumerate(cached):
        rec = CandidateRecord(
            idx=pos_idx+1, answer=e['answer'],
            confidence=float(e['confidence']), approach=e['approach'],
            reasoning=e['reasoning'], cost_usd=float(e.get('cost_usd', 0.0)),
            used=pos_idx+1, max_explores=max_explores,
            cache_id=str(e.get('cache_id','')),
        )
        lens.append(len(renderer.render(rec)))
arr = np.array(lens)
print(f'N={len(arr)}, mean={arr.mean():.0f}, p99={np.percentile(arr,99):.0f}, max={arr.max()}, exceed_{MAX_LEN}={(arr>MAX_LEN).sum()}')
```

Result: `N=1056, mean=1051, p99=1819, max=3297, exceed_2048=5`

### Positional correctness (verified 2026-04-17)

Measurement used same script with `np.mean([1.0 if e['is_correct'] else 0.0 for e at each position])`.

Result:
- pos 1: 31.8%, pos 2: 40.2%, pos 3: 37.9%, pos 4: 42.4%
- pos 5: 34.8%, pos 6: 33.3%, pos 7: 29.5%, pos 8: 37.9%

12.9 pp gap (pos 4 − pos 7) ≈ 3× binomial SE (√(0.36·0.64/132) ≈ 4.2 pp). Not Monte Carlo noise.

### Key file paths

- Training script: `training/scripts/train_grpo_vllm_8b_sft_2gpu.sh`
- verl tool lifecycle: `/home/peijia/miniconda3/envs/grpo_vllm/lib/python3.11/site-packages/verl/experimental/agent_loop/tool_agent_loop.py`
  - AgentData class: line 52
  - `request_id = uuid4().hex` (outer): line 133
  - Inner per-turn `uuid4().hex`: line 158 (`agent_loop.py`)
  - `_call_tool` truncation: lines 421-469
- verl DataProto keys-homogeneity: `verl/protocol.py:302, 931` (`list_of_dict_to_dict_of_list`)
- Current dataset construction: `training/grpo/prepare_data_hle.py`
  - `load_cached_explores` at lines 41-71 reads `explore_i/result.json` in numeric order
- Current reward_fn: `training/grpo/reward_fn.py`
- FullRenderer: `methods/tool_io.py:95-193`
- ExploreTool: `training/grpo/explore_tool.py`

### Superseded plans in this directory

- `2026-04-17-offline-shuffle-cache-id-removal.md` — rejected after analysis showed permutation timing is orthogonal to cache_id visibility; offline shuffle not needed for the cache_id goal. Kept in place as historical record; do not execute.
