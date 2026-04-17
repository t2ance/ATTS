# Offline Per-Sample Shuffle + Cache ID Hiding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the per-sample `cached_explores` shuffle from online rollout (ExploreTool) to offline dataset construction (prepare_data_hle.py), expand training data 5× via unique per-qid permutations with encoded PermutationID, stop rendering `Cache ID` in tool responses, switch reward_fn to positional indexing, lower rollout temperature to 0.7, restart GRPO from SFT checkpoint.

**Architecture:** Each qid gets exactly 5 unique permutations of its 8 cached_explores, sampled without replacement from 8! = 40320 possibilities. The permutation is encoded into `question_id` as `{qid}#{i0}_{i1}_..._{i7}` (0-based indices), making it self-contained and reversible. Val stays at 1 permutation per qid. Online ExploreTool consumes the pre-shuffled list in-order. FullRenderer omits `- Cache ID:`. reward_fn indexes `cached_explores[N-1]["is_correct"]` by Candidate #N positional index.

**Tech Stack:** Python 3.11 (grpo_vllm conda env), verl 0.7 tool_agent_loop, itertools.permutations, pandas/pyarrow for parquet, pytest for unit tests.

---

## Background

### Why this plan exists

1. **Cache ID visibility problem** — `FullRenderer.render` emits `- Cache ID: explore_N` to the model. This is removed to prevent semantic-ID shortcut learning.
2. **Positional bias measured** — Across 132 training+val samples, cached_explores correctness varies by position: pos 4 = 42.4%, pos 7 = 29.5% (gap 12.9 pp, ~3σ). Without shuffle, removing Cache ID alone exposes a positional-semantic shortcut.
3. **GRPO variance is policy-side** — RC-GRPO / RAGEN establishes that intra-group variance comes from policy token sampling. The previous online per-rollout shuffle did not contribute to GRPO variance. It only served positional-bias prevention.
4. **5× offline permutation expansion** — Each qid gets 5 unique permutations (without replacement from 40320 options, collision probability 0.025%). Train 530 rows, val 26 rows. PermutationID encodes the shuffle for traceability and reversibility.

### Files affected

- `training/grpo/prepare_data_hle.py` — add helpers + 5× expansion in main()
- `training/grpo/explore_tool.py` — remove online shuffle, remove `import random`
- `methods/tool_io.py` — remove `- Cache ID:` from render; fix `_self_check` round-trip
- `training/grpo/reward_fn.py` — switch to positional-index grade lookup; delete `_extract_cached_explore_grades`
- `training/scripts/train_grpo_vllm_8b_sft_2gpu.sh` — temperature 1.0→0.7; drop resume_from_path
- `training_data/grpo/train.parquet` + `val.parquet` — regenerated (530 + 26 rows)

### Test strategy

Unit tests live in `tests/` (new dir). Three focused test files:
- `tests/test_tool_io_round_trip.py` — verifies render/parse round-trip with `cache_id=""`
- `tests/test_permutation_encoding.py` — verifies PermutationID encode/decode + 5-sample uniqueness + distribution
- `tests/test_reward_fn_positional_lookup.py` — verifies reward_fn positional-index path

Smoke test: one step of GRPO training from SFT checkpoint with regenerated parquet.

---

## File Structure

### Created

- `tests/__init__.py` — empty package marker
- `tests/test_tool_io_round_trip.py` — render/parse round-trip without Cache ID
- `tests/test_permutation_encoding.py` — PermutationID encode/decode, sampling uniqueness, positional uniformity
- `tests/test_reward_fn_positional_lookup.py` — reward_fn end-to-end with positional lookup

### Modified

- `training/grpo/prepare_data_hle.py` — add `build_permutation_id`, `split_permutation_id`, `_apply_permutation`, constants `TRAIN_PERMUTATIONS_PER_QID=5` / `VAL_PERMUTATIONS_PER_QID=1`; rewrite main() to filter→split→expand
- `methods/tool_io.py` — `FullRenderer.render` drops cache_line; `_self_check` success record uses `cache_id=""`
- `training/grpo/explore_tool.py` — remove `rng = random.Random(...); rng.shuffle(cached)`; remove `import random`
- `training/grpo/reward_fn.py` — `_extract_explore_tool_responses` returns `list[tuple[int, str]]`; delete `_extract_cached_explore_grades`; `compute_score` uses positional lookup
- `training/scripts/train_grpo_vllm_8b_sft_2gpu.sh` — `temperature=1.0` → `temperature=0.7`; remove `trainer.resume_mode` + `trainer.resume_from_path`

### Regenerated (data artifacts, not source)

- `training_data/grpo/train.parquet` — 530 rows
- `training_data/grpo/val.parquet` — 26 rows

---

## Task 1: Round-trip test fixture for tool_io

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_tool_io_round_trip.py`

- [ ] **Step 1: Create tests package marker**

```bash
touch /data3/peijia/dr-claw/Explain/Experiment/core_code/tests/__init__.py
```

- [ ] **Step 2: Write the failing round-trip test**

Create `tests/test_tool_io_round_trip.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from methods.tool_io import CandidateRecord, FullRenderer


def test_round_trip_success_record_drops_cache_id():
    renderer = FullRenderer()
    original = CandidateRecord(
        idx=3, answer="D", confidence=0.75,
        approach="geographic verification", reasoning="Big Bend coordinates match",
        cost_usd=0.04, used=3, max_explores=8, cache_id="explore_5",
    )
    text = renderer.render(original)
    assert "Cache ID" not in text, f"Cache ID leaked into render: {text!r}"
    parsed = renderer.parse(text)
    assert parsed.cache_id == "", f"expected cache_id='', got {parsed.cache_id!r}"
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
        idx=2, answer="", confidence=0.0, approach="", reasoning="",
        cost_usd=0.0, used=2, max_explores=8, cache_id="explore_7", timed_out=True,
    )
    text = renderer.render(original)
    assert "Cache ID" not in text
    parsed = renderer.parse(text)
    assert parsed.timed_out is True
    assert parsed.cache_id == ""
    assert parsed.idx == original.idx
    assert parsed.used == original.used
    assert parsed.max_explores == original.max_explores
```

- [ ] **Step 3: Run to confirm it fails (Cache ID still present)**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m pytest tests/test_tool_io_round_trip.py -v
```

Expected: FAIL on `assert "Cache ID" not in text`.

- [ ] **Step 4: Commit failing test**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add tests/__init__.py tests/test_tool_io_round_trip.py
git commit -m "test: add round-trip test for FullRenderer without Cache ID"
```

---

## Task 2: Remove Cache ID rendering from FullRenderer

**Files:**
- Modify: `methods/tool_io.py`

- [ ] **Step 1: Replace render() method — remove cache_line**

Edit `methods/tool_io.py`. Replace lines 118–139:

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
        # cache_id is intentionally NOT rendered. reward_fn recovers the
        # corresponding cached explore by Candidate #N positional index.
        # parse() returns cache_id="" by design.
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

- [ ] **Step 2: Fix _self_check — success record cache_id=""**

In `methods/tool_io.py` `_self_check()`, replace the success CandidateRecord:

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

- [ ] **Step 3: Verify import + self_check passes**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -c "from methods.tool_io import FullRenderer; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Run round-trip test — expect PASS**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m pytest tests/test_tool_io_round_trip.py -v
```

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add methods/tool_io.py
git commit -m "refactor: stop rendering Cache ID in FullRenderer output"
```

---

## Task 3: Remove online shuffle from ExploreTool

**Files:**
- Modify: `training/grpo/explore_tool.py`

- [ ] **Step 1: Drop shuffle lines and `import random`**

Edit `training/grpo/explore_tool.py`.

Remove line at top:
```python
import random
```

Replace the `if bucket is None:` block (lines 67–87):

OLD:
```python
        if bucket is None:
            create_kwargs = agent_data.tools_kwargs.get("explore", {}).get("create_kwargs", {})
            cached = list(create_kwargs["cached_explores"])
            max_explores = int(create_kwargs.get("max_explores", len(cached)))
            assert len(cached) >= max_explores, (
                f"cached_explores ({len(cached)}) must have at least "
                f"max_explores ({max_explores}) items"
            )
            rng = random.Random(agent_data.request_id)
            rng.shuffle(cached)
            bucket = {
                "cached_explores": tuple(cached),
                "explore_state": ExploreStepState(max_explores=max_explores),
            }
            self._rollout_state[key] = bucket
```

NEW:
```python
        if bucket is None:
            create_kwargs = agent_data.tools_kwargs.get("explore", {}).get("create_kwargs", {})
            cached = tuple(create_kwargs["cached_explores"])
            max_explores = int(create_kwargs.get("max_explores", len(cached)))
            assert len(cached) >= max_explores, (
                f"cached_explores ({len(cached)}) must have at least "
                f"max_explores ({max_explores}) items"
            )
            # Order fixed by prepare_data_hle.py offline shuffle (per-qid seed).
            # All n rollouts of the same sample consume the same pre-shuffled order,
            # matching reward_fn's positional-index grade lookup.
            bucket = {
                "cached_explores": cached,
                "explore_state": ExploreStepState(max_explores=max_explores),
            }
            self._rollout_state[key] = bucket
```

- [ ] **Step 2: Import check**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -c "from training.grpo import explore_tool; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add training/grpo/explore_tool.py
git commit -m "refactor: remove online per-rollout shuffle from ExploreTool"
```

---

## Task 4: Permutation encoding tests

**Files:**
- Create: `tests/test_permutation_encoding.py`

- [ ] **Step 1: Write tests**

Create `tests/test_permutation_encoding.py`:

```python
from __future__ import annotations

import itertools
import random
import sys
from collections import Counter
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from training.grpo.prepare_data_hle import (
    TRAIN_PERMUTATIONS_PER_QID,
    build_permutation_id,
    split_permutation_id,
)

N_EXPLORES = 8
ALL_PERMS = list(itertools.permutations(range(N_EXPLORES)))


def _sample_perms(qid: str, n: int) -> list[tuple[int, ...]]:
    return random.Random(f"train:{qid}").sample(ALL_PERMS, n)


def test_encode_decode_round_trip():
    perm = [5, 0, 7, 3, 1, 4, 2, 6]
    pid = build_permutation_id("hle_q0123", perm)
    assert pid == "hle_q0123#5_0_7_3_1_4_2_6"
    qid, decoded = split_permutation_id(pid)
    assert qid == "hle_q0123"
    assert decoded == perm


def test_qid_with_underscores_splits_cleanly():
    perm = [0, 1, 2, 3, 4, 5, 6, 7]
    pid = build_permutation_id("hle_q_0_1_2", perm)
    qid, decoded = split_permutation_id(pid)
    assert qid == "hle_q_0_1_2"
    assert decoded == perm


def test_5_samples_are_unique_per_qid():
    sampled = _sample_perms("hle_q0123", TRAIN_PERMUTATIONS_PER_QID)
    assert len(set(sampled)) == TRAIN_PERMUTATIONS_PER_QID, (
        f"duplicate permutations in {sampled}"
    )


def test_position0_uniform_across_800_qids():
    pos0 = Counter(
        _sample_perms(f"hle_q{i:04d}", 1)[0][0] for i in range(800)
    )
    for idx in range(N_EXPLORES):
        assert 60 <= pos0[idx] <= 140, f"pos0 bucket {idx}: {pos0[idx]}"
```

- [ ] **Step 2: Run tests — expect FAIL (prepare_data_hle not updated yet)**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m pytest tests/test_permutation_encoding.py -v
```

Expected: ImportError on `build_permutation_id` (not defined yet).

- [ ] **Step 3: Commit failing test**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add tests/test_permutation_encoding.py
git commit -m "test: lock in PermutationID encoding + 5-sample uniqueness + uniformity"
```

---

## Task 5: 5× permutation expansion in prepare_data_hle.py

**Files:**
- Modify: `training/grpo/prepare_data_hle.py`

- [ ] **Step 1: Add imports and constants**

At top of `training/grpo/prepare_data_hle.py`, replace the import block:

OLD:
```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
```

NEW:
```python
from __future__ import annotations

import copy
import itertools
import json
import random
import sys
from pathlib import Path

import pandas as pd
```

Add constants after the existing constants (after `OUT_DIR = ...`):

```python
TRAIN_PERMUTATIONS_PER_QID = 5
VAL_PERMUTATIONS_PER_QID = 1
```

- [ ] **Step 2: Add helper functions**

After the `load_cached_explores` function, add:

```python
def build_permutation_id(qid: str, perm: list[int]) -> str:
    return f"{qid}#{'_'.join(str(i) for i in perm)}"


def split_permutation_id(pid: str) -> tuple[str, list[int]]:
    qid, perm_str = pid.split("#", 1)
    return qid, [int(x) for x in perm_str.split("_")]


def _apply_permutation(base: dict, perm: list[int]) -> dict:
    new = copy.deepcopy(base)
    orig = new["extra_info"]["tools_kwargs"]["explore"]["create_kwargs"]["cached_explores"]
    new["extra_info"]["tools_kwargs"]["explore"]["create_kwargs"]["cached_explores"] = [orig[i] for i in perm]
    qid = new["extra_info"]["question_id"]
    new["extra_info"]["question_id"] = build_permutation_id(qid, perm)
    return new
```

- [ ] **Step 3: Rewrite main() — filter → split → expand**

Replace the body of `main()` from the split section onward. OLD (lines 137–173):

```python
    rows: list[dict] = []
    for r in pool:
        rows.append(build_row(r))
    assert len(rows) == NUM_TRAIN_POOL

    rows = [
        r for r in rows
        if any(e["is_correct"] for e in r["extra_info"]["tools_kwargs"]["explore"]["create_kwargs"]["cached_explores"])
    ]
    print(f"Informative filter: kept {len(rows)}/{NUM_TRAIN_POOL} (dropped {NUM_TRAIN_POOL - len(rows)})")
    assert len(rows) > 0, "no informative rows after filter"

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
```

NEW:
```python
    base_rows: list[dict] = []
    for r in pool:
        base_rows.append(build_row(r))
    assert len(base_rows) == NUM_TRAIN_POOL

    # Filter before permutation: qids with no correct explore produce no useful
    # reward signal regardless of ordering.
    base_rows = [
        r for r in base_rows
        if any(e["is_correct"] for e in r["extra_info"]["tools_kwargs"]["explore"]["create_kwargs"]["cached_explores"])
    ]
    print(f"Informative filter: kept {len(base_rows)}/{NUM_TRAIN_POOL}")
    assert len(base_rows) > 0, "no informative rows after filter"

    # Split at qid level before expanding. All permutations of the same qid
    # stay in the same split — no leakage.
    val_size = max(1, int(len(base_rows) * VAL_FRACTION))
    val_pool = base_rows[:val_size]
    train_pool = base_rows[val_size:]
    print(f"Split: train={len(train_pool)} qids, val={len(val_pool)} qids")

    all_permutations = list(itertools.permutations(range(MAX_EXPLORES)))
    assert len(all_permutations) == 40320

    def expand(pool: list[dict], n_perm: int, seed_ns: str) -> list[dict]:
        rows: list[dict] = []
        for base in pool:
            qid = base["extra_info"]["question_id"]
            sampled = random.Random(f"{seed_ns}:{qid}").sample(all_permutations, n_perm)
            for perm in sampled:
                rows.append(_apply_permutation(base, list(perm)))
        return rows

    train_rows = expand(train_pool, TRAIN_PERMUTATIONS_PER_QID, "train")
    val_rows = expand(val_pool, VAL_PERMUTATIONS_PER_QID, "val")
    print(f"After expansion: train={len(train_rows)} rows, val={len(val_rows)} rows")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train_rows), ("val", val_rows)]:
        df = pd.DataFrame(data)
        out = OUT_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"Wrote {out} ({len(df)} rows)")
```

- [ ] **Step 4: Update schema sanity check — question_id now contains #**

In main(), the schema sanity check `assert sample["data_source"] == "atts_hle"` etc. is fine. No change needed to the schema check block.

- [ ] **Step 5: Dry-import check**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -c "from training.grpo.prepare_data_hle import build_permutation_id, split_permutation_id, TRAIN_PERMUTATIONS_PER_QID; print('ok')"
```

Expected: `ok`

- [ ] **Step 6: Run encoding tests — expect PASS**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m pytest tests/test_permutation_encoding.py -v
```

Expected: all 4 PASS.

- [ ] **Step 7: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add training/grpo/prepare_data_hle.py
git commit -m "feat: 5x per-qid offline permutation expansion with PermutationID encoding"
```

---

## Task 6: Switch reward_fn to positional indexing

**Files:**
- Modify: `training/grpo/reward_fn.py`

- [ ] **Step 1: Replace `_extract_explore_tool_responses` signature**

OLD (lines 165–189):
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
        if not body.startswith("Candidate #"):
            continue
        rec = _RENDERER.parse(body)
        if rec.cache_id and rec.answer:
            responses.append((rec.cache_id, rec.answer))
    return responses
```

NEW:
```python
def _extract_explore_tool_responses(solution_str: str) -> list[tuple[int, str]]:
    """Extract explorer (idx, self-reported answer) pairs from each
    <tool_response>...</tool_response> block.

    Returns 1-based Candidate #N idx instead of cache_id. Caller indexes
    cached_explores[idx-1] directly (offline-shuffled order from parquet).
    Timeout candidates and empty-answer responses are skipped.
    """
    responses: list[tuple[int, str]] = []
    for m in _TOOL_RESPONSE_RE.finditer(solution_str):
        body = m.group(1).strip()
        if not body.startswith("Candidate #"):
            continue
        rec = _RENDERER.parse(body)
        if rec.answer:
            responses.append((rec.idx, rec.answer))
    return responses
```

- [ ] **Step 2: Delete `_extract_cached_explore_grades` entirely**

Remove lines 192–215 (the entire `_extract_cached_explore_grades` function).

- [ ] **Step 3: Rewrite compute_score's explore grading block**

OLD (lines 254–275):
```python
    question = ""
    if extra_info is not None:
        question = extra_info.get("question", "") or ""
    cached_explore_grades = _extract_cached_explore_grades(extra_info)

    solution_str = _strip_think_blocks(solution_str)

    final_answer = _extract_final_answer(solution_str)
    y_T = _judge_remote(final_answer, ground_truth, question) if final_answer else 0.0

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
    question = ""
    if extra_info is not None:
        question = extra_info.get("question", "") or ""

    tools_kwargs = (extra_info or {}).get("tools_kwargs") or {}
    explore_kwargs = ((tools_kwargs.get("explore") or {}).get("create_kwargs") or {})
    cached_explores_ordered = explore_kwargs.get("cached_explores") or []
    assert cached_explores_ordered, (
        "extra_info.tools_kwargs.explore.create_kwargs.cached_explores missing; "
        "explore grading requires offline-prepared cached explores"
    )

    solution_str = _strip_think_blocks(solution_str)

    final_answer = _extract_final_answer(solution_str)
    y_T = _judge_remote(final_answer, ground_truth, question) if final_answer else 0.0

    explore_answers = _extract_explore_tool_responses(solution_str)
    y_per_step = []
    for idx, answer in explore_answers:
        if idx < 1 or idx > len(cached_explores_ordered):
            raise ValueError(
                f"Candidate #{idx} out of range; "
                f"cached_explores has {len(cached_explores_ordered)} entries"
            )
        is_correct = cached_explores_ordered[idx - 1].get("is_correct")
        if not isinstance(is_correct, bool):
            raise ValueError(
                f"cached_explores[{idx-1}] missing bool is_correct: "
                f"{cached_explores_ordered[idx-1]}"
            )
        y_per_step.append(1.0 if (is_correct and answer) else 0.0)
    V_star_N = max(y_per_step) if y_per_step else 0.0
```

- [ ] **Step 4: Import + syntax check**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -c "from training.grpo.reward_fn import compute_score, _extract_explore_tool_responses; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add training/grpo/reward_fn.py
git commit -m "refactor: reward_fn uses positional index instead of cache_id lookup"
```

---

## Task 7: reward_fn positional lookup end-to-end test

**Files:**
- Create: `tests/test_reward_fn_positional_lookup.py`

- [ ] **Step 1: Write test**

Create `tests/test_reward_fn_positional_lookup.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from training.grpo import reward_fn


def _candidate_block(idx: int, used: int, max_explores: int, answer: str) -> str:
    return (
        f"<tool_response>\nCandidate #{idx} recorded.\n"
        f"- Answer: {answer}\n"
        f"- Confidence: 0.5\n"
        f"- Approach: approach\n"
        f"- Reasoning: reasoning\n"
        f"- Cost: $0.02\n\n"
        f"Explore budget: {used}/{max_explores} used, {max_explores - used} remaining."
        f"</tool_response>"
    )


def _struct_output(answer: str) -> str:
    return (
        '<tool_call>{"name": "StructuredOutput", "arguments": {'
        f'"answer": "{answer}", "approach": "a", "reasoning": "r", "confidence": 0.8'
        "}}</tool_call>"
    )


def test_positional_lookup_hits_correct():
    cached = [
        {"cache_id": "x0", "answer": "A", "approach": "a", "reasoning": "r", "is_correct": False},
        {"cache_id": "x1", "answer": "B", "approach": "b", "reasoning": "r", "is_correct": True},
        {"cache_id": "x2", "answer": "C", "approach": "c", "reasoning": "r", "is_correct": False},
        {"cache_id": "x3", "answer": "D", "approach": "d", "reasoning": "r", "is_correct": False},
        {"cache_id": "x4", "answer": "E", "approach": "e", "reasoning": "r", "is_correct": False},
        {"cache_id": "x5", "answer": "F", "approach": "f", "reasoning": "r", "is_correct": False},
        {"cache_id": "x6", "answer": "G", "approach": "g", "reasoning": "r", "is_correct": False},
        {"cache_id": "x7", "answer": "H", "approach": "h", "reasoning": "r", "is_correct": False},
    ]
    # Model uses candidates at position 1 (wrong) and 2 (correct)
    solution_str = _candidate_block(1, 1, 8, "A") + _candidate_block(2, 2, 8, "B") + _struct_output("B")
    extra_info = {
        "question": "dummy",
        "tools_kwargs": {"explore": {"create_kwargs": {"cached_explores": cached, "max_explores": 8}}},
    }
    with patch.object(reward_fn, "_judge_remote", return_value=1.0):
        out = reward_fn.compute_score("atts_hle", solution_str, "B", extra_info)
    assert out["discovery"] == 1.0, out
    assert out["acc"] == 1.0, out
    assert out["num_explores"] == 2.0, out


def test_out_of_range_idx_raises():
    cached = [{"cache_id": "x", "answer": "A", "approach": "a", "reasoning": "r", "is_correct": False}]
    solution_str = _candidate_block(5, 1, 1, "A") + _struct_output("A")
    extra_info = {"tools_kwargs": {"explore": {"create_kwargs": {"cached_explores": cached, "max_explores": 1}}}}
    with patch.object(reward_fn, "_judge_remote", return_value=0.0):
        try:
            reward_fn.compute_score("atts_hle", solution_str, "A", extra_info)
        except ValueError as e:
            assert "out of range" in str(e)
            return
    raise AssertionError("expected ValueError on out-of-range idx")
```

- [ ] **Step 2: Run test**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m pytest tests/test_reward_fn_positional_lookup.py -v
```

Expected: both PASS.

- [ ] **Step 3: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add tests/test_reward_fn_positional_lookup.py
git commit -m "test: reward_fn positional-index grade lookup end-to-end"
```

---

## Task 8: Regenerate parquet dataset

**Files:** (no code edit — runs prepare_data_hle.py)

- [ ] **Step 1: Back up current parquet**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p training_data/grpo/backup_pre_shuffle_20260417
cp training_data/grpo/train.parquet training_data/grpo/backup_pre_shuffle_20260417/
cp training_data/grpo/val.parquet training_data/grpo/backup_pre_shuffle_20260417/
```

- [ ] **Step 2: Run prepare_data_hle.py**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m training.grpo.prepare_data_hle
```

Expected output:
```
tool_config.yaml StructuredOutput schema matches EXPLORE_SCHEMA.
Gold text-only: N questions
Training pool [100:400]: 300 questions
Informative filter: kept ~132/300
Split: train=~106 qids, val=~26 qids
After expansion: train=~530 rows, val=~26 rows
Wrote .../train.parquet (~530 rows)
Wrote .../val.parquet (~26 rows)
Schema sanity check: OK
```

- [ ] **Step 3: Verify permutation IDs in parquet**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -c "
import pandas as pd
df = pd.read_parquet('training_data/grpo/train.parquet')
print('train rows:', len(df))
sample_ids = [row['extra_info']['question_id'] for _, row in df.head(3).iterrows()]
for qid in sample_ids:
    assert '#' in qid, f'missing PermutationID in {qid}'
    orig, perm = qid.split('#', 1)
    indices = list(map(int, perm.split('_')))
    assert sorted(indices) == list(range(8)), f'invalid perm {indices}'
    print(f'  {qid} -> valid')
print('val rows:', len(pd.read_parquet(\"training_data/grpo/val.parquet\")))
"
```

Expected: 3 valid PermutationIDs printed, val ~26 rows.

- [ ] **Step 4: Verify position-correctness uniformity (optional sanity)**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -c "
import pandas as pd, numpy as np
df = pd.concat([pd.read_parquet(f'training_data/grpo/{n}.parquet') for n in ('train','val')])
pos_correct = [[] for _ in range(8)]
for ei in df['extra_info']:
    for i, e in enumerate(ei['tools_kwargs']['explore']['create_kwargs']['cached_explores']):
        pos_correct[i].append(1.0 if e['is_correct'] else 0.0)
for i in range(8):
    print(f'pos {i}: {np.mean(pos_correct[i]):.3f}  (n={len(pos_correct[i])})')
"
```

Expected: all positions within ~5 pp of each other (no more 42.4% vs 29.5% spread).

- [ ] **Step 5: Commit new parquet**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add training_data/grpo/train.parquet training_data/grpo/val.parquet
git commit -m "data: regenerate parquet with 5x per-qid offline permutation expansion"
```

---

## Task 9: Update training script

**Files:**
- Modify: `training/scripts/train_grpo_vllm_8b_sft_2gpu.sh`

- [ ] **Step 1: Lower rollout temperature**

Edit `training/scripts/train_grpo_vllm_8b_sft_2gpu.sh`, line 106:

OLD:
```
    actor_rollout_ref.rollout.temperature=1.0 \
```

NEW:
```
    actor_rollout_ref.rollout.temperature=0.7 \
```

`val_kwargs.temperature=0.0` (line 107) stays unchanged.

- [ ] **Step 2: Drop resume_from_path lines**

Remove these two lines (near the bottom of the verl CLI flags):

```
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=/data3/peijia/dr-claw/Explain/Experiment/core_code/checkpoints/atts-grpo/8b-sft-2gpu-bs96/global_step_3
```

- [ ] **Step 3: Shell syntax check**

```bash
bash -n /data3/peijia/dr-claw/Explain/Experiment/core_code/training/scripts/train_grpo_vllm_8b_sft_2gpu.sh && echo ok
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git add training/scripts/train_grpo_vllm_8b_sft_2gpu.sh
git commit -m "chore: temperature 1.0->0.7; restart from SFT checkpoint (drop global_step_3 resume)"
```

---

## Task 10: Smoke test + launch training

- [ ] **Step 1: Run all unit tests**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
/home/peijia/miniconda3/envs/grpo_vllm/bin/python -m pytest tests/ -v
```

Expected: all 7 tests PASS (2 round_trip + 4 encoding + 2 reward_fn_positional).

- [ ] **Step 2: Check GPU availability**

```bash
nvidia-smi
```

Identify two free GPUs. Adjust `CUDA_VISIBLE_DEVICES` in script if needed.

- [ ] **Step 3: Launch training**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
bash training/scripts/train_grpo_vllm_8b_sft_2gpu.sh \
  > tmp/grpo_offline_shuffle_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 4: Monitor first 5 minutes**

```bash
tail -f tmp/grpo_offline_shuffle_*.log | grep -E "Error|Traceback|wandb.ai|global_step|val-core|ValueError"
```

Expected:
- wandb URL printed
- No `Cache ID` in any tool response
- No `_extract_cached_explore_grades` AttributeError
- `val_before_train` completes without reward_fn crash

- [ ] **Step 5: Post-smoke W&B check**

After `val_before_train` and first train step:
- `val-core/atts_hle/acc/mean@4` recorded
- `discovery` + `num_explores` reported per step

---

## Rollback plan

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
git log --oneline training/grpo/reward_fn.py methods/tool_io.py training/grpo/explore_tool.py training/grpo/prepare_data_hle.py training/scripts/train_grpo_vllm_8b_sft_2gpu.sh
# Revert tasks in reverse order:
git revert --no-commit <sha_task9> <sha_task8> ... <sha_task1>
git commit -m "revert: offline shuffle plan; restore previous online shuffle"
cp training_data/grpo/backup_pre_shuffle_20260417/*.parquet training_data/grpo/
```

---

## Self-review summary

- All 5 code files in File Structure are touched by Tasks 2, 3, 5, 6, 9.
- Parquet regen in Task 8. Tests in Tasks 1, 4, 7. Smoke in Task 10.
- Type consistency: `_extract_explore_tool_responses` returns `list[tuple[int, str]]` in Task 6; same in Task 7 test. `build_permutation_id` / `split_permutation_id` defined in Task 5, imported in Task 4 test — no drift.
- No placeholders.
