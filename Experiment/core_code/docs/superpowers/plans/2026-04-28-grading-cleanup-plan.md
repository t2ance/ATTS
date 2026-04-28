# Grading Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move each benchmark's grading logic into its own `grade()` method that calls `grader.py` primitives directly. Delete `BenchmarkConfig.grade` dispatcher, `get_answer_type` abstraction, `_JUDGE_MODEL_CODEX` class constant. Add `grading_summary` class attribute printed in run startup banner so silent judge_model regressions (e.g. 2026-04-11 HLE incident) become impossible.

**Architecture:** Each benchmark's `grade()` is a 1-7 line function that picks one of three `grader.py` primitives (`check_answer`, `judge_answer`, `grade_code`) and calls it with that benchmark's specific decisions inlined. The `BenchmarkConfig` base class stops carrying any benchmark-specific dispatch logic. `eval.py:_grade_with_cache` contract preserved (still calls `benchmark.grade()` and reads `benchmark.judge_model` for cache invalidation).

**Tech Stack:** Python 3.11+, pytest, no new dependencies.

**Verified preconditions (grep confirmed 2026-04-28):**
- `BenchmarkConfig.grade` dispatcher at `benchmarks/base.py:379-400`
- `_JUDGE_MODEL_CODEX = {"claude-haiku-4-5-20251001": "gpt-5-codex-mini"}` at `benchmarks/base.py:377`
- `judge_model` already declared per-benchmark: lcb.py:91 (None), aime.py:72 (None), gpqa.py:58 (None), hle.py:123 (Haiku), babyvision.py:50 (Haiku), rbenchv.py:40 (Haiku)
- `get_answer_type` overridden in: babyvision.py:89, hle.py:148, gpqa.py:103
- `eval.py` reads `benchmark.judge_model` at lines 52, 99, 252, 295 → must preserve as class attr
- Codex backend grading triggers only in 2 scripts: `scripts/hle/gpt5.2_low/run_delegated.sh`, `scripts/hle/gpt5.4/run_no_integrate.sh` (other codex scripts are precache-only or use `judge_model=None` benchmarks)
- LCB and AIME already override `grade()` correctly — untouched by this plan

---

## File Structure

| File | Change |
|---|---|
| `benchmarks/base.py` | Remove: `grade()` (379-400), `get_answer_type()` default (372-374), `_JUDGE_MODEL_CODEX` (377), `judge_model = None` default (line where it appears in class body). Keep: `judge_model` declared as `judge_model: str | None` annotation only. |
| `benchmarks/gpqa.py` | Add: `grade()` override + `grading_summary`. Remove: `get_answer_type()` override (103-104). |
| `benchmarks/hle.py` | Add: `grade()` override (with codex bridge inline) + `grading_summary`. Remove: `get_answer_type()` override (148-149). |
| `benchmarks/babyvision.py` | Add: `grade()` override (hybrid) + `grading_summary`. Remove: `get_answer_type()` override (89-90). |
| `benchmarks/rbenchv.py` | Add: `grade()` override + `grading_summary`. |
| `benchmarks/lcb.py` | Add: `grading_summary`. (`grade()` already exists.) |
| `benchmarks/aime.py` | Add: `grading_summary`. (`grade()` already exists.) |
| `benchmarks/grader.py` | UNCHANGED (user-locked). `grade_answer` becomes 0-caller after Task 7 but is intentionally not deleted. |
| `eval.py` | Add 1 line printing `Grading: {benchmark.grading_summary}` in startup banner. |
| `tests/test_benchmark_grade.py` | NEW. Unit tests for each benchmark's new `grade()`, mocking `judge_answer` for deterministic verification. |

---

## Task 1: Regression Test Scaffold

**Files:**
- Create: `tests/test_benchmark_grade.py`

- [ ] **Step 1.1: Create the test file with deterministic check_answer cases**

```python
# tests/test_benchmark_grade.py
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from benchmarks.gpqa import GPQABenchmark
from benchmarks.hle import HLEBenchmark
from benchmarks.babyvision import BabyVisionBenchmark
from benchmarks.rbenchv import RBenchVBenchmark
from benchmarks.lcb import LCBBenchmark
from benchmarks.aime import AIMEBenchmark


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---- GPQA: pure string match on multipleChoice letter ----

def test_gpqa_grade_multiplechoice_correct():
    bench = GPQABenchmark()
    is_correct, cost = _run(bench.grade("c", "c", "q", {}, backend="claude"))
    assert is_correct is True
    assert cost == 0.0


def test_gpqa_grade_multiplechoice_wrong():
    bench = GPQABenchmark()
    is_correct, cost = _run(bench.grade("a", "c", "q", {}, backend="claude"))
    assert is_correct is False
    assert cost == 0.0


# ---- BabyVision: hybrid (choice -> string match, blank -> LLM judge) ----

def test_babyvision_choice_row_uses_string_match():
    bench = BabyVisionBenchmark()
    row = {"ansType": "choice"}
    is_correct, cost = _run(bench.grade("b", "b", "q", row, backend="claude"))
    assert is_correct is True
    assert cost == 0.0


def test_babyvision_blank_row_uses_judge_answer():
    bench = BabyVisionBenchmark()
    row = {"ansType": "blank"}
    with patch("benchmarks.babyvision.judge_answer", new=AsyncMock(return_value=(True, 0.001))) as m:
        is_correct, cost = _run(bench.grade("foo", "bar", "q", row, backend="claude"))
    assert is_correct is True
    assert cost == 0.001
    args, kwargs = m.call_args
    # judge_answer(predicted, gold, question, model, backend=..., out_dir=...)
    assert args[0] == "foo"
    assert args[1] == "bar"
    assert args[3] == "claude-haiku-4-5-20251001"
    assert kwargs["backend"] == "claude"


# ---- HLE: judge with codex bridge inline ----

def test_hle_grade_claude_backend_uses_haiku():
    bench = HLEBenchmark()
    row = {"answer_type": "exactMatch"}
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=(True, 0.002))) as m:
        is_correct, cost = _run(bench.grade("x", "y", "q", row, backend="claude"))
    args, _ = m.call_args
    assert args[3] == "claude-haiku-4-5-20251001"
    assert is_correct is True


def test_hle_grade_codex_backend_remaps_to_gpt5_codex_mini():
    bench = HLEBenchmark()
    row = {"answer_type": "exactMatch"}
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=(False, 0.003))) as m:
        is_correct, cost = _run(bench.grade("x", "y", "q", row, backend="codex"))
    args, kwargs = m.call_args
    assert args[3] == "gpt-5-codex-mini"
    assert kwargs["backend"] == "codex"


def test_hle_grade_vllm_backend_routes_to_claude():
    bench = HLEBenchmark()
    row = {"answer_type": "exactMatch"}
    with patch("benchmarks.hle.judge_answer", new=AsyncMock(return_value=(True, 0.0))) as m:
        _run(bench.grade("x", "y", "q", row, backend="vllm"))
    args, kwargs = m.call_args
    assert kwargs["backend"] == "claude"
    assert args[3] == "claude-haiku-4-5-20251001"


def test_hle_grade_multiplechoice_row_uses_string_match():
    bench = HLEBenchmark()
    row = {"answer_type": "multipleChoice"}
    is_correct, cost = _run(bench.grade("c", "c", "q", row, backend="claude"))
    assert is_correct is True
    assert cost == 0.0


# ---- RBenchV: always LLM judge ----

def test_rbenchv_uses_judge_answer():
    bench = RBenchVBenchmark()
    with patch("benchmarks.rbenchv.judge_answer", new=AsyncMock(return_value=(True, 0.004))) as m:
        is_correct, cost = _run(bench.grade("p", "g", "q", {}, backend="claude"))
    args, kwargs = m.call_args
    assert args[3] == "claude-haiku-4-5-20251001"
    assert kwargs["backend"] == "claude"
    assert is_correct is True


# ---- grading_summary class attr present on all 6 benchmarks ----

def test_all_benchmarks_have_grading_summary():
    for cls in (LCBBenchmark, AIMEBenchmark, GPQABenchmark, HLEBenchmark,
                BabyVisionBenchmark, RBenchVBenchmark):
        assert hasattr(cls, "grading_summary"), f"{cls.__name__} missing grading_summary"
        assert isinstance(cls.grading_summary, str)
        assert len(cls.grading_summary) > 10


# ---- judge_model class attr preserved (cache invalidation) ----

def test_judge_model_preserved_per_benchmark():
    assert LCBBenchmark.judge_model is None
    assert AIMEBenchmark.judge_model is None
    assert GPQABenchmark.judge_model is None
    assert HLEBenchmark.judge_model == "claude-haiku-4-5-20251001"
    assert BabyVisionBenchmark.judge_model == "claude-haiku-4-5-20251001"
    assert RBenchVBenchmark.judge_model == "claude-haiku-4-5-20251001"
```

- [ ] **Step 1.2: Run the tests; expect ALL tests to fail (no grade overrides yet, no grading_summary attr)**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
python -m pytest tests/test_benchmark_grade.py -v
```

Expected: most tests FAIL with `AttributeError` (grading_summary missing) or `AssertionError` (current `BenchmarkConfig.grade` calls `grader.grade_answer` not `judge_answer` directly).

- [ ] **Step 1.3: Commit the failing tests**

```bash
git add tests/test_benchmark_grade.py
git commit -m "test: regression suite for per-benchmark grade() refactor"
```

---

## Task 2: Add `grading_summary` Class Attribute to All 6 Benchmarks

**Files:**
- Modify: `benchmarks/lcb.py`, `benchmarks/aime.py`, `benchmarks/gpqa.py`, `benchmarks/hle.py`, `benchmarks/babyvision.py`, `benchmarks/rbenchv.py`

- [ ] **Step 2.1: Add `grading_summary` after `judge_model` declaration in each file**

`benchmarks/lcb.py` — add after `judge_model = None` (line 91):
```python
    grading_summary = "code execution (lcb_runner test cases)"
```

`benchmarks/aime.py` — add after `judge_model = None` (line 72):
```python
    grading_summary = "string match (integer normalize, modulo 1000)"
```

`benchmarks/gpqa.py` — add after `judge_model = None` (line 58):
```python
    grading_summary = "string match (multipleChoice letter A-E)"
```

`benchmarks/hle.py` — add after `judge_model = "claude-haiku-4-5-20251001"` (line 123):
```python
    grading_summary = (
        "LLM judge: claude-haiku-4-5-20251001 "
        "(codex backend remaps to gpt-5-codex-mini); "
        "multipleChoice rows fall through to string match"
    )
```

`benchmarks/babyvision.py` — add after `judge_model = "claude-haiku-4-5-20251001"` (line 50):
```python
    grading_summary = (
        "hybrid: string match for ansType=choice rows; "
        "LLM judge claude-haiku-4-5-20251001 for ansType=blank rows"
    )
```

`benchmarks/rbenchv.py` — add after `judge_model = "claude-haiku-4-5-20251001"` (line 40):
```python
    grading_summary = "LLM judge: claude-haiku-4-5-20251001"
```

- [ ] **Step 2.2: Run the grading_summary unit test; expect PASS**

```bash
python -m pytest tests/test_benchmark_grade.py::test_all_benchmarks_have_grading_summary -v
```

Expected: PASS.

- [ ] **Step 2.3: Commit**

```bash
git add benchmarks/lcb.py benchmarks/aime.py benchmarks/gpqa.py benchmarks/hle.py benchmarks/babyvision.py benchmarks/rbenchv.py
git commit -m "feat: add grading_summary class attr to each benchmark"
```

---

## Task 3: Print `grading_summary` in eval.py Startup Banner

**Files:**
- Modify: `eval.py:262`

- [ ] **Step 3.1: Insert one line into the existing banner block**

Find the banner at `eval.py:260-266`:
```python
    print(f"\n{'=' * 60}")
    print(f"{benchmark.name.upper()} Evaluation")
    print(f"Backend: {backend} | Orchestrator: {orchestrator_model} | Explorer: {explore_model} | Integrator: {integrate_model}")
    print(f"Questions to run: {len(pending)} ({len(done_records)} already completed, {total} total)")
    print(f"Max iterations per question: {num_explores} | Workers: {num_workers}")
    print(f"Logs:   {logger.run_dir}")
    print(f"{'=' * 60}\n")
```

After the `Backend: ...` line, add:
```python
    print(f"Grading: {benchmark.grading_summary}")
```

- [ ] **Step 3.2: Smoke verify banner output**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
python -c "
from benchmarks.hle import HLEBenchmark
print('Grading:', HLEBenchmark.grading_summary)
"
```

Expected stdout:
```
Grading: LLM judge: claude-haiku-4-5-20251001 (codex backend remaps to gpt-5-codex-mini); multipleChoice rows fall through to string match
```

- [ ] **Step 3.3: Commit**

```bash
git add eval.py
git commit -m "feat: print grading mode in run startup banner"
```

---

## Task 4: GPQA — Override `grade()`, Remove `get_answer_type`

**Files:**
- Modify: `benchmarks/gpqa.py`

- [ ] **Step 4.1: Add import + `grade()` method, remove `get_answer_type()` override**

Top of `benchmarks/gpqa.py` — add to existing imports:
```python
from benchmarks.grader import check_answer
```

Inside `GPQABenchmark` class — replace `get_answer_type` (lines 103-104):
```python
    # OLD
    def get_answer_type(self, row: dict) -> str:
        return "multipleChoice"
```

With:
```python
    async def grade(self, predicted, gold, question, row, backend, out_dir=None):
        return check_answer(predicted, gold, "multipleChoice"), 0.0
```

- [ ] **Step 4.2: Run GPQA tests**

```bash
python -m pytest tests/test_benchmark_grade.py::test_gpqa_grade_multiplechoice_correct tests/test_benchmark_grade.py::test_gpqa_grade_multiplechoice_wrong -v
```

Expected: PASS PASS.

- [ ] **Step 4.3: Commit**

```bash
git add benchmarks/gpqa.py
git commit -m "refactor: GPQA.grade calls check_answer directly"
```

---

## Task 5: RBenchV — Override `grade()`

**Files:**
- Modify: `benchmarks/rbenchv.py`

- [ ] **Step 5.1: Add import + `grade()` method**

Top of `benchmarks/rbenchv.py` — add to existing imports:
```python
from benchmarks.grader import judge_answer
```

Inside `RBenchVBenchmark` class — add method:
```python
    async def grade(self, predicted, gold, question, row, backend, out_dir=None):
        grade_backend = "claude" if backend == "vllm" else backend
        return await judge_answer(
            predicted, gold, question, self.judge_model,
            backend=grade_backend, out_dir=out_dir,
        )
```

- [ ] **Step 5.2: Run RBenchV test**

```bash
python -m pytest tests/test_benchmark_grade.py::test_rbenchv_uses_judge_answer -v
```

Expected: PASS.

- [ ] **Step 5.3: Commit**

```bash
git add benchmarks/rbenchv.py
git commit -m "refactor: RBenchV.grade calls judge_answer directly"
```

---

## Task 6: BabyVision — Override `grade()` (Hybrid), Remove `get_answer_type`

**Files:**
- Modify: `benchmarks/babyvision.py`

- [ ] **Step 6.1: Add imports + `grade()` method, remove `get_answer_type()` override**

Top of `benchmarks/babyvision.py` — add to existing imports:
```python
from benchmarks.grader import check_answer, judge_answer
```

Inside `BabyVisionBenchmark` class — replace `get_answer_type` (lines 89-90):
```python
    # OLD
    def get_answer_type(self, row: dict) -> str:
        return "multipleChoice" if row.get("ansType") == "choice" else "exactMatch"
```

With:
```python
    async def grade(self, predicted, gold, question, row, backend, out_dir=None):
        if row.get("ansType") == "choice":
            return check_answer(predicted, gold, "multipleChoice"), 0.0
        grade_backend = "claude" if backend == "vllm" else backend
        return await judge_answer(
            predicted, gold, question, self.judge_model,
            backend=grade_backend, out_dir=out_dir,
        )
```

- [ ] **Step 6.2: Run BabyVision tests**

```bash
python -m pytest tests/test_benchmark_grade.py::test_babyvision_choice_row_uses_string_match tests/test_benchmark_grade.py::test_babyvision_blank_row_uses_judge_answer -v
```

Expected: PASS PASS.

- [ ] **Step 6.3: Commit**

```bash
git add benchmarks/babyvision.py
git commit -m "refactor: BabyVision.grade does hybrid dispatch directly"
```

---

## Task 7: HLE — Override `grade()` (Codex Bridge Inline), Remove `get_answer_type`

**Files:**
- Modify: `benchmarks/hle.py`

- [ ] **Step 7.1: Add imports + `grade()` method, remove `get_answer_type()` override**

Top of `benchmarks/hle.py` — add to existing imports:
```python
from benchmarks.grader import check_answer, judge_answer
```

Inside `HLEBenchmark` class — replace `get_answer_type` (lines 148-149):
```python
    # OLD
    def get_answer_type(self, row: dict) -> str:
        return row.get("answer_type", "exactMatch")
```

With:
```python
    async def grade(self, predicted, gold, question, row, backend, out_dir=None):
        # HLE rows carry per-row answer_type; multipleChoice rows skip the LLM judge.
        answer_type = row.get("answer_type", "exactMatch")
        if answer_type == "multipleChoice":
            return check_answer(predicted, gold, "multipleChoice"), 0.0
        # vLLM serves the orchestrator, not the judge -> route judge through Claude.
        grade_backend = "claude" if backend == "vllm" else backend
        # Codex-backend HLE eval scripts (gpt5.2_low, gpt5.4) need a GPT judge model.
        judge_model = "gpt-5-codex-mini" if grade_backend == "codex" else self.judge_model
        return await judge_answer(
            predicted, gold, question, judge_model,
            backend=grade_backend, out_dir=out_dir,
        )
```

- [ ] **Step 7.2: Run HLE tests**

```bash
python -m pytest tests/test_benchmark_grade.py -k "hle" -v
```

Expected: 4 tests PASS (`test_hle_grade_claude_backend_uses_haiku`, `test_hle_grade_codex_backend_remaps_to_gpt5_codex_mini`, `test_hle_grade_vllm_backend_routes_to_claude`, `test_hle_grade_multiplechoice_row_uses_string_match`).

- [ ] **Step 7.3: Commit**

```bash
git add benchmarks/hle.py
git commit -m "refactor: HLE.grade inlines codex bridge and answer_type dispatch"
```

---

## Task 8: Demolish `BenchmarkConfig.grade` Dispatcher and Helpers

**Files:**
- Modify: `benchmarks/base.py`

This is the demolition step — only safe after Tasks 4-7 are committed.

- [ ] **Step 8.1: Pre-flight grep — confirm zero external readers of removed members**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
grep -rn "_JUDGE_MODEL_CODEX\|get_answer_type" --include="*.py" . | grep -v "tests/"
```

Expected output: empty (after Tasks 4-7, no remaining references). If anything appears, STOP — that consumer must be migrated first.

- [ ] **Step 8.2: Remove the dispatcher and helpers from `benchmarks/base.py`**

In `benchmarks/base.py`:

1. Remove the `judge_model: str | None` line that has the `= None` default (keep only the type annotation `judge_model: str | None`). Find the class body around line 333 and change:
   ```python
   judge_model: str | None
   ```
   (annotation only — no default value, since each subclass sets it explicitly).

2. Delete lines 372-374 (the `get_answer_type` default method).

3. Delete line 377 (the `_JUDGE_MODEL_CODEX = ...` class constant).

4. Delete lines 379-400 (the `BenchmarkConfig.grade` async method, including its docstring and `from benchmarks.grader import grade_answer` inner import).

After removal, `BenchmarkConfig` should still have: `normalize_answer`, `get_explore_schema`, `get_integrate_schema`, `get_explorer_system_prompt`, `get_integrator_system_prompt`, `build_explorer_message`, `build_integrator_message`, `get_answer_from_explore`, `get_answer_from_integrate`, `add_dataset_args`, `add_model_args`, `compute_metrics`, `print_metrics`, `_print_multi_model_metrics`, `save_plots`. **Do not touch any of those.**

- [ ] **Step 8.3: Run the full test suite**

```bash
python -m pytest tests/test_benchmark_grade.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 8.4: Import smoke — make sure no benchmark file imports a removed symbol**

```bash
python -c "
from benchmarks.lcb import LCBBenchmark
from benchmarks.aime import AIMEBenchmark
from benchmarks.gpqa import GPQABenchmark
from benchmarks.hle import HLEBenchmark
from benchmarks.babyvision import BabyVisionBenchmark
from benchmarks.rbenchv import RBenchVBenchmark
print('all benchmarks importable')
"
```

Expected: `all benchmarks importable` with no traceback.

- [ ] **Step 8.5: Commit**

```bash
git add benchmarks/base.py
git commit -m "refactor: remove BenchmarkConfig.grade dispatcher and codex map"
```

---

## Task 9: End-to-End Cache-Reuse Smoke

**Files:**
- No code changes; this task verifies the refactor preserved `eval.py:_grade_with_cache` behavior on real cached `grade.json` archives.

- [ ] **Step 9.1: Inspect a known-good GPQA grade.json**

```bash
cat /data3/peijia/dr-claw/Explain/Experiment/analysis/run/gpqa/sonnet_no_integrate_run4/run_20260328_060344/grading/recWXwn9v4IG9ZrM6/grade.json
```

Note: `judge_model: "none"` (because GPQA uses string match) and the `is_correct` value.

- [ ] **Step 9.2: Replay the cached `(predicted, gold)` through new GPQA.grade()**

Run inline:
```bash
python -c "
import asyncio, json
from benchmarks.gpqa import GPQABenchmark

p = '/data3/peijia/dr-claw/Explain/Experiment/analysis/run/gpqa/sonnet_no_integrate_run4/run_20260328_060344/grading/recWXwn9v4IG9ZrM6/grade.json'
d = json.load(open(p))
bench = GPQABenchmark()
ic, cost = asyncio.run(bench.grade(d['predicted'], d['gold'], 'q', {}, backend='claude'))
print(f'cached={d[\"is_correct\"]}, replay={ic}, match={d[\"is_correct\"] == ic}')
"
```

Expected: `match=True`. If False — STOP, the GPQA grade() does not match the prior dispatcher behavior.

- [ ] **Step 9.3: Verify the `judge_model` cache key still invalidates correctly**

```bash
python -c "
from benchmarks.hle import HLEBenchmark
from benchmarks.gpqa import GPQABenchmark
print('HLE.judge_model =', HLEBenchmark.judge_model)
print('GPQA.judge_model =', GPQABenchmark.judge_model)
"
```

Expected:
```
HLE.judge_model = claude-haiku-4-5-20251001
GPQA.judge_model = None
```

These must match the `judge_model` field stored in old `grade.json` files (HLE: `"claude-haiku-4-5-20251001"`, GPQA: `"none"`) so the existing cache continues to be trusted.

- [ ] **Step 9.4: Visual banner check on a one-question dry run**

Pick the smallest benchmark (AIME 2025) and run a single-question eval to verify the banner prints `Grading:`:

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
python eval.py --benchmark aime2025 \
    --backend claude --method socratic-self-refine \
    --num 1 --seed 42 --num-explores 1 --num-workers 1 \
    --orchestrator-model claude-sonnet-4-6 \
    --explore-model claude-sonnet-4-6 \
    --integrate-model claude-sonnet-4-6 \
    --no-cache-only \
    --log-dir ./tmp/grading_cleanup_smoke 2>&1 | head -20
```

Expected: banner contains a line `Grading: string match (integer normalize, modulo 1000)`.

- [ ] **Step 9.5: No commit needed (verification only). Mark task complete.**

---

## Self-Review (run before handoff)

**Spec coverage:**
- [x] Each benchmark grade() self-contained — Tasks 4-7
- [x] Delete BenchmarkConfig.grade dispatcher — Task 8
- [x] Delete get_answer_type abstraction — Tasks 4, 6, 7, 8
- [x] Delete _JUDGE_MODEL_CODEX class constant — Task 8 (after migration in Task 7)
- [x] judge_model class attribute preserved per-benchmark for cache invalidation — Task 8.1 grep gate + Task 9.3 verification
- [x] grading_summary class attr + banner — Tasks 2, 3
- [x] Codex HLE scripts continue to work — Task 7 inlines the bridge; Task 1 unit-tests the codex path
- [x] Cache-reuse contract preserved — Task 9 replay verification

**Placeholder scan:** No "TBD", "TODO", or "fill in details" tokens. All code blocks are complete and runnable.

**Type consistency:**
- `grade(predicted, gold, question, row, backend, out_dir=None)` signature uniform across all 4 new overrides + 2 existing (LCB, AIME).
- `check_answer(predicted, gold, answer_type)` and `judge_answer(predicted, gold, question, model, backend=..., out_dir=...)` signatures match `grader.py` definitions verified at lines 44 and 93.
- Return tuple `(bool, float)` consistent everywhere.

**Risk notes:**
- Task 8 is destructive on `base.py`. The Step 8.1 grep gate must produce empty output before proceeding. If non-empty, the violator must be migrated first.
- Task 9.2 replay test depends on the specific archived grade.json existing. If it has been deleted, pick another GPQA grade.json from the same run directory.
- Tests use `asyncio.get_event_loop().run_until_complete` for compatibility; if the test environment is on Python 3.12+ where this is deprecated, switch to `asyncio.run(coro)` (already used in Task 9.2).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-28-grading-cleanup-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — Dispatch one fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Run tasks sequentially in this session with checkpoints for review.

Which approach?
