# JudgeSpec Block + Multi-Bundle Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `judge_model` class attribute on each `BenchmarkConfig` subclass with a YAML-driven `judge:` block nested inside the `benchmark:` spec, and replace the single-verdict `grade.json` cache with a per-judge bundle directory (`judges/<label>/{config.json, grade.json, input.md, output.md, result.json}`) so multiple judges can grade the same explore side-by-side.

**Architecture:** `JudgeSpec` is a Pydantic discriminated union over `name in {claude, codex, vllm}`, embedded inside `HLESpec`/`BabyVisionSpec`/`RBenchVSpec` only. `LCBSpec`/`GPQASpec`/`AIME2025Spec`/`AIME2026Spec` reject `judge:` via `extra="forbid"`. `BenchmarkConfig` carries a `judge_spec: dict | None` instance attribute populated from the YAML at load time. `eval.py:_grade_with_cache` writes a 5-file bundle into `cache_base/<qid>/explore_<n>/judges/<label>/`. `find_cached_judge` does deep-equality on `config.json` to locate the right bundle. A separate `migrate_judge_layout.py` script does copy-then-cleanup migration of the existing 732+ `grade.json` files in the HLE Qwen36 cache.

**Tech Stack:** Python 3.11, Pydantic 2, pytest, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-01-judge-block-design.md`

**Verified preconditions (grep confirmed 2026-05-01):**
- `judge_model` class attribute declared in 6 benchmark files: `hle.py:122`, `babyvision.py:49`, `rbenchv.py:38` (all `claude-haiku-4-5-20251001`); `gpqa.py:57`, `lcb.py:89`, `aime.py:70` (all `None`).
- `eval.py:_grade_with_cache` (line 100) reads `benchmark.judge_model` at line 107, writes legacy `grade.json` at line 121-127.
- `eval.py:_grade_cached_explores` (line 131) reads cache at line 153-159.
- `eval.py:_grade_question_explores` calls `_grade_cached_explores` at line 197.
- `eval.py` line 316 logs `benchmark.judge_model` in the run config.
- `benchmarks/grader.py:grade_answer` signature: `(predicted, gold, question, answer_type, judge_model, backend, out_dir)`.
- `benchmarks/grader.py:judge_answer` writes 3 files to `out_dir`: `input.md` (via `TrajectoryWriter.create_simple`), `output.md`, `result.json` (via `save_sub_model_result`).
- HLE has bespoke routing in `hle.py:155-163`: `grade_backend = "claude" if backend == "vllm" else backend; judge_model = "gpt-5-codex-mini" if grade_backend == "codex" else self.judge_model`.
- Test scaffold exists at `tests/test_benchmark_specs.py` with the `_Holder` pattern; mirror for new tests.
- Cache to migrate: `analysis/cache/hle/qwen36_35b_a3b_fp8/gold/` (732 `grade.json` files at last count, mtime 2026-05-01).
- Other caches that may need migration: any directory matching `analysis/cache/**/grade.json`. Migration script must enumerate dynamically.

---

## File Structure

| File | Change |
|---|---|
| `benchmarks/specs.py` | Add `JudgeSpec` discriminated union (`ClaudeJudgeSpec`, `CodexJudgeSpec`, `VllmJudgeSpec`) + `SamplingConfig` re-export. Add `judge: JudgeSpec` field to `HLESpec`, `BabyVisionSpec`, `RBenchVSpec`. |
| `benchmarks/base.py` | Drop `judge_model: str \| None` annotation (line 333). Add `judge_spec: dict \| None = None` instance attribute (set via `__init__`). Add static `find_cached_judge(judges_dir, judge_spec)` and `judge_label(judge_spec)` helpers. |
| `benchmarks/grader.py` | Replace `grade_answer(... judge_model, backend, out_dir)` signature with `grade_answer(... judge_spec: dict \| None, out_dir)`. Replace `judge_answer(... model, backend, out_dir)` with `judge_answer(... judge_spec: dict, out_dir)`. Internally dispatch by `judge_spec["name"]`. Write `out_dir/config.json` alongside the existing `input.md`/`output.md`/`result.json`. |
| `benchmarks/hle.py` | Drop `judge_model = ...` (line 122). Drop bespoke routing in `.grade()` (lines 155-163). New `.grade()` calls `grade_answer(..., judge_spec=self.judge_spec, out_dir=...)`. |
| `benchmarks/babyvision.py` | Drop `judge_model = ...` (line 49). Hybrid `.grade()` calls `grade_answer` with `self.judge_spec` for blank questions, `check_answer` for choice questions (already the structure). |
| `benchmarks/rbenchv.py` | Drop `judge_model = ...` (line 38). `.grade()` calls `grade_answer(..., judge_spec=self.judge_spec, out_dir=...)`. |
| `benchmarks/gpqa.py`, `lcb.py`, `aime.py` | Drop `judge_model = None` declarations. No other change. |
| `benchmarks/__init__.py` (`get_benchmark`) | Update `BenchmarkConfig` constructor call to pass `judge_spec`. |
| `eval.py` | Replace every `benchmark.judge_model` read with new helpers. Update `_grade_with_cache` to write 5-file bundle into `judges/<label>/`. Update `_grade_cached_explores` cache lookup to use `find_cached_judge`. Update banner / log dict (line 316). |
| `precache_explores.py` | No change. |
| `scripts/babyvision/grpo/babyvision_qwen36_35b_a3b_exp_orch.yaml` | Add `judge:` block. |
| `scripts/babyvision/grpo/babyvision_qwen36_35b_a3b_temp.yaml` | Add `judge:` block. |
| `scripts/hle/grpo/hle_qwen36_35b_a3b_exp_orch.yaml` | Add `judge:` block. |
| `scripts/hle/grpo/hle_qwen36_35b_a3b_temp.yaml` | Add `judge:` block. |
| `scripts/rbenchv/sonnet/rbenchv_sonnet_precache_physics.yaml` | No `judge:` (precache only; remains unchanged). |
| All other `scripts/<bench>/<model>/<bench>_<model>_<method>.yaml` for HLE/BabyVision/RBenchV | Add `judge:` block (full sweep done in Task 14). |
| **(new)** `scripts/migrate_judge_layout.py` | Migration script with `--phase {dry-run, copy, cleanup}` flag + `--limit N` + `--cache-root <path>` flags. |
| **(new)** `tests/test_judge_spec.py` | Pydantic validation tests for `JudgeSpec` (per-name field rules, LCB/GPQA/AIME rejecting `judge:`). |
| **(new)** `tests/test_judge_cache_lookup.py` | Tests for `find_cached_judge`, `judge_label`, the deep-equality and label-collision raise. |
| **(new)** `tests/test_migrate_judge_layout.py` | Tests for the migration script's three phases on a tmp_path fixture. |
| `tests/test_benchmark_grade.py` | Update to pass `judge_spec` instead of relying on class-attribute `judge_model`. |

---

## Task 1: JudgeSpec discriminated union in specs.py

**Files:**
- Modify: `benchmarks/specs.py`
- Create: `tests/test_judge_spec.py`

- [ ] **Step 1.1: Write the failing tests for JudgeSpec validation**

Create `tests/test_judge_spec.py`:

```python
from __future__ import annotations
import pytest
from pydantic import BaseModel, ValidationError
from benchmarks.specs import (
    JudgeSpec, ClaudeJudgeSpec, CodexJudgeSpec, VllmJudgeSpec,
)


class _Holder(BaseModel):
    judge: JudgeSpec


def test_claude_judge_minimal():
    h = _Holder.model_validate({"judge": {"name": "claude", "model": "claude-haiku-4-5-20251001"}})
    assert isinstance(h.judge, ClaudeJudgeSpec)
    assert h.judge.model == "claude-haiku-4-5-20251001"


def test_codex_judge_minimal():
    h = _Holder.model_validate({"judge": {"name": "codex", "model": "gpt-5-codex-mini"}})
    assert isinstance(h.judge, CodexJudgeSpec)
    assert h.judge.model == "gpt-5-codex-mini"


def test_vllm_judge_with_sampling():
    h = _Holder.model_validate({
        "judge": {
            "name": "vllm",
            "model": "qwen36-35b-a3b-fp8",
            "sampling": {"temperature": 0.6, "max_tokens": 4096},
        }
    })
    assert isinstance(h.judge, VllmJudgeSpec)
    assert h.judge.model == "qwen36-35b-a3b-fp8"
    assert h.judge.sampling.temperature == 0.6


def test_vllm_judge_requires_sampling():
    with pytest.raises(ValidationError):
        _Holder.model_validate({"judge": {"name": "vllm", "model": "qwen36-35b-a3b-fp8"}})


def test_claude_judge_rejects_sampling():
    with pytest.raises(ValidationError):
        _Holder.model_validate({
            "judge": {"name": "claude", "model": "x", "sampling": {"temperature": 0.5}}
        })


def test_judge_unknown_name_rejected():
    with pytest.raises(ValidationError):
        _Holder.model_validate({"judge": {"name": "openrouter", "model": "x"}})
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/test_judge_spec.py -v
```

Expected: ImportError or ModuleNotFoundError on `JudgeSpec`/`ClaudeJudgeSpec`/etc., because they don't exist yet.

- [ ] **Step 1.3: Add JudgeSpec to specs.py**

Edit `benchmarks/specs.py`. After the `_Spec` base class, add:

```python
# ---------------------------------------------------------------------------
# JudgeSpec: per-judge config nested inside benchmarks that need an LLM judge.
# Discriminated union over `name`. claude/codex carry only `model`; vllm
# carries both `model` and a required `sampling` block (matches the
# SamplingConfig used by explorer/orchestrator in eval.py).
# ---------------------------------------------------------------------------

# Late import to avoid cycle: SamplingConfig lives in eval.py and references
# nothing from specs.py.
def _sampling_config_type():
    from eval import SamplingConfig
    return SamplingConfig


class _JudgeSpec(BaseModel):
    model_config = {"extra": "forbid"}


class ClaudeJudgeSpec(_JudgeSpec):
    name: Literal["claude"]
    model: str


class CodexJudgeSpec(_JudgeSpec):
    name: Literal["codex"]
    model: str


class VllmJudgeSpec(_JudgeSpec):
    name: Literal["vllm"]
    model: str
    sampling: "SamplingConfig"  # forward ref; resolved at model_validate time

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        # Resolve the forward ref now that we are inside a validation context.
        from eval import SamplingConfig  # noqa: F401
        cls.model_rebuild(_types_namespace={"SamplingConfig": SamplingConfig})
        return handler(cls)


JudgeSpec = Annotated[
    Union[ClaudeJudgeSpec, CodexJudgeSpec, VllmJudgeSpec],
    Field(discriminator="name"),
]
```

Note: The forward reference dance above is necessary because `eval.py` already imports from `specs.py`; importing the other direction at module-top creates a cycle. If the simpler form below works in your environment (Pydantic 2.5+ handles late binding), prefer it:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from eval import SamplingConfig

class VllmJudgeSpec(_JudgeSpec):
    name: Literal["vllm"]
    model: str
    sampling: "SamplingConfig"
```

If the late-binding pattern fails at validation time, factor `SamplingConfig` into a new module (`benchmarks/sampling.py` or `core/sampling.py`) and import it from both `eval.py` and `specs.py`. Decide at implementation time based on what works; the spec does not constrain this.

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
conda run -n explain --no-capture-output pytest tests/test_judge_spec.py -v
```

Expected: All 6 tests pass.

- [ ] **Step 1.5: Commit**

```bash
git add benchmarks/specs.py tests/test_judge_spec.py
git commit -m "feat(specs): add JudgeSpec discriminated union (claude/codex/vllm)"
```

---

## Task 2: Embed JudgeSpec into HLESpec / BabyVisionSpec / RBenchVSpec

**Files:**
- Modify: `benchmarks/specs.py`
- Modify: `tests/test_judge_spec.py`

- [ ] **Step 2.1: Add tests for benchmark-level judge embedding**

Append to `tests/test_judge_spec.py`:

```python
from benchmarks.specs import BenchmarkSpec, HLESpec, LCBSpec, GPQASpec


class _BenchHolder(BaseModel):
    benchmark: BenchmarkSpec


def test_hle_with_judge():
    h = _BenchHolder.model_validate({
        "benchmark": {
            "name": "hle", "subset": "gold",
            "judge": {"name": "claude", "model": "claude-haiku-4-5-20251001"},
        }
    })
    assert isinstance(h.benchmark, HLESpec)
    assert isinstance(h.benchmark.judge, ClaudeJudgeSpec)


def test_lcb_rejects_judge():
    with pytest.raises(ValidationError) as exc:
        _BenchHolder.model_validate({
            "benchmark": {"name": "lcb", "judge": {"name": "claude", "model": "x"}}
        })
    assert "judge" in str(exc.value).lower() or "extra" in str(exc.value).lower()


def test_gpqa_rejects_judge():
    with pytest.raises(ValidationError):
        _BenchHolder.model_validate({
            "benchmark": {"name": "gpqa", "judge": {"name": "claude", "model": "x"}}
        })
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
conda run -n explain --no-capture-output pytest tests/test_judge_spec.py::test_hle_with_judge -v
```

Expected: FAIL — Pydantic rejects unknown `judge` field on `HLESpec` (because `_Spec.extra="forbid"`).

- [ ] **Step 2.3: Add `judge: JudgeSpec` field to the three judge-using specs**

Edit `benchmarks/specs.py`:

```python
class HLESpec(_Spec):
    name: Literal["hle"]
    subset: Literal["gold", "revision", "uncertain"] | None = None
    category: str | None = None
    text_only: bool = False
    judge: JudgeSpec  # required; no implicit default

# (LEAVE GPQASpec, LCBSpec, AIME2025Spec, AIME2026Spec UNCHANGED — they reject `judge:` via extra="forbid")

class BabyVisionSpec(_Spec):
    name: Literal["babyvision"]
    type: str | None = None
    subtype: str | None = None
    judge: JudgeSpec

class RBenchVSpec(_Spec):
    name: Literal["rbenchv"]
    category: str | None = None
    judge: JudgeSpec
```

- [ ] **Step 2.4: Run all spec tests to verify pass**

```bash
conda run -n explain --no-capture-output pytest tests/test_judge_spec.py tests/test_benchmark_specs.py -v
```

Expected: All tests pass. Existing `test_hle_full` / `test_hle_minimal` will FAIL because they don't supply `judge:`. Update them to include `judge: {name: claude, model: claude-haiku-4-5-20251001}` or skip if conditional.

- [ ] **Step 2.5: Update existing tests to pass `judge:` for hle/babyvision/rbenchv**

In `tests/test_benchmark_specs.py`, every test that validates `{"name": "hle", ...}` / `babyvision` / `rbenchv` must include the `judge:` field. Rewrite each affected test inline; there are about 3-4 of them.

- [ ] **Step 2.6: Run all spec tests again**

```bash
conda run -n explain --no-capture-output pytest tests/test_judge_spec.py tests/test_benchmark_specs.py -v
```

Expected: All tests pass.

- [ ] **Step 2.7: Commit**

```bash
git add benchmarks/specs.py tests/test_judge_spec.py tests/test_benchmark_specs.py
git commit -m "feat(specs): embed JudgeSpec into HLE/BabyVision/RBenchV specs"
```

---

## Task 3: Plumb judge_spec into BenchmarkConfig

**Files:**
- Modify: `benchmarks/base.py`
- Modify: `benchmarks/__init__.py` (`get_benchmark` factory)
- Modify: `eval.py` (load_config path, infra construction)
- Modify: `precache_explores.py` (load_config path — no behavior change there)

- [ ] **Step 3.1: Read current BenchmarkConfig __init__**

```bash
grep -nE "class BenchmarkConfig|def __init__" /data3/peijia/dr-claw/Explain/Experiment/core_code/benchmarks/base.py | head
```

- [ ] **Step 3.2: Drop class-attribute judge_model annotation, add instance attribute**

Edit `benchmarks/base.py`. Find the `judge_model: str | None` annotation (per spec preconditions, line 333). Remove it. Add (or extend) `__init__`:

```python
class BenchmarkConfig:
    name: str
    grading_summary: str = ""

    def __init__(self, judge_spec: dict | None = None):
        # judge_spec is the per-run JudgeSpec dump from YAML, e.g.
        #   {"name": "claude", "model": "claude-haiku-4-5-20251001"}
        # or None for benchmarks that grade without an LLM judge.
        self.judge_spec = judge_spec
```

- [ ] **Step 3.3: Update `get_benchmark` to accept and pass `judge_spec`**

In `benchmarks/__init__.py`, find `get_benchmark`. Change its signature:

```python
def get_benchmark(name: str, judge_spec: dict | None = None) -> BenchmarkConfig:
    # ... existing dispatch ...
    return cls(judge_spec=judge_spec)
```

- [ ] **Step 3.4: Update eval.py and precache_explores.py to extract judge_spec from BenchmarkSpec and pass it through**

In `eval.py`, find the line that calls `get_benchmark(cfg.benchmark.name)` (the same pattern as `precache_explores.py:174`). Change to:

```python
bench_dict = cfg.benchmark.model_dump()
judge_spec = bench_dict.pop("judge", None)
benchmark = get_benchmark(cfg.benchmark.name, judge_spec=judge_spec)
bench_filters = {k: v for k, v in bench_dict.items() if k != "name"}
```

Same change in `precache_explores.py`. (Precache does not grade, but `judge_spec` is `None` for non-judge benchmarks — passing it through harmlessly.)

- [ ] **Step 3.5: Verify imports still resolve**

```bash
conda run -n explain --no-capture-output python -c "from benchmarks import get_benchmark; b = get_benchmark('hle', judge_spec={'name':'claude','model':'x'}); print(b.judge_spec)"
```

Expected: prints `{'name': 'claude', 'model': 'x'}` or, depending on existing class layout, raises a clean error (which Task 5/6 will fix).

- [ ] **Step 3.6: Commit**

```bash
git add benchmarks/base.py benchmarks/__init__.py eval.py precache_explores.py
git commit -m "feat(base): plumb judge_spec from YAML into BenchmarkConfig"
```

---

## Task 4: find_cached_judge + judge_label helpers + tests

**Files:**
- Modify: `benchmarks/base.py`
- Create: `tests/test_judge_cache_lookup.py`

- [ ] **Step 4.1: Write failing tests**

Create `tests/test_judge_cache_lookup.py`:

```python
from __future__ import annotations
import json
import pytest
from pathlib import Path
from benchmarks.base import find_cached_judge, judge_label


def test_judge_label_claude():
    spec = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    assert judge_label(spec) == "claude__claude-haiku-4-5-20251001"


def test_judge_label_vllm():
    spec = {"name": "vllm", "model": "qwen36-35b-a3b-fp8",
            "sampling": {"temperature": 0.6}}
    assert judge_label(spec) == "vllm__qwen36-35b-a3b-fp8"


def test_find_cached_judge_hit(tmp_path):
    judges_dir = tmp_path / "judges"
    spec = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    bundle = judges_dir / "claude__claude-haiku-4-5-20251001"
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text(json.dumps(spec))
    found = find_cached_judge(judges_dir, spec)
    assert found == bundle


def test_find_cached_judge_miss_returns_none(tmp_path):
    judges_dir = tmp_path / "judges"
    judges_dir.mkdir()
    spec = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    assert find_cached_judge(judges_dir, spec) is None


def test_find_cached_judge_label_collision_raises(tmp_path):
    judges_dir = tmp_path / "judges"
    stored_spec = {"name": "vllm", "model": "qwen36-35b-a3b-fp8",
                   "sampling": {"temperature": 0.6}}
    requested_spec = {"name": "vllm", "model": "qwen36-35b-a3b-fp8",
                      "sampling": {"temperature": 0.0}}
    bundle = judges_dir / "vllm__qwen36-35b-a3b-fp8"
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text(json.dumps(stored_spec))
    with pytest.raises(RuntimeError, match="Judge label collision"):
        find_cached_judge(judges_dir, requested_spec)


def test_find_cached_judge_no_judges_dir(tmp_path):
    # No judges/ directory yet — must not raise; treat as miss.
    spec = {"name": "claude", "model": "x"}
    assert find_cached_judge(tmp_path / "judges", spec) is None
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
conda run -n explain --no-capture-output pytest tests/test_judge_cache_lookup.py -v
```

Expected: ImportError on `find_cached_judge` / `judge_label`.

- [ ] **Step 4.3: Implement helpers in benchmarks/base.py**

Add at module level (top of `benchmarks/base.py`, after imports):

```python
import json
from pathlib import Path


def judge_label(judge_spec: dict) -> str:
    """Stable, human-readable label for a judge bundle directory."""
    return f"{judge_spec['name']}__{judge_spec['model']}"


def find_cached_judge(judges_dir: Path, judge_spec: dict) -> Path | None:
    """Locate the cached bundle matching judge_spec, or None on miss.

    Raises RuntimeError if a label-named directory exists but its config.json
    differs from judge_spec (label collision; user must rename manually).
    """
    label = judge_label(judge_spec)
    candidate = judges_dir / label
    if not candidate.exists():
        return None
    config_path = candidate / "config.json"
    if not config_path.exists():
        # Partial bundle from a crashed write. Treat as miss; let caller decide.
        return None
    stored = json.loads(config_path.read_text(encoding="utf-8"))
    if stored == judge_spec:
        return candidate
    raise RuntimeError(
        f"Judge label collision at {candidate}.\n"
        f"  Stored config:    {stored}\n"
        f"  Requested config: {judge_spec}\n"
        f"Two judges share the same backend+model label but differ on other "
        f"fields. Manually rename one of the conflicting bundles."
    )
```

- [ ] **Step 4.4: Run tests to verify they pass**

```bash
conda run -n explain --no-capture-output pytest tests/test_judge_cache_lookup.py -v
```

Expected: All 6 tests pass.

- [ ] **Step 4.5: Commit**

```bash
git add benchmarks/base.py tests/test_judge_cache_lookup.py
git commit -m "feat(base): add find_cached_judge and judge_label helpers"
```

---

## Task 5: Refactor grader.py to accept JudgeSpec

**Files:**
- Modify: `benchmarks/grader.py`
- Modify: `tests/test_benchmark_grade.py` (existing)

- [ ] **Step 5.1: Read existing grader.py grade_answer / judge_answer signatures**

Lines 94-147 of `benchmarks/grader.py` (per Verified preconditions). Read for context.

- [ ] **Step 5.2: Update existing grade tests to use judge_spec dict**

Find each test in `tests/test_benchmark_grade.py` that passes `judge_model="..."` and replace with `judge_spec={"name":"claude","model":"..."}`. Re-read the file first to enumerate touch sites.

- [ ] **Step 5.3: Run tests to verify the new signature is needed (they fail)**

```bash
conda run -n explain --no-capture-output pytest tests/test_benchmark_grade.py -v
```

Expected: TypeError on `judge_spec` kwarg not recognized.

- [ ] **Step 5.4: Replace grade_answer and judge_answer signatures**

In `benchmarks/grader.py`, replace `judge_answer` (lines 94-133) with:

```python
async def judge_answer(
    predicted: str, gold: str, question: str, judge_spec: dict,
    out_dir: Path | None = None,
) -> tuple[bool, float]:
    """Use an LLM to judge if predicted answer matches gold. Returns (correct, cost_usd)."""
    backend = judge_spec["name"]  # claude, codex, vllm
    model = judge_spec["model"]
    sampling = judge_spec.get("sampling")
    judge_prompt = _get_judge_system_prompt(backend)
    user_message = (
        f"[question]: {question}\n"
        f"[response]: {predicted}\n"
        f"[correct_answer]: {gold}"
    )
    writer = TrajectoryWriter.create_simple(out_dir / "output.md") if out_dir else TrajectoryWriter.noop()
    try:
        result, trajectory_text, cost_usd, usage = await call_sub_model(
            backend=backend,
            system_prompt=judge_prompt,
            user_message=user_message,
            image_data_url=None,
            model=model,
            output_schema=JUDGE_SCHEMA,
            writer=writer,
            sampling=sampling,
        )
    except asyncio.TimeoutError:
        print(f"  [judge] SDK timeout after all retries -- treating as incorrect")
        return False, 0.0
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(
            json.dumps(judge_spec, indent=2, sort_keys=True), encoding="utf-8")
        save_sub_model_result(
            out_dir=out_dir,
            result=result,
            trajectory_text=trajectory_text,
            cost_usd=cost_usd,
            usage=usage,
            duration_seconds=0.0,
            model=model,
        )
    if result.get("timed_out"):
        print(f"  [judge] timed out -- treating as incorrect")
        return False, 0.0
    return result["correct"], cost_usd
```

Also add `import json` at the top if not already present.

Replace `grade_answer` (lines 136-147) with:

```python
async def grade_answer(
    predicted: str, gold: str, question: str, answer_type: str,
    judge_spec: dict | None = None,
    out_dir: Path | None = None,
) -> tuple[bool, float]:
    """Grade an answer. Returns (correct, judge_cost_usd)."""
    if answer_type == "multipleChoice":
        return check_answer(predicted, gold, answer_type), 0.0
    if judge_spec is None:
        return check_answer(predicted, gold, answer_type), 0.0
    return await judge_answer(predicted, gold, question, judge_spec, out_dir=out_dir)
```

- [ ] **Step 5.5: Run grader tests to verify pass**

```bash
conda run -n explain --no-capture-output pytest tests/test_benchmark_grade.py -v
```

Expected: All pass.

- [ ] **Step 5.6: Commit**

```bash
git add benchmarks/grader.py tests/test_benchmark_grade.py
git commit -m "refactor(grader): accept judge_spec dict, write config.json into out_dir"
```

---

## Task 6: Drop class-attribute judge_model from all benchmark files

**Files:**
- Modify: `benchmarks/hle.py`, `benchmarks/babyvision.py`, `benchmarks/rbenchv.py` (drop `judge_model`, drop bespoke routing, switch `.grade()` to use `self.judge_spec`)
- Modify: `benchmarks/gpqa.py`, `benchmarks/lcb.py`, `benchmarks/aime.py` (drop `judge_model = None`)

- [ ] **Step 6.1: HLE — drop class attr and rewrite .grade()**

In `benchmarks/hle.py`:
- Delete the comment block + `judge_model = "claude-haiku-4-5-20251001"` (around line 116-122).
- Replace `.grade()` (lines 152-163, the bespoke `vllm -> claude` and `codex -> gpt-5-codex-mini` swaps) with:

```python
async def grade(self, predicted, gold, question, row, *, backend, out_dir):
    answer_type = row.get("answer_type", "exactMatch")
    if answer_type == "multipleChoice":
        return check_answer(predicted, gold, "multipleChoice"), 0.0
    return await judge_answer(predicted, gold, question, self.judge_spec, out_dir=out_dir)
```

The `backend` parameter is now unused inside HLE.grade — kept for interface symmetry. The judge backend lives entirely in `self.judge_spec["name"]`.

- [ ] **Step 6.2: BabyVision — drop class attr, switch grade()**

In `benchmarks/babyvision.py`:
- Delete `judge_model = "claude-haiku-4-5-20251001"` (line 49).
- Replace `.grade()` body so the blank-question branch calls `judge_answer(... self.judge_spec, ...)`. Choice branch unchanged.

- [ ] **Step 6.3: RBenchV — drop class attr, switch grade()**

In `benchmarks/rbenchv.py`:
- Delete `judge_model = "claude-haiku-4-5-20251001"` (line 38).
- Replace `.grade()` body to call `judge_answer(... self.judge_spec, ...)`.

- [ ] **Step 6.4: GPQA / LCB / AIME — drop dead `judge_model = None`**

In each: delete the `judge_model = None` line. No other change.

- [ ] **Step 6.5: Run benchmark grade tests**

```bash
conda run -n explain --no-capture-output pytest tests/test_benchmark_grade.py -v
```

Expected: All pass.

- [ ] **Step 6.6: Commit**

```bash
git add benchmarks/hle.py benchmarks/babyvision.py benchmarks/rbenchv.py benchmarks/gpqa.py benchmarks/lcb.py benchmarks/aime.py
git commit -m "refactor(benchmarks): drop judge_model class attribute, route via self.judge_spec"
```

---

## Task 7: New cache layout in eval.py

**Files:**
- Modify: `eval.py`

- [ ] **Step 7.1: Read current _grade_with_cache (lines 100-128) and _grade_cached_explores (lines 131-180)**

```bash
sed -n '95,180p' /data3/peijia/dr-claw/Explain/Experiment/core_code/eval.py
```

- [ ] **Step 7.2: Replace _grade_with_cache with judges/<label>/ layout**

Replace lines 100-128 with:

```python
async def _grade_with_cache(
    benchmark: BenchmarkConfig,
    predicted: str, gold: str, question: str, row: dict,
    backend: str, grade_dir: Path,
    quiet: bool = True,
) -> tuple[bool, float]:
    """Grade an answer, caching the bundle under grade_dir/judges/<label>/.

    grade_dir is the per-explore directory (e.g. cache_base/<qid>/explore_N/
    or run_dir/grading/<qid>/.../explore_N/). On cache hit, returns the cached
    verdict with judge_cost=0. On miss, calls the judge, writes the 5-file
    bundle (config.json, grade.json, input.md, output.md, result.json) into
    grade_dir/judges/<label>/, and returns (verdict, judge_cost).
    """
    judges_dir = grade_dir / "judges"

    # Benchmarks that grade without a judge (LCB, GPQA, AIME, BabyVision-choice)
    # short-circuit with the in-class .grade() and write nothing under judges/.
    if benchmark.judge_spec is None:
        return await benchmark.grade(predicted, gold, question, row,
                                     backend=backend, out_dir=None)

    # Cache hit?
    cached = find_cached_judge(judges_dir, benchmark.judge_spec)
    if cached is not None:
        grade_path = cached / "grade.json"
        if grade_path.exists():
            data = json.loads(grade_path.read_text(encoding="utf-8"))
            return data["is_correct"], 0.0
        # config.json present but grade.json missing -> partial bundle. Re-run.

    # Miss -> run the judge and write the 5-file bundle.
    label = judge_label(benchmark.judge_spec)
    bundle_dir = judges_dir / label
    bundle_dir.mkdir(parents=True, exist_ok=True)
    is_correct, judge_cost = await benchmark.grade(
        predicted, gold, question, row,
        backend=backend, out_dir=bundle_dir,
    )
    if not quiet:
        print(f"  [sub-model] judge: correct={is_correct}, predicted={str(predicted)[:60]}, gold={str(gold)[:60]}, cost=${judge_cost}")
    (bundle_dir / "grade.json").write_text(json.dumps({
        "judge_spec": benchmark.judge_spec,
        "is_correct": is_correct,
        "predicted": predicted,
        "gold": gold,
        "judge_cost_usd": judge_cost,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return is_correct, judge_cost
```

Add to imports at top of `eval.py`:

```python
from benchmarks.base import find_cached_judge, judge_label
```

- [ ] **Step 7.3: Update _grade_cached_explores cache lookup**

In `_grade_cached_explores` (around lines 152-176), replace the legacy `cache_grade_path` block:

```python
# OLD:
# cache_grade_path = cache_base / qid / f"explore_{idx}" / "grade.json"
# judge_key = benchmark.judge_model or "none"
# cached_grade = None
# if cache_grade_path.exists():
#     cached_grade = json.loads(cache_grade_path.read_text(encoding="utf-8"))
#     if cached_grade.get("judge_model") != judge_key:
#         cached_grade = None
# if cached_grade is not None:
#     is_correct_exp = cached_grade["is_correct"]
#     jc = 0.0
# else:
#     is_correct_exp, jc = await _grade_with_cache(... grade_dir=cache_base/qid/f"explore_{idx}", ...)

# NEW: Single call. _grade_with_cache reads the bundle from cache_base via
# its grade_dir/judges/<label>/ pattern, writes back on miss.
is_correct_exp, jc = await _grade_with_cache(
    benchmark, ans, gold_answer, question, row,
    backend=backend, grade_dir=cache_base / qid / f"explore_{idx}", quiet=quiet,
)
```

- [ ] **Step 7.4: Update banner / log dict**

`eval.py:316` logs `"judge_model": benchmark.judge_model`. Change to:

```python
"judge_spec": benchmark.judge_spec,
```

Find any other `benchmark.judge_model` reads via:

```bash
grep -n "judge_model" /data3/peijia/dr-claw/Explain/Experiment/core_code/eval.py
```

Replace each. The runtime of these reads (resume paths) needs the same logic: `find_cached_judge(judges_dir, benchmark.judge_spec) is not None` instead of string comparison.

- [ ] **Step 7.5: Sanity smoke test (not a unit test)**

This is a tricky path because it touches live cache. Run a no-questions dry sanity:

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output python -c "
from eval import load_config, EvalConfig
cfg = load_config('scripts/hle/grpo/hle_qwen36_35b_a3b_exp_orch.yaml', EvalConfig)
print('OK:', type(cfg.benchmark).__name__, 'judge=', cfg.benchmark.judge.model_dump())"
```

Expected: prints HLESpec + judge dict. Will fail until Task 14 adds `judge:` to the YAML; that is fine — defer this verification step until after Task 14.

- [ ] **Step 7.6: Commit**

```bash
git add eval.py
git commit -m "refactor(eval): move grade.json to judges/<label>/ bundle layout"
```

---

## Task 8: Migration script — skeleton + dry-run phase + tests

**Files:**
- Create: `scripts/migrate_judge_layout.py`
- Create: `tests/test_migrate_judge_layout.py`

- [ ] **Step 8.1: Write failing tests for dry-run phase**

Create `tests/test_migrate_judge_layout.py`:

```python
from __future__ import annotations
import json
import shutil
from pathlib import Path
import pytest
import importlib.util

# Import the script as a module (it lives under scripts/, not the package root).
SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "migrate_judge_layout.py"
spec = importlib.util.spec_from_file_location("migrate_judge_layout", SCRIPT_PATH)
mig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mig)


def _make_legacy_explore(explore_dir: Path, judge_model: str = "claude-haiku-4-5-20251001"):
    """Create a single legacy-layout explore_N directory."""
    explore_dir.mkdir(parents=True, exist_ok=True)
    (explore_dir / "result.json").write_text(json.dumps({"answer": "D", "cost_usd": 0.0}))
    (explore_dir / "input.md").write_text("question prompt")
    (explore_dir / "output.md").write_text("model output")
    (explore_dir / "grade.json").write_text(json.dumps({
        "judge_model": judge_model,
        "is_correct": True,
        "predicted": "D",
        "gold": "D",
        "judge_cost_usd": 0.0046,
    }))
    judge_dir = explore_dir / "judge"
    judge_dir.mkdir()
    (judge_dir / "input.md").write_text("judge prompt")
    (judge_dir / "output.md").write_text("judge output")
    (judge_dir / "result.json").write_text(json.dumps({"correct": True}))


def test_dry_run_counts(tmp_path, capsys):
    cache = tmp_path / "cache"
    _make_legacy_explore(cache / "qid1" / "explore_1")
    _make_legacy_explore(cache / "qid1" / "explore_2")
    _make_legacy_explore(cache / "qid2" / "explore_1")
    mig.run(cache_root=cache, phase="dry-run", limit=None)
    captured = capsys.readouterr()
    assert "Total: 3 explores eligible" in captured.out
    # Original files unchanged.
    assert (cache / "qid1" / "explore_1" / "grade.json").exists()
    assert (cache / "qid1" / "explore_1" / "judge").exists()


def test_dry_run_idempotent_skips_already_migrated(tmp_path, capsys):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    # Pre-create new layout to simulate already-migrated.
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text(json.dumps({
        "name": "claude", "model": "claude-haiku-4-5-20251001"
    }))
    (bundle / "grade.json").write_text("{}")
    mig.run(cache_root=cache, phase="dry-run", limit=None)
    captured = capsys.readouterr()
    assert "1 already migrated" in captured.out


def test_dry_run_skips_no_grade_json(tmp_path, capsys):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    explore.mkdir(parents=True)
    (explore / "result.json").write_text("{}")  # explorer-only, no grade
    mig.run(cache_root=cache, phase="dry-run", limit=None)
    captured = capsys.readouterr()
    assert "1 skipped (no grade.json)" in captured.out
```

- [ ] **Step 8.2: Run tests to verify they fail**

```bash
conda run -n explain --no-capture-output pytest tests/test_migrate_judge_layout.py -v
```

Expected: ImportError or FileNotFoundError on the script path.

- [ ] **Step 8.3: Implement skeleton + dry-run phase**

Create `scripts/migrate_judge_layout.py`:

```python
"""Migrate legacy grade.json + judge/ layout to judges/<label>/ bundle layout.

Three phases (run in order, one per invocation):
  --phase dry-run   Walk the cache tree, print what would be migrated. No I/O.
  --phase copy      Copy each explore's grade.json + judge/* into
                    judges/<label>/, write config.json. Verify hashes.
                    Original files remain untouched (safe).
  --phase cleanup   After verification + smoke eval pass, delete the legacy
                    grade.json and judge/ from each explore.

Usage:
  python scripts/migrate_judge_layout.py --phase dry-run --cache-root analysis/cache
  python scripts/migrate_judge_layout.py --phase copy    --cache-root analysis/cache --limit 5
  python scripts/migrate_judge_layout.py --phase copy    --cache-root analysis/cache
  # ... run smoke eval and verify $0 judge cost ...
  python scripts/migrate_judge_layout.py --phase cleanup --cache-root analysis/cache
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def _label(judge_model: str) -> str:
    # All current legacy entries use claude-* models. If a non-claude model is
    # encountered, abort with a clear message; user must extend this script.
    if judge_model.startswith("claude-"):
        return f"claude__{judge_model}"
    raise SystemExit(
        f"Unexpected judge_model {judge_model!r} in legacy grade.json. "
        f"Migration script only handles claude-* models. Extend manually."
    )


def _explore_dirs(cache_root: Path):
    """Yield each cache_root/<bench>/<model>[/<filter>...]/<qid>/explore_N/."""
    for grade in cache_root.rglob("grade.json"):
        explore_dir = grade.parent
        if explore_dir.name.startswith("explore_"):
            yield explore_dir


def _is_already_migrated(explore_dir: Path) -> bool:
    """A bundle exists under judges/* with a grade.json."""
    judges_dir = explore_dir / "judges"
    if not judges_dir.exists():
        return False
    for sub in judges_dir.iterdir():
        if sub.is_dir() and (sub / "grade.json").exists():
            return True
    return False


def run(cache_root: Path, phase: str, limit: int | None):
    if phase == "dry-run":
        return _phase_dry_run(cache_root, limit)
    if phase == "copy":
        return _phase_copy(cache_root, limit)
    if phase == "cleanup":
        return _phase_cleanup(cache_root, limit)
    raise SystemExit(f"Unknown phase: {phase}")


def _phase_dry_run(cache_root: Path, limit: int | None):
    eligible = 0
    already = 0
    skipped = 0
    samples_shown = 0
    examples = []
    for explore_dir in _explore_dirs(cache_root):
        grade_path = explore_dir / "grade.json"
        if not grade_path.exists():
            skipped += 1
            continue
        if _is_already_migrated(explore_dir):
            already += 1
            continue
        eligible += 1
        if samples_shown < 5:
            data = json.loads(grade_path.read_text(encoding="utf-8"))
            label = _label(data["judge_model"])
            examples.append((explore_dir, label))
            samples_shown += 1
    # Also count true-skipped: explore_N that have no grade.json at all.
    for ed in cache_root.rglob("explore_*"):
        if not (ed / "grade.json").exists() and ed.is_dir():
            if not (ed / "judges").exists():
                skipped += 1
    print(f"Cache root: {cache_root}")
    print(f"Sample destinations:")
    for explore_dir, label in examples:
        print(f"  {explore_dir.relative_to(cache_root)} -> judges/{label}/")
    print(f"Total: {eligible} explores eligible for migration; "
          f"{already} already migrated; {skipped} skipped (no grade.json).")


def _phase_copy(cache_root: Path, limit: int | None):
    raise NotImplementedError("Implemented in Task 9.")


def _phase_cleanup(cache_root: Path, limit: int | None):
    raise NotImplementedError("Implemented in Task 10.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--phase", choices=["dry-run", "copy", "cleanup"], required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N eligible explores (for piloting).")
    args = p.parse_args()
    run(cache_root=args.cache_root, phase=args.phase, limit=args.limit)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.4: Run tests to verify dry-run passes**

```bash
conda run -n explain --no-capture-output pytest tests/test_migrate_judge_layout.py -v
```

Expected: All 3 dry-run tests pass.

- [ ] **Step 8.5: Commit**

```bash
git add scripts/migrate_judge_layout.py tests/test_migrate_judge_layout.py
git commit -m "feat(scripts): migrate_judge_layout.py with dry-run phase"
```

---

## Task 9: Migration script — copy phase + tests

**Files:**
- Modify: `scripts/migrate_judge_layout.py`
- Modify: `tests/test_migrate_judge_layout.py`

- [ ] **Step 9.1: Write failing tests for copy phase**

Append to `tests/test_migrate_judge_layout.py`:

```python
def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


import hashlib  # add at top of file


def test_copy_creates_bundle(tmp_path):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    mig.run(cache_root=cache, phase="copy", limit=None)
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    assert bundle.exists()
    assert (bundle / "config.json").exists()
    assert (bundle / "grade.json").exists()
    assert (bundle / "input.md").exists()
    assert (bundle / "output.md").exists()
    assert (bundle / "result.json").exists()
    # Originals untouched (copy, not move).
    assert (explore / "grade.json").exists()
    assert (explore / "judge" / "input.md").exists()
    # config.json content correct.
    cfg = json.loads((bundle / "config.json").read_text())
    assert cfg == {"name": "claude", "model": "claude-haiku-4-5-20251001"}


def test_copy_byte_for_byte_fidelity(tmp_path):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    src_grade = _sha256(explore / "grade.json")
    src_judge_in = _sha256(explore / "judge" / "input.md")
    mig.run(cache_root=cache, phase="copy", limit=None)
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    assert _sha256(bundle / "grade.json") == src_grade
    assert _sha256(bundle / "input.md") == src_judge_in


def test_copy_idempotent(tmp_path):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    mig.run(cache_root=cache, phase="copy", limit=None)
    mtime1 = (explore / "judges" / "claude__claude-haiku-4-5-20251001" / "config.json").stat().st_mtime
    mig.run(cache_root=cache, phase="copy", limit=None)  # second run
    mtime2 = (explore / "judges" / "claude__claude-haiku-4-5-20251001" / "config.json").stat().st_mtime
    # Already-migrated explores skipped; mtime unchanged.
    assert mtime1 == mtime2


def test_copy_limit(tmp_path):
    cache = tmp_path / "cache"
    for i in range(5):
        _make_legacy_explore(cache / f"qid{i}" / "explore_1")
    mig.run(cache_root=cache, phase="copy", limit=2)
    migrated = list(cache.rglob("config.json"))
    assert len(migrated) == 2


def test_copy_aborts_on_unknown_judge_model(tmp_path):
    cache = tmp_path / "cache"
    _make_legacy_explore(cache / "qid1" / "explore_1", judge_model="gpt-4-turbo-preview")
    with pytest.raises(SystemExit, match="Unexpected judge_model"):
        mig.run(cache_root=cache, phase="copy", limit=None)
```

- [ ] **Step 9.2: Run tests to verify they fail**

```bash
conda run -n explain --no-capture-output pytest tests/test_migrate_judge_layout.py -v -k copy
```

Expected: NotImplementedError on copy phase.

- [ ] **Step 9.3: Implement copy phase**

Replace `_phase_copy` in `scripts/migrate_judge_layout.py`:

```python
def _atomic_write(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _verified_copy(src: Path, dst: Path):
    """Copy src to dst and assert byte-for-byte equality via sha256."""
    shutil.copy2(src, dst)
    src_h = hashlib.sha256(src.read_bytes()).hexdigest()
    dst_h = hashlib.sha256(dst.read_bytes()).hexdigest()
    if src_h != dst_h:
        raise SystemExit(
            f"Hash mismatch after copy: {src} -> {dst}\n"
            f"  src sha256: {src_h}\n  dst sha256: {dst_h}\n"
            f"Migration aborted; original {src} is intact, partial bundle at {dst.parent}."
        )


def _phase_copy(cache_root: Path, limit: int | None):
    processed = 0
    skipped_already = 0
    for explore_dir in _explore_dirs(cache_root):
        if limit is not None and processed >= limit:
            break
        grade_path = explore_dir / "grade.json"
        if not grade_path.exists():
            continue
        if _is_already_migrated(explore_dir):
            skipped_already += 1
            continue

        data = json.loads(grade_path.read_text(encoding="utf-8"))
        judge_model = data["judge_model"]
        label = _label(judge_model)

        bundle = explore_dir / "judges" / label
        bundle.mkdir(parents=True, exist_ok=True)

        # 1) config.json (atomic)
        config = {"name": "claude", "model": judge_model}
        _atomic_write(bundle / "config.json",
                      json.dumps(config, indent=2, sort_keys=True))

        # 2) grade.json (verified copy)
        _verified_copy(grade_path, bundle / "grade.json")

        # 3) judge/* (verified copy)
        legacy_judge = explore_dir / "judge"
        for fname in ("input.md", "output.md", "result.json"):
            src = legacy_judge / fname
            if src.exists():
                _verified_copy(src, bundle / fname)
            else:
                print(f"  WARN: {src} missing in legacy layout; skipping.")

        # 4) Sanity: re-read bundle config.json and confirm it deep-equals.
        stored = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        if stored != config:
            raise SystemExit(f"config.json content drift at {bundle}")

        processed += 1
        if processed % 50 == 0:
            print(f"  ... copied {processed} explores")

    print(f"Phase copy: {processed} migrated, {skipped_already} already migrated.")
    print(f"Originals untouched. Run smoke eval next; only then run --phase cleanup.")
```

- [ ] **Step 9.4: Run tests to verify pass**

```bash
conda run -n explain --no-capture-output pytest tests/test_migrate_judge_layout.py -v -k copy
```

Expected: All 5 copy tests pass.

- [ ] **Step 9.5: Commit**

```bash
git add scripts/migrate_judge_layout.py tests/test_migrate_judge_layout.py
git commit -m "feat(scripts): migrate copy phase with sha256 verification"
```

---

## Task 10: Migration script — cleanup phase + tests

**Files:**
- Modify: `scripts/migrate_judge_layout.py`
- Modify: `tests/test_migrate_judge_layout.py`

- [ ] **Step 10.1: Write failing tests for cleanup phase**

Append to `tests/test_migrate_judge_layout.py`:

```python
def test_cleanup_removes_legacy(tmp_path):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    mig.run(cache_root=cache, phase="copy", limit=None)
    # Pre-cleanup: both legacy and bundle exist.
    assert (explore / "grade.json").exists()
    assert (explore / "judge").exists()
    mig.run(cache_root=cache, phase="cleanup", limit=None)
    # Post-cleanup: legacy gone, bundle intact.
    assert not (explore / "grade.json").exists()
    assert not (explore / "judge").exists()
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    assert (bundle / "grade.json").exists()
    assert (bundle / "input.md").exists()


def test_cleanup_skips_explore_with_incomplete_bundle(tmp_path, capsys):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    mig.run(cache_root=cache, phase="copy", limit=None)
    # Corrupt the bundle: delete config.json.
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    (bundle / "config.json").unlink()
    mig.run(cache_root=cache, phase="cleanup", limit=None)
    # Cleanup must skip; legacy must still exist.
    assert (explore / "grade.json").exists()
    assert (explore / "judge").exists()
    captured = capsys.readouterr()
    assert "skipped" in captured.out.lower()


def test_cleanup_no_op_if_not_copied(tmp_path):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    # Run cleanup WITHOUT having run copy first.
    mig.run(cache_root=cache, phase="cleanup", limit=None)
    # Legacy must remain (no bundle to validate).
    assert (explore / "grade.json").exists()
    assert (explore / "judge").exists()
```

- [ ] **Step 10.2: Run tests to verify they fail**

```bash
conda run -n explain --no-capture-output pytest tests/test_migrate_judge_layout.py -v -k cleanup
```

Expected: NotImplementedError on cleanup phase.

- [ ] **Step 10.3: Implement cleanup phase**

Replace `_phase_cleanup`:

```python
def _validate_bundle(bundle: Path, expected_model: str) -> bool:
    """All 5 files present and config.json matches expected."""
    required = ["config.json", "grade.json", "input.md", "output.md", "result.json"]
    for fname in required:
        if not (bundle / fname).exists():
            return False
    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    if cfg.get("model") != expected_model:
        return False
    if cfg.get("name") != "claude":
        return False
    return True


def _phase_cleanup(cache_root: Path, limit: int | None):
    deleted = 0
    skipped = 0
    for explore_dir in _explore_dirs(cache_root):
        if limit is not None and deleted >= limit:
            break
        grade_path = explore_dir / "grade.json"
        if not grade_path.exists():
            continue
        data = json.loads(grade_path.read_text(encoding="utf-8"))
        judge_model = data["judge_model"]
        label = _label(judge_model)
        bundle = explore_dir / "judges" / label
        if not bundle.exists() or not _validate_bundle(bundle, judge_model):
            print(f"  skipped (incomplete bundle): {explore_dir}")
            skipped += 1
            continue
        # Safe to delete legacy.
        grade_path.unlink()
        legacy_judge = explore_dir / "judge"
        if legacy_judge.exists():
            shutil.rmtree(legacy_judge)
        deleted += 1
    print(f"Phase cleanup: {deleted} legacy entries removed, {skipped} skipped.")
```

- [ ] **Step 10.4: Run tests to verify pass**

```bash
conda run -n explain --no-capture-output pytest tests/test_migrate_judge_layout.py -v
```

Expected: All migration tests pass.

- [ ] **Step 10.5: Commit**

```bash
git add scripts/migrate_judge_layout.py tests/test_migrate_judge_layout.py
git commit -m "feat(scripts): migrate cleanup phase with bundle validation"
```

---

## Task 11: Update YAML configs to include judge: block

**Files:**
- Modify: every `scripts/<bench>/<model>/<bench>_<model>_<method>.yaml` for `bench in {hle, babyvision, rbenchv}`.

Use this command to enumerate first:

```bash
find /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/{hle,babyvision,rbenchv} -name "*.yaml" -not -name "*precache*" 2>&1
```

- [ ] **Step 11.1: Enumerate the touch list**

Run the find command above. Print the list. There should be on the order of 10-20 YAML files. Capture them as the touch list.

- [ ] **Step 11.2: Add `judge:` block to each non-precache YAML**

For each YAML in the touch list:

If the script is for HLE / BabyVision / RBenchV with a Claude judge (the dominant case today), append to the `benchmark:` block:

```yaml
benchmark:
  name: hle
  subset: gold
  judge:                                  # judge: explicit YAML block (was hardcoded class attr in benchmarks/hle.py before 2026-05-01)
    name: claude
    model: claude-haiku-4-5-20251001
```

If the script is for `qwen36_*_exp_orch` and the user wants the vLLM-served Qwen36 judge, swap to:

```yaml
  judge:
    name: vllm
    model: qwen36-35b-a3b-fp8
    sampling:
      temperature: 0.6
      top_p: 0.95
      max_tokens: 4096
      extra_body:
        chat_template_kwargs:
          enable_thinking: true
```

For each YAML, add a `# ...` comment explaining why this judge was chosen (e.g. `# judge: same Haiku judge as previously hardcoded; switching this is a research decision, not a config tweak`).

- [ ] **Step 11.3: Verify each YAML loads under the new schema**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output python -c "
import sys
from pathlib import Path
from eval import load_config, EvalConfig
for yp in Path('scripts').rglob('*.yaml'):
    if 'precache' in yp.name or '_template' in yp.name:
        continue
    try:
        cfg = load_config(str(yp), EvalConfig)
        print('OK', yp)
    except Exception as e:
        print('FAIL', yp, '->', repr(e))
        sys.exit(1)
"
```

Expected: every non-precache YAML for hle/babyvision/rbenchv loads cleanly. LCB / GPQA / AIME YAMLs also load cleanly (they don't carry `judge:`).

- [ ] **Step 11.4: Commit**

```bash
git add scripts/
git commit -m "feat(scripts): add explicit judge: block to HLE/BabyVision/RBenchV YAMLs"
```

---

## Task 12: Pilot migration on HLE Qwen36 cache (5 explores)

**Files:** none (operational task)

- [ ] **Step 12.1: Dry-run on the full HLE cache**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output python scripts/migrate_judge_layout.py \
    --phase dry-run \
    --cache-root /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/qwen36_35b_a3b_fp8/gold
```

Expected: prints `Total: 732 explores eligible for migration; 0 already migrated; <K> skipped (no grade.json).`

If the count differs significantly from 732, STOP and inspect.

- [ ] **Step 12.2: Pilot copy with --limit 5**

```bash
conda run -n explain --no-capture-output python scripts/migrate_judge_layout.py \
    --phase copy --limit 5 \
    --cache-root /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/qwen36_35b_a3b_fp8/gold
```

Expected: prints `Phase copy: 5 migrated, 0 already migrated.`

- [ ] **Step 12.3: Manually inspect one of the 5 migrated bundles**

```bash
ls /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/qwen36_35b_a3b_fp8/gold/<one-qid>/explore_1/judges/claude__claude-haiku-4-5-20251001/
diff <(cat /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/qwen36_35b_a3b_fp8/gold/<one-qid>/explore_1/grade.json) \
     <(cat /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/qwen36_35b_a3b_fp8/gold/<one-qid>/explore_1/judges/claude__claude-haiku-4-5-20251001/grade.json)
```

Expected: the bundle has 5 files (config.json + 4 copies); grade.json bytes are identical to the original.

- [ ] **Step 12.4: Spot-check an eval read against the pilot**

Run a 5-question HLE eval against the cache. Pick 5 question IDs that are in the pilot batch:

```bash
# Identify the migrated qids first:
for d in /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/qwen36_35b_a3b_fp8/gold/*/explore_1/judges/claude__claude-haiku-4-5-20251001; do
    echo "$(basename $(dirname $(dirname $(dirname $d))))"
done | head -5
```

Then construct a one-off YAML config that subsets HLE gold to just those 5 qids (or use the existing `_exp_orch` config and run `num: 5`), launch eval, and verify in the log that judge cost is $0.

If the pilot succeeds (all 5 cache hits, $0 judge cost), proceed. If not, STOP and inspect the bundle vs the eval reader.

- [ ] **Step 12.5: Full copy phase on HLE Qwen36 cache**

```bash
conda run -n explain --no-capture-output python scripts/migrate_judge_layout.py \
    --phase copy \
    --cache-root /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/qwen36_35b_a3b_fp8/gold
```

Expected: `Phase copy: 727 migrated, 5 already migrated.` (5 from the pilot.)

- [ ] **Step 12.6: Smoke eval (full 100-question HLE _exp_orch) — judge cost MUST be $0**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
    --config scripts/hle/grpo/hle_qwen36_35b_a3b_exp_orch.yaml \
    > tmp/eval_hle_judge_block_smoke.log 2>&1 &
```

Tail the log; verify the final summary shows `Judge cost: $0.0` (or close to it; only freshly graded explores would incur cost). Verify accuracy matches the prior run's 23/100.

If smoke eval shows judge cost > $0.10, the bundle isn't being read correctly — STOP and debug before cleanup.

- [ ] **Step 12.7: Cleanup phase (only after smoke eval passes)**

```bash
conda run -n explain --no-capture-output python scripts/migrate_judge_layout.py \
    --phase cleanup \
    --cache-root /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/qwen36_35b_a3b_fp8/gold
```

Expected: `Phase cleanup: 732 legacy entries removed, 0 skipped.`

- [ ] **Step 12.8: Commit (if anything changed in scripts)**

If you edited the migration script during pilot debugging:

```bash
git add scripts/migrate_judge_layout.py
git commit -m "fix(scripts): tweaks from pilot run"
```

If not, no commit; the cache is data, not code.

---

## Task 13: Migrate other caches and validate end-to-end

**Files:** none (operational)

- [ ] **Step 13.1: Enumerate remaining caches**

```bash
ls -d /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/*/* 2>&1
```

Expected: a list including HLE Sonnet (already migrated previously per CLAUDE.md note), HLE Opus, and any per-benchmark caches that exist.

- [ ] **Step 13.2: For each remaining cache, run dry-run + copy + smoke + cleanup**

For each cache directory:

```bash
ROOT=/data3/peijia/dr-claw/Explain/Experiment/analysis/cache/<bench>/<model>[/<filter>]

conda run -n explain --no-capture-output python scripts/migrate_judge_layout.py --phase dry-run --cache-root $ROOT
# inspect counts, then:
conda run -n explain --no-capture-output python scripts/migrate_judge_layout.py --phase copy --cache-root $ROOT
# Run a smoke eval against it; verify $0 judge cost.
# Only after smoke passes:
conda run -n explain --no-capture-output python scripts/migrate_judge_layout.py --phase cleanup --cache-root $ROOT
```

- [ ] **Step 13.3: Run the full unit test suite**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 13.4: Final acceptance: re-run HLE _exp_orch eval, verify**

```bash
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
    --config scripts/hle/grpo/hle_qwen36_35b_a3b_exp_orch.yaml \
    > tmp/eval_hle_post_migration.log 2>&1 &
```

Acceptance criteria:
- Final accuracy matches the pre-migration 23/100 result.
- `Judge cost: $0.0` in the run summary (or $0.something tiny if a few explores re-graded due to rare timing windows; investigate if > $0.50).
- Run dir contains a fresh `judges/claude__claude-haiku-4-5-20251001/` bundle in each per-run grading explore_N (the per-run dirs are written fresh, in new layout, by the migrated `eval.py`).

- [ ] **Step 13.5: Final commit (if any)**

```bash
git status  # should be clean
```

---

## Self-Review

Spec coverage check (run after writing the plan):

- **YAML schema (spec section "YAML schema"):** Tasks 1-2 cover JudgeSpec definition + embedding into HLE/BabyVision/RBenchV. Task 11 updates existing YAML files.
- **Filesystem layout (spec section "Filesystem layout"):** Task 7 implements the new layout in `_grade_with_cache`.
- **Lookup logic (spec section "Lookup logic"):** Task 4 implements `find_cached_judge` + `judge_label`.
- **Migration: pure mv -> NO, copy-then-cleanup per spec safety section:** Tasks 8-10 implement the three-phase script. Task 12 pilots and runs the HLE migration. Task 13 covers other caches.
- **Code changes table:** Each row of the spec's table corresponds to a task: specs.py = T1+T2; base.py = T3+T4; grader.py = T5; hle/babyvision/rbenchv.py = T6; gpqa/lcb/aime.py = T6; eval.py = T7; precache_explores.py = T3 (pass-through); migrate_judge_layout.py = T8+T9+T10; tests = T1, T4, T5, T8, T9, T10. YAMLs = T11.
- **Validation criteria (spec section "Validation"):**
  - Migration correctness: T12.3 (manual diff) + T12.6 (smoke eval).
  - Cache reuse: T12.6 ($0 judge cost).
  - Multi-judge coexistence: not explicitly tested in this plan; add a test in T13 if user wants. Acceptable to defer (user can run a vllm-judge eval on the migrated cache and observe a new bundle appears).
  - Schema rejection (LCB rejects judge): T2.1 test.
  - Label collision: T4.1 test.

Type consistency:
- `judge_spec: dict | None` used consistently across base.py, grader.py, eval.py.
- `find_cached_judge(judges_dir: Path, judge_spec: dict) -> Path | None` consistent with calls in T7.
- `judge_label(judge_spec: dict) -> str` consistent with calls in T7.

Placeholder scan:
- T11.1 has `<one-qid>` and `<bench>/<model>` placeholders. Acceptable: these are runtime values the engineer fills at execution time.
- T13.2 has `$ROOT` template. Acceptable: this is a shell variable.
- No "TBD", "TODO", "implement later" anywhere.

Open notes (not gaps; defer-friendly):
- The `SamplingConfig` import cycle in T1.3 has two strategies; pick at implementation time. If neither works, factor into a third module (small refactor).
- Multi-judge coexistence test is not explicit in tests; add only if user requests.
