# YAML-Discriminated-Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `benchmark: str` + top-level `filters: dict` into a single discriminated-union `benchmark:` block, and reduce eval/precache CLI to two flags (`--config`, `-o`) — making YAML the single source of truth.

**Architecture:** A new `benchmarks/specs.py` defines one Pydantic sub-schema per benchmark (HLE, GPQA, LCB, BabyVision, RBenchV, AIME2025, AIME2026), unified as a `Field(discriminator="name")` union. `EvalConfig` and `PrecacheConfig` replace the two old fields with one `benchmark: BenchmarkSpec` field. `parse_cli` in `eval.py` and `precache_explores.py` drops every flat flag including `--benchmark`; the only inputs are `--config FILE.yaml` and `-o key.path=value`. All 94 YAMLs are rewritten to the nested shape; all 94 shell scripts are stripped of `--benchmark <X>`. `BenchmarkConfig.add_dataset_args` / `add_model_args` / `make_filter_model` / `filter_keys` are deleted from the base class and 6 subclasses. `filter_dataset(rows, **kwargs)` keeps its signature (the spec's fields are already the kwargs).

**Tech Stack:** Pydantic v2 (discriminated unions via `Annotated[Union[...], Field(discriminator=...)]`), PyYAML, argparse (slim).

---

## Pre-Flight

- `git status` must be on the migration commits we already made (T1-T6 of the previous plan). The 94 YAMLs in `Experiment/core_code/configs/` already exist; they just have the wrong shape (top-level `filters:`).
- All 6 concrete benchmarks: `benchmarks/{aime,babyvision,gpqa,hle,lcb,rbenchv}.py`. Note `aime.py` defines BOTH `AIME2025Benchmark` and `AIME2026Benchmark` — they share `filter_keys = ("year",)`.
- The existing 25-test `tests/test_eval_config.py` and 6-test `tests/test_precache_config.py` ALL use the old `{benchmark: str, filters: {...}}` shape. They must be rewritten in-place when the schema changes (Tasks 2 and 3).
- Tests must complete in <60s. Subprocess-spawning tests are forbidden.

---

## Task 1: Add `BenchmarkSpec` discriminated-union module

**Files:**
- Create: `Experiment/core_code/benchmarks/specs.py`
- Test: `Experiment/core_code/tests/test_benchmark_specs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark_specs.py
from __future__ import annotations
import pytest
from pydantic import BaseModel, ValidationError
from benchmarks.specs import (
    BenchmarkSpec, HLESpec, GPQASpec, LCBSpec, BabyVisionSpec,
    RBenchVSpec, AIME2025Spec, AIME2026Spec,
)


class _Holder(BaseModel):
    benchmark: BenchmarkSpec


def test_hle_full():
    h = _Holder.model_validate({"benchmark": {"name": "hle", "subset": "gold", "text_only": True}})
    assert isinstance(h.benchmark, HLESpec)
    assert h.benchmark.subset == "gold"
    assert h.benchmark.text_only is True


def test_hle_minimal():
    h = _Holder.model_validate({"benchmark": {"name": "hle"}})
    assert isinstance(h.benchmark, HLESpec)
    assert h.benchmark.subset is None
    assert h.benchmark.text_only is False


def test_gpqa_domain():
    h = _Holder.model_validate({"benchmark": {"name": "gpqa", "domain": "physics"}})
    assert isinstance(h.benchmark, GPQASpec)
    assert h.benchmark.domain == "physics"


def test_lcb_difficulty():
    h = _Holder.model_validate({"benchmark": {"name": "lcb", "difficulty": "medium"}})
    assert isinstance(h.benchmark, LCBSpec)
    assert h.benchmark.difficulty == "medium"


def test_babyvision_type_subtype():
    h = _Holder.model_validate({"benchmark": {"name": "babyvision", "type": "ansType", "subtype": "choice"}})
    assert isinstance(h.benchmark, BabyVisionSpec)


def test_rbenchv_category():
    h = _Holder.model_validate({"benchmark": {"name": "rbenchv", "category": "Physics"}})
    assert isinstance(h.benchmark, RBenchVSpec)


def test_aime2025_year():
    h = _Holder.model_validate({"benchmark": {"name": "aime2025", "year": 2025}})
    assert isinstance(h.benchmark, AIME2025Spec)


def test_aime2026_year():
    h = _Holder.model_validate({"benchmark": {"name": "aime2026", "year": 2026}})
    assert isinstance(h.benchmark, AIME2026Spec)


def test_unknown_name_rejected():
    with pytest.raises(ValidationError):
        _Holder.model_validate({"benchmark": {"name": "nosuch"}})


def test_extra_filter_key_rejected():
    with pytest.raises(ValidationError):
        _Holder.model_validate({"benchmark": {"name": "hle", "domain": "x"}})


def test_missing_name_rejected():
    with pytest.raises(ValidationError):
        _Holder.model_validate({"benchmark": {"subset": "gold"}})
```

- [ ] **Step 2: Run failing test**

Run: `cd Experiment/core_code && pytest tests/test_benchmark_specs.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'benchmarks.specs'`.

- [ ] **Step 3: Create `benchmarks/specs.py`**

```python
"""Benchmark spec sub-schemas: discriminated union over benchmark name.

Replaces both EvalConfig.benchmark (str) and EvalConfig.filters (dict). Each
benchmark gets one Pydantic class enumerating exactly its valid filter keys.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class _Spec(BaseModel):
    model_config = {"extra": "forbid"}


class HLESpec(_Spec):
    name: Literal["hle"]
    subset: Literal["gold", "revision", "uncertain"] | None = None
    category: str | None = None
    text_only: bool = False


class GPQASpec(_Spec):
    name: Literal["gpqa"]
    domain: str | None = None


class LCBSpec(_Spec):
    name: Literal["lcb"]
    difficulty: str | None = None


class BabyVisionSpec(_Spec):
    name: Literal["babyvision"]
    type: str | None = None
    subtype: str | None = None


class RBenchVSpec(_Spec):
    name: Literal["rbenchv"]
    category: str | None = None


class AIME2025Spec(_Spec):
    name: Literal["aime2025"]
    year: int | None = None


class AIME2026Spec(_Spec):
    name: Literal["aime2026"]
    year: int | None = None


BenchmarkSpec = Annotated[
    Union[
        HLESpec, GPQASpec, LCBSpec, BabyVisionSpec, RBenchVSpec,
        AIME2025Spec, AIME2026Spec,
    ],
    Field(discriminator="name"),
]
```

- [ ] **Step 4: Run passing test**

Run: `cd Experiment/core_code && pytest tests/test_benchmark_specs.py -v`
Expected: 11 passed in <2s.

- [ ] **Step 5: Commit**

```bash
git add Experiment/core_code/benchmarks/specs.py Experiment/core_code/tests/test_benchmark_specs.py
git commit -m "feat(benchmarks): add BenchmarkSpec discriminated union"
```

---

## Task 2: Refactor `EvalConfig` to use `BenchmarkSpec`

**Files:**
- Modify: `Experiment/core_code/eval_config.py`
- Modify: `Experiment/core_code/tests/test_eval_config.py` (rewrite all 25 tests for new shape)

- [ ] **Step 1: Update `EvalConfig` to embed `BenchmarkSpec`**

Replace lines 23-119 of `eval_config.py` with:

```python
class EvalConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": False}

    # Benchmark + dataset selection (single discriminated block)
    from benchmarks.specs import BenchmarkSpec  # forward import is fine here
    benchmark: BenchmarkSpec
    backend: Literal["codex", "claude", "vllm"]
    explore_model: str

    # Method selection
    method: METHODS = "tts-agent"
    orchestrator_model: str | None = None
    integrate_model: str | None = None
    reward_model: str | None = None

    # Cache
    cache_dir: Path | None = None
    cache_dirs: dict[str, Path] = Field(default_factory=dict)

    # Multi/effort budgets
    model_budgets: dict[str, int] = Field(default_factory=dict)
    effort_budgets: dict[str, int] = Field(default_factory=dict)
    exploration_effort: Literal["low", "medium", "high"] | None = None

    # Dataset slicing
    num: int | None = None
    skip: int = 0
    seed: int = 42
    shuffle: bool = False

    # Model knobs
    budget_tokens: int = 32000
    effort: Literal["low", "medium", "high", "max"] = "low"
    num_explores: int = 8
    num_workers: int = 1
    explore_timeout: float = 1200.0
    max_output_chars: int | None = None

    # Run
    verbose: bool = False
    resume: str | None = None
    log_dir: str = "logs"
    no_cache_only: bool = False
    timeout: float = 1200.0
    no_integrate: bool = False
    num_rollouts: int = 1

    @model_validator(mode="after")
    def _validate_method_constraints(self):
        m = self.method
        if m == "tts-agent-multi":
            assert self.orchestrator_model, "tts-agent-multi requires orchestrator_model"
            assert self.cache_dirs, "tts-agent-multi requires cache_dirs (dict[model_alias, path])"
            assert self.model_budgets, "tts-agent-multi requires model_budgets"
        elif m == "tts-agent-effort":
            assert self.orchestrator_model, "tts-agent-effort requires orchestrator_model"
            assert self.cache_dirs, "tts-agent-effort requires cache_dirs (dict[level, path])"
            assert self.effort_budgets, "tts-agent-effort requires effort_budgets"
        elif m == "tts-agent":
            assert self.orchestrator_model, "tts-agent requires orchestrator_model"
            assert self.integrate_model, "tts-agent requires integrate_model"
        elif m == "rerank":
            assert self.reward_model, "rerank requires reward_model"
        elif m == "standalone-integrator":
            assert self.integrate_model, "standalone-integrator requires integrate_model"

        is_multi = m in ("tts-agent-multi", "tts-agent-effort")
        if is_multi:
            assert not self.cache_dir, f"{m} uses cache_dirs (dict), not cache_dir (single)"
        else:
            assert not self.cache_dirs, (
                f"{m} uses cache_dir (single Path), not cache_dirs (dict). "
                f"got cache_dirs={dict(self.cache_dirs)}"
            )

        if self.num_rollouts > 1:
            assert self.method == "tts-agent", (
                f"num_rollouts > 1 only supported for method=tts-agent, got {self.method}"
            )
            assert self.backend == "vllm", (
                f"num_rollouts > 1 only supported for backend=vllm, got {self.backend}"
            )
        assert self.num_rollouts >= 1, f"num_rollouts must be >= 1, got {self.num_rollouts}"
        return self
```

Also delete the `from benchmarks import get_benchmark` line and the `make_filter_model` / `model_dump(exclude_defaults=True)` filter-validation block at the end of the validator — the discriminated union takes over both jobs.

- [ ] **Step 2: Rewrite the eval_config tests**

`tests/test_eval_config.py` currently uses `{benchmark: "hle", filters: {subset: "gold"}}` in 25 places. Rewrite each test's input dict to `{benchmark: {name: "hle", subset: "gold"}}`. Delete tests that target dead concepts:

- `test_filter_unknown_key_rejected` → still valid, keep but rewrite to check the `extra="forbid"` rejection on the spec sub-model
- `test_dot_overrides_beat_yaml`, `test_o_override_beats_cli_filter_flag` → keep, use new override path `-o benchmark.subset=...`
- Any test asserting `cfg.filters` → rewrite to assert `cfg.benchmark.subset` etc.
- Any test that relied on `benchmark: str` directly → rewrite to access `cfg.benchmark.name`

Read the existing file once with the Read tool, then issue one Edit-tool replacement per test or one full Write to overwrite.

- [ ] **Step 3: Run tests**

Run: `cd Experiment/core_code && pytest tests/test_eval_config.py tests/test_benchmark_specs.py -v`
Expected: all 36 (25+11) pass in <5s. If a test failure says `'filters' is forbidden`, that test still uses the old shape — finish rewriting it.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/eval_config.py Experiment/core_code/tests/test_eval_config.py
git commit -m "refactor(EvalConfig): embed BenchmarkSpec; drop top-level filters"
```

---

## Task 3: Refactor `PrecacheConfig` to use `BenchmarkSpec`

**Files:**
- Modify: `Experiment/core_code/precache_config.py`
- Modify: `Experiment/core_code/tests/test_precache_config.py`

- [ ] **Step 1: Update `PrecacheConfig`**

Replace the file contents:

```python
"""Pydantic schema for precache_explores.py configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from benchmarks.specs import BenchmarkSpec


class PrecacheConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": False}

    benchmark: BenchmarkSpec
    backend: Literal["codex", "claude", "vllm"]
    explore_model: str
    cache_dir: Path

    num_explores: int = 8
    num_workers: int = 1
    num: int | None = None
    skip: int = 0
    seed: int = 42
    shuffle: bool = False
    budget_tokens: int = 32000
    effort: Literal["low", "medium", "high", "max"] = "low"
    explore_timeout: float = 1200.0
```

Delete the old `_validate_filters` method entirely — the discriminated union handles validation.

- [ ] **Step 2: Rewrite `tests/test_precache_config.py`**

Change every `"benchmark": "hle"` to `"benchmark": {"name": "hle"}`, and pull `subset` / `text_only` out of the (now deleted) `filters` block into the benchmark block. Verify the `extra="forbid"` test still rejects unknown top-level keys.

- [ ] **Step 3: Run tests**

Run: `cd Experiment/core_code && pytest tests/test_precache_config.py -v`
Expected: 6 passed in <2s.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/precache_config.py Experiment/core_code/tests/test_precache_config.py
git commit -m "refactor(PrecacheConfig): embed BenchmarkSpec; drop filter validator"
```

---

## Task 4: Slim `eval.py:parse_cli` to two flags

**Files:**
- Modify: `Experiment/core_code/eval.py:603-680` (parse_cli) and downstream usages of `cfg.benchmark` (find via grep).

- [ ] **Step 1: Replace `parse_cli` body**

Overwrite lines 603-680 with:

```python
def parse_cli() -> "EvalConfig":
    """Build EvalConfig from --config FILE.yaml + -o dot.path=value overrides.

    YAML is the single source of truth. The only CLI flags are --config and -o.
    """
    from eval_config import load_config, EvalConfig

    parser = argparse.ArgumentParser(description="Evaluate TTS agent")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="Dot-path override, e.g. -o benchmark.subset=gold")
    args = parser.parse_args()

    return load_config(
        config_path=args.config,
        dot_overrides=list(args.override),
        schema=EvalConfig,
    )
```

- [ ] **Step 2: Update the rest of `eval.py` for the new `cfg.benchmark` shape**

`cfg.benchmark` is now a Pydantic spec model, not a string. Find every site:

Run: `cd Experiment/core_code && grep -n "cfg\.benchmark\|cfg\.filters" eval.py`

For each hit:
- `get_benchmark(cfg.benchmark)` → `get_benchmark(cfg.benchmark.name)`
- `benchmark.filter_dataset(rows, **cfg.filters)` → `benchmark.filter_dataset(rows, **cfg.benchmark.model_dump(exclude={"name"}))`
- Any logging string `f"benchmark={cfg.benchmark}"` → `f"benchmark={cfg.benchmark.name}"`

Apply the edits with the Edit tool.

- [ ] **Step 3: Smoke-load one config**

Run: `cd Experiment/core_code && python -c "
import sys; sys.argv = ['eval.py', '--config', 'configs/aime2025_sonnet_precache.yaml']
"` — this won't work because YAMLs are still old shape. **Smoke deferred to Task 7** after YAML migration.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/eval.py
git commit -m "refactor(eval): parse_cli accepts only --config and -o"
```

---

## Task 5: Slim `precache_explores.py:parse_cli` and call sites

**Files:**
- Modify: `Experiment/core_code/precache_explores.py:133-180` (parse_cli) and main() body uses of `cfg.benchmark`/`cfg.filters`.

- [ ] **Step 1: Replace `parse_cli`**

```python
def parse_cli() -> "PrecacheConfig":
    """Build PrecacheConfig from --config + -o overrides only."""
    from precache_config import PrecacheConfig
    from eval_config import load_config

    parser = argparse.ArgumentParser(description="Pre-cache explore rollouts")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="Dot-path override, e.g. -o benchmark.subset=gold")
    args = parser.parse_args()

    return load_config(
        config_path=args.config,
        dot_overrides=list(args.override),
        schema=PrecacheConfig,
    )
```

- [ ] **Step 2: Update main() call sites**

Run: `cd Experiment/core_code && grep -n "cfg\.benchmark\|cfg\.filters" precache_explores.py`

Update:
- `get_benchmark(cfg.benchmark)` → `get_benchmark(cfg.benchmark.name)`
- `benchmark.filter_dataset(all_rows, **cfg.filters)` → `benchmark.filter_dataset(all_rows, **cfg.benchmark.model_dump(exclude={"name"}))`

- [ ] **Step 3: Commit**

```bash
git add Experiment/core_code/precache_explores.py
git commit -m "refactor(precache): parse_cli accepts only --config and -o"
```

---

## Task 6: Migrate the 94 YAMLs to nested benchmark form

**Files:**
- Modify: every file in `Experiment/core_code/configs/*.yaml` except `_template.yaml`.

For each YAML, transform:

```yaml
benchmark: hle
# ... other top-level keys ...
filters:
  subset: gold
  text_only: true
```

into:

```yaml
benchmark:
  name: hle
  subset: gold
  text_only: true
# ... other top-level keys (unchanged) ...
```

Files with no `filters:` block (e.g. `aime2026_sonnet_precache.yaml`) just rewrap `benchmark: aime2026` → `benchmark:\n  name: aime2026`.

- [ ] **Step 1: Verify the count**

Run: `cd Experiment/core_code && ls configs/*.yaml | wc -l`
Expected: 94 (or 93 + `_template.yaml`).

- [ ] **Step 2: Edit each YAML manually with Read+Edit**

For each file, Read it once, then issue one Edit replacing the `benchmark: <name>` line and the `filters:` block (if any) with the new nested form. The 76 newly-created YAMLs from the prior migration follow a tight template; the 18 pre-existing ones (e.g. `hle_multi_delegated.yaml`) may have additional comments — preserve them.

Concrete example, `configs/hle_sonnet_precache.yaml`:

Old:
```yaml
benchmark: hle
backend: claude
explore_model: claude-sonnet-4-6
effort: low
cache_dir: ../analysis/cache/hle/sonnet/gold
num: 100
num_explores: 8
num_workers: 8
seed: 42
filters:
  subset: gold
  text_only: true
```

New:
```yaml
benchmark:
  name: hle
  subset: gold
  text_only: true
backend: claude
explore_model: claude-sonnet-4-6
effort: low
cache_dir: ../analysis/cache/hle/sonnet/gold
num: 100
num_explores: 8
num_workers: 8
seed: 42
```

- [ ] **Step 3: Validate every YAML against the new schema**

Run:

```bash
cd Experiment/core_code && python -c "
from pathlib import Path
import yaml
from precache_config import PrecacheConfig
from eval_config import EvalConfig, load_config

failures = []
for yp in sorted(Path('configs').glob('*.yaml')):
    if yp.name == '_template.yaml':
        continue
    data = yaml.safe_load(yp.read_text()) or {}
    is_eval = any(k in data for k in ('method', 'orchestrator_model', 'reward_model', 'no_integrate'))
    schema = EvalConfig if is_eval else PrecacheConfig
    try:
        load_config(config_path=str(yp), dot_overrides=[], schema=schema)
    except Exception as e:
        failures.append((str(yp), str(e)[:200]))

if failures:
    for p, e in failures:
        print(f'FAIL {p}: {e}')
    raise SystemExit(1)
print(f'OK: {len(list(Path(\"configs\").glob(\"*.yaml\"))) - 1} YAMLs validate.')
"
```

Expected: `OK: 93 YAMLs validate.` Total runtime <60s (load_config is fast; this is just YAML parse + Pydantic validate).

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/configs/*.yaml
git commit -m "migrate(configs): nest filters under benchmark block (94 YAMLs)"
```

---

## Task 7: Strip `--benchmark <X>` from all 94 shell scripts

**Files:**
- Modify: every file in `Experiment/core_code/scripts/**/*.sh` that invokes `eval.py` or `precache_explores.py`.

- [ ] **Step 1: Enumerate**

Run: `cd Experiment/core_code && grep -rl -- '--benchmark' scripts/ | wc -l`
Expected: 94.

- [ ] **Step 2: Remove the line**

Each script currently has a fragment like:

```bash
PYTHONUNBUFFERED=1 nohup python eval.py \
	--benchmark hle \
	--config configs/hle_sonnet_precache.yaml \
	> ... 2>&1 &
```

Edit each one to drop the `--benchmark hle \` line entirely:

```bash
PYTHONUNBUFFERED=1 nohup python eval.py \
	--config configs/hle_sonnet_precache.yaml \
	> ... 2>&1 &
```

For each file: Read it, then issue one Edit replacing the `--benchmark <X> \\\n\t` fragment with `` (empty). The fragment is unique per file because no other line carries `--benchmark`.

- [ ] **Step 3: Verify no script still passes `--benchmark`**

Run: `cd Experiment/core_code && grep -rl -- '--benchmark' scripts/ | wc -l`
Expected: 0.

- [ ] **Step 4: Spot-check 3 scripts dry-parse end-to-end**

Run:

```bash
cd Experiment/core_code && python -c "
import sys
for cfg in [
    'configs/aime2025_sonnet_precache.yaml',
    'configs/hle_sonnet_delegated.yaml',
    'configs/lcb_grpo_8b.yaml',
]:
    sys.argv = ['x', '--config', cfg]
    if 'precache' in cfg:
        from precache_explores import parse_cli
    else:
        from eval import parse_cli
    cfg_obj = parse_cli()
    print(cfg, 'OK', cfg_obj.benchmark.name)
"
```

Expected: 3 OK lines printed in <10s.

- [ ] **Step 5: Commit**

```bash
git add Experiment/core_code/scripts
git commit -m "migrate(scripts): drop --benchmark flag (YAML is single source)"
```

---

## Task 8: Delete dead argparse / filter plumbing in benchmarks/

**Files:**
- Modify: `Experiment/core_code/benchmarks/base.py:328-461`
- Modify: `Experiment/core_code/benchmarks/aime.py`, `babyvision.py`, `gpqa.py`, `hle.py`, `lcb.py`, `rbenchv.py`

- [ ] **Step 1: Remove from `benchmarks/base.py`**

Delete:
- The `filter_keys: tuple[str, ...]` class attribute declaration (line 334)
- The `make_filter_model` method (line 425-433)
- The `add_dataset_args` method (line 435-443)
- The `add_model_args` method (line 445-461)
- The `import argparse` line if no other code uses it (it likely is unused after this).
- Update the docstring on `BenchmarkConfig` (line 326-331) to drop the "Subclasses must set ... filter_keys" sentence.

`filter_dataset(self, rows, **kwargs)` keeps its signature — kwargs now come from `cfg.benchmark.model_dump(exclude={"name"})`, which produces exactly the same keys as before.

- [ ] **Step 2: Remove from each subclass**

For each of the 6 benchmark files, delete:
- `filter_keys = (...)` class attribute
- `make_filter_model(self)` method
- `add_dataset_args(self, parser)` method (if defined; HLE has one, LCB has one, etc.)
- `add_model_args(self, parser)` (rare but check)
- `import argparse` if no longer used

Keep `_filter_dataset(...)` helper functions and the `filter_dataset` method (they take kwargs).

- [ ] **Step 3: Run all tests**

Run: `cd Experiment/core_code && pytest tests/ -v --timeout=30`
Expected: all tests pass; total wall time <60s.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/benchmarks
git commit -m "refactor(benchmarks): drop add_*_args / make_filter_model / filter_keys"
```

---

## Task 9: Update CLAUDE.md docs

**Files:**
- Modify: `Experiment/core_code/CLAUDE.md` — replace the "eval.py configuration" section.

- [ ] **Step 1: Replace the section**

Read `Experiment/core_code/CLAUDE.md`, locate the `## eval.py configuration` heading, and replace everything from that heading down to the next top-level `##` (or end of file) with:

````markdown
## eval.py configuration

`eval.py` and `precache_explores.py` accept exactly two CLI flags:

1. `--config configs/<name>.yaml` — required.
2. `-o key.path=value` — repeatable dot-path override, highest precedence.

The full schema lives in `Experiment/core_code/eval_config.py` (`EvalConfig`)
and `precache_config.py` (`PrecacheConfig`). Reference template:
`Experiment/core_code/configs/_template.yaml`.

### Benchmark selection lives inside the `benchmark:` block

Every YAML must declare `benchmark.name`; benchmark-specific filter fields
(subset, category, domain, difficulty, year, ...) sit alongside it under the
same key:

```yaml
# HLE with subset + text-only filter
benchmark:
  name: hle
  subset: gold
  text_only: true
backend: claude
explore_model: claude-sonnet-4-6
# ... rest of EvalConfig fields ...
```

```yaml
# RBenchV Physics
benchmark:
  name: rbenchv
  category: Physics
```

```yaml
# AIME 2026 with no filter
benchmark:
  name: aime2026
```

Valid `name` values: `hle`, `gpqa`, `lcb`, `babyvision`, `rbenchv`, `aime2025`,
`aime2026`. Each name picks a different sub-schema; misspelled or unknown
filter keys (e.g. `domain` on HLE) fail loudly at validation time.

### Override examples

```bash
# Single-path override
python eval.py --config configs/hle_sonnet_delegated.yaml \
  -o num=20 -o seed=99

# Filter override (note the dot path is benchmark.subset, not filters.subset)
python eval.py --config configs/hle_sonnet_delegated.yaml \
  -o benchmark.subset=revision

# Per-launch resume (kept out of the YAML so the YAML stays reusable)
python eval.py --config configs/lcb_sonnet_delegated.yaml \
  -o resume=../analysis/run/lcb/sonnet/run_20260308_230222
```

### Single-cache vs multi-cache

The schema has TWO mutually-exclusive cache fields, picked by method:

- `cache_dir: Path | None` — for `tts-agent`, `self-refine`, `socratic-self-refine`,
  `budget-forcing`, `rerank`, `standalone-integrator`.
- `cache_dirs: dict[str, Path]` — for `tts-agent-multi`, `tts-agent-effort`.

A pydantic validator asserts you do not mix them.

### Dict-typed fields are YAML/`-o` only

`cache_dirs` (multi-model), `model_budgets`, `effort_budgets` are dictionaries.
Set them via YAML mapping or per-key dot-path:

```yaml
cache_dirs:
  haiku: ../analysis/cache/hle/haiku/gold
  sonnet: ../analysis/cache/hle/sonnet/gold
model_budgets:
  haiku: 8
  sonnet: 8
```

```bash
python eval.py --config configs/hle_multi_delegated.yaml \
  -o model_budgets.haiku=4 -o cache_dirs.haiku=/new/path
```
````

- [ ] **Step 2: Commit**

```bash
git add Experiment/core_code/CLAUDE.md
git commit -m "docs: rewrite eval.py configuration section for 2-flag YAML CLI"
```

---

## Task 10: Update spec doc and supersede the old plan

**Files:**
- Modify: `docs/superpowers/specs/2026-04-29-yaml-only-cli-migration.md`
- Modify: `docs/superpowers/plans/2026-04-29-yaml-only-cli-migration.md` (mark superseded)

- [ ] **Step 1: Add an "Update 2026-04-29: 2-flag form" section to the spec**

At the top of the spec under the goal statement, prepend:

```markdown
> **Update 2026-04-29:** During execution we discovered `--benchmark` on the CLI
> duplicates the YAML's `benchmark:` field. The final design is **two flags
> only**: `--config` and `-o`. The `benchmark:` field is also restructured as a
> discriminated union with each benchmark's filters nested inside it (see
> `benchmarks/specs.py`). The corrected plan lives in
> `docs/superpowers/plans/2026-04-29-yaml-discriminated-benchmark.md`.
```

Then in the §Architecture and §CLI sections, replace any "3 flags" / "--benchmark required" wording with the 2-flag form and the nested-benchmark YAML example.

- [ ] **Step 2: Add a SUPERSEDED banner to the old plan**

Prepend to `docs/superpowers/plans/2026-04-29-yaml-only-cli-migration.md`:

```markdown
> **SUPERSEDED 2026-04-29:** Tasks T1-T6 of this plan are complete and committed.
> Tasks T7-T9 are obsolete because the architecture changed (CLI is 2 flags, not
> 3, and `filters:` is folded into `benchmark:`). See
> `docs/superpowers/plans/2026-04-29-yaml-discriminated-benchmark.md` for the
> in-flight plan.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-04-29-yaml-only-cli-migration.md \
        docs/superpowers/plans/2026-04-29-yaml-only-cli-migration.md
git commit -m "docs(spec): note 2-flag/discriminated-benchmark correction"
```

---

## Task 11: Final smoke run

- [ ] **Step 1: Pytest entire suite under 60s**

Run: `cd Experiment/core_code && pytest tests/ -v --timeout=30`
Expected: all tests pass; total wall time <60s.

- [ ] **Step 2: 5-script dry-parse spot check**

Run:

```bash
cd Experiment/core_code && python -c "
import sys
for cfg, kind in [
    ('configs/aime2025_sonnet_precache.yaml',  'precache'),
    ('configs/hle_sonnet_delegated.yaml',      'eval'),
    ('configs/lcb_grpo_8b.yaml',               'eval'),
    ('configs/rbenchv_sonnet_socratic_self_refine.yaml', 'eval'),
    ('configs/hle_multi_delegated.yaml',       'eval'),
]:
    sys.argv = ['x', '--config', cfg]
    mod = __import__('precache_explores' if kind == 'precache' else 'eval')
    obj = mod.parse_cli()
    print(cfg, '->', obj.benchmark.name, getattr(obj, 'method', '-'))
"
```

Expected: 5 lines, no exception, total runtime <15s.

- [ ] **Step 3: Verify no script regresses**

Run: `cd Experiment/core_code && grep -rl -- '--benchmark' scripts/`
Expected: empty.

Run: `cd Experiment/core_code && grep -rl '^filters:' configs/ | grep -v _template`
Expected: empty.

---

## Self-Review

**Spec coverage:** The spec asks for (a) YAML as single source of truth, (b) drop flat CLI flags, (c) per-benchmark filter validation. (a) and (b) → 2-flag CLI in Tasks 4-5. (c) → discriminated union in Task 1, used in Tasks 2-3. Old plan tasks T7-T9 covered by Tasks 4-5, 8, 9 here.

**Placeholder scan:** All steps have concrete code or concrete commands. No "TBD" / "etc." / "similar to X". Bulk YAML / shell-script edits in Tasks 6-7 are mechanical and explicitly demonstrate one before/after pair.

**Type consistency:** Field name `benchmark.name` used consistently across Task 2, 4, 5, 9; `cfg.benchmark.model_dump(exclude={"name"})` used identically in Tasks 4 and 5. `BenchmarkSpec` / `HLESpec` / `GPQASpec` / etc. names match between specs.py (Task 1) and tests (Task 1).

**Test runtime budget:** All `pytest` runs are unit-only (no subprocess spawn, no network). Total ~31 tests; budget <60s. The 94-YAML validation in Task 6 is also pure-python pydantic validation, not subprocess.
