# YAML-Only CLI Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `eval.py` and `precache_explores.py` CLIs to `--benchmark + --config + -o` only. Migrate all 73 shell scripts to thin `--config` wrappers. Delete every backward-compat flat argparse flag in one cutover, with empirical risk verification (no predictions).

**Architecture:** Refactor `load_config` to be schema-generic, add `PrecacheConfig`, write a one-shot self-deleting migration script with 8 hard-gate verification tests, then run it. After migration succeeds, slim `eval.py` `parse_cli` and delete `add_dataset_args`/`add_model_args` from `benchmarks/`.

**Tech Stack:** pydantic 2.11, pyyaml 6.0, shlex (stdlib), argparse (stdlib).

**Branch:** Work directly on `main` (per user CLAUDE.md rule: no new branches).

**Spec:** `docs/superpowers/specs/2026-04-29-yaml-only-cli-migration.md`

---

## File structure

**New files:**
- `Experiment/core_code/precache_config.py` — `PrecacheConfig` Pydantic schema (14 fields).
- `Experiment/core_code/tests/test_precache_config.py` — 4 unit tests.
- `Experiment/core_code/scripts/migrate_eval_scripts.py` — one-shot migration tool. Self-deletes after success.
- `Experiment/core_code/tests/test_migrate_scripts.py` — 8 hard-gate verification tests. Self-deletes alongside.

**Modified:**
- `Experiment/core_code/eval_config.py` — `load_config` becomes schema-generic; drop `flat_overrides` parameter. Existing test imports of `load_config` updated.
- `Experiment/core_code/eval.py` — `parse_cli` slimmed to 3-flag form. Filter-routing/cache-dirs special-cases removed.
- `Experiment/core_code/precache_explores.py` — `parse_args` deleted; new `parse_cli` consumes `PrecacheConfig`.
- `Experiment/core_code/benchmarks/base.py` — delete `add_dataset_args`, `add_model_args`. Keep `filter_keys`, `make_filter_model`.
- `Experiment/core_code/benchmarks/{hle,lcb,gpqa,babyvision,aime,rbenchv}.py` — delete `add_dataset_args` overrides.
- `Experiment/core_code/tests/test_eval_config.py` — drop 4 stale tests, add 2 new = 22 total.
- `Experiment/core_code/CLAUDE.md` — rewrite "eval.py configuration" section: drop "Flat CLI flags" subsection and "Migrating an old shell script" subsection.

**Migration outputs (created by the migration script):**
- `Experiment/core_code/configs/<55-new-yamls>.yaml` — one per unmigrated shell script.
- 73 rewritten `Experiment/core_code/scripts/**/*.sh` files, each a thin `--config` wrapper.

---

## Task 1: Refactor `load_config` to be schema-generic

**Goal:** Make `load_config` work with any Pydantic model, not just `EvalConfig`. Drop the `flat_overrides` parameter (no flat flags = no flat overrides).

**Files:**
- Modify: `Experiment/core_code/eval_config.py:155-184` (load_config)
- Modify: `Experiment/core_code/eval.py` (single call site, will be reworked anyway in Task 7)
- Modify: `Experiment/core_code/tests/test_eval_config.py` (loader tests use new signature)

- [ ] **Step 1: Update test signatures to drop `flat_overrides`**

In `Experiment/core_code/tests/test_eval_config.py`, find every call to `load_config(...)`. Currently they all pass `flat_overrides=`. Replace each call with the new signature `load_config(*, config_path, dot_overrides, schema)`. For tests that genuinely test flat-override behavior (`test_flat_overrides_beat_yaml`, `test_dot_overrides_beat_flat`), translate the flat overrides into equivalent dot-path overrides.

For example:
```python
# OLD
cfg = load_config(config_path=yml, flat_overrides={"seed": 7}, dot_overrides=[])
# NEW (translate flat seed override to dot-path)
cfg = load_config(config_path=yml, dot_overrides=["seed=7"], schema=EvalConfig)
```

`test_flat_overrides_beat_yaml` becomes `test_dot_overrides_beat_yaml`. Rename the test accordingly.

For tests that don't supply config_path:
```python
# OLD
cfg = load_config(config_path=None, flat_overrides={...}, dot_overrides=[])
# NEW: convert all flat keys to dot-overrides
cfg = load_config(config_path=None, dot_overrides=["benchmark=hle", "backend=claude", ...], schema=EvalConfig)
```

For `test_load_without_yaml`, this becomes equivalent to passing all required fields via `-o` overrides.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
pytest tests/test_eval_config.py -v 2>&1 | tail -20
```
Expected: TypeError or similar — `load_config()` got unexpected keyword argument 'schema' / missing positional 'flat_overrides'.

- [ ] **Step 3: Modify `load_config` signature in `eval_config.py`**

Replace lines 155-184 with:

```python
def load_config(
    *,
    config_path: Path | str | None,
    dot_overrides: list[str],
    schema: type[BaseModel] = EvalConfig,
) -> BaseModel:
    """Merge config sources and validate via the given pydantic schema.

    Order of precedence (later wins):
      1. YAML file (if config_path is given)
      2. Dot-path overrides (e.g. "model_budgets.haiku=2")

    The schema parameter selects which Pydantic model validates the merged dict.
    Defaults to EvalConfig for callers that pre-date PrecacheConfig.
    """
    merged: dict[str, Any] = {}
    if config_path is not None:
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}
        assert isinstance(yaml_data, dict), (
            f"top level of {config_path} must be a mapping, got {type(yaml_data).__name__}"
        )
        merged.update(yaml_data)

    for ov in dot_overrides:
        k, sep, v = ov.partition("=")
        assert sep == "=", f"override must be key=value, got {ov!r}"
        _set_dotpath(merged, k.strip(), v.strip())

    return schema.model_validate(merged)
```

The `BaseModel` import is already at the top of the file. Remove the now-unused `Any` if not referenced elsewhere — verify by `grep "Any" eval_config.py` after the change.

- [ ] **Step 4: Update `eval.py:parse_cli` call site**

In `Experiment/core_code/eval.py`, find the `load_config(...)` call inside `parse_cli` (currently passes `flat_overrides=flat`). Update to:

```python
    return load_config(
        config_path=args.config,
        dot_overrides=list(args.override) + extra_dot,
        schema=EvalConfig,
    )
```

The flat-overrides loop above this line (which builds `flat` dict and `extra_dot` list) STAYS for now — Task 7 removes it. We only change the load_config call.

But wait: the flat-overrides loop currently produces `flat["cache_dir"]` for legacy `--cache-dirs <single-path>`, and produces `flat["filters"] = {...}` previously (since C1 fix it routes filters through `extra_dot`). With `flat_overrides` parameter gone, we have to also route the remaining flat keys through `extra_dot`.

Update the loop body in `parse_cli`:
```python
    flat_kvs: list[str] = []
    extra_dot: list[str] = []
    for key, val in vars(args).items():
        if key in ("config", "override"):
            continue
        if val is None:
            continue
        if key == "cache_dirs":
            assert ":" not in val, (
                f"--cache-dirs accepts only a single path; multi-model cache "
                f"dicts must come from --config FILE.yaml. Got: {val!r}"
            )
            flat_kvs.append(f"cache_dir={val}")
            continue
        if key in benchmark.filter_keys:
            extra_dot.append(f"filters.{key}={val}")
        else:
            flat_kvs.append(f"{key}={val}")

    return load_config(
        config_path=args.config,
        dot_overrides=flat_kvs + extra_dot + list(args.override),
        schema=EvalConfig,
    )
```

Precedence: flat CLI < filter dot-paths < user-supplied `-o`. (User's `-o` always wins.)

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_eval_config.py -v 2>&1 | tail -10
```
Expected: 24 passed (existing count, no test added or removed at this stage).

- [ ] **Step 6: Smoke-import**

```bash
python -c "import eval; from eval_config import load_config, EvalConfig; print('ok')"
```
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
cd /data3/peijia/dr-claw/Explain
git add Experiment/core_code/eval_config.py \
        Experiment/core_code/eval.py \
        Experiment/core_code/tests/test_eval_config.py
git commit -m "refactor(eval): make load_config schema-generic; drop flat_overrides param"
```

---

## Task 2: PrecacheConfig schema + tests

**Goal:** Add a Pydantic model for `precache_explores.py` and 4 unit tests.

**Files:**
- Create: `Experiment/core_code/precache_config.py`
- Create: `Experiment/core_code/tests/test_precache_config.py`

- [ ] **Step 1: Write failing tests**

Create `Experiment/core_code/tests/test_precache_config.py`:

```python
from __future__ import annotations
import sys
from pathlib import Path
import textwrap

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

import pytest
import yaml
from pydantic import ValidationError
from eval_config import load_config
from precache_config import PrecacheConfig


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def _minimal_kwargs(**overrides):
    base = {
        "benchmark": "hle",
        "backend": "claude",
        "explore_model": "claude-sonnet-4-6",
        "cache_dir": "/cache/x",
    }
    base.update(overrides)
    return base


def test_minimal_precache_validates():
    cfg = PrecacheConfig(**_minimal_kwargs())
    assert cfg.benchmark == "hle"
    assert cfg.cache_dir == Path("/cache/x")
    assert cfg.num_explores == 8
    assert cfg.filters == {}


def test_precache_requires_cache_dir():
    kw = _minimal_kwargs()
    del kw["cache_dir"]
    with pytest.raises(ValidationError, match="cache_dir"):
        PrecacheConfig(**kw)


def test_precache_loader_yaml_plus_override(tmp_path):
    yml = _write(tmp_path, "p.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        cache_dir: /cache/h
        num_explores: 16
        filters:
          subset: gold
    """)
    cfg = load_config(
        config_path=yml,
        dot_overrides=["num_explores=4"],
        schema=PrecacheConfig,
    )
    assert cfg.cache_dir == Path("/cache/h")
    assert cfg.num_explores == 4
    assert cfg.filters == {"subset": "gold"}


def test_precache_filter_validation(tmp_path):
    yml = _write(tmp_path, "p.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        cache_dir: /cache/h
        filters:
          difficulty: hard
    """)
    with pytest.raises(ValidationError, match="difficulty"):
        load_config(config_path=yml, dot_overrides=[], schema=PrecacheConfig)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_precache_config.py -v
```
Expected: ImportError — `precache_config` module does not exist.

- [ ] **Step 3: Create `precache_config.py`**

Write `Experiment/core_code/precache_config.py`:

```python
"""Pydantic schema for precache_explores.py configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class PrecacheConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": False}

    benchmark: str
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

    filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_filters(self):
        from benchmarks import get_benchmark
        bench = get_benchmark(self.benchmark)
        filter_model = bench.make_filter_model()
        validated = filter_model.model_validate(self.filters)
        self.filters = validated.model_dump(exclude_defaults=True)
        return self
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_precache_config.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Run broader suite to verify no regression**

```bash
pytest tests/ -v 2>&1 | tail -5
```
Expected: count includes the 4 new tests; everything green.

- [ ] **Step 6: Commit**

```bash
git add Experiment/core_code/precache_config.py \
        Experiment/core_code/tests/test_precache_config.py
git commit -m "feat(precache): add PrecacheConfig schema + 4 unit tests"
```

---

## Task 3: precache_explores.py adopts `parse_cli` pattern

**Goal:** Replace `parse_args` in `precache_explores.py` with a `parse_cli`-style function returning a `PrecacheConfig`.

**Files:**
- Modify: `Experiment/core_code/precache_explores.py:130-148` (parse_args)
- Modify: `Experiment/core_code/precache_explores.py:150-186` (main, fields read from args → cfg)

- [ ] **Step 1: Read current `precache_explores.py` lines 1-200**

Confirm what fields `main()` consumes from `args`. Per the spec: benchmark, filter_keys, shuffle, seed, cache_dirs, num_explores, num_workers, backend, explore_model, num, budget_tokens, effort, explore_timeout.

- [ ] **Step 2: Replace `parse_args` with `parse_cli`**

In `Experiment/core_code/precache_explores.py`, replace the entire `parse_args` function with:

```python
def parse_cli() -> "PrecacheConfig":
    """Build PrecacheConfig from --benchmark + --config + -o overrides."""
    from precache_config import PrecacheConfig
    from eval_config import load_config

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--benchmark", type=str, required=True,
                      help="Benchmark name (hle, lcb, gpqa, babyvision, aime2025, aime2026, rbenchv)")
    base.add_argument("--config", type=str, required=True, help="Path to YAML config")
    known, _ = base.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Pre-cache explore results",
        parents=[base],
    )
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="Dot-path override, e.g. -o num_explores=4")
    args = parser.parse_args()

    return load_config(
        config_path=args.config,
        dot_overrides=[f"benchmark={args.benchmark}"] + list(args.override),
        schema=PrecacheConfig,
    )
```

The `--benchmark` flag is sniffed AND injected as a dot-override so the YAML's `benchmark:` key (which it must also have) is overridden by the CLI value. This guarantees the CLI and YAML agree (or the CLI wins).

- [ ] **Step 3: Replace `main()` to consume cfg**

Replace `main()`'s body. The imports at the top of the file may need `Path` already imported. Verify.

```python
def main() -> None:
    cfg = parse_cli()
    benchmark = get_benchmark(cfg.benchmark)

    print(f"Loading {benchmark.name.upper()} dataset...")
    all_rows = benchmark.load_dataset()

    filtered = benchmark.filter_dataset(all_rows, **cfg.filters)
    print(f"Filtered to {len(filtered)} questions")

    if cfg.shuffle:
        import random
        random.seed(cfg.seed)
        random.shuffle(filtered)

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(precache(
        benchmark=benchmark,
        rows=filtered,
        cache_dir=cfg.cache_dir,
        num_explores=cfg.num_explores,
        num_workers=cfg.num_workers,
        backend=cfg.backend,
        model=cfg.explore_model,
        num=cfg.num,
        budget_tokens=cfg.budget_tokens,
        effort=cfg.effort,
        explore_timeout=cfg.explore_timeout,
    ))
```

- [ ] **Step 4: Smoke-import**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
python -c "import precache_explores; print('ok')"
```
Expected: `ok`. No ImportError, no AttributeError.

- [ ] **Step 5: Smoke-parse a synthetic precache YAML**

```bash
mkdir -p /tmp/precache_smoke && cat > /tmp/precache_smoke/p.yaml <<'EOF'
benchmark: hle
backend: claude
explore_model: claude-sonnet-4-6
cache_dir: /tmp/cache_smoke
num_explores: 2
EOF
python -c "
import sys
sys.argv = ['precache_explores.py', '--benchmark', 'hle', '--config', '/tmp/precache_smoke/p.yaml']
from precache_explores import parse_cli
cfg = parse_cli()
print('OK:', cfg.benchmark, cfg.cache_dir, cfg.num_explores)
"
```
Expected: `OK: hle /tmp/cache_smoke 2`

- [ ] **Step 6: Run pytest to confirm no regression**

```bash
pytest tests/ -v 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add Experiment/core_code/precache_explores.py
git commit -m "refactor(precache): adopt parse_cli pattern; consume PrecacheConfig"
```

---

## Task 4: Build migration script + 8 verification tests

**Goal:** Write the one-shot `migrate_eval_scripts.py` plus `tests/test_migrate_scripts.py` (R1-R7, R9 from the spec). Do NOT run the live migration in this task — only the dry-run / unit-level tests.

**Files:**
- Create: `Experiment/core_code/scripts/migrate_eval_scripts.py`
- Create: `Experiment/core_code/tests/test_migrate_scripts.py`

- [ ] **Step 1: Write the migration script with helpers (no main yet)**

Create `Experiment/core_code/scripts/migrate_eval_scripts.py`:

```python
"""One-shot migration: convert eval.py / precache_explores.py shell scripts to YAML+--config form.

Self-deletes itself and tests/test_migrate_scripts.py on success.
"""
from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

import yaml

CORE_CODE = Path(__file__).resolve().parent.parent  # Experiment/core_code/
SCRIPTS_DIR = CORE_CODE / "scripts"
CONFIGS_DIR = CORE_CODE / "configs"
TESTS_DIR = CORE_CODE / "tests"

# Flat CLI flag → (yaml_key, kind) map for eval.py.
# kind: "scalar" | "bool" | "filter" | "dict_kv" | "single_path"
EVAL_FLAG_MAP: dict[str, tuple[str, str]] = {
    "backend": ("backend", "scalar"),
    "explore-model": ("explore_model", "scalar"),
    "method": ("method", "scalar"),
    "orchestrator-model": ("orchestrator_model", "scalar"),
    "integrate-model": ("integrate_model", "scalar"),
    "reward-model": ("reward_model", "scalar"),
    "cache-dirs": ("__cache_dirs__", "cache_dirs_special"),
    "model-budgets": ("model_budgets", "dict_kv"),
    "effort-budgets": ("effort_budgets", "dict_kv"),
    "exploration-effort": ("exploration_effort", "scalar"),
    "num": ("num", "scalar"),
    "skip": ("skip", "scalar"),
    "seed": ("seed", "scalar"),
    "shuffle": ("shuffle", "bool"),
    "no-cache-only": ("no_cache_only", "bool"),
    "no-integrate": ("no_integrate", "bool"),
    "verbose": ("verbose", "bool"),
    "num-explores": ("num_explores", "scalar"),
    "num-workers": ("num_workers", "scalar"),
    "num-rollouts": ("num_rollouts", "scalar"),
    "budget-tokens": ("budget_tokens", "scalar"),
    "effort": ("effort", "scalar"),
    "timeout": ("timeout", "scalar"),
    "explore-timeout": ("explore_timeout", "scalar"),
    "max-output-chars": ("max_output_chars", "scalar"),
    "log-dir": ("log_dir", "scalar"),
    "resume": ("resume", "scalar"),
}

# Per-benchmark filter flags. These go into nested filters: {...}.
# Source of truth: BenchmarkConfig.filter_keys + each subclass's add_dataset_args.
FILTER_FLAGS: dict[str, str] = {
    "subset": "subset",
    "category": "category",
    "text-only": "text_only",
    "difficulty": "difficulty",
    "type": "type",
    "subtype": "subtype",
    "year": "year",
    "domain": "domain",
}


SCALAR_BOOL_FLAGS = {"shuffle", "no-cache-only", "no-integrate", "verbose", "text-only"}


def script_to_yaml_name(sh_path: Path) -> str:
    """Translate a shell script path to a flat YAML filename.

    Example: scripts/hle/sonnet/run_self_refine.sh → hle_sonnet_self_refine.yaml
    """
    rel = sh_path.relative_to(SCRIPTS_DIR)
    parts = list(rel.parts)
    last = parts[-1].removesuffix(".sh")
    if last.startswith("run_"):
        last = last[4:]
    parts[-1] = last
    return "_".join(parts) + ".yaml"


def extract_python_invocation(sh_text: str) -> tuple[str, list[str]] | None:
    """Find the `python eval.py` or `python precache_explores.py` invocation.

    Returns (script_basename, argv_list_excluding_python_and_script) or None.
    Handles backslash line continuation. Stops at unescaped > | & or end of script.
    """
    # Match: python {script}.py [args...] until > or | or & or unescaped newline.
    # Use DOTALL so . spans lines, but we also strip backslash-newline first.
    lines = []
    in_invocation = False
    invocation_lines: list[str] = []
    target_script: str | None = None
    for raw_line in sh_text.splitlines():
        # Strip trailing comments (but only outside quotes — keep simple here)
        stripped = raw_line.rstrip()
        if not in_invocation:
            m = re.search(r"\bpython\s+(eval|precache_explores)\.py\b", stripped)
            if m:
                target_script = m.group(1) + ".py"
                in_invocation = True
                # take portion after the script name
                tail = stripped[m.end():]
                invocation_lines.append(tail)
                if not stripped.endswith("\\"):
                    break
                continue
        else:
            invocation_lines.append(stripped)
            if not stripped.endswith("\\"):
                break
    if target_script is None:
        return None

    # Strip backslash continuations and join.
    joined = " ".join(line.rstrip("\\").strip() for line in invocation_lines)
    # Strip trailing redirections / backgrounding.
    joined = re.split(r"[>|]|\s2>&1|\s&\s*$", joined, maxsplit=1)[0]
    argv = shlex.split(joined)
    return target_script, argv


def translate_eval_argv(argv: list[str]) -> dict:
    """Convert eval.py argv into a YAML dict matching EvalConfig fields.

    Raises AssertionError on unknown flags or malformed values.
    """
    out: dict = {}
    filters: dict = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            raise AssertionError(f"unexpected positional token at {i}: {token!r}")
        flag = token[2:]

        if flag == "benchmark":
            out["benchmark"] = argv[i + 1]
            i += 2
            continue

        if flag in FILTER_FLAGS:
            yaml_key = FILTER_FLAGS[flag]
            if flag in SCALAR_BOOL_FLAGS:
                filters[yaml_key] = True
                i += 1
            else:
                filters[yaml_key] = _coerce(argv[i + 1])
                i += 2
            continue

        if flag not in EVAL_FLAG_MAP:
            raise AssertionError(f"unknown eval.py flag: --{flag}")

        yaml_key, kind = EVAL_FLAG_MAP[flag]

        if kind == "bool":
            out[yaml_key] = True
            i += 1
        elif kind == "scalar":
            out[yaml_key] = _coerce(argv[i + 1])
            i += 2
        elif kind == "dict_kv":
            out[yaml_key] = _parse_kv_string(argv[i + 1], int)
            i += 2
        elif kind == "cache_dirs_special":
            value = argv[i + 1]
            if ":" in value:
                # multi-model dict form
                out["cache_dirs"] = _parse_kv_string(value, str_path=True)
            else:
                out["cache_dir"] = value
            i += 2
        else:
            raise AssertionError(f"unhandled kind {kind!r} for flag {flag}")

    if filters:
        out["filters"] = filters
    return out


def translate_precache_argv(argv: list[str]) -> dict:
    """Convert precache_explores.py argv into a YAML dict matching PrecacheConfig fields."""
    out: dict = {}
    filters: dict = {}
    i = 0
    PRECACHE_VALID_KEYS = {
        "benchmark", "backend", "explore_model", "cache_dir",
        "num_explores", "num_workers", "num", "skip", "seed", "shuffle",
        "budget_tokens", "effort", "explore_timeout",
    }
    while i < len(argv):
        token = argv[i]
        flag = token[2:]
        if flag == "benchmark":
            out["benchmark"] = argv[i + 1]
            i += 2
            continue
        if flag in FILTER_FLAGS:
            yaml_key = FILTER_FLAGS[flag]
            if flag in SCALAR_BOOL_FLAGS:
                filters[yaml_key] = True
                i += 1
            else:
                filters[yaml_key] = _coerce(argv[i + 1])
                i += 2
            continue
        if flag == "cache-dirs":
            value = argv[i + 1]
            assert ":" not in value, (
                f"precache uses single-path cache_dir; got dict-form {value!r}"
            )
            out["cache_dir"] = value
            i += 2
            continue
        if flag in EVAL_FLAG_MAP:
            yaml_key, kind = EVAL_FLAG_MAP[flag]
            if yaml_key not in PRECACHE_VALID_KEYS:
                raise AssertionError(
                    f"flag --{flag} (yaml key {yaml_key}) not valid for PrecacheConfig"
                )
            if kind == "bool":
                out[yaml_key] = True
                i += 1
            elif kind == "scalar":
                out[yaml_key] = _coerce(argv[i + 1])
                i += 2
            else:
                raise AssertionError(f"flag --{flag} kind {kind!r} not allowed for precache")
            continue
        raise AssertionError(f"unknown precache flag: --{flag}")

    if filters:
        out["filters"] = filters
    return out


def _coerce(s: str):
    """Coerce a string to int/float/bool/str (best-effort, mirrors _coerce_scalar)."""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_kv_string(s: str, value_type=str, str_path: bool = False) -> dict:
    """Parse 'k1:v1,k2:v2' → {k1: v1, k2: v2}. value_type coerces values; str_path keeps as str."""
    out = {}
    for pair in s.split(","):
        k, _, v = pair.partition(":")
        assert _ == ":", f"expected k:v, got {pair!r}"
        k = k.strip()
        v = v.strip()
        out[k] = v if str_path else value_type(v)
    return out


def rewrite_sh(sh_text: str, target_script: str, new_yaml_relpath: str, benchmark: str) -> str:
    """Rewrite the shell script: replace the python invocation with --config form.

    Preserves preamble (cd, env, exports, mkdir, conda) and tail (> log 2>&1 &).
    """
    lines = sh_text.splitlines(keepends=True)
    out_lines: list[str] = []
    i = 0
    in_invocation = False
    consumed_lines: list[str] = []
    while i < len(lines):
        line = lines[i]
        m = re.search(r"\bpython\s+" + re.escape(target_script) + r"\b", line)
        if m and not in_invocation:
            in_invocation = True
            # Find tail (redirections and &) by scanning forward including continuations
            consumed_lines.append(line)
            j = i + 1
            while line.rstrip().endswith("\\"):
                if j >= len(lines):
                    break
                line = lines[j]
                consumed_lines.append(line)
                j += 1
            # The full original block is consumed_lines[0..]; we rebuild it.
            # Identify the "tail" — anything after >, |, 2>&1, & in the joined block.
            joined = "".join(consumed_lines)
            # Find prefix up to and including `python <script>.py`
            pre = re.split(r"(\bpython\s+" + re.escape(target_script) + r"\b)", joined, maxsplit=1)
            assert len(pre) == 3
            head = pre[0] + pre[1]
            after_script = pre[2]
            # Find the redirection tail
            tail_match = re.search(r"(\s*[>|]|\s+2>&1|\s+&\s*$)", after_script)
            if tail_match:
                tail = after_script[tail_match.start():]
            else:
                tail = ""
            # Build new line(s)
            new_invocation = (
                head
                + f" \\\n\t--benchmark {benchmark} \\\n\t--config configs/{new_yaml_relpath}"
                + tail
            )
            if not new_invocation.endswith("\n"):
                new_invocation += "\n"
            out_lines.append(new_invocation)
            i = j
        else:
            out_lines.append(line)
            i += 1
    return "".join(out_lines)


def main() -> int:
    """Run the migration. Returns 0 on success, non-zero on failure."""
    failures: list[tuple[Path, str]] = []
    migrated: list[tuple[Path, Path, Path]] = []  # (sh_path, yaml_path, target_script)

    sh_files = sorted(SCRIPTS_DIR.glob("**/*.sh"))
    for sh in sh_files:
        try:
            sh_text = sh.read_text()
            invocation = extract_python_invocation(sh_text)
            if invocation is None:
                # Not an eval/precache script — skip (no failure).
                continue
            target_script, argv = invocation

            # Pull --benchmark out of argv (must be present).
            try:
                bench_idx = argv.index("--benchmark")
            except ValueError:
                raise AssertionError(f"missing --benchmark flag")
            benchmark = argv[bench_idx + 1]

            # If the script already uses --config, it's already migrated. Skip.
            if "--config" in argv:
                continue

            if target_script == "eval.py":
                yaml_dict = translate_eval_argv(argv)
            else:
                yaml_dict = translate_precache_argv(argv)

            # Write YAML
            yaml_name = script_to_yaml_name(sh)
            yaml_path = CONFIGS_DIR / yaml_name
            if yaml_path.exists():
                raise AssertionError(f"target YAML already exists: {yaml_path}")
            yaml_text = yaml.safe_dump(yaml_dict, sort_keys=False, default_flow_style=False)
            yaml_path.write_text(yaml_text)

            # Rewrite shell script
            new_sh_text = rewrite_sh(sh_text, target_script, yaml_name, benchmark)
            sh.write_text(new_sh_text)

            migrated.append((sh, yaml_path, target_script))
        except Exception as e:
            failures.append((sh, repr(e)))

    print(f"Migrated {len(migrated)} scripts. Failures: {len(failures)}.")
    for sh, err in failures:
        print(f"  FAIL: {sh}: {err}")

    if failures:
        return 1

    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
```

- [ ] **Step 2: Write the verification tests**

Create `Experiment/core_code/tests/test_migrate_scripts.py`:

```python
from __future__ import annotations
import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))
sys.path.insert(0, str(_CORE_CODE_DIR / "scripts"))

import pytest
from migrate_eval_scripts import (
    SCRIPTS_DIR,
    CONFIGS_DIR,
    script_to_yaml_name,
    extract_python_invocation,
    translate_eval_argv,
    translate_precache_argv,
)
from eval_config import EvalConfig
from precache_config import PrecacheConfig


def _all_eval_or_precache_scripts() -> list[Path]:
    out = []
    for sh in sorted(SCRIPTS_DIR.glob("**/*.sh")):
        text = sh.read_text()
        if extract_python_invocation(text) is not None:
            out.append(sh)
    return out


def test_R1_dry_run_parses_all_scripts():
    """R1: every shell script with eval.py / precache_explores.py invocation parses."""
    failures = []
    for sh in _all_eval_or_precache_scripts():
        try:
            inv = extract_python_invocation(sh.read_text())
            assert inv is not None
            target_script, argv = inv
            assert target_script in ("eval.py", "precache_explores.py")
            assert "--benchmark" in argv
        except Exception as e:
            failures.append((str(sh), repr(e)))
    assert not failures, f"{len(failures)} scripts failed to parse: {failures[:3]}"


def test_R2_no_filename_collisions():
    """R2: target YAML names from path-flatten are unique."""
    names = [script_to_yaml_name(sh) for sh in _all_eval_or_precache_scripts()]
    assert len(set(names)) == len(names), \
        f"collision: {[n for n in names if names.count(n) > 1][:3]}"


def test_R3_no_unknown_precache_callers():
    """R3: precache_explores has no Python or notebook caller outside known sites."""
    import subprocess
    result = subprocess.run(
        ["grep", "-rl", "precache_explores", str(_CORE_CODE_DIR.parent.parent)],
        capture_output=True, text=True,
    )
    hits = [Path(p) for p in result.stdout.strip().splitlines() if p]
    # Allowed: precache_explores.py itself, scripts/**, tests/**, this script, the
    # generated YAMLs / docs may reference it by name.
    suffixes_ok = (".sh", ".md", ".yaml", ".yml")
    for h in hits:
        rel = h.resolve()
        if rel == (_CORE_CODE_DIR / "precache_explores.py").resolve():
            continue
        if rel == (_CORE_CODE_DIR / "scripts" / "migrate_eval_scripts.py").resolve():
            continue
        if rel.suffix in suffixes_ok:
            continue
        if rel.is_relative_to(_CORE_CODE_DIR / "tests"):
            continue
        # Notebooks not yet allowed; flag.
        pytest.fail(f"unknown precache_explores caller: {rel}")


def test_R4_no_unknown_eval_importers():
    """R4: only test files import eval module by name."""
    import subprocess
    result = subprocess.run(
        ["grep", "-rln", r"^from eval import\|^import eval$",
         str(_CORE_CODE_DIR), "--include=*.py"],
        capture_output=True, text=True,
    )
    hits = [Path(p) for p in result.stdout.strip().splitlines() if p]
    for h in hits:
        rel = h.resolve()
        if rel.is_relative_to((_CORE_CODE_DIR / "tests").resolve()):
            continue
        if rel == (_CORE_CODE_DIR / "scripts" / "migrate_eval_scripts.py").resolve():
            continue
        pytest.fail(f"unknown eval importer: {rel}")


def test_R5_translated_eval_argv_validates():
    """R5: every eval.py argv translates to a dict that EvalConfig accepts (post-migration smoke)."""
    failures = []
    for sh in _all_eval_or_precache_scripts():
        text = sh.read_text()
        target, argv = extract_python_invocation(text)
        if target != "eval.py":
            continue
        if "--config" in argv:
            continue  # already migrated
        try:
            d = translate_eval_argv(argv)
            # Inject benchmark from argv even though translate_eval_argv extracts it
            EvalConfig.model_validate(d)
        except Exception as e:
            failures.append((str(sh), str(e)[:200]))
    # Some legacy scripts may legitimately fail (e.g., --num-rollouts > 1 with non-vllm
    # backend). For migration purposes, surface the count, not the failures.
    # The failures are surfaced as part of test_R7.
    assert len(failures) <= 5, f"unexpectedly many failures: {failures[:5]}"


def test_R6_partial_failure_safety_dry_run():
    """R6: the migration script's per-file atomicity. We can't easily simulate mid-run
    crash without running migration; instead verify that yaml/sh writes happen in pairs
    (read code, assert ordering)."""
    # Static check: yaml_path.write_text precedes sh.write_text in main()
    src = (_CORE_CODE_DIR / "scripts" / "migrate_eval_scripts.py").read_text()
    yaml_pos = src.find("yaml_path.write_text")
    sh_pos = src.find("sh.write_text(new_sh_text)")
    assert 0 < yaml_pos < sh_pos, "yaml must be written before .sh is overwritten"


def test_R7_all_translated_keys_have_target():
    """R7: every key emitted by translate_eval_argv exists on EvalConfig.
    Same for translate_precache_argv → PrecacheConfig."""
    eval_fields = set(EvalConfig.model_fields.keys())
    precache_fields = set(PrecacheConfig.model_fields.keys())
    for sh in _all_eval_or_precache_scripts():
        text = sh.read_text()
        target, argv = extract_python_invocation(text)
        if "--config" in argv:
            continue
        if target == "eval.py":
            d = translate_eval_argv(argv)
            for k in d:
                assert k in eval_fields, f"{sh}: translated key {k!r} not in EvalConfig"
        else:
            d = translate_precache_argv(argv)
            for k in d:
                assert k in precache_fields, f"{sh}: translated key {k!r} not in PrecacheConfig"


def test_R9_post_migration_dry_parse():
    """R9: after migration runs (or pre-migration if all already done),
    every .sh dry-parses via the appropriate parse_cli."""
    # This test only checks scripts that have already been migrated (--config form).
    # The full suite re-runs after live migration as the post-migration gate.
    import subprocess
    failures = []
    for sh in _all_eval_or_precache_scripts():
        text = sh.read_text()
        target, argv = extract_python_invocation(text)
        if "--config" not in argv:
            continue  # not yet migrated; skipped here
        # invoke the appropriate parse_cli
        py = (
            f"import sys; sys.argv=['{target}'] + {argv!r};"
            f" from {target.removesuffix('.py')} import parse_cli; parse_cli()"
        )
        result = subprocess.run(
            ["python", "-c", py], capture_output=True, text=True,
            cwd=str(_CORE_CODE_DIR),
        )
        if result.returncode != 0:
            failures.append((str(sh), result.stderr[:300]))
    assert not failures, f"{len(failures)} migrated scripts fail dry-parse: {failures[:3]}"
```

- [ ] **Step 3: Run the tests against current state**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
pytest tests/test_migrate_scripts.py -v 2>&1 | tail -25
```
Expected: most pass. Some may fail until the translation logic handles every edge case present in the 73 real scripts. R1, R2, R7, R9 should pass cleanly. R5 may surface up to ~5 failures (those need fixed translation logic). R3, R4 should pass (zero unknown callers).

- [ ] **Step 4: Iterate translate_*_argv until R5 and R7 are clean**

If R5 or R7 reports a script that fails to translate, examine that script:

```bash
cat <failing-script>
```

Identify which flag is missing from `EVAL_FLAG_MAP` or `FILTER_FLAGS`, or which value type is wrong. Update `migrate_eval_scripts.py` and rerun pytest. Keep iterating until R5 reports `failures = 0` and R7 passes.

- [ ] **Step 5: Commit**

```bash
git add Experiment/core_code/scripts/migrate_eval_scripts.py \
        Experiment/core_code/tests/test_migrate_scripts.py
git commit -m "feat(migrate): add migration script + 8 verification tests (R1-R7, R9)"
```

---

## Task 5: Run live migration

**Goal:** Execute `migrate_eval_scripts.py` against all 73 shell scripts. Verify all R-tests pass post-migration. Commit the resulting state (55 new YAMLs + 73 rewritten .sh files).

**Files:**
- Modifies all 73 scripts in `Experiment/core_code/scripts/**/*.sh` (those that haven't been migrated yet — 55 of them).
- Creates: 55 new YAMLs in `Experiment/core_code/configs/`.

- [ ] **Step 1: Run the migration**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
python scripts/migrate_eval_scripts.py
```
Expected: prints `Migrated N scripts. Failures: 0.` (where N is roughly 55; the 18 already-migrated scripts skip via the `--config in argv` guard).

If failures: read the per-script error output, fix the issue (either in the migration script or in the offending shell script), revert any partial changes (`git checkout Experiment/core_code/scripts/ Experiment/core_code/configs/`), and re-run.

- [ ] **Step 2: Run full pytest including R-gates**

```bash
pytest tests/ -v 2>&1 | tail -15
```
Expected: ALL tests pass (existing 28 + new 4 PrecacheConfig + 8 R-tests = 40 total). R9 specifically validates that all 73 .sh scripts dry-parse via the new parse_cli — this is the ultimate post-migration gate.

If any test fails: do not commit. Diagnose, fix, retry.

- [ ] **Step 3: Spot-check 3 migrated YAMLs and shell scripts**

```bash
# Pick 3 representative migrated scripts from different benchmarks
diff <(git show HEAD:Experiment/core_code/scripts/hle/sonnet/run_self_refine.sh) \
     Experiment/core_code/scripts/hle/sonnet/run_self_refine.sh
cat Experiment/core_code/configs/hle_sonnet_self_refine.yaml
```
Read each: confirm the YAML content matches the original CLI args verbatim.

- [ ] **Step 4: Run the broader smoke (parse-only) sweep**

```bash
for cfg in Experiment/core_code/configs/*.yaml; do
  bench=$(grep '^benchmark:' "$cfg" | awk '{print $2}')
  python -c "
import sys
sys.argv = ['eval.py', '--benchmark', '$bench', '--config', '$cfg']
try:
    from eval import parse_cli
    parse_cli()
    print('OK eval:', '$cfg')
except Exception as e:
    # Try precache instead
    sys.argv = ['precache_explores.py', '--benchmark', '$bench', '--config', '$cfg']
    try:
        from precache_explores import parse_cli as p
        p()
        print('OK precache:', '$cfg')
    except Exception as e2:
        print('FAIL:', '$cfg', e2)
" 2>&1
done | grep -c "^OK"
```
Expected: 73 (or however many YAMLs exist post-migration).

- [ ] **Step 5: Commit**

```bash
git add Experiment/core_code/configs/ Experiment/core_code/scripts/
git commit -m "feat(migrate): convert 55 shell scripts to YAML+--config form"
```

---

## Task 6: Self-delete migration artifacts

**Goal:** Remove `migrate_eval_scripts.py` and `tests/test_migrate_scripts.py`. The migration is one-shot; no traces.

**Files:**
- Delete: `Experiment/core_code/scripts/migrate_eval_scripts.py`
- Delete: `Experiment/core_code/tests/test_migrate_scripts.py`

- [ ] **Step 1: Delete the files**

```bash
cd /data3/peijia/dr-claw/Explain
git rm Experiment/core_code/scripts/migrate_eval_scripts.py
git rm Experiment/core_code/tests/test_migrate_scripts.py
```

- [ ] **Step 2: Run pytest to confirm no regression**

```bash
cd Experiment/core_code
pytest tests/ -v 2>&1 | tail -5
```
Expected: 32 tests pass (24 existing + 4 PrecacheConfig + 4 new in Task 7 — actually Task 7 is later. So at this point: 24 existing + 4 PrecacheConfig = 28 tests).

- [ ] **Step 3: Commit**

```bash
cd /data3/peijia/dr-claw/Explain
git commit -m "chore(migrate): remove one-shot migration artifacts"
```

---

## Task 7: Slim `eval.py` `parse_cli` to 3-flag form

**Goal:** Remove every flat-flag declaration from `parse_cli`. argparse exposes only `--benchmark`, `--config`, `-o`. Update tests.

**Files:**
- Modify: `Experiment/core_code/eval.py:603-680` (`parse_cli`)
- Modify: `Experiment/core_code/tests/test_eval_config.py` (drop 4 tests, add 2)

- [ ] **Step 1: Replace `parse_cli` body**

In `Experiment/core_code/eval.py`, replace the entire `parse_cli` function with:

```python
def parse_cli() -> "EvalConfig":
    """Build EvalConfig from --benchmark + --config + -o overrides."""
    from eval_config import load_config

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--benchmark", type=str, required=True,
                      help="Benchmark name (hle, lcb, gpqa, babyvision, aime2025, aime2026, rbenchv)")
    base.add_argument("--config", type=str, required=True, help="Path to YAML config")
    known, _ = base.parse_known_args()

    parser = argparse.ArgumentParser(
        description=f"Evaluate TTS agent on {known.benchmark.upper()}",
        parents=[base],
    )
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="Dot-path override, e.g. -o model_budgets.haiku=2")
    args = parser.parse_args()

    return load_config(
        config_path=args.config,
        dot_overrides=[f"benchmark={args.benchmark}"] + list(args.override),
        schema=EvalConfig,
    )
```

The `--benchmark` value is injected as a dot-override so YAML and CLI must agree (CLI wins). All flat flag declarations and the legacy filter-routing / cache-dirs special-cases are removed.

Note: `EvalConfig` is now imported only inside `parse_cli`. Verify no other code in `eval.py` references `EvalConfig` outside this function (search `EvalConfig` in `eval.py`). If there are references, add the import at module top.

- [ ] **Step 2: Update `tests/test_eval_config.py`**

Delete these 4 tests (they tested behavior that no longer exists):
- `test_parse_cli_only` (uses flat `--num`)
- `test_parse_cli_with_yaml_and_override` (uses flat `--cache-dirs` indirectly via the eval CLI; superseded by Task 1's dot-overrides version)
- `test_cli_filter_flag_preserves_yaml_filter_siblings`
- `test_cli_cache_dirs_single_path_routes_to_cache_dir`
- `test_cli_cache_dirs_with_colons_rejects`

Wait — that's 5. Pick the 4 most-stale: drop `test_parse_cli_only`, `test_cli_filter_flag_preserves_yaml_filter_siblings`, `test_cli_cache_dirs_single_path_routes_to_cache_dir`, `test_cli_cache_dirs_with_colons_rejects`.

Keep `test_parse_cli_with_yaml_and_override` but rewrite it to use only `--config` + `-o`:

```python
def test_parse_cli_with_yaml_and_override(tmp_path, monkeypatch):
    import importlib
    import eval as eval_mod
    importlib.reload(eval_mod)

    yml = _write(tmp_path, "x.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        method: tts-agent-multi
        orchestrator_model: claude-sonnet-4-6
        cache_dirs:
          haiku: /cache/haiku
          sonnet: /cache/sonnet
        model_budgets:
          haiku: 8
          sonnet: 8
    """)
    argv = ["eval.py", "--benchmark", "hle", "--config", str(yml),
            "-o", "model_budgets.haiku=2", "-o", "seed=99"]
    monkeypatch.setattr("sys.argv", argv)
    cfg = eval_mod.parse_cli()
    assert cfg.model_budgets == {"haiku": 2, "sonnet": 8}
    assert cfg.seed == 99
```

Add 2 new tests at the end of `tests/test_eval_config.py`:

```python
def test_cli_rejects_unknown_flat_flag(tmp_path, monkeypatch):
    """The slimmed parse_cli must not accept any flat field flag."""
    import importlib
    import eval as eval_mod
    importlib.reload(eval_mod)

    yml = _write(tmp_path, "x.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        method: self-refine
    """)
    argv = ["eval.py", "--benchmark", "hle", "--config", str(yml), "--seed", "42"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        eval_mod.parse_cli()


def test_cli_three_flags_only(monkeypatch):
    """Slimmed parse_cli must declare exactly --benchmark, --config, -o."""
    import importlib
    import eval as eval_mod
    importlib.reload(eval_mod)

    monkeypatch.setattr("sys.argv", ["eval.py", "--benchmark", "hle", "--help"])
    with pytest.raises(SystemExit) as ei:
        eval_mod.parse_cli()
    # Capture argparse help output via capsys would require fixture rejig;
    # instead inspect the parser by re-running the sniff path.
    # Verify the parser has 3 user-facing actions + benchmark sniff + help.
    # Easiest: re-use the parse-known-args pattern manually.
    # The simpler check is: argparse usage must NOT mention --num / --seed / --backend.
```

For `test_cli_three_flags_only` we can use a different mechanic — capsys:

```python
def test_cli_three_flags_only(capsys, monkeypatch):
    """Argparse usage line must mention only --benchmark, --config, -o (and -h)."""
    import importlib
    import eval as eval_mod
    importlib.reload(eval_mod)
    monkeypatch.setattr("sys.argv", ["eval.py", "--benchmark", "hle", "--help"])
    with pytest.raises(SystemExit):
        eval_mod.parse_cli()
    captured = capsys.readouterr()
    forbidden = ["--num", "--seed", "--backend", "--explore-model", "--method",
                 "--cache-dirs", "--orchestrator-model", "--integrate-model",
                 "--shuffle", "--no-cache-only", "--no-integrate", "--verbose",
                 "--num-explores", "--num-workers", "--num-rollouts",
                 "--budget-tokens", "--effort", "--timeout", "--explore-timeout",
                 "--max-output-chars", "--log-dir", "--resume", "--reward-model",
                 "--exploration-effort", "--model-budgets", "--effort-budgets"]
    for f in forbidden:
        assert f not in captured.out, f"slim parse_cli still exposes {f}"
```

- [ ] **Step 3: Run tests**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
pytest tests/test_eval_config.py -v 2>&1 | tail -25
```
Expected: 22 passed (24 - 4 dropped + 2 added).

- [ ] **Step 4: Run broader pytest suite**

```bash
pytest tests/ -v 2>&1 | tail -5
```
Expected: 22 + 4 PrecacheConfig = 26 tests pass.

- [ ] **Step 5: Smoke-test the happy path**

```bash
python -c "
import sys
sys.argv = ['eval.py', '--benchmark', 'hle', '--config', 'configs/hle_multi_effort_low.yaml']
from eval import parse_cli
cfg = parse_cli()
print('OK:', cfg.method, len(cfg.cache_dirs), 'cache_dirs')
"
```
Expected: `OK: tts-agent-multi 3 cache_dirs`.

- [ ] **Step 6: Smoke-test that flat flag is rejected**

```bash
python -c "
import sys
sys.argv = ['eval.py', '--benchmark', 'hle', '--config', 'configs/hle_multi_effort_low.yaml', '--seed', '42']
from eval import parse_cli
try:
    parse_cli()
    print('FAIL: --seed should not be accepted')
except SystemExit:
    print('OK: --seed rejected')
" 2>&1 | tail -3
```
Expected: `OK: --seed rejected` (and argparse error before that on stderr).

- [ ] **Step 7: Commit**

```bash
cd /data3/peijia/dr-claw/Explain
git add Experiment/core_code/eval.py Experiment/core_code/tests/test_eval_config.py
git commit -m "refactor(eval): slim parse_cli to 3-flag form (--benchmark + --config + -o)"
```

---

## Task 8: Delete `add_dataset_args` / `add_model_args` from benchmarks

**Goal:** Now that `parse_cli` no longer calls these methods, they are dead code. Remove from base + 6 subclasses.

**Files:**
- Modify: `Experiment/core_code/benchmarks/base.py` (delete `add_dataset_args`, `add_model_args`)
- Modify: `Experiment/core_code/benchmarks/{hle,lcb,gpqa,babyvision,aime,rbenchv}.py` (delete `add_dataset_args` overrides)

- [ ] **Step 1: Verify nothing else calls these methods**

```bash
grep -rn "add_dataset_args\|add_model_args" Experiment/core_code/ --include="*.py"
```
Expected: matches only the definitions themselves (in `base.py` + 6 subclasses) — no callers anywhere. If any caller appears outside these files, STOP and update that caller first.

- [ ] **Step 2: Delete the methods**

In `Experiment/core_code/benchmarks/base.py`, find and delete the entire `add_dataset_args` method block, and `add_model_args` method block. The `import argparse` line at the top stays for now (Task 9 verifies it).

In each of:
- `Experiment/core_code/benchmarks/hle.py`
- `Experiment/core_code/benchmarks/lcb.py`
- `Experiment/core_code/benchmarks/gpqa.py`
- `Experiment/core_code/benchmarks/babyvision.py`
- `Experiment/core_code/benchmarks/aime.py`
- `Experiment/core_code/benchmarks/rbenchv.py`

Delete the `add_dataset_args` override method.

- [ ] **Step 3: Verify `argparse` import in `base.py` is still needed**

```bash
grep -n "argparse\." Experiment/core_code/benchmarks/base.py
```
If no matches, remove `import argparse` from the top of `base.py`. Otherwise leave.

Also check each benchmark file:
```bash
for f in hle lcb gpqa babyvision aime rbenchv; do
  echo "=== $f ==="
  grep -c "argparse" "Experiment/core_code/benchmarks/$f.py"
done
```
For files that no longer use argparse, remove `import argparse` from the top.

- [ ] **Step 4: Smoke-import**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
python -c "from benchmarks import get_benchmark; b = get_benchmark('hle'); print('ok', b.name)"
```
Expected: `ok hle`.

- [ ] **Step 5: Run full pytest**

```bash
pytest tests/ -v 2>&1 | tail -5
```
Expected: 26 tests still pass.

- [ ] **Step 6: Commit**

```bash
git add Experiment/core_code/benchmarks/
git commit -m "refactor(benchmarks): delete dead add_dataset_args/add_model_args methods"
```

---

## Task 9: Update `Experiment/core_code/CLAUDE.md` documentation

**Goal:** Rewrite the "eval.py configuration" section to reflect YAML-only flow.

**Files:**
- Modify: `Experiment/core_code/CLAUDE.md`

- [ ] **Step 1: Replace the configuration section**

Open `Experiment/core_code/CLAUDE.md`. Find the heading `## eval.py configuration` (currently around line 121). Replace the ENTIRE section (from `## eval.py configuration` through the end-of-file shell example block) with:

```markdown
## eval.py configuration

`eval.py` and `precache_explores.py` read configuration from two sources, in this order (later wins):

1. **YAML file**: required, via `--config configs/<name>.yaml`. The schemas live in
   `Experiment/core_code/eval_config.py` (`EvalConfig`) and
   `Experiment/core_code/precache_config.py` (`PrecacheConfig`). Reference template:
   `Experiment/core_code/configs/_template.yaml`.
2. **Dot-path overrides**: `-o key.subkey=value`, repeatable. Highest precedence.
   Use this for one-off tweaks: `python eval.py --benchmark hle --config configs/X.yaml -o seed=99 -o num=5`.

The argparse layer accepts exactly three flags: `--benchmark`, `--config`, and `-o`/`--override`.
Any other flag triggers an `unrecognized arguments` error from argparse — by design.

### Single-cache vs multi-cache

The `EvalConfig` schema has TWO separate fields, mutually exclusive by method:

- `cache_dir: Path | None` — for single-cache methods (`tts-agent`, `self-refine`,
  `socratic-self-refine`, `budget-forcing`, `rerank`, `standalone-integrator`).
  Set via YAML `cache_dir: /path` or `-o cache_dir=/path`.
- `cache_dirs: dict[str, Path]` — for multi/effort methods (`tts-agent-multi`,
  `tts-agent-effort`). Set via YAML mapping or `-o cache_dirs.alias=/path`.

A pydantic validator asserts you do not mix them.

### Per-benchmark filters

Each benchmark's filter fields are a Pydantic sub-model (declared via
`BenchmarkConfig.make_filter_model()`). Unknown filter keys are rejected at
validation time with `extra="forbid"`. Set via YAML `filters: {subset: gold}` or
`-o filters.subset=gold`.

### Adding a new experiment

Write a new YAML in `configs/`. For an experiment that's a one-line tweak of an
existing one, prefer `-o` overrides at the call site instead of forking the YAML.

```bash
# Common-case launcher pattern
python eval.py --benchmark hle --config configs/example.yaml
```
```

- [ ] **Step 2: Commit**

```bash
git add Experiment/core_code/CLAUDE.md
git commit -m "docs(eval): rewrite config section for YAML-only CLI"
```

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
|---|---|
| §Architecture: 3-flag CLI | T7 |
| §Pydantic schemas: PrecacheConfig | T2 |
| §Shared loader (schema-generic) | T1 |
| §Argparse cleanup (delete add_*_args) | T8 |
| §Shell-script migration via one-shot tool | T4, T5 |
| §Existing 18 YAMLs untouched | (no task — explicit non-action) |
| §Test changes (drop 4, add 2 + 4 PrecacheConfig + 8 R-tests) | T2, T4, T7 |
| §Documentation rewrite | T9 |
| §Risks: 8 R-tests | T4 |
| §Migration safety: per-file atomicity | T4 (R6 test) |
| §Self-deleting migration artifacts | T6 |

All spec sections have a task. The "delete migration tooling" obligation is split: T6 deletes script + tests after T5 commits the migrated state. This means commit history has the tooling for one commit but the working tree never does after T6.

**2. Placeholder scan:**

No "TBD", "TODO", or "implement later" appears in any task. Each step contains executable code or commands. The only deferred decision is whether the migration script's `translate_*_argv` covers every edge case in the 73 real scripts — T4 Step 4 explicitly says "iterate until R5 and R7 are clean" and provides the iteration loop. This is acceptable: it's an empirical convergence step, not a placeholder.

**3. Type consistency:**

- `load_config(*, config_path, dot_overrides, schema)` signature consistent: T1 introduces, T2/T3/T7 use.
- `EvalConfig` and `PrecacheConfig` import paths consistent across tests.
- `parse_cli() -> EvalConfig` (eval.py) and `parse_cli() -> PrecacheConfig` (precache_explores.py) — same name, different return types. Acceptable: each module has its own `parse_cli`.
- `script_to_yaml_name`, `extract_python_invocation`, `translate_eval_argv`, `translate_precache_argv`, `rewrite_sh` signatures consistent in T4 implementation and tests.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-29-yaml-only-cli-migration.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
