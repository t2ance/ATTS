# Method Block Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor method-related YAML config from flat top-level fields into a discriminated-union `method:` block (mirroring `BenchmarkSpec`), and collapse all `cfg.method == "..."` if/elif dispatches in eval.py into a single polymorphic `MethodConfig` lookup.

**Architecture:** Two cooperating layers — `MethodSpec` (Pydantic discriminated union for YAML schema, in `methods/specs.py`) and `MethodConfig` (runtime behavior class hierarchy, in `methods/registry.py`). Each method name maps to one Spec class + one Config class. EvalConfig holds `method: MethodSpec`; eval.py's main() does `method = get_method(cfg.method.name)` and dispatches everything through it.

**Tech Stack:** Python 3.11, Pydantic v2, PyYAML.

**Scope:** In: `eval.py`, `methods/specs.py` (new), `methods/registry.py` (new), `methods/__init__.py`, `methods/base.py`, 83 YAMLs under `Experiment/core_code/scripts/`, project CLAUDE.md. Out: `precache_explores.py` (separate schema), `methods/<method>.py` solve function bodies (only the `build_solve_fn` adapter changes how kwargs are passed in), `backends/`, `benchmarks/`.

**Working directory for all paths in this plan:** `/data3/peijia/dr-claw/Explain/Experiment/core_code/`. Plan/spec docs live under `/data3/peijia/dr-claw/Explain/docs/superpowers/`.

---

## File Structure

| File | Role | Action |
|---|---|---|
| `methods/specs.py` | YAML schema layer: 8 method spec classes + discriminated union | Create |
| `methods/registry.py` | Behavior layer: `MethodConfig` base + 8 subclasses + `METHODS` dict + `get_method()` | Create |
| `methods/__init__.py` | Re-export `get_method`, `MethodSpec` for callers | Modify |
| `methods/base.py` | Drop residual `if infra.cache_only: assert ...` block (lines 334-335) | Modify (-2 lines) |
| `eval.py` | EvalConfig: drop scattered method fields, add `method: MethodSpec`, drop `_validate_method_constraints`, drop `no_cache_only`. main(): replace if/elif dispatch with `method.build_solve_fn(spec)`, replace pre_filter/preflight blocks with method calls | Modify (~-110 lines net) |
| `tools/migrate_yaml_to_method_block.py` | One-shot migration script for 83 YAMLs | Create (delete after use) |
| `scripts/**/*.yaml` (83 files with `method:`) | Reshape: wrap method-specific fields under `method:` block, drop dead fields, drop `no_cache_only` | Auto-modify via script |
| `CLAUDE.md` (project + Experiment/core_code) | Update YAML schema docs | Modify |

---

## Migration Order Rationale

1. **Specs and Registry first** (Tasks 1-3) — pure additions, no existing code touched. If we break something, only new files need rollback.
2. **Migration script before eval.py rewire** (Tasks 4-5) — YAMLs must be in new shape BEFORE eval.py expects them; otherwise YAML loading breaks.
3. **EvalConfig and dispatch together** (Tasks 6-7) — these are coupled; partial commit leaves a non-running system.
4. **Cleanup last** (Tasks 8-9-10) — small, safe deletions, then smoke tests, then docs.

---

## Conventions

- **Conda env:** all Python execution uses `conda run -n explain --no-capture-output python ...` (per project CLAUDE.md). Inside this plan, the working dir is `Experiment/core_code/`.
- **Commits:** one commit per task. Commit messages follow the existing repo style (`refactor(methods): ...`, `feat(methods): ...`, etc.).
- **Verification commands:** every task ends with a verification step that runs in <30 seconds and prints concrete output the engineer can compare against.

---

### Task 1: Create `methods/specs.py` — discriminated MethodSpec union

**Files:**
- Create: `Experiment/core_code/methods/specs.py`

**Goal:** Define 8 Pydantic Spec classes (one per method) + the `MethodSpec` Annotated Union with `discriminator="name"`. Mirror the structure of `Experiment/core_code/benchmarks/specs.py` exactly.

- [ ] **Step 1: Inspect benchmark specs.py for the exact pattern to mirror**

Run: `cat Experiment/core_code/benchmarks/specs.py | head -40`
Expected: shows `_Spec` base with `model_config = {"extra": "forbid"}`, `Annotated[Union[...], Field(discriminator="name")]`. Confirm the pattern.

- [ ] **Step 2: Locate the `SamplingConfig` import (used by TTSAgentSpec)**

Run: `grep -n "class SamplingConfig" Experiment/core_code/eval.py`
Expected: one line, e.g. `class SamplingConfig(BaseModel):`. Note the line number for the import path (`from eval import SamplingConfig` would create a circular import — instead, lift `SamplingConfig` into `methods/specs.py` directly OR import lazily from inside TTSAgentSpec). Decision: define `SamplingConfig` in `methods/specs.py` and have eval.py re-export from there.

- [ ] **Step 3: Write `methods/specs.py`**

```python
"""Method spec sub-schemas: discriminated union over method name.

Mirror of benchmarks/specs.py. Each method gets one Pydantic class enumerating
exactly its valid fields. extra="forbid" rejects unknown YAML keys, so dead
fields (e.g. orchestrator_model in self-refine YAMLs) fail validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


class SamplingConfig(BaseModel):
    """vLLM sampling block. Lifted here so methods/specs.py can own the schema
    end-to-end without circular import to eval.py."""
    model_config = {"extra": "forbid"}
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    enable_thinking: bool | None = None
    max_tokens: int | None = None


class _MethodSpec(BaseModel):
    model_config = {"extra": "forbid"}


class TTSAgentSpec(_MethodSpec):
    name: Literal["tts-agent"]
    orchestrator_model: str
    explore_model: str
    integrate_model: str | None = None
    cache_dir: Path
    no_integrate: bool = False
    num_explores: int = 8
    num_rollouts: int = 1
    sampling: SamplingConfig | None = None

    @model_validator(mode="after")
    def _check_integrate(self):
        if not self.no_integrate:
            assert self.integrate_model, (
                "tts-agent requires integrate_model unless no_integrate=true"
            )
        return self


class TTSAgentMultiSpec(_MethodSpec):
    name: Literal["tts-agent-multi"]
    orchestrator_model: str
    cache_dirs: dict[str, Path]
    model_budgets: dict[str, int]
    exploration_effort: Literal["low", "medium", "high"] | None = None
    num_explores: int = 8


class TTSAgentEffortSpec(_MethodSpec):
    name: Literal["tts-agent-effort"]
    orchestrator_model: str
    explore_model: str
    cache_dirs: dict[str, Path]
    effort_budgets: dict[str, int]
    num_explores: int = 8


class SelfRefineSpec(_MethodSpec):
    name: Literal["self-refine"]
    explore_model: str
    cache_dir: Path
    num_explores: int = 8


class SocraticSelfRefineSpec(_MethodSpec):
    name: Literal["socratic-self-refine"]
    explore_model: str
    cache_dir: Path
    num_explores: int = 8


class BudgetForcingSpec(_MethodSpec):
    name: Literal["budget-forcing"]
    explore_model: str
    cache_dir: Path
    num_explores: int = 8


class RerankSpec(_MethodSpec):
    name: Literal["rerank"]
    reward_model: str
    cache_dir: Path
    # No explore_model / integrate_model / num_explores: rerank reads cached
    # explores and scores them with a reward model, no LLM call, no new explore.


class StandaloneIntegratorSpec(_MethodSpec):
    name: Literal["standalone-integrator"]
    integrate_model: str
    cache_dir: Path
    # No explore_model / num_explores: reads cached explores, single integrate call.


MethodSpec = Annotated[
    Union[
        TTSAgentSpec, TTSAgentMultiSpec, TTSAgentEffortSpec,
        SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
        RerankSpec, StandaloneIntegratorSpec,
    ],
    Field(discriminator="name"),
]
```

- [ ] **Step 4: Verify file parses and discriminator works**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import sys; sys.path.insert(0, 'Experiment/core_code')
from methods.specs import MethodSpec, TTSAgentSpec, RerankSpec
from pydantic import TypeAdapter
adapter = TypeAdapter(MethodSpec)

# Valid tts-agent (with no_integrate=true, no integrate_model)
s = adapter.validate_python({
    'name': 'tts-agent',
    'orchestrator_model': 'm',
    'explore_model': 'm',
    'cache_dir': '/tmp/x',
    'no_integrate': True,
})
print('tts-agent OK:', type(s).__name__, '| no_integrate=', s.no_integrate)
assert isinstance(s, TTSAgentSpec)

# Valid rerank
r = adapter.validate_python({
    'name': 'rerank',
    'reward_model': 'opengvlab/visualprm',
    'cache_dir': '/tmp/x',
})
print('rerank OK:', type(r).__name__)
assert isinstance(r, RerankSpec)

# Invalid: orchestrator_model in rerank (should fail with extra_forbidden)
try:
    adapter.validate_python({
        'name': 'rerank',
        'reward_model': 'm',
        'cache_dir': '/tmp/x',
        'orchestrator_model': 'should not be here',
    })
    print('FAIL: rerank accepted orchestrator_model')
except Exception as e:
    print('rerank correctly rejects orchestrator_model:', 'extra_forbidden' in str(e))

# Invalid: tts-agent no_integrate=False without integrate_model
try:
    adapter.validate_python({
        'name': 'tts-agent',
        'orchestrator_model': 'm',
        'explore_model': 'm',
        'cache_dir': '/tmp/x',
        'no_integrate': False,
    })
    print('FAIL: tts-agent accepted missing integrate_model')
except Exception as e:
    print('tts-agent correctly rejects missing integrate_model:', 'integrate_model' in str(e))
"
```
Expected output (3 lines + 2 verification lines):
```
tts-agent OK: TTSAgentSpec | no_integrate= True
rerank OK: RerankSpec
rerank correctly rejects orchestrator_model: True
tts-agent correctly rejects missing integrate_model: True
```

- [ ] **Step 5: Commit**

```bash
git add Experiment/core_code/methods/specs.py
git commit -m "feat(methods): add MethodSpec discriminated union schema

Mirrors BenchmarkSpec. Each method gets one Pydantic class with
extra='forbid' so dead/unknown YAML fields fail validation. Sets
up the schema layer for the upcoming method-block refactor."
```

---

### Task 2: Create `methods/registry.py` — MethodConfig base + 8 subclasses

**Files:**
- Create: `Experiment/core_code/methods/registry.py`

**Goal:** Behavior layer. Each method name maps to a class with `cache_only`, `pre_flight_check`, `pre_filter_by_cache`, `supports_num_rollouts`, `consumes_sampling_block`, plus methods `build_solve_fn(spec)`, `derive_evaluate_args(spec)`, `filter_rows(rows, cache_dir, benchmark)`, `preflight(rows, cache_dir, num_explores, num)`.

- [ ] **Step 1: Write `methods/registry.py`**

```python
"""Method behavior registry: per-method runtime configuration + dispatch.

Pairs with methods/specs.py (data layer). Each method name maps to one
MethodConfig subclass that declares its runtime properties (cache_only,
pre_flight_check, etc.) and provides build_solve_fn / filter_rows /
preflight hooks that eval.py calls polymorphically.
"""
from __future__ import annotations

import functools
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal

from methods.specs import (
    MethodSpec, TTSAgentSpec, TTSAgentMultiSpec, TTSAgentEffortSpec,
    SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
    RerankSpec, StandaloneIntegratorSpec,
)


class MethodConfig(ABC):
    name: str
    cache_only: bool                              # explore cache miss → AssertionError?
    pre_flight_check: bool = False                # banner-time cache completeness?
    pre_filter_by_cache: bool = False             # drop rows with no cache before run?
    supports_num_rollouts: bool = False           # rejection-sampling K>1?
    consumes_sampling_block: bool = False         # YAML sampling block transparently passed?

    @abstractmethod
    def build_solve_fn(self, spec: MethodSpec) -> Callable:
        """Return the partialed solve function for this method, bound to spec fields."""

    def derive_evaluate_args(self, spec: MethodSpec) -> dict:
        """Return kwargs for evaluate(): orchestrator_model / explore_model /
        integrate_model / cache_dirs_multi. Each method picks what makes sense."""
        return {
            "orchestrator_model": getattr(spec, "orchestrator_model", ""),
            "explore_model": getattr(spec, "explore_model", ""),
            "integrate_model": getattr(spec, "integrate_model", "") or "",
            "cache_dirs_multi": getattr(spec, "cache_dirs", None),
        }

    def filter_rows(self, rows: list[dict], cache_dir: Path | None, benchmark) -> list[dict]:
        """Default: pass-through. Override in pre_filter_by_cache methods."""
        if not self.pre_filter_by_cache or cache_dir is None:
            return rows
        cached_ids = {
            p.name for p in cache_dir.iterdir()
            if p.is_dir() and (p / "explore_1" / "result.json").exists()
        }
        before = len(rows)
        out = [r for r in rows if benchmark.get_id(r) in cached_ids]
        print(f"{self.name}: {len(out)} questions with cache (from {before})")
        return out

    def preflight(
        self,
        rows: list[dict],
        cache_dir: Path | None,
        num_explores: int,
        num: int | None,
        benchmark,
    ) -> None:
        """Default: no-op. Override in pre_flight_check methods."""
        if not self.pre_flight_check or cache_dir is None:
            return
        rows_to_check = rows if num is None else rows[:num]
        qids = [benchmark.get_id(r) for r in rows_to_check]
        missing: list[tuple[str, int]] = []
        for qid in qids:
            for idx in range(1, num_explores + 1):
                if not (cache_dir / qid / f"explore_{idx}" / "result.json").exists():
                    missing.append((qid, idx))
        if missing:
            sample = ", ".join(f"({q}, explore_{i})" for q, i in missing[:10])
            raise AssertionError(
                f"Cache pre-flight FAILED: {len(missing)} missing entries "
                f"(of {len(qids) * num_explores} expected) under {cache_dir}. "
                f"First 10: {sample}"
            )
        print(
            f"Cache pre-flight OK: {len(qids)} qids x {num_explores} explores "
            f"= {len(qids) * num_explores} cache files present at {cache_dir}"
        )


class TTSAgentMethod(MethodConfig):
    name = "tts-agent"
    cache_only = True
    pre_flight_check = True
    supports_num_rollouts = True
    consumes_sampling_block = True

    def build_solve_fn(self, spec: TTSAgentSpec):
        from methods.tts_agent import solve
        return solve


class TTSAgentMultiMethod(MethodConfig):
    name = "tts-agent-multi"
    cache_only = True

    def build_solve_fn(self, spec: TTSAgentMultiSpec):
        from methods.tts_agent_multi import solve
        return functools.partial(
            solve,
            cache_dirs=spec.cache_dirs,
            model_budgets=spec.model_budgets,
            exploration_effort=spec.exploration_effort,
        )

    def derive_evaluate_args(self, spec: TTSAgentMultiSpec) -> dict:
        # multi reuses orchestrator_model as integrate_model downstream
        return {
            "orchestrator_model": spec.orchestrator_model,
            "explore_model": "",   # unused by multi
            "integrate_model": spec.orchestrator_model,
            "cache_dirs_multi": spec.cache_dirs,
        }


class TTSAgentEffortMethod(MethodConfig):
    name = "tts-agent-effort"
    cache_only = True

    def build_solve_fn(self, spec: TTSAgentEffortSpec):
        from methods.tts_agent_effort import solve
        return functools.partial(
            solve,
            cache_dirs=spec.cache_dirs,
            effort_budgets=spec.effort_budgets,
        )

    def derive_evaluate_args(self, spec: TTSAgentEffortSpec) -> dict:
        return {
            "orchestrator_model": spec.orchestrator_model,
            "explore_model": spec.explore_model,
            "integrate_model": spec.orchestrator_model,
            "cache_dirs_multi": spec.cache_dirs,
        }


class SelfRefineMethod(MethodConfig):
    name = "self-refine"
    cache_only = False                  # generates new refines on top of cached drafts
    pre_flight_check = True

    def build_solve_fn(self, spec: SelfRefineSpec):
        from methods.self_refine import solve
        return solve


class SocraticSelfRefineMethod(MethodConfig):
    name = "socratic-self-refine"
    cache_only = False
    pre_flight_check = True

    def build_solve_fn(self, spec: SocraticSelfRefineSpec):
        from methods.socratic_self_refine import solve
        return solve


class BudgetForcingMethod(MethodConfig):
    name = "budget-forcing"
    cache_only = False
    pre_flight_check = True

    def build_solve_fn(self, spec: BudgetForcingSpec):
        from methods.budget_forcing import solve
        return solve


class RerankMethod(MethodConfig):
    name = "rerank"
    cache_only = True
    pre_filter_by_cache = True

    def build_solve_fn(self, spec: RerankSpec):
        from methods.reward_rerank import solve
        return functools.partial(solve, reward_model_name=spec.reward_model)


class StandaloneIntegratorMethod(MethodConfig):
    name = "standalone-integrator"
    cache_only = True
    pre_filter_by_cache = True

    def build_solve_fn(self, spec: StandaloneIntegratorSpec):
        from methods.standalone_integrator import solve
        return functools.partial(solve, integrate_model=spec.integrate_model)


METHODS: dict[str, type[MethodConfig]] = {
    "tts-agent": TTSAgentMethod,
    "tts-agent-multi": TTSAgentMultiMethod,
    "tts-agent-effort": TTSAgentEffortMethod,
    "self-refine": SelfRefineMethod,
    "socratic-self-refine": SocraticSelfRefineMethod,
    "budget-forcing": BudgetForcingMethod,
    "rerank": RerankMethod,
    "standalone-integrator": StandaloneIntegratorMethod,
}


def get_method(name: str) -> MethodConfig:
    assert name in METHODS, f"Unknown method: {name!r}. Available: {list(METHODS)}"
    return METHODS[name]()
```

- [ ] **Step 2: Verify all 8 methods register and instantiate**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import sys; sys.path.insert(0, 'Experiment/core_code')
from methods.registry import METHODS, get_method
print('Registered methods:', list(METHODS.keys()))
for name in METHODS:
    m = get_method(name)
    assert m.name == name
    print(f'  {name}: cache_only={m.cache_only}, pre_flight={m.pre_flight_check}, pre_filter={m.pre_filter_by_cache}')
"
```
Expected:
```
Registered methods: ['tts-agent', 'tts-agent-multi', 'tts-agent-effort', 'self-refine', 'socratic-self-refine', 'budget-forcing', 'rerank', 'standalone-integrator']
  tts-agent: cache_only=True, pre_flight=True, pre_filter=False
  tts-agent-multi: cache_only=True, pre_flight=False, pre_filter=False
  tts-agent-effort: cache_only=True, pre_flight=False, pre_filter=False
  self-refine: cache_only=False, pre_flight=True, pre_filter=False
  socratic-self-refine: cache_only=False, pre_flight=True, pre_filter=False
  budget-forcing: cache_only=False, pre_flight=True, pre_filter=False
  rerank: cache_only=True, pre_flight=False, pre_filter=True
  standalone-integrator: cache_only=True, pre_flight=False, pre_filter=True
```

- [ ] **Step 3: Commit**

```bash
git add Experiment/core_code/methods/registry.py
git commit -m "feat(methods): add MethodConfig runtime registry

Behavior layer paired with methods/specs.py. MethodConfig declares
per-method properties (cache_only, pre_flight_check, etc.) and
provides build_solve_fn / filter_rows / preflight hooks that eval.py
will call polymorphically once the dispatcher is rewired."
```

---

### Task 3: Wire `methods/__init__.py`

**Files:**
- Modify: `Experiment/core_code/methods/__init__.py`

**Goal:** Re-export `MethodSpec` and `get_method` so callers do `from methods import MethodSpec, get_method` instead of reaching into submodules.

- [ ] **Step 1: Inspect current __init__.py**

Run: `cat Experiment/core_code/methods/__init__.py`
Expected: empty file (1 line).

- [ ] **Step 2: Write re-exports**

Replace contents with:
```python
"""Method registry: schema layer (specs) + behavior layer (registry)."""
from methods.specs import MethodSpec
from methods.registry import MethodConfig, METHODS, get_method

__all__ = ["MethodSpec", "MethodConfig", "METHODS", "get_method"]
```

- [ ] **Step 3: Verify imports work**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import sys; sys.path.insert(0, 'Experiment/core_code')
from methods import MethodSpec, MethodConfig, get_method
print('imports ok')
print('  MethodSpec:', MethodSpec)
print('  get_method(\"tts-agent\"):', get_method('tts-agent'))
"
```
Expected: `imports ok` plus the two pretty-printed values.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/methods/__init__.py
git commit -m "feat(methods): re-export MethodSpec and get_method from package root"
```

---

### Task 4: Write migration script `tools/migrate_yaml_to_method_block.py`

**Files:**
- Create: `Experiment/core_code/tools/migrate_yaml_to_method_block.py`

**Goal:** A one-shot script that reads each YAML, identifies its `method:` value, lifts the method-specific fields into a nested `method:` block, drops dead fields and `no_cache_only`, and writes the new YAML. Includes `--dry-run` flag.

- [ ] **Step 1: Create `tools/` directory if missing**

Run: `mkdir -p Experiment/core_code/tools`

- [ ] **Step 2: Write migration script**

```python
"""One-shot migration: flat YAML → method-block YAML.

Per method name, the script knows which fields belong INSIDE the method block
vs. which stay at top level. For each YAML in scripts/**/*.yaml that has a
top-level `method:` field, it rewrites the file with:
  method:
    name: <old method value>
    <method-specific fields lifted here>
  <generic fields stay at top level>

Drops these fields outright (they are dead in the new schema):
  - no_cache_only (no longer a user toggle)
  - dead model fields per method (e.g. orchestrator_model in self-refine)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Per-method: which top-level fields move INTO the method block.
# Fields not listed here either stay outside (generic) or are dropped if
# they appear but are not allowed for this method.
METHOD_BLOCK_FIELDS: dict[str, set[str]] = {
    "tts-agent": {
        "orchestrator_model", "explore_model", "integrate_model",
        "cache_dir", "no_integrate", "num_explores", "num_rollouts",
        "sampling",
    },
    "tts-agent-multi": {
        "orchestrator_model", "cache_dirs", "model_budgets",
        "exploration_effort", "num_explores",
    },
    "tts-agent-effort": {
        "orchestrator_model", "explore_model", "cache_dirs",
        "effort_budgets", "num_explores",
    },
    "self-refine": {"explore_model", "cache_dir", "num_explores"},
    "socratic-self-refine": {"explore_model", "cache_dir", "num_explores"},
    "budget-forcing": {"explore_model", "cache_dir", "num_explores"},
    "rerank": {"reward_model", "cache_dir"},
    "standalone-integrator": {"integrate_model", "cache_dir"},
}

# Always-dropped fields (irrespective of method).
ALWAYS_DROP: set[str] = {"no_cache_only"}


def migrate(yaml_path: Path, dry_run: bool = False) -> bool:
    """Rewrite one YAML in-place. Returns True if file changed."""
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"{yaml_path}: top-level not a dict"

    if "method" not in data:
        return False  # precache YAMLs etc., not in scope
    if isinstance(data["method"], dict):
        return False  # already migrated

    method_name = data["method"]
    if method_name not in METHOD_BLOCK_FIELDS:
        print(f"  WARN: {yaml_path}: unknown method {method_name!r}, skipping")
        return False
    allowed_inside = METHOD_BLOCK_FIELDS[method_name]

    method_block: dict = {"name": method_name}
    new_top: dict = {}
    dropped: list[str] = []

    for k, v in data.items():
        if k == "method":
            continue
        if k in ALWAYS_DROP:
            dropped.append(k)
            continue
        if k in allowed_inside:
            method_block[k] = v
        else:
            new_top[k] = v
    # Detect dead method-specific fields (e.g. orchestrator_model in self-refine
    # YAMLs) — these are top-level fields that look method-related but the spec
    # for THIS method does not allow them.
    METHOD_RELATED = {
        "orchestrator_model", "explore_model", "integrate_model", "reward_model",
        "cache_dir", "cache_dirs", "model_budgets", "effort_budgets",
        "exploration_effort", "no_integrate", "num_explores", "num_rollouts",
        "sampling",
    }
    for k in list(new_top.keys()):
        if k in METHOD_RELATED and k not in allowed_inside:
            dropped.append(k)
            del new_top[k]

    # Reassemble: benchmark first if present, then method, then everything else.
    out: dict = {}
    if "benchmark" in new_top:
        out["benchmark"] = new_top.pop("benchmark")
    if "backend" in new_top:
        out["backend"] = new_top.pop("backend")
    out["method"] = method_block
    for k, v in new_top.items():
        out[k] = v

    if dry_run:
        print(f"--- {yaml_path} (dry-run) ---")
        print(yaml.safe_dump(out, sort_keys=False, default_flow_style=False))
        if dropped:
            print(f"  dropped: {dropped}")
        return True

    with yaml_path.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
    if dropped:
        print(f"  {yaml_path}: dropped {dropped}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("Experiment/core_code/scripts"),
        help="Directory to walk for *.yaml files",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N files")
    args = parser.parse_args()

    yamls = sorted(args.root.rglob("*.yaml"))
    if args.limit:
        yamls = yamls[: args.limit]

    changed = 0
    for p in yamls:
        if migrate(p, dry_run=args.dry_run):
            changed += 1
    print(f"\n{'Would migrate' if args.dry_run else 'Migrated'} {changed}/{len(yamls)} files.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test the script in dry-run mode on ONE file**

Run:
```bash
conda run -n explain --no-capture-output python Experiment/core_code/tools/migrate_yaml_to_method_block.py \
  --root Experiment/core_code/scripts/aime2025/sonnet \
  --dry-run --limit 1
```
Expected output: a printed YAML with `method:` as a nested block, e.g.:
```yaml
benchmark:
  name: aime2025
backend: claude
method:
  name: tts-agent
  orchestrator_model: claude-sonnet-4-6
  explore_model: claude-sonnet-4-6
  integrate_model: claude-sonnet-4-6
  cache_dir: ...
  num_explores: 8
num: ...
seed: 42
log_dir: ...
resume: ...
```
The original file on disk is unchanged (dry-run).

- [ ] **Step 4: Verify ONE file's dry-run output parses through MethodSpec**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import yaml
from pydantic import TypeAdapter
from methods.specs import MethodSpec

# Read post-migration form by running the script in-process for one file
import sys
sys.path.insert(0, 'Experiment/core_code')
from tools.migrate_yaml_to_method_block import migrate
import shutil, tempfile
src = 'Experiment/core_code/scripts/aime2025/sonnet/aime2025_sonnet_delegated.yaml'
with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+', delete=False) as f:
    shutil.copy(src, f.name)
    migrate(__import__('pathlib').Path(f.name))
    f.seek(0)
    data = yaml.safe_load(open(f.name))
adapter = TypeAdapter(MethodSpec)
spec = adapter.validate_python(data['method'])
print('parsed', type(spec).__name__, '— name=', spec.name)
"
```
Expected: `parsed TTSAgentSpec — name= tts-agent`

- [ ] **Step 5: Commit**

```bash
git add Experiment/core_code/tools/migrate_yaml_to_method_block.py
git commit -m "tools: one-shot migration script for YAML method-block reshape

Reshapes flat method config into discriminated-union method block.
Drops no_cache_only (deprecated) and dead per-method fields. Run
once via Task 5; can be deleted afterward (kept here for record)."
```

---

### Task 5: Run migration on all 83 YAMLs and verify each parses

**Files:**
- Modify: 83 YAMLs under `Experiment/core_code/scripts/` (auto via script)

**Goal:** All eval YAMLs end up in the new method-block shape. Every one parses cleanly through `MethodSpec.validate_python(data["method"])`.

- [ ] **Step 1: Run migration script for real (writes to disk)**

Run:
```bash
conda run -n explain --no-capture-output python Experiment/core_code/tools/migrate_yaml_to_method_block.py \
  --root Experiment/core_code/scripts
```
Expected output ends with: `Migrated 83/<N> files.` where N is the total number of YAMLs in scripts/ (precache YAMLs are skipped because they have no `method:` field).

- [ ] **Step 2: Spot-check 4 representative YAMLs**

Run:
```bash
for f in \
  Experiment/core_code/scripts/aime2025/sonnet/aime2025_sonnet_delegated.yaml \
  Experiment/core_code/scripts/hle/sonnet/hle_sonnet_self_refine.yaml \
  Experiment/core_code/scripts/hle/multi_model/hle_multi_delegated.yaml \
  Experiment/core_code/scripts/babyvision/sonnet/babyvision_sonnet_visualprm_rerank.yaml \
; do
  echo "=== $f ==="
  cat "$f"
  echo ""
done
```
Expected: every YAML has `method:` as a nested mapping with `name:` inside. self-refine YAML has no `orchestrator_model`/`integrate_model` at any level. rerank YAML has `method.reward_model`. None of them has `no_cache_only`.

- [ ] **Step 3: Parse every YAML through MethodSpec — fail loud on any error**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import yaml
from pathlib import Path
from pydantic import TypeAdapter
from methods.specs import MethodSpec

adapter = TypeAdapter(MethodSpec)
yamls = sorted(Path('Experiment/core_code/scripts').rglob('*.yaml'))
ok = 0
fails: list[tuple[str, str]] = []
for p in yamls:
    data = yaml.safe_load(open(p))
    if 'method' not in data:
        continue
    try:
        adapter.validate_python(data['method'])
        ok += 1
    except Exception as e:
        fails.append((str(p), str(e)[:200]))
print(f'OK: {ok}')
print(f'Failed: {len(fails)}')
for p, msg in fails[:5]:
    print(f'  {p}: {msg}')
assert not fails, f'{len(fails)} YAML(s) failed MethodSpec validation'
"
```
Expected: `OK: 83` and `Failed: 0`. If any failure, the script prints the path + first 200 chars of the pydantic error — fix the migration script (Task 4 Step 2) and re-run from Step 1.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/scripts/
git commit -m "refactor(scripts): migrate 83 YAMLs to method-block schema

Run by tools/migrate_yaml_to_method_block.py. Each method-specific
field moved under nested method: block; dead fields dropped (e.g.
orchestrator_model in self-refine YAMLs); no_cache_only removed.
All YAMLs verified to parse through MethodSpec."
```

---

### Task 6: Wire `EvalConfig` to use MethodSpec block

**Files:**
- Modify: `Experiment/core_code/eval.py` (EvalConfig class section, ~lines 80-165)

**Goal:** Replace scattered method-related fields on EvalConfig with one `method: MethodSpec` field. Delete `_validate_method_constraints` (46 lines). Delete `no_cache_only`. Re-export `SamplingConfig` from `methods.specs` (since the field moved into TTSAgentSpec, eval.py no longer hosts it directly).

- [ ] **Step 1: Read current EvalConfig to identify exact line ranges**

Run: `grep -nE "^class EvalConfig|class SamplingConfig|_validate_method_constraints|^    [a-z_]+: " Experiment/core_code/eval.py | head -40`

Expected: a list of EvalConfig fields and the validator method. Note line numbers for the changes.

- [ ] **Step 2: Modify `eval.py` EvalConfig — replace scattered fields with method: MethodSpec**

Open `Experiment/core_code/eval.py`. In the EvalConfig class:

DELETE these top-level fields:
- `method: Literal[...]` (the string discriminator — replaced by `method: MethodSpec`)
- `orchestrator_model: str = ""`
- `explore_model: str = ""`
- `integrate_model: str = ""`
- `reward_model: str = ""`
- `cache_dir: Path | None = None`
- `cache_dirs: dict[str, Path] = Field(default_factory=dict)`
- `model_budgets: dict[str, int] = Field(default_factory=dict)`
- `effort_budgets: dict[str, int] = Field(default_factory=dict)`
- `exploration_effort: Literal[...] | None = None`
- `no_integrate: bool = False`
- `num_explores: int = 8`
- `num_rollouts: int = 1`
- `sampling: SamplingConfig | None = None`
- `no_cache_only: bool = False`

ADD (in their place):
```python
method: MethodSpec
```

DELETE the entire `_validate_method_constraints` model_validator (the `@model_validator(mode="after") def _validate_method_constraints(self): ...` block, ~46 lines).

DELETE the in-eval `class SamplingConfig(BaseModel): ...` definition (it now lives in `methods/specs.py`).

ADD imports at the top of eval.py:
```python
from methods import MethodSpec, get_method
from methods.specs import SamplingConfig
```

- [ ] **Step 3: Verify eval.py still parses (Python syntax)**

Run: `conda run -n explain --no-capture-output python -c "import ast; ast.parse(open('Experiment/core_code/eval.py').read()); print('parse ok')"`
Expected: `parse ok`.

- [ ] **Step 4: Verify each migrated YAML loads through the new EvalConfig**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import sys
sys.path.insert(0, 'Experiment/core_code')
import yaml
from pathlib import Path
from eval import EvalConfig

yamls = sorted(Path('Experiment/core_code/scripts').rglob('*.yaml'))
ok = 0
fails = []
for p in yamls:
    data = yaml.safe_load(open(p))
    if 'method' not in data:
        continue
    try:
        EvalConfig.model_validate(data)
        ok += 1
    except Exception as e:
        fails.append((str(p), str(e)[:300]))
print(f'OK: {ok} / Failed: {len(fails)}')
for p, msg in fails[:3]:
    print(f'  {p}:')
    print(f'    {msg}')
assert not fails, 'fix migration or schema before continuing'
"
```
Expected: `OK: 83 / Failed: 0`.

- [ ] **Step 5: Commit**

```bash
git add Experiment/core_code/eval.py
git commit -m "refactor(eval): replace scattered method fields with MethodSpec block

EvalConfig now carries one 'method: MethodSpec' field instead of 14
top-level method-related fields. _validate_method_constraints (46
lines of if/elif over method names) is gone — validation moves into
each Spec class via Pydantic. SamplingConfig migrated to methods/specs.py.
no_cache_only deleted (cache_only is now a class attribute on MethodConfig)."
```

---

### Task 7: Wire `eval.py` main() dispatch through MethodConfig

**Files:**
- Modify: `Experiment/core_code/eval.py` (lines ~798-955, the main dispatch + filter + preflight + InfraConfig construction + evaluate() call)

**Goal:** Replace the if/elif method dispatch (33 lines), the integrate_model/cache_dirs_multi derivation (9 lines), the rerank/standalone filter (5 lines), and the pre-flight (28 lines) with method-method calls. Update InfraConfig to source `cache_only` from method class and `cache_dir` from `cfg.method`.

- [ ] **Step 1: Read current main() dispatch section**

Run: `sed -n '795,955p' Experiment/core_code/eval.py`
Expected: shows the if/elif method dispatch, integrate_model/cache_dirs_multi derivation, rerank filter, pre-flight, num_rollouts expansion, InfraConfig, evaluate() call.

- [ ] **Step 2: Replace the dispatch + derivation + filter + preflight blocks**

In `eval.py` `async def async_main()`:

DELETE the entire `if cfg.method == "self-refine": ... else: from methods.tts_agent import solve` block (lines ~798-830, 33 lines).

DELETE the `if cfg.method in ("tts-agent-multi", "tts-agent-effort"):` integrate_model/cache_dirs_multi block (lines ~832-840, 9 lines).

DELETE the `if cfg.method in ("rerank", "standalone-integrator") and cache_dir:` filter block (lines ~867-871, 5 lines).

DELETE the `if not cfg.no_cache_only and cache_dir is not None and cfg.method in (...)` pre-flight block (lines ~873-900, 28 lines).

ADD in their place — directly after `bench_filters = cfg.benchmark.model_dump(...)` (around line 848):
```python
method = get_method(cfg.method.name)
solve = method.build_solve_fn(cfg.method)
runtime = method.derive_evaluate_args(cfg.method)
integrate_model = runtime["integrate_model"]
cache_dirs_multi = runtime["cache_dirs_multi"]
cache_dir = getattr(cfg.method, "cache_dir", None)
```

And right before the `if cfg.num_rollouts > 1:` row-expansion block (around line 902, but `cfg.num_rollouts` no longer exists at top level — it lives in `cfg.method.num_rollouts` for tts-agent only), replace `cfg.num_rollouts` everywhere in main() with `getattr(cfg.method, "num_rollouts", 1)`.

ADD the filter and preflight calls after `filtered = benchmark.filter_dataset(...)` (around line 846, BEFORE the shuffle/skip lines):
```python
filtered = method.filter_rows(filtered, cache_dir, benchmark)
```

ADD the preflight call AFTER the `if cfg.skip > 0:` block (around line 858) but BEFORE the row-expansion block:
```python
method.preflight(filtered, cache_dir, getattr(cfg.method, "num_explores", 8), cfg.num, benchmark)
```

UPDATE the `InfraConfig(...)` construction (line ~918) to source from `cfg.method`:
```python
infra = InfraConfig(
    backend=cfg.backend,
    max_iterations=getattr(cfg.method, "num_explores", 8),
    cache_dir=cache_dir,
    cache_only=method.cache_only,
    budget_tokens=cfg.budget_tokens,
    effort=cfg.effort,
    timeout=cfg.timeout,
    benchmark=benchmark,
    quiet=not cfg.verbose,
    logger=None,
    enable_integrate=not getattr(cfg.method, "no_integrate", False),
    max_output_tokens=cfg.max_output_tokens,
)
```

UPDATE the `evaluate(...)` call to source `sampling` from `cfg.method`:
```python
sampling=cfg.method.sampling.model_dump() if getattr(cfg.method, "sampling", None) else None,
```

- [ ] **Step 3: Verify eval.py still parses**

Run: `conda run -n explain --no-capture-output python -c "import ast; ast.parse(open('Experiment/core_code/eval.py').read()); print('parse ok')"`
Expected: `parse ok`.

- [ ] **Step 4: Verify --help still works (catches import errors)**

Run: `conda run -n explain --no-capture-output python Experiment/core_code/eval.py --help 2>&1 | tail -10`
Expected: shows `usage: eval.py [-h] --config CONFIG`.

- [ ] **Step 5: Dry-load every YAML through full EvalConfig + dispatch through method.build_solve_fn**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import sys
sys.path.insert(0, 'Experiment/core_code')
import yaml
from pathlib import Path
from eval import EvalConfig
from methods import get_method

yamls = sorted(Path('Experiment/core_code/scripts').rglob('*.yaml'))
ok = 0
fails = []
for p in yamls:
    data = yaml.safe_load(open(p))
    if 'method' not in data:
        continue
    try:
        cfg = EvalConfig.model_validate(data)
        m = get_method(cfg.method.name)
        _ = m.build_solve_fn(cfg.method)
        _ = m.derive_evaluate_args(cfg.method)
        ok += 1
    except Exception as e:
        fails.append((str(p), str(e)[:300]))
print(f'Dispatched OK: {ok} / Failed: {len(fails)}')
for p, msg in fails[:3]:
    print(f'  {p}: {msg}')
assert not fails
"
```
Expected: `Dispatched OK: 83 / Failed: 0`.

- [ ] **Step 6: Commit**

```bash
git add Experiment/core_code/eval.py
git commit -m "refactor(eval): dispatch through MethodConfig polymorphism

Replaces ~85 lines of method-by-method if/elif blocks (dispatch,
integrate_model derivation, rerank filter, pre-flight check) with
unified calls into method.build_solve_fn / derive_evaluate_args /
filter_rows / preflight. New methods now require zero changes to
eval.py — only a new spec class + new MethodConfig subclass."
```

---

### Task 8: Drop residual `cache_only` assertion in `methods/base.py`

**Files:**
- Modify: `Experiment/core_code/methods/base.py:334-335`

**Goal:** Remove the `if infra.cache_only: assert infra.cache_dir is not None` block. This constraint is now expressed by the spec layer (multi/effort use cache_dirs not cache_dir; the spec validates).

- [ ] **Step 1: Read current lines**

Run: `sed -n '330,340p' Experiment/core_code/methods/base.py`
Expected: shows the `if infra.cache_only: assert infra.cache_dir is not None, "cache_only=True requires cache_dir"` block.

- [ ] **Step 2: Delete the 2-line block**

Edit `Experiment/core_code/methods/base.py`. Delete:
```python
    if infra.cache_only:
        assert infra.cache_dir is not None, "cache_only=True requires cache_dir"
```

- [ ] **Step 3: Verify file parses**

Run: `conda run -n explain --no-capture-output python -c "import ast; ast.parse(open('Experiment/core_code/methods/base.py').read()); print('parse ok')"`
Expected: `parse ok`.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/methods/base.py
git commit -m "refactor(methods): drop residual cache_only=>cache_dir assert

The constraint moved into the spec layer: multi/effort use cache_dirs
(no cache_dir at all), single-cache methods declare cache_dir as
required in their spec. The runtime check is redundant."
```

---

### Task 9: Smoke-test 4 YAML shapes end-to-end

**Files:** none modified

**Goal:** Confirm the refactored pipeline runs against real YAMLs without errors. One YAML per representative shape: tts-agent (single cache, paper main), tts-agent-multi (multi cache), self-refine (cache_only=False), rerank (pre_filter_by_cache).

- [ ] **Step 1: Pick 4 YAMLs and confirm they have tiny `num` for fast smoke**

Each smoke YAML should set `num: 2` (not the production `num: 100`). If they don't, copy them under a `tmp/` smoke dir with overridden `num`. Run:
```bash
mkdir -p Experiment/core_code/tmp/smoke
for src in \
  Experiment/core_code/scripts/aime2025/sonnet/aime2025_sonnet_delegated.yaml \
  Experiment/core_code/scripts/hle/sonnet/hle_sonnet_self_refine.yaml \
  Experiment/core_code/scripts/hle/multi_model/hle_multi_delegated.yaml \
  Experiment/core_code/scripts/babyvision/sonnet/babyvision_sonnet_visualprm_rerank.yaml \
; do
  base=$(basename "$src")
  conda run -n explain --no-capture-output python -c "
import yaml, sys
data = yaml.safe_load(open('$src'))
data['num'] = 2
data.pop('resume', None)  # smoke is fresh, no resume
yaml.safe_dump(data, open('Experiment/core_code/tmp/smoke/$base', 'w'), sort_keys=False)
print('wrote tmp/smoke/$base')
"
done
```
Expected: 4 lines `wrote tmp/smoke/<file>`.

- [ ] **Step 2: Run each smoke YAML through eval.py (just the launch banner — kill after preflight passes)**

For each smoke YAML, run with a 60-second timeout to capture the banner output:
```bash
for cfg in Experiment/core_code/tmp/smoke/*.yaml; do
  echo "=== smoke: $cfg ==="
  timeout 60 conda run -n explain --no-capture-output python Experiment/core_code/eval.py --config "$cfg" 2>&1 | head -30 || true
  echo ""
done
```

Expected per YAML:
- A `<BENCHMARK> Evaluation` banner with `Backend: ... | Orchestrator: ... | Explorer: ... | Integrator: ...`.
- Either `Cache pre-flight OK: ...` (for tts-agent / self-refine / rerank-NOT — multi has no preflight) or `<method>: N questions with cache (from M)` for rerank.
- No AssertionError, no KeyError, no AttributeError.

- [ ] **Step 3: Clean up smoke dir**

Run: `rm -rf Experiment/core_code/tmp/smoke`
Expected: silent.

- [ ] **Step 4: Commit (no code change — smoke verifies prior commits)**

If Step 2 showed any failure, go back and fix the offending Task. If all 4 banners were clean, no commit needed for this task — record the smoke pass in the next task's commit message instead.

---

### Task 10: Update docs in CLAUDE.md

**Files:**
- Modify: `Experiment/core_code/CLAUDE.md` (the "eval.py configuration" section)

**Goal:** Replace the description of the old flat schema with the new method-block schema. Add an example for each shape.

- [ ] **Step 1: Find the current section**

Run: `grep -n "## eval.py configuration\|^### " Experiment/core_code/CLAUDE.md | head -20`
Expected: location of the "eval.py configuration" section and its subsections.

- [ ] **Step 2: Replace the section content**

Edit `Experiment/core_code/CLAUDE.md`. Find the `## eval.py configuration` section. Replace its body with:

```markdown
## eval.py configuration

`eval.py` accepts exactly one CLI flag:

1. `--config <path>.yaml` — required. YAML is the single source of truth.

The YAML has two discriminated-union blocks plus generic top-level fields.

### `benchmark:` block (discriminated by `name:`)

See `benchmarks/specs.py`. Each benchmark name maps to a Pydantic spec with
`extra: forbid`, so unknown filter keys fail validation.

### `method:` block (discriminated by `name:`)

See `methods/specs.py`. Each method has its own Spec class. Method-specific
fields (orchestrator_model, explore_model, integrate_model, reward_model,
cache_dir, cache_dirs, model_budgets, effort_budgets, exploration_effort,
no_integrate, num_explores, num_rollouts, sampling) live INSIDE the method
block. extra="forbid" means writing a field that the spec doesn't allow
(e.g. `orchestrator_model:` in a self-refine YAML) fails validation.

| Method | Required fields inside `method:` block |
|---|---|
| `tts-agent` | orchestrator_model, explore_model, cache_dir; integrate_model unless no_integrate=true |
| `tts-agent-multi` | orchestrator_model, cache_dirs, model_budgets |
| `tts-agent-effort` | orchestrator_model, explore_model, cache_dirs, effort_budgets |
| `self-refine` | explore_model, cache_dir |
| `socratic-self-refine` | explore_model, cache_dir |
| `budget-forcing` | explore_model, cache_dir |
| `rerank` | reward_model, cache_dir |
| `standalone-integrator` | integrate_model, cache_dir |

### Top-level generic fields

`backend`, `budget_tokens`, `effort`, `timeout`, `max_output_tokens`,
`num`, `num_workers`, `seed`, `shuffle`, `skip`, `log_dir`, `resume`,
`verbose`. None of these vary by method.

### Example: tts-agent

```yaml
benchmark:
  name: hle
  subset: gold
backend: claude
method:
  name: tts-agent
  orchestrator_model: claude-sonnet-4-6
  explore_model: claude-sonnet-4-6
  cache_dir: /cache/hle/sonnet/gold
  no_integrate: true
  num_explores: 8
num: 100
seed: 42
log_dir: /run/hle/sonnet
```

### Example: rerank

```yaml
benchmark:
  name: babyvision
backend: claude
method:
  name: rerank
  reward_model: OpenGVLab/VisualPRM-8B
  cache_dir: /cache/babyvision/sonnet
num_workers: 16
log_dir: /run/babyvision/sonnet_visualprm_rerank
```

### Single-cache vs multi-cache

`cache_dir: Path` — single-cache methods (tts-agent, self-refine, socratic-self-refine, budget-forcing, rerank, standalone-integrator).
`cache_dirs: dict[str, Path]` — multi-cache methods (tts-agent-multi, tts-agent-effort).
The spec layer rejects mixing them.

### Adding a new method

Three files only:
1. `methods/<new_method>.py` — implement `async def solve(infra, problem, ...)`.
2. `methods/specs.py` — add `<NewMethod>Spec(_MethodSpec)` and add it to the `MethodSpec` Union.
3. `methods/registry.py` — add `<NewMethod>Method(MethodConfig)` and register it in `METHODS`.

eval.py needs zero changes.
```

- [ ] **Step 3: Verify CLAUDE.md still reads cleanly**

Run: `head -200 Experiment/core_code/CLAUDE.md | tail -100`
Expected: the new section content as written above.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/CLAUDE.md
git commit -m "docs(CLAUDE): describe new method-block schema

Update eval.py configuration section to reflect the discriminated-union
method block. Document the 8 method specs and their required fields.
Adding a new method now requires changes only in methods/."
```

---

## Self-Review

After completing all 10 tasks, the engineer should sanity-check:

1. **`grep -rn "cfg.method ==" Experiment/core_code/eval.py`** → no matches (all dispatch goes through `method.build_solve_fn` etc.)
2. **`grep -rn "no_cache_only" Experiment/core_code/`** → no matches (deleted everywhere)
3. **`grep -rn "_validate_method_constraints" Experiment/core_code/`** → no matches
4. **`grep -rln "^method:" Experiment/core_code/scripts/`** → 0 matches (all `method:` are now nested mappings, not strings — top-level grep won't see them as a flat key. Use `grep -lE "^method:\s*$" -A 1` if you want to confirm structure.)
5. **`conda run -n explain --no-capture-output python Experiment/core_code/eval.py --help`** → still shows `--config` only.
6. **All 4 smoke YAMLs (Task 9) printed clean banners with no AssertionError.**

If all 6 pass, the refactor is closed.
