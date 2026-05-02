# Backend-into-Method Block Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the last "dead-for-some-method" fields at EvalConfig top level by introducing `BackendConfig` (a Pydantic sub-block carrying `name`, `budget_tokens`, `effort`, `timeout`, `max_output_tokens`) and embedding it inside the 7 method specs that actually consume backend; `RerankSpec` keeps no backend (it doesn't call any LLM). Also delete the orphan `explore_timeout` field on EvalConfig.

**Architecture:** Add `BackendConfig` to `methods/specs.py`. Add `backend: BackendConfig` field to TTSAgent / TTSAgentMulti / TTSAgentEffort / SelfRefine / SocraticSelfRefine / BudgetForcing / StandaloneIntegrator specs. RerankSpec is unchanged. Drop `backend` / `budget_tokens` / `effort` / `timeout` / `max_output_tokens` / `explore_timeout` from EvalConfig top level. eval.py main() reads from `cfg.method.backend.<field>` for the 7 backend-using methods; for rerank, the InfraConfig keeps a sentinel `backend=""` (the InfraConfig field stays — the runtime carrier is allowed to hold "no backend"). Rewrite 83 YAMLs: 76 LLM-using lift fields into `method.backend:` block, 7 rerank YAMLs drop the dead fields outright.

**Tech Stack:** Python 3.11, Pydantic v2, PyYAML.

**Scope:** In: `methods/specs.py`, `eval.py`, `methods/registry.py` (RerankMethod gets a default-backend hook), 83 YAMLs under `Experiment/core_code/scripts/`, `_template.yaml`, `Experiment/core_code/CLAUDE.md`, `tests/test_eval_config.py`, `tests/test_precache_config.py` (only if PrecacheConfig also has these fields, check). Out: `precache_explores.py` and `PrecacheConfig` (their backend stays flat — precache is single-method, never had the dead-field problem). Out: `methods/<method>.py` solve function bodies.

**Working directory:** `/data3/peijia/dr-claw/Explain` (repo root). All bash commands run from here.

---

## File Structure

| File | Role | Action |
|---|---|---|
| `Experiment/core_code/methods/specs.py` | Add `BackendConfig`; add `backend: BackendConfig` field to 7 specs | Modify |
| `Experiment/core_code/eval.py` | Drop 6 top-level fields (backend, budget_tokens, effort, timeout, max_output_tokens, explore_timeout); update main() to read from cfg.method.backend.* | Modify |
| `Experiment/core_code/methods/registry.py` | RerankMethod constructs InfraConfig with sentinel backend (since spec has no backend) — handled in eval.py main() not here | No change unless needed |
| `Experiment/core_code/tools/migrate_backend_into_method.py` | One-shot script reshaping 83 YAMLs: 76 lift backend fields into method.backend:; 7 rerank YAMLs drop fields outright | Create |
| `Experiment/core_code/scripts/**/*.yaml` (83 files) | Auto-modify via script | Auto |
| `Experiment/core_code/_template.yaml` | Update annotated reference | Modify |
| `Experiment/core_code/CLAUDE.md` | Update method-block schema docs | Modify |
| `Experiment/core_code/tests/test_eval_config.py` | Update fixtures: every method block now needs `backend: { name: claude }` (except rerank); top-level no longer has those fields | Modify |

---

## Conventions

- All Python execution: `conda run -n explain --no-capture-output python ...`. Inside this plan, `sys.path.insert(0, 'Experiment/core_code')` is used for portability from repo root.
- One commit per task.
- Each task ends with a verification step under 30 seconds with concrete expected output.

---

### Task 1: Add `BackendConfig` to `methods/specs.py` + thread into 7 method specs

**Files:**
- Modify: `Experiment/core_code/methods/specs.py`

- [ ] **Step 1: Inspect current specs.py to find insertion points**

Run: `grep -nE "^class|model_config" Experiment/core_code/methods/specs.py`
Expected: Lists all class definitions including the 8 method specs.

- [ ] **Step 2: Add `BackendConfig` class right after `SamplingConfig`**

Edit `Experiment/core_code/methods/specs.py`. After the `class SamplingConfig` definition (immediately before `class _MethodSpec`), insert:

```python
class BackendConfig(BaseModel):
    """LLM backend configuration. Embedded in method specs that call an LLM
    (everything except rerank). Groups `name` with the per-call knobs that
    only make sense alongside it (budget_tokens / effort / timeout /
    max_output_tokens). Methods that do not call any LLM (rerank, which
    scores cached candidates with a local reward model) omit this field
    entirely -- their YAMLs have no backend reference at any level.
    """
    model_config = {"extra": "forbid"}
    name: Literal["codex", "claude", "vllm"]
    budget_tokens: int = 32000
    effort: Literal["low", "medium", "high", "max"] = "low"
    timeout: float = 1200.0
    max_output_tokens: int | None = None
```

- [ ] **Step 3: Add `backend: BackendConfig` field to 7 method specs**

Add `backend: BackendConfig` as the FIRST field after `name:` in each of these classes:
- `TTSAgentSpec`
- `TTSAgentMultiSpec`
- `TTSAgentEffortSpec`
- `SelfRefineSpec`
- `SocraticSelfRefineSpec`
- `BudgetForcingSpec`
- `StandaloneIntegratorSpec`

Do NOT add it to `RerankSpec`.

For example, `TTSAgentSpec` becomes:
```python
class TTSAgentSpec(_MethodSpec):
    name: Literal["tts-agent"]
    backend: BackendConfig
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
```

- [ ] **Step 4: Verify all 7 specs include backend, RerankSpec does not**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import sys; sys.path.insert(0, 'Experiment/core_code')
from methods.specs import (
    BackendConfig, TTSAgentSpec, TTSAgentMultiSpec, TTSAgentEffortSpec,
    SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
    RerankSpec, StandaloneIntegratorSpec,
)
should_have = [TTSAgentSpec, TTSAgentMultiSpec, TTSAgentEffortSpec,
               SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
               StandaloneIntegratorSpec]
for c in should_have:
    assert 'backend' in c.model_fields, f'{c.__name__} missing backend'
    assert c.model_fields['backend'].annotation is BackendConfig, f'{c.__name__} backend type wrong'
    print(f'  {c.__name__}: backend OK')
assert 'backend' not in RerankSpec.model_fields, 'RerankSpec must not have backend'
print('  RerankSpec: no backend (correct)')
" 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn"
```
Expected: 7 OK lines for the LLM-using specs + "RerankSpec: no backend (correct)".

- [ ] **Step 5: Verify discriminator-level parse with new shape**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import sys; sys.path.insert(0, 'Experiment/core_code')
from methods.specs import MethodSpec
from pydantic import TypeAdapter
adapter = TypeAdapter(MethodSpec)
ok = adapter.validate_python({
    'name': 'tts-agent',
    'backend': {'name': 'claude', 'budget_tokens': 32000, 'effort': 'low'},
    'orchestrator_model': 'm', 'explore_model': 'm', 'cache_dir': '/tmp/x',
    'no_integrate': True,
})
print('tts-agent with nested backend OK:', type(ok).__name__, '|', ok.backend.name)
rr = adapter.validate_python({
    'name': 'rerank', 'reward_model': 'rm', 'cache_dir': '/tmp/x',
})
print('rerank without backend OK:', type(rr).__name__)
try:
    adapter.validate_python({
        'name': 'rerank', 'reward_model': 'rm', 'cache_dir': '/tmp/x',
        'backend': {'name': 'claude'},
    })
    print('FAIL: rerank accepted backend')
except Exception as e:
    print('rerank correctly rejects backend:', 'extra_forbidden' in str(e))
" 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn"
```
Expected:
```
tts-agent with nested backend OK: TTSAgentSpec | claude
rerank without backend OK: RerankSpec
rerank correctly rejects backend: True
```

- [ ] **Step 6: Commit**

```bash
git add Experiment/core_code/methods/specs.py
git commit -m "$(cat <<'EOF'
feat(methods): add BackendConfig sub-block to 7 LLM-using specs

Groups backend name + per-call knobs (budget_tokens, effort, timeout,
max_output_tokens) into a single BackendConfig sub-block embedded in
the 7 method specs that actually call an LLM. RerankSpec is unchanged
-- it scores cached candidates with a local reward model and has no
backend concept.

Sets up the schema layer for the upcoming YAML migration that lifts
backend-related fields from EvalConfig top level into method.backend
block, eliminating dead-for-rerank fields.
EOF
)"
```

---

### Task 2: Write migration script `tools/migrate_backend_into_method.py`

**Files:**
- Create: `Experiment/core_code/tools/migrate_backend_into_method.py`

- [ ] **Step 1: Write the migration script**

Create `Experiment/core_code/tools/migrate_backend_into_method.py`:

```python
"""One-shot migration: lift top-level backend fields into method.backend block.

For each YAML in scripts/**/*.yaml that has a top-level `method:` block:

- If method.name == "rerank": delete top-level backend / budget_tokens /
  effort / timeout / max_output_tokens / explore_timeout outright (rerank
  doesn't call any LLM, these were always dead).
- Otherwise: lift those fields into a nested `method.backend:` block:
    method:
      name: tts-agent
      backend:
        name: claude
        budget_tokens: 32000
        effort: low
        timeout: 1200
        max_output_tokens: null
      orchestrator_model: ...

Always drops `explore_timeout` from EvalConfig YAMLs (it was an orphan
field never read by eval.py; only PrecacheConfig uses it, and precache
YAMLs aren't touched).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

BACKEND_FIELDS: list[str] = [
    "name",            # populated from top-level "backend"
    "budget_tokens",
    "effort",
    "timeout",
    "max_output_tokens",
]

# Top-level keys that the script consumes.
TOP_LEVEL_BACKEND_KEYS: set[str] = {
    "backend", "budget_tokens", "effort", "timeout", "max_output_tokens",
}
ALWAYS_DROP_TOP_LEVEL: set[str] = {"explore_timeout"}


def migrate(yaml_path: Path, dry_run: bool = False) -> bool:
    """Rewrite one YAML in-place. Returns True if file changed."""
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    if data is None or not isinstance(data, dict):
        return False
    if "method" not in data or not isinstance(data["method"], dict):
        return False  # precache YAML or pre-migration shape; skip

    method_block = data["method"]
    method_name = method_block.get("name")

    # Already migrated? (method.backend already present)
    if "backend" in method_block and isinstance(method_block.get("backend"), dict):
        return False

    is_rerank = method_name == "rerank"

    # Extract top-level backend-related values (if present)
    extracted: dict = {}
    for k in TOP_LEVEL_BACKEND_KEYS:
        if k in data:
            extracted[k] = data.pop(k)
    for k in ALWAYS_DROP_TOP_LEVEL:
        if k in data:
            data.pop(k)

    if is_rerank:
        # Drop entirely. Rerank doesn't have a backend concept.
        action_summary = f"dropped {sorted(extracted.keys())}"
    else:
        # Build method.backend sub-block.
        if "backend" not in extracted:
            print(f"  WARN: {yaml_path}: method={method_name!r} but no top-level backend; cannot lift")
            return False
        backend_block: dict = {"name": extracted.pop("backend")}
        # Carry over the other knobs that were present
        for k in ("budget_tokens", "effort", "timeout", "max_output_tokens"):
            if k in extracted:
                backend_block[k] = extracted.pop(k)
        # Inject backend as the second field of method block (right after name)
        new_method: dict = {}
        for k, v in method_block.items():
            new_method[k] = v
            if k == "name":
                new_method["backend"] = backend_block
        # If method block had no 'name' first, fall back to prepend
        if "backend" not in new_method:
            new_method = {"name": method_name, "backend": backend_block, **method_block}
        data["method"] = new_method
        action_summary = "lifted backend into method.backend"

    # Reassemble: benchmark / method first if present, then everything else.
    out: dict = {}
    if "benchmark" in data:
        out["benchmark"] = data.pop("benchmark")
    if "method" in data:
        out["method"] = data.pop("method")
    for k, v in data.items():
        out[k] = v

    if dry_run:
        print(f"--- {yaml_path} (dry-run, {action_summary}) ---")
        print(yaml.safe_dump(out, sort_keys=False, default_flow_style=False))
        return True

    with yaml_path.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
    print(f"  {yaml_path}: {action_summary}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("Experiment/core_code/scripts"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
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

- [ ] **Step 2: Dry-run on one tts-agent YAML to confirm lift works**

Run:
```bash
conda run -n explain --no-capture-output python Experiment/core_code/tools/migrate_backend_into_method.py \
  --root Experiment/core_code/scripts/aime2025/sonnet \
  --dry-run --limit 1 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn"
```
Expected: dry-run output shows method block with nested `backend:` sub-block and the action `lifted backend into method.backend`.

- [ ] **Step 3: Dry-run on a rerank YAML to confirm drop works**

Run:
```bash
conda run -n explain --no-capture-output python Experiment/core_code/tools/migrate_backend_into_method.py \
  --root Experiment/core_code/scripts/babyvision/sonnet --dry-run 2>&1 \
  | grep -A 100 "babyvision_sonnet_visualprm_rerank" | head -25
```
Expected: visualprm rerank YAML dry-run shows NO backend reference at any level (top dropped, method has no backend); action says `dropped ['backend', 'budget_tokens', 'effort', ...]`.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/tools/migrate_backend_into_method.py
git commit -m "$(cat <<'EOF'
tools: one-shot script lifting backend into method.backend block

Per method.name: lift backend / budget_tokens / effort / timeout /
max_output_tokens from EvalConfig top level into method.backend
sub-block (76 LLM-using YAMLs); drop them entirely for rerank YAMLs
(7 files, where they were always dead). Also drops the orphan
explore_timeout field (only PrecacheConfig used it; eval YAMLs
that had it were never reading it).
EOF
)"
```

---

### Task 3: Run migration on all 83 YAMLs

**Files:**
- Modify: 83 YAMLs under `Experiment/core_code/scripts/`

- [ ] **Step 1: Run for real**

Run:
```bash
conda run -n explain --no-capture-output python Experiment/core_code/tools/migrate_backend_into_method.py \
  --root Experiment/core_code/scripts 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn" | tail -50
```
Expected: prints "lifted backend into method.backend" for 76 files, "dropped [...]" for 7 rerank files, ends with `Migrated 83/<N> files.`

- [ ] **Step 2: Verify all 83 YAMLs parse through new MethodSpec**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import sys; sys.path.insert(0, 'Experiment/core_code')
import yaml
from pathlib import Path
from pydantic import TypeAdapter
from methods.specs import MethodSpec

adapter = TypeAdapter(MethodSpec)
ok, fails = 0, []
for p in sorted(Path('Experiment/core_code/scripts').rglob('*.yaml')):
    data = yaml.safe_load(open(p))
    if not isinstance(data, dict) or 'method' not in data:
        continue
    try:
        adapter.validate_python(data['method'])
        ok += 1
    except Exception as e:
        fails.append((str(p), str(e)[:200]))
print(f'OK: {ok}, Failed: {len(fails)}')
for p, msg in fails[:5]:
    print(f'  {p}: {msg}')
assert not fails
" 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn"
```
Expected: `OK: 83, Failed: 0`.

- [ ] **Step 3: Spot-check 4 representative shapes**

Run:
```bash
for f in \
  Experiment/core_code/scripts/aime2025/sonnet/aime2025_sonnet_delegated.yaml \
  Experiment/core_code/scripts/hle/sonnet/hle_sonnet_self_refine.yaml \
  Experiment/core_code/scripts/hle/multi_model/hle_multi_delegated.yaml \
  Experiment/core_code/scripts/babyvision/sonnet/babyvision_sonnet_visualprm_rerank.yaml \
; do echo "=== $f ==="; cat "$f"; echo ""; done
```
Expected: tts-agent / self-refine / multi YAMLs each have `method.backend:` sub-block with `name: claude`. Rerank YAML has no `backend` at any level.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/scripts/
git commit -m "$(cat <<'EOF'
refactor(scripts): lift backend into method.backend block (83 YAMLs)

Run by tools/migrate_backend_into_method.py. 76 LLM-using YAMLs gain
a method.backend: sub-block; 7 rerank YAMLs drop the dead fields
entirely. orphan explore_timeout dropped from any YAML that had it.
All 83 verified to parse through MethodSpec.
EOF
)"
```

---

### Task 4: Modify `EvalConfig` — drop the 6 fields, no other change

**Files:**
- Modify: `Experiment/core_code/eval.py` (EvalConfig class section)

- [ ] **Step 1: Read EvalConfig current state**

Run: `sed -n '20,55p' Experiment/core_code/eval.py`
Expected: shows EvalConfig class with `backend`, `budget_tokens`, `effort`, `timeout`, `max_output_tokens`, `explore_timeout` fields.

- [ ] **Step 2: Delete the 6 fields**

Edit `Experiment/core_code/eval.py`. In the `EvalConfig` class, delete these lines:

```python
    backend: Literal["codex", "claude", "vllm"]
```
```python
    budget_tokens: int = 32000
    effort: Literal["low", "medium", "high", "max"] = "low"
    explore_timeout: float = 1200.0
    max_output_tokens: int | None = None
    timeout: float = 1200.0
```

The remaining EvalConfig fields are:
```python
class EvalConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": False}
    benchmark: BenchmarkSpec
    method: MethodSpec
    num: int | None = None
    skip: int = 0
    seed: int = 42
    shuffle: bool = False
    num_workers: int = 1
    verbose: bool = False
    resume: str | None = None
    log_dir: str = "logs"
```

If `Literal` is no longer used elsewhere in eval.py after this edit, leave the import alone — it's also imported in benchmarks/specs.py and elsewhere; only delete it if a strict `grep -n "Literal" eval.py` returns zero matches.

- [ ] **Step 3: Verify eval.py parses**

Run: `conda run -n explain --no-capture-output python -c "import ast; ast.parse(open('Experiment/core_code/eval.py').read()); print('parse ok')" 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn"`
Expected: `parse ok`.

- [ ] **Step 4: Verify all 83 YAMLs parse through new EvalConfig**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import sys; sys.path.insert(0, 'Experiment/core_code')
import yaml
from pathlib import Path
from eval import EvalConfig
ok, fails = 0, []
for p in sorted(Path('Experiment/core_code/scripts').rglob('*.yaml')):
    data = yaml.safe_load(open(p))
    if not isinstance(data, dict) or 'method' not in data:
        continue
    try:
        EvalConfig.model_validate(data)
        ok += 1
    except Exception as e:
        fails.append((str(p), str(e)[:300]))
print(f'OK: {ok}, Failed: {len(fails)}')
for p, msg in fails[:3]:
    print(f'  {p}: {msg}')
assert not fails
" 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn"
```
Expected: `OK: 83, Failed: 0`.

- [ ] **Step 5: Commit**

```bash
git add Experiment/core_code/eval.py
git commit -m "$(cat <<'EOF'
refactor(eval): drop 6 backend-level fields from EvalConfig top level

Backend-level config (backend, budget_tokens, effort, timeout,
max_output_tokens) now lives inside method.backend block per the
'no dead fields at top level' principle. explore_timeout was
already orphan (never read in eval.py), dropped outright.

Note: eval.py main() still reads cfg.backend / cfg.budget_tokens
etc. and will break at runtime. Task 5 rewires it.
EOF
)"
```

---

### Task 5: Wire `eval.py` main() to read from `cfg.method.backend.*`

**Files:**
- Modify: `Experiment/core_code/eval.py` (main() InfraConfig construction + evaluate() call)

- [ ] **Step 1: Inventory references to the 6 deleted fields**

Run: `grep -nE "cfg\.(backend|budget_tokens|effort|timeout|max_output_tokens|explore_timeout)\b" Experiment/core_code/eval.py`
Expected: ~6-8 matches, all in main() InfraConfig construction or pass-through to evaluate().

- [ ] **Step 2: Read the relevant block**

Run: `sed -n '700,770p' Experiment/core_code/eval.py`
Expected: shows the InfraConfig construction block.

- [ ] **Step 3: Update InfraConfig construction**

In `eval.py` `async_main()`, find the existing `infra = InfraConfig(...)` block and replace the field reads. The new pattern: rerank has no `cfg.method.backend`, so we provide a sentinel.

Replace the InfraConfig construction with:

```python
    # Backend block lives inside method spec for the 7 LLM-using methods;
    # rerank has no backend (it scores cached candidates with a local reward
    # model, never calls an LLM API). For rerank, populate InfraConfig with
    # an empty-string backend; create_solve_context still imports
    # backends.<name> but rerank.solve never calls the resulting caller.
    backend_block = getattr(cfg.method, "backend", None)
    backend_name = backend_block.name if backend_block else ""
    backend_budget_tokens = backend_block.budget_tokens if backend_block else 32000
    backend_effort = backend_block.effort if backend_block else "low"
    backend_timeout = backend_block.timeout if backend_block else 1200.0
    backend_max_output_tokens = backend_block.max_output_tokens if backend_block else None

    infra = InfraConfig(
        backend=backend_name,
        max_iterations=num_explores,
        cache_dir=cache_dir,
        cache_only=method.cache_only,
        budget_tokens=backend_budget_tokens,
        effort=backend_effort,
        timeout=backend_timeout,
        benchmark=benchmark,
        quiet=not cfg.verbose,
        logger=None,
        enable_integrate=not getattr(cfg.method, "no_integrate", False),
        max_output_tokens=backend_max_output_tokens,
    )
```

If `make_sub_model_caller` would crash on `backend=""` for rerank, handle that in `methods/base.py:make_sub_model_caller` — but rerank never invokes `ctx.call_sub_model`, so the import line never runs in rerank path. Verify by Step 4.

- [ ] **Step 4: Update the `evaluate(...)` call site to drop deleted top-level field reads**

Find the existing `await evaluate(...)` call in async_main(). Replace any remaining `cfg.backend` / `cfg.budget_tokens` etc. reads in surrounding code with the local `backend_*` variables defined in Step 3, OR remove them if they were only being passed to InfraConfig (already updated).

Specifically, where the dispatcher computes a banner-line `backend` for logger config (search for `"backend": ...` in the run config dict around line ~316 inside `evaluate`'s logger setup), make sure it reads `infra.backend` (which we set in Step 3) rather than `cfg.backend`.

- [ ] **Step 5: Verify eval.py parses + --help still works**

Run:
```bash
conda run -n explain --no-capture-output python -c "import ast; ast.parse(open('Experiment/core_code/eval.py').read()); print('parse ok')" 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn"
conda run -n explain --no-capture-output python Experiment/core_code/eval.py --help 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn"
```
Expected: `parse ok` and the standard --help output with only `--config`.

- [ ] **Step 6: Smoke-test 4 shapes with full dry-load**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import sys; sys.path.insert(0, 'Experiment/core_code')
import yaml
from pathlib import Path
from eval import EvalConfig
from methods import get_method
from methods.base import InfraConfig
from benchmarks import get_benchmark

shapes = [
    ('Experiment/core_code/scripts/aime2025/sonnet/aime2025_sonnet_delegated.yaml', 'tts-agent', 'claude'),
    ('Experiment/core_code/scripts/hle/sonnet/hle_sonnet_self_refine.yaml', 'self-refine', 'claude'),
    ('Experiment/core_code/scripts/hle/multi_model/hle_multi_delegated.yaml', 'tts-agent-multi', 'claude'),
    ('Experiment/core_code/scripts/babyvision/sonnet/babyvision_sonnet_visualprm_rerank.yaml', 'rerank', ''),
]
for path, expected_method, expected_backend in shapes:
    print(f'=== {expected_method} ===')
    data = yaml.safe_load(open(path))
    cfg = EvalConfig.model_validate(data)
    assert cfg.method.name == expected_method
    backend_block = getattr(cfg.method, 'backend', None)
    backend_name = backend_block.name if backend_block else ''
    assert backend_name == expected_backend, f'{expected_method}: backend={backend_name!r} expected {expected_backend!r}'
    print(f'  method.name={cfg.method.name}  backend.name={backend_name!r}')
print('ALL 4 SHAPES PASS')
" 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn"
```
Expected:
```
=== tts-agent ===
  method.name=tts-agent  backend.name='claude'
=== self-refine ===
  method.name=self-refine  backend.name='claude'
=== tts-agent-multi ===
  method.name=tts-agent-multi  backend.name='claude'
=== rerank ===
  method.name=rerank  backend.name=''
ALL 4 SHAPES PASS
```

- [ ] **Step 7: Commit**

```bash
git add Experiment/core_code/eval.py
git commit -m "$(cat <<'EOF'
refactor(eval): read backend config from cfg.method.backend block

main() now sources backend_name / budget_tokens / effort / timeout /
max_output_tokens from the method's BackendConfig sub-block. Rerank
falls through to sentinel backend='' since it has no backend; rerank
never invokes ctx.call_sub_model, so the imported backend module is
unused at runtime.
EOF
)"
```

---

### Task 6: Update `tests/test_eval_config.py` for new method-block shape

**Files:**
- Modify: `Experiment/core_code/tests/test_eval_config.py`

- [ ] **Step 1: Update `_minimal_kwargs` to include backend block**

Edit `Experiment/core_code/tests/test_eval_config.py`. Find `_minimal_kwargs` and update its default method block to include the backend sub-block (since self-refine now requires it):

```python
def _minimal_kwargs(method_block=None, **overrides):
    base = {
        "benchmark": {"name": "hle"},
        "method": method_block or {
            "name": "self-refine",
            "backend": {"name": "claude"},
            "explore_model": "claude-sonnet-4-6",
            "cache_dir": "/cache/x",
        },
    }
    base.update(overrides)
    return base
```

Remove the top-level `"backend": "claude"` since it no longer lives at top level.

- [ ] **Step 2: Update each method block in tests that previously had no backend**

Search for every literal `"name": "tts-agent"` / `"name": "tts-agent-multi"` / `"name": "self-refine"` / `"name": "standalone-integrator"` etc. in the test file and add `"backend": {"name": "claude"}` to each method dict.

DO NOT add backend to rerank method blocks.

For example:
```python
def test_tts_agent_happy_path_no_integrate():
    cfg = EvalConfig(**_minimal_kwargs(method={
        "name": "tts-agent",
        "backend": {"name": "claude"},
        "orchestrator_model": "m",
        "explore_model": "m",
        "cache_dir": "/cache/x",
        "no_integrate": True,
    }))
    ...
```

Do this for: `test_tts_agent_happy_path_no_integrate`, `test_tts_agent_requires_integrate_model_when_no_integrate_false`, `test_tts_agent_multi_happy_path`, `test_tts_agent_multi_missing_required_fields`, `test_self_refine_rejects_orchestrator_model`, `test_standalone_integrator_requires_integrate_model`, `test_load_config_yaml`, `test_load_config_multi_method_yaml`, `test_hle_filters_validate`, `test_hle_filters_reject_unknown_field`, `test_gpqa_filters_validate`, `test_filters_empty_validates_for_all_benchmarks`.

- [ ] **Step 3: Update `test_top_level_has_no_method_specific_fields`**

The dead-field set has grown. Update the test:

```python
def test_top_level_has_no_method_specific_fields():
    """Method-related fields no longer live at top level. Setting them
    there must fail under extra='forbid'."""
    for dead in ("orchestrator_model", "explore_model", "integrate_model",
                 "reward_model", "cache_dir", "cache_dirs", "model_budgets",
                 "no_integrate", "num_explores", "num_rollouts",
                 "no_cache_only", "backend", "budget_tokens", "effort",
                 "timeout", "max_output_tokens", "explore_timeout"):
        with pytest.raises(ValidationError, match=dead):
            EvalConfig(**_minimal_kwargs(**{dead: "x" if "model" in dead or dead in ("backend",) else 1}))
```

- [ ] **Step 4: Add a new test asserting rerank rejects backend field**

Add this test at the bottom of the per-method section:

```python
def test_rerank_rejects_backend_field():
    """Rerank doesn't have a backend; setting one in the method block must fail."""
    with pytest.raises(ValidationError, match="backend|extra"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "rerank",
            "reward_model": "rm",
            "cache_dir": "/cache/x",
            "backend": {"name": "claude"},
        }))


def test_backend_config_extra_forbidden():
    """BackendConfig itself has extra:forbid."""
    with pytest.raises(ValidationError, match="extra"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "self-refine",
            "backend": {"name": "claude", "typoed": True},
            "explore_model": "m",
            "cache_dir": "/cache/x",
        }))
```

- [ ] **Step 5: Update the YAML loader tests**

For the YAML tests (`test_load_config_yaml`, `test_load_config_multi_method_yaml`, the `test_hle_filters_*` and `test_gpqa_filters_validate`), add `backend:` block inside the `method:` YAML body. Example for `test_load_config_yaml`:

```python
def test_load_config_yaml(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark:
          name: hle
        method:
          name: self-refine
          backend:
            name: claude
          explore_model: claude-sonnet-4-6
          cache_dir: /cache/single
        num: 50
    """)
    cfg = load_config(config_path=yml, schema=EvalConfig)
    assert cfg.method.cache_dir == Path("/cache/single")
    assert cfg.method.backend.name == "claude"
    assert cfg.num == 50
```

Apply the same pattern (add `backend: { name: claude }` inside method block, remove top-level `backend: claude`) to the other YAML loader tests.

- [ ] **Step 6: Run the test file**

Run: `conda run -n explain pytest Experiment/core_code/tests/test_eval_config.py -v 2>&1 | tail -30`
Expected: all tests pass.

- [ ] **Step 7: Run the full test suite to confirm no regression elsewhere**

Run: `conda run -n explain pytest Experiment/core_code/tests/ 2>&1 | tail -5`
Expected: `XX passed in N.NNs` where XX is at least 63 (the prior baseline) plus any new tests added in this task.

- [ ] **Step 8: Commit**

```bash
git add Experiment/core_code/tests/test_eval_config.py
git commit -m "$(cat <<'EOF'
test(eval_config): update for backend-into-method-block schema

Every non-rerank method block now requires nested backend: { name: ... }
sub-block. Adds two new tests: rerank-rejects-backend-field and
backend_config_extra_forbidden. Updates the dead-top-level set to
include the 6 newly-moved fields (backend, budget_tokens, effort,
timeout, max_output_tokens, explore_timeout).
EOF
)"
```

---

### Task 7: Update `_template.yaml`

**Files:**
- Modify: `Experiment/core_code/_template.yaml`

- [ ] **Step 1: Read current template**

Run: `cat Experiment/core_code/_template.yaml`

- [ ] **Step 2: Rewrite to nest backend inside method**

Replace `Experiment/core_code/_template.yaml` content with:

```yaml
# _template.yaml
# Annotated template for eval.py. Every field shown here is OPTIONAL unless flagged required.

# === Benchmark block (discriminated by name) ===
# See benchmarks/specs.py.
benchmark:
  name: hle                 # required: hle / gpqa / lcb / babyvision / rbenchv / aime2025 / aime2026
  subset: gold              # HLE-only filter; remove for other benchmarks
  text_only: true           # HLE-only filter

# === Method block (discriminated by name) ===
# See methods/specs.py. Method-specific fields live inside this block.
# Methods that call an LLM include a backend: sub-block grouping the
# backend name with its per-call knobs. RerankSpec OMITS backend entirely
# (it scores cached candidates with a local reward model, no LLM call).
method:
  name: tts-agent           # one of: tts-agent / tts-agent-multi / tts-agent-effort /
                            # self-refine / socratic-self-refine / budget-forcing /
                            # rerank / standalone-integrator

  # Backend block (REMOVE this block entirely if method.name == rerank):
  backend:
    name: claude            # one of: codex, claude, vllm
    budget_tokens: 32000
    effort: low             # low / medium / high / max
    timeout: 1200
    max_output_tokens: null

  # Model fields (per method) -- see methods/specs.py for which spec accepts which:
  orchestrator_model: claude-sonnet-4-6
  explore_model: claude-sonnet-4-6
  integrate_model: claude-sonnet-4-6
  # reward_model: ...        # required only for rerank
  # exploration_effort: low  # tts-agent-multi only

  # Cache:
  cache_dir: ../analysis/cache/hle/sonnet/gold
  # cache_dirs:              # tts-agent-multi / tts-agent-effort
  #   haiku: /cache/haiku

  # Method-specific budgets:
  # model_budgets: { haiku: 8 }      # tts-agent-multi
  # effort_budgets: { low: 8 }       # tts-agent-effort

  # Per-method behavior knobs:
  no_integrate: false        # tts-agent only
  num_explores: 8
  num_rollouts: 1            # tts-agent only; > 1 requires backend.name=vllm
  # sampling: ...            # tts-agent only; vLLM SamplingConfig

# === Dataset ===
num: null
skip: 0
seed: 42
shuffle: false

# === Truly generic top-level (apply to every method) ===
num_workers: 1
verbose: false
resume: null
log_dir: logs
```

- [ ] **Step 3: Verify template parses**

Run:
```bash
conda run -n explain --no-capture-output python -c "
import sys; sys.path.insert(0, 'Experiment/core_code')
import yaml
from eval import EvalConfig
data = yaml.safe_load(open('Experiment/core_code/_template.yaml'))
cfg = EvalConfig.model_validate(data)
print(f'OK: method={cfg.method.name} backend={cfg.method.backend.name}')
" 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn"
```
Expected: `OK: method=tts-agent backend=claude`.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/_template.yaml
git commit -m "docs(_template): nest backend inside method block

Reference template now reflects: backend block nested under method
(for LLM-using methods); rerank YAMLs have no backend at any level."
```

---

### Task 8: Update `Experiment/core_code/CLAUDE.md` docs

**Files:**
- Modify: `Experiment/core_code/CLAUDE.md`

- [ ] **Step 1: Find the "method block" section**

Run: `grep -n "### \`method:\` block\|### Top-level generic\|### Single-cache vs multi-cache\|### Example: tts-agent\|### Example: rerank" Experiment/core_code/CLAUDE.md`

- [ ] **Step 2: Update the per-method required-fields table**

Find the table starting with `| Method | Required fields inside \`method:\` block |`. Replace its body with:

```markdown
| Method | Required fields inside `method:` block |
|---|---|
| `tts-agent` | `backend`, `orchestrator_model`, `explore_model`, `cache_dir`; `integrate_model` unless `no_integrate=true` |
| `tts-agent-multi` | `backend`, `orchestrator_model`, `cache_dirs`, `model_budgets` |
| `tts-agent-effort` | `backend`, `orchestrator_model`, `explore_model`, `cache_dirs`, `effort_budgets` |
| `self-refine` | `backend`, `explore_model`, `cache_dir` |
| `socratic-self-refine` | `backend`, `explore_model`, `cache_dir` |
| `budget-forcing` | `backend`, `explore_model`, `cache_dir` |
| `rerank` | `reward_model`, `cache_dir` (no backend — uses local reward model) |
| `standalone-integrator` | `backend`, `integrate_model`, `cache_dir` |

Where `backend` is a sub-block:
```yaml
backend:
  name: claude            # one of: codex, claude, vllm
  budget_tokens: 32000    # default 32000
  effort: low             # default low
  timeout: 1200           # default 1200
  max_output_tokens: null # default null
```
```

- [ ] **Step 3: Update the top-level generic field list**

Find the section `### Top-level generic fields`. Update its body to:

```markdown
### Top-level generic fields (do not vary per method)

`num`, `skip`, `seed`, `shuffle`, `num_workers`, `log_dir`, `resume`, `verbose`.

Backend-related fields (`backend`, `budget_tokens`, `effort`, `timeout`,
`max_output_tokens`) live inside `method.backend` for the 7 LLM-using methods,
and don't exist at all in rerank YAMLs (rerank uses a local reward model).
```

- [ ] **Step 4: Update Example: tts-agent**

Find `### Example: tts-agent (paper main config)` and replace the YAML body with:

```yaml
benchmark:
  name: hle
  subset: gold
method:
  name: tts-agent
  backend:
    name: claude
  orchestrator_model: claude-sonnet-4-6
  explore_model: claude-sonnet-4-6
  cache_dir: /cache/hle/sonnet/gold
  no_integrate: true
  num_explores: 8
num: 100
seed: 42
log_dir: /run/hle/sonnet
```

- [ ] **Step 5: Update Example: rerank**

Find `### Example: rerank` and replace its YAML body with:

```yaml
benchmark:
  name: babyvision
method:
  name: rerank
  reward_model: OpenGVLab/VisualPRM-8B
  cache_dir: /cache/babyvision/sonnet
num_workers: 16
log_dir: /run/babyvision/sonnet_visualprm_rerank
```

(Note: no `backend:` field anywhere — illustrating the principle.)

- [ ] **Step 6: Commit**

```bash
git add Experiment/core_code/CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(CLAUDE): document backend-into-method-block schema

Backend (and its 4 per-call knobs) live inside method.backend sub-block
for the 7 LLM-using methods. Rerank YAMLs have no backend at any level.
Top-level generic fields shrunk from 14 to 8.
EOF
)"
```

---

### Task 9: Final self-review

**Files:** none modified

- [ ] **Step 1: Verify zero references to deleted EvalConfig fields**

Run:
```bash
grep -rn "cfg\.\(backend\|budget_tokens\|effort\|timeout\|max_output_tokens\|explore_timeout\)\b" Experiment/core_code/eval.py || echo "no matches"
```
Expected: `no matches`.

- [ ] **Step 2: Verify zero stale top-level backend in YAMLs**

Run:
```bash
grep -rlE "^backend:" Experiment/core_code/scripts/ 2>/dev/null | grep -v precache || echo "no eval YAML has top-level backend"
```
Expected: `no eval YAML has top-level backend`. (Precache YAMLs are excluded; their `PrecacheConfig` schema is unchanged.)

- [ ] **Step 3: Verify zero rerank YAML has any backend reference**

Run:
```bash
for f in $(find Experiment/core_code/scripts -name "*.yaml" | xargs grep -lE "^  name: rerank$"); do
  if grep -q "backend" "$f"; then
    echo "STALE: $f still mentions backend"
  fi
done
echo "done"
```
Expected: only `done` (no STALE lines).

- [ ] **Step 4: Run full test suite**

Run:
```bash
conda run -n explain pytest Experiment/core_code/tests/ 2>&1 | tail -5
```
Expected: `XX passed`.

- [ ] **Step 5: Final smoke test on 4 shapes (no commit, just verification)**

Run the same dry-load script as Task 5 Step 6. Expected output unchanged.

If all 5 self-review steps pass, the refactor is closed.

---

## Self-Review (writing-plans skill spec coverage check)

- **Spec coverage:** `BackendConfig` (Task 1), 7 method specs gain `backend` (Task 1), RerankSpec unchanged (Task 1 verified Step 4), 83 YAMLs migrated (Tasks 2-3), EvalConfig drops 6 fields (Task 4), main() reads from cfg.method.backend (Task 5), tests updated (Task 6), template updated (Task 7), docs updated (Task 8), final review (Task 9). Every spec requirement has a task.
- **Placeholder scan:** No "TBD" / "TODO" / "implement later" / "Add appropriate error handling" / "Similar to Task N". Every code-changing step shows complete code.
- **Type consistency:** `BackendConfig` declared in Task 1 Step 2, used by name in Tasks 5, 6, 7, 8. Field names (`name`, `budget_tokens`, `effort`, `timeout`, `max_output_tokens`) consistent across all tasks.
