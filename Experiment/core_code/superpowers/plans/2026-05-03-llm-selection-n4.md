# LLM Selection (N=4) on HLE and LCB — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the standalone-integrator (LLM Selection) baseline with N=4 candidates on HLE and LCB, producing two new Pareto-frontier rows for the existing N=8 row in main.tex `tab:main-results` panels (a) and (b).

**Architecture:** Add a `num_explores: int = 8` field to `StandaloneIntegratorSpec` (default reproduces the existing N=8 behavior, no breaking change), thread it through `methods/registry.py` into the `solve()` function, and slice `candidates = candidates[:num_explores]` after loading from cache. Two new yamls + two launchers consume the existing Sonnet explore caches at `analysis/cache/{hle,lcb}/sonnet/`. No new explore generation. Judge cache (HLE only, 800 entries) hits for free; only 100 integrator-result judges are new spend.

**Tech Stack:** Python 3.11, Pydantic 2 (specs), pytest, conda env `explain`. Anthropic Claude API via project's `backends/claude.py` shim.

**Companion files:**
- Gate-discipline TODO files (operator-facing, complementary to this plan):
  - `/data3/peijia/dr-claw/Explain/Experiment/core_code/todo_llm_selection_n4_hle.md`
  - `/data3/peijia/dr-claw/Explain/Experiment/core_code/todo_llm_selection_n4_lcb.md`

**Status snapshot at plan-write time (2026-05-03):**
- Task 1 Step 1-2 already complete: `methods/specs.py` and `methods/registry.py` edited and verified — Pydantic dry-load confirms `num_explores=8` default and `num_explores=4` override both parse. Remaining work picks up at Task 1 Step 3.

**Cost estimate:** ~$8 HLE + ~$10 LCB ≈ $18 fresh API spend. Cache replay accounting will report ~$227 (HLE) / ~$157 (LCB) paper-style total per run.

---

## File Structure

| File | Status | Purpose |
|---|---|---|
| `methods/specs.py` | DONE | `StandaloneIntegratorSpec` gains `num_explores: int = 8` field with explanatory comment |
| `methods/registry.py` | DONE | `StandaloneIntegratorMethod.build_solve_fn` threads `num_explores` into the partialed solve fn |
| `methods/standalone_integrator.py` | TODO | `solve()` accepts `num_explores`, truncates `candidates[:num_explores]`, recomputes `explore_cost_total` from kept entries |
| `tests/test_standalone_integrator_num_explores.py` | TODO (new) | Unit test asserting spec + registry thread `num_explores` correctly; integration test with fixture cache asserting truncation |
| `scripts/hle/sonnet/hle_sonnet_standalone_integrator_n4.yaml` | TODO (new) | HLE N=4 config, same cache_dir as N=8, distinct log_dir |
| `scripts/hle/sonnet/run_standalone_integrator_n4.sh` | TODO (new) | HLE launcher |
| `scripts/lcb/sonnet/lcb_sonnet_standalone_integrator_n4.yaml` | TODO (new) | LCB N=4 config |
| `scripts/lcb/sonnet/run_standalone_integrator_n4.sh` | TODO (new) | LCB launcher |

---

## Task 1: Truncate candidates in `standalone_integrator.solve()`

**Files:**
- Modify: `/data3/peijia/dr-claw/Explain/Experiment/core_code/methods/standalone_integrator.py`
- Test: `/data3/peijia/dr-claw/Explain/Experiment/core_code/tests/test_standalone_integrator_num_explores.py` (create)

- [x] **Step 1: Spec field added** (DONE 2026-05-03)

`methods/specs.py:147-160` now has:
```python
class StandaloneIntegratorSpec(_MethodSpec):
    name: Literal["standalone-integrator"]
    backend: BackendConfig
    integrate_model: str
    cache_dir: Path
    # Default 8 reproduces the paper's "LLM Selection (N=8)" baseline (main.tex
    # tab:main-results). Override e.g. `num_explores: 4` in yaml to take only
    # the first N cached candidates per question -- enables cost-vs-accuracy
    # Pareto sweeps on the same fixed cache without regenerating explores.
    # Coupling: integrator response cache key is `integrate_standalone_{N}` so
    # different N values do NOT collide on the cached integrator output.
    num_explores: int = 8
```

- [x] **Step 2: Registry threads field** (DONE 2026-05-03)

`methods/registry.py:193-199`:
```python
def build_solve_fn(self, spec: StandaloneIntegratorSpec):
    from methods.standalone_integrator import solve
    return functools.partial(
        solve,
        integrate_model=spec.integrate_model,
        num_explores=spec.num_explores,
    )
```

- [ ] **Step 3: Write the failing test**

Create `/data3/peijia/dr-claw/Explain/Experiment/core_code/tests/test_standalone_integrator_num_explores.py`:

```python
"""Verify num_explores plumbing for standalone-integrator.

- spec accepts num_explores with default 8
- spec accepts override (e.g. 4)
- registry's partialed solve carries num_explores
- solve() truncates candidates to first N when num_explores < cache size
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter

from methods.specs import MethodSpec, StandaloneIntegratorSpec
from methods.registry import METHODS


def _make_spec(num_explores: int | None = None) -> dict:
    spec = {
        "name": "standalone-integrator",
        "backend": {"name": "claude"},
        "integrate_model": "claude-sonnet-4-6",
        "cache_dir": "/tmp/fake-cache",
    }
    if num_explores is not None:
        spec["num_explores"] = num_explores
    return spec


def test_spec_default_is_eight():
    parsed = TypeAdapter(MethodSpec).validate_python(_make_spec())
    assert isinstance(parsed, StandaloneIntegratorSpec)
    assert parsed.num_explores == 8


def test_spec_accepts_override():
    parsed = TypeAdapter(MethodSpec).validate_python(_make_spec(num_explores=4))
    assert parsed.num_explores == 4


def test_spec_rejects_unknown_field():
    bad = _make_spec()
    bad["explore_model"] = "claude-sonnet-4-6"  # not a standalone-integrator field
    with pytest.raises(Exception):
        TypeAdapter(MethodSpec).validate_python(bad)


def test_registry_partial_carries_num_explores():
    parsed = TypeAdapter(MethodSpec).validate_python(_make_spec(num_explores=4))
    method_cls = METHODS["standalone-integrator"]
    partial = method_cls().build_solve_fn(parsed)
    assert partial.keywords["num_explores"] == 4
    assert partial.keywords["integrate_model"] == "claude-sonnet-4-6"


def test_load_and_truncate_with_fixture_cache(tmp_path: Path):
    """Build a fake 6-explore cache; load_cached_candidates returns 6;
    after slicing [:4] the kept cost equals the sum of first 4 cost_usd."""
    from methods.base import load_cached_candidates

    class _FakeBenchmark:
        @staticmethod
        def get_answer_from_explore(d):
            return d["answer"]

    qid = "qfake1"
    qdir = tmp_path / qid
    costs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    for i, c in enumerate(costs, 1):
        ed = qdir / f"explore_{i}"
        ed.mkdir(parents=True)
        (ed / "result.json").write_text(json.dumps({
            "answer": f"a{i}", "approach": "", "reasoning": "",
            "confidence": 0.5, "cost_usd": c, "timed_out": False,
        }))

    candidates, total = load_cached_candidates(tmp_path, qid, _FakeBenchmark())
    assert len(candidates) == 6
    assert total == pytest.approx(sum(costs))

    # truncation behavior expected of solve() after the new edit:
    n = 4
    kept = candidates[:n]
    kept_cost = sum(c.cost_usd for c in kept)
    assert len(kept) == 4
    assert kept_cost == pytest.approx(sum(costs[:4]))
```

- [ ] **Step 4: Run test to verify it fails on the truncation assertion path**

Run from `/data3/peijia/dr-claw/Explain/Experiment/core_code`:
```bash
conda run -n explain --no-capture-output pytest tests/test_standalone_integrator_num_explores.py -v
```

Expected:
- `test_spec_default_is_eight` PASS (specs.py edit done)
- `test_spec_accepts_override` PASS
- `test_spec_rejects_unknown_field` PASS
- `test_registry_partial_carries_num_explores` PASS (registry.py edit done)
- `test_load_and_truncate_with_fixture_cache` PASS (this test only exercises base helpers, not the patched solve())

The plumbing is sound; Step 5 below adds the *runtime* truncation inside solve().

- [ ] **Step 5: Patch `standalone_integrator.solve()` to consume `num_explores`**

In `/data3/peijia/dr-claw/Explain/Experiment/core_code/methods/standalone_integrator.py`, change the signature and body:

```python
async def solve(
    infra: InfraConfig,
    problem: str,
    image_data_url: str | None = None,
    question_id: str | None = None,
    integrate_model: str = "claude-sonnet-4-6",
    num_explores: int = 8,
    **_extra,
) -> SolveResult:
    """Synthesize from up to `num_explores` pre-cached candidates in one LLM call."""
    assert infra.cache_dir is not None, "cache_dir is required for standalone-integrator"
    assert question_id is not None, "question_id is required for standalone-integrator"
    assert num_explores >= 1, f"num_explores must be >= 1, got {num_explores}"
    ...
```

Below the existing `load_cached_candidates(...)` call (currently line 45-47), add the truncation:

```python
candidates, _ = load_cached_candidates(
    infra.cache_dir, question_id, ctx.benchmark,
)
# Truncate to first N cached candidates; recompute cost from kept entries only
# so paper-style $/q reporting reflects the candidates actually integrated.
# Default num_explores=8 keeps the original "consume all" behavior.
candidates = candidates[:num_explores]
explore_cost_total = sum(c.cost_usd for c in candidates)
```

Leave `cache_key=f"integrate_standalone_{len(candidates)}"` unchanged — the existing format already includes the count, so N=4 → `integrate_standalone_4` and N=8 → `integrate_standalone_8` do not collide on the cached integrator output.

- [ ] **Step 6: Re-run the unit tests**

```bash
conda run -n explain --no-capture-output pytest tests/test_standalone_integrator_num_explores.py -v
```

Expected: 5/5 PASS.

- [ ] **Step 7: Smoke-load existing N=8 yamls (no behavior change for default)**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output python -c "
import yaml
from pydantic import TypeAdapter
from methods.specs import MethodSpec
adapter = TypeAdapter(MethodSpec)
for p in ['scripts/hle/sonnet/hle_sonnet_standalone_integrator.yaml',
          'scripts/lcb/sonnet/lcb_sonnet_standalone_integrator.yaml']:
    spec = yaml.safe_load(open(p))['method']
    parsed = adapter.validate_python(spec)
    assert parsed.num_explores == 8, f'{p}: default broke'
    print(f'OK: {p} -> num_explores={parsed.num_explores}')
"
```

Expected: 2 lines of `OK: ... -> num_explores=8`.

- [ ] **Step 8: Commit Task 1**

```bash
cd /data3/peijia/dr-claw/Explain
git add Experiment/core_code/methods/specs.py \
        Experiment/core_code/methods/registry.py \
        Experiment/core_code/methods/standalone_integrator.py \
        Experiment/core_code/tests/test_standalone_integrator_num_explores.py
git commit -m "feat(standalone-integrator): expose num_explores knob (default 8)"
```

---

## Task 2: HLE N=4 yaml + launcher

**Files:**
- Create: `/data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/hle/sonnet/hle_sonnet_standalone_integrator_n4.yaml`
- Create: `/data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/hle/sonnet/run_standalone_integrator_n4.sh`

- [ ] **Step 1: Write the yaml**

Path: `scripts/hle/sonnet/hle_sonnet_standalone_integrator_n4.yaml`

```yaml
benchmark:
  name: hle
  subset: gold
  text_only: true
  judge:
    name: claude
    model: claude-haiku-4-5-20251001
method:
  name: standalone-integrator
  backend:
    name: claude
  integrate_model: claude-sonnet-4-6
  cache_dir: ../analysis/cache/hle/sonnet/gold
  # Override default 8 -> 4. Halved candidate budget for a cost-vs-accuracy
  # Pareto point on the same fixed cache. Reuses the 800 cached Haiku grade
  # bundles (judge_spec unchanged), so only 100 integrator-result judges incur
  # new API spend. Coupling: paper main.tex tab:main-results panel (a) gets a
  # new "LLM Selection (N=4)" row sibling to the existing N=8 row at line 241.
  num_explores: 4
num: 100
num_workers: 16
seed: 42
log_dir: ../analysis/run/hle/sonnet_standalone_integrator_n4
```

- [ ] **Step 2: Write the launcher**

Path: `scripts/hle/sonnet/run_standalone_integrator_n4.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/hle/sonnet_standalone_integrator_n4
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/hle/sonnet/hle_sonnet_standalone_integrator_n4.yaml \
	> ../analysis/run/hle/sonnet_standalone_integrator_n4/run.log 2>&1 &
echo "PID=$!"
echo "log=/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_standalone_integrator_n4/run.log"
```

- [ ] **Step 3: chmod and syntax-check**

```bash
chmod +x /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/hle/sonnet/run_standalone_integrator_n4.sh
bash -n /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/hle/sonnet/run_standalone_integrator_n4.sh
```

Expected: zero output (clean syntax).

- [ ] **Step 4: Dry-load the yaml**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output python -c "
import yaml
from pydantic import TypeAdapter
from methods.specs import MethodSpec
spec = yaml.safe_load(open('scripts/hle/sonnet/hle_sonnet_standalone_integrator_n4.yaml'))['method']
parsed = TypeAdapter(MethodSpec).validate_python(spec)
assert parsed.num_explores == 4
print(f'OK: HLE N=4 spec parses, num_explores={parsed.num_explores}')
"
```

Expected: `OK: HLE N=4 spec parses, num_explores=4`.

- [ ] **Step 5: Commit Task 2**

```bash
cd /data3/peijia/dr-claw/Explain
git add Experiment/core_code/scripts/hle/sonnet/hle_sonnet_standalone_integrator_n4.yaml \
        Experiment/core_code/scripts/hle/sonnet/run_standalone_integrator_n4.sh
git commit -m "feat(eval): add HLE N=4 LLM Selection config + launcher"
```

---

## Task 3: LCB N=4 yaml + launcher

**Files:**
- Create: `/data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/lcb/sonnet/lcb_sonnet_standalone_integrator_n4.yaml`
- Create: `/data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/lcb/sonnet/run_standalone_integrator_n4.sh`

- [ ] **Step 1: Write the yaml**

Path: `scripts/lcb/sonnet/lcb_sonnet_standalone_integrator_n4.yaml`

```yaml
benchmark:
  name: lcb
method:
  name: standalone-integrator
  backend:
    name: claude
  integrate_model: claude-sonnet-4-6
  cache_dir: ../analysis/cache/lcb/sonnet
  # Override default 8 -> 4. Halved candidate budget for a cost-vs-accuracy
  # Pareto point on the same fixed cache. LCB grades via lcb_runner test cases
  # (no LLM judge), so judge cost is $0; only the 175 integrator calls incur
  # new API spend. Coupling: paper main.tex tab:main-results panel (b) gets a
  # new "LLM Selection (N=4)" row sibling to the existing N=8 row at line 267.
  num_explores: 4
num_workers: 16
seed: 42
log_dir: ../analysis/run/lcb/sonnet_standalone_integrator_n4
```

- [ ] **Step 2: Write the launcher**

Path: `scripts/lcb/sonnet/run_standalone_integrator_n4.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/lcb/sonnet_standalone_integrator_n4
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/lcb/sonnet/lcb_sonnet_standalone_integrator_n4.yaml \
	> ../analysis/run/lcb/sonnet_standalone_integrator_n4/run.log 2>&1 &
echo "PID=$!"
echo "log=/data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/run.log"
```

- [ ] **Step 3: chmod and syntax-check**

```bash
chmod +x /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/lcb/sonnet/run_standalone_integrator_n4.sh
bash -n /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/lcb/sonnet/run_standalone_integrator_n4.sh
```

Expected: zero output.

- [ ] **Step 4: Dry-load the yaml**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output python -c "
import yaml
from pydantic import TypeAdapter
from methods.specs import MethodSpec
spec = yaml.safe_load(open('scripts/lcb/sonnet/lcb_sonnet_standalone_integrator_n4.yaml'))['method']
parsed = TypeAdapter(MethodSpec).validate_python(spec)
assert parsed.num_explores == 4
print(f'OK: LCB N=4 spec parses, num_explores={parsed.num_explores}')
"
```

Expected: `OK: LCB N=4 spec parses, num_explores=4`.

- [ ] **Step 5: Commit Task 3**

```bash
cd /data3/peijia/dr-claw/Explain
git add Experiment/core_code/scripts/lcb/sonnet/lcb_sonnet_standalone_integrator_n4.yaml \
        Experiment/core_code/scripts/lcb/sonnet/run_standalone_integrator_n4.sh
git commit -m "feat(eval): add LCB N=4 LLM Selection config + launcher"
```

---

## Task 4: Execute HLE N=4 run

**Files:** none modified. Produces `analysis/run/hle/sonnet_standalone_integrator_n4/`.

- [ ] **Step 1: Launch HLE run; share PID + absolute log path**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
bash scripts/hle/sonnet/run_standalone_integrator_n4.sh
```

The launcher prints `PID=<pid>` and `log=/data3/peijia/dr-claw/.../run.log`. Share both with the user immediately (per `feedback_share_long_running_logs`).

- [ ] **Step 2: 10-minute heartbeat — confirm banner is correct**

After ~30 s, check banner lines. Expected lines in the log:
- `standalone-integrator: 100 questions with cache (from 668)`
- For first qid: `[standalone-integrator] N candidates -> answer=...` where `N == 4`

If banner instead shows 0 cache hits or `N==8` for first qid, STOP — Phase 0 truncation didn't take effect.

- [ ] **Step 3: Wait for `EVALUATION COMPLETE`; gate 1 — completeness**

```bash
grep -E "EVALUATION COMPLETE|Total:|Integrated:" /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_standalone_integrator_n4/run.log | tail -3
```

Expected: `Total: 100`, `Integrated: X/100` for some X.

- [ ] **Step 4: Gate 2 — cost ceiling**

```bash
grep -A 5 "Cost breakdown" /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_standalone_integrator_n4/run.log
```

Expected (hard ceilings):
- `Integrator   $X` with X ≤ 7 (estimate ≈ $5.5; ceiling 27% margin)
- `Judge        $X` with X ≤ 5 (estimate ≈ $2.9; 800 explore-judges hit cache at $0)

If Judge > $5, STOP — cache lookup is broken (R4 fired); do not proceed to Task 5 until diagnosed.

- [ ] **Step 5: Gate 3 — Pass@1 sanity**

```bash
grep "best-of-1" /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_standalone_integrator_n4/run.log
```

Expected: `best-of-1   48.0%   ...` (deterministic — same first explore from same cache as the existing N=8 run; tolerance ±1pp).

If best-of-1 deviates beyond ±1pp, STOP — explorer cache changed (or ordering changed); do not write the row to paper.

- [ ] **Step 6: Gate 4 — no Tracebacks**

```bash
grep -c "Traceback" /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_standalone_integrator_n4/run.log
```

Expected: 0.

- [ ] **Step 7: Update `todo_llm_selection_n4_hle.md` Phase 2 / 3 with Evidence**

Edit the TODO file in place. Fill each Gate's `Evidence ·` line with the concrete numeric value just measured. Flip `☐ → ✓` only after every Gate passes.

---

## Task 5: Execute LCB N=4 run

**Files:** none modified. Produces `analysis/run/lcb/sonnet_standalone_integrator_n4/`.

- [ ] **Step 1: Launch LCB run; share PID + absolute log path**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
bash scripts/lcb/sonnet/run_standalone_integrator_n4.sh
```

Share PID + `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/run.log`.

- [ ] **Step 2: 10-minute heartbeat — banner check**

Expected banner: `standalone-integrator: 175 questions with cache (from 175)`.

- [ ] **Step 3: Wait for `EVALUATION COMPLETE`; completeness**

```bash
grep -E "EVALUATION COMPLETE|Total:|Integrated:" /data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/run.log | tail -3
```

Expected: `Total: 175`, `Integrated: X/175`.

- [ ] **Step 4: Cost ceiling**

```bash
grep -A 5 "Cost breakdown" /data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/run.log
```

Expected:
- `Integrator   $X` with X ≤ 13 (estimate ≈ $10; 30% margin)
- `Judge        $0.0` (LCB grades via test execution, no LLM judge)

- [ ] **Step 5: Pass@1 sanity**

```bash
grep "best-of-1" /data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/run.log
```

Expected: `best-of-1   77.14%   ...` ± 1pp.

- [ ] **Step 6: Explore-distribution check**

```bash
grep -A 12 "Explore distribution" /data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/run.log
```

Expected lines (raw cache contents — N=4 truncation is per-call, not per-cache):
```
0 explores: 9 questions
1 explores: 1 questions
2 explores: 3 questions
3 explores: 2 questions
4 explores: 2 questions
5 explores: 2 questions
6 explores: 2 questions
7 explores: 3 questions
8 explores: 151 questions
```

- [ ] **Step 7: No Tracebacks**

```bash
grep -c "Traceback" /data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/run.log
```

Expected: 0.

- [ ] **Step 8: Update `todo_llm_selection_n4_lcb.md` Phase 2 / 3 with Evidence**

Same discipline as Task 4 Step 7.

---

## Self-Review (post-write)

**Spec coverage:**
- User intent ("HLE N=4 + LCB N=4 LLM Selection ≈ $8 + $10") → covered by Tasks 4 + 5 with cost gates ≤ $7+$5 and ≤ $13 respectively.
- "Default N=8, override in yaml" → covered by Task 1 Step 1 (default `num_explores: int = 8`) and Tasks 2-3 Step 1 (yaml `num_explores: 4` with rationale comment).
- Cache discipline (no fresh explore generation, judge cache reuse) → covered by Task 4 Steps 2 and 4 gates.
- Operator-side gate discipline → cross-referenced TODO files at top of plan.

**Placeholder scan:** none — every step has an exact command + expected output, no "TBD" / "similar to" / "fill in".

**Type consistency:** field name `num_explores` used identically in `specs.py`, `registry.py`, `standalone_integrator.py`, both yamls. Default `8` consistent across the three call sites (spec field, solve signature, comment). Cache key `integrate_standalone_{N}` is preserved verbatim from the existing implementation.
