# Explore Cache Owner Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ExploreVariant` the single owner of the explore cache. Make `JudgeOutcome` a self-describing data type that owns its own persistence schema. Eliminate scattered cache I/O across `eval.py` / `precache_explores.py` / `methods/base.py` / `benchmarks/grader.py`.

**Architecture:** Two new dataclasses (`Exploration`, `JudgeOutcome`) carry their own `.persist(target_dir)` methods. `ExploreVariant` gains two intent-driven methods: `get_exploration` (single-by-idx, generates on miss, optionally grades) and `get_all_explorations` (list-by-qid, read-only, optionally grades). All explore-cache I/O routes through these methods. Integrate cache is dropped (it was pseudo-deterministic). Grader is a `(answer, qid) -> JudgeOutcome` closure constructed in eval setup. Resume's done-records grading becomes pure deserialization; record schema is extended to carry `per_variant_candidates` for multi-variant tts-agent.

**Tech Stack:** Python 3.11, pydantic v2, asyncio, pytest. Existing project conventions: `from __future__ import annotations`, `extra: forbid` on all yaml-facing schemas, no defensive programming (assert at boundaries), no try-except for unexpected errors.

**Locked design contracts (12)** — these are decisions, not open questions:

1. `JudgeOutcome` dataclass + `.label_for(spec)` static + `.label` property + `.persist(dir)` method
2. `judge_answer` returns `JudgeOutcome`; `out_dir` parameter removed
3. `BenchmarkConfig.grade(...)` returns `JudgeOutcome`; `out_dir` parameter removed
4. `Exploration` dataclass with `.persist(dir)` method; carries `verdict: JudgeOutcome | None` and `rollout_idx: int | None`
5. `ExploreVariant.get_exploration(qid, idx, *, rollout_idx=None, generate_fn, grader=None)` — `generate_fn` is mandatory; returns `Exploration` always; never returns `None`
6. `ExploreVariant.get_all_explorations(qid, *, rollout_idx=None, grader=None) -> list[Exploration]` — pure read; empty list is legitimate
7. Integrate has no cache: `RoleSlot.cache_dir` field deleted; yaml `integrate.cache_dir:` removed from every config; integrate calls go directly through backend
8. `EvalConfig.cache_only: bool | None = None` yaml field; `False` skips preflight + lets explore cache miss → API; `True` makes the caller-side `generate_fn` raise on miss
9. `JudgeOutcome.label_for(None) → None`; rule-based grading (LCB/GPQA/AIME, judge_spec=None) skips judge bundle persistence — caller checks `if outcome.label is not None` before persisting
10. Cache subpath includes `rollout_idx`: `<cache_dir>/<qid>/[rollout_<r>/]explore_<idx>/` — destructive cache migration is part of this refactor
11. `methods/base.py:315` `cache_key.startswith("integrate_")` exemption deleted with the entire `cache_only` raise branch
12. `eval.py:grade_done_record` deleted; resume done-records path is pure record-to-tuple deserialize. Record schema gains `per_variant_candidates: dict[label, list[CandidateDict]] | None`. NO `schema_version` field — presence/absence of `per_variant_candidates` is itself the schema indicator. Multi-variant resume reads `rec["per_variant_candidates"]` directly; KeyError on unmigrated record is the loud signal (consistent with project policy: prefer breaking change over version-tagged dual-read paths).

**Architecture diagram:**

```
JudgeOutcome (dataclass)              Exploration (dataclass)
  is_correct, cost_usd                  qid, idx, rollout_idx
  judge_spec_snapshot                   answer, trajectory, cost
  input_md, output_md, result_dict      model, timed_out
  .label_for(spec) static               verdict: JudgeOutcome | None
  .label property                       .persist(target_dir)
  .persist(target_dir)

judge_answer(predicted, gold, q, spec) -> JudgeOutcome    [pure, no I/O]
BenchmarkConfig.grade(...) -> JudgeOutcome                 [pure, no I/O]

Grader Protocol = async (answer, qid) -> JudgeOutcome      [closure]
   constructed in eval setup, captures rows_by_id + benchmark + judge_spec

ExploreVariant
  .get_exploration(qid, idx, *, rollout_idx, generate_fn, grader=None)
        -> Exploration            [generate on miss; persist exp; persist judge if outcome.label is not None]
  .get_all_explorations(qid, *, rollout_idx, grader=None)
        -> list[Exploration]      [read-only; per-idx judge optional]
  ._explore_dir, ._judge_dir, ._has_explore, ._load_explore, ._load_judge   [internal helpers]

Consumers:
  precache_explores.py     -> variant.get_exploration(generate_fn=...)
  tts_agent.run_explore    -> variant.get_exploration(generate_fn=...)
  eval.py grading phase    -> variant.get_all_explorations(grader=g)
  eval.py final answer     -> grader(predicted, qid) + outcome.persist(run_dir/.../judges/<label>)
  rerank/standalone-integ  -> variant.get_all_explorations(grader=None)

Deleted:
  methods/base.py:make_sub_model_caller                  [collapsed into ExploreVariant]
  benchmarks/base.py:judge_label                         [collapsed into JudgeOutcome.label_for]
  benchmarks/base.py:find_cached_judge                   [collapsed into ExploreVariant._load_judge]
  eval.py:_grade_with_cache                              [collapsed into ExploreVariant + grader]
  eval.py:_grade_cached_explores                         [replaced by get_all_explorations]
  eval.py:_grade_question_explores / _multi              [replaced by get_all_explorations]
  eval.py:grade_done_record                              [replaced by pure deserialize]
  RoleSlot.cache_dir field                               [integrate has no cache]
```

**Migration / breaking change disclosure:**

- All existing `<cache_dir>/<qid>/explore_<idx>/result.json` files remain valid (path didn't change for K=1 / no-rollout case).
- For yaml configs that set `num_rollouts > 1`: existing cached explores under the bare `explore_<idx>/` slot were already corrupted by mutual rollout overwrites. Path becomes `<cache_dir>/<qid>/rollout_<r>/explore_<idx>/`. Operators must accept that K>1 cache is being rebuilt.
- All yaml configs with `integrate.cache_dir:` field will fail validation after task 13. `tools/migrate_yaml_drop_integrate_cache_dir.py` strips that field across all `Experiment/core_code/scripts/**/*.yaml`.
- Old `results.jsonl` records that lack `per_variant_candidates` field: when resumed under multi-variant tts-agent, will fail with `KeyError`. Single-variant resume continues to work (uses existing `explore_candidates` field). See "Reality Check / Issue R3" for the one-shot migration script that backfills `per_variant_candidates` from cache_dir contents (zero API for normal cases).

---

## Reality Check (added 2026-05-05 after disk audit)

This section was added after auditing 5861 existing `judges/<label>/grade.json` files on disk. Two issues the plan body underspecifies; both must be resolved AFTER the refactor lands, before re-running any benchmark that wants to reuse the existing judge cache.

### Issue R1 — `grade.json` is currently 3 schemas in the wild, not 1

Disk audit (2026-05-05, `Experiment/analysis/cache/**/judges/claude__claude-haiku-*/grade.json`):

| Schema | Count | % | Shape |
|---|---|---|---|
| A (oldest) | 2332 | 39.8% | `{"judge_model": "claude-haiku-...", "is_correct": ..., "predicted": ..., "gold": ..., "judge_cost_usd": ...}` |
| B (middle) | 2987 | 51.0% | `{"judge_spec": {"name": "claude", "model": "..."}, "is_correct": ..., "predicted": ..., "gold": ..., "judge_cost_usd": ...}` |
| NEW (target) | 542 | 9.2% | `{"judge_spec": {"backend": "claude", "model": "..."}, "is_correct": ..., "cost_usd": ...}` |

The body's `JudgeOutcome.persist` writes the NEW shape. The body's `_load_judge` reads `grade.get("cost_usd", 0.0)` — which for Schema A & B silently returns 0 because the old key is `judge_cost_usd`. This violates project policy (CLAUDE.md: "no silent fallback to default value"). Judge cost accounting on resume across old caches will be silently undercounted to 0.

Side effect of the same audit: file completeness inside `judges/<label>/` directories is also degraded — only 27.2% have `input.md`, 89.3% have `config.json`/`output.md`/`result.json`, 100% have `grade.json`. The 627 "skeleton" bundles (only grade.json) are babyvision rule-based MC verdicts that got stamped with an LLM judge label by accident; they have `judge_cost_usd: 0.0`. This does NOT block grade.json migration but means `JudgeOutcome.input_md` will be empty when reading old bundles.

**Fix (deferred until after refactor):** Write `tools/migrate_grade_json.py` with a 3-branch reader:

```python
def migrate_one(grade_path: Path) -> None:
    g = json.loads(grade_path.read_text())
    if "judge_spec" in g and isinstance(g["judge_spec"], dict) and "backend" in g["judge_spec"]:
        return  # NEW, skip
    if "judge_spec" in g and isinstance(g["judge_spec"], dict) and "name" in g["judge_spec"]:
        spec = {"backend": g["judge_spec"]["name"], "model": g["judge_spec"]["model"]}
    elif "judge_model" in g:
        backend, model = grade_path.parent.name.split("__", 1)
        assert model == g["judge_model"], f"directory/grade.json model mismatch at {grade_path}"
        spec = {"backend": backend, "model": model}
    else:
        raise AssertionError(f"unknown grade.json schema at {grade_path}")
    new_grade = {
        "judge_spec": spec,
        "is_correct": g["is_correct"],
        "cost_usd": g.get("judge_cost_usd", g.get("cost_usd", 0.0)),
    }
    tmp = grade_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new_grade, indent=2, ensure_ascii=False))
    os.replace(tmp, grade_path)  # atomic
```

Pure local file ops. Zero API. <10s wall-clock for 5861 bundles. Run with `--dry-run` first to confirm 0 OTHER classifications, then in-place.

### Issue R2 — `_load_judge` silently regresses `find_cached_judge`'s best-effort matching

The body proposes:

```python
# specs.py _load_judge (plan body line ~824)
assert config == judge_spec, (
    f"Cached judge bundle config mismatch at {jd}: "
    f"stored={config} requested={judge_spec}. "
    "Refuse to silently use mismatched cache."
)
```

Current code at `benchmarks/base.py:65-136` (`find_cached_judge`) implements a deliberate **4-branch best-effort policy**:

1. `stored == judge_spec` → exact hit (silent)
2. `stored ⊂ requested` → best-effort hit (legitimate schema evolution: cache predates a new optional field)
3. shared keys with disagreeing values → RuntimeError (true conflict)
4. `stored ⊋ requested` → RuntimeError (cache was made under stricter spec; refusing to inherit its verdict for a less-specific request)

Process-level counters `_JUDGE_CACHE_STATS` track exact_hits vs best_effort_hits and aggregate-print one banner line at run end. Banner-level visibility was an explicit design choice to avoid 800 per-call warning lines per HLE run.

The plan body collapses all four branches into a single `assert config == judge_spec`. Practical impact today: any yaml that adds an optional judge field (`vllm_sampling`, `effort`, `budget_tokens` — `grader.py:132-138` already reads these) will assertion-fail on every existing cached bundle, wiping the entire judge cache.

This regression is not declared in the locked contracts (1-12). It is a silent behavioral narrowing.

**Decision (2026-05-05, user binding):** "单向 Best Effort 是不应该改变的，他做的这个 strict matching 是个错误的行为，这里不应该改变." → revert the silent narrowing. Plan body Task 5 Step 5 has been edited in-place to use the 4-branch best-effort logic shown below. `_JUDGE_CACHE_STATS` counters and `summarize_judge_cache()` move from `benchmarks/base.py` (which Task 18 deletes) to `cache_types.py` so eval.py's run-end banner line keeps working. Sketch (now matches plan body Task 5 Step 5 verbatim):

```python
def _load_judge(self, qid, idx, judge_spec, rollout_idx=None):
    from cache_types import JudgeOutcome
    label = JudgeOutcome.label_for(judge_spec)
    if label is None:
        return None
    jd = self._judge_dir(qid, idx, label, rollout_idx)
    cp, gp = jd / "config.json", jd / "grade.json"
    if not (cp.exists() and gp.exists()):
        return None
    stored = json.loads(cp.read_text(encoding="utf-8"))
    # Branch 1: exact equality
    if stored == judge_spec:
        pass
    else:
        shared = set(stored) & set(judge_spec)
        conflicts = {k: (stored[k], judge_spec[k]) for k in shared if stored[k] != judge_spec[k]}
        if conflicts:
            raise RuntimeError(f"Judge config conflict at {jd}: {conflicts}")
        only_stored = sorted(set(stored) - set(judge_spec))
        if only_stored:
            raise RuntimeError(f"Judge cache spec narrowing at {jd}: stored has extra keys {only_stored}")
        # else: stored ⊂ requested → best-effort hit (legitimate schema evolution)
    grade = json.loads(gp.read_text(encoding="utf-8"))
    return JudgeOutcome(
        is_correct=grade["is_correct"],
        cost_usd=grade["cost_usd"],   # post-Issue-R1 migration: required key, no .get fallback
        judge_spec_snapshot=stored,
        input_md=(jd / "input.md").read_text(encoding="utf-8") if (jd / "input.md").exists() else "",
        output_md=(jd / "output.md").read_text(encoding="utf-8") if (jd / "output.md").exists() else "",
        result_dict=json.loads((jd / "result.json").read_text(encoding="utf-8")) if (jd / "result.json").exists() else {},
    )
```

Note: after Issue-R1 migration, `cost_usd` is a required key in every grade.json on disk → use `grade["cost_usd"]` not `.get(..., 0.0)`. The strict access is the assertion; missing key would mean the migration was skipped for that bundle.

### Issue R3 — Old `results.jsonl` records lack `per_variant_candidates`

Plan body Task 14 adds `per_variant_candidates` to the record schema. Plan body Task 15 reads `rec["per_variant_candidates"]` directly (no version field, no `.get` fallback) — KeyError is the loud signal that an unmigrated record snuck through.

User-binding decision (2026-05-05): "你也没必要用那个 version 编号了... 你记这个 version 编号反而是画蛇添足." → no `schema_version` field is introduced; field presence is the schema indicator. Aligns with CLAUDE.md "DO NOT USE VERSION TO DENOTE THE CHANGES... ALWAYS OVERWRITE THE PREVIOUS VERSION".

**Fix (deferred until after refactor):** Write `tools/migrate_results_jsonl_per_variant.py`. For every old multi-variant `results.jsonl` row, walk `cache_dirs_multi[label] / qid / explore_*/` and reconstruct `per_variant_candidates`:

```python
def migrate_record(rec: dict, cache_dirs_multi: dict, benchmark, judge_spec) -> dict:
    if "per_variant_candidates" in rec:
        return rec  # already migrated
    pvc: dict[str, list[dict]] = {}
    for label, cache_dir in cache_dirs_multi.items():
        cands = []
        qdir = cache_dir / rec["id"]
        for ed in sorted(qdir.glob("explore_*"), key=lambda p: int(p.name.split("_")[1])):
            r = json.loads((ed / "result.json").read_text())
            answer = r["answer"]
            # Resolve verdict: prefer cached judge bundle, else rule-based grade
            label_dir = ed / "judges" / f"{judge_spec['backend']}__{judge_spec['model']}"
            if (label_dir / "grade.json").exists():
                is_correct = json.loads((label_dir / "grade.json").read_text())["is_correct"]
            else:
                # rule-based path (LCB/GPQA/AIME or MC short-circuit) — recompute locally
                is_correct = benchmark.rule_based_check(answer, rec["gold_answer"], rec)
            cands.append({
                "normalized_answer": benchmark.normalize_answer(answer),
                "is_correct": is_correct,
                "cost_usd": r["cost_usd"],
            })
        pvc[label] = cands
    rec["per_variant_candidates"] = pvc
    return rec
```

Pure local file ops + at most cheap rule-based checks. Zero LLM API for the normal case (judge cache present). For LLM-judge benchmarks (HLE/RBenchV/BabyVision-blank) where a judge bundle is missing, the script raises with the missing path — operator either pre-runs the judge OR the script bumps a `--allow-judge-call` flag with explicit cost accounting. Default = strict, refuse to silently judge-on-migration.

**Edge cases handled:**
- `explore_*/result.json` missing → raise (cache itself is corrupt, operator must investigate)
- `cache_dirs_multi` not provided to script → require `--yaml <path>` flag pointing at the original yaml
- record already has `per_variant_candidates` → skip (idempotent)

### Execution order

1. Land the refactor (Tasks 1-19) as written. Plan body has been edited in-place to reflect the user-binding decisions on R2 (best-effort matching restored) and R3 (no schema_version field).
2. Run `tools/migrate_grade_json.py --dry-run` (R1); verify 0 OTHER bundles. Then run in-place.
3. For each old multi-variant `results.jsonl` to be resumed: run `tools/migrate_results_jsonl_per_variant.py --yaml <yaml> --in-place` (R3).
4. Re-run any benchmark — judge cache hits via 4-branch best-effort, multi-variant resume reads `per_variant_candidates` directly.

---

## File Structure

**New files:**
- `Experiment/core_code/cache_types.py` — `Exploration` and `JudgeOutcome` dataclasses (single source of truth for both schemas)
- `Experiment/core_code/tests/test_cache_types.py` — unit tests for the two dataclasses
- `Experiment/core_code/tests/test_explore_variant_methods.py` — unit tests for `get_exploration` / `get_all_explorations` (extends existing `test_explore_variant.py`)
- `Experiment/core_code/tools/migrate_yaml_drop_integrate_cache_dir.py` — one-shot yaml migration

**Modified files:**
- `Experiment/core_code/benchmarks/grader.py` — `judge_answer` returns `JudgeOutcome`; drop `out_dir` parameter
- `Experiment/core_code/benchmarks/base.py` — `BenchmarkConfig.grade` signature; delete `judge_label`, `find_cached_judge`, `summarize_judge_cache`
- `Experiment/core_code/benchmarks/{hle,gpqa,aime,lcb,babyvision,rbenchv}.py` — each `grade(...)` returns `JudgeOutcome`
- `Experiment/core_code/methods/specs.py` — `ExploreVariant` gains methods; `RoleSlot.cache_dir` removed
- `Experiment/core_code/methods/base.py` — delete `make_sub_model_caller`'s cache layer; backend.call_sub_model becomes the only LLM-call surface
- `Experiment/core_code/methods/tts_agent.py` — `run_explore` uses `variant.get_exploration`; integrate goes direct to backend
- `Experiment/core_code/methods/self_refine.py` / `socratic_self_refine.py` / `budget_forcing.py` — same migration
- `Experiment/core_code/methods/standalone_integrator.py` — `get_all_explorations` for read; integrate goes direct to backend
- `Experiment/core_code/methods/reward_rerank.py` — `get_all_explorations` for read
- `Experiment/core_code/precache_explores.py` — uses `variant.get_exploration`
- `Experiment/core_code/eval.py` — `EvalConfig.cache_only` field; preflight skip; rewrite resume done-records path; rewrite per-question grading path; delete dead helpers
- `Experiment/core_code/methods/registry.py` — `MethodConfig.preflight` accepts spec instead of cache_dir+num_explores
- All `Experiment/core_code/scripts/**/*.yaml` — drop `integrate.cache_dir:` lines

**Deleted symbols:**
- `methods/base.py:make_sub_model_caller` (function and all callers)
- `benchmarks/base.py:judge_label` (function — collapsed into `JudgeOutcome.label_for`)
- `benchmarks/base.py:find_cached_judge` (function — 4-branch best-effort logic moved to `ExploreVariant._load_judge`, NOT collapsed to strict-equality)
- `eval.py:_grade_with_cache` / `_grade_cached_explores` / `_grade_question_explores` / `_grade_question_explores_multi` / `grade_done_record` (functions)
- `methods/specs.py:RoleSlot.cache_dir` (field)

**Relocated symbols (NOT deleted):**
- `benchmarks/base.py:_JUDGE_CACHE_STATS` / `reset_judge_cache_stats` / `summarize_judge_cache` → `cache_types.py` (Task 18). Run-end banner aggregation policy preserved verbatim — moved alongside the matching logic that feeds them.

---

## Phase 1 — New Data Types (No Behavior Change Yet)

### Task 1: `JudgeOutcome` dataclass with persist method

**Files:**
- Create: `Experiment/core_code/cache_types.py`
- Test: `Experiment/core_code/tests/test_cache_types.py`

- [ ] **Step 1: Write the failing test for `JudgeOutcome.label_for`**

```python
# tests/test_cache_types.py
from cache_types import JudgeOutcome


def test_label_for_with_spec():
    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    assert JudgeOutcome.label_for(spec) == "claude__claude-haiku-4-5-20251001"


def test_label_for_none_returns_none():
    assert JudgeOutcome.label_for(None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Experiment/core_code && conda run -n explain --no-capture-output pytest tests/test_cache_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cache_types'`

- [ ] **Step 3: Write minimal `cache_types.py` with `JudgeOutcome` skeleton**

```python
# cache_types.py
"""Owner-blind dataclasses: each carries its own persistence schema.

JudgeOutcome  — one verdict + its trace artifacts. Pure data. .persist(dir)
                writes the on-disk bundle. .label_for(spec) is the canonical
                judge-bundle directory naming function.

Exploration   — one explore call's result + its trace. Optional verdict.
                .persist(dir) writes input.md / output.md / result.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JudgeOutcome:
    is_correct: bool
    cost_usd: float
    judge_spec_snapshot: dict | None
    input_md: str
    output_md: str
    result_dict: dict[str, Any]

    @staticmethod
    def label_for(judge_spec: dict | None) -> str | None:
        """Canonical bundle directory name for a judge_spec.

        Returns None for rule-based grading (LCB/GPQA/AIME) — caller checks
        `if outcome.label is not None` before persisting.
        """
        if judge_spec is None:
            return None
        return f"{judge_spec['backend']}__{judge_spec['model']}"

    @property
    def label(self) -> str | None:
        return self.label_for(self.judge_spec_snapshot)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd Experiment/core_code && conda run -n explain --no-capture-output pytest tests/test_cache_types.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the failing test for `.persist`**

Append to `tests/test_cache_types.py`:

```python
def test_persist_writes_five_files(tmp_path):
    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    outcome = JudgeOutcome(
        is_correct=True, cost_usd=0.0012,
        judge_spec_snapshot=spec,
        input_md="judge prompt here",
        output_md="judge response here",
        result_dict={"correct": True, "explanation": "matches"},
    )
    target = tmp_path / "judges" / outcome.label
    outcome.persist(target)

    assert (target / "config.json").exists()
    assert json.loads((target / "config.json").read_text()) == spec
    assert (target / "input.md").read_text() == "judge prompt here"
    assert (target / "output.md").read_text() == "judge response here"
    assert (target / "result.json").exists()
    grade = json.loads((target / "grade.json").read_text())
    assert grade["is_correct"] is True
    assert grade["cost_usd"] == 0.0012
    assert grade["judge_spec"] == spec
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_cache_types.py::test_persist_writes_five_files -v`
Expected: FAIL with `AttributeError: 'JudgeOutcome' object has no attribute 'persist'`

- [ ] **Step 7: Add `.persist` method to `JudgeOutcome`**

Add to `cache_types.py` inside the `JudgeOutcome` class:

```python
    def persist(self, target_dir: Path) -> None:
        """Write 5-file judge bundle to target_dir.

        Caller must NOT call this when self.label is None (rule-based grading
        has no LLM trace to archive). Asserts if invoked in that state.
        """
        assert self.label is not None, (
            "JudgeOutcome.persist called for rule-based grading "
            "(judge_spec_snapshot is None). Caller must guard with "
            "`if outcome.label is not None`."
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "config.json").write_text(
            json.dumps(self.judge_spec_snapshot, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        (target_dir / "input.md").write_text(self.input_md, encoding="utf-8")
        (target_dir / "output.md").write_text(self.output_md, encoding="utf-8")
        (target_dir / "result.json").write_text(
            json.dumps({**self.result_dict, "cost_usd": self.cost_usd},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (target_dir / "grade.json").write_text(
            json.dumps({
                "judge_spec": self.judge_spec_snapshot,
                "is_correct": self.is_correct,
                "cost_usd": self.cost_usd,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
```

- [ ] **Step 8: Add the rule-based assertion test**

```python
def test_persist_asserts_when_rule_based(tmp_path):
    outcome = JudgeOutcome(
        is_correct=True, cost_usd=0.0,
        judge_spec_snapshot=None,
        input_md="", output_md="",
        result_dict={"correct": True},
    )
    import pytest
    with pytest.raises(AssertionError, match="rule-based grading"):
        outcome.persist(tmp_path / "judges" / "should_not_exist")
```

- [ ] **Step 9: Run all tests**

Run: `pytest tests/test_cache_types.py -v`
Expected: PASS (4 tests)

- [ ] **Step 10: Commit**

```bash
git add Experiment/core_code/cache_types.py Experiment/core_code/tests/test_cache_types.py
git commit -m "feat(cache_types): add JudgeOutcome dataclass with label_for + persist"
```

---

### Task 2: `Exploration` dataclass with persist method

**Files:**
- Modify: `Experiment/core_code/cache_types.py`
- Test: `Experiment/core_code/tests/test_cache_types.py`

- [ ] **Step 1: Write the failing test for `Exploration.persist`**

Append to `tests/test_cache_types.py`:

```python
from cache_types import Exploration


def test_exploration_persist_writes_three_files(tmp_path):
    exp = Exploration(
        qid="abc", idx=3, rollout_idx=None,
        answer="42",
        trajectory="reasoning here",
        cost_usd=0.05,
        model="claude-sonnet-4-6",
        timed_out=False,
        verdict=None,
    )
    exp.persist(tmp_path / "explore_3")
    assert (tmp_path / "explore_3" / "result.json").exists()
    assert (tmp_path / "explore_3" / "input.md").exists()
    assert (tmp_path / "explore_3" / "output.md").read_text() == "reasoning here"
    payload = json.loads((tmp_path / "explore_3" / "result.json").read_text())
    assert payload["answer"] == "42"
    assert payload["cost_usd"] == 0.05
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload.get("timed_out") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cache_types.py::test_exploration_persist_writes_three_files -v`
Expected: FAIL with `ImportError: cannot import name 'Exploration'`

- [ ] **Step 3: Add `Exploration` to `cache_types.py`**

Append to `cache_types.py`:

```python
@dataclass
class Exploration:
    """One explore call's full record. Self-describing schema.

    On-disk layout under target_dir:
      input.md         — the prompt sent (rendered markdown)
      output.md        — the model's raw trajectory text
      result.json      — structured: answer, cost_usd, model, timed_out, ...
    """
    qid: str
    idx: int
    rollout_idx: int | None
    answer: str
    trajectory: str
    cost_usd: float
    model: str
    timed_out: bool = False
    # Optional grading layer attached by ExploreVariant when grader is provided.
    verdict: JudgeOutcome | None = None
    # Free-form fields preserved from backend response (usage, finish_reason, …)
    extra: dict[str, Any] = field(default_factory=dict)
    # Inputs needed to write input.md (caller fills these in before persist).
    system_prompt: str = ""
    user_message: str = ""

    def persist(self, target_dir: Path) -> None:
        """Write input.md / output.md / result.json into target_dir.

        Used by ExploreVariant for cache_dir persistence and by callers
        for run_dir/trajectories/ mirror writes — same schema, different
        target.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "input.md").write_text(
            f"## System Prompt\n\n{self.system_prompt}\n\n## User Message\n\n{self.user_message}",
            encoding="utf-8",
        )
        (target_dir / "output.md").write_text(self.trajectory, encoding="utf-8")
        result_payload = {
            "answer": self.answer,
            "cost_usd": self.cost_usd,
            "model": self.model,
            "timed_out": self.timed_out,
            **self.extra,
        }
        (target_dir / "result.json").write_text(
            json.dumps(result_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_cache_types.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Add a test for `verdict` round-trip**

```python
def test_exploration_with_verdict_field():
    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    verdict = JudgeOutcome(
        is_correct=True, cost_usd=0.001, judge_spec_snapshot=spec,
        input_md="...", output_md="...", result_dict={"correct": True},
    )
    exp = Exploration(
        qid="q1", idx=1, rollout_idx=None,
        answer="42", trajectory="...", cost_usd=0.05, model="m",
        verdict=verdict,
    )
    assert exp.verdict is verdict
    assert exp.verdict.is_correct is True
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_cache_types.py -v`
Expected: PASS (6 tests)

- [ ] **Step 7: Commit**

```bash
git add Experiment/core_code/cache_types.py Experiment/core_code/tests/test_cache_types.py
git commit -m "feat(cache_types): add Exploration dataclass with persist + verdict field"
```

---

### Task 3: Migrate `judge_answer` to return `JudgeOutcome`

**Files:**
- Modify: `Experiment/core_code/benchmarks/grader.py:109-201`
- Test: `Experiment/core_code/tests/test_benchmark_grade.py`

- [ ] **Step 1: Read existing `judge_answer` to understand structure**

Run: `cat Experiment/core_code/benchmarks/grader.py | head -210 | tail -110`
Note: function currently returns `tuple[bool, float]` and writes 4 files when `out_dir` is set.

- [ ] **Step 2: Write failing test for new return type**

In `tests/test_benchmark_grade.py` add:

```python
import pytest
from cache_types import JudgeOutcome


@pytest.mark.asyncio
async def test_judge_answer_returns_judge_outcome(monkeypatch):
    """judge_answer is pure: returns JudgeOutcome, writes nothing."""
    from benchmarks.grader import judge_answer

    async def fake_call_sub_model(system, user, image, model, schema, writer, **kwargs):
        return ({"correct": True, "explanation": "OK"}, "trajectory text", 0.001, {})

    import benchmarks.grader as grader_mod
    monkeypatch.setattr(grader_mod.backends_claude, "call_sub_model", fake_call_sub_model, raising=False)

    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    outcome = await judge_answer(
        predicted="42", gold="42", question="what is 6*7?", judge_spec=spec,
        max_retries=1,
    )
    assert isinstance(outcome, JudgeOutcome)
    assert outcome.is_correct is True
    assert outcome.cost_usd == 0.001
    assert outcome.judge_spec_snapshot == spec
    assert outcome.label == "claude__claude-haiku-4-5-20251001"
    assert "trajectory text" in outcome.output_md
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_benchmark_grade.py::test_judge_answer_returns_judge_outcome -v`
Expected: FAIL because `judge_answer` still returns tuple and accepts `out_dir`.

- [ ] **Step 4: Refactor `judge_answer` signature + body**

In `benchmarks/grader.py:109-201`:
- Remove `out_dir: Path | None = None` from signature.
- Remove the entire `if out_dir is not None:` block (lines 175-192).
- Capture the user_message string built inside (the prompt sent to judge LLM) into a local for the JudgeOutcome.
- Change return statement from `return last_result["correct"], total_cost` to construct and return `JudgeOutcome`.

Reference shape (replace lines 109-201):

```python
async def judge_answer(
    predicted: str, gold: str, question: str, judge_spec: dict,
    max_retries: int = 3,
) -> JudgeOutcome:
    """Pure judge call. Returns the verdict + full trace. Caller persists."""
    from cache_types import JudgeOutcome
    backend = judge_spec["backend"]
    model = judge_spec["model"]
    user_message = _build_judge_prompt(question, predicted, gold)   # extract existing inline construction
    system_prompt = _judge_system_prompt(backend)
    sub_model_kwargs = _judge_sub_model_kwargs(judge_spec)

    total_cost = 0.0
    last_result: dict | None = None
    last_trajectory = ""
    last_usage: dict = {}
    backend_mod = import_module(f"backends.{backend}")
    for attempt in range(1, max_retries + 1):
        result, trajectory_text, cost_usd, usage = await backend_mod.call_sub_model(
            system_prompt, user_message, None, model, JUDGE_SCHEMA,
            writer=TrajectoryWriter.noop(),
            **sub_model_kwargs,
        )
        total_cost += cost_usd
        last_result, last_trajectory, last_usage = result, trajectory_text, usage
        if not (result.get("timed_out") or result.get("parse_failed")):
            break
        logger.warning(...)  # existing retry log

    if last_result.get("timed_out") or last_result.get("parse_failed"):
        raise RuntimeError(...)  # existing: refuse to silently judge as incorrect

    return JudgeOutcome(
        is_correct=last_result["correct"],
        cost_usd=total_cost,
        judge_spec_snapshot=dict(judge_spec),
        input_md=f"## System Prompt\n\n{system_prompt}\n\n## User Message\n\n{user_message}",
        output_md=last_trajectory,
        result_dict={**last_result, "usage": last_usage},
    )
```

(Engineer should adapt to existing helper names; the `_build_judge_prompt` extraction may already be inline — pull it into a named local string before the loop.)

- [ ] **Step 5: Run grader test**

Run: `pytest tests/test_benchmark_grade.py -v`
Expected: New test PASS. Existing tests may FAIL where they unpack `(bool, float)` — these will be fixed in Task 4.

- [ ] **Step 6: Commit**

```bash
git add Experiment/core_code/benchmarks/grader.py Experiment/core_code/tests/test_benchmark_grade.py
git commit -m "refactor(grader): judge_answer returns JudgeOutcome; drop out_dir param

Pure function: no disk I/O. Caller (ExploreVariant in upcoming commits) is
responsible for persisting the bundle via outcome.persist(target_dir)."
```

---

### Task 4: Migrate all 7 `BenchmarkConfig.grade(...)` to return `JudgeOutcome`

**Files:**
- Modify: `Experiment/core_code/benchmarks/{hle,rbenchv,babyvision,gpqa,aime,lcb}.py` (each `grade` method)
- Modify: `Experiment/core_code/benchmarks/grader.py` (the `grade_answer` wrapper at line 204)
- Modify: `Experiment/core_code/benchmarks/base.py` (BenchmarkConfig abstract method signature if it has one)
- Test: `Experiment/core_code/tests/test_benchmark_grade.py`

- [ ] **Step 1: Failing test for HLE.grade**

```python
@pytest.mark.asyncio
async def test_hle_grade_returns_judge_outcome(monkeypatch):
    from benchmarks.hle import HLEBenchmark
    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    bench = HLEBenchmark(judge_spec=spec)

    async def fake_judge_answer(predicted, gold, question, judge_spec, max_retries=3):
        return JudgeOutcome(is_correct=True, cost_usd=0.001,
                            judge_spec_snapshot=spec, input_md="i", output_md="o",
                            result_dict={"correct": True})
    monkeypatch.setattr("benchmarks.hle.judge_answer", fake_judge_answer)

    row = {"answer_type": "exactMatch"}
    outcome = await bench.grade(predicted="42", gold="42", question="q", row=row, backend="claude")
    assert isinstance(outcome, JudgeOutcome)
    assert outcome.is_correct is True
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest tests/test_benchmark_grade.py::test_hle_grade_returns_judge_outcome -v`
Expected: FAIL — return type is still tuple.

- [ ] **Step 3: Update `HLEBenchmark.grade` (hle.py:153-164)**

```python
# benchmarks/hle.py
from cache_types import JudgeOutcome

async def grade(self, predicted, gold, question, row, backend) -> JudgeOutcome:
    # Note: out_dir parameter removed. Caller persists via outcome.persist(...).
    answer_type = row.get("answer_type", "exactMatch")
    if answer_type == "multipleChoice":
        is_correct = check_answer(predicted, gold, "multipleChoice")
        return JudgeOutcome(
            is_correct=is_correct, cost_usd=0.0,
            judge_spec_snapshot=None,  # rule-based: no LLM trace
            input_md="", output_md="",
            result_dict={"correct": is_correct, "kind": "rule_based_mc"},
        )
    return await judge_answer(
        predicted, gold, question, self.judge_spec,
        max_retries=self.judge_max_retries,
    )
```

- [ ] **Step 4: Update other benchmark grade methods analogously**

**`benchmarks/rbenchv.py:90`**:
```python
async def grade(self, predicted, gold, question, row, backend) -> JudgeOutcome:
    return await judge_answer(predicted, gold, question, self.judge_spec,
                              max_retries=self.judge_max_retries)
```

**`benchmarks/babyvision.py:100`** (hybrid):
```python
async def grade(self, predicted, gold, question, row, backend) -> JudgeOutcome:
    if row.get("ansType") == "choice":
        is_correct = check_answer(predicted, gold, "multipleChoice")
        return JudgeOutcome(is_correct=is_correct, cost_usd=0.0,
                            judge_spec_snapshot=None, input_md="", output_md="",
                            result_dict={"correct": is_correct, "kind": "rule_based_mc"})
    return await judge_answer(predicted, gold, question, self.judge_spec,
                              max_retries=self.judge_max_retries)
```

**`benchmarks/gpqa.py:102`** (rule-based MC):
```python
async def grade(self, predicted, gold, question, row, backend) -> JudgeOutcome:
    is_correct = check_answer(predicted, gold, "multipleChoice")
    return JudgeOutcome(is_correct=is_correct, cost_usd=0.0,
                        judge_spec_snapshot=None, input_md="", output_md="",
                        result_dict={"correct": is_correct, "kind": "rule_based_mc"})
```

**`benchmarks/aime.py:117`** (rule-based exactMatch):
```python
async def grade(self, predicted, gold, question, row, backend) -> JudgeOutcome:
    pred_norm = _normalize_aime_answer(predicted)
    gold_norm = _normalize_aime_answer(gold)
    is_correct = pred_norm == gold_norm
    return JudgeOutcome(is_correct=is_correct, cost_usd=0.0,
                        judge_spec_snapshot=None, input_md="", output_md="",
                        result_dict={"correct": is_correct, "kind": "rule_based_exact",
                                     "pred_norm": pred_norm, "gold_norm": gold_norm})
```

**`benchmarks/lcb.py:140`** (rule-based code execution):
```python
async def grade(self, predicted, gold, question, row, backend) -> JudgeOutcome:
    is_correct, _ = await grade_code(predicted, row)
    return JudgeOutcome(is_correct=is_correct, cost_usd=0.0,
                        judge_spec_snapshot=None, input_md="", output_md="",
                        result_dict={"correct": is_correct, "kind": "rule_based_code"})
```

- [ ] **Step 5: Update `grader.py:grade_answer` wrapper (line 204-220)**

```python
async def grade_answer(predicted, gold, question, answer_type, judge_spec, max_retries=3) -> JudgeOutcome:
    if answer_type == "multipleChoice" or judge_spec is None:
        is_correct = check_answer(predicted, gold, answer_type)
        return JudgeOutcome(is_correct=is_correct, cost_usd=0.0,
                            judge_spec_snapshot=None, input_md="", output_md="",
                            result_dict={"correct": is_correct})
    return await judge_answer(predicted, gold, question, judge_spec, max_retries=max_retries)
```

- [ ] **Step 6: Update `BenchmarkConfig.grade` abstract method signature in `base.py`**

Update the signature in `base.py` near line 379 to remove `out_dir` param and update return type annotation to `JudgeOutcome`.

- [ ] **Step 7: Run all benchmark grade tests**

Run: `pytest tests/test_benchmark_grade.py -v`
Expected: PASS (all 7 benchmark grade methods return JudgeOutcome).

- [ ] **Step 8: Commit**

```bash
git add Experiment/core_code/benchmarks/
git commit -m "refactor(benchmark.grade): return JudgeOutcome; drop out_dir param

Rule-based benchmarks (LCB/GPQA/AIME, plus MC paths in HLE/BabyVision)
return JudgeOutcome with judge_spec_snapshot=None and empty trace fields.
LLM-judge benchmarks (HLE/RBenchV/BabyVision-blank) delegate to
judge_answer which now returns JudgeOutcome directly."
```

---

## Phase 2 — `ExploreVariant` Methods (Old Call Sites Still Work)

### Task 5: `ExploreVariant` internal helpers (path construction + cache I/O)

**Files:**
- Modify: `Experiment/core_code/methods/specs.py` (`ExploreVariant` class)
- Test: `Experiment/core_code/tests/test_explore_variant_methods.py` (new)

- [ ] **Step 1: Failing test for `_explore_dir`**

```python
# tests/test_explore_variant_methods.py
from pathlib import Path
from methods.specs import ExploreVariant, ModelConfig


def _make_variant(tmp_path):
    return ExploreVariant(
        label="default",
        model=ModelConfig(backend="claude", model="claude-sonnet-4-6"),
        cache_dir=tmp_path,
        num_explores=8,
    )


def test_explore_dir_no_rollout(tmp_path):
    v = _make_variant(tmp_path)
    assert v._explore_dir("q1", 3) == tmp_path / "q1" / "explore_3"


def test_explore_dir_with_rollout(tmp_path):
    v = _make_variant(tmp_path)
    assert v._explore_dir("q1", 3, rollout_idx=2) == tmp_path / "q1" / "rollout_2" / "explore_3"


def test_judge_dir(tmp_path):
    v = _make_variant(tmp_path)
    label = "claude__claude-haiku-4-5-20251001"
    assert v._judge_dir("q1", 3, label) == tmp_path / "q1" / "explore_3" / "judges" / label
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_explore_variant_methods.py -v`
Expected: FAIL — methods don't exist.

- [ ] **Step 3: Add helpers to `ExploreVariant` (specs.py:141-154)**

Append inside `ExploreVariant` class body:

```python
    # ---- Internal helpers (path construction + atomic I/O) ----

    def _explore_dir(self, qid: str, idx: int, rollout_idx: int | None = None) -> Path:
        """Path to one explore's bundle directory.
        K=1 (rollout_idx=None) → cache_dir/<qid>/explore_<idx>
        K>1 (rollout_idx=k)    → cache_dir/<qid>/rollout_<k>/explore_<idx>
        """
        base = self.cache_dir / qid
        if rollout_idx is not None:
            base = base / f"rollout_{rollout_idx}"
        return base / f"explore_{idx}"

    def _judge_dir(self, qid: str, idx: int, label: str,
                   rollout_idx: int | None = None) -> Path:
        return self._explore_dir(qid, idx, rollout_idx) / "judges" / label

    def _has_explore(self, qid: str, idx: int, rollout_idx: int | None = None) -> bool:
        return (self._explore_dir(qid, idx, rollout_idx) / "result.json").exists()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_explore_variant_methods.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Add `_load_explore` and `_load_judge` helpers + tests**

```python
def test_load_explore_returns_none_on_miss(tmp_path):
    v = _make_variant(tmp_path)
    assert v._load_explore("q1", 1) is None


def test_load_explore_reads_persisted(tmp_path):
    from cache_types import Exploration
    v = _make_variant(tmp_path)
    exp = Exploration(
        qid="q1", idx=1, rollout_idx=None,
        answer="42", trajectory="t", cost_usd=0.05, model="m",
    )
    exp.persist(v._explore_dir("q1", 1))
    loaded = v._load_explore("q1", 1)
    assert loaded is not None
    assert loaded.answer == "42"
    assert loaded.cost_usd == 0.05
```

Implementation in `specs.py`:

```python
    def _load_explore(self, qid: str, idx: int,
                      rollout_idx: int | None = None) -> "Exploration | None":
        """Load Exploration from disk; None if result.json missing."""
        from cache_types import Exploration  # avoid circular import
        d = self._explore_dir(qid, idx, rollout_idx)
        rp = d / "result.json"
        if not rp.exists():
            return None
        payload = json.loads(rp.read_text(encoding="utf-8"))
        traj = (d / "output.md").read_text(encoding="utf-8") if (d / "output.md").exists() else ""
        return Exploration(
            qid=qid, idx=idx, rollout_idx=rollout_idx,
            answer=payload.get("answer", ""),
            trajectory=traj,
            cost_usd=payload.get("cost_usd", 0.0),
            model=payload.get("model", ""),
            timed_out=payload.get("timed_out", False),
            extra={k: v for k, v in payload.items()
                   if k not in {"answer", "cost_usd", "model", "timed_out"}},
        )

    def _load_judge(self, qid: str, idx: int, judge_spec: dict,
                    rollout_idx: int | None = None) -> "JudgeOutcome | None":
        """Port of `benchmarks/base.py:find_cached_judge` (4-branch best-effort).

        Match policy is UNIDIRECTIONAL by design — preserved verbatim from
        the pre-refactor implementation (do NOT collapse to `stored == spec`):
          - stored == spec                  -> exact hit
          - stored is strict subset of spec -> best-effort hit (legitimate
              schema evolution: cache predates a new optional field)
          - shared key with disagreeing val -> RuntimeError (true conflict)
          - stored has key absent from spec -> RuntimeError (cache was made
              under stricter spec; refusing to inherit its verdict for a
              less-specific request)
          - config / grade missing          -> None (real cache miss)

        Process-level counters _JUDGE_CACHE_STATS that lived in benchmarks/base.py
        move to a module-level dict in cache_types.py (kept identical: exact_hits,
        best_effort_hits, best_effort_extras). eval.py's run-end banner reads
        cache_types.summarize_judge_cache() instead of the deleted base.py one.
        """
        from cache_types import JudgeOutcome, _JUDGE_CACHE_STATS
        label = JudgeOutcome.label_for(judge_spec)
        if label is None:
            return None  # rule-based: never persisted
        jd = self._judge_dir(qid, idx, label, rollout_idx)
        gp = jd / "grade.json"
        cp = jd / "config.json"
        rp = jd / "result.json"
        if not (gp.exists() and cp.exists()):
            return None
        stored = json.loads(cp.read_text(encoding="utf-8"))

        if stored == judge_spec:
            _JUDGE_CACHE_STATS["exact_hits"] += 1
        else:
            shared = set(stored) & set(judge_spec)
            conflicts = {k: (stored[k], judge_spec[k]) for k in shared
                         if stored[k] != judge_spec[k]}
            if conflicts:
                raise RuntimeError(
                    f"Judge config value conflict at {jd}.\n"
                    f"  Conflicting keys (stored vs requested): {conflicts}\n"
                    f"  Stored:    {stored}\n"
                    f"  Requested: {judge_spec}\n"
                    f"Wipe the bundle or rename the label before re-running."
                )
            only_stored = sorted(set(stored) - set(judge_spec))
            if only_stored:
                raise RuntimeError(
                    f"Judge cache spec narrowing at {jd}.\n"
                    f"  Stored has keys absent from requested: {only_stored}\n"
                    f"  Stored:    {stored}\n"
                    f"  Requested: {judge_spec}\n"
                    f"Cached verdict was produced under a non-default spec; "
                    f"refusing to reuse it for a less-specific request."
                )
            # Only legitimate non-exact path: stored ⊂ requested (schema evolution).
            only_requested = sorted(set(judge_spec) - set(stored))
            _JUDGE_CACHE_STATS["best_effort_hits"] += 1
            _JUDGE_CACHE_STATS["best_effort_extras"].update(only_requested)

        grade = json.loads(gp.read_text(encoding="utf-8"))
        return JudgeOutcome(
            is_correct=grade["is_correct"],
            # Post-Issue-R1 grade.json migration: cost_usd is required in every
            # bundle on disk. Strict access (no .get fallback) — missing key
            # would mean the migration script was skipped for that bundle and
            # we want to fail loudly, not silently return 0.
            cost_usd=grade["cost_usd"],
            judge_spec_snapshot=stored,
            input_md=(jd / "input.md").read_text(encoding="utf-8") if (jd / "input.md").exists() else "",
            output_md=(jd / "output.md").read_text(encoding="utf-8") if (jd / "output.md").exists() else "",
            result_dict=json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {},
        )
```

Add `import json` at the top of `specs.py` if not already there.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_explore_variant_methods.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add Experiment/core_code/methods/specs.py Experiment/core_code/tests/test_explore_variant_methods.py
git commit -m "feat(ExploreVariant): add internal cache I/O helpers (_explore_dir, _judge_dir, _has_explore, _load_explore, _load_judge)

Pure path construction + JSON I/O. No public-facing change yet.
Strict judge config identity check on _load_judge — refuse silent
mismatches."
```

---

### Task 6: `ExploreVariant.get_exploration` — single-by-idx with mandatory `generate_fn`

**Files:**
- Modify: `Experiment/core_code/methods/specs.py`
- Test: `Experiment/core_code/tests/test_explore_variant_methods.py`

- [ ] **Step 1: Failing test for cache-hit path**

```python
@pytest.mark.asyncio
async def test_get_exploration_cache_hit_does_not_call_generate(tmp_path):
    from cache_types import Exploration
    v = _make_variant(tmp_path)
    cached = Exploration(
        qid="q1", idx=1, rollout_idx=None,
        answer="cached", trajectory="t", cost_usd=0.0, model="m",
    )
    cached.persist(v._explore_dir("q1", 1))

    called = False
    async def gen():
        nonlocal called
        called = True
        raise AssertionError("generate_fn should not be called on cache hit")

    exp = await v.get_exploration("q1", 1, generate_fn=gen)
    assert exp.answer == "cached"
    assert called is False
```

- [ ] **Step 2: Failing test for cache-miss path**

```python
@pytest.mark.asyncio
async def test_get_exploration_cache_miss_calls_generate_and_persists(tmp_path):
    from cache_types import Exploration
    v = _make_variant(tmp_path)

    async def gen():
        return Exploration(
            qid="q1", idx=1, rollout_idx=None,
            answer="fresh", trajectory="generated", cost_usd=0.05, model="m",
        )

    exp = await v.get_exploration("q1", 1, generate_fn=gen)
    assert exp.answer == "fresh"
    assert (v._explore_dir("q1", 1) / "result.json").exists()  # persisted
```

- [ ] **Step 3: Run, verify fail**

Run: `pytest tests/test_explore_variant_methods.py -v`
Expected: FAIL — `get_exploration` doesn't exist.

- [ ] **Step 4: Implement `get_exploration` (no grader yet)**

Add to `ExploreVariant` in `specs.py`:

```python
    async def get_exploration(
        self,
        qid: str,
        idx: int,
        *,
        rollout_idx: int | None = None,
        generate_fn,                                  # async () -> Exploration; mandatory
        grader=None,                                  # async (answer, qid) -> JudgeOutcome, optional
    ) -> "Exploration":
        """Cache hit → return cached Exploration. Cache miss → call generate_fn,
        persist, return. Optional grader → also produce + persist verdict.

        Always returns Exploration. Never None. Cache miss with a generate_fn
        that fails (raises) propagates that failure — there is no silent
        degradation.
        """
        # Layer 1: explore cache
        exp = self._load_explore(qid, idx, rollout_idx)
        if exp is None:
            exp = await generate_fn()
            assert isinstance(exp, Exploration), (
                f"generate_fn must return Exploration, got {type(exp).__name__}"
            )
            # generate_fn returns a possibly-incomplete Exploration; backfill identity.
            exp.qid, exp.idx, exp.rollout_idx = qid, idx, rollout_idx
            exp.persist(self._explore_dir(qid, idx, rollout_idx))

        # Layer 2: judge (optional, layered on top)
        if grader is not None:
            cached_outcome = self._load_judge(qid, idx, grader.judge_spec, rollout_idx)
            if cached_outcome is not None:
                exp.verdict = cached_outcome
            else:
                outcome = await grader(exp.answer, qid)
                exp.verdict = outcome
                if outcome.label is not None:
                    # Rule-based grading has no trace to persist.
                    outcome.persist(self._judge_dir(qid, idx, outcome.label, rollout_idx))
        return exp
```

Add a `from cache_types import Exploration` import at top of `specs.py`.

- [ ] **Step 5: Run tests, verify pass**

Run: `pytest tests/test_explore_variant_methods.py -v`
Expected: PASS.

- [ ] **Step 6: Add test for `grader` layer**

```python
@pytest.mark.asyncio
async def test_get_exploration_with_grader_attaches_verdict(tmp_path):
    from cache_types import Exploration, JudgeOutcome

    class FakeGrader:
        judge_spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
        async def __call__(self, answer, qid):
            return JudgeOutcome(
                is_correct=True, cost_usd=0.001,
                judge_spec_snapshot=self.judge_spec,
                input_md="i", output_md="o", result_dict={"correct": True},
            )

    v = _make_variant(tmp_path)
    async def gen():
        return Exploration(qid="q1", idx=1, rollout_idx=None,
                           answer="42", trajectory="t", cost_usd=0.0, model="m")

    exp = await v.get_exploration("q1", 1, generate_fn=gen, grader=FakeGrader())
    assert exp.verdict is not None
    assert exp.verdict.is_correct is True
    # Judge bundle persisted
    assert (v._judge_dir("q1", 1, exp.verdict.label) / "grade.json").exists()


@pytest.mark.asyncio
async def test_get_exploration_with_rule_based_grader_skips_judge_persist(tmp_path):
    from cache_types import Exploration, JudgeOutcome

    class RuleBasedGrader:
        judge_spec = None  # rule-based
        async def __call__(self, answer, qid):
            return JudgeOutcome(is_correct=True, cost_usd=0.0,
                                judge_spec_snapshot=None, input_md="", output_md="",
                                result_dict={"correct": True})

    v = _make_variant(tmp_path)
    async def gen():
        return Exploration(qid="q1", idx=1, rollout_idx=None,
                           answer="A", trajectory="", cost_usd=0.0, model="m")

    exp = await v.get_exploration("q1", 1, generate_fn=gen, grader=RuleBasedGrader())
    assert exp.verdict is not None
    assert exp.verdict.is_correct is True
    # No judge bundle persisted (rule-based)
    assert not (v._explore_dir("q1", 1) / "judges").exists()
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_explore_variant_methods.py -v`
Expected: PASS (all tests).

- [ ] **Step 8: Commit**

```bash
git add Experiment/core_code/methods/specs.py Experiment/core_code/tests/test_explore_variant_methods.py
git commit -m "feat(ExploreVariant): add get_exploration(qid, idx, generate_fn, grader=None)

Single-by-idx accessor. Cache hit returns cached; miss calls generate_fn
and persists. Optional grader layers verdict on top with separate cache.
Rule-based grading skips judge bundle persistence. Always returns
Exploration — never None, never silent degradation."
```

---

### Task 7: `ExploreVariant.get_all_explorations` — list-by-qid, read-only

**Files:**
- Modify: `Experiment/core_code/methods/specs.py`
- Test: `Experiment/core_code/tests/test_explore_variant_methods.py`

- [ ] **Step 1: Failing tests for list semantics**

```python
@pytest.mark.asyncio
async def test_get_all_explorations_empty(tmp_path):
    v = _make_variant(tmp_path)
    explorations = await v.get_all_explorations("nonexistent_qid")
    assert explorations == []


@pytest.mark.asyncio
async def test_get_all_explorations_sorted(tmp_path):
    from cache_types import Exploration
    v = _make_variant(tmp_path)
    for i in [3, 1, 2]:
        Exploration(qid="q1", idx=i, rollout_idx=None,
                    answer=f"ans{i}", trajectory="", cost_usd=0.0, model="m"
                   ).persist(v._explore_dir("q1", i))
    explorations = await v.get_all_explorations("q1")
    assert [e.idx for e in explorations] == [1, 2, 3]
    assert [e.answer for e in explorations] == ["ans1", "ans2", "ans3"]


@pytest.mark.asyncio
async def test_get_all_explorations_with_grader_attaches_verdicts(tmp_path):
    from cache_types import Exploration, JudgeOutcome

    class FakeGrader:
        judge_spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
        async def __call__(self, answer, qid):
            return JudgeOutcome(is_correct=(answer == "ans1"), cost_usd=0.001,
                                judge_spec_snapshot=self.judge_spec,
                                input_md="i", output_md="o",
                                result_dict={"correct": True})

    v = _make_variant(tmp_path)
    for i in [1, 2]:
        Exploration(qid="q1", idx=i, rollout_idx=None,
                    answer=f"ans{i}", trajectory="", cost_usd=0.0, model="m"
                   ).persist(v._explore_dir("q1", i))

    explorations = await v.get_all_explorations("q1", grader=FakeGrader())
    assert len(explorations) == 2
    assert explorations[0].verdict.is_correct is True   # ans1
    assert explorations[1].verdict.is_correct is False  # ans2
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/test_explore_variant_methods.py -v`
Expected: FAIL — method doesn't exist.

- [ ] **Step 3: Implement `get_all_explorations`**

Add to `ExploreVariant` in `specs.py`:

```python
    async def get_all_explorations(
        self,
        qid: str,
        *,
        rollout_idx: int | None = None,
        grader=None,
    ) -> list["Exploration"]:
        """Pure read of cached explorations under (qid, rollout_idx).

        Walks the directory, returns Explorations in ascending idx order.
        Empty list is a legitimate result, not an error.

        If grader provided, attach verdict to each Exploration (using judge
        cache when present; calling grader on miss).
        """
        from cache_types import Exploration

        base = self.cache_dir / qid
        if rollout_idx is not None:
            base = base / f"rollout_{rollout_idx}"
        if not base.exists():
            return []
        # Collect explore_<idx> subdirs that have result.json
        found: list[int] = []
        for p in base.iterdir():
            if not (p.is_dir() and p.name.startswith("explore_")):
                continue
            if not (p / "result.json").exists():
                continue
            try:
                found.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                continue
        found.sort()

        explorations: list[Exploration] = []
        for idx in found:
            exp = self._load_explore(qid, idx, rollout_idx)
            assert exp is not None  # we just verified result.json exists
            if grader is not None:
                cached_outcome = self._load_judge(qid, idx, grader.judge_spec, rollout_idx)
                if cached_outcome is not None:
                    exp.verdict = cached_outcome
                else:
                    outcome = await grader(exp.answer, qid)
                    exp.verdict = outcome
                    if outcome.label is not None:
                        outcome.persist(self._judge_dir(qid, idx, outcome.label, rollout_idx))
            explorations.append(exp)
        return explorations
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_explore_variant_methods.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Experiment/core_code/methods/specs.py Experiment/core_code/tests/test_explore_variant_methods.py
git commit -m "feat(ExploreVariant): add get_all_explorations(qid, grader=None) -> list

Pure read; never generates. Returns explorations in ascending idx order;
empty list is legitimate. Optional grader attaches verdicts using judge
cache + on-miss grader call. For rerank/standalone-integrator/eval grading
phase consumers."
```

---

## Phase 3 — Migrate Call Sites

### Task 8: Migrate `tts_agent.run_explore` to use `variant.get_exploration`

**Files:**
- Modify: `Experiment/core_code/methods/tts_agent.py:198-244`

- [ ] **Step 1: Read existing `run_explore`**

Run: `sed -n '198,244p' Experiment/core_code/methods/tts_agent.py`

- [ ] **Step 2: Replace `caller(...)` with `variant.get_exploration(generate_fn=...)`**

Rewrite the body of `run_explore`. The `generate_fn` closure captures backend.call_sub_model + prompt + schema:

```python
async def run_explore(ctx: SolveContext, spec, variants_by_label: dict[str, ExploreVariant], label: str) -> str:
    multi = len(spec.explore) > 1
    if ctx.state.explore.is_exhausted:
        return f"Explore quota exhausted ({ctx.state.explore.max_explores} explores already used). ..."
    if multi and ctx.state.explore.variant_exhausted(label):
        ...
    variant = variants_by_label[label]
    in_idx = ctx.state.explore.variant_call_counts.get(label, 0) + 1

    user_msg = ctx.benchmark.build_explorer_message(ctx.state.problem)
    explorer_system_prompt = ctx.benchmark.get_explorer_system_prompt(variant.model.backend)
    explore_schema = ctx.benchmark.get_explore_schema()

    async def generate_fn():
        from cache_types import Exploration
        backend_mod = import_module(f"backends.{variant.model.backend}")
        result, traj, cost, usage = await backend_mod.call_sub_model(
            explorer_system_prompt, user_msg, ctx.image_data_url,
            variant.model.model, explore_schema,
            writer=TrajectoryWriter.noop(),
            budget_tokens=variant.model.budget_tokens,
            effort=variant.model.effort,
            sampling=(variant.model.vllm_sampling.model_dump()
                      if variant.model.vllm_sampling is not None else None),
            provider_order=variant.model.openrouter_provider_order,
            provider_allow_fallbacks=variant.model.openrouter_provider_allow_fallbacks,
        )
        return Exploration(
            qid=ctx.question_id, idx=in_idx, rollout_idx=ctx.rollout_idx,
            answer=ctx.benchmark.get_answer_from_explore(result),
            trajectory=traj,
            cost_usd=cost,
            model=variant.model.model,
            timed_out=result.get("timed_out", False),
            extra={"usage": usage, **{k: v for k, v in result.items() if k != "answer"}},
            system_prompt=explorer_system_prompt,
            user_message=user_msg,
        )

    exp = await variant.get_exploration(
        ctx.question_id, in_idx,
        rollout_idx=ctx.rollout_idx,
        generate_fn=generate_fn,
        # No grader here — orchestrator does not see gold during agent loop.
    )

    # Mirror to traj_dir for this run's working record.
    if ctx.traj_dir is not None:
        exp.persist(ctx.traj_dir / f"explore_{in_idx}")

    return process_explore_result(
        ctx, exp,
        label=label,
        model_label=label if multi else "",
        extra_budget_text=_budget_status_text(spec, ctx),
    )
```

(Engineer should adapt `process_explore_result` to accept `Exploration` instead of `(result, cost, usage)`. Update its body to read `exp.answer / exp.cost_usd / exp.extra` etc.)

- [ ] **Step 3: Update `process_explore_result` signature (lines 96-156)**

```python
def process_explore_result(
    ctx: SolveContext,
    exp: "Exploration",
    *,
    label: str,
    model_label: str,
    extra_budget_text: str,
) -> str:
    if exp.timed_out:
        ctx.writer.write_explore_timeout()
        ...
    ctx.cost.add(exp.cost_usd, exp.extra.get("usage", {}), component="explorer")
    state.candidates.append(Candidate(
        answer=exp.answer,
        ...
        cost_usd=exp.cost_usd,
    ))
    ...
```

- [ ] **Step 4: Update caller of `run_explore` to pass `variants_by_label` dict**

In `tts_agent.solve` (around line 471-480), build:

```python
variants_by_label: dict[str, ExploreVariant] = {v.label: v for v in spec.explore}
```

And pass `variants_by_label` to `_run_orchestrator` / `run_explore` instead of the old `variant_callers`.

- [ ] **Step 5: Run existing tts_agent tests**

Run: `pytest Experiment/core_code/tests/ -v -k "tts_agent or explore_variant" --no-header 2>&1 | tail -30`
Expected: green or only-pre-existing failures unrelated to this refactor.

- [ ] **Step 6: Commit**

```bash
git add Experiment/core_code/methods/tts_agent.py
git commit -m "refactor(tts_agent): run_explore uses ExploreVariant.get_exploration

Orchestrator agent loop does not pass a grader (cannot see gold during
solve). Generate_fn closure captures backend + prompts + schema; ExploreVariant
handles cache + persistence. Trajectory mirror to traj_dir is the caller's
responsibility (Exploration.persist into run_dir/trajectories/<qid>/explore_N/)."
```

---

### Task 9: Migrate `self_refine` / `socratic_self_refine` / `budget_forcing`

**Files:**
- Modify: `Experiment/core_code/methods/self_refine.py:178-281`
- Modify: `Experiment/core_code/methods/socratic_self_refine.py:317-420`
- Modify: `Experiment/core_code/methods/budget_forcing.py:64-69`

For each: replace `ctx.call_sub_model(..., cache_key="explore_<i>")` with `variant.get_exploration(qid, i, generate_fn=...)`. Keep `feedback_<i>` cache key path unchanged for now — Phase 4 will handle that decision.

- [ ] **Step 1: Audit feedback_<i> cache uses**

Run: `grep -n "feedback_\|cache_key=f" Experiment/core_code/methods/self_refine.py Experiment/core_code/methods/socratic_self_refine.py`

The `feedback_<i>` keys are auxiliary to the refine cycle. They're written via `make_sub_model_caller` and live alongside the explore cache. Decision: **keep the auxiliary `feedback_*` cache writes via a thin shim that uses `backend.call_sub_model` + `Exploration.persist` directly** — feedback is not an Exploration semantically (it's an internal refine artifact), but the persistence mechanics are the same.

- [ ] **Step 2: Migrate self_refine.py**

Replace each `ctx.call_sub_model(..., cache_key="explore_X")` with the variant.get_exploration pattern. For `feedback_<i>`, keep a direct backend call but write the trace to `<cache_dir>/<qid>/feedback_<i>/`:

```python
# self_refine.py — explore_1 (initial draft)
async def gen_initial():
    from cache_types import Exploration
    backend_mod = import_module(f"backends.{variant.model.backend}")
    result, traj, cost, usage = await backend_mod.call_sub_model(...)
    return Exploration(
        qid=question_id, idx=1, rollout_idx=rollout_idx,
        answer=ctx.benchmark.get_answer_from_explore(result),
        trajectory=traj, cost_usd=cost, model=variant.model.model,
        ...
    )
exp1 = await variant.get_exploration(question_id, 1, rollout_idx=rollout_idx, generate_fn=gen_initial)
ctx.cost.add(exp1.cost_usd, ...)
```

Repeat for explore_2..N (refine outputs). For feedback_<i>, retain a thin direct-backend call with manual disk write to `<variant.cache_dir>/<qid>/feedback_<i>/`. Add a comment marking feedback as "auxiliary, not an Exploration".

- [ ] **Step 3: Same for socratic_self_refine.py**

Mirror Task 9 Step 2.

- [ ] **Step 4: Same for budget_forcing.py (only `explore_<i>` cache_keys)**

```python
# budget_forcing.py
exp_i = await variant.get_exploration(question_id, i, rollout_idx=rollout_idx, generate_fn=...)
```

- [ ] **Step 5: Run tests for these 3 methods**

Run: `pytest Experiment/core_code/tests/ -v -k "self_refine or socratic or budget_forcing" --no-header 2>&1 | tail -30`

- [ ] **Step 6: Commit**

```bash
git add Experiment/core_code/methods/self_refine.py Experiment/core_code/methods/socratic_self_refine.py Experiment/core_code/methods/budget_forcing.py
git commit -m "refactor(self_refine + socratic + budget_forcing): migrate explore-cache calls to ExploreVariant.get_exploration

Auxiliary feedback_<i> writes retained as direct backend calls (they are
not Explorations semantically). All explore_<i> reads/writes route through
ExploreVariant."
```

---

### Task 10: Migrate `precache_explores.py`

**Files:**
- Modify: `Experiment/core_code/precache_explores.py:39, 127-176`

- [ ] **Step 1: Replace `make_sub_model_caller` use with `variant.get_exploration`**

```python
# precache_explores.py
# Drop: from methods.base import make_sub_model_caller

# Inside per-question loop:
for explore_idx in pending_idxes:
    async def gen():
        from cache_types import Exploration
        backend_mod = import_module(f"backends.{variant.model.backend}")
        result, traj, cost, usage = await backend_mod.call_sub_model(
            explorer_system_prompt, user_msg, image_data_url,
            variant.model.model, explore_schema,
            writer=TrajectoryWriter.noop(),
            budget_tokens=variant.model.budget_tokens,
            effort=variant.model.effort,
            sampling=(...),
            provider_order=variant.model.openrouter_provider_order,
            provider_allow_fallbacks=variant.model.openrouter_provider_allow_fallbacks,
        )
        return Exploration(
            qid=qid, idx=explore_idx, rollout_idx=None,
            answer=benchmark.get_answer_from_explore(result),
            trajectory=traj, cost_usd=cost, model=variant.model.model,
            timed_out=result.get("timed_out", False),
            extra={"usage": usage, **{k: v for k, v in result.items() if k != "answer"}},
            system_prompt=explorer_system_prompt,
            user_message=user_msg,
        )

    exp = await variant.get_exploration(
        qid, explore_idx,
        # rollout_idx is None for precache (precache is always K=1 lane)
        generate_fn=gen,
        # No grader: precache doesn't grade.
    )
    progress_logger.record_task(qid, explore_idx, exp)  # adapt PrecacheLogger if needed
```

- [ ] **Step 2: Run smoke**

Run on a tiny yaml: `python precache_explores.py --config tests/openrouter_e2e_precache.yaml` (if exists). Expect it to either complete or fail with the same behavior as before (this is a refactor, not a fix).

- [ ] **Step 3: Commit**

```bash
git add Experiment/core_code/precache_explores.py
git commit -m "refactor(precache): use ExploreVariant.get_exploration

Drops make_sub_model_caller dependency. precache and eval/orchestrator
now share the same single entry point for explore generation + persistence."
```

---

### Task 11: Replace `eval.py` grading-phase helpers with `get_all_explorations`

**Files:**
- Modify: `Experiment/core_code/eval.py:172-311` (delete `_grade_with_cache`, `_grade_cached_explores`, `_grade_question_explores`, `_grade_question_explores_multi`)
- Modify: `Experiment/core_code/eval.py:546-700` (`process_question` per-question grading section)

- [ ] **Step 1: Construct the grader closure in `async_main`**

```python
# eval.py:async_main, after benchmark instantiation
rows_by_id = {benchmark.get_id(r): r for r in filtered}

class _Grader:
    def __init__(self, benchmark, rows_by_id, judge_spec):
        self.benchmark = benchmark
        self.rows_by_id = rows_by_id
        self.judge_spec = judge_spec

    async def __call__(self, answer: str, qid: str):
        row = self.rows_by_id[qid]
        return await self.benchmark.grade(
            predicted=answer,
            gold=self.benchmark.get_answer(row),
            question=self.benchmark.get_question(row),
            row=row,
            backend="",  # legacy; benchmarks no longer need it
        )

grader = _Grader(benchmark, rows_by_id, benchmark.judge_spec)
```

- [ ] **Step 2: Replace per-question grading in `process_question`**

```python
# Replace lines around 597-621 (_grade_question_explores call) with:
question_cands_explorations = await variant.get_all_explorations(
    qid, rollout_idx=rollout_idx, grader=grader,
)
question_cands = [
    (benchmark.normalize_answer(e.answer),
     e.verdict.is_correct if e.verdict else False,
     e.cost_usd)
    for e in question_cands_explorations
]
qbon_jc = sum((e.verdict.cost_usd if e.verdict else 0.0)
              for e in question_cands_explorations)

# For multi-variant tts-agent:
pm_cands = None
if cache_dirs_multi:
    pm_cands = {}
    pm_jc = 0.0
    for label, v in variants_by_label.items():
        explorations = await v.get_all_explorations(qid, rollout_idx=rollout_idx, grader=grader)
        pm_cands[label] = [
            (benchmark.normalize_answer(e.answer),
             e.verdict.is_correct if e.verdict else False,
             e.cost_usd)
            for e in explorations
        ]
        pm_jc += sum((e.verdict.cost_usd if e.verdict else 0.0) for e in explorations)
    qbon_jc += pm_jc
```

- [ ] **Step 3: Replace integrated-answer grading**

Replace `is_correct, judge_cost_1 = await _grade_with_cache(...)` (eval.py:592-595) with:

```python
# Final-answer grading: lives in run_dir, not cache_dir.
grade_target_dir = _rollout_subpath(run_logger.run_dir / "grading", qid, rollout_idx)
final_label = JudgeOutcome.label_for(benchmark.judge_spec)
cached_grade_path = (grade_target_dir / "judges" / final_label / "grade.json"
                     if final_label else None)
if cached_grade_path and cached_grade_path.exists():
    grade_data = json.loads(cached_grade_path.read_text())
    is_correct = grade_data["is_correct"]
    judge_cost_1 = 0.0
else:
    final_outcome = await grader(predicted, qid)
    is_correct = final_outcome.is_correct
    judge_cost_1 = final_outcome.cost_usd
    if final_outcome.label is not None:
        final_outcome.persist(grade_target_dir / "judges" / final_outcome.label)
```

- [ ] **Step 4: Delete dead helpers**

Remove from `eval.py`:
- `_grade_with_cache` (lines 172-218)
- `_grade_cached_explores` (lines 221-254)
- `_grade_question_explores` (lines 257-288)
- `_grade_question_explores_multi` (lines 291-311)

- [ ] **Step 5: Run rbenchv smoke (if achievable without launch)**

Run: `python eval.py --config Experiment/core_code/scripts/rbenchv/sonnet/rbenchv_sonnet_delegated.yaml --num 1` against a yaml with `cache_only: true` (still using current default behavior — Task 13 adds the yaml field).

If this requires Task 13 first, skip and note dependency.

- [ ] **Step 6: Commit**

```bash
git add Experiment/core_code/eval.py
git commit -m "refactor(eval.py): grading phase routes through ExploreVariant.get_all_explorations

Deletes _grade_with_cache / _grade_cached_explores / _grade_question_explores /
_grade_question_explores_multi (4 helpers, ~150 lines).
Constructs Grader closure in setup; ExploreVariant owns explore-side judge
bundle persistence; eval owns run-side grading bundles for the integrated
answer."
```

---

## Phase 4 — Drop Integrate Cache

### Task 12: Drop `RoleSlot.cache_dir`; integrate goes direct to backend

**Files:**
- Modify: `Experiment/core_code/methods/specs.py:130-138`
- Modify: `Experiment/core_code/methods/tts_agent.py:247-280`
- Modify: `Experiment/core_code/methods/standalone_integrator.py:60-95`

- [ ] **Step 1: Drop `cache_dir` field from `RoleSlot`**

```python
class RoleSlot(BaseModel):
    """A model invocation. Used by integrate role.
    No cache: integrate input is candidates content, which is not encoded
    in any cache_key; caching by (qid, count) is content-blind and unsafe."""
    model_config = {"extra": "forbid"}
    model: ModelConfig
    # cache_dir removed (was a leaky abstraction; see plan 2026-05-05)
```

- [ ] **Step 2: Migrate yaml configs (drop `integrate.cache_dir:` lines)**

Run-once script:

```python
# Experiment/core_code/tools/migrate_yaml_drop_integrate_cache_dir.py
import yaml
from pathlib import Path
import sys

ROOT = Path("Experiment/core_code/scripts")
for yp in ROOT.rglob("*.yaml"):
    text = yp.read_text()
    data = yaml.safe_load(text) or {}
    if isinstance(data.get("method"), dict):
        integrate = data["method"].get("integrate")
        if isinstance(integrate, dict) and "cache_dir" in integrate:
            del integrate["cache_dir"]
            yp.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
            print(f"Migrated: {yp}")
```

Run: `python Experiment/core_code/tools/migrate_yaml_drop_integrate_cache_dir.py`

- [ ] **Step 3: Replace tts_agent integrate cache call with direct backend call**

In `methods/tts_agent.py:run_integrate` (line 247-285):

```python
async def run_integrate(ctx: SolveContext, spec) -> str:
    assert spec.integrate is not None
    state = ctx.state
    assert state.candidates

    integrator_system_prompt = ctx.benchmark.get_integrator_system_prompt(spec.integrate.model.backend)
    integrate_schema = ctx.benchmark.get_integrate_schema()
    user_msg = ctx.benchmark.build_integrator_message(state.problem, state.candidates)

    # Direct backend call. Trajectory written to run_dir/trajectories only.
    backend_mod = import_module(f"backends.{spec.integrate.model.backend}")
    result, traj, cost_usd, usage = await backend_mod.call_sub_model(
        integrator_system_prompt, user_msg, ctx.image_data_url,
        spec.integrate.model.model, integrate_schema,
        writer=ctx.writer,
        budget_tokens=spec.integrate.model.budget_tokens,
        effort=spec.integrate.model.effort,
        sampling=(spec.integrate.model.vllm_sampling.model_dump()
                  if spec.integrate.model.vllm_sampling is not None else None),
        provider_order=spec.integrate.model.openrouter_provider_order,
        provider_allow_fallbacks=spec.integrate.model.openrouter_provider_allow_fallbacks,
    )
    ctx.cost.add(cost_usd, usage, component="integrator")
    final_answer = ctx.benchmark.get_answer_from_integrate(result)
    state.final_answer = final_answer
    return f"Integrated answer: {final_answer}"
```

- [ ] **Step 4: Same for `standalone_integrator.py:67-95`**

```python
backend_mod = import_module(f"backends.{spec.integrate.model.backend}")
result, traj, cost_usd, usage = await backend_mod.call_sub_model(...)
# (no caching; trajectory still written to ctx.writer)
```

- [ ] **Step 5: Update test_explore_variant or test for RoleSlot**

If existing tests reference `RoleSlot(cache_dir=...)`, update them to drop that field.

- [ ] **Step 6: Run tests**

Run: `pytest Experiment/core_code/tests/ -v --no-header 2>&1 | tail -20`

- [ ] **Step 7: Commit**

```bash
git add Experiment/core_code/methods/specs.py Experiment/core_code/methods/tts_agent.py Experiment/core_code/methods/standalone_integrator.py Experiment/core_code/scripts Experiment/core_code/tools/migrate_yaml_drop_integrate_cache_dir.py
git commit -m "refactor(integrate): drop cache (key was content-blind)

cache_key=integrate_<count> only encodes candidate count, not candidate
content or integrate model identity. Cache silently lied when integrate
model changed or candidates differed. Re-running integrate is cheap
(1 call/question vs 8 explore calls); resume already skips done records.
RoleSlot.cache_dir removed; all yaml integrate.cache_dir: lines stripped
by migration script."
```

---

### Task 13: Drop `cache_only` `integrate_*` exemption + `make_sub_model_caller`

**Files:**
- Modify: `Experiment/core_code/methods/base.py:285-352`

- [ ] **Step 1: Verify no remaining callers**

Run: `grep -rn "make_sub_model_caller" Experiment/core_code/ --include="*.py"`
Expected: zero non-import references after Tasks 8-12.

- [ ] **Step 2: Delete `make_sub_model_caller` function**

Delete `methods/base.py:263-352` entirely. Keep `save_sub_model_input` / `save_sub_model_result` (they may still be used as helpers for direct backend calls).

- [ ] **Step 3: Audit `infra.cache_only` usage**

Run: `grep -rn "cache_only" Experiment/core_code/ --include="*.py" | grep -v tests/`
Expected: usage shifted to either `EvalConfig.cache_only` (yaml) or removed.

- [ ] **Step 4: Run all tests**

Run: `pytest Experiment/core_code/tests/ -v --no-header 2>&1 | tail -20`

- [ ] **Step 5: Commit**

```bash
git add Experiment/core_code/methods/base.py
git commit -m "refactor: delete make_sub_model_caller (collapsed into ExploreVariant)

The cache_only/integrate_* exemption disappears with it. cache_only is now
expressed at the call site: caller passes a generate_fn that raises if
cache_only=True is requested on a miss."
```

---

## Phase 5 — Resume + Record Schema

### Task 14: Extend record schema with `per_variant_candidates`

**Files:**
- Modify: `Experiment/core_code/eval.py:625-650` (record write site)

Decision (2026-05-05, user binding): NO `schema_version` field. Record schema is identified by the presence/absence of `per_variant_candidates` itself. Project policy (CLAUDE.md): "DO NOT USE VERSION TO DENOTE CHANGES... ALWAYS OVERWRITE THE PREVIOUS VERSION". Migration path (Reality Check Issue R3) backfills the field once; after migration, all records carry it; new code reads it directly with no version branching.

- [ ] **Step 1: Add field to record dict**

```python
# eval.py:process_question, in the record = {...} block (around line 625)
record = {
    "id": qid,
    ...
    "explore_candidates": [
        {"normalized_answer": na, "is_correct": ic, "cost_usd": c}
        for na, ic, c in question_cands
    ],
    "per_variant_candidates": (    # only for multi-variant tts-agent
        {label: [{"normalized_answer": na, "is_correct": ic, "cost_usd": c}
                 for na, ic, c in cands]
         for label, cands in pm_cands.items()}
        if pm_cands else None
    ),
    ...
}
```

- [ ] **Step 2: Commit**

```bash
git add Experiment/core_code/eval.py
git commit -m "feat(eval.py): add per_variant_candidates to record schema

Multi-variant resume reconstructs per-model arrays directly from this
field instead of re-walking cache_dir on every resume. No version field —
field presence/absence is the schema indicator."
```

---

### Task 15: Replace `grade_done_record` with pure deserialize

**Files:**
- Modify: `Experiment/core_code/eval.py:425-541`

- [ ] **Step 1: Delete `grade_done_record` function**

Delete `eval.py:472-518`.
Delete the `grade_sem`, `grade_done_count`, `grade_done_lock`, `_bundle_cached`, `total_to_grade` book-keeping (lines 425-470).
Delete the `results = await asyncio.gather(...)` and the subsequent `for rec, gr in zip(...)` loop (lines 520-541).

- [ ] **Step 2: Replace with pure deserialize**

```python
# eval.py:async_main, in place of the deleted block
for i, rec in enumerate(done_records):
    if str(rec.get("predicted_answer", "")).startswith("ERROR:"):
        errors += 1
    total_cost += rec.get("cost_usd", 0.0)
    explore_counts.append(rec.get("num_explores", 0))
    for comp, c in rec.get("cost_by_component", {}).items():
        total_cost_by_component[comp] = total_cost_by_component.get(comp, 0.0) + c

    cands = [(c["normalized_answer"], c["is_correct"], c["cost_usd"])
             for c in rec.get("explore_candidates", [])]
    all_candidates[i] = cands
    all_integrated[i] = (benchmark.normalize_answer(rec["predicted_answer"]), rec["is_correct"])

    if rec["is_correct"]:
        correct += 1
    if cands and cands[0][1]:
        first_correct += 1

    # Multi-variant per_model: read directly from record. Missing field on
    # an old record raises KeyError — that IS the loud signal to run the
    # one-shot migration script (Reality Check Issue R3). No version field,
    # no .get fallback, no silent default.
    if cache_dirs_multi:
        pvc = rec["per_variant_candidates"]   # KeyError on unmigrated record
        for label, cand_list in pvc.items():
            per_model_all_candidates[label][i] = [
                (c["normalized_answer"], c["is_correct"], c["cost_usd"])
                for c in cand_list
            ]
    all_records.append(rec)
```

- [ ] **Step 3: Test resume on a small results.jsonl**

Hand-craft a minimal results.jsonl with 2 records (each carrying `per_variant_candidates`). Run a no-op resume — should print correct stats without any judge calls.

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/eval.py
git commit -m "refactor(eval.py): resume done-records is pure deserialize, no grader calls

grade_done_record was 99% recomputing data already in record. The 1% it
did add (per_model dimension for multi-variant) is now stored in record
as per_variant_candidates directly. Eliminates ~80 lines of resume-time
judge replay. Old records that predate this field need a one-shot
migration (Reality Check Issue R3)."
```

---

## Phase 6 — `cache_only` YAML + rollout_idx Cache Path

### Task 16: Add `EvalConfig.cache_only` yaml field + preflight skip logic

**Files:**
- Modify: `Experiment/core_code/eval.py:112-136`
- Modify: `Experiment/core_code/eval.py:819-848`
- Modify: `Experiment/core_code/methods/registry.py:57-85`

- [ ] **Step 1: Add field to `EvalConfig`**

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
    judge_max_retries: int = 3
    resume: str | None = None
    log_dir: str = "logs"
    # When set: True forces strict cache (caller's generate_fn raises on
    # miss); False permits explore cache miss → API → writeback (online
    # cache) AND skips banner-time preflight. None inherits MethodConfig
    # default (current behavior).
    cache_only: bool | None = None
```

- [ ] **Step 2: Update preflight call site (eval.py:819)**

```python
# eval.py:async_main
if cfg.cache_only is not False:
    method.preflight(filtered, spec, cfg.num, benchmark)
```

- [ ] **Step 3: Update `MethodConfig.preflight` signature (registry.py:57)**

Change signature from `(rows, cache_dir, num_explores, num, benchmark)` to `(rows, spec, num, benchmark)`. Inside, walk variants from spec and use `variant._has_explore(qid, idx)`:

```python
def preflight(self, rows, spec, num, benchmark) -> None:
    if not self.pre_flight_check:
        return
    variants = spec.explore if isinstance(spec.explore, list) else [spec.explore]
    qids = [benchmark.get_id(r) for r in (rows if num is None else rows[:num])]
    missing = [
        (qid, v.label, idx)
        for qid in qids
        for v in variants
        for idx in range(1, v.num_explores + 1)
        if not v._has_explore(qid, idx)
    ]
    if missing:
        sample = ", ".join(f"({q}/{lbl}, explore_{i})" for q, lbl, i in missing[:10])
        raise AssertionError(
            f"Cache pre-flight FAILED: {len(missing)} missing entries. "
            f"First 10: {sample}"
        )
    logger.info(f"Cache pre-flight OK: {sum(v.num_explores for v in variants)} expected per qid × {len(qids)} qids")
```

- [ ] **Step 4: Generate_fn behavior for cache_only=True**

Update each `generate_fn` closure to honor `cfg.cache_only is True` by raising. One way: pass a `cache_only` flag through `infra` and have the closure check. Cleanest: in eval.py `process_question`, before solve, set:

```python
def _make_strict_generate_fn(real_gen_fn, cache_only_flag):
    async def gen():
        if cache_only_flag is True:
            raise AssertionError(
                f"cache_only=True: explore cache miss in {tts_agent.py} — "
                f"refusing to generate fresh API call."
            )
        return await real_gen_fn()
    return gen
```

Or simpler: `infra.cache_only` propagates to `tts_agent.run_explore` → wraps `generate_fn`.

- [ ] **Step 5: Add yaml-level test**

```python
def test_eval_config_cache_only_field():
    from eval import EvalConfig, load_config
    import tempfile
    import yaml as pyyaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        pyyaml.dump({
            "benchmark": {"name": "gpqa"},
            "method": {"name": "self-refine",
                       "backend": {"name": "claude"},
                       "explore": {"label": "default",
                                   "model": {"backend": "claude", "model": "claude-sonnet-4-6"},
                                   "cache_dir": "/tmp/x"}},
            "cache_only": False,
        }, f)
        cfg = load_config(config_path=f.name)
    assert cfg.cache_only is False
```

- [ ] **Step 6: Commit**

```bash
git add Experiment/core_code/eval.py Experiment/core_code/methods/registry.py
git commit -m "feat(eval.py): add cache_only yaml field; preflight signature accepts spec

cache_only=False enables online-cache mode (preflight skipped, miss→API→
writeback). cache_only=True enforces strict cache. None retains
MethodConfig hardcoded default (current behavior, zero migration cost
for existing yamls)."
```

---

### Task 17: Verify `rollout_idx` propagates through full call chain

**Files:**
- Audit: `Experiment/core_code/methods/tts_agent.py`
- Audit: `Experiment/core_code/eval.py:546-680`

- [ ] **Step 1: Audit rollout_idx usage**

Run: `grep -n "rollout_idx" Experiment/core_code/methods/*.py Experiment/core_code/eval.py`

Verify: `ctx.rollout_idx` is passed into `variant.get_exploration(rollout_idx=...)` and `variant.get_all_explorations(rollout_idx=...)` in every site that has access to it.

- [ ] **Step 2: Add an explicit test**

```python
@pytest.mark.asyncio
async def test_get_exploration_isolates_rollouts(tmp_path):
    from cache_types import Exploration
    v = _make_variant(tmp_path)

    async def gen_for(rollout, answer):
        async def g():
            return Exploration(qid="q1", idx=1, rollout_idx=rollout,
                               answer=answer, trajectory="", cost_usd=0.0, model="m")
        return g

    exp0 = await v.get_exploration("q1", 1, rollout_idx=0, generate_fn=await gen_for(0, "rollout0_ans"))
    exp1 = await v.get_exploration("q1", 1, rollout_idx=1, generate_fn=await gen_for(1, "rollout1_ans"))
    assert exp0.answer == "rollout0_ans"
    assert exp1.answer == "rollout1_ans"
    # Each persisted to its own subpath
    assert (tmp_path / "q1" / "rollout_0" / "explore_1" / "result.json").exists()
    assert (tmp_path / "q1" / "rollout_1" / "explore_1" / "result.json").exists()
```

- [ ] **Step 3: Run, fix any propagation gaps**

- [ ] **Step 4: Commit**

```bash
git add Experiment/core_code/tests/test_explore_variant_methods.py Experiment/core_code/methods/ Experiment/core_code/eval.py
git commit -m "test+fix: rollout_idx propagation isolates K>1 rollouts

K>1 cache previously shared explore_<idx> slot across rollouts (mutual
overwrites under non-zero temperature). With rollout_idx in path, each
rollout gets its own slot. Old K>1 cache (if any) is invalid and must be
rebuilt."
```

---

## Phase 7 — Cleanup

### Task 18: Move best-effort matching helpers to `cache_types.py`; delete from `benchmarks/base.py`

**Files:**
- Modify: `Experiment/core_code/cache_types.py` (add counters + summarize)
- Modify: `Experiment/core_code/benchmarks/base.py:35-130` (delete originals)
- Modify: `Experiment/core_code/eval.py` (update import path for `summarize_judge_cache`)

This task was originally "delete dead helpers". Decision 2026-05-05 (user binding, see "Reality Check / Issue R2"): the 4-branch best-effort matching policy is intentional and must be preserved. `find_cached_judge`'s logic now lives in `ExploreVariant._load_judge` (Task 5 Step 5). The supporting counters move alongside.

- [ ] **Step 1: Add counters + helpers to `cache_types.py`**

Append to `cache_types.py`:

```python
# Process-level counters mirroring the original benchmarks/base.py policy.
# Banner-aggregated to avoid 800 per-call warning lines per HLE run.
_JUDGE_CACHE_STATS: dict = {
    "exact_hits": 0,
    "best_effort_hits": 0,
    "best_effort_extras": set(),
}


def reset_judge_cache_stats() -> None:
    _JUDGE_CACHE_STATS["exact_hits"] = 0
    _JUDGE_CACHE_STATS["best_effort_hits"] = 0
    _JUDGE_CACHE_STATS["best_effort_extras"] = set()


def summarize_judge_cache() -> dict:
    return {
        "exact_hits": _JUDGE_CACHE_STATS["exact_hits"],
        "best_effort_hits": _JUDGE_CACHE_STATS["best_effort_hits"],
        "best_effort_extras": sorted(_JUDGE_CACHE_STATS["best_effort_extras"]),
    }
```

- [ ] **Step 2: Update `eval.py` imports**

Change `from benchmarks.base import summarize_judge_cache, reset_judge_cache_stats` to `from cache_types import summarize_judge_cache, reset_judge_cache_stats`. Banner-print site stays as-is.

- [ ] **Step 3: Verify zero remaining references to `judge_label` / `find_cached_judge`**

```bash
grep -rn "judge_label\|find_cached_judge" Experiment/core_code/ --include="*.py"
```
Expected: only the definitions themselves in benchmarks/base.py.

- [ ] **Step 4: Delete `judge_label`, `find_cached_judge` from `benchmarks/base.py`**

Remove `benchmarks/base.py:35-37` (`judge_label`) and `benchmarks/base.py:65-136` (`find_cached_judge`). Also remove `_JUDGE_CACHE_STATS`, `reset_judge_cache_stats`, `summarize_judge_cache` from base.py since they now live in `cache_types.py`.

- [ ] **Step 5: Run tests**

Run: `pytest Experiment/core_code/tests/ -v --no-header 2>&1 | tail -10`

- [ ] **Step 6: Commit**

```bash
git add Experiment/core_code/cache_types.py Experiment/core_code/benchmarks/base.py Experiment/core_code/eval.py
git commit -m "refactor: move judge cache best-effort matching to cache_types.py

judge_label -> JudgeOutcome.label_for (Task 1).
find_cached_judge 4-branch matching -> ExploreVariant._load_judge (Task 5).
_JUDGE_CACHE_STATS / reset_judge_cache_stats / summarize_judge_cache stay
intact, just relocated. Run-end banner output unchanged."
```

---

### Task 19: End-to-end smoke on rbenchv yaml

**Files:**
- Modify: `Experiment/core_code/scripts/rbenchv/sonnet/rbenchv_sonnet_delegated.yaml` (add `cache_only: false`, `num: 5`)

- [ ] **Step 1: Update yaml for smoke**

```yaml
# rbenchv_sonnet_delegated.yaml
benchmark:
  name: rbenchv
  judge:
    model: claude-haiku-4-5-20251001
    backend: claude
method:
  name: tts-agent
  orchestrator_prompt: single
  orchestrator:
    backend: claude
    model: claude-sonnet-4-6
  explore:
    - label: default
      model:
        backend: claude
        model: claude-sonnet-4-6
      cache_dir: ../analysis/cache/rbenchv/sonnet
      num_explores: 4
  integrate:
    model:
      backend: claude
      model: claude-sonnet-4-6
  # Removed: integrate.cache_dir (Task 12 migration)
num_workers: 4
seed: 42
log_dir: ../analysis/run/rbenchv/sonnet

# Smoke: online cache mode + 5-question subset
# Online cache: preflight skipped, missing explores generate fresh.
# num=5 caps blast radius (~$5).
cache_only: false
num: 5
```

- [ ] **Step 2: Run smoke**

```bash
cd Experiment/core_code
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
    --config scripts/rbenchv/sonnet/rbenchv_sonnet_delegated.yaml \
    > /tmp/rbenchv_smoke.log 2>&1 &
SMOKE_PID=$!
echo "PID=$SMOKE_PID, log=/tmp/rbenchv_smoke.log"
```

- [ ] **Step 3: Verify banner doesn't crash**

`tail -f /tmp/rbenchv_smoke.log` — expect:
- `Cache pre-flight OK` line absent (skipped because cache_only=false)
- "Questions to run: 5" or similar
- No AssertionError on banner
- Per-question `[explore_X] cache miss` or `cache hit` messages (if those logs were added)

- [ ] **Step 4: Verify cache writeback**

After smoke completes:

```bash
ls -la Experiment/analysis/cache/rbenchv/sonnet/<some_qid>/explore_5/
# expect: result.json, input.md, output.md, judges/<label>/grade.json
```

- [ ] **Step 5: Final commit**

```bash
git add Experiment/core_code/scripts/rbenchv/sonnet/rbenchv_sonnet_delegated.yaml
git commit -m "test: rbenchv smoke under cache_only=false (online cache mode)

Verifies the full refactor path:
- preflight skipped (cache_only=false)
- explore cache miss → API call → writeback to <cache_dir>/<qid>/explore_<idx>/
- judge bundle written to <cache_dir>/<qid>/explore_<idx>/judges/<label>/
- final answer judge written to <run_dir>/grading/<qid>/judges/<label>/"
```

---

## Self-Review Checklist (run after writing all tasks)

- **Spec coverage:** Each of the 12 contracts maps to a task — verified ✓ (Tasks 1-2 → contracts 1, 4; Task 3 → contract 2; Task 4 → contract 3; Tasks 5-7 → contracts 5, 6, 9, 10; Task 12 → contract 7; Task 16 → contract 8; Task 13 → contract 11; Task 15 → contract 12).
- **Placeholder scan:** No "TBD", "implement appropriate error handling", "similar to Task N" anywhere. Each step has full code or exact commands.
- **Type consistency:** `JudgeOutcome.label`, `JudgeOutcome.label_for`, `Exploration.persist`, `Exploration.verdict` all match the dataclass definition in cache_types.py. `ExploreVariant.get_exploration` signature consistent across all 5 task references.
- **Incidental fixes outside scope:** Task 17 fixes K>1 rollout cache slot collision (was a pre-existing bug exposed by abstraction). Task 14 adds `per_variant_candidates` to record schema (no version field — field presence is the schema indicator). Both are documented as part of this refactor's scope, not silent.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-05-explore-cache-owner-refactor.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Good for a 19-task refactor where context bloat from accumulated diffs is the main risk.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Good if you want to pause for design adjustments mid-flight.

Which approach?
