# ModelConfig Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-method shared `BackendConfig` + per-role bare model strings with per-role full `ModelConfig`, and merge `TTSAgentSpec` / `TTSAgentMultiSpec` / `TTSAgentEffortSpec` into one unified `TTSAgentSpec` whose `explore: list[ExploreVariant]` covers single, multi-model, and multi-effort runs.

**Architecture:**
- `ModelConfig`: per-role `(backend, model, budget_tokens, effort, timeout, max_output_tokens, [vllm_sampling, openrouter_provider_*])` with backend-prefixed fields and a model_validator that fails loud on cross-backend leakage.
- `RoleSlot{model, cache_dir}` and `ExploreVariant{label, model, cache_dir, num_explores}` wrap `ModelConfig` for roles that need per-role caching.
- Unified `TTSAgentSpec` carries `orchestrator: ModelConfig`, `explore: list[ExploreVariant]`, `integrate: RoleSlot|None`, plus an explicit `orchestrator_prompt: Literal["single","multi_model","effort"]` field that prompts.py routes to the right system prompt via `select_orchestrator_prompt(spec)`.
- The orchestrator's explore tool exposes `variant: enum[<labels>]` when `len(explore) > 1`, no parameter when `len(explore) == 1`.

**Tech Stack:** Python 3.11 (`explain` conda env), pydantic v2, ruamel.yaml for migration, pytest. All runtime invocations use `conda run -n explain --no-capture-output python ...` per project CLAUDE.md.

**Scope:**
- Modify: `Experiment/core_code/methods/{specs.py, base.py, tts_agent.py, registry.py, self_refine.py, socratic_self_refine.py, budget_forcing.py, standalone_integrator.py, tool_state.py}`, `eval.py`, `precache_explores.py`, `backends/openrouter.py`, `benchmarks/specs.py`, `benchmarks/base.py`, `prompts.py`.
- Delete: `methods/tts_agent_multi.py` (290 lines), `methods/tts_agent_effort.py` (293 lines).
- Create: `scripts/maintenance/migrate_yamls_to_modelconfig.py`, `scripts/maintenance/migrate_vllm_judge_cache.py`, plus ~6 test files under `tests/`.
- Migrate: 155 yamls under `Experiment/core_code/scripts/**/*.yaml` (17 of those are `tts-agent-multi`; 49 are precache yamls; 0 are `tts-agent-effort`).
- Migrate: 3 vllm-judge cache directories under `Experiment/analysis/cache/<bench>/<method>/<qid>/explore_<n>/judges/vllm__*/config.json` (rename `sampling` key to `vllm_sampling`).

**Out of scope (per design §12):** `training/grpo/grade_cache.py`, `scripts/hle/sonnet/regrade_socratic_self_refine.py`, yaml inheritance/anchors, `RerankSpec.reward_model` (stays a local-PyTorch `str`).

---

## File Structure

| Path | Role |
|---|---|
| `methods/specs.py` | New `ModelConfig`, `RoleSlot`, `ExploreVariant`; unified `TTSAgentSpec`; deletes `BackendConfig`/`TTSAgentMultiSpec`/`TTSAgentEffortSpec`. |
| `methods/base.py` | Slimmed `InfraConfig` (drops shared backend/effort/budget/timeout/max_output_tokens/orchestrator_effort); `SolveContext.call_sub_model(*, model_cfg, ...)`. |
| `methods/tts_agent.py` | One unified solver; per-variant sub-model callers; explore tool with optional `variant` param; reads `select_orchestrator_prompt(spec)`. |
| `methods/tool_state.py` | `ExploreStepState` extended with per-variant counters. |
| `methods/registry.py` | `TTSAgentMultiMethod`/`TTSAgentEffortMethod` deleted; `derive_evaluate_args` returns spec-shaped data instead of flat `orchestrator_model/explore_model/integrate_model/cache_dirs_multi`. |
| `methods/{self_refine,socratic_self_refine,budget_forcing}.py` | Take `explore: ExploreVariant`. |
| `methods/standalone_integrator.py` | Takes `integrate: RoleSlot, num_explores: int`. |
| `prompts.py` | Adds `select_orchestrator_prompt(spec) -> str`. Existing four prompt strings unchanged byte-for-byte. |
| `backends/openrouter.py` | Module-globals `_PROVIDER_ORDER`/`_PROVIDER_ALLOW_FALLBACKS`/`set_provider()`/`_maybe_inject_provider()` deleted; provider routing read from per-call `ModelConfig` and threaded via the `call_sub_model` / `run_tool_conversation` kwargs. |
| `benchmarks/specs.py` | `ClaudeJudgeSpec`/`CodexJudgeSpec`/`VllmJudgeSpec`/`_JudgeSpec`/`JudgeSpec` deleted; `judge: ModelConfig | None` in `HLESpec`/`BabyVisionSpec`/`RBenchVSpec`. |
| `benchmarks/base.py` | `judge_label(judge_spec)` updated to read `judge_spec.backend` instead of `judge_spec.name`. |
| `eval.py` | Re-instantiates `InfraConfig`; pulls per-role `ModelConfig`s through `method.derive_evaluate_args`; deletes the `openrouter.set_provider(...)` call. |
| `precache_explores.py` | `PrecacheConfig.explore: ExploreVariant` collapses 10 scattered backend-related fields. |
| `scripts/maintenance/migrate_vllm_judge_cache.py` | One-shot: renames `sampling` key to `vllm_sampling` in vllm judge `config.json` files. |
| `scripts/maintenance/migrate_yamls_to_modelconfig.py` | One-shot: rewrites all yamls under `scripts/**/*.yaml`. |
| `tests/test_modelconfig.py` | Validator tests for cross-backend field leakage. |
| `tests/test_role_slot.py` | Field validation. |
| `tests/test_explore_variant.py` | Field validation. |
| `tests/test_tts_agent_unified.py` | Length-1/3 yaml round-trip; deleted `tts-agent-multi`/`tts-agent-effort` rejection. |
| `tests/test_judge_spec.py` | Rewritten for single `ModelConfig` judge field. |
| `tests/test_eval_config.py`, `tests/test_precache_config.py` | yaml-load round-trip tests. |

---

## Task 1: Add `ModelConfig` / `RoleSlot` / `ExploreVariant` types

**Files:**
- Modify: `Experiment/core_code/methods/specs.py`
- Test: `Experiment/core_code/tests/test_modelconfig.py` (create)
- Test: `Experiment/core_code/tests/test_role_slot.py` (create)
- Test: `Experiment/core_code/tests/test_explore_variant.py` (create)

This task adds the new types alongside the old `BackendConfig`. Old code still compiles — the new types are unreferenced.

- [ ] **Step 1: Write the validator tests for `ModelConfig`**

Create `Experiment/core_code/tests/test_modelconfig.py`:

```python
"""Validator tests for ModelConfig (per-role backend invocation config)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from methods.specs import ModelConfig, SamplingConfig


def test_minimal_claude_config_loads():
    cfg = ModelConfig(backend="claude", model="claude-sonnet-4-6")
    assert cfg.budget_tokens == 32000
    assert cfg.effort == "low"
    assert cfg.timeout == 1200.0
    assert cfg.max_output_tokens is None
    assert cfg.vllm_sampling is None
    assert cfg.openrouter_provider_order is None
    assert cfg.openrouter_provider_allow_fallbacks is True


def test_vllm_config_with_sampling():
    cfg = ModelConfig(
        backend="vllm",
        model="qwen36-35b-a3b-fp8",
        vllm_sampling=SamplingConfig(temperature=0.7),
    )
    assert cfg.vllm_sampling.temperature == 0.7


def test_vllm_sampling_rejected_on_non_vllm_backend():
    with pytest.raises(ValidationError) as exc:
        ModelConfig(
            backend="claude",
            model="claude-sonnet-4-6",
            vllm_sampling=SamplingConfig(temperature=0.7),
        )
    assert "vllm_sampling is vllm-only" in str(exc.value)


def test_openrouter_provider_order_rejected_on_non_openrouter_backend():
    with pytest.raises(ValidationError) as exc:
        ModelConfig(
            backend="claude",
            model="claude-sonnet-4-6",
            openrouter_provider_order=["Parasail"],
        )
    assert "openrouter_provider_order is openrouter-only" in str(exc.value)


def test_openrouter_allow_fallbacks_default_true_rejected_when_overridden_on_other_backend():
    with pytest.raises(ValidationError) as exc:
        ModelConfig(
            backend="claude",
            model="claude-sonnet-4-6",
            openrouter_provider_allow_fallbacks=False,
        )
    assert "openrouter_provider_allow_fallbacks is openrouter-only" in str(exc.value)


def test_unknown_backend_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(backend="nope", model="x")


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(backend="claude", model="claude-sonnet-4-6", typo_field=1)
```

- [ ] **Step 2: Run tests to verify they fail with import error**

```
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output python -m pytest tests/test_modelconfig.py -v
```
Expected: ImportError ("cannot import name 'ModelConfig'") or all tests fail.

- [ ] **Step 3: Add `ModelConfig` to `methods/specs.py`**

Insert immediately after `class SamplingConfig(...)` (before `class BackendConfig`):

```python
class ModelConfig(BaseModel):
    """Per-role backend invocation config.

    Replaces the shared method-level BackendConfig + bare-string model name
    pattern. Each model role (orchestrator, explore variant, integrate, judge)
    carries its own complete invocation parameters, including optional
    backend-specific blocks (vllm_sampling for vllm, openrouter_provider_*
    for openrouter). The model_validator hard-fails when a backend-specific
    field is set on the wrong backend, replacing the prior silent-no-op
    behavior of BackendConfig.
    """
    model_config = {"extra": "forbid"}

    backend: Literal["codex", "claude", "vllm", "openrouter"]
    model: str
    budget_tokens: int = 32000
    effort: Literal["low", "medium", "high", "max"] = "low"
    timeout: float = 1200.0
    max_output_tokens: int | None = None

    vllm_sampling: SamplingConfig | None = None

    openrouter_provider_order: list[str] | None = None
    openrouter_provider_allow_fallbacks: bool = True

    @model_validator(mode="after")
    def _check_backend_specific(self):
        if self.backend != "vllm":
            assert self.vllm_sampling is None, (
                f"vllm_sampling is vllm-only but backend={self.backend!r}; remove from yaml"
            )
        if self.backend != "openrouter":
            assert self.openrouter_provider_order is None, (
                f"openrouter_provider_order is openrouter-only but backend={self.backend!r}"
            )
            assert self.openrouter_provider_allow_fallbacks is True, (
                f"openrouter_provider_allow_fallbacks is openrouter-only but "
                f"backend={self.backend!r}"
            )
        return self
```

- [ ] **Step 4: Run `ModelConfig` tests to verify they pass**

```
conda run -n explain --no-capture-output python -m pytest tests/test_modelconfig.py -v
```
Expected: all 7 tests pass.

- [ ] **Step 5: Write `RoleSlot` and `ExploreVariant` tests**

Create `Experiment/core_code/tests/test_role_slot.py`:

```python
"""Validator tests for RoleSlot (single-call cached role)."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from methods.specs import ModelConfig, RoleSlot


def test_role_slot_loads():
    slot = RoleSlot(
        model=ModelConfig(backend="claude", model="claude-sonnet-4-6"),
        cache_dir=Path("/tmp/x"),
    )
    assert slot.model.model == "claude-sonnet-4-6"
    assert slot.cache_dir == Path("/tmp/x")


def test_role_slot_rejects_num_explores_typo():
    """RoleSlot is for single-call roles; num_explores would belong on ExploreVariant."""
    with pytest.raises(ValidationError):
        RoleSlot(
            model=ModelConfig(backend="claude", model="claude-sonnet-4-6"),
            cache_dir=Path("/tmp/x"),
            num_explores=8,
        )
```

Create `Experiment/core_code/tests/test_explore_variant.py`:

```python
"""Validator tests for ExploreVariant (per-variant explore call config)."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from methods.specs import ExploreVariant, ModelConfig


def test_explore_variant_loads():
    v = ExploreVariant(
        label="haiku",
        model=ModelConfig(backend="claude", model="claude-haiku-4-5-20251001"),
        cache_dir=Path("/tmp/cache/haiku"),
    )
    assert v.label == "haiku"
    assert v.num_explores == 8


def test_explore_variant_label_required():
    with pytest.raises(ValidationError):
        ExploreVariant(
            model=ModelConfig(backend="claude", model="claude-haiku-4-5-20251001"),
            cache_dir=Path("/tmp/cache/haiku"),
        )


def test_explore_variant_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ExploreVariant(
            label="haiku",
            model=ModelConfig(backend="claude", model="claude-haiku-4-5-20251001"),
            cache_dir=Path("/tmp/cache/haiku"),
            unknown_key=1,
        )
```

- [ ] **Step 6: Run tests, expect import errors**

```
conda run -n explain --no-capture-output python -m pytest tests/test_role_slot.py tests/test_explore_variant.py -v
```
Expected: ImportError on `RoleSlot` and `ExploreVariant`.

- [ ] **Step 7: Add `RoleSlot` and `ExploreVariant` to `methods/specs.py`**

Insert after the new `ModelConfig` class:

```python
class RoleSlot(BaseModel):
    """A model invocation + the cache_dir its outputs land in.

    Used by single-call cached roles (e.g. integrate). Pydantic extra="forbid"
    rejects num_explores or other ExploreVariant-shaped fields at config-load.
    """
    model_config = {"extra": "forbid"}
    model: ModelConfig
    cache_dir: Path


class ExploreVariant(BaseModel):
    """One explore-pass: model + its cache_dir + how many candidates to draw.

    The label is required and serves as the orchestrator-visible identifier
    (the unified explore tool exposes `variant: enum[<labels>]` when there
    is more than one variant in a TTSAgentSpec.explore list). Cache layout
    on disk is per-variant via cache_dir, so cache_keys remain
    `f"explore_{idx}"` within each variant.
    """
    model_config = {"extra": "forbid"}
    label: str
    model: ModelConfig
    cache_dir: Path
    num_explores: int = 8
```

- [ ] **Step 8: Run tests, expect all pass**

```
conda run -n explain --no-capture-output python -m pytest tests/test_role_slot.py tests/test_explore_variant.py -v
```
Expected: 5 tests pass.

- [ ] **Step 9: Commit**

```
git add Experiment/core_code/methods/specs.py Experiment/core_code/tests/test_modelconfig.py Experiment/core_code/tests/test_role_slot.py Experiment/core_code/tests/test_explore_variant.py
git commit -m "feat(specs): add ModelConfig/RoleSlot/ExploreVariant types"
```

---

## Task 2: Replace `TTSAgentSpec` family with unified shape; delete `BackendConfig`

**Files:**
- Modify: `Experiment/core_code/methods/specs.py`
- Test: `Experiment/core_code/tests/test_tts_agent_unified.py` (create)

This task merges `TTSAgentSpec` / `TTSAgentMultiSpec` / `TTSAgentEffortSpec` into one. Old `BackendConfig` is deleted in this commit. Solver / registry / eval still reference deleted symbols — those break and get fixed in later tasks. Acceptable mid-tree state because we will not run anything between Task 2 and Task 14.

- [ ] **Step 1: Write `TTSAgentSpec` round-trip tests**

Create `Experiment/core_code/tests/test_tts_agent_unified.py`:

```python
"""Round-trip tests for the unified TTSAgentSpec (single / multi-model / effort)."""
from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from methods.specs import TTSAgentSpec


_SINGLE_NO_INTEGRATE = """
name: tts-agent
orchestrator_prompt: single
orchestrator:
  backend: openrouter
  model: x-ai/grok-4.1-fast
  effort: high
explore:
  - label: default
    model: {backend: openrouter, model: x-ai/grok-4.1-fast, effort: low}
    cache_dir: /cache/hle/grok/gold
    num_explores: 4
"""

_MULTI_MODEL = """
name: tts-agent
orchestrator_prompt: multi_model
orchestrator: {backend: claude, model: claude-sonnet-4-6}
explore:
  - label: haiku
    model: {backend: claude, model: claude-haiku-4-5-20251001}
    cache_dir: /cache/hle/haiku/gold
    num_explores: 8
  - label: sonnet
    model: {backend: claude, model: claude-sonnet-4-6}
    cache_dir: /cache/hle/sonnet/gold
    num_explores: 8
  - label: opus
    model: {backend: claude, model: claude-opus-4-6}
    cache_dir: /cache/hle/opus/gold
    num_explores: 4
"""

_EFFORT = """
name: tts-agent
orchestrator_prompt: effort
orchestrator: {backend: vllm, model: qwen36-35b-a3b-fp8}
explore:
  - label: low
    model: {backend: vllm, model: qwen36-35b-a3b-fp8, effort: low}
    cache_dir: /cache/hle/qwen_low/gold
    num_explores: 6
  - label: medium
    model: {backend: vllm, model: qwen36-35b-a3b-fp8, effort: medium}
    cache_dir: /cache/hle/qwen_medium/gold
    num_explores: 6
  - label: high
    model: {backend: vllm, model: qwen36-35b-a3b-fp8, effort: high}
    cache_dir: /cache/hle/qwen_high/gold
    num_explores: 6
"""


def test_single_no_integrate_loads():
    spec = TTSAgentSpec.model_validate(yaml.safe_load(_SINGLE_NO_INTEGRATE))
    assert spec.orchestrator_prompt == "single"
    assert len(spec.explore) == 1
    assert spec.integrate is None
    assert spec.orchestrator.effort == "high"
    assert spec.explore[0].model.effort == "low"


def test_multi_model_loads():
    spec = TTSAgentSpec.model_validate(yaml.safe_load(_MULTI_MODEL))
    assert spec.orchestrator_prompt == "multi_model"
    assert len(spec.explore) == 3
    assert {v.label for v in spec.explore} == {"haiku", "sonnet", "opus"}
    assert spec.integrate is None


def test_effort_loads():
    spec = TTSAgentSpec.model_validate(yaml.safe_load(_EFFORT))
    assert spec.orchestrator_prompt == "effort"
    assert {v.label for v in spec.explore} == {"low", "medium", "high"}


def test_single_with_three_variants_rejected():
    bad = yaml.safe_load(_SINGLE_NO_INTEGRATE)
    bad["explore"] = yaml.safe_load(_MULTI_MODEL)["explore"]
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_multi_model_with_one_variant_rejected():
    bad = yaml.safe_load(_MULTI_MODEL)
    bad["explore"] = bad["explore"][:1]
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_multi_model_with_wrong_labels_rejected():
    bad = yaml.safe_load(_MULTI_MODEL)
    bad["explore"][0]["label"] = "qwen"
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_effort_with_wrong_labels_rejected():
    bad = yaml.safe_load(_EFFORT)
    bad["explore"][0]["label"] = "tiny"
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_multi_model_with_integrate_rejected():
    bad = yaml.safe_load(_MULTI_MODEL)
    bad["integrate"] = {
        "model": {"backend": "claude", "model": "claude-sonnet-4-6"},
        "cache_dir": "/cache/integrate",
    }
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_duplicate_labels_rejected():
    bad = yaml.safe_load(_MULTI_MODEL)
    bad["explore"][1]["label"] = "haiku"
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_old_multi_method_name_rejected():
    """The deleted TTSAgentMultiSpec used name: tts-agent-multi.

    Validation routes via discriminated union on `name`; the new union has no
    tts-agent-multi variant, so validation fails before TTSAgentSpec is
    constructed. Confirm the error surfaces."""
    from methods.specs import MethodSpec
    from pydantic import TypeAdapter
    bad = yaml.safe_load(_MULTI_MODEL)
    bad["name"] = "tts-agent-multi"
    with pytest.raises(ValidationError):
        TypeAdapter(MethodSpec).validate_python(bad)
```

- [ ] **Step 2: Run tests; expect import errors**

```
conda run -n explain --no-capture-output python -m pytest tests/test_tts_agent_unified.py -v
```

- [ ] **Step 3: Replace the three old method specs in `methods/specs.py`**

In `methods/specs.py`:
1. Delete `BackendConfig` (lines 57-77 in current file).
2. Delete `TTSAgentSpec` (lines 84-110), `TTSAgentMultiSpec` (lines 113-120), `TTSAgentEffortSpec` (lines 123-130).
3. In their place, insert one unified spec:

```python
class TTSAgentSpec(_MethodSpec):
    """Unified spec covering single, multi-model, and multi-effort runs.

    `explore: list[ExploreVariant]` length encodes the operating mode:
    - length 1 with `orchestrator_prompt: single`: standard ATTS.
    - length 3 with `orchestrator_prompt: multi_model` and labels
      {haiku, sonnet, opus}: replaces the old TTSAgentMultiSpec.
    - length 3 with `orchestrator_prompt: effort` and labels {low, medium, high}:
      replaces the old TTSAgentEffortSpec.

    The orchestrator's explore tool exposes `variant: enum[<labels>]` when
    `len(explore) > 1`, no parameter when `len(explore) == 1`. `select_orchestrator_prompt`
    in prompts.py routes (orchestrator_prompt, integrate is None) to the
    matching system prompt; the multi_model and effort prompts hardcode their
    label sets, so the validator below enforces label-set match.
    """
    name: Literal["tts-agent"]
    orchestrator: ModelConfig
    explore: list[ExploreVariant]
    integrate: RoleSlot | None = None
    orchestrator_prompt: Literal["single", "multi_model", "effort"]
    num_rollouts: int = 1

    @model_validator(mode="after")
    def _check_consistency(self):
        n = len(self.explore)
        p = self.orchestrator_prompt
        assert (p == "single") == (n == 1), (
            f"orchestrator_prompt={p!r} requires "
            f"{'len(explore)==1' if p == 'single' else 'len(explore)>1'}, got {n}"
        )
        labels = [v.label for v in self.explore]
        assert len(set(labels)) == len(labels), (
            f"duplicate labels in explore list: {labels}"
        )
        if p == "multi_model":
            assert set(labels) == {"haiku", "sonnet", "opus"}, (
                f"orchestrator_prompt=multi_model hardcodes haiku/sonnet/opus; "
                f"yaml has labels={set(labels)}"
            )
        if p == "effort":
            assert set(labels) == {"low", "medium", "high"}, (
                f"orchestrator_prompt=effort hardcodes low/medium/high; "
                f"yaml has labels={set(labels)}"
            )
        if p in ("multi_model", "effort"):
            assert self.integrate is None, (
                f"orchestrator_prompt={p!r} assumes orchestrator self-synthesizes; "
                f"integrate must be None"
            )
        return self
```

4. In the `MethodSpec` union, remove `TTSAgentMultiSpec` and `TTSAgentEffortSpec`. Resulting union:

```python
MethodSpec = Annotated[
    Union[
        TTSAgentSpec,
        SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
        RerankSpec, StandaloneIntegratorSpec,
    ],
    Field(discriminator="name"),
]
```

- [ ] **Step 4: Run tests, expect new tests pass**

```
conda run -n explain --no-capture-output python -m pytest tests/test_tts_agent_unified.py -v
```
Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```
git add Experiment/core_code/methods/specs.py Experiment/core_code/tests/test_tts_agent_unified.py
git commit -m "feat(specs): unify tts-agent spec; delete BackendConfig/Multi/Effort"
```

---

## Task 3: Refactor remaining method specs

**Files:**
- Modify: `Experiment/core_code/methods/specs.py`
- Test: existing `tests/test_eval_config.py` if present, otherwise no new tests (covered in Task 17 yaml-load test).

- [ ] **Step 1: Update `SelfRefineSpec` / `SocraticSelfRefineSpec` / `BudgetForcingSpec`**

In `methods/specs.py`, replace the three current classes:

```python
class SelfRefineSpec(_MethodSpec):
    name: Literal["self-refine"]
    explore: ExploreVariant


class SocraticSelfRefineSpec(_MethodSpec):
    name: Literal["socratic-self-refine"]
    explore: ExploreVariant


class BudgetForcingSpec(_MethodSpec):
    name: Literal["budget-forcing"]
    explore: ExploreVariant
```

`num_explores` and `cache_dir` are now nested inside `explore: ExploreVariant`; `backend.{...}` block is folded into `explore.model: ModelConfig`.

- [ ] **Step 2: Update `StandaloneIntegratorSpec`**

```python
class StandaloneIntegratorSpec(_MethodSpec):
    name: Literal["standalone-integrator"]
    integrate: RoleSlot
    num_explores: int = 8
    # num_explores stays at top level (not on RoleSlot) because it controls
    # how many cached candidates this method consumes, not how many to call.
```

- [ ] **Step 3: Leave `RerankSpec` unchanged**

Per design §5.2, `RerankSpec.reward_model: str` + `cache_dir: Path` stays. Add an inline comment if not already present:

```python
class RerankSpec(_MethodSpec):
    name: Literal["rerank"]
    reward_model: str
    cache_dir: Path
    # Kept asymmetric: reward_model is a HuggingFace model id loaded by
    # transformers, not a backend-routed remote call. Forcing through
    # ModelConfig would require adding a "local" backend Literal which
    # mixes "remote backend" and "local PyTorch" semantics.
```

- [ ] **Step 4: Verify import surface**

```
conda run -n explain --no-capture-output python -c "from methods.specs import (
    ModelConfig, RoleSlot, ExploreVariant, MethodSpec,
    TTSAgentSpec, SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
    RerankSpec, StandaloneIntegratorSpec, SamplingConfig,
)"
```
Expected: no errors.

- [ ] **Step 5: Verify deleted symbols are gone**

```
conda run -n explain --no-capture-output python -c "
from methods.specs import BackendConfig
" 2>&1 | grep -q "ImportError\|cannot import" && echo OK_DELETED || echo STILL_EXISTS
```
Expected output: `OK_DELETED`.

- [ ] **Step 6: Commit**

```
git add Experiment/core_code/methods/specs.py
git commit -m "refactor(specs): port self-refine/socratic/budget-forcing/standalone to ModelConfig"
```

---

## Task 4: Refactor `benchmarks/specs.py` judge field

**Files:**
- Modify: `Experiment/core_code/benchmarks/specs.py`
- Modify: `Experiment/core_code/benchmarks/base.py` (`judge_label` reads `.backend`)
- Test: `Experiment/core_code/tests/test_judge_spec.py` (create or rewrite)

- [ ] **Step 1: Write judge spec tests**

Create `Experiment/core_code/tests/test_judge_spec.py`:

```python
"""Validator tests for the unified judge: ModelConfig field."""
from __future__ import annotations

import pytest
from pydantic import ValidationError, TypeAdapter

from benchmarks.specs import BenchmarkSpec
from methods.specs import ModelConfig


def _hle(judge_block):
    return {"name": "hle", "subset": "gold", "judge": judge_block}


def test_claude_judge_loads():
    spec = TypeAdapter(BenchmarkSpec).validate_python(_hle({
        "backend": "claude",
        "model": "claude-haiku-4-5-20251001",
    }))
    assert isinstance(spec.judge, ModelConfig)
    assert spec.judge.backend == "claude"
    assert spec.judge.effort == "low"  # CLAUDE.md non-thinking-judge default


def test_codex_judge_loads():
    spec = TypeAdapter(BenchmarkSpec).validate_python(_hle({
        "backend": "codex", "model": "gpt-5.2",
    }))
    assert spec.judge.backend == "codex"


def test_vllm_judge_with_sampling_loads():
    spec = TypeAdapter(BenchmarkSpec).validate_python(_hle({
        "backend": "vllm",
        "model": "qwen36-35b-a3b-fp8",
        "vllm_sampling": {"temperature": 0.0},
    }))
    assert spec.judge.vllm_sampling.temperature == 0.0


def test_old_field_name_sampling_rejected():
    """Existing yamls using `sampling:` (the old VllmJudgeSpec field name)
    must fail loud; migration script renames to `vllm_sampling`."""
    with pytest.raises(ValidationError):
        TypeAdapter(BenchmarkSpec).validate_python(_hle({
            "backend": "vllm",
            "model": "qwen36-35b-a3b-fp8",
            "sampling": {"temperature": 0.0},
        }))


def test_lcb_rejects_judge_block():
    """LCB grades by code execution; extra=forbid blocks judge: in yaml."""
    with pytest.raises(ValidationError):
        TypeAdapter(BenchmarkSpec).validate_python({
            "name": "lcb", "judge": {"backend": "claude", "model": "x"},
        })
```

- [ ] **Step 2: Replace `_JudgeSpec` family in `benchmarks/specs.py`**

Delete `_JudgeSpec`, `ClaudeJudgeSpec`, `CodexJudgeSpec`, `VllmJudgeSpec`, and the `JudgeSpec` Annotated union (lines 26-68 in current file).

In their place, change the three benchmark specs that carry a judge:

```python
from methods.specs import ModelConfig  # add to existing import block

# (delete _JudgeSpec / ClaudeJudgeSpec / CodexJudgeSpec / VllmJudgeSpec / JudgeSpec)


class HLESpec(_Spec):
    name: Literal["hle"]
    subset: Literal["gold", "revision", "uncertain"] | None = None
    category: str | None = None
    text_only: bool = False
    judge: ModelConfig  # required for HLE


class BabyVisionSpec(_Spec):
    name: Literal["babyvision"]
    type: str | None = None
    subtype: str | None = None
    judge: ModelConfig


class RBenchVSpec(_Spec):
    name: Literal["rbenchv"]
    category: str | None = None
    judge: ModelConfig
```

`GPQASpec`, `LCBSpec`, `AIME2025Spec`, `AIME2026Spec` keep their existing shape (no `judge:` field; rejected by `extra="forbid"`).

- [ ] **Step 3: Update `judge_label` in `benchmarks/base.py`**

Find the existing `judge_label(judge_spec)` function. Update its body to read `.backend` instead of `.name`. The on-disk path stays `f"{backend}__{model}"` — same string content, just sourced from the renamed field.

```bash
grep -n "judge_label" /data3/peijia/dr-claw/Explain/Experiment/core_code/benchmarks/base.py
```

Expected: a function definition like `def judge_label(judge_spec) -> str:`. Inside it, change `judge_spec.name` to `judge_spec.backend` everywhere (single-call site).

If the function takes a `dict` (from `model_dump()`), update the dict access from `["name"]` to `["backend"]`.

- [ ] **Step 4: Run judge spec tests**

```
conda run -n explain --no-capture-output python -m pytest tests/test_judge_spec.py -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```
git add Experiment/core_code/benchmarks/specs.py Experiment/core_code/benchmarks/base.py Experiment/core_code/tests/test_judge_spec.py
git commit -m "refactor(specs): collapse JudgeSpec union into ModelConfig"
```

---

## Task 5: Add `select_orchestrator_prompt` to `prompts.py`

**Files:**
- Modify: `Experiment/core_code/prompts.py`

- [ ] **Step 1: Insert `select_orchestrator_prompt` immediately after the four prompt strings**

Find `ORCHESTRATOR_EFFORT_SYSTEM_PROMPT = """..."""` (ends around line 317). Right after the closing `"""`, add:

```python
def select_orchestrator_prompt(spec) -> str:
    """Pick the orchestrator system prompt for a TTSAgentSpec.

    Currently dispatches on (orchestrator_prompt, integrate is None). Future
    routing axes (e.g. benchmark-family overrides, custom prompt registry)
    can be added here without touching solver call sites.

    The four prompt strings stay byte-identical with their pre-refactor form
    so existing run trajectories (analysis/run/hle/multi_model_effort_*) can
    be reproduced under the unified solver. The label-set assertions on the
    multi_model and effort branches are spec-side validators, not enforced
    here.
    """
    no_integrate = spec.integrate is None
    if spec.orchestrator_prompt == "single":
        return (
            ORCHESTRATOR_NO_INTEGRATE_SYSTEM_PROMPT
            if no_integrate
            else ORCHESTRATOR_SYSTEM_PROMPT
        )
    if spec.orchestrator_prompt == "multi_model":
        assert no_integrate
        return ORCHESTRATOR_MULTI_MODEL_SYSTEM_PROMPT
    if spec.orchestrator_prompt == "effort":
        assert no_integrate
        return ORCHESTRATOR_EFFORT_SYSTEM_PROMPT
    raise AssertionError(
        f"unknown orchestrator_prompt: {spec.orchestrator_prompt!r}"
    )
```

- [ ] **Step 2: Verify import**

```
conda run -n explain --no-capture-output python -c "from prompts import select_orchestrator_prompt; print(select_orchestrator_prompt)"
```
Expected: function repr.

- [ ] **Step 3: Commit**

```
git add Experiment/core_code/prompts.py
git commit -m "feat(prompts): add select_orchestrator_prompt(spec) router"
```

---

## Task 6: Extend `ExploreStepState` for per-variant counters

**Files:**
- Read: `Experiment/core_code/methods/tool_state.py`
- Modify: `Experiment/core_code/methods/tool_state.py`

This task extends the explore-state struct with per-variant counters used by the unified solver. Single-variant runs degenerate to the current behavior.

- [ ] **Step 1: Read the current shape**

```
cat /data3/peijia/dr-claw/Explain/Experiment/core_code/methods/tool_state.py
```

Note `ExploreStepState`'s current fields (likely `max_explores`, `call_count`, `used`, `is_exhausted`).

- [ ] **Step 2: Add per-variant tracking**

Add to `ExploreStepState` (alongside the existing fields):

```python
@dataclass
class ExploreStepState:
    max_explores: int                       # sum across variants for length>1
    call_count: int = 0                     # total across all variants
    # Per-variant in-variant indexer used to derive cache_key f"explore_{idx}"
    # within each variant's cache_dir. variant_call_counts[label] -> int.
    # Empty dict for length-1 (single variant) runs; the legacy global
    # `call_count + 1` is used for cache_key in that case.
    variant_call_counts: dict[str, int] = field(default_factory=dict)
    # Per-variant budget cap, populated from ExploreVariant.num_explores at
    # solver init. Solver asserts variant_call_counts[label] < variant_caps[label]
    # before dispatch (mirrors old tts_agent_multi.py budget guard).
    variant_caps: dict[str, int] = field(default_factory=dict)

    @property
    def used(self) -> int:
        return self.call_count

    @property
    def is_exhausted(self) -> bool:
        return self.call_count >= self.max_explores

    def variant_exhausted(self, label: str) -> bool:
        return self.variant_call_counts.get(label, 0) >= self.variant_caps.get(label, 0)
```

If the existing `advance(state)` helper exists, update it to take an optional `label: str | None = None` argument; when set, also bump `variant_call_counts[label]`.

- [ ] **Step 3: Verify import**

```
conda run -n explain --no-capture-output python -c "from methods.tool_state import ExploreStepState; s = ExploreStepState(max_explores=8); print(s.is_exhausted, s.variant_call_counts)"
```
Expected: `False {}`.

- [ ] **Step 4: Commit**

```
git add Experiment/core_code/methods/tool_state.py
git commit -m "feat(tool_state): per-variant counters for unified explore loop"
```

---

## Task 7: Slim `InfraConfig` and refactor `SolveContext.call_sub_model`

**Files:**
- Modify: `Experiment/core_code/methods/base.py`

- [ ] **Step 1: Edit `InfraConfig`**

In `methods/base.py`, replace the current dataclass:

```python
@dataclass
class InfraConfig:
    """Per-run infrastructure shared by all solving methods.

    Per-role backend/model/effort/budget/timeout previously shared at this
    level moved into per-role ModelConfig. InfraConfig now carries only
    cross-role context (cache_dir for legacy methods, max_iterations,
    benchmark, logger, enable_integrate).
    """
    max_iterations: int
    cache_dir: Path | None
    cache_only: bool
    benchmark: Any
    logger: RunLogger | None
    enable_integrate: bool = True
```

- [ ] **Step 2: Edit `SolveContext`**

Replace the current dataclass and `call_sub_model` method:

```python
@dataclass
class SolveContext:
    """Common state for all solve methods, created by create_solve_context()."""
    state: SolvingState
    cost: CostTracker
    rounds: list[RoundLog]
    _sub_model_fn: Any
    writer: TrajectoryWriter
    traj_dir: Path | None
    question_cache_dir: Path | None
    image_data_url: str | None
    benchmark: Any
    logger: RunLogger | None
    question_id: str | None
    cache_only: bool                  # mirrored from infra so integrate-role
                                      # callers built inside solvers don't need
                                      # to thread infra through every helper
    rollout_idx: int | None = None

    async def call_sub_model(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model_cfg,  # methods.specs.ModelConfig (typed at call site)
        output_schema: dict[str, Any],
        cache_key: str = "",
        writer: TrajectoryWriter | None = None,
    ) -> tuple[dict[str, Any], str, float, dict[str, Any], float]:
        """Dispatch a single backend call using a ModelConfig.

        All backend selection / budget / effort / timeout / sampling /
        provider routing comes from model_cfg. The infra-level _sub_model_fn
        was constructed at create_solve_context() time and is bound to a
        single backend module — so model_cfg.backend MUST match what
        _sub_model_fn was built against. A solver that needs to dispatch
        across backends (cross-backend explore variants) must keep one
        _sub_model_fn per backend; see methods/tts_agent.py per-variant
        callers."""
        return await self._sub_model_fn(
            system_prompt, user_message, self.image_data_url,
            model_cfg.model, output_schema,
            cache_key=cache_key,
            writer=writer or TrajectoryWriter.noop(),
            budget_tokens=model_cfg.budget_tokens,
            effort=model_cfg.effort,
            sampling=model_cfg.vllm_sampling.model_dump() if model_cfg.vllm_sampling else None,
            provider_order=model_cfg.openrouter_provider_order,
            provider_allow_fallbacks=model_cfg.openrouter_provider_allow_fallbacks,
        )

    def result(self, answer: str) -> SolveResult:
        return SolveResult(answer=answer, cost=self.cost, rounds=self.rounds, writer=self.writer)
```

- [ ] **Step 3: Update `make_sub_model_caller` signature**

`make_sub_model_caller(backend, ...)` already takes a `backend: str`; the inner `call(...)` already accepts `budget_tokens` / `effort` / `sampling` kwargs. Add two new kwargs to the inner `call`:

```python
async def call(
    system_prompt: str,
    user_message: str,
    image_data_url: str | None,
    model: str,
    output_schema: dict[str, Any],
    *,
    cache_key: str = "",
    writer: TrajectoryWriter = TrajectoryWriter.noop(),
    budget_tokens: int = 32000,
    effort: str | None = None,
    sampling: dict | None = None,
    provider_order: list[str] | None = None,            # NEW
    provider_allow_fallbacks: bool = True,              # NEW
):
    ...
    api_coro = backend_mod.call_sub_model(
        system_prompt, user_message, image_data_url, model, output_schema,
        writer, budget_tokens=budget_tokens, effort=effort, sampling=sampling,
        provider_order=provider_order,                  # NEW
        provider_allow_fallbacks=provider_allow_fallbacks,  # NEW
    )
```

(The kwargs are silently ignored by non-openrouter backends in Task 13's openrouter.py refactor, by being accepted in their `call_sub_model` signature without effect.)

- [ ] **Step 4: Update `create_solve_context`**

Replace the current `create_solve_context(*, infra, ...)` body so it stops reading `infra.backend` / `infra.budget_tokens` / `infra.effort` / `infra.timeout` / etc. The factory now needs the backend string and the cache-related fields explicitly to build `make_sub_model_caller`. Solvers will pass them in directly:

```python
def create_solve_context(
    *,
    infra: InfraConfig,
    backend: str,
    timeout: float,
    problem: str,
    image_data_url: str | None = None,
    question_id: str | None = None,
    writer_system_prompt: str,
    writer_user_message: str,
    writer_header_lines: list[str],
    writer_title_suffix: str,
    rollout_idx: int | None = None,
) -> SolveContext:
    """Build the per-question solve context.

    `backend` and `timeout` are passed in by the solver because they belong
    on the role's ModelConfig, not on infra. The default _sub_model_fn is
    built against `backend`; solvers that need cross-backend dispatch (e.g.
    multi-variant explore) build additional _sub_model_fn instances per
    variant inline."""
    state = SolvingState(
        problem=problem,
        explore=ExploreStepState(max_explores=infra.max_iterations),
    )
    ...
    sub_model_fn = make_sub_model_caller(
        backend, question_cache_dir, infra.cache_only,
        traj_dir=traj_dir, timeout=timeout,
    )
    ...
    return SolveContext(
        state=state, cost=cost, rounds=rounds,
        _sub_model_fn=sub_model_fn, writer=writer,
        traj_dir=traj_dir, question_cache_dir=question_cache_dir,
        image_data_url=image_data_url,
        benchmark=infra.benchmark, logger=infra.logger,
        question_id=question_id,
        cache_only=infra.cache_only,
        rollout_idx=rollout_idx,
    )
```

- [ ] **Step 5: Verify imports compile**

```
conda run -n explain --no-capture-output python -c "
from methods.base import InfraConfig, SolveContext, create_solve_context, make_sub_model_caller
"
```
Expected: no error. (Solver call sites are still broken — fixed in Tasks 8/11/12.)

- [ ] **Step 6: Commit**

```
git add Experiment/core_code/methods/base.py
git commit -m "refactor(base): InfraConfig drops shared backend; ctx.call_sub_model takes ModelConfig"
```

---

## Task 8: Rewrite `methods/tts_agent.py` solver

**Files:**
- Modify: `Experiment/core_code/methods/tts_agent.py`

This is the largest single-file change. The solver consolidates the per-variant dispatch logic from the deleted `tts_agent_multi.py` into the existing single-variant flow.

- [ ] **Step 1: Replace the explore tool definition with a variant-aware factory**

Currently `EXPLORE_TOOL` is a no-parameter constant. Replace it with a builder:

```python
def _build_explore_tool(variants: list) -> dict[str, Any]:
    """Build the explore tool schema. No parameter when length 1; variant
    enum otherwise."""
    if len(variants) == 1:
        return {
            "name": "explore",
            "description": (
                "Dispatch a fresh, independent solver to generate a new candidate answer. "
                "Takes no parameters -- a separate model will solve the problem from scratch."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        }
    labels = [v.label for v in variants]
    return {
        "name": "explore",
        "description": (
            "Dispatch a fresh, independent solver to generate a new candidate answer. "
            "You must specify which variant to use. Each variant has its own "
            "budget; do not call a variant whose budget is exhausted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "variant": {
                    "type": "string",
                    "enum": labels,
                    "description": "Which variant to dispatch for this explore call.",
                },
            },
            "required": ["variant"],
            "additionalProperties": False,
        },
    }
```

`INTEGRATE_TOOL` stays as it is.

- [ ] **Step 2: Add a per-variant caller registry inside `solve`**

Current `solve()` signature uses `orchestrator_model` / `explore_model` / `integrate_model` strings. Replace with the spec object:

```python
async def solve(
    infra: InfraConfig,
    problem: str,
    *,
    spec,  # methods.specs.TTSAgentSpec
    image_data_url: str | None = None,
    question_id: str | None = None,
    rollout_idx: int | None = None,
    temperature: float | None = None,
    **_extra,
) -> SolveResult:
    """Unified TTS agent solve.

    spec.explore length 1 = single-variant ATTS; length > 1 = old
    multi-model / effort runs. The orchestrator's explore tool exposes
    `variant: enum[<labels>]` when length > 1.
    """
    from methods.specs import TTSAgentSpec  # circular-import guard
    assert isinstance(spec, TTSAgentSpec), type(spec)

    max_iterations = sum(v.num_explores for v in spec.explore)
    # Override infra's max_iterations: yaml-side num_explores per variant
    # is the source of truth; eval.py builds InfraConfig with sum already,
    # but we recompute here so the solver is self-consistent.
    assert infra.max_iterations == max_iterations, (
        f"infra.max_iterations={infra.max_iterations} does not match "
        f"sum(num_explores)={max_iterations}"
    )

    user_message_text = build_user_message(
        problem, max_iterations,
        # Pass per-variant budgets into the user message for orchestrator
        # to see. Map label -> num_explores; build_user_message renders it
        # under the "Per-variant limits" line. Replaces the old
        # model_budgets / effort_budgets dicts.
        variant_budgets={v.label: v.num_explores for v in spec.explore} if len(spec.explore) > 1 else None,
    )
```

(`build_user_message` needs to accept `variant_budgets` — see Step 4.)

Build context using the orchestrator's backend/timeout (the orchestrator's run_tool_conversation uses ctx.backend, set later; explore variants get their own callers):

```python
    from prompts import select_orchestrator_prompt
    system_prompt = select_orchestrator_prompt(spec)

    ctx = create_solve_context(
        infra=infra,
        backend=spec.orchestrator.backend,
        timeout=spec.orchestrator.timeout,
        problem=problem,
        image_data_url=image_data_url,
        question_id=question_id,
        writer_system_prompt=system_prompt,
        writer_user_message=user_message_text,
        writer_header_lines=[
            f"**Orchestrator**: {spec.orchestrator.backend}/{spec.orchestrator.model}",
            *(f"**Variant {v.label}**: {v.model.backend}/{v.model.model} (n={v.num_explores})" for v in spec.explore),
            *([f"**Integrate**: {spec.integrate.model.backend}/{spec.integrate.model.model}"] if spec.integrate else []),
            f"**Max iterations**: {max_iterations}",
        ],
        writer_title_suffix="(unified)",
        rollout_idx=rollout_idx,
    )
    # Populate per-variant caps for budget guard
    ctx.state.explore.variant_caps = {v.label: v.num_explores for v in spec.explore}

    # Per-variant sub-model callers, keyed by label. Each caller is bound to
    # its variant's cache_dir + backend + timeout. Mirrors the per-alias
    # caller dict from the deleted tts_agent_multi.py:267.
    variant_callers: dict[str, Any] = {}
    for v in spec.explore:
        question_cache_dir = (v.cache_dir / question_id) if question_id else None
        variant_callers[v.label] = make_sub_model_caller(
            v.model.backend, question_cache_dir, infra.cache_only,
            traj_dir=ctx.traj_dir, timeout=v.model.timeout,
        )
```

- [ ] **Step 3: Replace `run_explore` and `run_integrate`**

```python
async def run_explore(ctx: SolveContext, spec, variant_callers: dict, label: str) -> str:
    """Run an explore call against a specific variant. Returns tool result text.

    `label` selects which ExploreVariant in spec.explore is used. Per-variant
    cache key is `f"explore_{in_variant_idx}"`."""
    if ctx.state.explore.is_exhausted:
        return (
            f"Explore quota exhausted ({ctx.state.explore.max_explores} explores already used). "
            f"You must call submit_answer with the best candidate from prior explores now."
        )
    if ctx.state.explore.variant_exhausted(label):
        cap = ctx.state.explore.variant_caps[label]
        return (
            f"Variant {label!r} budget exhausted ({cap} used). "
            f"Call explore with a different variant or submit_answer."
        )
    variant = next(v for v in spec.explore if v.label == label)
    in_idx = ctx.state.explore.variant_call_counts.get(label, 0) + 1
    ctx.state.explore.variant_call_counts[label] = in_idx

    user_msg = ctx.benchmark.build_explorer_message(ctx.state.problem)
    explorer_system_prompt = ctx.benchmark.get_explorer_system_prompt(variant.model.backend)
    explore_schema = ctx.benchmark.get_explore_schema()

    caller = variant_callers[label]
    result, traj, cost, usage, duration = await caller(
        explorer_system_prompt, user_msg, ctx.image_data_url,
        variant.model.model, explore_schema,
        cache_key=f"explore_{in_idx}",
        writer=TrajectoryWriter.noop(),
        budget_tokens=variant.model.budget_tokens,
        effort=variant.model.effort,
        sampling=variant.model.vllm_sampling.model_dump() if variant.model.vllm_sampling else None,
        provider_order=variant.model.openrouter_provider_order,
        provider_allow_fallbacks=variant.model.openrouter_provider_allow_fallbacks,
    )
    return process_explore_result(
        ctx, result, cost, usage,
        model_label=label if len(spec.explore) > 1 else "",
    )


async def run_integrate(ctx: SolveContext, spec) -> str:
    """Run integrate against the cached integrate role."""
    assert spec.integrate is not None, "integrate called when spec.integrate is None"
    assert ctx.state.candidates, "integrate called with no candidates"
    integrator_system_prompt = ctx.benchmark.get_integrator_system_prompt(spec.integrate.model.backend)
    integrate_schema = ctx.benchmark.get_integrate_schema()
    user_msg = ctx.benchmark.build_integrator_message(ctx.state.problem, ctx.state.candidates)

    # integrate has its own cache_dir so we build a one-shot caller.
    question_cache_dir = (spec.integrate.cache_dir / ctx.question_id) if ctx.question_id else None
    integrate_caller = make_sub_model_caller(
        spec.integrate.model.backend, question_cache_dir, ctx.cache_only,
        traj_dir=ctx.traj_dir, timeout=spec.integrate.model.timeout,
    )
    result, _traj, cost, usage, _dur = await integrate_caller(
        integrator_system_prompt, user_msg, ctx.image_data_url,
        spec.integrate.model.model, integrate_schema,
        cache_key=f"integrate_{ctx.state.explore.call_count + 1}",
        budget_tokens=spec.integrate.model.budget_tokens,
        effort=spec.integrate.model.effort,
        sampling=spec.integrate.model.vllm_sampling.model_dump() if spec.integrate.model.vllm_sampling else None,
        provider_order=spec.integrate.model.openrouter_provider_order,
        provider_allow_fallbacks=spec.integrate.model.openrouter_provider_allow_fallbacks,
    )
    ctx.cost.add(cost, usage, component="integrator")
    final = ctx.benchmark.get_answer_from_integrate(result)
    ctx.state.final_answer = final
    ctx.state.final_reasoning = result.get("reasoning")
    ctx.state.final_analysis = result.get("analysis")
    return "Final answer recorded."
```

- [ ] **Step 4: Update `prompts.build_user_message`**

In `prompts.py`, replace the `model_budgets` / `effort_budgets` / `exploration_effort` parameters with one `variant_budgets`:

```python
def build_user_message(
    problem: str,
    max_iterations: int,
    variant_budgets: dict[str, int] | None = None,
) -> str:
    """Build the initial user message for the orchestrator."""
    budget_lines = f"You have up to {max_iterations} explore rounds in total."
    if variant_budgets:
        per_variant = ", ".join(f"{lab}: max {n}" for lab, n in variant_budgets.items())
        budget_lines += f"\nPer-variant limits: {per_variant}."
        budget_lines += "\nOnce a variant's limit is reached, you cannot use it again."
    budget_lines += "\nBegin by calling explore to dispatch the first solver."
    return (
        f"## Problem\n\n{problem}\n\n"
        f"## Budget\n\n"
        f"{budget_lines}"
    )
```

- [ ] **Step 5: Replace `_run_orchestrator`**

```python
async def _run_orchestrator(
    ctx: SolveContext,
    spec,
    variant_callers: dict,
    user_message_text: str,
    system_prompt: str,
    temperature: float | None = None,
) -> None:
    backend_mod = import_module(f"backends.{spec.orchestrator.backend}")

    async def tool_handler(name: str, args: dict) -> tuple[str, bool]:
        if name == "explore":
            if len(spec.explore) == 1:
                # No `variant` parameter exposed to orchestrator in this case.
                label = spec.explore[0].label
            else:
                # Length>1: `variant` is required by the tool schema. Fail
                # loud rather than silently defaulting to spec.explore[0] —
                # a missing field means the orchestrator/backend dropped it
                # and we want the operator to see the gap, not silently
                # bias every multi-variant run toward the first variant.
                assert "variant" in args, (
                    f"explore tool called without `variant` param under "
                    f"length>1 spec; args={args!r}"
                )
                label = args["variant"]
            n_before = len(ctx.state.candidates)
            result_text = await run_explore(ctx, spec, variant_callers, label)
            if len(ctx.state.candidates) > n_before:
                cand = ctx.state.candidates[-1]
                _log_round(ctx, RoundLog(
                    round_num=ctx.state.explore.used,
                    action="explore",
                    tool_input={
                        "variant": label,
                        "answer": cand.answer,
                        "reasoning": cand.reasoning,
                        "approach": cand.approach,
                        "confidence": cand.confidence,
                        "cost_usd": cand.cost_usd,
                    },
                ))
            return result_text, False
        elif name == "integrate":
            result_text = await run_integrate(ctx, spec)
            _log_round(ctx, RoundLog(
                round_num=ctx.state.explore.call_count + 1,
                action="integrate",
                tool_input={
                    "final_answer": ctx.state.final_answer,
                    "reasoning": ctx.state.final_reasoning,
                    "analysis": ctx.state.final_analysis,
                },
            ))
            return result_text, True
        else:
            assert False, f"Unknown tool: {name}"

    explore_tool = _build_explore_tool(spec.explore)
    if spec.integrate is not None:
        tools = [explore_tool, INTEGRATE_TOOL]
        output_format = None
    else:
        tools = [explore_tool]
        output_format = {"type": "json_schema", "schema": ctx.benchmark.get_explore_schema()}

    cost, usage, exit_reason = await backend_mod.run_tool_conversation(
        system_prompt=system_prompt,
        user_message=user_message_text,
        image_data_url=ctx.image_data_url,
        model=spec.orchestrator.model,
        tools=tools,
        max_turns=ctx.state.explore.max_explores + 2,
        tool_handler=tool_handler,
        effort=spec.orchestrator.effort,
        output_format=output_format,
        writer=ctx.writer,
        on_structured_output=make_structured_output_handler(ctx),
        max_output_tokens=spec.orchestrator.max_output_tokens,
        temperature=temperature,
        sampling=spec.orchestrator.vllm_sampling.model_dump() if spec.orchestrator.vllm_sampling else None,
        provider_order=spec.orchestrator.openrouter_provider_order,
        provider_allow_fallbacks=spec.orchestrator.openrouter_provider_allow_fallbacks,
    )
    ctx._exit_reason = exit_reason
    ctx.cost.add(cost, usage, component="orchestrator")
    ctx.writer.write_session_summary(ctx.cost.total_cost_usd, {
        "input_tokens": ctx.cost.total_input_tokens,
        "output_tokens": ctx.cost.total_output_tokens,
    })
```

- [ ] **Step 6: Wire `solve` to `_run_orchestrator`**

Continue inside `solve` after the `variant_callers` setup:

```python
    await _run_orchestrator(
        ctx, spec, variant_callers, user_message_text, system_prompt,
        temperature=temperature,
    )

    if ctx.state.final_answer is None:
        ctx.state.final_answer = ""

    logger.info(
        f"\nTotal cost: ${ctx.cost.total_cost_usd}"
        f" (input: {ctx.cost.total_input_tokens}, output: {ctx.cost.total_output_tokens})"
    )
    result = ctx.result(ctx.state.final_answer)
    result.exit_reason = getattr(ctx, "_exit_reason", "incomplete")
    return result
```

- [ ] **Step 7: Verify file syntax**

```
conda run -n explain --no-capture-output python -c "import methods.tts_agent"
```
Expected: no errors.

- [ ] **Step 8: Commit**

```
git add Experiment/core_code/methods/tts_agent.py Experiment/core_code/prompts.py
git commit -m "refactor(tts_agent): unified solver consumes spec; per-variant dispatch"
```

---

## Task 9: Delete `tts_agent_multi.py` and `tts_agent_effort.py`

**Files:**
- Delete: `Experiment/core_code/methods/tts_agent_multi.py`
- Delete: `Experiment/core_code/methods/tts_agent_effort.py`

- [ ] **Step 1: Delete the files**

```
rm /data3/peijia/dr-claw/Explain/Experiment/core_code/methods/tts_agent_multi.py
rm /data3/peijia/dr-claw/Explain/Experiment/core_code/methods/tts_agent_effort.py
```

- [ ] **Step 2: Find any remaining imports**

```
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
grep -rn "from methods.tts_agent_multi\|from methods.tts_agent_effort\|import methods.tts_agent_multi\|import methods.tts_agent_effort" --include="*.py" || echo "no remaining imports"
```
Expected: `no remaining imports` (registry.py imports are removed in Task 10).

If anything shows up that's not registry.py (handled in Task 10), stop and fix it before continuing.

- [ ] **Step 3: Commit**

```
git add -A Experiment/core_code/methods/
git commit -m "refactor(methods): delete tts_agent_multi and tts_agent_effort"
```

---

## Task 10: Update `methods/registry.py`

**Files:**
- Modify: `Experiment/core_code/methods/registry.py`

- [ ] **Step 1: Update imports**

Replace the import block at top:

```python
from methods.specs import (
    MethodSpec, TTSAgentSpec,
    SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
    RerankSpec, StandaloneIntegratorSpec,
)
```

(Removed `TTSAgentMultiSpec`, `TTSAgentEffortSpec`.)

- [ ] **Step 2: Delete `TTSAgentMultiMethod` and `TTSAgentEffortMethod` classes**

Remove the two class definitions.

- [ ] **Step 3: Update `TTSAgentMethod` and the rest**

Each `MethodConfig.derive_evaluate_args` previously returned a flat dict. Replace with a single `spec` pass-through; eval.py now consumes the spec directly. Sketch:

```python
class MethodConfig(ABC):
    name: str
    cache_only: bool
    pre_flight_check: bool = False
    pre_filter_by_cache: bool = False
    supports_num_rollouts: bool = False
    consumes_sampling_block: bool = False

    @abstractmethod
    def build_solve_fn(self, spec: MethodSpec) -> Callable: ...

    def filter_rows(self, rows, cache_dir, benchmark): ...   # unchanged
    def preflight(self, rows, cache_dir, num_explores, num, benchmark): ...  # unchanged


class TTSAgentMethod(MethodConfig):
    name = "tts-agent"
    cache_only = True
    pre_flight_check = True
    supports_num_rollouts = True
    consumes_sampling_block = True

    def build_solve_fn(self, spec: TTSAgentSpec):
        from methods.tts_agent import solve
        import functools
        return functools.partial(solve, spec=spec)


class SelfRefineMethod(MethodConfig):
    name = "self-refine"
    cache_only = False
    pre_flight_check = True

    def build_solve_fn(self, spec: SelfRefineSpec):
        from methods.self_refine import solve
        import functools
        return functools.partial(solve, spec=spec)


class SocraticSelfRefineMethod(MethodConfig):
    name = "socratic-self-refine"
    cache_only = False
    pre_flight_check = True

    def build_solve_fn(self, spec: SocraticSelfRefineSpec):
        from methods.socratic_self_refine import solve
        import functools
        return functools.partial(solve, spec=spec)


class BudgetForcingMethod(MethodConfig):
    name = "budget-forcing"
    cache_only = False
    pre_flight_check = True

    def build_solve_fn(self, spec: BudgetForcingSpec):
        from methods.budget_forcing import solve
        import functools
        return functools.partial(solve, spec=spec)


class RerankMethod(MethodConfig):
    name = "rerank"
    cache_only = True
    pre_filter_by_cache = True

    def build_solve_fn(self, spec: RerankSpec):
        from methods.reward_rerank import solve
        import functools
        return functools.partial(solve, reward_model_name=spec.reward_model)


class StandaloneIntegratorMethod(MethodConfig):
    name = "standalone-integrator"
    cache_only = True
    pre_filter_by_cache = True

    def build_solve_fn(self, spec: StandaloneIntegratorSpec):
        from methods.standalone_integrator import solve
        import functools
        return functools.partial(solve, spec=spec)


METHODS: dict[str, type[MethodConfig]] = {
    "tts-agent": TTSAgentMethod,
    "self-refine": SelfRefineMethod,
    "socratic-self-refine": SocraticSelfRefineMethod,
    "budget-forcing": BudgetForcingMethod,
    "rerank": RerankMethod,
    "standalone-integrator": StandaloneIntegratorMethod,
}
```

(Note `derive_evaluate_args` removed from base class — eval.py reads spec fields directly going forward.)

- [ ] **Step 4: Verify**

```
conda run -n explain --no-capture-output python -c "
from methods.registry import METHODS, get_method
print(sorted(METHODS.keys()))
"
```
Expected: `['budget-forcing', 'rerank', 'self-refine', 'socratic-self-refine', 'standalone-integrator', 'tts-agent']`.

- [ ] **Step 5: Commit**

```
git add Experiment/core_code/methods/registry.py
git commit -m "refactor(registry): drop multi/effort; build_solve_fn uses functools.partial(spec=spec)"
```

---

## Task 11: Update `self_refine.py` / `socratic_self_refine.py` / `budget_forcing.py`

**Files:**
- Modify: `Experiment/core_code/methods/self_refine.py`
- Modify: `Experiment/core_code/methods/socratic_self_refine.py`
- Modify: `Experiment/core_code/methods/budget_forcing.py`

For each file the existing `solve(infra, problem, *, explore_model, ..., cache_dir, num_explores, ...)` signature needs to change to consume the new `spec.explore: ExploreVariant`.

- [ ] **Step 1: Read each file's current `solve` signature**

```
grep -n "^async def solve" /data3/peijia/dr-claw/Explain/Experiment/core_code/methods/{self_refine,socratic_self_refine,budget_forcing}.py
```

- [ ] **Step 2: Update each to accept `spec`**

Pattern (apply to all three):

```python
async def solve(
    infra: InfraConfig,
    problem: str,
    *,
    spec,  # methods.specs.SelfRefineSpec / SocraticSelfRefineSpec / BudgetForcingSpec
    image_data_url: str | None = None,
    question_id: str | None = None,
    rollout_idx: int | None = None,
    **_extra,
) -> SolveResult:
    variant = spec.explore
    # backend / model / num_explores / cache_dir come from variant
    ...
```

Each method's body that previously called `ctx.call_sub_model(model=explore_model, ...)` now passes `model_cfg=variant.model`. The `cache_dir` was previously read from `infra.cache_dir`; now it's `variant.cache_dir`.

For each file, the concrete kwargs to remove from the `solve(...)` signature are:
`backend`, `explore_model`, `cache_dir`, `num_explores`, `budget_tokens`, `effort`, `timeout`, `max_output_tokens`, `sampling`, `provider_order`, `provider_allow_fallbacks`, `orchestrator_model`, `integrate_model`. Add a single `spec` kwarg. Then:

1. Replace bare-string model usage `model=explore_model` with `model=variant.model.model`; pull `variant = spec.explore` once at top of body.
2. Replace `num_explores` reads with `variant.num_explores`.
3. Replace `cache_dir` reads with `variant.cache_dir`. The `infra.cache_dir` is set by eval.py Task 14 to `variant.cache_dir` already, so `create_solve_context(infra=infra, ...)` is consistent — no extra threading needed.
4. Each `ctx.call_sub_model(model=variant.model.model, ...)` call site needs to pass `model_cfg=variant.model` (per Task 7's refactored signature). Audit the inner trajectory-write helpers and ensure `variant.model.backend` flows wherever the old `infra.backend` was used (e.g. `benchmark.get_explorer_system_prompt(variant.model.backend)`).

(Concretely, this tracks `infra.cache_dir` already being a Path argument in `create_solve_context`. eval.py in Task 14 sets `infra.cache_dir = spec.explore.cache_dir` for self-refine / socratic / budget-forcing methods.)

- [ ] **Step 3: Verify each compiles**

```
conda run -n explain --no-capture-output python -c "
import methods.self_refine, methods.socratic_self_refine, methods.budget_forcing
"
```

- [ ] **Step 4: Commit**

```
git add Experiment/core_code/methods/self_refine.py Experiment/core_code/methods/socratic_self_refine.py Experiment/core_code/methods/budget_forcing.py
git commit -m "refactor(methods): self-refine/socratic/budget-forcing consume ExploreVariant"
```

---

## Task 12: Update `methods/standalone_integrator.py`

**Files:**
- Modify: `Experiment/core_code/methods/standalone_integrator.py`

- [ ] **Step 1: Update `solve` signature**

```python
async def solve(
    infra: InfraConfig,
    problem: str,
    *,
    spec,  # methods.specs.StandaloneIntegratorSpec
    image_data_url: str | None = None,
    question_id: str | None = None,
    rollout_idx: int | None = None,
    **_extra,
) -> SolveResult:
    integrate_role = spec.integrate
    num_explores = spec.num_explores
    # cache_dir for cached candidates is integrate_role.cache_dir? No -
    # cached explores live under the per-benchmark base cache_dir; this
    # method reads them via load_cached_candidates. The ROLE_SLOT.cache_dir
    # is for the integrator's own output. Audit:
    #   1. Where does this method call load_cached_candidates? → that path
    #      is the cached-explore base, NOT integrate_role.cache_dir.
    #   2. integrate_role.cache_dir is where integrator output lands.
```

- [ ] **Step 2: Audit the existing implementation**

Read the file and identify:
- Where `cache_dir` was used to find cached explores (passed to `load_cached_candidates`)
- Where `cache_dir` was used to write integrator output

For (1), the cached-explore directory was being passed in via `infra.cache_dir`. Keep using `infra.cache_dir` for that — eval.py in Task 14 sets it.
For (2), use `integrate_role.cache_dir` for the integrator output.

This file's structure may be simple enough that the change is local; cite the relevant lines once you read it.

- [ ] **Step 3: Verify**

```
conda run -n explain --no-capture-output python -c "import methods.standalone_integrator"
```

- [ ] **Step 4: Commit**

```
git add Experiment/core_code/methods/standalone_integrator.py
git commit -m "refactor(standalone-integrator): consume RoleSlot for integrator output"
```

---

## Task 13: `backends/openrouter.py` per-call provider routing

**Files:**
- Modify: `Experiment/core_code/backends/openrouter.py`
- Modify: `Experiment/core_code/backends/{codex,claude,vllm}.py` (accept new kwargs as no-ops)

- [ ] **Step 1: Delete the module-globals**

In `backends/openrouter.py`, remove:
- Lines 67-68 (`_PROVIDER_ORDER`, `_PROVIDER_ALLOW_FALLBACKS`).
- Lines 71-82 (`set_provider` function).
- Lines 85-91 (`_maybe_inject_provider` function).

- [ ] **Step 2: Add `provider_order` / `provider_allow_fallbacks` kwargs to both call sites**

Update `call_sub_model` signature (around line 148):

```python
async def call_sub_model(
    system_prompt: str,
    user_message: str,
    image_data_url: str | None,
    model: str,
    output_schema: dict[str, Any],
    writer,
    budget_tokens: int = 32000,
    effort: str | None = None,
    sampling: dict | None = None,
    provider_order: list[str] | None = None,        # NEW
    provider_allow_fallbacks: bool = True,          # NEW
) -> tuple[dict[str, Any], str, float, dict[str, Any]]:
    ...
    # In the block that builds extra_body (around line 184-191), replace
    # _maybe_inject_provider(extra_body) with:
    if provider_order:
        extra_body["provider"] = {
            "order": provider_order,
            "allow_fallbacks": provider_allow_fallbacks,
        }
```

Update `run_tool_conversation` signature (around line 331):

```python
async def run_tool_conversation(
    *,
    system_prompt: str,
    user_message: str,
    image_data_url: str | None,
    model: str,
    tools: list[dict[str, Any]],
    max_turns: int,
    tool_handler,
    effort: str | None = None,
    output_format: dict[str, Any] | None = None,
    writer=None,
    on_structured_output=None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    sampling: dict | None = None,
    provider_order: list[str] | None = None,        # NEW
    provider_allow_fallbacks: bool = True,          # NEW
):
    ...
    # Replace _maybe_inject_provider(extra_body) (around line 375) with the
    # same pattern as call_sub_model.
```

- [ ] **Step 3: Add the kwargs to the other backends as no-op accepts**

`backends/claude.py`, `backends/codex.py`, `backends/vllm.py` all need to accept `provider_order` and `provider_allow_fallbacks` kwargs in their `call_sub_model` and `run_tool_conversation`, then ignore them (since only OpenRouter routes through providers).

For each, add the two kwargs to the signature; do not use them. This keeps the `make_sub_model_caller` inner `call` agnostic — the same kwargs flow through whichever backend is selected.

- [ ] **Step 4: Verify**

```
conda run -n explain --no-capture-output python -c "
import backends.openrouter, backends.claude, backends.vllm, backends.codex
print('backends loaded')
"
```
Expected: `backends loaded`.

```
conda run -n explain --no-capture-output python -c "
from backends.openrouter import set_provider
" 2>&1 | grep -q "ImportError\|cannot import" && echo OK_DELETED || echo STILL_EXISTS
```
Expected: `OK_DELETED`.

- [ ] **Step 5: Commit**

```
git add Experiment/core_code/backends/openrouter.py Experiment/core_code/backends/claude.py Experiment/core_code/backends/vllm.py Experiment/core_code/backends/codex.py
git commit -m "refactor(backends): per-call provider routing; delete openrouter.set_provider"
```

---

## Task 14: Rewrite `eval.py` `InfraConfig` flow

**Files:**
- Modify: `Experiment/core_code/eval.py`

- [ ] **Step 1: Replace the `InfraConfig` instantiation block**

Find lines around 712-793 (the section that pulls `runtime = method.derive_evaluate_args(...)` and builds `InfraConfig`). Replace with:

```python
    method = get_method(cfg.method.name)
    solve = method.build_solve_fn(cfg.method)

    # Compute per-method total iterations + the cache_dir we pre-flight against.
    # Different specs put cache_dir in different places; centralize the lookup
    # here so InfraConfig can stay slim and method-shape-agnostic.
    spec = cfg.method
    if isinstance(spec, TTSAgentSpec):
        # length-1 single-variant: pre-flight that one variant's cache.
        # length-3 multi/effort: pre-flight is per-variant; for the banner
        # check we use the first variant's directory, mirroring the
        # behavior of the old TTSAgentMultiMethod which only checked one.
        # Future improvement: extend method.preflight() to enumerate variants.
        cache_dir = spec.explore[0].cache_dir
        num_explores = sum(v.num_explores for v in spec.explore)
        num_rollouts = spec.num_rollouts
    elif isinstance(spec, (SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec)):
        cache_dir = spec.explore.cache_dir
        num_explores = spec.explore.num_explores
        num_rollouts = 1
    elif isinstance(spec, StandaloneIntegratorSpec):
        # Cached explore base lives at integrate.cache_dir's PARENT? No —
        # standalone-integrator's cache_dir for cached explores has historically
        # been the same as the eval's cache_dir; we keep that contract by
        # treating spec.integrate.cache_dir as the pre-existing explore cache.
        cache_dir = spec.integrate.cache_dir
        num_explores = spec.num_explores
        num_rollouts = 1
    elif isinstance(spec, RerankSpec):
        cache_dir = spec.cache_dir
        num_explores = 8
        num_rollouts = 1
    else:
        raise AssertionError(f"unknown spec type: {type(spec).__name__}")

    logger.info(f"Loading {benchmark.name.upper()} dataset...")
    all_rows = benchmark.load_dataset()
    logger.info(f"Loaded {len(all_rows)} total questions")

    filtered = benchmark.filter_dataset(all_rows, **bench_filters)
    logger.info(f"Filtered to {len(filtered)} questions")
    if not filtered:
        logger.info("No questions match the filter criteria.")
        return

    if cfg.shuffle:
        import random
        random.seed(cfg.seed)
        random.shuffle(filtered)
    if cfg.skip > 0:
        logger.info(f"Skipping first {cfg.skip} questions")
        filtered = filtered[cfg.skip:]

    filtered = method.filter_rows(filtered, cache_dir, benchmark)
    method.preflight(filtered, cache_dir, num_explores, cfg.num, benchmark)

    if num_rollouts > 1:
        # ... (existing rejection-sampling expansion logic, unchanged)
        ...
        effective_num = len(expanded)
    else:
        effective_num = cfg.num

    infra = InfraConfig(
        max_iterations=num_explores,
        cache_dir=cache_dir,
        cache_only=method.cache_only,
        benchmark=benchmark,
        logger=None,
        enable_integrate=isinstance(spec, TTSAgentSpec) and spec.integrate is not None,
    )

    # Note: no openrouter.set_provider call — provider routing is now per-role
    # via ModelConfig.openrouter_provider_order, threaded through call sites
    # in tts_agent.py and friends.

    await evaluate(
        infra=infra,
        rows=filtered,
        solve_fn=solve,
        num=effective_num,
        num_workers=cfg.num_workers,
        resume_run_dir=cfg.resume,
        log_dir=cfg.log_dir,
        # No more orchestrator_model / explore_model / integrate_model /
        # cache_dirs_multi: solve_fn is partialed with spec=spec via
        # registry.build_solve_fn.
        dataset_config={
            "benchmark": benchmark.name,
            **bench_filters,
            "seed": cfg.seed,
            "shuffle": cfg.shuffle,
            "num": cfg.num,
            "num_rollouts": num_rollouts,
        },
    )
```

- [ ] **Step 2: Update `evaluate()` signature**

The existing `evaluate(infra, rows, solve_fn, num, num_workers, resume_run_dir, log_dir, orchestrator_model, explore_model, integrate_model, dataset_config, cache_dirs_multi, sampling)` needs the model-name and cache_dirs_multi kwargs removed. Find the function definition and:

1. Remove `orchestrator_model`, `explore_model`, `integrate_model`, `cache_dirs_multi`, `sampling` parameters.
2. Inside the function, every reference to these kwargs is gone — they previously flowed into `solve_fn(...)` as kwargs, but now `solve_fn` is partialed with `spec=spec` and reads everything from there.

Audit any logging that printed `orchestrator_model` etc. in the run banner — replace with reads from the spec via the partialed `solve_fn` introspection (or pass the spec into evaluate explicitly as a new kwarg `spec=spec`).

Cleanest path: add `spec` as an evaluate() kwarg, log spec details from there:

```python
async def evaluate(
    *,
    infra: InfraConfig,
    rows: list[dict],
    solve_fn,
    spec,  # for logging only
    num: int | None,
    num_workers: int,
    resume_run_dir: str | None,
    log_dir: str,
    dataset_config: dict,
):
    ...
    # Banner: use spec for human-readable summary
    ...
```

And in `async_main`'s call to `evaluate`, pass `spec=cfg.method`.

- [ ] **Step 3: Update imports at top of eval.py**

```python
from methods.specs import (
    SamplingConfig, TTSAgentSpec,
    SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
    StandaloneIntegratorSpec, RerankSpec,
)
```

- [ ] **Step 4: Verify import**

```
conda run -n explain --no-capture-output python -c "from eval import EvalConfig, load_config, async_main"
```
Expected: no error.

- [ ] **Step 5: Commit**

```
git add Experiment/core_code/eval.py
git commit -m "refactor(eval): InfraConfig slim; spec-driven evaluate; drop set_provider"
```

---

## Task 15: Rebuild `precache_explores.py`

**Files:**
- Modify: `Experiment/core_code/precache_explores.py`

- [ ] **Step 1: Replace `PrecacheConfig`**

```python
class PrecacheConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": False}

    benchmark: BenchmarkSpec
    explore: ExploreVariant   # carries backend, model, cache_dir, num_explores

    # Top-level precache concerns (parallelism, dataset slicing) stay flat
    # because they are not part of the explore call itself.
    num_workers: int = 1
    num: int | None = None
    skip: int = 0
    seed: int = 42
    shuffle: bool = False
```

`backend`, `explore_model`, `cache_dir`, `num_explores`, `budget_tokens`, `effort`, `explore_timeout`, `provider_order`, `provider_allow_fallbacks`, `sampling` — all gone (folded into `explore.model`).

- [ ] **Step 2: Update imports**

```python
from methods.specs import ExploreVariant
```

- [ ] **Step 3: Update the `precache(...)` function signature**

```python
async def precache(
    benchmark: BenchmarkConfig,
    rows: list[dict],
    variant: ExploreVariant,    # was: cache_dir, num_explores, backend, model, budget_tokens, effort, explore_timeout, sampling, ...
    num_workers: int,
    num: int | None = None,
) -> None:
    ...
```

Inside, read everything from `variant.model.*`. Replace direct uses of:
- `backend` → `variant.model.backend`
- `model` → `variant.model.model`
- `budget_tokens` → `variant.model.budget_tokens`
- `effort` → `variant.model.effort`
- `explore_timeout` → `variant.model.timeout`
- `sampling` → `variant.model.vllm_sampling.model_dump() if variant.model.vllm_sampling else None`
- `provider_order` → `variant.model.openrouter_provider_order`
- `provider_allow_fallbacks` → `variant.model.openrouter_provider_allow_fallbacks`
- `cache_dir` → `variant.cache_dir`
- `num_explores` → `variant.num_explores`

The `make_sub_model_caller` call inside the worker now receives the per-call provider kwargs:

```python
sub_model_fn = make_sub_model_caller(
    variant.model.backend, cache_dir=question_cache_dir, cache_only=False,
    traj_dir=question_cache_dir, timeout=variant.model.timeout,
)
result, traj, cost_usd, usage, duration = await sub_model_fn(
    system_prompt=explorer_prompt,
    user_message=input_text,
    image_data_url=image_data_url,
    model=variant.model.model,
    output_schema=explore_schema,
    cache_key=f"explore_{explore_idx}",
    budget_tokens=variant.model.budget_tokens,
    effort=variant.model.effort,
    sampling=variant.model.vllm_sampling.model_dump() if variant.model.vllm_sampling else None,
    provider_order=variant.model.openrouter_provider_order,
    provider_allow_fallbacks=variant.model.openrouter_provider_allow_fallbacks,
)
```

- [ ] **Step 4: Update `main()`**

```python
def main():
    setup_console_logging()
    from eval import load_config

    parser = argparse.ArgumentParser(description="Pre-cache explore results")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(config_path=args.config, schema=PrecacheConfig)

    bench_dump = cfg.benchmark.model_dump()
    judge_spec = bench_dump.pop("judge", None)
    benchmark = get_benchmark(cfg.benchmark.name, judge_spec=judge_spec)
    bench_filters = cfg.benchmark.model_dump(exclude={"name", "judge"}, exclude_defaults=True)

    logger.info(f"Loading {benchmark.name.upper()} dataset...")
    all_rows = benchmark.load_dataset()
    filtered = benchmark.filter_dataset(all_rows, **bench_filters)
    if cfg.shuffle:
        import random
        random.seed(cfg.seed)
        random.shuffle(filtered)
    cfg.explore.cache_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(precache(
        benchmark=benchmark,
        rows=filtered,
        variant=cfg.explore,
        num_workers=cfg.num_workers,
        num=cfg.num,
    ))
```

(The `set_provider(...)` call is gone — provider routing rides on `variant.model.openrouter_provider_order`.)

- [ ] **Step 5: Verify**

```
conda run -n explain --no-capture-output python -c "from precache_explores import PrecacheConfig, precache, main"
```

- [ ] **Step 6: Commit**

```
git add Experiment/core_code/precache_explores.py
git commit -m "refactor(precache): PrecacheConfig.explore: ExploreVariant"
```

---

## Task 16: Migrate vLLM judge cache directories

**Files:**
- Create: `Experiment/core_code/scripts/maintenance/migrate_vllm_judge_cache.py`

- [ ] **Step 1: Verify the cache directory structure exists on disk**

```
find /data3/peijia/dr-claw/Explain/Experiment/analysis -type d -name "vllm__*" | head -5
```

If empty: this task becomes a no-op. If non-empty: continue.

- [ ] **Step 2: Write the one-shot migration script**

Create `Experiment/core_code/scripts/maintenance/migrate_vllm_judge_cache.py`:

```python
#!/usr/bin/env python
"""One-shot data migration: rename vllm-judge config field `sampling`
to `vllm_sampling` to match the new ModelConfig schema.

Run ONCE before the ModelConfig refactor merges. The new
benchmarks/specs.py + benchmarks.base.find_cached_judge raises a
hard RuntimeError if a stored config.json carries an unknown field
(stored-superset-of-requested), so failure to run this script
manifests as eval crashes the moment a vllm judge cache hit is
attempted.

Idempotent: re-running on already-migrated configs is a no-op.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ANALYSIS_DIR = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis")


def main() -> None:
    cfg_paths = list(ANALYSIS_DIR.rglob("judges/vllm__*/config.json"))
    if not cfg_paths:
        print("No vllm judge config.json files found; nothing to migrate.")
        return

    migrated = 0
    skipped = 0
    for p in cfg_paths:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        if "sampling" in cfg and "vllm_sampling" not in cfg:
            cfg["vllm_sampling"] = cfg.pop("sampling")
            p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            migrated += 1
            print(f"migrated: {p}")
        else:
            skipped += 1
    print(f"\nDone. migrated={migrated} skipped={skipped} total={len(cfg_paths)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run dry-run by listing affected files**

```
find /data3/peijia/dr-claw/Explain/Experiment/analysis -path "*judges/vllm__*/config.json" | head -10
find /data3/peijia/dr-claw/Explain/Experiment/analysis -path "*judges/vllm__*/config.json" | wc -l
```
Note the count.

- [ ] **Step 4: Run the migration**

```
conda run -n explain --no-capture-output python /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/maintenance/migrate_vllm_judge_cache.py
```
Expected output: `migrated=N skipped=0 total=N` where N matches Step 3.

- [ ] **Step 5: Spot-check a migrated file**

```
find /data3/peijia/dr-claw/Explain/Experiment/analysis -path "*judges/vllm__*/config.json" | head -1 | xargs -I{} python -c "import json; print(list(json.load(open('{}')).keys()))"
```
Expected: list contains `vllm_sampling` and not `sampling`.

- [ ] **Step 6: Commit the script**

```
git add Experiment/core_code/scripts/maintenance/migrate_vllm_judge_cache.py
git commit -m "chore(maintenance): one-shot vllm judge cache field rename"
```

---

## Task 17: Migrate yamls

**Files:**
- Create: `Experiment/core_code/scripts/maintenance/migrate_yamls_to_modelconfig.py`

- [ ] **Step 1: Write the migration script — full skeleton**

Create `Experiment/core_code/scripts/maintenance/migrate_yamls_to_modelconfig.py` with this complete content:

```python
#!/usr/bin/env python
"""One-shot yaml migration: rewrite all yamls under scripts/**/*.yaml to the
new ModelConfig-based schema.

Per-file: detect shape, apply the relevant rewrite, validate via pydantic,
write back (preserving comments/ordering via ruamel.yaml). Idempotent:
already-migrated files are skipped on the basis of marker fields
(`orchestrator_prompt` for tts-agent, presence of `explore.model` for
self-refine etc., presence of `judge.backend` for any benchmark).

Run modes:
  --dry-run --path <one yaml>   prints rewritten yaml to stdout
  --bulk --globs <pattern>      rewrites every match in place; aborts on
                                first validation failure (rolling back .bak)
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import traceback
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[3]   # repo root
sys.path.insert(0, str(ROOT / "Experiment" / "core_code"))

from eval import EvalConfig, load_config                # noqa: E402
from precache_explores import PrecacheConfig            # noqa: E402

YAML_RW = YAML(typ="rt")
YAML_RW.preserve_quotes = True
YAML_RW.indent(mapping=2, sequence=4, offset=2)

# Old multi-model alias-to-canonical-model-id mapping (from the deleted
# methods/tts_agent_multi.py:MODEL_ALIASES).
_MODEL_ALIASES = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

# Backend-config keys carried over to the new ModelConfig.
_BACKEND_PASSTHROUGH = (
    "budget_tokens", "effort", "timeout", "max_output_tokens",
)


def _build_model_block(old_backend: dict, model_name: str) -> dict:
    """Convert old method.backend dict + bare model string into a new ModelConfig dict."""
    out = {"backend": old_backend["name"], "model": model_name}
    for k in _BACKEND_PASSTHROUGH:
        if k in old_backend:
            out[k] = old_backend[k]
    if old_backend.get("name") == "openrouter":
        if "provider_order" in old_backend and old_backend["provider_order"] is not None:
            out["openrouter_provider_order"] = old_backend["provider_order"]
        if "provider_allow_fallbacks" in old_backend:
            out["openrouter_provider_allow_fallbacks"] = old_backend["provider_allow_fallbacks"]
    return out


def _migrate_judge(bench: dict) -> None:
    """In-place: rename judge.name -> judge.backend, judge.sampling -> judge.vllm_sampling."""
    judge = bench.get("judge")
    if judge is None or "backend" in judge:
        return  # absent or already migrated
    judge["backend"] = judge.pop("name")
    if judge["backend"] == "vllm" and "sampling" in judge:
        judge["vllm_sampling"] = judge.pop("sampling")


def _migrate_tts_agent(method: dict) -> dict:
    """Rewrite a tts-agent (single-variant) method block."""
    backend = method["backend"]
    model_block_orchestrator = _build_model_block(backend, method["orchestrator_model"])
    if method.get("orchestrator_effort") is not None:
        model_block_orchestrator["effort"] = method["orchestrator_effort"]
    explore_variant = {
        "label": "default",
        "model": _build_model_block(backend, method["explore_model"]),
        "cache_dir": method["cache_dir"],
        "num_explores": method.get("num_explores", 8),
    }
    new = {
        "name": "tts-agent",
        "orchestrator_prompt": "single",
        "orchestrator": model_block_orchestrator,
        "explore": [explore_variant],
    }
    if not method.get("no_integrate", False):
        new["integrate"] = {
            "model": _build_model_block(backend, method["integrate_model"]),
            "cache_dir": method["cache_dir"],   # historically shared with explore
        }
    if method.get("num_rollouts", 1) != 1:
        new["num_rollouts"] = method["num_rollouts"]
    if method.get("sampling") is not None:
        new["orchestrator"]["vllm_sampling"] = method["sampling"]
    return new


def _migrate_tts_agent_multi(method: dict) -> dict:
    """Rewrite a tts-agent-multi block to unified tts-agent shape."""
    backend = method["backend"]
    cache_dirs = method["cache_dirs"]
    model_budgets = method["model_budgets"]
    explore = []
    for alias in ("haiku", "sonnet", "opus"):
        if alias not in cache_dirs:
            continue
        explore.append({
            "label": alias,
            "model": _build_model_block(backend, _MODEL_ALIASES[alias]),
            "cache_dir": cache_dirs[alias],
            "num_explores": model_budgets[alias],
        })
    new = {
        "name": "tts-agent",
        "orchestrator_prompt": "multi_model",
        "orchestrator": _build_model_block(backend, method["orchestrator_model"]),
        "explore": explore,
    }
    return new


def _migrate_tts_agent_effort(method: dict) -> dict:
    """Rewrite a tts-agent-effort block to unified tts-agent shape."""
    backend = method["backend"]
    cache_dirs = method["cache_dirs"]
    effort_budgets = method["effort_budgets"]
    explore = []
    for level in ("low", "medium", "high"):
        if level not in cache_dirs:
            continue
        model_block = _build_model_block(backend, method["explore_model"])
        model_block["effort"] = level
        explore.append({
            "label": level,
            "model": model_block,
            "cache_dir": cache_dirs[level],
            "num_explores": effort_budgets[level],
        })
    return {
        "name": "tts-agent",
        "orchestrator_prompt": "effort",
        "orchestrator": _build_model_block(backend, method["orchestrator_model"]),
        "explore": explore,
    }


def _migrate_explore_only(method: dict, name: str) -> dict:
    """Rewrite self-refine / socratic-self-refine / budget-forcing."""
    return {
        "name": name,
        "explore": {
            "label": "default",
            "model": _build_model_block(method["backend"], method["explore_model"]),
            "cache_dir": method["cache_dir"],
            "num_explores": method.get("num_explores", 8),
        },
    }


def _migrate_standalone_integrator(method: dict) -> dict:
    return {
        "name": "standalone-integrator",
        "integrate": {
            "model": _build_model_block(method["backend"], method["integrate_model"]),
            "cache_dir": method["cache_dir"],
        },
        "num_explores": method.get("num_explores", 8),
    }


def _migrate_method(method: dict) -> dict:
    """Dispatch to the right per-method migrator."""
    n = method["name"]
    if n == "tts-agent":
        return _migrate_tts_agent(method)
    if n == "tts-agent-multi":
        return _migrate_tts_agent_multi(method)
    if n == "tts-agent-effort":
        return _migrate_tts_agent_effort(method)
    if n in ("self-refine", "socratic-self-refine", "budget-forcing"):
        return _migrate_explore_only(method, n)
    if n == "standalone-integrator":
        return _migrate_standalone_integrator(method)
    if n == "rerank":
        return method  # unchanged
    raise ValueError(f"unknown method.name: {n!r}")


def _looks_migrated_eval(data: dict) -> bool:
    """Detect already-migrated eval yamls."""
    method = data.get("method", {})
    n = method.get("name")
    if n == "tts-agent":
        return "orchestrator_prompt" in method
    if n in ("self-refine", "socratic-self-refine", "budget-forcing"):
        explore = method.get("explore", {})
        return isinstance(explore, dict) and "model" in explore
    if n == "standalone-integrator":
        return "integrate" in method
    return n == "rerank"  # rerank is unchanged either way


def _migrate_precache(data: dict) -> dict:
    """Rewrite a precache yaml to use a top-level explore: ExploreVariant."""
    backend_name = data["backend"]
    model_block = {
        "backend": backend_name,
        "model": data["explore_model"],
        "budget_tokens": data.get("budget_tokens", 32000),
        "effort": data.get("effort", "low"),
        "timeout": data.get("explore_timeout", 1200.0),
    }
    if backend_name == "vllm" and data.get("sampling") is not None:
        model_block["vllm_sampling"] = data["sampling"]
    if backend_name == "openrouter":
        if data.get("provider_order") is not None:
            model_block["openrouter_provider_order"] = data["provider_order"]
        if "provider_allow_fallbacks" in data:
            model_block["openrouter_provider_allow_fallbacks"] = data["provider_allow_fallbacks"]
    explore = {
        "label": "default",
        "model": model_block,
        "cache_dir": data["cache_dir"],
        "num_explores": data.get("num_explores", 8),
    }
    new = {
        "benchmark": data["benchmark"],
        "explore": explore,
    }
    for k in ("num_workers", "num", "skip", "seed", "shuffle"):
        if k in data:
            new[k] = data[k]
    return new


def _looks_migrated_precache(data: dict) -> bool:
    explore = data.get("explore")
    return isinstance(explore, dict) and "model" in explore and "label" in explore


def _is_precache(path: Path) -> bool:
    return "precache" in path.name


def _migrate_one(path: Path, *, dry_run: bool) -> str:
    """Returns one of: 'migrated', 'skipped', 'rerank-noop'."""
    raw = YAML_RW.load(path.read_text(encoding="utf-8"))
    if raw is None:
        return "skipped"
    if _is_precache(path):
        if _looks_migrated_precache(raw):
            return "skipped"
        new = _migrate_precache(raw)
        # Migrate the benchmark.judge block too
        if isinstance(new.get("benchmark"), dict):
            _migrate_judge(new["benchmark"])
    else:
        method = raw.get("method", {})
        if not method:
            return "skipped"
        if _looks_migrated_eval(raw):
            return "skipped"
        new_method = _migrate_method(method)
        new = dict(raw)
        new["method"] = new_method
        if isinstance(new.get("benchmark"), dict):
            _migrate_judge(new["benchmark"])

    # Validate via pydantic before writing
    schema = PrecacheConfig if _is_precache(path) else EvalConfig
    schema.model_validate(new)

    if dry_run:
        buf = io.StringIO()
        YAML_RW.dump(new, buf)
        print(buf.getvalue())
        return "migrated"

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    with path.open("w", encoding="utf-8") as f:
        YAML_RW.dump(new, f)
    return "migrated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--path", type=Path, help="Single yaml (with --dry-run)")
    ap.add_argument("--bulk", action="store_true", help="Walk scripts/**/*.yaml")
    ap.add_argument("--globs", default="scripts/**/*.yaml")
    args = ap.parse_args()

    if args.path:
        verdict = _migrate_one(args.path, dry_run=args.dry_run)
        print(f"{verdict}: {args.path}", file=sys.stderr)
        return

    assert args.bulk, "use --path for single-file or --bulk for tree-walk"
    base = Path(__file__).resolve().parents[1]
    paths = sorted(base.glob(args.globs))
    counts = {"migrated": 0, "skipped": 0}
    for p in paths:
        try:
            verdict = _migrate_one(p, dry_run=False)
        except Exception as e:
            backup = p.with_suffix(p.suffix + ".bak")
            if backup.exists():
                shutil.copy2(backup, p)
                backup.unlink()
            print(f"FAIL {p}: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"{verdict}: {p}")
    print(f"\nDone. {counts}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check the script on one file first**

Pick one tts-agent yaml (e.g. the grok HLE one):

```
conda run -n explain --no-capture-output python /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/maintenance/migrate_yamls_to_modelconfig.py --dry-run --path Experiment/core_code/scripts/hle/openrouter/hle_grok-4.1-fast_eval.yaml
```

(Add a `--dry-run` flag to the script that prints the rewritten yaml to stdout without overwriting.)

Compare the output against the expected shape (from design §8.2 second example — "After — tts-agent base"). They should match field-for-field.

- [ ] **Step 3: Run on one tts-agent-multi yaml**

```
conda run -n explain --no-capture-output python ... --dry-run --path <one tts-agent-multi yaml>
```
Confirm the resulting structure matches design §8.2 fourth example.

- [ ] **Step 4: Bulk-run on all eval yamls**

```
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output python scripts/maintenance/migrate_yamls_to_modelconfig.py --bulk --globs "scripts/**/*.yaml"
```
Expected: per-file lines like `migrated: <path>` for each of 162 yamls. If any single file fails validation, the script aborts after rolling back to the `.bak`.

- [ ] **Step 5: Spot-check three yamls**

Read three representative migrated files and confirm:
1. `scripts/hle/openrouter/hle_grok-4.1-fast_eval.yaml` — has `orchestrator: ..., explore: [...], orchestrator_prompt: single`.
2. Any `tts-agent-multi` yaml under `scripts/hle/multi_model/` — has `orchestrator_prompt: multi_model` and length-3 `explore`.
3. Any precache yaml under `scripts/hle/openrouter/` — has top-level `explore: ExploreVariant`.

- [ ] **Step 6: Verify pydantic round-trip across all yamls**

```
conda run -n explain --no-capture-output python -c "
from pathlib import Path
from eval import EvalConfig, load_config
from precache_explores import PrecacheConfig
import sys

scripts = Path('Experiment/core_code/scripts').rglob('*.yaml')
fails = []
for p in scripts:
    try:
        if 'precache' in p.name:
            load_config(config_path=p, schema=PrecacheConfig)
        else:
            load_config(config_path=p, schema=EvalConfig)
    except Exception as e:
        fails.append((p, str(e)[:200]))
for p, e in fails:
    print(f'FAIL {p}: {e}')
print(f'fails={len(fails)}')
"
```
Expected: `fails=0`.

- [ ] **Step 7: Commit migrated yamls + migration script**

```
git add Experiment/core_code/scripts/maintenance/migrate_yamls_to_modelconfig.py Experiment/core_code/scripts/
git commit -m "chore(yamls): migrate all yamls to ModelConfig schema"
```

- [ ] **Step 8: Remove .bak files**

```
find /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts -name "*.yaml.bak" -delete
git add -A Experiment/core_code/scripts/
git status   # expect clean tree, no leftover .bak
```

---

## Task 18: Smoke test

**Files:** none modified (validation only).

The motivating use case: grok-4.1-fast HLE-num=2 with `orchestrator.effort=high` over the existing 400-explore cache (effort=low). Validates that the cache layer is reused (no new explore calls) and the orchestrator turn fires at effort=high.

- [ ] **Step 1: Pre-launch checks**

Per project CLAUDE.md API-key freshness rule:

```
curl -sS -o /dev/null -w "HTTP=%{http_code}\n" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/auth/key
```
Expected: `HTTP=200`. If `HTTP=401`, run the bashrc-source recipe from project CLAUDE.md before continuing.

```
ls /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/openrouter_grok-4.1-fast/gold | wc -l
```
Note the count (expected ~100 question subdirs).

- [ ] **Step 2: Edit the migrated grok yaml to num=2 for smoke**

Make a temp copy under `scripts/hle/openrouter/_smoke_modelconfig.yaml`:

```yaml
benchmark:
  name: hle
  subset: gold
  text_only: true
  judge:
    backend: claude
    model: claude-haiku-4-5-20251001
method:
  name: tts-agent
  orchestrator_prompt: single
  orchestrator:
    backend: openrouter
    model: x-ai/grok-4.1-fast
    effort: high                    # the whole point of this smoke
    timeout: 1200.0
    max_output_tokens: 8000
  explore:
    - label: default
      model:
        backend: openrouter
        model: x-ai/grok-4.1-fast
        effort: low                 # cached at effort=low; reused
        timeout: 1200.0
        max_output_tokens: 8000
      cache_dir: ../analysis/cache/hle/openrouter_grok-4.1-fast/gold
      num_explores: 4
num: 2
num_workers: 2
seed: 42
log_dir: ../analysis/run/hle/_smoke_modelconfig
```

- [ ] **Step 3: Launch the smoke run**

```
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
    --config scripts/hle/openrouter/_smoke_modelconfig.yaml \
    > /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/_smoke_modelconfig.log 2>&1 &
echo $! > /tmp/_smoke_pid
cat /tmp/_smoke_pid
```

Share with the user immediately (per project memory `feedback_share_long_running_logs`):

| | |
|---|---|
| PID | from `cat /tmp/_smoke_pid` |
| Log | `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/_smoke_modelconfig.log` |

- [ ] **Step 4: Tail the log until completion (~1-3 minutes)**

```
tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/_smoke_modelconfig.log
```

Look for:
- `Cache pre-flight OK: 2 qids x 4 explores = 8 cache files present` (no new explore calls fired).
- 2 questions complete with `committed` exit_reason.
- Per-question banners log `Orchestrator: openrouter/x-ai/grok-4.1-fast` and per-variant entries.

- [ ] **Step 5: Verify orchestrator effort=high actually engaged**

Find one trajectory.md from the smoke run:

```
ls /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/_smoke_modelconfig/run_*/trajectories/*/trajectory.md | head -1
```

Open it, confirm orchestrator's reasoning blocks are longer than the explore-side reasoning blocks (proxy for effort=high vs effort=low). If reasoning blocks look identical to a known prior effort=low run, halt — the per-role effort plumbing failed somewhere.

- [ ] **Step 6: Delete the smoke yaml + smoke run dir before committing**

```
rm /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/hle/openrouter/_smoke_modelconfig.yaml
# Keep the log file under analysis/run/ for forensic reference; just don't commit it.
```

---

## Task 19: Final sanity sweep + commit

- [ ] **Step 1: Verify no orphan references to deleted symbols**

```
cd /data3/peijia/dr-claw/Explain
git grep -n "BackendConfig\|TTSAgentMultiSpec\|TTSAgentEffortSpec\|ClaudeJudgeSpec\|VllmJudgeSpec\|CodexJudgeSpec\|orchestrator_effort\b" \
  -- 'Experiment/core_code/*.py' 'Experiment/core_code/**/*.py'
```
Expected: empty. (Grep across `.py` only — the design-spec doc and old plan docs may still reference old names; that's fine.)

- [ ] **Step 2: Verify deleted modules**

```
ls /data3/peijia/dr-claw/Explain/Experiment/core_code/methods/tts_agent_multi.py 2>&1 | grep -q "No such file" && echo OK_DELETED || echo STILL_EXISTS
ls /data3/peijia/dr-claw/Explain/Experiment/core_code/methods/tts_agent_effort.py 2>&1 | grep -q "No such file" && echo OK_DELETED || echo STILL_EXISTS
```
Expected: both `OK_DELETED`.

- [ ] **Step 3: Run the full pytest scope**

```
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain --no-capture-output python -m pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 4: Final commit (if any uncommitted leftovers)**

```
git status
git diff
# If clean, no commit needed. If anything is left over (e.g. forgotten import cleanup),
# commit with: git commit -m "chore: post-refactor cleanup"
```

- [ ] **Step 5: Tag the migration**

Optional: tag the merge commit so a hypothetical revert can pivot off one anchor:

```
git tag modelconfig-refactor-20260504 -a -m "ModelConfig refactor lands"
```

---

## Completion Criteria

The refactor is done when ALL of:

1. `git grep "BackendConfig\|TTSAgentMultiSpec\|TTSAgentEffortSpec\|ClaudeJudgeSpec\|VllmJudgeSpec\|CodexJudgeSpec\|JudgeSpec\b\|orchestrator_effort\b"` over `Experiment/core_code/**/*.py` returns nothing.
2. `methods/tts_agent_multi.py` and `methods/tts_agent_effort.py` no longer exist.
3. All 162 yamls under `Experiment/core_code/scripts/` validate against `EvalConfig` / `PrecacheConfig` (Task 17 Step 6).
4. `tests/` pass.
5. The grok-4.1-fast HLE-num=2 smoke test (Task 18) ran end-to-end with the existing 400-explore cache reused at effort=low and the orchestrator turn at effort=high.
6. vllm judge cache directories under `Experiment/analysis/.../judges/vllm__*/config.json` carry `vllm_sampling` (Task 16 Step 5) and the judge cache hits when run.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-04-modelconfig-refactor.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
