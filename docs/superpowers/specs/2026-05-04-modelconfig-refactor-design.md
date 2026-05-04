# ModelConfig Refactor — Design Spec

**Date**: 2026-05-04
**Author**: peijia + Claude
**Status**: Draft, pending user review
**Scope**: Replace per-method-shared `BackendConfig` + per-role string `xxx_model: str` with per-role full `ModelConfig` carrying every backend invocation parameter independently.

---

## 1. Goal

Make every model role inside a method spec — `orchestrator`, `explore`, `integrate`, `judge`, plus future roles — carry its own complete, independent backend invocation configuration. Allow cross-provider, cross-effort, cross-timeout, cross-budget, cross-max-output-tokens, cross-provider-routing combinations without sharing a single method-level `backend:` block.

The immediate motivating use case: run grok-4.1-fast HLE eval where cached explores were generated at `effort=low` but the orchestrator needs `effort=high` to make better integration decisions over those candidates. Larger goal: enable any heterogeneous combination (e.g., explore on local vLLM, orchestrator on OpenRouter Claude, integrate on Sonnet) without code changes.

The current shape — one `BackendConfig` shared across all roles + per-role bare model strings — forced every role into the same backend / effort / timeout / budget / provider routing. Per-role `effort` was added as a one-off patch on 2026-05-04 (see `methods/specs.py:TTSAgentSpec.orchestrator_effort`); this spec replaces that patch with the complete per-role design and removes the `orchestrator_effort` field.

## 2. Non-Goals

- No yaml inheritance / defaults / anchors (option A: every role spells out every field explicitly).
- No CLI-level model overrides — yaml remains single source of truth.
- No backward-compatibility shims. Old yamls hard-fail on load and must be migrated by the migration script.
- No changes to `training/grpo/grade_cache.py` (uses an independent `JUDGE_MODEL` constant; out of scope).
- No changes to `RerankSpec.reward_model` (local PyTorch model, not a backend-routed call; it stays a `str` with method-level `cache_dir`).

## 3. New Type: `ModelConfig`

Replaces `BackendConfig` and absorbs the three `JudgeSpec` subclasses (`ClaudeJudgeSpec`, `CodexJudgeSpec`, `VllmJudgeSpec`).

```python
class ModelConfig(BaseModel):
    model_config = {"extra": "forbid"}

    # Shared dimensions (all backends understand these)
    name: Literal["codex", "claude", "vllm", "openrouter"]
    model: str
    budget_tokens: int = 32000
    effort: Literal["low", "medium", "high", "max"] = "low"
    timeout: float = 1200.0
    max_output_tokens: int | None = None

    # vLLM-only (prefix + validator)
    vllm_sampling: SamplingConfig | None = None

    # OpenRouter-only (prefix + validator)
    openrouter_provider_order: list[str] | None = None
    openrouter_provider_allow_fallbacks: bool = True

    @model_validator(mode="after")
    def _check_backend_specific(self):
        if self.name != "vllm":
            assert self.vllm_sampling is None, (
                f"vllm_sampling is vllm-only but name={self.name!r}; remove from yaml"
            )
        if self.name != "openrouter":
            assert self.openrouter_provider_order is None, (
                f"openrouter_provider_order is openrouter-only but name={self.name!r}"
            )
            assert self.openrouter_provider_allow_fallbacks is True, (
                f"openrouter_provider_allow_fallbacks is openrouter-only but "
                f"name={self.name!r}"
            )
        return self
```

Why backend-specific prefix + validator: per CLAUDE.md "expose as much error as possible". The prior `BackendConfig` accepted `provider_order: [...]` even when `name=vllm` — silent no-op, the user's pin was discarded. The 2026-05-04 deepseek-v4-flash incident burned hours partly because of this silent fallback. Prefixed names make the intent visible in yaml, and the validator turns the silent ignore into a hard error.

## 4. New Helper Types: `RoleSlot` and `ExploreVariant`

Two thin wrappers carry per-role caching alongside the model invocation:

```python
class RoleSlot(BaseModel):
    """A model invocation + where its outputs are cached.
    Used by integrate (and future single-shot cached roles)."""
    model_config = {"extra": "forbid"}
    model: ModelConfig
    cache_dir: Path

class ExploreVariant(BaseModel):
    """One explore-pass: model + its cache_dir + how many candidates to draw."""
    model_config = {"extra": "forbid"}
    model: ModelConfig
    cache_dir: Path
    num_explores: int = 8
```

Why two types instead of one with `Optional[num_explores]`: pydantic `extra="forbid"` will reject `num_explores` set on an integrate yaml block, catching typos at config load time rather than at runtime. Single-class with optional num_explores would silently accept misplaced fields.

Why not put `cache_dir` inside `ModelConfig`: cache_dir is not a model invocation parameter — it's an output-archival location. Putting it inside `ModelConfig` would mix two semantics. The wrapper pattern keeps `ModelConfig` pure ("how to call this model") and `RoleSlot`/`ExploreVariant` adds the role-specific archival concern.

## 5. Method Spec Changes

### 5.1. TTSAgentSpec (three-into-one consolidation)

Old shape (three separate specs):
- `TTSAgentSpec(backend, orchestrator_model, explore_model, integrate_model, cache_dir, num_explores, ...)`
- `TTSAgentMultiSpec(backend, orchestrator_model, cache_dirs: dict[str, Path], model_budgets: dict[str, int], exploration_effort, num_explores)`
- `TTSAgentEffortSpec(backend, orchestrator_model, explore_model, cache_dirs: dict[str, Path], effort_budgets: dict[str, int], num_explores)`

New shape (one unified spec):

```python
class TTSAgentSpec(_MethodSpec):
    name: Literal["tts-agent"]
    orchestrator: ModelConfig
    explore: list[ExploreVariant]              # length 1 = old base; length N = old multi/effort
    integrate: RoleSlot | None = None
    no_integrate: bool = False
    num_rollouts: int = 1
```

The `multi` and `effort` cases collapse into list-length:
- Old base case → list length 1
- Old multi (haiku/sonnet/opus) → list length 3, three different `model.model` strings
- Old effort (low/medium/high) → list length 3, three different `model.effort` values

The `string-key dict` join between `cache_dirs: dict[str, ...]` and `model_budgets: dict[str, ...]` (which silently mis-joined on label typos) is replaced by structural alignment — each `ExploreVariant` carries its own model, cache_dir, and num_explores together.

### 5.2. Other method specs (mechanical field rename)

| Spec | Old fields | New fields |
|---|---|---|
| `SelfRefineSpec` | `backend, explore_model: str, cache_dir, num_explores` | `explore: ExploreVariant` |
| `SocraticSelfRefineSpec` | `backend, explore_model: str, cache_dir, num_explores` | `explore: ExploreVariant` |
| `BudgetForcingSpec` | `backend, explore_model: str, cache_dir, num_explores` | `explore: ExploreVariant` |
| `RerankSpec` | `reward_model: str, cache_dir` | `reward_model: str, cache_dir` (UNCHANGED — local PyTorch) |
| `StandaloneIntegratorSpec` | `backend, integrate_model: str, cache_dir, num_explores` | `integrate: RoleSlot, num_explores: int` |

`RerankSpec` is intentionally unchanged. Its `reward_model` is a HuggingFace model id loaded by `transformers`, not a backend-routed remote call. Forcing it through `ModelConfig` would require adding `name="local"` to the backend Literal, which mixes "remote backend" and "local PyTorch" semantics. The cleaner outcome is to admit the asymmetry: rerank is a different kind of operation.

### 5.3. Deletions

- `TTSAgentMultiSpec` — merged into TTSAgentSpec
- `TTSAgentEffortSpec` — merged into TTSAgentSpec
- `BackendConfig` — replaced by ModelConfig
- `_JudgeSpec`, `ClaudeJudgeSpec`, `CodexJudgeSpec`, `VllmJudgeSpec`, `JudgeSpec` union — all replaced by single `ModelConfig`
- `methods/tts_agent_multi.py` (289 lines) — solver merged into tts_agent.py
- `methods/tts_agent_effort.py` (292 lines) — solver merged into tts_agent.py
- `TTSAgentSpec.orchestrator_effort` (added 2026-05-04 as one-off patch) — replaced by `orchestrator: ModelConfig` carrying its own `effort`

Net class change: 15 → 11 (three deletions, one rename, two new helpers).
Net code change: ~600 lines deleted, ~150 lines added.

## 6. Benchmark Spec Changes

The `judge` block in benchmark specs becomes a plain `ModelConfig`:

```python
# Before (benchmarks/specs.py)
class ClaudeJudgeSpec: name="claude", model: str, effort, budget_tokens
class CodexJudgeSpec:  name="codex", model: str
class VllmJudgeSpec:   name="vllm", model: str, sampling: SamplingConfig
JudgeSpec = Union[Claude, Codex, Vllm]

# After
judge: ModelConfig | None = None
```

`judge_label(judge_spec) = f"{name}__{model}"` is unchanged — both keys exist on ModelConfig, so existing cache directories (e.g., `judges/claude__claude-haiku-4-5-20251001/`) are still found by name.

CLAUDE.md non-thinking-judge rule encoded as default: `ModelConfig.effort = "low"` (already the default).

### vLLM judge cache compatibility (Pitfall #1 from review)

Existing cache directories under `judges/vllm__*/` contain `config.json` files with field name `sampling: {...}`. The new `ModelConfig` schema uses `vllm_sampling` instead. The unidirectional `find_cached_judge` policy treats stored-superset-of-requested as a hard `RuntimeError`, so existing vLLM judge caches would break.

**Mitigation**: write a one-shot data-migration script `tmp/migrate_vllm_judge_cache.py` that scans every `judges/vllm__*/config.json` under `analysis/cache/` and renames the `sampling` key to `vllm_sampling`. Run once before the spec change is merged. No alias logic in production code — keep `find_cached_judge` simple; pay the migration cost once.

## 7. Plumbing Changes

### 7.1. `methods/base.py`

`InfraConfig` and `SolveContext` no longer carry single shared `backend / effort / budget_tokens / max_output_tokens / timeout / orchestrator_effort` fields. Instead they hold per-role `ModelConfig` references that the solver picks based on which call it's making:

```python
@dataclass
class SolveContext:
    state, cost, rounds, _sub_model_fn, writer, traj_dir, question_cache_dir,
    image_data_url, benchmark, logger, question_id, rollout_idx,
    # per-role configs (None when method doesn't need that role)
    orchestrator: ModelConfig | None = None
    integrate: RoleSlot | None = None
    # explore is a list (TTSAgentSpec carries list[ExploreVariant])
    explore_variants: list[ExploreVariant] = field(default_factory=list)
```

`SolveContext.call_sub_model` takes the `ModelConfig` to use as a parameter:

```python
async def call_sub_model(self, *, system_prompt, user_message, output_schema,
                         model_cfg: ModelConfig, cache_key="", writer=None):
    return await self._sub_model_fn(
        system_prompt, user_message, self.image_data_url,
        model=model_cfg.model, output_schema=output_schema,
        backend=model_cfg.name,
        cache_key=cache_key, writer=writer or TrajectoryWriter.noop(),
        budget_tokens=model_cfg.budget_tokens, effort=model_cfg.effort,
        timeout=model_cfg.timeout, max_output_tokens=model_cfg.max_output_tokens,
        sampling=model_cfg.vllm_sampling,
        provider_order=model_cfg.openrouter_provider_order,
        provider_allow_fallbacks=model_cfg.openrouter_provider_allow_fallbacks,
    )
```

### 7.2. `methods/tts_agent.py` solver

Two state changes:

1. `ExploreStepState` extended from a single `(call_count, max_explores)` indexer to track which variant is currently being drawn from. Existing logic that uses `state.explore.call_count + 1` for cache_key naming becomes (variant_idx, in_variant_idx) — but **cache_key stays `f"explore_{in_variant_idx}"` because each variant has its own `cache_dir`**. The `state.explore` data structure tracks both, but cache_key naming lives within a variant.

2. Explore-tool handler picks the right variant. The orchestrator's explore tool call selects which variant to invoke (probably round-robin or method-driven; matches behavior of old `tts_agent_multi.py:rotate`). For length-1 explore list (the base case), this degenerates to the current behavior.

3. Orchestrator turn (`run_tool_conversation` at tts_agent.py:280) reads from `ctx.orchestrator: ModelConfig`. Integrate tool reads from `ctx.integrate: RoleSlot`.

### 7.3. `eval.py` (InfraConfig instantiation, ~767-780)

```python
# OLD
backend_block = getattr(cfg.method, "backend", None)
infra = InfraConfig(
    backend=backend_block.name, ..., effort=backend_block.effort, ...
)

# NEW — InfraConfig no longer carries backend/effort/etc; SolveContext
# pulls per-role ModelConfig directly from the method spec
infra = InfraConfig(
    cache_dir=...,  # method-level for legacy methods that still need it
                    # (rerank, standalone-integrator)
    cache_only=method.cache_only,
    max_iterations=...,
    benchmark=benchmark,
    logger=None,
    enable_integrate=not getattr(cfg.method, "no_integrate", False),
)
# per-role ModelConfigs are passed to create_solve_context separately,
# extracted by registry.derive_evaluate_args from the method spec
```

### 7.4. `backends/openrouter.py`

The module-level globals `_PROVIDER_ORDER, _PROVIDER_ALLOW_FALLBACKS` and `set_provider()` function are deleted. `_maybe_inject_provider` reads the provider settings from the per-call `ModelConfig` instead. Each `call_sub_model` and `run_tool_conversation` call now carries its own provider routing — different roles can use different providers within a single eval run.

### 7.5. `precache_explores.py`

`PrecacheConfig` collapses its scattered fields (`backend, explore_model, budget_tokens, effort, max_output_tokens, provider_order, ...`) into one field: `explore: ExploreVariant`. `cache_dir` and `num_explores` come from `explore.cache_dir` and `explore.num_explores`.

### 7.6. `methods/registry.py`

`TTSAgentMultiMethod` and `TTSAgentEffortMethod` classes are deleted (they registered the now-deleted specs). The remaining methods' `derive_evaluate_args` is updated to extract per-role `ModelConfig`s from the new spec shapes.

## 8. yaml Migration

### 8.1. Migration script

Write `tmp/migrate_yamls_to_modelconfig.py` (one-shot, archived after use).

Behavior per yaml:
1. Parse the file with `ruamel.yaml` (preserves comments and ordering).
2. Detect `method.name`:
   - `tts-agent` → fold `backend.{...}` + each `xxx_model: str` into per-role `ModelConfig`s; build `explore: [ExploreVariant{model, cache_dir, num_explores}]` with list length 1.
   - `tts-agent-multi` → rename to `tts-agent`; fold `cache_dirs: dict + model_budgets: dict + exploration_effort` into `explore: list[ExploreVariant]` with length N; the orchestrator field comes from `orchestrator_model` + the shared backend.
   - `tts-agent-effort` → rename to `tts-agent`; fold `cache_dirs: dict + effort_budgets: dict` into `explore: list[ExploreVariant]` with each variant carrying a different `effort`.
   - `self-refine`, `socratic-self-refine`, `budget-forcing` → fold `backend.{...}` + `explore_model: str` + `cache_dir` + `num_explores` into `explore: ExploreVariant`.
   - `standalone-integrator` → fold `backend.{...}` + `integrate_model: str` + `cache_dir` into `integrate: RoleSlot`; preserve top-level `num_explores`.
   - `rerank` → unchanged.
3. Detect `benchmark.judge` block; if present, fold it into a flat `ModelConfig` (the only structural change is the union-discriminator collapse — fields stay the same names except `sampling` → `vllm_sampling` for vLLM judges).
4. Write back, preserving comments where possible.
5. Validate: `python -c "import yaml; from methods.specs import MethodSpec; ..."` round-trips through pydantic.

Run it once across:
- `scripts/**/*.yaml` (~30 files including 17 multi/effort)
- `tests/**/*.yaml` if any (search and update test fixtures)

After validation, commit migrated yamls + migration script in same commit. Archive the migration script under `archive/scripts/2026-05-04-modelconfig-migration.py`.

### 8.2. yaml shape examples

```yaml
# Before — tts-agent base
benchmark:
  name: hle
  judge:
    name: claude
    model: claude-haiku-4-5-20251001
    effort: low
method:
  name: tts-agent
  backend:
    name: openrouter
    effort: low
    timeout: 1200
  orchestrator_model: x-ai/grok-4.1-fast
  explore_model: x-ai/grok-4.1-fast
  cache_dir: ../analysis/cache/hle/openrouter_grok-4.1-fast/gold
  no_integrate: true
  num_explores: 4
```

```yaml
# After — tts-agent base
benchmark:
  name: hle
  judge:
    name: claude
    model: claude-haiku-4-5-20251001
    effort: low
method:
  name: tts-agent
  orchestrator:
    name: openrouter
    model: x-ai/grok-4.1-fast
    effort: high                # ← was lockstep with explore.effort; now independent
    timeout: 1200
  explore:
    - model:
        name: openrouter
        model: x-ai/grok-4.1-fast
        effort: low
        timeout: 1200
      cache_dir: ../analysis/cache/hle/openrouter_grok-4.1-fast/gold
      num_explores: 4
  no_integrate: true
```

```yaml
# Before — tts-agent-multi
method:
  name: tts-agent-multi
  backend:
    name: claude
  orchestrator_model: claude-sonnet-4-6
  cache_dirs:
    haiku: ../analysis/cache/hle/haiku/gold
    sonnet: ../analysis/cache/hle/sonnet/gold
    opus: ../analysis/cache/hle/opus/gold
  model_budgets:
    haiku: 8
    sonnet: 8
    opus: 4
  num_explores: 8
```

```yaml
# After — tts-agent (multi as length-3 explore list)
method:
  name: tts-agent
  orchestrator:
    name: claude
    model: claude-sonnet-4-6
    effort: low
  explore:
    - model: {name: claude, model: claude-haiku-4-5-20251001, effort: low}
      cache_dir: ../analysis/cache/hle/haiku/gold
      num_explores: 8
    - model: {name: claude, model: claude-sonnet-4-6, effort: low}
      cache_dir: ../analysis/cache/hle/sonnet/gold
      num_explores: 8
    - model: {name: claude, model: claude-opus-4-5, effort: low}
      cache_dir: ../analysis/cache/hle/opus/gold
      num_explores: 4
```

## 9. Cache Compatibility (Pitfall Mitigations)

Two cache layers are affected:

### 9.1. Explore cache (`cache/<bench>/<model>/<subset>/<qid>/explore_N/result.json`) — UNCHANGED layout

Per-variant `cache_dir` keeps the existing directory layout. cache_key inside a variant remains `f"explore_{idx}"`. The 400 grok-4.1-fast HLE explores already on disk are reused on the very next eval run with the new spec.

For multi-cache cases (haiku/sonnet/opus), each old `cache/hle/<model>/gold/` becomes one ExploreVariant.cache_dir — also reused as-is.

**Zero data migration for the explore cache layer.**

### 9.2. Judge cache (`*/explore_N/judges/<label>/config.json`) — VLLM-ONLY MIGRATION

Three vLLM judge cache locations on disk hold `config.json` files with key `sampling`:
- `judges/vllm__gemma4-26b-a4b-it/`
- `judges/vllm__qwen36-35b-a3b-fp8/`
- `judges/vllm__gptoss-20b/`

The new `ModelConfig` uses `vllm_sampling`. Run `tmp/migrate_vllm_judge_cache.py` once before the spec change merges:

```python
# Pseudocode
for cfg_path in glob("analysis/**/judges/vllm__*/config.json"):
    cfg = json.loads(cfg_path.read_text())
    if "sampling" in cfg:
        cfg["vllm_sampling"] = cfg.pop("sampling")
        cfg_path.write_text(json.dumps(cfg, indent=2))
```

Claude / Codex / OpenRouter judge caches are unaffected (no field rename for them).

### 9.3. Old run resume

Old `run_dir/run_config.json` files dump the old yaml shape. They cannot resume under the new spec. Acceptable per CLAUDE.md "DO NOT maintain backward compatibility by default" — old runs become unresumable but their `results.jsonl` data and grade.json files remain readable for forensic analysis.

## 10. Testing

### 10.1. New unit tests

- `tests/test_modelconfig.py`
  - Accept valid combinations (each backend × each effort).
  - Reject `vllm_sampling` set when `name != "vllm"`.
  - Reject `openrouter_provider_order` set when `name != "openrouter"`.
  - Reject unknown `name` value.
  - Default field values match expectation.
- `tests/test_role_slot.py`
  - Accept `{model: ModelConfig, cache_dir: Path}`.
  - Reject extra fields like `num_explores`.
- `tests/test_explore_variant.py`
  - Accept `{model, cache_dir, num_explores}`.
  - Default `num_explores = 8`.
- `tests/test_tts_agent_unified.py`
  - Length-1 `explore: list[ExploreVariant]` parses successfully (base case).
  - Length-3 `explore` parses successfully (multi case).
  - Length-3 with varying `effort` per variant parses (effort case).
  - `name: tts-agent-multi` and `name: tts-agent-effort` fail validation (specs deleted).

### 10.2. Updated tests

- `tests/test_judge_spec.py` — assert single `ModelConfig`-based judge field; assert vllm-prefix validator behavior.
- `tests/test_judge_cache_best_effort.py` — exercise stored-subset hit with the larger ModelConfig schema; verify vllm judge cache after migration script runs.
- `tests/test_eval_config.py` — yaml load round-trips through new spec shape.
- `tests/test_precache_config.py` — `explore: ExploreVariant` round-trips.

### 10.3. Integration smoke

- `num=2` HLE eval with the migrated grok yaml using `orchestrator.effort=high` + `explore.effort=low` (cached). Verify:
  - All 4 cached explores load (no cache miss / re-fire).
  - Orchestrator turn fires at effort=high (visible in OpenRouter logs as higher reasoning token usage on the orchestrator call).
  - Final integrated answer recorded.
- `num=2` LCB eval with the migrated multi yaml (haiku/sonnet/opus three variants).
- VLLM judge eval after running `migrate_vllm_judge_cache.py`: confirm cache hits at `judges/vllm__*` rather than re-judging.

## 11. Risk Register

| # | Risk | Mitigation | Severity if missed |
|---|---|---|---|
| 1 | VLLM judge cache `sampling` key rename hard-fails `find_cached_judge` | Run `tmp/migrate_vllm_judge_cache.py` before merging spec change | High — eval crashes immediately on vllm judge runs |
| 2 | Solver state.explore extension wrong; cache_keys cross variants | Tests cover length-1 / length-3 cases; per-variant cache_dir ensures key isolation by directory | High — wrong cache hits silently mix candidates |
| 3 | yaml migration script misses an edge case (e.g., resume: ../old_run_dir) | Validate every migrated yaml with pydantic round-trip; manual review of multi/effort yamls (17 files, tractable) | Medium — caught by validator |
| 4 | Old run_dir's run_config.json no longer parseable | Documented as expected break; forensic data still readable | Low — acceptable |
| 5 | `RerankSpec.reward_model` asymmetry (kept as `str`) confusing | Inline comment in spec explaining local-PyTorch nature | Low — design choice documented |
| 6 | Existing 17 multi/effort yamls in production directories — running script could break in-flight experiments | Migration is one-shot, not per-eval; running experiments not affected during migration window | Low |
| 7 | tests/test_judge_spec.py and friends fail under new spec | Update tests as part of the same PR | Low |

## 12. Out of Scope

- `training/grpo/grade_cache.py` — uses an independent `JUDGE_MODEL` constant, separate from the main spec system. Out of scope for this PR.
- `scripts/hle/sonnet/regrade_socratic_self_refine.py` — references `bench.judge_model` class attribute already removed on 2026-05-01; broken in main, not addressed here.
- yaml inheritance / defaults / anchors (option A locked in §1).
- ModelConfig adding `name="local"` to support `RerankSpec.reward_model` — kept asymmetric per §5.2.
- Per-method-spec yaml schema versioning — yamls are not versioned; migration is one-shot.

## 13. Implementation Order

To minimize broken-window time, implement in this order:

1. Add new types (`ModelConfig`, `RoleSlot`, `ExploreVariant`) alongside old `BackendConfig`. Tests pass for both.
2. Migrate `methods/specs.py` method specs to use new types; old specs deleted.
3. Update `methods/base.py` (`InfraConfig`, `SolveContext`, `call_sub_model`).
4. Update `methods/tts_agent.py` solver (consolidate multi/effort logic; explore loop becomes list-of-variants).
5. Delete `methods/tts_agent_multi.py` and `methods/tts_agent_effort.py`.
6. Update `methods/registry.py` (delete multi/effort registrations).
7. Update `eval.py`, `precache_explores.py`, `backends/openrouter.py` plumbing.
8. Update `benchmarks/specs.py` (collapse JudgeSpec union into ModelConfig).
9. Run `tmp/migrate_vllm_judge_cache.py`.
10. Run `tmp/migrate_yamls_to_modelconfig.py`.
11. Update existing tests; add new tests.
12. Smoke test: grok HLE num=2 with effort split.
13. Commit everything in one breaking-change commit.

## 14. Acceptance Criteria

- All existing yaml files load successfully under new spec (after migration).
- The grok-4.1-fast HLE-100 eval can run with `orchestrator.effort=high` over the existing 400-explore cache (effort=low) without re-firing explores.
- `tests/` pass.
- `git grep "BackendConfig\|TTSAgentMultiSpec\|TTSAgentEffortSpec\|ClaudeJudgeSpec\|VllmJudgeSpec\|CodexJudgeSpec\|JudgeSpec"` returns nothing.
- `methods/tts_agent_multi.py` and `methods/tts_agent_effort.py` no longer exist.
- VLLM judge cache hits work after running `migrate_vllm_judge_cache.py`.
