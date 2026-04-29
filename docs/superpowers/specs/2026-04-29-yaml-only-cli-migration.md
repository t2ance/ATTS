# YAML-Only CLI Migration: Design Spec

**Date:** 2026-04-29
**Owner:** Peijia
**Status:** Draft

## Goal

Collapse `eval.py` and `precache_explores.py` from a three-source CLI (YAML + flat argparse + dot-path overrides) to a two-source CLI (YAML + dot-path overrides). Eliminate every flat argparse flag that mirrors a Pydantic field. Migrate all 73 shell scripts to thin `--config` wrappers in a single, traceless cutover.

## Why

The previous migration (commits `08ef0c7..7b10288` on main) preserved flat argparse flags as a backward-compatibility layer for 67 single-model shell scripts. That decision is now reversed: the user has confirmed (a) no backward compatibility, (b) one-shot migration, (c) no traces. Keeping flat flags carries permanent costs:

- Three-source precedence (YAML / flat / dot-path) is one source too many; reasoning about the merge is non-obvious.
- `parse_cli` carries special-case logic for legacy paths: `--cache-dirs` single-path routing to `cfg.cache_dir`, filter dot-path translation. All of this exists only because flat filter / cache flags exist.
- Every new `EvalConfig` field forces an "expose as flat flag?" decision with no canonical answer.
- 67 shell scripts are research artifacts (CLAUDE.md AD4: "Scripts hardcode all constants"). Each script is morally a frozen YAML; the shell-arg form is just incidental syntax.

The design principle is **single source of truth**: one YAML per concrete experiment, plus optional `-o key=value` patches for ad-hoc tweaks.

## Non-Goals

- Do not preserve any deprecated flat flag.
- Do not introduce ad-hoc CLI shortcuts (`-o num=1 -o seed=42` is the terminal experience for one-off tweaks).
- Do not touch the `EvalConfig` schema (already stable from prior migration).
- Do not modify any of the 18 existing YAMLs except for renames called out in §Renames.
- Do not touch the `worktree-knot-compute` branch.

## Architecture

### CLI surface (final state, both `eval.py` and `precache_explores.py`)

```
python <script>.py --benchmark X --config configs/Y.yaml [-o key=value ...]
```

Three argparse flags, exactly:

| Flag | Purpose |
|---|---|
| `--benchmark` | Required. Sniffed via `parse_known_args` so the right benchmark class is picked (used for filter validator dispatch). |
| `--config` | Required. Path to a YAML config file. |
| `-o`/`--override` | Optional, repeatable. Dot-path override `-o key.subkey=value`. Highest precedence. |

No other flags. Misspelled or extra flags hit argparse `unrecognized arguments` (loud failure).

### Pydantic schemas

- **`eval_config.py` `EvalConfig`** — unchanged from current main. `parse_cli` becomes a thin wrapper.
- **`precache_config.py` `PrecacheConfig`** — new file, declares ~14 fields covering precache_explores.py needs (no `method`, no `orchestrator_model`, no `cache_dirs` dict, no `model_budgets` / `effort_budgets`):

  ```python
  class PrecacheConfig(BaseModel):
      model_config = {"extra": "forbid"}
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
          model = bench.make_filter_model()
          self.filters = model.model_validate(self.filters).model_dump(exclude_defaults=True)
          return self
  ```

  Note: `cache_dir` is required (no default); precache cannot proceed without one. Same `--config + -o` loader infrastructure as `eval_config.py` (`load_config()` is generic — extracted to `config_loader.py` if duplication grows; for now `precache_config.py` imports `_set_dotpath` and `_coerce_scalar` from `eval_config`).

### Shared loader

`load_config(*, config_path, dot_overrides) -> ConfigT` extracts to a generic helper. The `flat_overrides` parameter goes away (no flat flags = no flat overrides). Signature:

```python
def load_config(
    *,
    config_path: Path | str,
    dot_overrides: list[str],
    schema: type[BaseModel],
) -> BaseModel: ...
```

Both `eval.py` and `precache_explores.py` call this with `schema=EvalConfig` / `schema=PrecacheConfig`.

### Argparse cleanup

- **`benchmarks/base.py`**: delete `add_dataset_args` and `add_model_args` methods entirely. `filter_keys` tuple stays (Pydantic filter validators still use it). `make_filter_model` stays.
- **`benchmarks/{hle,lcb,gpqa,babyvision,aime,rbenchv}.py`**: delete the `add_dataset_args` overrides.
- **`eval.py` `parse_cli`**: delete benchmark-args attachment, all flat-flag declarations, and the filter-routing / cache-dirs special-case loop. Final body is ~15 lines.
- **`precache_explores.py`**: delete `parse_args`, replace with a `parse_cli`-shaped function returning a `PrecacheConfig`.

### Shell-script migration

Write `Experiment/core_code/scripts/migrate_eval_scripts.py` (one-shot tool, deletes itself on success):

1. Walk `Experiment/core_code/scripts/**/*.sh`.
2. For each script that calls `python eval.py` or `python precache_explores.py`:
   - Parse the invocation with `shlex.split` (handles `\` continuation, quoted strings, comments).
   - Translate flat args → YAML keys per the lookup table below.
   - Generate `Experiment/core_code/configs/<flat_name>.yaml` where `<flat_name>` = the script path with `/` → `_`, `.sh` stripped, `run_` prefix stripped.
     - `scripts/hle/sonnet/run_self_refine.sh` → `configs/hle_sonnet_self_refine.yaml`
     - `scripts/lcb/grpo/run_eval_grpo_8b.sh` → `configs/lcb_grpo_eval_grpo_8b.yaml`
     - Filename collisions hard-fail (no auto-suffix; the namespace is flat enough that collisions are bugs).
   - Rewrite the `.sh` in place: keep its preamble (cd, env vars, `unset`, `export`, `mkdir`, `conda activate`), keep its tail (`> log 2>&1 &`), replace the `python ... ARGS` middle line with `python <script>.py --benchmark X --config configs/<flat_name>.yaml`.
3. After all 73 scripts complete: `os.remove(__file__)`.
4. Hard-fails on any unparseable script (no silent skip).

**Arg → YAML key map** (eval.py):

| Flat arg | YAML key | Notes |
|---|---|---|
| `--benchmark X` | (kept on shell `--benchmark X`) | Sniffed by argparse, also written to YAML for completeness. |
| `--backend X` | `backend: X` | |
| `--explore-model X` | `explore_model: X` | |
| `--method X` | `method: X` | |
| `--orchestrator-model X` | `orchestrator_model: X` | |
| `--integrate-model X` | `integrate_model: X` | Omit if equal to orchestrator_model AND method is multi/effort (auto-aliased). |
| `--reward-model X` | `reward_model: X` | |
| `--cache-dirs PATH` | `cache_dir: PATH` | Single-path string → singular field. |
| `--cache-dirs "k:p,k:p"` | `cache_dirs: {k: p, k: p}` | Dict form. |
| `--model-budgets "k:n,k:n"` | `model_budgets: {k: n, k: n}` | |
| `--effort-budgets "k:n,k:n"` | `effort_budgets: {k: n, k: n}` | |
| `--exploration-effort X` | `exploration_effort: X` | |
| `--num N` / `--skip N` / `--seed N` | scalar | |
| `--shuffle` / `--no-cache-only` / `--no-integrate` / `--verbose` | bool true | |
| `--num-explores N` / `--num-workers N` | scalar | |
| `--num-rollouts N` | `num_rollouts: N` | |
| `--budget-tokens N` / `--effort X` / `--timeout N` / `--explore-timeout N` | scalar | |
| `--max-output-chars N` | scalar | |
| `--log-dir PATH` | `log_dir: PATH` | |
| `--resume PATH` | `resume: PATH` | |
| `--subset X` / `--text-only` / `--category X` / `--difficulty X` / `--type X` / `--subtype X` / `--year N` / `--domain X` | nested into `filters: {...}` | Per-benchmark filter keys go inside `filters:`. |

**Arg → YAML key map** (precache_explores.py): same as eval.py minus `method` / orchestrator / integrate / reward / multi-cache / budgets / exploration_effort / num_rollouts / no_integrate / verbose / resume / log_dir / max_output_chars / no_cache_only / timeout. The 14 fields from `PrecacheConfig`.

### Renames in existing 18 YAMLs

The current `configs/` has two `_v2` / dual-named pairs that encode real semantic distinctions (verified by diff):

- `gpqa_multi_delegated_v2.yaml` differs from `gpqa_multi_effort_high.yaml`: the former has NO `exploration_effort`, the latter has `exploration_effort: high`.
- `gpqa_multi_delegated_effort_high.yaml` differs from `gpqa_multi_effort_high.yaml`: only by `log_dir` (`..._effort_high` vs `..._effort_high_v2`).

Renames during migration:

| Current | New | Reason |
|---|---|---|
| `gpqa_multi_delegated_v2.yaml` | `gpqa_multi_delegated_freeform.yaml` | "freeform" = no `exploration_effort` set; "v2" was a launcher-version suffix not a semantic distinction |
| `gpqa_multi_delegated_effort_high.yaml` | (consolidated; delete the file) | Identical to `gpqa_multi_effort_high.yaml` modulo log_dir. The shell script for `run_delegated_effort_high.sh` after migration points to `configs/gpqa_multi_effort_high.yaml` with `-o log_dir=../analysis/run/gpqa/multi_model_effort_high` to override the suffixed log_dir. Net: one fewer YAML. |

The 16 other existing YAMLs keep their names.

### Test changes

- **Drop**: tests asserting legacy `--cache-dirs` flat-flag routing (3 tests in `tests/test_eval_config.py` from C2 fix: `test_cli_filter_flag_preserves_yaml_filter_siblings`, `test_cli_cache_dirs_single_path_routes_to_cache_dir`, `test_cli_cache_dirs_with_colons_rejects`). Replace with two tests:
  - `test_cli_rejects_unknown_flat_flag` — verify `--seed 42` causes `argparse: unrecognized arguments`
  - `test_cli_three_flags_only` — assert argparse parser has only `{--benchmark, --config, -o, --override, -h, --help}`
- **Drop**: existing `test_parse_cli_only` (it passes `--num 20` flat; that flag is gone). Replace with a `--config` smoke test.
- **Add**: 4 tests for `PrecacheConfig` — schema, loader, filter validation, dot-path override.

Total expected after migration: ~22 tests in `test_eval_config.py` + 4 in `test_precache_config.py` = 26.

### Documentation

`Experiment/core_code/CLAUDE.md` "eval.py configuration" section rewritten:
- Section 2 ("Flat CLI flags") is deleted.
- Sections renumber: YAML file → `-o` overrides (now §1 and §2).
- "Migrating an old shell script" section drops (no shell scripts left in old form by the time docs land).
- New mention: `precache_explores.py` follows the same `--config + -o` pattern.

## Components & file map

| Action | Path |
|---|---|
| MODIFY | `Experiment/core_code/eval.py` (`parse_cli` body, async_main filter call site) |
| MODIFY | `Experiment/core_code/eval_config.py` (drop `flat_overrides` from `load_config`; extract loader to be schema-generic) |
| CREATE | `Experiment/core_code/precache_config.py` |
| MODIFY | `Experiment/core_code/precache_explores.py` (replace `parse_args` with `parse_cli`-style loader) |
| MODIFY | `Experiment/core_code/benchmarks/base.py` (delete `add_dataset_args`, `add_model_args`) |
| MODIFY | `Experiment/core_code/benchmarks/{hle,lcb,gpqa,babyvision,aime,rbenchv}.py` (delete `add_dataset_args` overrides) |
| MODIFY | `Experiment/core_code/tests/test_eval_config.py` (drop 3 legacy tests, drop 1, add 2) |
| CREATE | `Experiment/core_code/tests/test_precache_config.py` |
| CREATE then DELETE | `Experiment/core_code/scripts/migrate_eval_scripts.py` |
| CREATE | `Experiment/core_code/configs/<55 new YAMLs>` |
| RENAME | `configs/gpqa_multi_delegated_v2.yaml` → `configs/gpqa_multi_delegated_freeform.yaml` |
| DELETE | `configs/gpqa_multi_delegated_effort_high.yaml` (consolidated) |
| MODIFY | All 73 shell scripts under `Experiment/core_code/scripts/` (thin wrappers) |
| MODIFY | `Experiment/core_code/CLAUDE.md` (rewrite eval.py configuration section) |

## Data flow (final)

```
shell script
  └── python eval.py --benchmark X --config configs/Y.yaml [-o k=v ...]
      └── argparse: parses 3 flags only
      └── load_config(config_path, dot_overrides, schema=EvalConfig)
          ├── yaml.safe_load(config_path)
          ├── for ov in dot_overrides: _set_dotpath(merged, ...)
          └── EvalConfig.model_validate(merged)
      └── async_main(cfg) — unchanged from current main
```

## Error handling

- **Missing `--config`**: argparse error (required flag).
- **Missing YAML file**: `FileNotFoundError` from `open()`.
- **Malformed YAML (non-dict at top level)**: assertion in `load_config`.
- **Unknown YAML key**: pydantic `ValidationError` (`extra="forbid"`).
- **Bad dot-path syntax** (`-o foo`): assertion in `_set_dotpath` (must contain `=`).
- **Bad dot-path target** (`-o nonexistent.field=v`): pydantic `ValidationError`.
- **Filter key not declared by benchmark**: pydantic `ValidationError` (per-benchmark filter sub-model has `extra="forbid"`).
- **Method/cache mismatch** (e.g., `tts-agent-multi` without `cache_dirs`): existing `EvalConfig` validator handles this.
- **`migrate_eval_scripts.py` cannot parse a shell script**: hard-fails with the script path, prints suggested manual fix. Migration aborts; no partial state because the script writes-then-renames atomically per file.

## Migration safety

The migration script processes one shell script at a time, atomically:
1. Read `<script>.sh`.
2. Generate `<configs/...>.yaml` to a temp path.
3. Generate the rewritten `<script>.sh` to a temp path.
4. `os.rename(temp_yaml, final_yaml_path)` and `os.rename(temp_sh, original_sh)`. Both renames are atomic on the same filesystem.
5. If any step fails, the original `.sh` is intact.

After all scripts process: a final pass dry-parses every `.sh` via the new `parse_cli` (Task-8-style) to verify zero failures. Then `os.remove(__file__)`.

Git workflow: per the user's CLAUDE.md rule, all work happens directly on `main` (no new branch). Commits are split per logical step (see implementation plan).

## Testing strategy

1. **Unit tests for `PrecacheConfig`**: schema, loader, filter validation, dot-path override. 4 tests.
2. **Updated `test_eval_config.py`**: 21 existing - 4 stale + 2 new = 19 tests.
3. **End-to-end smoke**: pick 1 representative migrated YAML per benchmark family (5 YAMLs), parse-only invoke `parse_cli`, assert no exception.
4. **Dry-parse harness from Task 8**: re-run on all 73 migrated `.sh` files, assert 0 failures.
5. **broad pytest suite**: must stay green (43 tests previously passing).

## Open questions

None. (`precache_explores.py` scope: confirmed in. Self-deleting migration script: confirmed. v2-name cleanup: confirmed.)

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Migration script `shlex.split` mishandles edge-case shell quoting (backticks, `${VAR}` inside argv) | Low (verified 0/73 scripts use vars in argv) | Hard-fail on unparseable; user manually migrates that one |
| YAML filename collision after path-flatten | Low (paths verified unique) | Hard-fail on collision |
| precache_explores.py callers (other than shell scripts) | Unknown | Grep verifies no other callers (`grep -r "precache_explores" Experiment/`) before deleting `parse_args` |
| `cache_dir: Path` required in PrecacheConfig but a script omits it | Low (existing scripts always pass `--cache-dirs`) | Pydantic validation hard-fails the YAML load; user updates YAML |
| async_main reads `cfg.filters` but new validator emits `exclude_defaults=True` (drop unset keys) | None | This is current behavior on main; no change |
| Loader extraction (generic schema arg) breaks existing tests | Medium | `eval_config.load_config` keeps current signature; new schema-generic helper sits one layer beneath |

## Success criteria

- [ ] `parse_cli` body in `eval.py` ≤ 25 lines
- [ ] argparse exposes exactly 3 user-facing flags + `-h`
- [ ] `benchmarks/base.py` and 6 subclass files have NO `add_dataset_args` / `add_model_args` methods
- [ ] All 73 shell scripts use `--config FILE.yaml` form
- [ ] `Experiment/core_code/configs/` contains 72 YAMLs (55 new + 17 retained, after the 2 renames/consolidation)
- [ ] `migrate_eval_scripts.py` is gone (deleted itself)
- [ ] `pytest tests/` green
- [ ] Dry-parse all 73 `.sh` succeeds (0 failures)
- [ ] `Experiment/core_code/CLAUDE.md` "eval.py configuration" section reflects new flow only

## Out of scope (deferred)

- Schema versioning (e.g., `version: 1` in YAML)
- YAML includes / anchors / templating
- ENV var interpolation in YAML
- A separate "experiment registry" or run tracker
- Migrating non-eval, non-precache shell scripts (training scripts, etc.)
- `worktree-knot-compute` branch reconciliation
