# JudgeSpec Block: Configurable Judges with Multi-Bundle Cache

**Date:** 2026-05-01
**Status:** Spec (pending implementation plan)

## Problem

The judge model used for LLM-graded benchmarks is currently a hardcoded class attribute on each `BenchmarkConfig` subclass. Six files declare it: `hle.py`, `babyvision.py`, `rbenchv.py` carry `judge_model = "claude-haiku-4-5-20251001"`; `gpqa.py`, `lcb.py`, `aime.py` carry `judge_model = None`. The judge backend and any bespoke routing (HLE's `vllm -> claude` swap, `codex -> gpt-5-codex-mini` swap) live inside each benchmark's `.grade()` method.

Three problems follow:

1. There is no YAML knob to switch judge model. To change it, the user edits source code.
2. There is no place to attach configuration that is judge-specific. A vLLM-served judge needs its own sampling block; today there is no schema slot for it.
3. The cache key for a graded explore is the judge model name (single string). Changing judge invalidates every prior verdict, even though those verdicts cost real money and would still be valid as historical records.

This spec replaces the class-attribute pattern with a YAML-driven `judge:` block nested inside each benchmark's spec, and replaces the single-verdict cache file with a per-judge bundle directory so that multiple judges can grade the same explore side-by-side without overwriting each other.

## Verified preconditions

Confirmed 2026-05-01 by reading the codebase:

- `judge_model` class attribute declared in 6 benchmark files (paths above).
- `eval.py:_grade_with_cache` (line 201) takes `benchmark.judge_model` as the cache key on read (line 208) and write (line 223).
- `_grade_cached_explores` (line 230) and resume paths (lines 421-446, 461, 480, 492-494, 527) all read `benchmark.judge_model` for cache lookups.
- `benchmarks/grader.py:grade_answer` signature: `(predicted, gold, question, answer_type, judge_model, backend, out_dir)`.
- `benchmarks/grader.py:judge_answer` writes `input.md`, `output.md`, `result.json` into `out_dir`.
- `eval.py:_grade_with_cache` writes `grade.json` next to the `judge/` subdir at `out_dir.parent`.
- HLE has bespoke routing in `.grade()` (line 158-160): swaps `vllm -> claude` for grading, swaps `codex -> gpt-5-codex-mini` for judge model.
- One existing run dir has the legacy layout intact: `analysis/run/hle/opus/run_20260307_084618/grading/<qid>/{grade.json, input.md, output.md, result.json, judge/}`.
- `analysis/cache/hle/qwen36_35b_a3b_fp8/gold/` has 732 `grade.json` files written under the legacy layout (today, 2026-05-01).

## YAML schema

The `judge` block is a sub-field of `benchmark`. Benchmarks that grade without an LLM judge (LCB by code execution, GPQA and AIME by string match) do not declare `judge`. Pydantic `extra="forbid"` rejects `judge:` from those benchmark specs at load time.

HLE / BabyVision / RBenchV with a Claude judge:

```yaml
benchmark:
  name: hle
  subset: gold
  judge:
    name: claude
    model: claude-haiku-4-5-20251001
```

BabyVision with a vLLM-served judge that carries its own sampling block:

```yaml
benchmark:
  name: babyvision
  type: choice
  judge:
    name: vllm
    model: qwen36-35b-a3b-fp8
    sampling:
      temperature: 0.6
      max_tokens: 4096
```

LCB / GPQA / AIME never carry `judge`:

```yaml
benchmark:
  name: lcb
  difficulty: medium
  # writing `judge:` here triggers a Pydantic validation error
```

`JudgeSpec` is itself a discriminated union over `name`:

| `judge.name` | Required fields | Notes |
|---|---|---|
| `claude` | `model: str` | Routes through Claude SDK. |
| `codex`  | `model: str` | Routes through Codex SDK. |
| `vllm`   | `model: str`, `sampling: SamplingConfig` | Routes through vLLM HTTP. `sampling` is required (no implicit defaults), matching the existing `SamplingConfig` used for explorer / orchestrator in `eval.py`. |

## Filesystem layout

A graded explore directory holds explorer artifacts at the top level (judge-agnostic) and a `judges/` subdirectory containing one bundle per judge that has graded this explore:

```
explore_N/
├── input.md            # explorer prompt (judge-agnostic)
├── output.md           # explorer raw output
├── result.json         # explorer structured result
└── judges/
    ├── claude__claude-haiku-4-5-20251001/
    │   ├── config.json   # full JudgeSpec dump (identity source-of-truth)
    │   ├── grade.json    # this judge's verdict
    │   ├── input.md      # OPTIONAL: judge prompt (only present in older caches)
    │   ├── output.md     # this judge's raw output
    │   └── result.json   # this judge's structured output
    └── vllm__qwen36-35b-a3b-fp8/
        ├── config.json
        ├── grade.json
        ├── output.md
        └── result.json
```

The label `<backend>__<model>` is a human-readable directory name. The source of truth for judge identity is `config.json`, which contains the full `JudgeSpec` dump (including `sampling` for vLLM judges).

This layout applies to two locations:

1. `analysis/cache/<bench>/<model>[/<filter>]/<qid>/explore_N/` — long-lived, shared across runs.
2. `analysis/run/<bench>/<model>/run_<ts>/grading/<qid>/[/rollout_<r>]/explore_N/` — per-run record.

## Lookup logic

```python
def find_cached_judge(judges_dir: Path, requested_spec: dict) -> Path | None:
    label = f"{requested_spec['name']}__{requested_spec['model']}"
    candidate = judges_dir / label
    if not candidate.exists():
        return None
    stored = json.loads((candidate / "config.json").read_text())
    if stored == requested_spec:
        return candidate
    raise RuntimeError(
        f"Judge label collision at {candidate}.\n"
        f"  Stored config:    {stored}\n"
        f"  Requested config: {requested_spec}\n"
        f"Two judges share the same backend+model label but differ on other "
        f"fields (e.g. sampling). Manually rename one of the conflicting "
        f"bundles before re-running."
    )
```

### Cache miss behavior

`find_cached_judge` is a pure read. The caller decides what to do on miss:

- **Eval mode (default caller):** on miss, call the judge, write the 5-file bundle to `judges/<label>/`.
- **Cache-only caller** (analysis tools that must not spawn fresh judge calls, signalled by a `cache_only=True` flag on the caller side, mirroring the existing `make_sub_model_caller(cache_only=...)` pattern): on miss, raise. The user explicitly requested a specific judge that has not yet graded this explore, and the caller is not allowed to fill the gap.

### Label collision behavior

If two bundles would share the same `<backend>__<model>` label but their `config.json` contents differ (for example, two vLLM judges on the same model but with different sampling), `find_cached_judge` raises. The user resolves manually by renaming one of the conflicting bundles. No automatic suffix or hash disambiguation. Rationale: cross-sampling judge comparison is a research path, not a daily flow; making it explicit prevents silent duplication.

## Migration

Pure file-move migration over `analysis/cache/<bench>/<model>[/<filter>]/<qid>/explore_N/`. No file content is rewritten; the only new file created is `config.json`.

Pre-migration layout:

```
explore_N/
├── grade.json          # {"judge_model": "<m>", "is_correct": <b>, ...}
├── input.md
├── output.md
├── result.json
└── judge/
    ├── input.md
    ├── output.md
    └── result.json
```

Migration steps for each `explore_N` that has both `grade.json` and `judge/`:

1. Read `grade.json`, extract `judge_model`. All current entries are `claude-haiku-4-5-20251001`; if any other value appears, the migration script aborts and asks the user to inspect.
2. Compute `<label>` = `claude__<judge_model>`.
3. `mkdir -p judges/<label>`.
4. `mv grade.json judges/<label>/grade.json`.
5. `mv judge/input.md judges/<label>/input.md`.
6. `mv judge/output.md judges/<label>/output.md`.
7. `mv judge/result.json judges/<label>/result.json`.
8. `rmdir judge`.
9. Write `judges/<label>/config.json` with `{"name": "claude", "model": "<judge_model>"}`.

Per-run grading dirs (`analysis/run/.../grading/`) are not migrated. They are per-run records; they are not consumed by future runs. New runs write the new layout from day 1.

### Migration safety (mandatory)

The migration touches 732 graded explores in `analysis/cache/hle/qwen36_35b_a3b_fp8/gold/` worth $28.17 of judge cost, and an unknown additional count across other benchmarks' caches. A bug in the script could lose all of it. The script must therefore use **copy-then-cleanup**, not `mv`, with a verification pass between the two:

The script exposes one CLI flag `--phase {dry-run, copy, cleanup}` with no default. Each invocation runs exactly one phase. Phases must be run in order; later phases assert earlier phases completed.

**Phase 1: dry-run.** Walks the cache tree and prints, for each `explore_N`, the destination paths it would create. No I/O. Output is a human-readable report ending with `Total: <N> explores eligible for migration; <M> already migrated; <K> skipped (no grade.json)`. Run this first, sanity-check the counts and a few sample paths.

**Phase 2: copy.** For each eligible `explore_N`:

1. `mkdir -p judges/<label>/` (idempotent).
2. Write `config.json` with the inferred `JudgeSpec` dump. Use `tempfile + os.rename` for atomic write.
3. Copy (not move) `grade.json -> judges/<label>/grade.json`. Verify byte-for-byte equality of source and destination via `hashlib.sha256` before continuing.
4. Copy each of `judge/{input.md, output.md, result.json}` -> `judges/<label>/{input.md, output.md, result.json}`, with the same byte-equality verification per file.
5. Read `judges/<label>/grade.json` and verify `judge_model` matches the inferred `<label>`'s model. If not, raise.
6. On any failure in steps 1-5, raise immediately. Original `grade.json` and `judge/` are untouched on disk; the new bundle is partial but identifiable by missing/incomplete files; re-running phase 2 is safe (idempotent).

After phase 2 completes for the entire tree, run a smoke eval (the existing 100-question HLE `_exp_orch` config) against the migrated cache and assert: judge cost == $0, accuracy matches the pre-migration run's 23/100. If either check fails, the migration is wrong; do not proceed to phase 3.

**Phase 3: cleanup.** Walks the cache tree again. For each `explore_N` where `judges/<label>/` is fully populated (all 5 files present + config.json hashes match expected `JudgeSpec`):

1. `rm explore_N/grade.json`.
2. `rm -rf explore_N/judge/`.

Phase 3 only deletes after explicit per-file verification of the new bundle's completeness. If verification fails for a given explore, the script logs and skips (does not delete) — leaving that explore's legacy files intact for manual inspection.

Recommended pilot: before running phase 2 on the full tree, run it with `--limit 5` to migrate 5 explores, then manually `diff -r` the source and destination files, then run a 5-question eval and verify cache hits. Only after the pilot passes, run the full phase 2.

## Code changes

| File | Change |
|---|---|
| `benchmarks/specs.py` | Add `JudgeSpec` discriminated union (`ClaudeJudgeSpec`, `CodexJudgeSpec`, `VllmJudgeSpec`). Add `judge: JudgeSpec` field to `HLESpec`, `BabyVisionSpec`, `RBenchVSpec`. Other specs unchanged; their `extra="forbid"` rejects `judge:`. |
| `benchmarks/base.py` | Drop `judge_model: str \| None` annotation. Add `BenchmarkConfig.__init__(self, judge_spec: dict \| None = None)`. Add `find_cached_judge(judges_dir, judge_spec)` static helper. |
| `benchmarks/grader.py` | Replace `grade_answer(... judge_model, backend, out_dir)` with `grade_answer(... judge_spec, out_dir)`. Internally dispatch to `claude` / `codex` / `vllm` backend by `judge_spec["name"]`. `judge_answer` similarly takes `judge_spec` and unpacks. |
| `benchmarks/hle.py` | Drop `judge_model = ...`. Drop the bespoke `vllm -> claude` and `codex -> gpt-5-codex-mini` routing inside `.grade()`. The replacement is purely YAML-driven. |
| `benchmarks/babyvision.py` | Drop `judge_model = ...`. `.grade()` reads `self.judge_spec`. |
| `benchmarks/rbenchv.py` | Same as `babyvision.py`. |
| `benchmarks/gpqa.py`, `benchmarks/lcb.py`, `benchmarks/aime.py` | Drop `judge_model = None` (the attribute no longer exists). No other change. |
| `eval.py` | Replace every read of `benchmark.judge_model` with a `find_cached_judge` call against the explore's `judges/` dir. Update `_grade_with_cache` to write the 5-file bundle (including `config.json`) into `judges/<label>/`. |
| `precache_explores.py` | No change. Precache does not grade. |
| **(new)** `scripts/migrate_judge_layout.py` | One-shot migration. Walks every `cache_base/<bench>/<model>[/<filter>]/<qid>/explore_N/` and applies the steps above. Idempotent: if `judges/<label>/` already exists, skip. |
| **(new)** `tests/test_judge_spec.py` | Unit tests for `JudgeSpec` validation (LCB rejecting `judge:`, vLLM judge requiring `sampling`, etc.) and for `find_cached_judge` (hit, miss, label collision). |

## Validation

- **Migration correctness.** After running the migration script over `analysis/cache/hle/qwen36_35b_a3b_fp8/gold/`, every previously-graded `explore_N` directory has `judges/claude__claude-haiku-4-5-20251001/` containing all 5 files, `grade.json` content unchanged, and the old `grade.json` and `judge/` paths no longer exist.
- **Cache reuse.** Re-running the HLE `_exp_orch` eval against the migrated cache must hit `find_cached_judge` for every explore -> 0 judge calls -> $0 Haiku cost. Today the same eval cost $28.17.
- **Multi-judge coexistence.** Running a second eval with `judge.name: vllm` on the same cache produces new bundles under `judges/vllm__<model>/` without touching the existing `judges/claude__<...>/` bundles.
- **Schema rejection.** A YAML with `benchmark.name: lcb` and `benchmark.judge: { ... }` fails Pydantic validation at load time.
- **Label collision.** Crafting two `judges/vllm__<model>/` bundles whose `config.json` contents differ on `sampling` and pointing eval at one of them must raise `RuntimeError` with the diff in the message.

## Out of scope

- Per-run grading dirs are not migrated. They were never read across runs.
- Concurrent multi-judge grading (running multiple judges on the same explore in the same run) is not part of this spec. Each run grades with one judge.
- The `grading_summary` class attribute (added in the 2026-04-28 grading-cleanup plan) is preserved as-is. It will reference the YAML-supplied judge instead of a hardcoded class attribute.
- No changes to non-judge cache layers (`result.json` for explorer outputs).
