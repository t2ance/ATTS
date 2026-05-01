# CLAUDE.md

For generic preferences (communication style, coding principles, monitoring, model training defaults) see the global `~/.claude/CLAUDE.md`. For verl / FSDP / vLLM training perf knobs, the symmetric remaining-memory principle, the offload-last-resort hierarchy, the production-token-budget rule, and stress-test recipe — see the `verl` skill (`references/hyper-param-tuning.md`). For vLLM `--safetensors-load-strategy` and inference save discipline — see the `vllm` skill (`references/optimization.md`). For agent-SDK transcript naming and tokenizer-vs-char rules — see the `claude-code` skill (`references/engineering-discipline.md`).

This file holds only the Explain-project-specific facts.

## Local box GPU topology

On this physical machine:

- GPU 0 — carries the claude embedding daemon (~1.37 GB resident). Use as the judge + auxiliary card.
- GPU 1, GPU 2 — symmetric free memory, use these for training (FSDP + colocated vLLM).
- GPU 3 — blocked by another user.

This split exists to satisfy the symmetric remaining-memory principle (see `verl` skill `references/hyper-param-tuning.md` §8): training ranks must have equal free memory at init, so we keep the daemon off the training set.

## Conda Environment

**MANDATORY: All experiment scripts (eval.py, precache_explores.py, and any other research code under `Experiment/`) MUST run in the `explain` conda env.** This is a hard requirement — the base env has different package versions and Python 3.13 vs explain's 3.11; results across envs are NOT comparable.

Invocation pattern for every launcher .sh:

```bash
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
    --config <path> ... \
    >  <log> 2>&1 &
```

- `-n explain` selects the env without `conda activate` (works inside scripts where activation hooks may not be sourced).
- `--no-capture-output` (alias `-s` / `--live-stream`) is **mandatory** for long-running background jobs. Without it `conda run` buffers stdout/stderr until process exit, leaving the log file empty until the run finishes — defeating the entire purpose of `tail -f`.
- `PYTHONUNBUFFERED=1` is kept as belt-and-suspenders even though `--no-capture-output` already streams.

Never call bare `python` in a launcher; that inherits whatever env was active in the parent shell (usually `base`), which is silently wrong.

## Per-benchmark grading reference

Each benchmark's grading logic is FIXED — we never swap judge models or grading strategies in practice. The `judge_model` class attribute and `_JUDGE_MODEL_CODEX` mapping in `benchmarks/base.py` are flexibility hooks that are not currently exercised. This table is the source of truth for "how is X graded".

Verified by reading `benchmarks/*.py` and `benchmarks/grader.py` on 2026-04-28.

### Routing logic (`benchmarks/grader.py:135-147`)

`grade_answer(predicted, gold, question, answer_type, judge_model, ...)` decides:

```
if answer_type == "multipleChoice":   return check_answer(...)         # string match A-E
if judge_model is None:               return check_answer(...)         # string match
else:                                 return judge_answer(...)         # LLM judge
```

So the routing depends on TWO things: per-row `answer_type` and per-class `judge_model`. Below is the materialized truth table per benchmark.

### Per-benchmark table

| Benchmark | grade() override? | answer_type | judge_model | Effective grader |
|---|---|---|---|---|
| LCB | `lcb.py:142` → `grade_code(predicted, row)` | (bypassed) | `None` | Run predicted code against `public_test_cases` + `private_test_cases` via `lcb_runner.evaluation.compute_code_generation_metrics`. is_correct = all tests pass. |
| AIME (2025/2026) | `aime.py:119` → string equality | `exactMatch` | `None` | `_normalize_aime_answer(predicted) == _normalize_aime_answer(gold)`. Strips `\boxed{}`, `$…$`, lowercases. Integer-string match. |
| GPQA | none (uses `base.py:379`) | `multipleChoice` (`gpqa.py:104`) | `None` | grader.py short-circuits at `multipleChoice` → `check_answer` → `_extract_mc_letter` regex extracts A-E from predicted, compares to gold letter. |
| HLE | none (uses `base.py:379`) | per-row from dataset (usually `exactMatch`) | `claude-haiku-4-5-20251001` (`hle.py:123`) | LLM judge via `judge_answer`. Haiku reads (question, predicted, gold) and emits a yes/no semantic-equivalence verdict. Required because HLE answers are LaTeX/free-form text. |
| BabyVision | none (uses `base.py:379`) | hybrid: `multipleChoice` if `ansType=="choice"` else `exactMatch` (`babyvision.py:89`) | `claude-haiku-4-5-20251001` (`babyvision.py:50`) | Hybrid: choice questions → string match A-E; blank questions → LLM judge (Haiku). The judge_model only fires on blank questions. |
| RBenchV | none (uses `base.py:379`) | `exactMatch` (default from `base.py:372`) | `claude-haiku-4-5-20251001` (`rbenchv.py:40`) | LLM judge (Haiku). Visual reasoning answers are free-form, semantic equivalence required. |

### Key facts to remember

- **judge_model is always Haiku or None.** No benchmark currently uses any other judge model. The `_JUDGE_MODEL_CODEX` codex-mapping is dormant.
- **Three grading mechanisms exist**: code execution (LCB), string match (AIME / GPQA / BabyVision-choice), LLM judge (HLE / BabyVision-blank / RBenchV).
- **`judge_model = None` is correct for LCB / AIME / GPQA**, not a leftover. LCB doesn't need a judge (test cases). AIME's integer-string match doesn't need one. GPQA's multipleChoice short-circuits before judge_model is consulted.
- **Once-burned: HLE.** `hle.py:123`'s `judge_model` was temporarily set to `None` for a 2026-04-11 smoke test and not restored, causing all 100 HLE socratic-self-refine grades to fall through to `check_answer` with `gold=""`, underestimating accuracy by ~8 pp. Restored 2026-04-28. Lesson: any non-trivial override of these class defaults must carry a comment with date + rationale + rollback instruction.

### Cache validity

`eval.py:_grade_with_cache` (line 45-73) caches verdicts as `grade.json` and trusts the cache only if the stored `judge_model` matches the current `benchmark.judge_model`. This protects against the HLE-class bug (changing the class attribute auto-invalidates old cache). It does NOT protect against:

- Backend swap (claude → codex): not part of cache key.
- Gold answer changes (dataset upstream updates): stored in cache file but not checked on read.
- Question text changes: same as above.
- LCB: `judge_model` is always `None` → cache key always matches → stale grades from lcb_runner bugs would persist forever. Mitigation: never modify lcb_runner without invalidating the LCB cache directory.

If you change any of the above (backend, dataset version, lcb_runner version, judge prompt template), bump or wipe the affected cache directory; the auto-invalidation only catches `judge_model`.
