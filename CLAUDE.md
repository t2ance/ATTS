# CLAUDE.md

For generic preferences (communication style, coding principles, monitoring, model training defaults) see the global `~/.claude/CLAUDE.md`. For verl / FSDP / vLLM training perf knobs, the symmetric remaining-memory principle, the offload-last-resort hierarchy, the production-token-budget rule, and stress-test recipe — see the `verl` skill (`references/hyper-param-tuning.md`). For vLLM `--safetensors-load-strategy` and inference save discipline — see the `vllm` skill (`references/optimization.md`). For agent-SDK transcript naming and tokenizer-vs-char rules — see the `claude-code` skill (`references/engineering-discipline.md`).

This file holds only the Explain-project-specific facts.

RUNNING CLAUDE AS BACKEND DOES NOT USE AN API KEY. JUST RUN IT.

## Writing
Whenever having modified the main.tex, remember to use compile.sh to re-compile the source file to get the latest pdf.

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

## API-key freshness in long-running shells (incident 2026-05-04)

Pre-flight: before any eval that calls a paid API, curl the credential. Do NOT trust that `$OPENROUTER_API_KEY` / `$ANTHROPIC_API_KEY` "is set" — verify the endpoint returns HTTP 200, not just that the var is non-empty.

```bash
curl -sS -o /dev/null -w "HTTP=%{http_code}\n" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/auth/key
# expect HTTP=200; HTTP=401 with body {"error":{"message":"User not found."}} = dead key
```

The trap: a long-lived bash session (and any Claude Code session forked from it) **fixates env at startup**. Editing `~/.bashrc` afterward does NOT propagate to the running shell or its children. The parent bash on this box has been alive >1 day; commenting out an old `OPENROUTER_API_KEY` line and adding a new one does not refresh the env until the shell is restarted.

The naive fix `bash -c 'source ~/.bashrc; ...'` does NOT work either: stock `~/.bashrc` opens with `case $- in *i*) ;; *) return;; esac` — non-interactive subshells `return` at line ~9 before reaching any `export`. Symptom: launch looks normal, eval starts, OpenRouter returns 401 "User not found." on every orchestrator turn → predicted="" → judge grades empty answers as wrong → Pass@1 collapses to 0-3% on questions that should score ~9%.

To pull a fresh `export FOO=...` line out of `~/.bashrc` into a non-interactive launcher, grep the active line (regex anchored to `^[[:space:]]*export` excludes `#export` commented lines) and `eval` it:

```bash
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py ... &
```

This sources only the one line you need, never inlines the secret on the command line, and bypasses the `case $-` guard. The key still never appears in `ps`, `nohup.out`, or shell history.

Concrete cost of skipping pre-flight (2026-05-04): two consecutive eval launches (run_20260504_051728, run_20260504_054240) each ran for several minutes, made 36-58 dead-on-arrival OpenRouter calls, and graded empty answers with Haiku before being killed and archived as `_BROKEN_orch401`. Both broken runs are preserved under `analysis/run/hle/openrouter_gpt-oss-20b_free_low/run_*_BROKEN_orch401/` for forensic reference.

## OpenRouter provider compatibility cheat-sheet (verified 2026-05-04 16:30 UTC)

`backends/openrouter.py` forces `tool_choice={"type":"function","function":{"name":"StructuredOutput"}}` on every call to guarantee structured output (the `response_format=json_schema` path triggers a vllm grammar-loop bug; see `call_sub_model` docstring 2026-05-03 incident). Forced `tool_choice` is NOT universally supported across OpenRouter's upstream provider list — `tools=True` in `/api/v1/models` is necessary but not sufficient. The probe tool `tmp/probe_provider_compat.py` exhaustively tests every provider for every candidate model.

For any new model in this project, run `python tmp/probe_provider_compat.py` (edit MODELS list) before writing yamls. The compatibility classes:

- **STRUCTURAL OK** — provider returns 200 + `tool_calls` populated. Safe to use.
- **STRUCTURAL 404** — provider listed in `/endpoints` API but actually not deployed (OpenRouter metadata lag). Strict pin to it returns "No endpoints found for ...".
- **STRUCTURAL 400** — provider deployed but rejects forced `tool_choice` schema. Common pattern: reasoning-only inference engines (DeepSeek's deepseek-reasoner endpoint).
- **STRUCTURAL 200-no-tool_calls** — provider returns 200 but ignores `tool_choice`, emits free-form text. Caches as `timed_out reason=no_tool_call`. Model-side issue.
- **TRANSIENT 429 / 502** — rate limit or upstream incident. Snapshot-only state; survives in production via SDK retry, but bursting num_workers=4 against rate-limited providers wastes time.

### Verified pins per model (2026-05-04)

Use `provider_order: [...]` and `provider_allow_fallbacks: false` in yaml. Strict pin = OpenRouter never falls through to broken providers. If pinned provider is fully down, request 404s loudly rather than silently failing on a broken fallback.

| Model | `provider_order` | Notes |
|---|---|---|
| `openai/gpt-oss-120b:free` | `[]` (default routing) | Single provider OpenInference; default route empirically works for 100/100 evals despite occasional 400 in single-shot probes |
| `openai/gpt-oss-20b:free` | `[]` | Same as above; 100/100 HLE eval landed 2026-05-04 |
| `google/gemma-4-26b-a4b-it:free` | (UNUSABLE) | Google AI Studio is the only provider; structurally returns 200 without `tool_calls` (model ignores forced `tool_choice`) AND throttles aggressively. DEFERRED in `todo_openrouter_hle_gemma-4-26b-a4b-it_free.md`. Revival requires either paid Google endpoint or different prompting strategy. |
| `x-ai/grok-4.1-fast` | `[]` | Single provider xAI; default works |
| `deepseek/deepseek-v4-flash` | (UNUSABLE without BYOK) | DeepSeek (official) 400's tool_choice; 4 providers 404 (listed-not-deployed); DeepInfra 429+502 unreliable; AkashML is the only structurally-clean route, but is throttled at the SHARED OpenRouter→AkashML key — even a single `max_tokens=10` call returns 429 "temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits". Production verdict 2026-05-04: 1 explore takes ~8 min through SDK 429-retry storm; 800 explores would need >5 days wall-clock. Use `deepseek-v4-pro` instead, OR BYOK at OpenRouter `/settings/integrations`. |
| `deepseek/deepseek-v4-pro` | `[]` (default routing) | Default routes successfully to a non-DeepSeek provider (Parasail confirmed working in single-call probe); 4 providers 404 (GMICloud/AtlasCloud/SiliconFlow/Novita listed-not-deployed); DeepSeek (official) still 400's tool_choice; Together 429-flaky. Pricing $0.435/$0.870 per M (3× v4-flash). For HLE-100 budget ~$13-20. |
| `moonshotai/kimi-k2.6` | `["Io Net", "Parasail", "Inceptron", "Venice"]` | 4 verified providers; multi-pin gives natural fallback within strict mode (OpenRouter tries them in order before failing). |
| `google/gemini-3-flash-preview` | `[]` | Both Google AI Studio and Google providers work; default routing fine |

### When the matrix is wrong

The probe is a moment-in-time snapshot. Production findings override:

- DeepInfra showed 429 in our probe but earlier in the same day successfully served 1 request. Rate limits churn.
- OpenInference 400'd gpt-oss-* in the probe, but full 100/100 HLE evals landed earlier the same day. Single-shot probes can mislead on transient outages.

**If a production run reports `BadRequestError` or `transient_api_error` rates >5%**, re-run the probe to refresh the matrix before adjusting `provider_order`. Don't blindly add providers based on stale matrix entries.

### How `provider_order` plumbs through

1. `BackendConfig.provider_order` (eval yamls; `methods/specs.py:65-78`) and `PrecacheConfig.provider_order` (precache yamls; `precache_explores.py:55-64`) accept the list at config-load time.
2. `eval.py:781-788` and `precache_explores.py:209-214` call `backends.openrouter.set_provider(...)` once at process startup.
3. `backends/openrouter.py:_maybe_inject_provider` injects `extra_body["provider"] = {"order": [...], "allow_fallbacks": bool}` on every `chat.completions.create` call in both `call_sub_model` (line ~149) and `run_tool_conversation` (line ~332).

If you add a new model and don't see provider pin in extra_body, check both call sites are covered.

## Per-benchmark grading reference

Each benchmark's grading logic is FIXED — we never swap judge models or grading strategies in practice. The `judge_model` class attribute and `_JUDGE_MODEL_CODEX` mapping in `benchmarks/base.py` are flexibility hooks that are not currently exercised. This table is the source of truth for "how is X graded".

Verified by reading `benchmarks/*.py` and `benchmarks/grader.py` on 2026-04-28.

## Thining budget configuration
For judge of answer, ALWAYS use non-thinking to save money.

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
