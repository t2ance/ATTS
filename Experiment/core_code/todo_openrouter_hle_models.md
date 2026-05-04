# TODO: HLE Eval — 7 OpenRouter Models via claude.py Backend (free auto, paid gated)

## What this is

Run ATTS evaluation on the HLE benchmark for 7 OpenRouter models via the
already-verified `backends/openrouter.py` path. Goal: collect HLE-Verified
results across a model panel that spans (a) cost-zero free models for
methodology validation and (b) representative paid frontier models for paper
backbone-ablation rows. Tool-calling capability has been confirmed for all 7
via `/api/v1/models` `supported_parameters` (each lists `tools` + `tool_choice`
+ `reasoning`); end-to-end ATTS pipeline integrity has been confirmed on
`openai/gpt-oss-120b:free` (trajectory + cache files + $0 OpenRouter usage delta
on 1-question precache; see `todo_openrouter_via_claude.md` Phase 3 verdict).

Three free models run unattended; four paid models require explicit per-model
user approval before any API call is made (cost-consent rule). HLE is the
single benchmark in scope here — other benchmarks (BabyVision / LCB / GPQA)
are deferred to a separate TODO once this lands.

## Output target

Per model: `analysis/run/hle/openrouter_<model_slug>/run_<TS>/results.jsonl`
with one row per HLE question, plus per-explore caches at
`analysis/cache/hle/openrouter_<model_slug>/<qid>/explore_<n>/result.json`.

Paper integration: tab:backbone-ablation at `Publication/paper/main.tex` line
402-420 (existing GPT-5.2 vs Sonnet rows). Add 3-7 rows for these models
depending on which paid runs are approved.

## Discipline

Every event below has explicit Gates with checkboxes. An event flips from `☐`
to `✓` only after all Gates pass AND each Gate's `Evidence ·` line is filled
with the actual measurement (qid counts, OpenRouter usage delta, log line
ref, results.jsonl line count — not "looks fine"). No silent skipping. No
narrative-only claims. No marking done before Evidence is recorded. A failed
Gate stops the run; do not advance until the failing Gate is either resolved
or escalated to the user.

## Cost-consent gate (NON-NEGOTIABLE)

- Items 03 / 04 / 05 (free models): proceed automatically when prerequisite
  Gates pass.
- Items 06 / 07 / 08 / 09 (paid models): **STOP at the start of each item;
  surface estimated cost to user; wait for explicit "approved" reply** before
  any API call. Do NOT batch the four paid models for a single approval —
  approve per-model so an over-budget line item can be skipped without
  re-evaluating the rest.

## Model lineup (in execution order)

| # | Model id | Pricing | Tool support (verified `/v1/models`) | Auto/Gated |
|---|---|---|---|---|
| 1 | `openai/gpt-oss-120b:free` | $0 / $0 | tools + tool_choice + reasoning | Auto |
| 2 | `openai/gpt-oss-20b:free` | $0 / $0 | tools + tool_choice + reasoning | Auto |
| 3 | `x-ai/grok-4.1-fast` | $0.20 / $0.50 per 1M | tools + tool_choice + reasoning | Gated |
| 4 | `deepseek/deepseek-v4-pro` | $0.435 / $0.870 per 1M | tools + tool_choice + reasoning | Gated |
| 5 | `moonshotai/kimi-k2.6` | $0.74 / $3.49 per 1M | tools + tool_choice + reasoning | Gated |
| 6 | `google/gemini-3-flash-preview` | $0.50 / $3.00 per 1M | tools + tool_choice + reasoning | Gated |

Pricing is "USD per 1M tokens"; verified by `/api/v1/models` 2026-05-03 / 2026-05-04.

### Removed from lineup (2026-05-04 — verified unusable for this pipeline)

The lineup originally listed 7 models. Two were removed after end-to-end provider probing on 2026-05-04 confirmed they cannot run our forced-`tool_choice={"type":"function","function":{"name":"StructuredOutput"}}` pipeline reliably enough to commit a paper-row run. **Focus the panel on the 6 above.** Probe artifacts: `tmp/probe_provider_compat.py` script + `Publication/.../CLAUDE.md` "OpenRouter provider compatibility cheat-sheet".

- **`google/gemma-4-26b-a4b-it:free`** — STRUCTURAL: the only upstream provider (Google AI Studio) returns HTTP 200 but emits free-form text without populating `tool_calls`, regardless of how the request is shaped. Caches as `timed_out reason=no_tool_call` on every explore. The model layer ignores the forced-tool contract. The `:free` route is also frequently 429-throttled. Companion file `todo_openrouter_hle_gemma-4-26b-a4b-it_free.md` already carries the DEFERRED status block. Revival requires either swapping to free-form output parsing (drop forced `tool_choice`) OR a paid Google endpoint that honors the forced-tool contract.

- **`deepseek/deepseek-v4-flash`** — INFRASTRUCTURAL: 7 upstream providers listed; 4 return `404 No endpoints found` when strict-pinned (listed-but-not-deployed metadata lag), DeepSeek's official endpoint returns `400 deepseek-reasoner does not support this tool_choice`, DeepInfra intermittently 429+502, leaving AkashML as the only structurally clean route. AkashML is throttled at the SHARED OpenRouter→AkashML key — verified 2026-05-04 a single `max_tokens=10` probe still returns `429 temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits`. Production-load test 2026-05-04: 1 explore took ~8 min through SDK 429-retry storm, so 800 explores would need >5 days wall-clock. **Replaced by `deepseek/deepseek-v4-pro`** (~3× per-token cost but routes through Parasail by default; single-call probe returns 200 + populated `tool_calls`). If a v4-flash row is ever required, the path forward is BYOK at OpenRouter `/settings/integrations` with a personal AkashML API key.

## HLE benchmark scope

- Subset: `gold` (100 questions), text-only filter (paper main subset matches
  this configuration; `_template.yaml` style). Match what already exists in
  the paper main results so the new rows are directly comparable.
- ATTS config: `method: tts-agent`, `num_explores: 8`, `no_integrate: true`,
  `num_workers: 8`. Identical to the autonomous run that produced the existing
  Qwen-FP8 / Sonnet / GPT-5.2 backbone-ablation rows. **Do not vary these** —
  cross-row comparability requires fixed method config.
- Per-model cache_dir: `../analysis/cache/hle/openrouter_<slug>/gold` where
  `<slug>` is the model id with `/` and `:` and `.` replaced by `_`. E.g.
  `openrouter_openai_gpt-oss-120b_free`. NEW directory per model — never
  reuse another model's cache.

## Rate-limit anchor (free tier)

- OpenRouter free-model limits, this account: **1000 requests/day per model**
  (because account has ≥$10 credits; verified via `/auth/key.is_free_tier=false`
  + total_credits=$1660). Without credits the limit would be 50/day.
- Per-question call budget: 1 orchestrator + 8 explores = 9 calls/q.
- Per HLE-100 run: 100 × 9 = 900 calls per model — fits inside the 1000/day
  quota with ~10% headroom. Consecutive same-day runs of the same free model
  would bust the quota.
- Per-minute limit: 20 rpm. With `num_workers=8` and ~9 calls/q, the worst-case
  burst is 8 concurrent calls; a 60-second window can cap throughput. Soft
  fail expected: `precache_explores.py` already handles 429 retries at the
  HTTP layer.
- If HTTP 429 surfaces despite SDK retries, item-level Gate fails and the
  remediation is `--num_workers=4` (halve concurrency). Verify post-recovery.

## Per-model cost estimate (HLE-100, ATTS num_explores=8, no_integrate)

Empirical token-per-call baseline from probe A on `gpt-oss-120b:free`: 12226
input + 94 output. Probe B (orchestrator): 14179 input + 172 output. Mixed
average: ~13k input + ~150 output per call × 9 calls × 100 q = ~11.7M input +
~135k output per model.

| # | Model | Estimated $ for HLE-100 |
|---|---|---|
| 1 | gpt-oss-120b:free | $0 |
| 2 | gpt-oss-20b:free | $0 |
| 3 | grok-4.1-fast | 11.7M × $0.20/1M + 135k × $0.50/1M = $2.40 |
| 4 | deepseek-v4-pro | 11.7M × $0.435/1M + 135k × $0.870/1M = $5.21 |
| 5 | kimi-k2.6 | 11.7M × $0.74/1M + 135k × $3.49/1M = $9.13 |
| 6 | gemini-3-flash-preview | 11.7M × $0.50/1M + 135k × $3.00/1M = $6.26 |
| **Paid total (if all 4 approved)** | — | **~$23.0** |

Note: `deepseek-v4-pro` cost is upper-bound by token math; long-reasoning explores at `effort=low` may push output 5-10× the 135k average; if the live `auth/key.usage` delta exceeds 2× this estimate, treat as the R8 hard-stop case.

Estimates are upper-bounds (real input is shorter for non-thinking models).
Actual per-question $/q recorded in results.jsonl after each run via
OpenRouter `/auth/key.usage` delta, not the SDK's fabricated cost field.

## Co-monitor — log paths for parallel watching

| Phase | Log path |
|---|---|
| 0 (preflight) | `tmp/openrouter_hle_preflight_<TS>.log` |
| 1 (per-model precache + eval) | `tmp/openrouter_hle_<slug>_<TS>.log` |
| 2 (paper integration) | `tmp/openrouter_hle_paper_<TS>.log` |

Absolute path prefix: `/data3/peijia/dr-claw/Explain/Experiment/core_code/`.

## Risk register

| # | Failure | Root cause | Defense in this TODO |
|---|---|---|---|
| R1 | Free-tier daily quota busts (HTTP 429 after ~1000 reqs) | OpenRouter limit; even with $10 credits cap is 1000/day | Item 03/04/05 G3 — verify usage <950 reqs at end-of-run; if hit, defer next free run by 24h |
| R2 | per-model cache pollution from cross-model overlap | reused cache_dir | Item 02 G1 — assert cache_dir is empty AND unique per model |
| R3 | paid model billed for failed precache restart | cache miss + retry without checkpoint | Always pass `--cache-dirs <DIR>` AND `--resume <RUN>` per the project cache discipline |
| R4 | Hidden Haiku tax re-emerges if claude.py is edited and DISABLE flags drift | someone removes the 11-flag setdefault block | Item 02 G2 — grep verify 11 flags present in `backends/claude.py` |
| R5 | Free model silently downgraded to paid by OpenRouter | pricing change without renaming endpoint | Item 02 G3 — re-check `/api/v1/models` pricing per model right before launching |
| R6 | One free model's 429 cascades into batch failure for others | naive launcher kills all on first error | run free models SEQUENTIALLY, not in parallel — separate processes, separate quotas, no shared blast radius |
| R7 | gemma-4 thinking-mode bug (per `todo_gemma4_thinking_fix.md`) | jinja chat template + skip_special_tokens stripping | OpenRouter routes via Anthropic-Skin → bypasses local vLLM serve → Gemma jinja path NOT in scope. The thinking bug only fires on local vLLM serve, not when OpenRouter terminates the call upstream. Confirmed by `/api/v1/models` reasoning=true. |
| R8 | Paid run blast radius if estimate is wrong | thinking-mode generates 5-10× tokens than estimate | Per-paid-item G2 hard-stops at 2× estimate via `max_output_tokens` cap in YAML |

## Resume / restart procedure

| Failure point | Recover by | Banner verification |
|---|---|---|
| Precache crashes mid-run | Re-launch with `--cache-dirs <DIR>` (same dir); SDK skips already-cached qids | `Tasks: K to run, J already cached` with J>0 |
| Eval crashes mid-run | Re-launch with `--resume <RUN_DIR>` AND `--cache-dirs <DIR>` | `Resuming ...: N rollouts already completed` with N>0 AND `Questions to run: M (N already completed, M+N total)` with N>0 |
| 429 throttle | Wait until quota resets (UTC midnight per OpenRouter docs); halve `--num-workers` if recurs | log line `[run] resuming after rate-limit recovery` |
| Wrong model billed (paid response on free model) | Compare OpenRouter usage delta vs expected $0; if non-zero on a `:free` model, STOP; investigate provider-side fallback | `usage delta = $0` post-run |

## Phase 0 — Pre-flight [0/2]

01 ☐ Backend reuse stack still healthy
   ├ G1 ☐ Gate · `backends/openrouter.py` exists at expected path AND contains `from backends.claude import call_sub_model, run_tool_conversation`
   │      Evidence · 
   ├ G2 ☐ Gate · `backends/claude.py` line 17-50 contains the 11-flag setdefault block (DISABLE_TELEMETRY + 10 others). Without these, Haiku tax returns; free-tier guarantee breaks.
   │      Evidence · 
   ├ G3 ☐ Gate · OpenRouter `/auth/key` returns HTTP 200 (key still valid)
   │      Evidence · 
   └ How  · `python -c "from importlib import import_module; b=import_module('backends.openrouter'); print(b.call_sub_model)"`; `grep -c 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC' backends/claude.py` (expect ≥1); `curl /auth/key`

02 ☐ All 7 model entries valid + tool-capable + free pricing for the 3 free
   ├ G1 ☐ Gate · `/api/v1/models` lookup returns each of the 7 model ids; none return NOT FOUND
   │      Evidence · 
   ├ G2 ☐ Gate · For each of the 7, `supported_parameters` contains BOTH `tools` AND `tool_choice`
   │      Evidence · 
   ├ G3 ☐ Gate · For the 3 free models (gpt-oss-120b, gpt-oss-20b, gemma-4-26b-a4b-it), `pricing.prompt == "0" AND pricing.completion == "0"`
   │      Evidence · 
   └ How  · single inline python: query `/api/v1/models`, filter by 7 ids, print pricing + supported_parameters per id

## Phase 1 — Free models (auto-run) [0/2]

Cache discipline: each item creates a fresh cache_dir under `../analysis/cache/hle/openrouter_<slug>/gold`. Sequential execution (one model at a time) to keep blast radius small. No parallelism across free models.

03 ☐ `openai/gpt-oss-120b:free` — HLE-100 ATTS
   ├ G1 ☐ Gate · YAML built at `scripts/hle/openrouter/hle_gpt-oss-120b_free.yaml` with method=tts-agent, num_explores=8, no_integrate=true, num=100, num_workers=8, backend.name=openrouter
   │      Evidence · 
   ├ G2 ☐ Gate · Precache + eval complete; `wc -l` of results.jsonl returns 100 rows
   │      Evidence · 
   ├ G3 ☐ Gate · OpenRouter usage delta across the run is < $0.01 (≤ noise threshold; should be $0 — non-zero indicates provider-side paid fallback)
   │      Evidence · 
   ├ G4 ☐ Gate · timed_out rate ≤ 5% (5/100); zero `Traceback` in run log
   │      Evidence · 
   ├ G5 ☐ Soft-Gate · Pass@1 accuracy reasonable for HLE-Verified scale (no hard threshold — just sanity-check vs paper baselines around 9-15% range)
   │      Justification · cite the measured Pass@1, the range it falls within, and 1-2 example correct/incorrect qids
   │      Evidence · 
   └ How  · `bash scripts/hle/openrouter/run_hle_gpt-oss-120b_free.sh > tmp/openrouter_hle_openai_gpt-oss-120b_free_$(date +%Y%m%d_%H%M%S).log 2>&1 &` then share PID + log path

04 ☐ `openai/gpt-oss-20b:free` — HLE-100 ATTS
   ├ G1 ☐ Gate · YAML at `scripts/hle/openrouter/hle_gpt-oss-20b_free.yaml`; same shape as item 03 G1
   │      Evidence · 
   ├ G2 ☐ Gate · same as 03 G2 (100 rows)
   │      Evidence · 
   ├ G3 ☐ Gate · same as 03 G3 (delta < $0.01)
   │      Evidence · 
   ├ G4 ☐ Gate · same as 03 G4
   │      Evidence · 
   ├ G5 ☐ Soft-Gate · same as 03 G5
   │      Justification · 
   │      Evidence · 
   └ How  · same as 03 with model swapped

05 ⊘ `google/gemma-4-26b-a4b-it:free` — REMOVED 2026-05-04, see "Removed from lineup" section above. The Google AI Studio upstream provider returns HTTP 200 without populated `tool_calls` regardless of forced `tool_choice`, AND aggressively 429-throttles `:free` traffic; both failure modes verified by `tmp/probe_provider_compat.py` 2026-05-04. Companion `todo_openrouter_hle_gemma-4-26b-a4b-it_free.md` carries the DEFERRED status block. Do NOT re-add this item without first solving the no-tool-call symptom (likely requires switching to free-form output parsing — non-trivial pipeline change).

## Phase 2 — Paid models (cost-consent gated) [0/4]

For EACH paid item: STOP at item start; surface estimated $$$; wait for user reply containing "approved" before any API call. Do NOT pre-build YAMLs or warm caches before approval — pre-build is fine, API call is not.

06 ☐ `x-ai/grok-4.1-fast` — HLE-100 ATTS [requires approval; est ~$2.40]
   ├ G1 ☐ Gate · User explicit approval recorded in this Evidence line ("approved $X" or similar)
   │      Evidence · 
   ├ G2 ☐ Gate · YAML at `scripts/hle/openrouter/hle_grok-4.1-fast.yaml`; max_output_tokens=8000 to cap blast radius (R8)
   │      Evidence · 
   ├ G3 ☐ Gate · 100 rows in results.jsonl
   │      Evidence · 
   ├ G4 ☐ Gate · Final OpenRouter usage delta ≤ 2× estimate ($4.80); if exceeded, STOP and report
   │      Evidence · 
   ├ G5 ☐ Gate · timed_out rate ≤ 5%
   │      Evidence · 
   ├ G6 ☐ Soft-Gate · Pass@1 accuracy
   │      Justification · 
   │      Evidence · 
   └ How  · same launcher as item 03; share PID + log path

07 ☐ `deepseek/deepseek-v4-pro` — HLE-100 ATTS [requires approval; est ~$5.21]
   (REPLACED `deepseek/deepseek-v4-flash` 2026-05-04 — see "Removed from lineup"
   section above for v4-flash's AkashML shared-quota incompatibility.)
   ├ G1 ☐ Gate · User approval; same shape as 06 G1
   │      Evidence · 
   ├ G2 ☐ Gate · YAML at `scripts/hle/openrouter/hle_deepseek-v4-pro_eval.yaml`
   │      (precache + smoke + eval + 3 launcher .sh already migrated 2026-05-04;
   │      `provider_order` left empty since default routing lands on Parasail
   │      cleanly per probe). See companion file
   │      `todo_openrouter_hle_deepseek-v4-pro.md` for per-Phase Gates.
   │      Evidence · 
   ├ G3 ☐ Gate · 100 rows
   │      Evidence · 
   ├ G4 ☐ Gate · usage delta ≤ 2× estimate ($10.42)
   │      Evidence · 
   ├ G5 ☐ Gate · timed_out ≤ 5%
   │      Evidence · 
   ├ G6 ☐ Soft-Gate · Pass@1
   │      Justification · 
   │      Evidence · 
   └ How  · same launcher

08 ☐ `moonshotai/kimi-k2.6` — HLE-100 ATTS [requires approval; est ~$9.13]
   ├ G1 ☐ Gate · User approval
   │      Evidence · 
   ├ G2 ☐ Gate · YAML; max_output_tokens=8000
   │      Evidence · 
   ├ G3 ☐ Gate · 100 rows
   │      Evidence · 
   ├ G4 ☐ Gate · usage delta ≤ 2× estimate ($18.26)
   │      Evidence · 
   ├ G5 ☐ Gate · timed_out ≤ 5%
   │      Evidence · 
   ├ G6 ☐ Soft-Gate · Pass@1
   │      Justification · 
   │      Evidence · 
   └ How  · same launcher

09 ☐ `google/gemini-3-flash-preview` — HLE-100 ATTS [requires approval; est ~$6.26]
   ├ G1 ☐ Gate · User approval
   │      Evidence · 
   ├ G2 ☐ Gate · YAML; max_output_tokens=8000
   │      Evidence · 
   ├ G3 ☐ Gate · 100 rows
   │      Evidence · 
   ├ G4 ☐ Gate · usage delta ≤ 2× estimate ($12.52)
   │      Evidence · 
   ├ G5 ☐ Gate · timed_out ≤ 5%
   │      Evidence · 
   ├ G6 ☐ Soft-Gate · Pass@1
   │      Justification · 
   │      Evidence · 
   └ How  · same launcher

## Phase 3 — Paper integration [0/1]

10 ☐ Append rows for completed models to tab:backbone-ablation
   ├ G1 ☐ Gate · `Publication/paper/main.tex` line 402-420 (tab:backbone-ablation) gains one row per Phase 1+2 ☑ item with: model id, Pass@1, total $/run, num_explores=8, no_integrate
   │      Evidence · 
   ├ G2 ☐ Gate · `cd Publication/paper && bash compile.sh` produces `build/main.pdf` with no `Overfull \hbox` warnings
   │      Evidence · 
   └ How  · edit main.tex; run compile.sh; visual check of the table in build/main.pdf

## Out of scope (deferred to a separate TODO)

- Other benchmarks (BabyVision / LCB / GPQA-Diamond / RBenchV) on these 7 models
- DISABLE-flag minimum-subset bisection (currently using all 11 flags as a safe default; minimum subset would save env-var noise but is not load-bearing for cost or correctness)
- Other free OpenRouter models not in this lineup (`z-ai/glm-4.5-air:free`, `minimax/minimax-m2.5:free`, etc.)
