# TODO: `google/gemma-4-26b-a4b-it:free` — HLE-100 ATTS via OpenRouter (n_explore=4)

## STATUS: DEFERRED 2026-05-04 — free tier UNUSABLE; revival REQUIRES paid model

User directive 2026-05-04 03:54: "如果要跑的话，不能跑免费的，必须跑那个收费的"
(if revived, must use paid; free is not allowed).

**Root cause**: Google AI Studio (the upstream provider OpenRouter routes
`google/gemma-*:free` to) is currently throttling the entire `:free` family.
Single isolated curl probe to `gemma-4-26b-a4b-it:free` returns 429 with
explicit metadata: `"google/gemma-4-26b-a4b-it:free is temporarily
rate-limited upstream. Please retry shortly, or add your own key to
accumulate your rate limits"`. Same 429 verified across `gemma-4-31b-it:free`,
`gemma-3-27b-it:free`, `gemma-3-12b-it:free` — all Google AI Studio,
all 429. Control: `openai/gpt-oss-20b:free` (provider OpenInference) returns
200 in same shell second. SDK 8-retry path exhausted in Phase 1 smoke; 4/4
explores TIMED OUT with `RateLimitError`. STOP-the-run gate triggered.

**HARD CONSTRAINT for revival**: must change `explore_model` in all three
yamls from `google/gemma-4-26b-a4b-it:free` to `google/gemma-4-26b-a4b-it`
(NO `:free`). Cost: ~$0.80 precache + ~$0.50 eval ≈ $1.30 total at $0.06/M
input + $0.33/M output (verified `/api/v1/models` 2026-05-04). Cache_dir
must change to `..._paid/gold` to avoid mixing tier results. Qwen3.6 self-
hosted judge can be re-launched on demand via
`bash scripts/gpqa/grpo/serve_qwen36_35b_a3b_dp4.sh` (DP=4 on GPU 0+1+2+3).

Phase 1 item 01 (yaml/sh creation) artifacts CREATED but use `:free` model;
re-run for paid path requires editing those files OR creating `..._paid_*.yaml`
siblings. Phase 1 item 02 smoke FAILED (4/4 timed_out due to upstream 429).
Cache stub at `analysis/cache/hle/openrouter_google_gemma-4-26b-a4b-it_free_smoke/`
(4 timed_out result.json) retained for forensic reference; do NOT reuse.

Below is the original plan as-written. Execute only after confirming paid-model
constraint is honored.

---

## What this is

Run ATTS evaluation on the HLE-Verified gold-100 subset using
`google/gemma-4-26b-a4b-it:free` via OpenRouter, replacing a previously-failed
local vLLM-serve attempt of the same model (cache stub at
`analysis/cache/hle/gemma4_26b_a4b_it_thinking_smoke_v2/` is the abandoned
local-vLLM artifact; do NOT reuse). The local vLLM path was blocked by a
gemma-4 jinja chat-template + `skip_special_tokens` stripping bug that
corrupts the thinking-mode response. OpenRouter terminates the call upstream
(Anthropic-Skin route), bypassing local jinja entirely — the bug does not
fire on this code path. Tool-call + tool_choice + reasoning support verified
on `/api/v1/models` 2026-05-04.

This run uses **`num_explores: 4`** instead of the paper's standard 8 (user
directive 2026-05-04: "之前的 8 个样本好像有点太多了，没有必要"). Total
explores = 100 × 4 = 400 (vs 800 at n=8). Wall-clock ~halved; oracle Pass@k
ceiling is bounded at k=4 instead of k=8 (this is a known methodological
trade-off and acceptable for this model — not a paper-main row).

This run uses **`effort: medium`** as the default (user directive 2026-05-04:
"如果你只用 low 的话，可能效果不太好"). Rationale: medium reasoning depth gives
better per-explore quality without burning the 32768 output cap (gpt-oss-20b's
8192 cap was the constraint that forced LOW there; Gemma's 4× larger cap
relaxes the constraint). **Auto-fallback rule**: if Phase 2 G3 measures
medium-effort timed_out rate > 10%, the run halts and falls back to
`effort: low` into a separate cache_dir (`..._free_low_fallback/gold`),
preserving the medium cache for archival, then re-runs Phase 2.

## Output target

- Per-explore cache: `analysis/cache/hle/openrouter_google_gemma-4-26b-a4b-it_free/gold/<qid>/explore_<n>/result.json`
- Per-question results: `analysis/run/hle/openrouter_google_gemma-4-26b-a4b-it_free/run_<TS>/results.jsonl`
- Paper integration: `tab:backbone-ablation` at `Publication/paper/main.tex` line 402-420 (existing GPT-5.2 vs Sonnet rows). Add 1 row for Gemma with footnote noting `n_explore=4` deviation.

## Discipline

Every event below has explicit Gates with checkboxes. An event flips from `☐`
to `✓` only after all Gates pass AND each Gate's `Evidence ·` line is filled
with the actual measurement (qid counts, OpenRouter usage delta, log line
ref, results.jsonl line count — not "looks fine"). No silent skipping. No
narrative-only claims. No marking done before Evidence is recorded. A failed
Gate stops the run; do not advance until the failing Gate is either resolved
or escalated to the user.

## Judge architecture (NEW 2026-05-04: self-hosted Qwen3.6, NOT Haiku)

User directive 2026-05-04: drop Claude Haiku 4.5 judge ($0.64/100q on HLE) in
favor of self-hosted `qwen36-35b-a3b-fp8` via vLLM DP=4 on `:8000` (alias
`qwen36-35b-a3b-fp8`). Cost per eval = $0. Quality trade-off: Qwen3.6-35B-A3B
is a strong reasoning model and is NOT the model under evaluation (Gemma is)
— so this is a stronger-judge-than-model setup, NOT self-judging. No
self-preference bias.

Judge yaml block (used in all three Gemma yamls verbatim):
```yaml
judge:
  name: vllm
  model: qwen36-35b-a3b-fp8
  sampling:
    temperature: 1.0
    top_p: 0.95
    top_k: 64
    enable_thinking: false   # CLAUDE.md non-thinking judge rule
    max_tokens: 4096
```

**Prerequisite for ANY phase below**: `curl http://localhost:8000/v1/models`
returns `qwen36-35b-a3b-fp8`. If not up, launch via
`bash scripts/gpqa/grpo/serve_qwen36_35b_a3b_dp4.sh` (DP=4 on GPU 0+1+2+3).
Serve script alias is hardcoded; do not change.

## Free-tier limits anchor (Gemma-specific)

| Field | Value | Source |
|---|---|---|
| context_length | 262144 (256K) | OpenRouter `/api/v1/models` 2026-05-04 |
| **max_completion_tokens** | **32768** (HARD cap; ignores `max_tokens` request) | OpenRouter `/api/v1/models` 2026-05-04 |
| daily request budget | 1000 req/day per model | OpenRouter free-tier policy |
| 429 handling | SDK auto-retries up to 8× with exponential backoff (`backends/openrouter.py:76`) | gpt-oss-20b LOW empirical: 11/11 429 absorbed in <2s, 0 escalations |
| effort=medium expected reasoning budget | ~8K-16K tokens (rough estimate; under the 32768 cap with ≥16K headroom for tool_call emit) | extrapolated from gpt-oss-20b LOW (~3-6K) and HIGH (28-36K capped at 8192) |

**Calibration anchor (gpt-oss-20b:free LOW baseline measured 2026-05-04 03:09):**
- 547 explores, 91.2% answered, 8.8% timed_out
- Timed_out breakdown: 4.9% `invalid_json_in_tool_args` (LaTeX `\X` unescaped),
  3.3% `no_tool_call`, 0.5% `transient_api_error`
- Use this as the floor for "what counts as an acceptable timed_out rate" — Gemma should match or beat (its 32768 cap leaves more headroom for tool_call emit after reasoning, so `no_tool_call` should drop).

## n_explore=4 implication on G3 of Phase 2

At n=4, oracle Pass@k saturates at k=4. If the paper-main config (n=8) for
gpt-oss-20b LOW saturated at Pass@k=10% (k≥3, see archived progress note),
Gemma at n=4 should be in a similar order. Do not compare `n=4` numbers
against `n=8` paper rows — they are not directly comparable; that is why the
paper row needs a footnote.

## Resume / restart procedure

| Failure point | Recover by | Banner verification |
|---|---|---|
| Precache process killed mid-run | Re-launch with same yaml; cache discipline auto-skips any qid+explore_idx that already has `result.json` | Log line `Tasks: K to run, J already cached` with J > 0 |
| Eval process killed mid-run | Re-launch with `resume: <run_dir>` added to yaml (or use launcher arg) | Log line `Resuming ...: N rollouts already completed` with N > 0 |
| 429 burst > 8 retries | Drop `num_workers` from 4 → 2 in yaml and restart | Log line shows reduced concurrent in-flight |
| `output_tokens=32768` clusters in cache | Switch `effort:` from `low` → `low` (no fix, fundamental cap); reduce `budget_tokens` if explicitly set; escalate to user — likely indicates the model + prompt combo is intrinsically over-budget for free tier | Cache audit shows >5% explores at exactly 32768 |

## Risk register (known failure modes; consolidated from gpt-oss-20b runs + Gemma jinja history)

| # | Failure | Root cause | Defense in this todo |
|---|---|---|---|
| R1 | Free-tier `max_completion_tokens` cap silently clips reasoning + tool_call | OpenRouter ignores caller's `max_tokens` if larger than provider cap; gpt-oss-20b:free has 8192, Gemma has 32768 | Phase 1 G3 Pre-flight smoke checks output_tokens distribution; Phase 2 G6 audits p95 against cap |
| R2 | LaTeX `\X` in tool_call.arguments → `json.JSONDecodeError` → `invalid_json_in_tool_args` timed_out | OpenAI parser does not unescape LaTeX backslashes in JSON strings before our code re-parses | `backends/openrouter.py` already returns `timed_out` cache record (no crash); Phase 2 G3 measures rate; if >5% escalate code fix |
| R3 | Model returns reasoning-only with no tool_call | Free-tier model occasionally ignores forced `tool_choice` | `backends/openrouter.py` returns `timed_out` (no crash); Phase 2 G3 cap rolls into 15% threshold |
| R4 | 429 burst exceeds SDK retries (8) | OpenRouter free-tier per-minute cap shared across all users | `backends/openrouter.py:76` `max_retries=8`; if Phase 2 G5 throughput drops, restart Procedure step 3 (drop num_workers) |
| R5 | OpenRouter returns HTTP 200 with `choices=None` | Free-tier provider edge case | `backends/openrouter.py:207` returns `timed_out` `empty_choices` (incident 2026-05-03 fix) |
| R6 | OpenRouter returns non-JSON HTTP body | Provider HTML error page on overload | `backends/openrouter.py:174` `except (... + json.JSONDecodeError)` (incident 2026-05-03 fix) |
| R7 | Gemma-4 thinking-mode jinja bug | gemma-4 chat template + `skip_special_tokens` corrupts thinking on local vLLM | OpenRouter routes via Anthropic-Skin → bypasses local vLLM → R7 NOT in scope. Confirmed by `/api/v1/models` `reasoning=True` returning structured reasoning_details. |
| R8 | Cache pollution from previous failed local-vLLM smoke | `analysis/cache/hle/gemma4_26b_a4b_it_thinking_smoke_v2/` exists from a different (failed) attempt | New cache_dir uses `openrouter_` prefix; never share with local-vLLM cache |

## Co-monitor — log paths for parallel watching

Absolute paths (working dir: `/data3/peijia/dr-claw/Explain/Experiment/core_code`):

| Phase | Run log | Cache count |
|---|---|---|
| Phase 1 smoke | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_gemma-4-26b-a4b-it_free_smoke_<DATE>.log` | `find /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/openrouter_google_gemma-4-26b-a4b-it_free_smoke/gold -name 'result.json' \| wc -l` |
| Phase 2 precache | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_gemma-4-26b-a4b-it_free_precache_<DATE>.log` | `find /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/openrouter_google_gemma-4-26b-a4b-it_free/gold -name 'result.json' \| wc -l` |
| Phase 3 eval | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_gemma-4-26b-a4b-it_free_eval_<DATE>.log` | `wc -l /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/openrouter_google_gemma-4-26b-a4b-it_free/run_*/results.jsonl` |

---

## Phase 1 — Setup [0 done / 2]

01 ☐ Create yamls + launcher scripts
   ├ G1 ☐ Gate · `scripts/hle/openrouter/hle_gemma-4-26b-a4b-it_free_precache.yaml` exists with `explore_model: google/gemma-4-26b-a4b-it:free`, `effort: medium` (NOT low — see header rationale; fallback path tracked in G3 of item 03), `num_explores: 4`, `num: 100`, `num_workers: 4`, `cache_dir: ../analysis/cache/hle/openrouter_google_gemma-4-26b-a4b-it_free/gold`, `seed: 42`, `explore_timeout: 600.0`, judge `claude-haiku-4-5-20251001`, all non-default values commented per config-override discipline
   │      Evidence · 
   ├ G2 ☐ Gate · `scripts/hle/openrouter/hle_gemma-4-26b-a4b-it_free_eval.yaml` exists; identical except `method: tts-agent`, `no_integrate: true`, points to same `cache_dir`, `log_dir: ../analysis/run/hle/openrouter_google_gemma-4-26b-a4b-it_free`
   │      Evidence · 
   ├ G3 ☐ Gate · `scripts/hle/openrouter/hle_gemma-4-26b-a4b-it_free_smoke.yaml` exists with `num: 2`, `num_explores: 2`, `effort: medium` (matches production), separate `cache_dir: ../analysis/cache/hle/openrouter_google_gemma-4-26b-a4b-it_free_smoke/gold` (do NOT pollute production cache)
   │      Evidence · 
   ├ G4 ☐ Gate · launcher `scripts/hle/openrouter/run_gemma-4-26b-a4b-it_free_precache.sh` and `_eval.sh` exist; both use `PYTHONUNBUFFERED=1`, `conda run -n explain --no-capture-output`, inline `OPENROUTER_API_KEY=...`, no shell args, write log under `tmp/`
   │      Evidence · 
   └ How · Mirror `scripts/hle/openrouter/hle_gpt-oss-20b_free_low_precache.yaml` (and its `eval` sibling); diff: `explore_model`, `num_explores: 4`, `cache_dir` slug. Use absolute paths in launchers.

02 ☐ Pre-flight smoke (num=2, n_explore=2 → 4 explores total)
   ├ G1 ☐ Gate · Smoke completes 4/4 explores within 5 min wall-clock without uncaught crash (`grep -c Traceback <log> == 0` and `find <smoke_cache> -name result.json | wc -l == 4`)
   │      Evidence · 
   ├ G2 ☐ Gate · ≥3/4 result.json have non-null `answer` field (i.e. ≤1/4 timed_out — proves end-to-end ATTS works; if 4/4 timed_out, fundamental tool_call routing problem with this model)
   │      Evidence · 
   ├ G3 ☐ Gate · `output_tokens` distribution across 4 explores: at least 1 sample is < 32768 (proves the 32768 cap is NOT systematically hit at effort=low — if all 4 are exactly 32768, calibration matches the gpt-oss-20b:free disaster pattern and effort=low is also infeasible on Gemma)
   │      Evidence · 
   ├ G4 ☐ Gate · `finish_reason='length'` rate < 50% across the 4 explores (calibration: gpt-oss-20b:free HIGH on free was 82.6% — anything in that range means stop)
   │      Evidence · 
   ├ G5 ☐ Gate · trajectory text in ≥1 result.json contains BOTH a non-empty reasoning excerpt AND a parsed `StructuredOutput` JSON block (proves the Anthropic-Skin → tool-call path is working end-to-end, not just stub)
   │      Evidence · 
   └ How · `bash scripts/hle/openrouter/run_gemma-4-26b-a4b-it_free_smoke.sh` ; tail log; after exit, audit smoke cache with one-shot Python (`for f in result.json files: print finish_reason, output_tokens, reason if timed_out`)

---

## Phase 2 — Precache [0 done / 1]

03 ☐ Full precache 400/400 explores (100 qids × 4 explores each)
   ├ G1 ☐ Gate · `find <cache_dir> -name "result.json" | wc -l` returns exactly 400
   │      Evidence · 
   ├ G2 ☐ Gate · `find <cache_dir> -mindepth 1 -maxdepth 1 -type d | wc -l` returns exactly 100; every qid dir contains exactly 4 sub-dirs (`for d in $cache/*/; do n=$(ls $d | wc -l); [ $n -eq 4 ] || echo "$d $n"; done` returns empty)
   │      Evidence · 
   ├ G3 ☐ Gate · cumulative `timed_out` rate ≤ **10%** at `effort: medium` (calibration: gpt-oss-20b:free LOW achieved 8.8%; medium adds reasoning depth which slightly raises `no_tool_call` risk but Gemma's 32768 cap leaves >16K headroom for tool_call emit; 10% is the user-set hard ceiling for medium quality)
   │      On-fail · IF measured > 10%: STOP. Do NOT advance to Phase 3. Write fallback yaml `scripts/hle/openrouter/hle_gemma-4-26b-a4b-it_free_low_fallback_precache.yaml` (same as production but `effort: low` and `cache_dir: ../analysis/cache/hle/openrouter_google_gemma-4-26b-a4b-it_free_low_fallback/gold`); preserve the medium cache untouched for archival comparison; re-run Phase 2 against the fallback yaml; replace this Gate's Evidence with both medium AND low timed_out rates side-by-side; only advance to Phase 3 once low achieves ≤ 10%. Rationale: 10% is the user-set quality bar; falling back to low trades reasoning depth for stability, matching the gpt-oss-20b regime.
   │      Evidence · 
   ├ G4 ☐ Gate · zero `Traceback` in run log (`grep -c Traceback <log> == 0`); the 5 known soft-failure modes (no_tool_call / invalid_json / transient / empty_choices / json_decode_body) all return cached `timed_out` records, not crashes
   │      Evidence · 
   ├ G5 ☐ Gate · throughput ≥ 0.5 explores/min over a rolling 10-min window in steady state (i.e. excluding warm-up first 5 min and burst recovery from 429); calibration: gpt-oss-20b:free LOW achieved ~2.25/min at num_workers=4. Gemma should be ≥0.5/min — if <0.5/min for ≥30 consecutive min, OpenRouter routing for Gemma free-tier is degraded; restart with halved num_workers
   │      Evidence · 
   ├ G6 ☐ Gate · `output_tokens` p95 < 32768 (audit cache: extract `usage.output_tokens` from all 400 result.json; sort; check p95). Proves the 32768 cap is NOT the bottleneck. If p95 == 32768, ≥5% of explores hit the wall — flag as Gemma-specific risk before eval
   │      Evidence · 
   ├ G7 ☐ Soft-Gate · sample 5 random non-`timed_out` result.json (use `find ... | shuf -n 5`); for each verify:
   │      (a) trajectory text non-empty AND contains both reasoning_details and StructuredOutput JSON
   │      (b) `answer` field populated and matches the schema (string for free-form, single letter A-E for multiple-choice)
   │      (c) `confidence` ∈ [0, 1]
   │      (d) reasoning text shows actual problem-solving (not "I don't know" / not garbled / not English-only when problem is multilingual)
   │      (e) no LaTeX repair regression (if problem contains LaTeX, the trajectory should reference the math symbols correctly, not as escaped strings like `"\\frac"` mistakenly emitted)
   │      Justification (required) · Write 1-2 sentences per sub-check (a-e) citing the qid, the result.json path, and the concrete evidence (e.g. "qid `66e8...` line `<n>` of trajectory contains `<excerpt>`"). Do NOT say "looks fine" or "verified manually" — both are auto-rejected.
   │      Evidence · 
   └ How · `bash scripts/hle/openrouter/run_gemma-4-26b-a4b-it_free_precache.sh` ; capture PID and log path immediately and surface to user; periodic check via `find <cache_dir> -name "result.json" | wc -l`. ETA ~3-4 hours at num_workers=4 free-tier rate.

---

## Phase 3 — Eval [0 done / 1]

04 ☐ Eval n=100 against precache → Pass@1 + oracle Pass@4
   ├ G1 ☐ Gate · `wc -l results.jsonl` == 100
   │      Evidence · 
   ├ G2 ☐ Gate · `progress.json` (or final log line) reports `total_correct`, `total_cost`, `judge_cost` numerically (not null)
   │      Evidence · 
   ├ G3 ☐ Gate · Pass@1 ∈ [3%, 50%] sanity envelope. Lower bound calibrated to gpt-oss-20b:free LOW (5%) minus 2pp safety margin; upper bound is Gemma-4-26b's published HLE-Verified leaderboard ballpark (no exact published number, but 26B-class MoE models typically <50% on HLE without RL post-train). If Pass@1 < 3% the pipeline likely has a grading collapse (HLE judge_model regression, like the 2026-04-28 incident). If Pass@1 > 50% suspect data leak. STOP and escalate either way.
   │      Evidence · 
   ├ G4 ☐ Gate · judge cost (Haiku 4.5) < $1.50 (calibration: gpt-oss-20b smoke20 cost $0.128 for 20 questions ≈ $0.64/100q; Gemma may be longer responses → Haiku reads more → 2× headroom)
   │      Evidence · 
   ├ G5 ☐ Gate · zero `Traceback` in eval log
   │      Evidence · 
   └ How · `bash scripts/hle/openrouter/run_gemma-4-26b-a4b-it_free_eval.sh` ; cache discipline ensures explores are not re-generated (`cache_only` mode); only Haiku judge calls cost money.

---

## Phase 4 — Paper integration [0 done / 1]

05 ☐ Add 1 row to `tab:backbone-ablation` in `Publication/paper/main.tex`
   ├ G1 ☐ Gate · row inserted between line 402 and line 420 with: model name `gemma-4-26b-a4b-it`, n_explore=4 (footnoted), Pass@1, oracle Pass@4 numbers from Phase 3 G2
   │      Evidence · 
   ├ G2 ☐ Gate · footnote attached to the row reading exactly "n_explore=4 vs paper main 8; OpenRouter free tier; max_completion_tokens=32768"
   │      Evidence · 
   ├ G3 ☐ Gate · `cd ../../Publication/paper && bash compile.sh` exits with status 0 (no LaTeX errors)
   │      Evidence · 
   ├ G4 ☐ Gate · `Publication/paper/build/main.pdf` mtime is newer than the modification time of `main.tex` from G1; PDF visually inspected — table renders without `Overfull \hbox` warning beyond existing baseline; no margin overflow
   │      Evidence · 
   └ How · edit main.tex, add row + footnote; recompile; visually verify rendered PDF page containing tab:backbone-ablation.
