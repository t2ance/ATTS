# TODO: get `openai/gpt-oss-20b:free` HLE-100 ATTS running with `effort=low`

Single objective: produce a usable HLE-Verified num=100 ATTS run with `effort=low` on OpenRouter free tier. Effort=high is DEFERRED (see bottom).

## Decision log (read first)

- **Verdict on `effort=low`**: GO. Cumulative cache audit on 2026-05-04 03:09 over 547 result.json: timed_out = 48 (8.8%), answered = 499 (91.2%). Only 4.9pp of timed_out is `invalid_json_in_tool_args` (LaTeX backslash) which is code-fixable. This is publication-grade.
- **Verdict on `effort=high` (free tier)**: ABANDONED. OpenRouter `/api/v1/models` ground truth: `openai/gpt-oss-20b:free` has hard `max_completion_tokens=8192` (paid `openai/gpt-oss-20b` has 131072). With effort=high, reasoning alone exceeds 8192 → finish_reason='length' on 82.6% of explores. Physically infeasible on free tier. Killed 2026-05-04 02:57 after 213/800.
- **HIGH cache retained**: `analysis/cache/hle/openrouter_gpt-oss-20b_free_high/gold/` (212 result.json, mostly timed_out). Do NOT discard — if HIGH is ever revived (only on PAID `openai/gpt-oss-20b`, NEVER `:free`), the few `answer=...` entries stay valid via cache discipline.

## Current cache state (verified 2026-05-04 03:09)

- Cache root: `/data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/openrouter_gpt-oss-20b_free_low/gold/`
- **547 / 800 explores cached** (68.4%)
- **53 / 100 qids** have at least one result.json
- Remaining work: 800 - 547 = **253 explores** to run; 100 - 53 = **47 qids** uncovered

## Items

1. [x] **Resume LOW precache to 800/800** ✓ COMPLETE 2026-05-04 04:04
   - Evidence: 100 canonical qids × 8 explores = 800/800 in cache (verified 2026-05-04 04:05). One non-canonical qid (101st) carries 135 result.json from prior larger-num runs — does not affect eval. Final timed_out rate across all 935 cached: 103/935 = 11.0%, comfortably under the 15% relaxed gate. Run completed 04:04 ("Done. 388 cached, 412 skipped (already existed)" log line).
   - PID 2936707 (precache_explores.py, started 03:17:10) terminated cleanly.
   - Log: `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_gpt-oss-20b_free_low_precache_20260504_031710.log`

2. [ ] **Eval num=100 against the LOW cache → Pass@1 + oracle Pass@k**
   - Gate: `results.jsonl` has 100 lines AND `progress.json` reports `total_correct`. Numbers logged here.
   - How: `scripts/hle/openrouter/hle_gpt-oss-20b_free_low_eval.yaml` (already exists). Smoke yaml `_low_eval_smoke20.yaml` available for a 20-question sanity pass first if desired.

3. [ ] **Integrate into paper `tab:backbone-ablation` (main.tex line 402-420)**
   - Gate: `Publication/paper/build/main.pdf` rebuilt via `compile.sh`; row reads "gpt-oss-20b:free (effort=low) HLE: <Pass@1>" with footnote noting 8.8% timed_out + 8192-token free-tier cap as known limitations.
   - How: edit `Publication/paper/main.tex`, run `compile.sh`, verify pdf renders.

## Resume / restart cookbook (LOW)

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
  --config scripts/hle/openrouter/hle_gpt-oss-20b_free_low_precache.yaml \
  > tmp/openrouter_hle_gpt-oss-20b_free_low_precache_<DATE>.log 2>&1 &
```

Cache count check:
```bash
find /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/openrouter_gpt-oss-20b_free_low/gold/ -name "result.json" | wc -l
```

## Backend hardening already in place (DO NOT regress)

`backends/openrouter.py` — five `timed_out` cache-record paths (consolidated 2026-05-03/04):
- `no_tool_call` (model emits reasoning only, no tool_call)
- `invalid_json_in_tool_args` (LaTeX `\X` unescaped in tool args) ← dominant LOW failure mode (4.9% of 547)
- `transient_api_error` (RateLimitError / APIConnectionError / APITimeoutError / APIError, max_retries=8)
- `JSONDecodeError on response body`
- `empty_choices` (HTTP 200 with `choices=None`/`[]`)

Tests: `tests/test_openrouter_backend.py` — 15/15 passing.

---

## DEFERRED: effort=high path

Status: PAUSED indefinitely. Re-activate ONLY if the LOW path produces interesting-but-insufficient results AND budget is approved for paid tier.

**HARD CONSTRAINT (do not violate)**: if revived, MUST use `openai/gpt-oss-20b` (paid, $0.14/M output) — NEVER `openai/gpt-oss-20b:free`. Reason: free tier `max_completion_tokens=8192` is a hard provider cap that cannot accommodate effort=high reasoning. Verified 2026-05-04 via OpenRouter `/api/v1/models` and 213 cached `output_tokens=8192` data points.

Estimated paid cost for full 800 explores at effort=high: ~$3.40 (24M output tokens × $0.14/M).

If revived, deferred steps are:
- D1. Create `scripts/hle/openrouter/hle_gpt-oss-20b_PAID_high_precache.yaml` with `explore_model: openai/gpt-oss-20b` (no `:free`) and a fresh `cache_dir = ../analysis/cache/hle/openrouter_gpt-oss-20b_paid_high/gold`. Do NOT reuse the `_free_high/` cache — different tier, billing, and provider routing; treat as a separate experiment.
- D2. Full 800-explore precache, expect <5% timed_out (no length cap to fight).
- D3. Eval n=100 against paid HIGH cache.
- D4. Optionally add as a second row in `tab:backbone-ablation` for an effort=low vs effort=high contrast.
