# TODO: GPT-5.4 ATTS run on HLE + GPQA — paper main `tab:backbone-ablation`

## Status: DEFERRED (paused 2026-05-03 by user)

Paused mid-Phase-2 (HLE precache) at user request. Resume by relaunching `bash /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/hle/gpt5.4_high/supervisor.sh` (precache_explores.py auto-skips already-cached explores).

**Final HLE precache state at pause:** cache=356/800 (44.5%), 0 timeout placeholders, 0 zombie processes. Codex rate window has been cycling between open and throttled in ~2-3h windows; supervisor v5 (setsid + group kill + per-Round stall counter reset) handled it autonomously over 6h35min, no manual intervention needed during the window.

**Outstanding work when resumed:** 444 HLE explores remaining + HLE eval + GPQA precache + GPQA eval + paper integration (Phases 2-4).

## What this is

Add 2 rows to the paper's backbone ablation table (`tab:backbone-ablation` at `Publication/paper/main.tex` line 417-418 area), `GPT-5.4 & high & HLE` and `GPT-5.4 & high & GPQA`, mirroring the existing GPT-5.2 high-effort methodology exactly. Methodology is the same two-step that produced the existing GPT-5.2 high rows:

- **Step A (precache):** `precache_explores.py` calls codex backend with `gpt-5.4`, `effort=low`. Generates 8 explore candidates per question, writes `result.json` per explore to a NEW cache dir. This step pays the explore tokens at the cheaper low-effort price (mirroring the same trick used by `cache/hle/gpt5.2_low/gold/` which feeds both HLE-low and HLE-high eval rows).
- **Step B (eval):** `eval.py` runs `tts-agent` method with `cache_only=true`, `effort=high`. Orchestrator + integrator call codex at high effort; explorer is read from the cache built in Step A — zero new explore API calls during eval.

Output: 2 paper rows directly comparable to the existing 2 GPT-5.2 high rows. Single-Variant; one TODO file.

Why GPT-5.4 over 5.2: tests whether the +2.0 / +15.0 ATTS gain pattern reproduces (or strengthens) on the newer reasoning model at the same effort budget; user directive 2026-05-03.

## Output target

2 new rows inserted into `Publication/paper/main.tex` line 419 (immediately below the existing `GPT-5.2 & high & GPQA` row, before the `\midrule` separating GPT block from Qwen block):

```latex
GPT-5.4 & high & HLE  & <pass1> & <acc> & <gain> & <$/q> \\
GPT-5.4 & high & GPQA & <pass1> & <acc> & <gain> & <$/q> \\
```

Compiled `Publication/paper/build/main.pdf` shows Table 3 with 3 rows in the GPT block (5.2-low-HLE, 5.2-high-HLE, 5.2-high-GPQA — already there) followed by 2 new GPT-5.4 rows.

## Discipline

Every event has Gates with checkboxes. An event flips `☐` → `✓` only after **all** its Gates pass AND each Gate's `Evidence ·` line is filled with the actual measurement (e.g. "27/800 = 3.4% timed_out", not "looks fine"). Soft-Gates require Justification with concrete qid + line-number evidence. No silent skipping, no marking done before evidence is recorded. STOP-the-run gates (acc-must-beat-5.2, timeout-rate, cost-budget, auth) are non-negotiable — failing them means do NOT write to the paper, debug first.

## Codex backend anchors (5.2 history-derived)

- **Auth:** codex backend reads `~/.codex/auth.json` (`backends/codex.py:28`). Token last refreshed for the GPT-5.2 runs on 2026-03-26/27. Pre-flight verification mandatory before any paid call.
- **Pricing (`backends/codex.py:32`):** GPT-5.4 = $2.50/M input, $15.00/M output. GPT-5.2 = $1.75/M input, $14.00/M output. Blended ratio for reasoning workloads ≈ ×1.10–1.15.
- **429 rate-limit history:** Across HLE-low (`run_20260326_050015`), HLE-high (`run_20260326_171818`), and GPQA-high (`run_20260327_173451`), `grep -c 429 *.log = 0`. Zero 429s in main eval at the parallelism used (HLE workers=4-8, GPQA workers=1). LCB precache at workers=4 hit 429 hard (multiple `precache_*w*.log` files testify, all aborted on `httpx.HTTPStatusError 429 Too Many Requests`). Inference: HLE & GPQA at the worker counts below are 429-safe historically, but 5.4 may have tighter limits — smoke test before any large-scale launch.
- **Methodology mirror:** `Experiment/analysis/run/hle/gpt5.2_high/run_20260326_171818/run_config.json` documents the exact 5.2-high config: `cache_dir=../analysis/cache/hle/gpt5.2_low/gold` (shared with low), `cache_only=true`, `num_workers=4`, `budget_tokens=32000`, `effort=high`, `judge_model=claude-haiku-4-5-20251001`. GPQA-high analog: `num_workers=1`, `judge_model=null` (GPQA uses string-match, no judge).

## Result quality anchor (USER REQUIREMENT — STOP-the-run)

GPT-5.4 result must beat GPT-5.2 result on each benchmark. Hard threshold from the existing rows in `tab:backbone-ablation` (lines 417-418):

| Benchmark | GPT-5.2 high Acc | 5.4 must achieve | Stderr (binom @ N) | 1-σ band |
|---|---:|---:|---:|---|
| HLE | 57.00% (line 417) | **> 57.00%** (strict greater) | √(0.57·0.43/100)=4.95pp | 52.0 – 62.0 |
| GPQA | 71.07% (line 418) | **> 71.07%** (strict greater) | √(0.71·0.29/198)=3.23pp | 67.8 – 74.3 |

Note: 1pp difference is well inside 1σ noise on both benchmarks. "Strict greater" means we require nominal Acc > 5.2 baseline as a sanity floor; if Acc is between 5.2 baseline and (5.2 baseline + 1σ), flag as "marginal but compliant" in Soft-Gate. If Acc ≤ 5.2 baseline → STOP, do NOT write to paper, debug.

## Timeout anchor (USER REQUIREMENT — STOP-the-run)

GPT-5.4 inference under `effort=high` may produce long reasoning traces. Hard cap from `backends/codex.py` and `methods/specs.py` is 1200s per call (default `timeout`). Each question has 1 orchestrator turn × multiple explore turns × 1 integrate turn → multiple timeout opportunities per row.

- **Per-row gate:** results.jsonl `timed_out=true` rate ≤ **10%** of total rows. Above 10% → STOP, indicates either (a) 5.4 is genuinely too slow at high effort and we need to bump `timeout`, or (b) auth / rate-limit is silently degrading throughput. Either case: debug before paper.
- **Per-explore gate:** in cache, `result.json` rows with `timed_out=true` ≤ 10%. Same threshold for the precache phase.

## Cost budget anchor (Scheme B — high-effort precache, recalibrated from smoke)

**Methodology change 2026-05-03**: switched from scheme A (5.2-style: low explores + high orch) to scheme B (full high effort: high explores + high orch). User directive: prioritize absolute acc over methodology match with 5.2.

**Smoke measurement (2026-05-03 06:13–06:49 UTC, 24 high-effort calls)**:
- mean cost/call: **$0.1343**
- median: $0.1033, max: $0.3034 (single-call worst case)
- mean wall/call: 176s, max: 536s (worst case), 0/24 timed_out
- 0 × 429, 0 × Traceback

True wallet (NOT sum of `cost_usd` in results.jsonl — that double-counts cached explores):

| Step | Smoke-derived estimate | Cap (×1.15 over estimate) |
|---|---:|---:|
| HLE precache (100q × 8 = 800 calls @ $0.134) | $107 | **≤ $123** |
| HLE eval (orch+integrate+judge, cache_only) | TBD post-precache | **≤ $15** (~ $107×0.10 + Haiku judge $1) |
| GPQA precache (198q × 8 = 1584 calls @ $0.134) | $212 | **≤ $244** |
| GPQA eval (orch+integrate, cache_only, no judge) | TBD post-precache | **≤ $25** |
| **Subtotal HLE only** | $122 | **≤ $138** |
| **Total HLE + GPQA** | $339 | **≤ $407** |

Why ×1.15 and not tighter: smoke n=24 is a small sample for cost variance estimation; max single-call cost was 2.3× the mean, so cap needs headroom for rare expensive questions. If precache spend exceeds cap, one of: (a) full 100-q distribution has heavier tail than smoke 3-q sample (just a noise issue, raise cap or rerun smoke at n=10), (b) pricing changed mid-run, (c) silent bug. STOP and check.

**Sequencing decision (user 2026-05-03): HLE precache → HLE eval → STOP and decide whether GPQA budget is available.** Don't pre-commit GPQA cost.

## Resume / restart procedure

| Failure point | Recover by | Banner verification |
|---|---|---|
| Mid precache (e.g. 400/800 done) | Re-launch same `precache_explores.py` command. Cache hits via existing `result.json` files; `precache_explores.py` reports `Tasks: K to run, J already cached` with J>0. | Banner line `Tasks: K to run, J already cached` with J>0 must appear; J should equal the count from prior run. |
| Mid eval (e.g. Q50/100) | Add `resume: <RUN_DIR>` to the eval YAML pointing to the dying run's `analysis/run/<bench>/gpt5.4_*/run_<timestamp>/`. Banner: both lines `Resuming ...: N rollouts already completed` (N>0) AND `Questions to run: M (N already completed, M+N total)` (N>0) must appear. | both lines mandatory; if `0 already completed` → resume path broken, STOP. |
| Codex 429 hits exhaustion (5x backoff fails) | Wait 5–15 min for window reset; relaunch with same cache_dir + resume; reduce `num_workers` by half if recurs. | restart banner shows partial cache from prior attempt; sanity-check by `find cache/<bench>/gpt5.4*/ -name 'result.json' \| wc -l` matches prior progress. |
| Auth expired mid-run (401) | Refresh `~/.codex/auth.json` (codex CLI login flow); resume from cache. | curl `https://chatgpt.com/backend-api/codex/responses` health check returns 200 not 401. |
| Pick which `RUN_DIR` to resume from | Pick by **largest** `wc -l results.jsonl`, NOT mtime. | n/a |

## Risk register (known failure modes, derived from 5.2 history + LCB-precache incident)

| # | Failure | Root cause | Defense |
|---|---|---|---|
| R1 | Codex 429 cascading retries → `httpx.HTTPStatusError` raises out of `precache_explores.py` (LCB-precache 2026-03-27 history, 6 abort logs in `analysis/run/lcb/gpt5.2_no_integrate_high/`) | OpenAI per-token-per-minute limit; HLE/GPQA never hit it at workers=4-8 because each call is shorter than LCB's code generations | Phase 1 item 01 (5-question smoke at workers=2) confirms 5.4 at this latency is 429-safe before scaling up; Phase 2/3 precache uses workers=4 (HLE/GPQA proven safe in 5.2) |
| R2 | `~/.codex/auth.json` token expired (last refresh 2026-03-27, 5+ weeks ago) → 401 on first call | OAuth token TTL | Phase 1 item 01 G1 (curl auth health check before any launch) |
| R3 | `gpt-5.4` model name not in codex catalog → 404 on first call | model deprecation / API drift | Phase 1 item 01 G2 (smoke does a real gpt-5.4 call, not just /v1/models) |
| R4 | `cache_dir` typo or accidental reuse of GPT-5.2 cache → contaminated explores | path drift | Phase 1 item 02 G2/G3 (yaml cache_dir is FRESH `cache/<bench>/gpt5.4*/...`, never `gpt5.2`); Phase 2 item 04 G1 banner shows `Tasks: 800 to run, 0 already cached` on first launch |
| R5 | `cache_only=true` flag silently bypassed during eval → unwanted explorer API calls (RB1 from Gemma TODO) | yaml typo / spec drift | Phase 2 item 05 G7 + Phase 3 item 07 G7: `cost_by_component.explorer × pricing_inverse` should match precache cost (since cache_only=true means eval reads precache cost faithfully); large divergence = silently bypassed |
| R6 | HLE judge silently swapped (e.g. set to None like the 2026-04-11 incident, underestimating Acc by ~8pp per project memory) | class-attribute drift | Phase 2 item 05 G6 (verify `run_config.json` field `judge_model == "claude-haiku-4-5-20251001"`) |
| R7 | GPQA grader regex extraction misses A-E in 5.4's verbose final answer (`b06751a` / `d35f925` history) | regex too strict | Phase 3 item 07 G3 (extracted-letter rate ≥ 90%; if lower, regex broken on 5.4 output style) |
| R8 | Pricing table not updated for gpt-5.4 → cost numbers in results.jsonl are zero | `PRICING_PER_1M` mapping fallback `(0.0, 0.0)` at `backends/codex.py:49` | Phase 1 item 01 G3 (smoke results.jsonl `cost_usd > 0`) — already verified `(2.50, 15.00)` is in the table at line 32 |
| R9 | 5.4 takes drastically longer per call → 10%+ timeout at default 1200s | high-effort reasoning length | Phase 2 item 05 G2 + Phase 3 item 07 G2 (timeout rate ≤ 10%); on-fail: bump `timeout` in YAML backend block to 1800 and rerun the timed-out subset |
| R10 | 5.4 Acc not better than 5.2 → invalidates the contribution of this run | actual model regression OR pipeline bug | Phase 2 item 05 G4 + Phase 3 item 07 G4 (HARD STOP gate: Acc > 5.2 baseline) |

## Co-monitor — log paths for parallel watching (absolute, from working dir `/data3/peijia/dr-claw/Explain/Experiment/core_code/`)

| Phase item | Run log (stdout/stderr) |
|---|---|
| 01 smoke (HLE, num=5, low effort) | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/smoke_hle_gpt5.4.log` |
| 04 HLE precache (workers=4, low effort, 100Q × 8 = 800 explores) | `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/gpt5.4_high/precache.log` |
| 05 HLE eval (workers=4, high effort, cache_only) | `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/gpt5.4_high/delegated.log` |
| 06 GPQA precache (workers=4, low effort, 198Q × 8 = 1584 explores) | `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/gpqa/gpt5.4_no_integrate_high/precache.log` |
| 07 GPQA eval (workers=1, high effort, cache_only, no_integrate) | `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/gpqa/gpt5.4_no_integrate_high/delegated.log` |

User can `tail -f <path>` for any of these.

## Phase 1 — Pre-flight & config [0/3]

01 ✓ Codex auth + 3-question HLE smoke (workers=2, gpt-5.4, effort=**high** — scheme B)
   ├ G1 ✓ Gate · `~/.codex/auth.json` exists; smoke calls returned HTTP 200 (not 401/404)
   │      Evidence · 24/24 codex requests returned 200; auth file mtime 2026-04-24 (5w old, still valid)
   ├ G2 ✓ Gate · 3 questions × 8 explores = 24 result.json files written to `analysis/cache/hle/gpt5.4_smoke/gold/`
   │      Evidence · `find ../analysis/cache/hle/gpt5.4_smoke/gold -name result.json | wc -l` = 24; 3 qid subdirs × 8 explores each
   ├ G3 ✓ Gate · sum cost_usd > 0 (pricing table populated)
   │      Evidence · sum=$3.22 across 24 calls; mean=$0.1343, median=$0.1033, max=$0.3034 (RB8 — gpt-5.4 entry @ codex.py:32 active)
   ├ G4 ✓ Gate · `timed_out=true` rate ≤ 10%
   │      Evidence · 0/24 = 0%; max wall/call = 536s (45% of default 1200s timeout, comfortable headroom)
   ├ G5 ✓ Gate · 429 count ≤ 5
   │      Evidence · `grep -c 429 tmp/smoke_hle_gpt5.4_high.log` = 0; zero retry-recovered too — workers=2 at high-effort is well below rate limit
   ├ G6 ⚠ Gate · smoke spend ≤ $2.00 — **EXCEEDED but acknowledged: cap was scheme-A based ($0.04/call low-effort estimate); scheme B reality is $0.13/call ⇒ $3.22**
   │      Evidence · actual $3.22 vs $2.00 cap; cap re-derived for scheme B production runs as ≤ $123 (HLE) / ≤ $244 (GPQA) per anchor table above. Not blocking.
   └ How (executed) · `tmp/smoke_hle_gpt5.4_high.yaml` (effort=high, num=3, workers=2); ran 06:13–06:49 UTC, ~36 min wall

02 ☐ Create 4 production YAML files for HLE + GPQA
   ├ G1 ☐ Gate · `scripts/hle/gpt5.4_high/hle_gpt5.4_precache.yaml` parses via project's eval-config schema; key fields: `benchmark.name=hle`, `subset=gold`, `text_only=true`, `backend=codex`, `explore_model=gpt-5.4`, `effort=low`, `cache_dir=../analysis/cache/hle/gpt5.4_high/gold`, `num_explores=8`, `num=200`, `num_workers=4`, `seed=42`
   │      Evidence · 
   ├ G2 ☐ Gate · `scripts/hle/gpt5.4_high/hle_gpt5.4_high_delegated.yaml` parses; key fields: `method.name=tts-agent`, `backend.name=codex`, `effort=high`, `explore_model=gpt-5.4`, `orchestrator_model=gpt-5.4`, `integrate_model=gpt-5.4`, `cache_dir=../analysis/cache/hle/gpt5.4_high/gold`, `cache_only=true`, `num_explores=8`, `num=200`, `num_workers=4`, `seed=42`, `judge.model=claude-haiku-4-5-20251001`, `log_dir=../analysis/run/hle/gpt5.4_high`
   │      Evidence · 
   ├ G3 ☐ Gate · `scripts/gpqa/gpt5.4/gpqa_gpt5.4_precache.yaml` parses; key fields: `benchmark.name=gpqa`, `backend=codex`, `explore_model=gpt-5.4`, `effort=low`, `cache_dir=../analysis/cache/gpqa/gpt5.4`, `num_explores=8`, `num_workers=4`, `seed=42`
   │      Evidence · 
   ├ G4 ☐ Gate · `scripts/gpqa/gpt5.4/gpqa_gpt5.4_no_integrate_high.yaml` parses; key fields: `method.name=tts-agent`, `backend.name=codex`, `effort=high`, all 3 model fields = `gpt-5.4`, `no_integrate=true`, `cache_dir=../analysis/cache/gpqa/gpt5.4`, `cache_only=true`, `num_explores=8`, `num_workers=1`, `seed=42`, `log_dir=../analysis/run/gpqa/gpt5.4_no_integrate_high` (NO judge block — GPQA uses string-match)
   │      Evidence · 
   ├ G5 ☐ Gate · all 4 yamls have inline comments on overrides that drift from the analogous gpt-5.2 yaml (per project memory `comment_on_config_overrides`); specifically the cache_dir change and the model id swap each carry a `# ...` reason line
   │      Evidence · 
   └ How  · clone the 4 corresponding gpt-5.2 yamls under `scripts/hle/gpt5.2_low/` and `scripts/gpqa/gpt5.2/`, swap `gpt-5.2`→`gpt-5.4`, swap cache_dir, change log_dir suffix to `_high` (HLE) and keep `_no_integrate_high` (GPQA)

03 ☐ Create 4 launcher .sh files (precache + eval × 2 benchmarks)
   ├ G1 ☐ Gate · `scripts/hle/gpt5.4_high/run_precache.sh` is executable (mode 755), uses absolute path `cd /data3/peijia/dr-claw/Explain/Experiment/core_code`, invokes `conda run -n explain --no-capture-output python precache_explores.py --config scripts/hle/gpt5.4_high/hle_gpt5.4_precache.yaml > ../analysis/run/hle/gpt5.4_high/precache.log 2>&1 &`
   │      Evidence · 
   ├ G2 ☐ Gate · `scripts/hle/gpt5.4_high/run_delegated.sh` similar, invoking `eval.py --config ...delegated.yaml > ../analysis/run/hle/gpt5.4_high/delegated.log`; sets `unset CLAUDECODE` (mirror gpt-5.2 pattern)
   │      Evidence · 
   ├ G3 ☐ Gate · `scripts/gpqa/gpt5.4/run_precache.sh` and `scripts/gpqa/gpt5.4/run_no_integrate_high.sh` ditto, paths point to gpqa cache + run dirs
   │      Evidence · 
   ├ G4 ☐ Gate · all 4 sh scripts have `mkdir -p` for the target log dir before launching (eval.py needs the dir to exist for results.jsonl)
   │      Evidence · 
   └ How  · clone 4 sh files from `scripts/hle/gpt5.2_low/run_precache.sh` / `run_delegated.sh` and `scripts/gpqa/gpt5.2/run_precache.sh` / `run_no_integrate_high.sh`, edit paths

## Phase 2 — HLE [0/2] — DEFERRED

04 ⏸ HLE precache (gpt-5.4, effort=low, workers=4, 200 → ~100 questions × 8 explores) — DEFERRED at 356/800 (44.5%)
   ├ G1 ☐ Gate · banner line `Tasks: K to run, 0 already cached` appears with K = expected total (≈800) on first launch
   │      Evidence · 
   ├ G2 ☐ Gate · `find ../analysis/cache/hle/gpt5.4_high/gold -name result.json | wc -l` ≥ 800 after run completes (≥100 qids × 8 explores; may be slightly lower if HLE-gold subset has <100 items)
   │      Evidence · 
   ├ G3 ☐ Gate · timed_out rate across all cache result.json files ≤ 10% (R9 defense): `python -c "import json,glob; rs=[json.load(open(p)) for p in glob.glob('../analysis/cache/hle/gpt5.4_high/gold/*/explore_*/result.json')]; n_to=sum(1 for r in rs if r.get('timed_out')); print(f'{n_to}/{len(rs)}={100*n_to/len(rs):.1f}%')"`
   │      On-fail · bump backend.timeout in `hle_gpt5.4_precache.yaml` from 1200 to 1800; resume run via cache hits
   │      Evidence · 
   ├ G4 ☐ Gate · `grep -c 'HTTPStatusError' ../analysis/run/hle/gpt5.4_high/precache.log` == 0 (R1 defense: no terminal 429 cascade)
   │      Evidence · 
   ├ G5 ☐ Gate · **Cost cap** — sum of `cost_usd` from cache files ≤ **$123** (smoke-derived: 800 calls × $0.1343 × 1.15); STOP if higher
   │      On-fail · investigate: (a) 100-q distribution has heavier cost tail than 3-q smoke (raise cap or rerun smoke at n=10), (b) pricing stub stale, (c) silent token-burn bug
   │      Evidence · 
   ├ G6 ☐ Gate · **Cache completeness** — every qid subdir under `../analysis/cache/hle/gpt5.4_high/gold/` has EXACTLY 8 `explore_*/result.json` files (not ≥8, not ≤8); `python -c "from pathlib import Path; bad=[q for q in Path('../analysis/cache/hle/gpt5.4_high/gold').iterdir() if q.is_dir() and len(list(q.glob('explore_*/result.json')))!=8]; print(f'qids != 8 explores: {len(bad)}', bad[:5])"` returns count 0
   │      On-fail · the offending qids had partial 429 / timeout / crash; rerun precache with --resume; cache_only eval will crash on these qids if not fixed
   │      Evidence · 
   ├ G7 ☐ Gate · zero `Traceback` in `precache.log` other than retry-recovered httpx errors
   │      Evidence · 
   └ How  · `bash scripts/hle/gpt5.4_high/run_precache.sh`; after launch share PID + absolute log path; estimated 40–60min wall

05 ☐ HLE eval (gpt-5.4, effort=high, cache_only=true, workers=4, num=200 → 100 graded rows)
   ├ G1 ☐ Gate · `wc -l ../analysis/run/hle/gpt5.4_high/run_*/results.jsonl` ≥ 100
   │      Evidence · 
   ├ G2 ☐ Gate · **Timeout rate** — `timed_out=true` rate across results.jsonl rows ≤ **10%** (USER REQUIREMENT)
   │      On-fail · bump backend.timeout in `hle_gpt5.4_high_delegated.yaml` from 1200 to 1800; rerun the timed-out subset only via resume
   │      Evidence · 
   ├ G3 ☐ Gate · non-empty `predicted_answer` rate ≥ 95% (high-effort reasoning should always commit to an answer)
   │      Evidence · 
   ├ G4 ☐ Gate · **Acc must beat 5.2** — `Acc(5.4 high) > 57.00%` (USER REQUIREMENT, line 417 of `tab:backbone-ablation`); STOP if `Acc ≤ 57.00`
   │      On-fail · do NOT write to paper; investigate (a) prompt regression on 5.4, (b) judge regression, (c) cache contamination, (d) genuine model regression
   │      Evidence · 
   ├ G5 ☐ Gate · 0 rows with empty `predicted_answer` AND `is_correct=true` (judge integrity, RB-class defense)
   │      Evidence · 
   ├ G6 ☐ Gate · **Judge integrity** (R6 defense — combined check):
   │      (a) `run_config.json` field `judge_model == "claude-haiku-4-5-20251001"` (sticky to 5.2 baseline for apples-to-apples comparison)
   │      (b) `sum(judge_cost_usd) > 0` across all results.jsonl rows — non-zero proves the LLM judge actually fired; zero = 2026-04-11 disaster (judge silently None → string fallback → ~8pp acc underestimate)
   │      Evidence · 
   ├ G7 ☐ Gate · **Cost cap** — true wallet `sum(cost_by_component.orchestrator + cost_by_component.integrate + judge_cost_usd)` across results.jsonl ≤ **$15** (10% of HLE precache cap as orch+integrate share + Haiku judge ≤ $1); STOP if higher
   │      On-fail · same investigation as item 04 G5
   │      Evidence · 
   ├ G8 ☐ Gate · zero `Traceback` in `delegated.log` other than codex 429 retry-recovered events
   │      Evidence · 
   ├ G9 ☐ Soft-Gate · **Post-eval sanity review + observational stats** — sample 5 random qids (seed=42); for each verify:
   │      (a) trajectory shows orchestrator visibly READ cached explores (`cost_by_component.explorer > 0` per row, attribution from cache), and submitted answer references explore content
   │      (b) `predicted_answer` length distribution: median ≥ 10 chars, IQR > 5 chars (HLE answers are mostly short LaTeX expressions)
   │      (c) judge `verdict_reasoning` cites gold value verbatim in its decision (Haiku-judge integrity)
   │      Observational stats (PRINT only, no auto-stop — 玄学 noise band):
   │      · `Pass@1(5.4) − Pass@1(5.2 high) = ?pp` — 5.2 was 54.00%; report delta and your read on whether explorer behaviour shifted
   │      · per-row max cost / median cost ratio across 100 rows; flag rows above 5× median for human inspection (latent reasoning loop)
   │      · avg input/output tokens per call (compare against 5.2 if memory; just print otherwise)
   │      Justification (required) · 1-2 sentences per (a)-(c) citing qid + `trajectories/<qid>/trajectory.md` line + `results.jsonl` row index. The Observational stats are FYI only, not stop conditions.
   │      Evidence · 
   └ How  · `bash scripts/hle/gpt5.4_high/run_delegated.sh`; estimated 15–25min wall

## Phase 3 — GPQA [0/2] — DEFERRED (blocked on Phase 2)

06 ☐ GPQA precache (gpt-5.4, effort=low, workers=4, 198 questions × 8 explores)
   ├ G1 ☐ Gate · banner `Tasks: K to run, 0 already cached` on first launch with K ≈ 1584
   │      Evidence · 
   ├ G2 ☐ Gate · `find ../analysis/cache/gpqa/gpt5.4 -name result.json | wc -l` ≥ 1584
   │      Evidence · 
   ├ G3 ☐ Gate · timed_out rate across cache files ≤ 10% (R9 defense)
   │      On-fail · bump backend.timeout in `gpqa_gpt5.4_precache.yaml` to 1800; resume
   │      Evidence · 
   ├ G4 ☐ Gate · `grep -c 'HTTPStatusError' ../analysis/run/gpqa/gpt5.4_no_integrate_high/precache.log` == 0 (R1 defense)
   │      Evidence · 
   ├ G5 ☐ Gate · **Cost cap** — sum of `cost_usd` from cache files ≤ **$244** (smoke-derived: 1584 calls × $0.1343 × 1.15); STOP if higher
   │      On-fail · same investigation as item 04 G5; ALSO: GPQA precache only launches if HLE eval clears budget — this gate may not even fire if user halts after HLE
   │      Evidence · 
   ├ G6 ☐ Gate · **Cache completeness** — every qid subdir under `../analysis/cache/gpqa/gpt5.4/` has EXACTLY 8 `explore_*/result.json`; same one-liner as item 04 G6 with cache path swapped
   │      On-fail · same as item 04 G6
   │      Evidence · 
   ├ G7 ☐ Gate · zero `Traceback` in precache.log other than retry-recovered events
   │      Evidence · 
   └ How  · `bash scripts/gpqa/gpt5.4/run_precache.sh`; estimated 1–1.5h wall

07 ☐ GPQA eval (gpt-5.4, effort=high, cache_only=true, no_integrate=true, workers=1, 198 graded rows)
   ├ G1 ☐ Gate · `wc -l ../analysis/run/gpqa/gpt5.4_no_integrate_high/run_*/results.jsonl` == 198
   │      Evidence · 
   ├ G2 ☐ Gate · **Timeout rate** — `timed_out=true` rate ≤ **10%** (USER REQUIREMENT)
   │      On-fail · bump backend.timeout in `gpqa_gpt5.4_no_integrate_high.yaml` to 1800; rerun timed-out subset via resume
   │      Evidence · 
   ├ G3 ☐ Gate · `predicted_answer` matches A-E letter regex rate ≥ 90% (R7 defense — GPQA grader regex extraction sane on 5.4 output)
   │      Evidence · 
   ├ G4 ☐ Gate · **Acc must beat 5.2** — `Acc(5.4 high) > 71.07%` (USER REQUIREMENT, line 418 of `tab:backbone-ablation`); STOP if `Acc ≤ 71.07`
   │      On-fail · do NOT write to paper; investigate as in 05-G4
   │      Evidence · 
   ├ G5 ☐ Gate · 0 rows with `predicted_answer` not in {A,B,C,D,E} AND `is_correct=true` (grader can't be right on a non-letter)
   │      Evidence · 
   ├ G6 ☐ Gate · `run_config.json` field `judge_model == null` (GPQA uses string-match, no judge — verify nothing snuck in)
   │      Evidence · 
   ├ G7 ☐ Gate · **Cost cap** — true wallet `sum(cost_by_component.orchestrator + cost_by_component.integrate)` (judge=0 because GPQA uses string-match) ≤ **$25** (10% of GPQA precache cap as orch+integrate share); STOP if higher
   │      On-fail · same investigation as item 04 G5
   │      Evidence · 
   ├ G8 ☐ Gate · zero `Traceback` in `delegated.log` other than codex retry-recovered events
   │      Evidence · 
   ├ G9 ☐ Soft-Gate · **Post-eval sanity review + observational stats** — sample 5 random qids (seed=42); for each verify:
   │      (a) extracted MC letter matches the orchestrator's committed final letter in trajectory (no stray-letter regex extraction)
   │      (b) orchestrator visibly aggregated 8 cached explore answers; NOT first-explore copy; trajectory shows reasoning over multiple candidates
   │      (c) extracted letter distribution across 198 rows: A/B/C/D within ±15% of uniform (24.3% ± 3.6pp); huge skew = grader pulling wrong letter from prose
   │      Observational stats (PRINT only, no auto-stop):
   │      · `Pass@1(5.4) − Pass@1(5.2 high) = ?pp` — 5.2 was 56.06%; report delta and your read on whether explorer behaviour shifted
   │      · per-row max/median cost ratio across 198 rows; flag rows above 5× median for human inspection
   │      · avg input/output tokens per call (compare against 5.2's roughly known ratio)
   │      Justification (required) · 1-2 sentences per (a)-(c) with qid + trajectory line + row index. Observational stats are FYI only.
   │      Evidence · 
   └ How  · `bash scripts/gpqa/gpt5.4/run_no_integrate_high.sh`; estimated 1.5–2h wall

## Phase 4 — Paper integration [0/3] — DEFERRED (blocked on Phases 2-3)

08 ☐ Parse 2 results.jsonl → Pass@1 / Acc / Gain / true-wallet $/q
   ├ G1 ☐ Gate · 2 metric rows printed: HLE-high and GPQA-high
   │      Evidence · 
   ├ G2 ☐ Gate · `Acc - Pass@1 == Gain` for each row (within 0.1pp rounding)
   │      Evidence · 
   ├ G3 ☐ Gate · `$/q` = (precache_total_cost + sum(non-explorer cost components in eval) + sum(judge_cost_usd)) / num_questions; HLE expected ≈ ($32 + $6 + $0.07)/100 = ~$0.38/q; GPQA expected ≈ ($35 + $10)/198 = ~$0.23/q. Sanity check: ratios stay in same ballpark as 5.2's 0.26/0.16 × ~1.1 pricing.
   │      Evidence · 
   ├ G4 ☐ Gate · **5.4 vs 5.2 differential**: explicitly print `Acc(5.4) - Acc(5.2)` per benchmark; both must be > 0 (otherwise items 05/07 G4 would have already STOPPED, but verify here as a final guard before paper write)
   │      Evidence · 
   └ How  · python script reading `analysis/run/hle/gpt5.4_high/run_*/results.jsonl` and `analysis/run/gpqa/gpt5.4_no_integrate_high/run_*/results.jsonl`, computing Pass@1 (`first_candidate_correct` rate), Acc (`is_correct` rate), Gain, and true-wallet $/q (precache cost from cache + eval orch+integrate cost from results.jsonl, EXCLUDING the explorer re-attribution to avoid double-counting)

09 ☐ Insert 2 GPT-5.4 rows into `tab:backbone-ablation` at `Publication/paper/main.tex`
   ├ G1 ☐ Gate · 2 new rows inserted at line 419 area, BETWEEN the existing `GPT-5.2 & high & GPQA` row (line 418) and the next `\midrule` (line 419), so the GPT block now reads: GPT-5.2 low HLE / GPT-5.2 high HLE / GPT-5.2 high GPQA / GPT-5.4 high HLE / GPT-5.4 high GPQA / [midrule] / Qwen
   │      Evidence · 
   ├ G2 ☐ Gate · row schema matches surrounding GPT-5.2 rows: `<Backbone> & <Effort> & <Bench> & <Pass@1> & <Acc> & <Gain> & <$/q> \\`
   │      Evidence · 
   ├ G3 ☐ Gate · existing 5.2 / Sonnet / Qwen rows UNTOUCHED (verify by `git diff` showing only +2 lines, no other modifications inside the table); existing bold marks (`\textbf{71.07}`, `\textbf{+15.0}`, `\textbf{0.16}`) preserved
   │      Evidence · 
   ├ G4 ☐ Gate · narrative paragraph below the table (line 429-431) not modified UNLESS user explicitly asks; the new rows speak via the table only
   │      Evidence · 
   └ How  · Edit tool, exact insertion above the third `\midrule` in the table, after the GPQA gpt-5.2 row on line 418

10 ☐ Recompile paper + verify table renders correctly
   ├ G1 ☐ Gate · `cd Publication/paper && bash compile.sh` exits 0
   │      Evidence · 
   ├ G2 ☐ Gate · `Publication/paper/build/main.pdf` mtime updated (within last 10 minutes); file size > 200 KiB
   │      Evidence · 
   ├ G3 ☐ Gate · `tab:backbone-ablation` in PDF shows 5 rows in the GPT block (3 GPT-5.2 + 2 GPT-5.4) followed by `\midrule` then 4 Qwen rows
   │      Evidence · 
   ├ G4 ☐ Gate · zero `Overfull \hbox` warnings on the page containing Table 3 (compile log filtered by line range of the table)
   │      Evidence · 
   ├ G5 ☐ Gate · 5.4 row numerics in PDF match item 08 computed values (visual cross-check: open PDF, read table, compare to printed metrics)
   │      Evidence · 
   └ How  · `cd /data3/peijia/dr-claw/Explain/Publication/paper && bash compile.sh`; visual inspection of Table 3 page in `build/main.pdf`
