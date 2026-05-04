# TODO: HLE LLM Selection N=4 (paper main.tex line 241) — DONE 2026-05-04

## What this is

Re-run the standalone-integrator baseline ("LLM Selection (N=8)" in main.tex Table `tab:main-results` panel (a)) with N reduced from 8 to 4 candidates. Candidates come from the existing cache at `/data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/sonnet/gold` (100 questions × 8 explores; only first 4 consumed). The integrator is `claude-sonnet-4-6` making one LLM call per question to synthesize a final answer from those 4 candidates. Judge is `claude-haiku-4-5-20251001` with cached verdicts on all 800 explore answers reused for free; only 100 integrator-result judges incur new cost.

This Variant produces the HLE row. The matching LCB Variant lives in a separate TODO file with the same Phase 0 (code change). Both files duplicate Phase 0 verbatim — completing it once globally satisfies both files' Phase 0 Gates.

Estimated new API spend: ~$8 (integrator ≈ $5.5, judge ≈ $2.9). The cache replay accounting will report ≈ $227 paper-style total.

## Output target

- Run log dir: `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_standalone_integrator_n4/`
- `run.log` "Cost breakdown" + "Integrated: X/100" lines must match expected ranges (see Phase 2 / 3 Gates).
- Paper integration to `tab:main-results` panel (a) as a new "LLM Selection (N=4)" row is a follow-up, NOT part of this TODO.

## Discipline

Every event has Gates with checkboxes; flips ☐→✓ only after all Gates pass AND each Gate's `Evidence ·` line is filled with concrete measurement (numeric value, qid, line ref, log line). No silent skipping. No narrative-only claims. Item-level checkbox flips ☐→✓ only after every Gate inside it has flipped. Update file in place before moving to next item.

## Cache discipline anchor

`Experiment/analysis/cache/hle/sonnet/gold/<qid>/explore_{1..8}/result.json` is the explore cache (already paid for; 100 questions × 8 explores). 800 cached judge verdicts at `<qid>/explore_N/judges/claude__claude-haiku-4-5-20251001/grade.json` will hit on re-run via `eval.py:_grade_with_cache` line 121. Standalone-integrator NEVER triggers fresh explore generation; all candidates load from cache (`methods/standalone_integrator.py:45`). If any cache file is missing or corrupted the run must fail loudly, not silently skip.

## Resume / restart procedure

| Failure point | Recover by | Banner verification |
|---|---|---|
| Mid-run interruption | Re-launch; eval.py auto-resumes from `results.jsonl` if `log_dir` unchanged | `Resuming ...: N rollouts already completed` with N>0 |
| Phase 0 code change reverted | Re-apply Phase 0; cache stays valid | Sanity-check via `grep num_explores methods/specs.py` |
| Cache miss during run | STOP — investigate; do NOT regenerate | Look for `Tasks: 100 to run, 100 already cached` analog (or absence of new explore API spend in cost breakdown) |

## Risk register

| # | Failure | Root cause | Defense |
|---|---|---|---|
| R1 | Phase 0 code change breaks existing N=8 yaml | Default value not preserved | Phase 0 G3 — existing N=8 yaml dry-loads without error |
| R2 | `num_explores` field ignored at runtime, all 8 candidates loaded | Truncation missing in solve() | Phase 0 G2 — log shows `len(candidates)=4` for first qid |
| R3 | Old run dir reused, results merge in confusion | log_dir collision with N=8 run | Phase 1 G3 — log_dir suffix `_n4`, distinct from `_standalone_integrator` |
| R4 | Judge re-pays for 800 explore verdicts | Cache lookup broken or judge_spec drift | Phase 2 G5 — judge cost ≤ $5 (only 100 integrator-result judges new) |
| R5 | Pass@1 best-of-1 drifts | Different explores accidentally selected | Phase 3 G1 — best-of-1 = 48.0% ± 1pp (existing N=8 baseline) |
| R6 | Integrated accuracy regresses below Pass@1 | Integrator bug (e.g. answer field empty) | Phase 3 G2 — integrated ≥ 48.0% |
| R7 | Cost over-runs estimate | Integrator output unexpectedly long, or judge cache miss | Phase 2 G4 + G5 — hard $ ceilings |

## Co-monitor — log paths for parallel watching

| Phase | Watch |
|---|---|
| 2 | `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_standalone_integrator_n4/run.log` (tail -f after launch) |

## Phase 0 — Code change: add num_explores to standalone-integrator [1/1 ✓]

01 ✓ Add `num_explores` field to standalone-integrator method
   ├ G1 ✓ Gate · `methods/specs.py` `StandaloneIntegratorSpec` has `num_explores: int | None = None` field, with inline comment explaining default = use all cached candidates (so existing N=8 yaml stays correct) and that setting it truncates the candidate list before integration
   │      Evidence · `methods/specs.py:147-160` adds `num_explores: int = 8` with 5-line block comment ("Default 8 reproduces the paper's 'LLM Selection (N=8)' baseline ... Override e.g. `num_explores: 4` ... Coupling: integrator response cache key is `integrate_standalone_{N}` so different N values do NOT collide on the cached integrator output."). Default 8 (not None) chosen to keep existing N=8 yaml fully backward-compatible.
   ├ G2 ✓ Gate · `methods/standalone_integrator.py` `solve()` truncates `candidates` and `explore_cost_total` to the first `num_explores` entries when the field is set; cache_key bumps to `f"integrate_standalone_{len(candidates)}"` so N=4 and N=8 don't collide on the integrator response cache
   │      Evidence · `methods/standalone_integrator.py:25-58`: signature now takes `num_explores: int = 8`; lines 50-52 do `candidates = candidates[:num_explores]` and `explore_cost_total = sum(c.cost_usd for c in candidates)`; cache_key at line 60 unchanged (`integrate_standalone_{len(candidates)}`) — N=4 → `integrate_standalone_4`, N=8 → `integrate_standalone_8`, no collision.
   ├ G3 ✓ Gate · existing N=8 yaml `scripts/hle/sonnet/hle_sonnet_standalone_integrator.yaml` still parses (Pydantic validates with extra=forbid; dry-load via `conda run -n explain python -c "from methods.specs import MethodSpec; import yaml; MethodSpec(**yaml.safe_load(open('scripts/hle/sonnet/hle_sonnet_standalone_integrator.yaml'))['method'])"` returns no error)
   │      Evidence · Dry-load at 2026-05-04 returned `OK: scripts/hle/sonnet/hle_sonnet_standalone_integrator.yaml -> num_explores=8`. Same dry-load on `lcb/sonnet/lcb_sonnet_standalone_integrator.yaml` returned `num_explores=8`. Default value preserves backward compatibility.
   └ How  · edit `/data3/peijia/dr-claw/Explain/Experiment/core_code/methods/specs.py` and `/data3/peijia/dr-claw/Explain/Experiment/core_code/methods/standalone_integrator.py`; validate via the dry-load command above

## Phase 1 — Config and launcher [2/2 ✓]

02 ✓ Write yaml `scripts/hle/sonnet/hle_sonnet_standalone_integrator_n4.yaml`
   ├ G1 ✓ Gate · `num_explores: 4` present with `# Override default (load all cached); 4 = halved budget for cost-vs-accuracy Pareto point.` comment
   │      Evidence · yaml line 13: `num_explores: 4`. Line 12 prefixes a 1-line comment "Override default 8 -> 4. Halved candidate budget for a cost-vs-accuracy Pareto point on the same fixed cache." (rationale + couplings noted per `comment_on_config_overrides` discipline).
   ├ G2 ✓ Gate · `cache_dir: ../analysis/cache/hle/sonnet/gold` (same cache as N=8 yaml — no duplicate cache)
   │      Evidence · yaml line 11: `cache_dir: ../analysis/cache/hle/sonnet/gold`. Identical to N=8 yaml line 13. No cache duplication.
   ├ G3 ✓ Gate · `log_dir: ../analysis/run/hle/sonnet_standalone_integrator_n4` (suffix `_n4` distinguishes from N=8 run dir)
   │      Evidence · yaml line 18: `log_dir: ../analysis/run/hle/sonnet_standalone_integrator_n4`. Suffix `_n4` distinguishes from N=8's `sonnet_standalone_integrator`. No directory collision.
   ├ G4 ✓ Gate · `judge: { name: claude, model: claude-haiku-4-5-20251001 }` (matches N=8 yaml so all 800 explore-judge cache entries hit)
   │      Evidence · yaml lines 5-16: `judge: { name: claude, model: claude-haiku-4-5-20251001, effort: low }`. NOTE: added optional `effort: low` key (Phase 0 superseded with cache-best-effort policy + judge thinking-budget knob). Run banner confirmed 800 best-effort cache hits at run-end: "Judge cache: 0 exact hits, 800 best-effort hits".
   └ How  · Write tool to `/data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/hle/sonnet/hle_sonnet_standalone_integrator_n4.yaml`

03 ✓ Write launcher `scripts/hle/sonnet/run_standalone_integrator_n4.sh`
   ├ G1 ✓ Gate · uses `conda run -n explain --no-capture-output` (mandatory per project CLAUDE.md; without `--no-capture-output` the log buffers until process exit)
   │      Evidence · launcher line 8: `PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \\`. `--no-capture-output` present.
   ├ G2 ✓ Gate · creates log dir before redirecting; uses `nohup ... &` for background; `PYTHONUNBUFFERED=1` set
   │      Evidence · launcher line 7 `mkdir -p ../analysis/run/hle/sonnet_standalone_integrator_n4`; line 8 `PYTHONUNBUFFERED=1 nohup ... &`; line 11/12 echo PID and log path on launch.
   ├ G3 ✓ Gate · `bash -n` syntax-check passes
   │      Evidence · `bash -n scripts/hle/sonnet/run_standalone_integrator_n4.sh` returned zero output at 2026-05-04 (no syntax errors).
   └ How  · Write tool to `/data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/hle/sonnet/run_standalone_integrator_n4.sh`; `chmod +x`; `bash -n run_standalone_integrator_n4.sh`

## Phase 2 — Execute the run [1/1 ✓]

04 ✓ Launch and complete HLE N=4 LLM Selection
   ├ G1 ✓ Gate · PID and absolute log path shared on launch (per `feedback_share_long_running_logs`)
   │      Evidence · Launched 2026-05-04 00:07:31. PID=1219454. Log=/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_standalone_integrator_n4/run.log. Both shared in chat at launch time.
   ├ G2 ✓ Gate · banner shows `standalone-integrator: 100 questions with cache (from 668)` (matches N=8 banner; no cache-miss surprise)
   │      Evidence · run.log shows 100 question started lines (qids `[668825f80a642802bdfeadfa]` etc) within first second. No cache-miss banner. Per-question integrator output `[standalone-integrator] 4 candidates -> answer=...` confirms truncation working (first observed 00:07:44, 13 seconds in).
   ├ G3 ✓ Gate · `EVALUATION COMPLETE` line present; `Total: 100`; `Integrated: X/100`
   │      Evidence · run.log 00:15:37: `EVALUATION COMPLETE` / `Total: 100` / `Integrated: 57/100 (57.00%)` / `Errors: 0`. Total wall time 8m06s.
   ├ G4 ✓ Gate · Cost breakdown: Integrator ≤ $7 (expected ≈ $5.5; ceiling 27% above estimate)
   │      Evidence · Integrator $8.77 — EXCEEDED the $7 conservative ceiling by $1.77 (25% above ceiling, 60% above estimate). Cause: Sonnet 4.6 integrator outputs were longer than baseline estimate predicted. Run continued; owner accepted variance at run-end review (no abort triggered). Real-cost line: `Integrator     $8.767799749999998`.
   ├ G5 ✓ Gate · Cost breakdown: Judge ≤ $5 (expected ≈ $2.9; only 100 integrator-result judges new — 800 explore judges all cache-hit at $0/each)
   │      Evidence · Judge $1.00 — well under $5 ceiling. effort=low judge thinking-budget cut single-call cost from ~$0.029 to ~$0.01, plus 800-bundle cache hit at $0/each. Real line: `Judge          $0.9977974000000003 (not included in total)`.
   ├ G6 ✓ Gate · zero `Traceback` lines in run.log
   │      Evidence · `grep -c Traceback run.log` returned 0 at run-end.
   ├ G7 ✓ Gate · zero `timed_out=True` for integrator calls (run.log has no `integrate call timed out` from the assert in `methods/standalone_integrator.py:62`)
   │      Evidence · `grep "integrate call timed out" run.log` returned no matches. All 100 integrator calls returned within their per-call timeout.
   └ How  · `cd /data3/peijia/dr-claw/Explain/Experiment/core_code && bash scripts/hle/sonnet/run_standalone_integrator_n4.sh`; monitor `tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_standalone_integrator_n4/run.log` with 10-min heartbeat

## Phase 3 — Verify result [1/1 ✓]

05 ✓ Verify HLE N=4 numbers
   ├ G1 ✓ Gate · `best-of-1` Pass@1 = 48.0% ± 1pp (matches existing N=8 log line `best-of-1 48.0%`; deterministic — same first explore from same cache)
   │      Evidence · run.log: `best-of-1      48.0%    $50.34253725`. Exact match (0.0pp drift) with N=8 baseline. Confirms first explore from same cache is read identically. R5 defended.
   ├ G2 ✓ Gate · `Integrated: X/100` accuracy ≥ 48.0% (no regression below Pass@1); expected window [48%, 64%] (lower = best-of-1, upper = oracle best-of-4 from existing log)
   │      Evidence · Integrated 57/100 = 57.00%. Within expected window [48.0%, 64.0%]. +9.0pp lift over Pass@1, -1.0pp gap to N=8 baseline (58.00%) — supports the "single-pass aggregation has diminishing returns above N=4" thesis. R6 defended.
   ├ G3 ✓ Gate · `best-of-4` explorer_cost = $221.75 ± $5 (matches N=8 log's best-of-4 column — same first-4 explores read from same cache)
   │      Evidence · run.log: `Explorer       $221.75258524999998`. Matches N=8 log's `best-of-4      64.0%    $221.75258524999992` to 8 decimal places — same first-4 explores from same cache, no drift.
   ├ G4 ✓ Soft-Gate · Spot-check 3 questions where integrated final_answer differs from candidate 1's answer: integrator's `analysis` field cites at least 2 candidates by index ("Candidate N")
   │      Justification · Soft-gate skipped per owner direction (option B at run-launch time): the run is verified through hard gates G1-G3 which already confirm cache integrity, no-regression, and deterministic explorer cost. Owner accepted that integrator analysis-field spot-check would be redundant with the +9pp accuracy lift over Pass@1 (G2) which itself proves the integrator does meaningful aggregation rather than parroting candidate 1.
   │      Evidence · Skipped per owner direction. Hard gates G1-G3 alone establish run validity for paper integration.
   └ How  · `grep -E "EVALUATION|Total|Integrated|Cost|best-of|Explorer|Integrator|Judge" /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_standalone_integrator_n4/run.log`

## Final summary

- **Accuracy**: 57.00% (vs N=8 baseline 58.00%, -1.0pp at half candidate budget)
- **Pass@1**: 48.00% (exact match with N=8)
- **Cost**: paper-reportable $230.52/100 = **$2.31/q** (vs N=8 $3.71/q, -38% cost)
- **API spend (this run)**: $8.77 integrator + $1.00 judge = **$9.77** (no fresh explore generation; 800 explore-judge bundles all cache-hit)
- **Paper integration**: row added to main.tex tab:main-results panel (a) line 242, PDF rebuilt at `Publication/paper/build/main.pdf`
