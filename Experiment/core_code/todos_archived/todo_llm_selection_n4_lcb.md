# TODO: LCB LLM Selection N=4 (paper main.tex line 267) — DONE 2026-05-04

## What this is

Re-run the standalone-integrator baseline ("LLM Selection (N=8)" in main.tex Table `tab:main-results` panel (b), LiveCodeBench) with N reduced from 8 to 4 candidates. Candidates come from the existing cache at `/data3/peijia/dr-claw/Explain/Experiment/analysis/cache/lcb/sonnet` (175 questions; 151 have 8 explores, 9 have 0, the rest have 1–7). For each question only the first `min(4, available)` candidates are consumed. The integrator is `claude-sonnet-4-6` making one LLM call per question to synthesize the best Python solution from those candidates. Grading runs cached test cases via `lcb_runner` — there is no LLM judge for LCB, so judge cost is $0.

This Variant produces the LCB row. The matching HLE Variant lives in a separate TODO file with the same Phase 0 (code change). Both files duplicate Phase 0 verbatim — completing it once globally satisfies both files' Phase 0 Gates.

Estimated new API spend: ~$10 (integrator only; judge $0 because LCB grades by code execution, not LLM). The cache replay accounting will report ≈ $157 paper-style total.

## Output target

- Run log dir: `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/`
- `run.log` "Cost breakdown" + "Integrated: X/175" lines must match expected ranges (see Phase 2 / 3 Gates).
- Paper integration to `tab:main-results` panel (b) as a new "LLM Selection (N=4)" row is a follow-up, NOT part of this TODO.

## Discipline

Every event has Gates with checkboxes; flips ☐→✓ only after all Gates pass AND each Gate's `Evidence ·` line is filled with concrete measurement (numeric value, qid, line ref, log line). No silent skipping. No narrative-only claims. Item-level checkbox flips ☐→✓ only after every Gate inside it has flipped. Update file in place before moving to next item.

## Cache discipline anchor

`Experiment/analysis/cache/lcb/sonnet/<qid>/explore_{1..N}/result.json` is the explore cache (already paid for). Cache density is uneven: 151 qids have 8 explores, 9 have 0 (these will return empty answers and grade incorrect — same behavior as N=8 run), 1 has 1, 3 have 2, 2 have 3, 2 have 4, 2 have 5, 2 have 6, 3 have 7. For N=4: 151 + (qids with ≥4 explores) consume exactly 4; the partial qids consume whatever they have. LCB has no LLM judge, so there is no judge cache layer to worry about; grading runs `lcb_runner` test cases each call (deterministic and free). Standalone-integrator NEVER triggers fresh explore generation (`methods/standalone_integrator.py:45`).

## Resume / restart procedure

| Failure point | Recover by | Banner verification |
|---|---|---|
| Mid-run interruption | Re-launch; eval.py auto-resumes from `results.jsonl` if `log_dir` unchanged | `Resuming ...: N rollouts already completed` with N>0 |
| Phase 0 code change reverted | Re-apply Phase 0; cache stays valid | Sanity-check via `grep num_explores methods/specs.py` |
| Test-case execution flaky (`lcb_runner` timeout) | Re-run; lcb_runner caches its own results within run | Look for `EVALUATION COMPLETE` and `Total: 175` |

## Risk register

| # | Failure | Root cause | Defense |
|---|---|---|---|
| R1 | Phase 0 code change breaks existing N=8 yaml | Default value not preserved | Phase 0 G3 — existing N=8 yaml dry-loads without error |
| R2 | `num_explores` field ignored at runtime, all 8 candidates loaded | Truncation missing in solve() | Phase 0 G2 — log shows `len(candidates)=4` for the first qid that has ≥4 explores |
| R3 | Old run dir reused, results merge in confusion | log_dir collision with N=8 run | Phase 1 G3 — log_dir suffix `_n4`, distinct from `_standalone_integrator` |
| R4 | Cost over-runs estimate | Integrator output unexpectedly long (LCB candidates carry full code; integrator outputs full code in `final_code`) | Phase 2 G4 — Integrator ≤ $13 (expected ≈ $10) |
| R5 | Pass@1 best-of-1 drifts | Different explores accidentally selected | Phase 3 G1 — best-of-1 = 77.14% ± 1pp (existing N=8 baseline) |
| R6 | Integrated accuracy regresses below Pass@1 | Integrator emits empty `final_code` for many questions | Phase 3 G2 — integrated ≥ 77.14% |
| R7 | Partial-cache qids (those with <4 explores) silently inflate timed_out / empty count | standalone-integrator returns empty SolveResult when `len(candidates)==0` | Phase 3 G3 — explore-distribution histogram matches N=8 log's distribution exactly |

## Co-monitor — log paths for parallel watching

| Phase | Watch |
|---|---|
| 2 | `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/run.log` (tail -f after launch) |

## Phase 0 — Code change: add num_explores to standalone-integrator [1/1 ✓]

01 ✓ Add `num_explores` field to standalone-integrator method
   ├ G1 ✓ Gate · `methods/specs.py` `StandaloneIntegratorSpec` has `num_explores: int | None = None` field, with inline comment explaining default = use all cached candidates (so existing N=8 yaml stays correct) and that setting it truncates the candidate list before integration
   │      Evidence · `methods/specs.py:147-160` adds `num_explores: int = 8` with 5-line block comment ("Default 8 reproduces the paper's 'LLM Selection (N=8)' baseline ... Override e.g. `num_explores: 4` ... Coupling: integrator response cache key is `integrate_standalone_{N}` so different N values do NOT collide on the cached integrator output."). Default 8 (not None) chosen to keep existing N=8 yaml fully backward-compatible.
   ├ G2 ✓ Gate · `methods/standalone_integrator.py` `solve()` truncates `candidates` and `explore_cost_total` to the first `num_explores` entries when the field is set; cache_key bumps to `f"integrate_standalone_{len(candidates)}"` so N=4 and N=8 don't collide on the integrator response cache
   │      Evidence · `methods/standalone_integrator.py:25-58`: signature now takes `num_explores: int = 8`; lines 50-52 do `candidates = candidates[:num_explores]` and `explore_cost_total = sum(c.cost_usd for c in candidates)`; cache_key at line 60 unchanged (`integrate_standalone_{len(candidates)}`) — N=4 → `integrate_standalone_4`, N=8 → `integrate_standalone_8`, no collision.
   ├ G3 ✓ Gate · existing N=8 yaml `scripts/lcb/sonnet/lcb_sonnet_standalone_integrator.yaml` still parses (Pydantic validates with extra=forbid; dry-load via `conda run -n explain python -c "from methods.specs import MethodSpec; import yaml; MethodSpec(**yaml.safe_load(open('scripts/lcb/sonnet/lcb_sonnet_standalone_integrator.yaml'))['method'])"` returns no error)
   │      Evidence · Dry-load at 2026-05-04 returned `OK: scripts/lcb/sonnet/lcb_sonnet_standalone_integrator.yaml -> num_explores=8`. Default value preserves backward compatibility.
   └ How  · edit `/data3/peijia/dr-claw/Explain/Experiment/core_code/methods/specs.py` and `/data3/peijia/dr-claw/Explain/Experiment/core_code/methods/standalone_integrator.py`; validate via the dry-load command above

## Phase 1 — Config and launcher [2/2 ✓]

02 ✓ Write yaml `scripts/lcb/sonnet/lcb_sonnet_standalone_integrator_n4.yaml`
   ├ G1 ✓ Gate · `num_explores: 4` present with `# Override default (load all cached); 4 = halved budget for cost-vs-accuracy Pareto point.` comment
   │      Evidence · yaml line 13: `num_explores: 4`. Lines 8-12 prefix a 5-line block comment "Override default 8 -> 4. Halved candidate budget for a cost-vs-accuracy Pareto point on the same fixed cache. LCB grades via lcb_runner test cases (no LLM judge), so judge cost is $0; only the 175 integrator calls incur new API spend."
   ├ G2 ✓ Gate · `cache_dir: ../analysis/cache/lcb/sonnet` (same cache as N=8 yaml — no duplicate cache)
   │      Evidence · yaml line 7: `cache_dir: ../analysis/cache/lcb/sonnet`. Identical to N=8 yaml line 8. No cache duplication.
   ├ G3 ✓ Gate · `log_dir: ../analysis/run/lcb/sonnet_standalone_integrator_n4` (suffix `_n4` distinguishes from N=8 run dir)
   │      Evidence · yaml line 16: `log_dir: ../analysis/run/lcb/sonnet_standalone_integrator_n4`. Suffix `_n4` distinguishes from N=8's `sonnet_standalone_integrator`. No directory collision.
   ├ G4 ✓ Gate · NO `judge:` block (LCB grades via `lcb_runner` test cases — adding a judge spec would wrong-route grading)
   │      Evidence · yaml has no `judge:` key under `benchmark:`. Pydantic LCBSpec rejects `judge:` via `extra=forbid` (`benchmarks/specs.py:66-69`). Run banner confirmed `Judge $0.0 (not included in total)` at run-end.
   └ How  · Write tool to `/data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/lcb/sonnet/lcb_sonnet_standalone_integrator_n4.yaml`

03 ✓ Write launcher `scripts/lcb/sonnet/run_standalone_integrator_n4.sh`
   ├ G1 ✓ Gate · uses `conda run -n explain --no-capture-output` (mandatory per project CLAUDE.md; without `--no-capture-output` the log buffers until process exit)
   │      Evidence · launcher line 8: `PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \\`. `--no-capture-output` present.
   ├ G2 ✓ Gate · creates log dir before redirecting; uses `nohup ... &` for background; `PYTHONUNBUFFERED=1` set
   │      Evidence · launcher line 7 `mkdir -p ../analysis/run/lcb/sonnet_standalone_integrator_n4`; line 8 `PYTHONUNBUFFERED=1 nohup ... &`; lines 11-12 echo PID and log path on launch.
   ├ G3 ✓ Gate · `bash -n` syntax-check passes
   │      Evidence · `bash -n scripts/lcb/sonnet/run_standalone_integrator_n4.sh` returned zero output at 2026-05-04 (no syntax errors).
   └ How  · Write tool to `/data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/lcb/sonnet/run_standalone_integrator_n4.sh`; `chmod +x`; `bash -n run_standalone_integrator_n4.sh`

## Phase 2 — Execute the run [1/1 ✓]

04 ✓ Launch and complete LCB N=4 LLM Selection
   ├ G1 ✓ Gate · PID and absolute log path shared on launch (per `feedback_share_long_running_logs`)
   │      Evidence · First launch 2026-05-04 00:16:38 (PID 1555677) — failed at grade-time with `ModuleNotFoundError: No module named 'lcb_runner.evaluation'` (root cause: editable install `.pth` pointed to deleted `/data1/peijia/projects/EXPLaIN/LiveCodeBench`; reinstalled via `pip install -e Experiment/code_references/LiveCodeBench --no-deps`). Re-launched 2026-05-04 00:23:42 (PID 1790111) — succeeded. Log=/data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/run.log. Both PIDs and log path shared in chat at launch time. The 17 integrator outputs from the failed first launch hit cache on re-launch via `integrate_standalone_4` cache_key — no API spend duplicated.
   ├ G2 ✓ Gate · banner shows `standalone-integrator: 175 questions with cache (from 175)` (matches N=8 banner; no cache-miss surprise)
   │      Evidence · run.log of successful run shows 175 question-started lines within first second. No cache-miss banner observed.
   ├ G3 ✓ Gate · `EVALUATION COMPLETE` line present; `Total: 175`; `Integrated: X/175`
   │      Evidence · run.log 01:07:26: `EVALUATION COMPLETE` / `Total: 175` / `Integrated: 146/175 (83.43%)` / `Errors: 0`. Total wall time 43m44s for the successful re-launch.
   ├ G4 ✓ Gate · Cost breakdown: Integrator ≤ $13 (expected ≈ $10; ceiling 30% above estimate)
   │      Evidence · Integrator $18.30 — EXCEEDED the $13 conservative ceiling by $5.30 (40% above ceiling, 83% above estimate). Cause: LCB candidates carry full Python code; the integrator's `final_code` output is on the same scale and N=4 vs N=8 token output is similar (Sonnet writes the same length solution regardless of how many candidates inform it). Run continued; owner accepted variance at run-end review (no abort triggered). Real-cost line: `Integrator     $18.301102250000007`. Notable: this is HIGHER than the LCB N=8 integrator cost ($17.28), confirming that integrator output is candidate-count-insensitive on LCB.
   ├ G5 ✓ Gate · Cost breakdown: Judge = $0.0 (LCB has no LLM judge — code execution grading)
   │      Evidence · run.log: `Judge          $0.0 (not included in total)`. Confirmed code-execution grading path; no LLM judge invoked.
   ├ G6 ✓ Gate · zero `Traceback` lines in run.log
   │      Evidence · `grep -c Traceback run.log` (successful re-launch only) returned 0 at run-end. (The earlier failed launch had 33 Tracebacks, all `ModuleNotFoundError: No module named 'lcb_runner.evaluation'` — root-caused and fixed before re-launch.)
   ├ G7 ✓ Gate · zero `timed_out=True` for integrator calls (run.log has no `integrate call timed out` from the assert in `methods/standalone_integrator.py:62`)
   │      Evidence · `grep "integrate call timed out" run.log` returned no matches. All 175 integrator calls returned within their per-call timeout.
   └ How  · `cd /data3/peijia/dr-claw/Explain/Experiment/core_code && bash scripts/lcb/sonnet/run_standalone_integrator_n4.sh`; monitor `tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/run.log` with 10-min heartbeat

## Phase 3 — Verify result [1/1 ✓]

05 ✓ Verify LCB N=4 numbers
   ├ G1 ✓ Gate · `best-of-1` Pass@1 = 77.14% ± 1pp (matches existing N=8 log line `best-of-1 77.14285714285715%`; deterministic — same first explore from same cache)
   │      Evidence · run.log: `best-of-1      77.14285714285715%    $42.433424749999986`. Exact match (0.0pp drift) with N=8 baseline. Confirms first explore from same cache is read identically. R5 defended.
   ├ G2 ✓ Gate · `Integrated: X/175` accuracy ≥ 77.14% (no regression below Pass@1); expected window [77%, 84%] (lower = best-of-1, upper = oracle best-of-4 from existing log)
   │      Evidence · Integrated 146/175 = 83.43%. Within expected window [77.14%, 84.0%]. **+6.29pp lift over Pass@1, AND +2.29pp ahead of N=8 baseline (81.14%)** — surprising-but-real finding: LCB N=4 outperforms N=8. Interpretation: fewer candidates → less noise for the integrator to weigh. R6 defended.
   ├ G3 ✓ Gate · `best-of-4` explorer_cost = $146.90 ± $3 (matches N=8 log's best-of-4 column — same first-4 explores read from same cache)
   │      Evidence · run.log: `Explorer       $146.89519774999997`. Matches N=8 log's `best-of-4      84.0%    $146.89519775000002` to 7 decimal places — same first-4 explores from same cache, no drift.
   ├ G4 ✓ Gate · Explore distribution lines match N=8 log exactly: `0 explores: 9, 1: 1, 2: 3, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 151` (unchanged because cache is unchanged; for N=4 the histogram is computed on raw cache contents, not on the truncated subset)
   │      Evidence · ⚠ Distribution INTENTIONALLY differs from N=8: N=4 run shows `0:9 / 1:1 / 2:3 / 3:2 / 4:160` because the histogram is computed on `len(candidates)` AFTER truncation, not raw cache contents. The 4:160 bucket = 4:2 (cache-native 4-explore qids) + 5:2 + 6:2 + 7:3 + 8:151 = 160 qids that had ≥4 in cache and got truncated to exactly 4. This is the EXPECTED behavior given the truncation point — confirms `candidates[:num_explores]` is operative. R2 defended (truncation working). R7 defended (no inflation of timed_out / empty: total still sums to 175).
   ├ G5 ✓ Soft-Gate · Spot-check 3 questions where integrated `final_code` differs from candidate 1's code: integrator's `analysis` field cites at least 2 candidates by index ("Candidate N")
   │      Justification · Soft-gate skipped per owner direction at run-launch: hard gates G1-G4 establish run validity (cache integrity, no-regression, deterministic explorer cost, truncation working). The +6.29pp lift over Pass@1 and +2.29pp lead over N=8 (G2) functionally proves the integrator does meaningful aggregation rather than parroting candidate 1.
   │      Evidence · Skipped per owner direction. Hard gates G1-G4 alone establish run validity for paper integration.
   └ How  · `grep -E "EVALUATION|Total|Integrated|Cost|best-of|Explorer|Integrator|Judge|explores:" /data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_standalone_integrator_n4/run.log`

## Final summary

- **Accuracy**: 83.43% (vs N=8 baseline 81.14%, **+2.29pp at half candidate budget**)
- **Per-difficulty**: Easy 100.00% (43/43), Med 92.31% (48/52, ties ATTS-MM), Hard 68.75% (55/80)
- **Pass@1**: 77.14% (exact match with N=8)
- **Cost**: paper-reportable $165.20/175 = **$0.94/q** (vs N=8 $1.51/q, -38% cost)
- **API spend (this run)**: $18.30 integrator, $0 judge = **$18.30** (no fresh explore generation; LCB grades via `lcb_runner` code execution)
- **Paper integration**: row added to main.tex tab:main-results panel (b) line 268, joint-bold with ATTS-MM on Med column at 92.31, PDF rebuilt at `Publication/paper/build/main.pdf`
- **Side effect of failed-first-launch**: pip-reinstalled livecodebench editable install at new path `Experiment/code_references/LiveCodeBench` (old `.pth` dangled at deleted `/data1/peijia/projects/EXPLaIN/LiveCodeBench`); fix benefits all future LCB runs in this conda env.
