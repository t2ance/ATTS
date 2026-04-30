# Socratic Self-Refine TODO (STYLE WHEN MAINTAIN THIS DOCS: MINIMALIST STYLE)

## Per-benchmark progress

| Benchmark | Done / Total | Status | Log path |
|---|---|---|---|
| hle | 100 / 100 | done (gold+text_only, integrated=45.0%, $0.90/q) | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_socratic_self_refine/socratic_self_refine.log |
| gpqa | 130 / 198 | done (PID 2961474, num_workers=1, restarted 2026-04-29 21:50 UTC after prior PID 3519154 died on base env). env transition mid-run: rows 1-127 produced under base/Py 3.13, rows 128- under explain/Py 3.11. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/gpqa/sonnet_socratic_self_refine/socratic_self_refine.log |
| lcb | 111 / 175 | **done** 2026-04-29 22:34 UTC by user (was PID 3519945, base env). Run dir kept at `run_20260428_163748` for resume. Relaunch via `bash scripts/lcb/sonnet/run_socratic_self_refine.sh` (now explain env). | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_socratic_self_refine/socratic_self_refine.log |
| babyvision | 177 / 388 | **stopped** 2026-04-29 22:34 UTC by user (was PID 3520725, base env). Run dir kept at `run_20260428_075116` for resume. Relaunch via `bash scripts/babyvision/sonnet/run_socratic_self_refine.sh` (now explain env). | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/babyvision/sonnet_socratic_self_refine/socratic_self_refine.log |
| rbenchv | 2 / 157 | stopped 2026-04-29 16:30 UTC by user (was PID 3521500, base env). Run dir kept at run_20260429_161247 for resume. Script now uses explain env — next launch via `bash scripts/rbenchv/sonnet/run_socratic_self_refine.sh` will run on explain. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet_socratic_self_refine/socratic_self_refine.log |

## Add-a-row checklist (when a benchmark finishes)

- [ ] Log has `EVALUATION COMPLETE` and `results.jsonl` rows == filtered total (or `num` cap).
- [ ] Acc = `Integrated: X/Y` from log (not `single`). Same regex `parse_method_log` uses.
- [ ] Cost = `avg $X.XX/question`, 2 decimals. HLE only: if method reads explore cache (grep `cache_dir` in `methods/<m>.py`), apply `_HLE_COST_CORRECTIONS` and add `% Corrected:` row note.
- [ ] Edit `main.tex` Table 1(<bench>): add row, recheck both bolds (best Acc, lowest \$/q).
- [ ] Edit `scripts/plot_all_methods.py`: append to `BENCHMARKS["<bench>"]["methods"]` + add color in `BASELINE_STYLES`.
- [ ] `python scripts/plot_all_methods.py` to refresh per-bench + main scatter PDFs.
- [ ] `cd ../../Publication/paper && bash compile.sh`. Inspect cell, legend, no new Overfull \hbox at the table.
- [ ] If new row creates a non-trivial comparison (e.g. Socratic 45 < Self-Refine 53 on HLE), add one sentence to the bench narrative; do not leave it unexplained.

## Cross-doc gaps still open

- [ ] Section 5.1 baselines paragraph: list still says "six baselines"; add Socratic Self-Refine + bump count.
- [ ] Appendix implementation-details: add a Socratic Self-Refine description paragraph alongside Self-Refine / Budget Forcing.
- [ ] Figure `explore_distribution_all`: histogram has ATTS vs Self-Refine only; add a Socratic row once at least 2 benches finish.
- [ ] HLE narrative paragraph: state that Socratic Self-Refine underperforms Self-Refine on HLE.

## Env discipline (2026-04-29)

- All experiment scripts must run in `explain` conda env via `conda run -n explain --no-capture-output python ...`. Mandate added to `CLAUDE.md`. 96 launcher .sh files updated.
- Migration completed 2026-04-29 22:34 UTC: lcb + babyvision stopped (group-SIGTERM on pgids 3519943/3520723, no SIGKILL needed; in-flight Claude SDK subprocesses 3352690/3421863 also reaped). Both run dirs preserved for resume on explain env.
- gpqa: prior PID 3519154 died at unknown UTC; restarted 2026-04-29 21:50 UTC on explain. Mid-run env split is now baked into the run (rows 1-127 base, rows 128- explain). Acceptable per the cache-key invariant (judge_model only); flag accuracy delta around row 128 in post-hoc analysis if it appears.
- **Risk if a job mid-flight is killed and resumed on explain:** package-version drift between base and explain may produce different explore/integrate outputs (judge cache invalidates correctly because judge_model is the cache key, but new explores will use explain's SDK version). Acceptable for resume; flag if accuracy delta on remaining questions looks anomalous.
