# Socratic Self-Refine TODO (STYLE WHEN MAINTAIN THIS DOCS: MINIMALIST STYLE)

## Per-benchmark progress

| Benchmark | Done / Total | Status | Log path |
|---|---|---|---|
| hle | 100 / 100 | done (gold+text_only, integrated=45.0%, $0.90/q) | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_socratic_self_refine/socratic_self_refine.log |
| gpqa | 198 / 198 | done 2026-04-30 04:24 UTC (Integrated 147/198 = 74.24%, $0.56/q). env transition mid-run: rows 1-127 base/Py 3.13, rows 128- explain/Py 3.11. Run dir `run_20260428_163751`. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/gpqa/sonnet_socratic_self_refine/socratic_self_refine.log |
| lcb | 175 / 175 | done 2026-04-30 05:25 UTC (Integrated 144/175 = 82.29%, $0.60/q). Run dir `run_20260428_163748`. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet_socratic_self_refine/socratic_self_refine.log |
| babyvision | 388 / 388 | done 2026-04-30 19:29 UTC (Integrated 82/388 = 21.13%, $0.58/q). Run dir `run_20260428_075116`. Resume from 2026-04-29 22:34 UTC stop completed cleanly under explain env. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/babyvision/sonnet_socratic_self_refine/socratic_self_refine.log |
| rbenchv | 2 / 157 | **deferred** to follow-up revision; main.tex Appendix~\ref{app:rbenchv} already discloses RBenchV bottom-group rows as TBD. Run dir kept at run_20260429_161247 for future resume. Script uses explain env — next launch via `bash scripts/rbenchv/sonnet/run_socratic_self_refine.sh`. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet_socratic_self_refine/socratic_self_refine.log |

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

- [x] Section 5.1 baselines paragraph: now says "seven baselines" with Socratic Self-Refine listed (2026-05-01).
- [x] Appendix implementation-details: Socratic Self-Refine description paragraph added between Self-Refine and Budget Forcing (2026-05-01).
- [x] Figure `explore_distribution_all`: BabyVision SSR panel filled; `INCOMPLETE_PANELS` set emptied; figure caption updated (2026-05-01).
- [x] HLE narrative paragraph: states that Socratic Self-Refine underperforms Self-Refine on HLE (45.00% vs 53.00% at 2.2x cost) (2026-05-01).

## Env discipline (2026-04-29)

- All experiment scripts must run in `explain` conda env via `conda run -n explain --no-capture-output python ...`. Mandate added to `CLAUDE.md`. 96 launcher .sh files updated.
- Migration completed 2026-04-29 22:34 UTC: lcb + babyvision stopped (group-SIGTERM on pgids 3519943/3520723, no SIGKILL needed; in-flight Claude SDK subprocesses 3352690/3421863 also reaped). Both run dirs preserved for resume on explain env.
- gpqa: prior PID 3519154 died at unknown UTC; restarted 2026-04-29 21:50 UTC on explain. Mid-run env split is now baked into the run (rows 1-127 base, rows 128- explain). Acceptable per the cache-key invariant (judge_model only); flag accuracy delta around row 128 in post-hoc analysis if it appears.
- **Risk if a job mid-flight is killed and resumed on explain:** package-version drift between base and explain may produce different explore/integrate outputs (judge cache invalidates correctly because judge_model is the cache key, but new explores will use explain's SDK version). Acceptable for resume; flag if accuracy delta on remaining questions looks anomalous.
