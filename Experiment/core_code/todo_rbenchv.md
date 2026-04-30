# RBench-V TODO (STYLE WHEN MAINTAIN THIS DOCS: MINIMALIST STYLE)

Pivot axis: **per-method progress under the rbenchv benchmark**. Mirror of paper
Table `tab:lb-rbenchv` bottom group (8 method rows × 7 columns). Cross-cuts the
per-method `todo_ssc.md` etc., which pivot the other way.

Dataset families (n totals from `main.tex` line 740): Math 176, Physics 157,
Counting 195, Games 275, Overall 803, w/o Math 627. Judge: Claude Haiku 4.5
(`benchmarks/rbenchv.py:40`). Precache target = 803 × 8 explores = 6424
`result.json` files under `analysis/cache/rbenchv/sonnet/`.

## Gating milestone: sonnet explore precache

| Step | Done / Total | Status | Log path |
|---|---|---|---|
| precache (Counting + Games) | 72 qids × 8 = 576 explores cached | partial; last log halted on rate-limit at qid 72 explore_4 (UTC 2026-04-29). 70 of 72 qids fully cached (8/8), 2 partial (4/8 and 5/8). | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet/precache.log |
| precache (Physics) | 213 / 1256 explores cached (≈27 qids fully + physics_26 mid-run) | **active** PID 3249630 on explain env (Py 3.11), restarted 2026-04-29 22:15 UTC after rate-limit reset. 1043 explores remain. Currently running physics_26 explore_7. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet/precache_physics.log |
| precache (Math) | 0 / 176 qids | not started. Launcher: `scripts/rbenchv/sonnet/run_precache_math.sh` (added 2026-04-29; mirrors physics with `category: Math`, `num_workers=1`). |  |
| precache (Counting) | 0 / 195 qids | not started. Launcher: `scripts/rbenchv/sonnet/run_precache_counting.sh` (added 2026-04-29; `category: Counting`). Note: 72 numeric-prefix qids in cache predate the qid-prefix change and may overlap with this family. |  |
| precache (Game) | 0 / 275 qids | not started. Launcher: `scripts/rbenchv/sonnet/run_precache_game.sh` (added 2026-04-29; `category: Game` — singular per dataset's `catagory` field, not "Games"). |  |
| **Total cache coverage** | **99 / 803 qids fully cached + 2 partial** | unique qids: 72 numeric + 27 `physics_*`. Method-row evaluation cannot start until at least one full family is cached. | `find analysis/cache/rbenchv/sonnet -name result.json \| wc -l` (currently 785 of 6424) |

## Per-method progress (paper Table tab:lb-rbenchv bottom group)

Each row corresponds to one row in `main.tex` line 759-766. Output columns
needed: Overall (n=803), w/o Math (n=627), Math (n=176), Physics (n=157),
Counting (n=195), Games (n=275), \$/q.

| Method | Done / Total | Status | Run dir |
|---|---|---|---|
| Pass@1 | 0 / 803 | blocked on precache. Reads explore_1 from cache; no separate launch. Compute via `analysis/parse_method_log.py` once cache complete. | (cache-derived) |
| Majority Voting | 0 / 803 | blocked on precache. Reads all 8 explores from cache; aggregator only. | (no run dir; reads cache) |
| Self-Refine | 0 / 803 | not started. Launcher exists: `scripts/rbenchv/sonnet/run_self_refine.sh`. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet_self_refine/ (empty) |
| Budget Forcing | 0 / 803 | not started. Launcher exists: `scripts/rbenchv/sonnet/run_budget_forcing.sh`. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet_budget_forcing/ (empty) |
| VisualPRM | 0 / 803 | not started. Launcher exists: `scripts/rbenchv/sonnet/run_visualprm_rerank.sh`. Reranks cached explores; depends on full precache. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet_visualprm_rerank/ (empty) |
| LLM Selection (N=8) | 0 / 803 | not started. Launcher: `scripts/rbenchv/sonnet/run_standalone_integrator.sh` (added 2026-04-29; method=`standalone-integrator`, num_workers=4). Reads cached explores; depends on full precache. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet_standalone_integrator/ (not yet created) |
| ATTS (\our{}) | 0 / 803 | not started. Launcher exists: `scripts/rbenchv/sonnet/run_delegated.sh` (this is the ATTS orchestrator entry, naming inherited from earlier sweeps). | (n/a) |
| ATTS-MM (\attsmm{}) | 0 / 803 | not started. No launcher yet — needs a multi-model variant. Verify whether multi-model cache (haiku + sonnet + opus) is even planned for rbenchv before scaffolding; current `cache/rbenchv/haiku/` has 72 result.json (9 qids × 8) and `cache/rbenchv/opus/` does not exist. |  |

Side experiment (NOT in paper Table 1, tracked under `todo_ssc.md` row 11):
Socratic Self-Refine 2/157 on Physics subset, run dir
`run_20260429_161247`, stopped 2026-04-29 16:30 UTC (env-discipline migration).

## Add-a-row checklist (when a method row finishes)

- [ ] Log has `EVALUATION COMPLETE` and `results.jsonl` rows == 803 (or filtered family total if running family-by-family).
- [ ] Compute the 7 cells per family from results.jsonl: Overall, w/o Math (= mean of Physics+Counting+Games weighted by n), Math, Physics, Counting, Games, \$/q. Confirm n adds up to 176/157/195/275 per family.
- [ ] Re-derive bolds in Table `tab:lb-rbenchv` bottom group (best Acc per column among the 8 our-eval rows; lowest \$/q among them). Top group bolds (cited from RBench-V paper) are fixed and must not be touched.
- [ ] Edit `main.tex` line 759-766: replace the `\textit{TBD}` cells in this method's row with computed numbers. Remove the deferred-evaluation paragraph (line 771-) only when ALL 8 rows are done.
- [ ] `cd ../../Publication/paper && bash compile.sh`. Inspect Table 1(d?), check no Overfull \hbox at `tabcolsep=3pt scriptsize`. If overflow appears, drop scriptsize → tiny BEFORE shrinking labels.
- [ ] If a row's number contradicts the "RBench-V is hard, image-emit gap" framing in line 730 / 772, surface it: do not silently fix the prose to match.
- [ ] Update this file: move row from "0 / 803" to "Y / 803, accuracy=...".

## Cross-doc gaps (will close as rows fill)

- [ ] `main.tex` line 771-774 deferred-evaluation paragraph says "has not yet been run". Delete this paragraph the moment all 8 rows are filled. Leaving the paragraph + filled numbers is contradictory.
- [ ] `main.tex` line 730 framing paragraph claims "image-emitting reasoning is part of RBench-V's official capability target but not part of \our{}'s current explorer interface". If \our{} ends up scoring above some of the cited zero-shot models on a non-Math family, this framing needs softening, not deletion.
- [ ] `scripts/plot_all_methods.py`: rbenchv is not currently in `BENCHMARKS`. Decide whether RBench-V appears in the cost-vs-accuracy main scatter or stays appendix-only. If main scatter, add `BENCHMARKS["rbenchv"]` + colors per method.
- [ ] `analysis/parse_method_log.py`: confirm it handles the rbenchv answer_type=hybrid (Math/Counting often multipleChoice, Physics/Games often exactMatch) — same dual-mode codepath that `babyvision.py:89` uses.

## Env discipline (2026-04-29)

- All experiment scripts must run in `explain` conda env via `conda run -n explain --no-capture-output python ...` (CLAUDE.md mandate). The 7 launcher .sh in `scripts/rbenchv/sonnet/` are already updated.
- Existing precache log `precache.log` was started in **base** env before the env-discipline mandate (per `todo_ssc.md` line 35). Resuming this precache via the updated `run_precache.sh` will switch to explain — cache validity is preserved by the cache key (which does not include env), but explore content may show subtle Python 3.13 → 3.11 SDK-version drift on freshly-generated explores. Acceptable: rbenchv judge is Haiku, not the explore output, and judge cache-key checks `judge_model` only.
- Do NOT delete `cache/rbenchv/sonnet/` to "start clean" — 785 explores already paid for. The next precache resume must keep this cache and only fill gaps.

## Risks worth calling out

- **Precache cost gradient**: 6424 - 785 = 5639 remaining explores at sonnet rates. Before resuming Math/Physics precache, get a per-explore cost estimate (`grep '\$' precache.log | tail -50 | awk` or read 10 cached `result.json` cost fields) to avoid surprise burn.
- **Family-by-family vs full**: running a method on the partial cache (Counting+Games only) gives a 470-question partial result usable for early signal, but the paper table requires per-family numbers including Math+Physics. Decide explicitly whether to publish partial-row numbers as an interim, or wait for full cache.
- **ATTS-MM path uncertainty**: cited in main.tex but no launcher and no haiku/opus cache. Either build the path or remove the row from the table.
