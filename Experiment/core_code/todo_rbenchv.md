# RBench-V TODO (STYLE WHEN MAINTAIN THIS DOCS: MINIMALIST STYLE)

Pivot axis: **per-method progress under the rbenchv benchmark**. Mirror of paper
Table `tab:lb-rbenchv` bottom group (8 method rows × 7 columns). Cross-cuts the
per-method `todo_ssc.md` etc., which pivot the other way.

Dataset families (n totals from `main.tex` line 740): Math 176, Physics 157,
Counting 195, Games 275, Overall 803, w/o Math 627. Judge: Claude Haiku 4.5
(`benchmarks/rbenchv.py:40`). Precache target = 803 × 8 explores = 6424
`result.json` files under `analysis/cache/rbenchv/sonnet/`.

## Gating milestone: sonnet explore precache

**STATUS 2026-05-01 03:10**: precache **DEAD**. Last process (Physics) crashed
2026-04-30 01:28 UTC with `Claude API error (authentication_failed): 403
permission_error — Account is no longer a member of the organization`. Re-tested
SDK auth at 2026-05-01 03:10 — still broken (`claude_agent_sdk._errors.ProcessError: Command failed with exit code 1`). **Cannot resume any precache until org/token is restored**; no precache process has run since 2026-04-30 01:28.

| Step | Done / Total | Status | Log path |
|---|---|---|---|
| precache (Counting + Games numeric qids) | 71 qids × 8 = 568 + 1 partial (qid 69: 4/8) = 572 result.json | halted 2026-04-29; pre-dates Physics crash. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet/precache.log |
| precache (Physics) | 30 qids × 8 = 240 + 1 partial (physics_30: 4/8) = 244 result.json | **dead** (auth_failed). 1012 of 1256 physics explores remain. | /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet/precache_physics.log |
| precache (Math) | 0 / 1408 explores | not started. Launcher: `scripts/rbenchv/sonnet/run_precache_math.sh`. |  |
| precache (Counting) | 0 / 1560 explores (separately from numeric-qid early run — qid namespace diverged) | not started. Launcher: `scripts/rbenchv/sonnet/run_precache_counting.sh`. |  |
| precache (Game) | 0 / 2200 explores | not started. Launcher: `scripts/rbenchv/sonnet/run_precache_game.sh` (`category: Game` singular). |  |
| **Total sonnet cache coverage** | **101 / 803 qids fully cached + 2 partial** | 71 numeric + 30 `physics_*`. 816 of 6424 result.json present (12.7%). | `find analysis/cache/rbenchv/sonnet -name result.json \| wc -l` |
| haiku side-cache | 9 qids × 8 = 72 result.json | unchanged since pre-2026-04-29; not the gating cache for the bottom-group rows. | `analysis/cache/rbenchv/haiku/` |

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

- **BLOCKER: Claude API auth (2026-04-30 → still broken 2026-05-01 03:10)**.
  Org membership lost; all rbenchv precache plus any new Sonnet/Haiku call
  fails with 403. The 4-benchmark Qwen3.6-FP8 sweep (HLE/GPQA/LCB/BabyVision)
  is unaffected because it runs `cache_only=True` against pre-existing caches —
  cost numbers there are billed from cached `cost_usd`, not new API calls.
  Restoring auth is the only way to fill the remaining 5608 sonnet explores.
- **Precache cost gradient**: 6424 - 816 = 5608 remaining explores at sonnet
  rates. Before resuming, sample `result.json` cost fields to project burn.
- **Family-by-family vs full**: 71 Counting+Games numeric qids are usable for
  early signal but qid namespace diverged when launchers switched to
  `category:`-prefixed qids — the new precache will likely re-cache under a
  different qid string and not collide with the old 71. Verify before relaunch.
- **ATTS-MM path uncertainty**: cited in main.tex but no launcher and no
  haiku/opus cache. Either build the path or remove the row from the table.
