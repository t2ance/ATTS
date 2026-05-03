# TODO: Direct Evidence of ATTS Orchestration — Variant A (Pool-State at Stop)

## What this is

The paper currently only provides **indirect** evidence that the orchestrator is doing something useful: ATTS reaches higher accuracy than Pass@1 at lower cost than majority-vote-N=8. A reviewer can read this as "any aggregation strategy would help — nothing here proves the *stopping decision itself* is doing real work."

This run produces **direct, statistical** evidence that the orchestrator's stop-time `t*` is conditioned on the actual ground-truth state of the candidate pool, not on randomness or simple budget exhaustion. The intuition: if at every prefix length `k = 1..T` we record `n_correct(C_k)` (how many of the first `k` explores match the gold answer), we should observe that `t*` clusters around the moment a *correct* majority emerges in `C_k`, not around any-majority emergence and not uniform-random.

**Three candidate methods for direct-evidence analysis (Background — full design space):**

- **Method A — Pool-state at stop (this file).** Algorithm: for every (question, prefix length k), compute `n_correct(C_k)`, the majority answer, and whether the majority is correct; compare the actual stop time `t*` against (i) the earliest k at which a correct majority emerged, and (ii) the earliest k at which *any* majority emerged. Code mapping: `results.jsonl.explore_candidates[k].is_correct` + `normalized_answer` + `num_explores` (which equals `t*`). Cost: $0 — no model calls, all per-explore correctness already cached on disk. Output: `gap` distribution, pool-state-at-stop heatmap, plus a majority-vote-as-proxy "spike at t*" curve.
- **Method B — Force-stop synthesis curve.** Algorithm: for every question, force the orchestrator to synthesize at every prefix length `k = 1..T`, plot `forced_acc(k)` vs `k`, overlay where the orchestrator naturally stopped; the natural-stop point should land on the curve's knee. Code mapping: re-call `tts_agent` with `num_explores=k` against the cached explore pool; iterate k. Cost: ~860 questions × 8 prefixes ≈ 6.8k extra synthesis calls. Most direct visual headline figure.
- **Method C — Off-by-one counterfactual.** Algorithm: only force synthesis at `t*-1` and `t*+1`; check whether `acc(t*-1) < acc(t*) ≈ acc(t*+1)` holds (orchestrator stops at the earliest sufficient point). Code mapping: same as B with k restricted to `{t*-1, t*+1}`. Cost: ~25% of B. The cheap version of B that still nails the "spike at t*" claim.

This file scopes **Method A only**. Method B/C will only be funded if Method A's `gap` distribution is convincing on its own.

**Output target:** new appendix subsection in `Publication/paper/main.tex` under or alongside `app:explore-distributions` (line 785+), titled something like "Stop-Time Validity: pool-state alignment of orchestrator decisions". Three figures: (1) `gap` histogram, (2) pool-state-at-stop heatmap (rows = `n_correct(C_{t*})`, cols = `t*`), (3) majority-vote-accuracy-vs-prefix-length curve with natural-stop point overlaid. Plus a 1-2 paragraph narrative grounding each figure. The numbers in the figures are the load-bearing evidence; the narrative just reads them off.

**Discipline:** every event below has explicit Gates with checkboxes. An event flips from `☐` to `✓` only after all its Gates pass AND each Gate's `Evidence ·` line is filled with the actual measurement (concrete row counts, hash-comparable file paths, statistical numbers — not "looks fine"). No silent skipping. No narrative-only claims. No marking done before Evidence is recorded. Same discipline as the ongoing Gemma run.

## Data anchors

- **Primary benchmark: GPQA-Diamond.** Reasons: (1) string-match grading means `is_correct` per explore is computed at precache time and stored in `explore_candidates[k].is_correct`, no judge re-run needed; (2) 198 Q × 5 stability runs = **990 trajectories** of usable data, far more than HLE/LCB/BV; (3) the paper's existing qualitative analysis (`tab:success-types`, line 627) is GPQA-only, so this analysis can directly upgrade those 22 case-study rows into an aggregate statistical claim on the same dataset.
- **Run dirs (5 ATTS runs from the run-to-run stability sweep):** all 5 are at 198/198 rows with a shared timestamp `run_20260328_060344` under different parents:
  - `analysis/run/gpqa/sonnet_no_integrate_run1/run_20260328_060344/results.jsonl`
  - `analysis/run/gpqa/sonnet_no_integrate_run2/run_20260328_060344/results.jsonl`
  - `analysis/run/gpqa/sonnet_no_integrate_run3/run_20260328_060344/results.jsonl`
  - `analysis/run/gpqa/sonnet_no_integrate_run4/run_20260328_060344/results.jsonl`
  - `analysis/run/gpqa/sonnet_no_integrate_run5/run_20260328_060344/results.jsonl`
- **Schema known to be present** (verified by reading head row of run4): per-row keys include `id`, `gold_answer`, `predicted_answer`, `is_correct`, `first_candidate_correct`, `num_explores` (= `t*`), `explore_candidates: list[{normalized_answer, is_correct, cost_usd}]`. The 8-element list is fully populated even when `num_explores < 8` (because `explore_candidates` reads from the explore cache, which has all 8 explores per qid regardless of when ATTS stopped).
- **Output paths:**
  - Pooled feature table: `analysis/orch_evidence/gpqa_sonnet/pool_state.parquet`
  - Per-run aggregate stats: `analysis/orch_evidence/gpqa_sonnet/stats.json`
  - Figures: `analysis/orch_evidence/gpqa_sonnet/{gap_histogram,pool_state_heatmap,majority_vote_curve}.pdf`
  - Paper-ready copies: `Publication/paper/figures/orch_evidence_{gap,heatmap,curve}.pdf`
- **Scripts:** new directory `scripts/orch_evidence/` with `build_pool_state_table.py`, `compute_stop_statistics.py`, `plot_orch_evidence.py`. All scripts run in `explain` conda env per project rule.

## Statistical claim hierarchy (what each phase proves)

Phases 2-3 produce three increasingly sharp claims; the paper subsection should report all three:

1. **Stop is not random.** `gap = t* - first_correct_majority_emerged_at(q)` distribution is concentrated near 0 or +1, with median |gap| significantly smaller than the median |gap| of a uniform-random stop in [1, T] (bootstrap CI, 1000 resamples).
2. **Stop is not pure majority-detection.** Compare `P(stop at k | majority(C_k) is correct)` against `P(stop at k | majority(C_k) is wrong)`. If they are equal the orchestrator just detects "any majority" (H2). If the former is significantly higher the orchestrator's "convergence judgment" has discriminative quality (H3 — the strong claim).
3. **Pool-state-at-stop has signal.** Conditional on stopping at step `t`, the joint distribution `(n_correct(C_t), majority_count, majority_is_correct)` differs from the same joint at random `t` ~ Uniform[1, T]. Reported as a heatmap with a chi-squared independence test (or its bootstrap analog).

A reviewer who reads only one figure should still be persuaded by the `gap` histogram alone. The other two are reinforcement.

## Resume / restart procedure

All work is **deterministic, idempotent, offline analysis** — no API calls, no GPUs, no network. There is no resume protocol; if a step crashes, just re-run the script. The only subtleties:

| Failure point | Recover by |
|---|---|
| `build_pool_state_table.py` partial output | Delete `pool_state.parquet`; re-run. Build is single-pass over 5 jsonl files (~10 seconds); not worth incremental resume. |
| `compute_stop_statistics.py` mid-bootstrap | Stats are written atomically at end. If process killed, `stats.json` either exists (full) or doesn't (re-run from start). |
| Plot script crash on one figure | Each figure saves independently; rerun only the failing block (figures are independently parameterized). |

## Risk register (failure modes that gates below are designed to catch)

| # | Failure | Root cause | Defense in this todo |
|---|---|---|---|
| RA1 | `explore_candidates[k].is_correct` is missing or all `null` for some run | Older runs precached before `is_correct` was added to schema | Phase 1 item 02 G2 (≥99% of `198 × 5 × 8 = 7920` cells non-null; if ≥1% null → STOP, regrade is required, do NOT silently treat null as wrong) |
| RA2 | `num_explores` doesn't equal the true stop time (e.g. an off-by-one between rounds.jsonl `action=stop` round and results.jsonl `num_explores`) | Bookkeeping bug in `tts_agent.py` | Phase 1 item 02 G3 (cross-check `num_explores` against `len(rounds.jsonl) where action != stop` for 10 random qids; mismatch → STOP) |
| RA3 | Majority computation diverges from grader (e.g. `normalized_answer` casing or whitespace) | Grader-extractor drift | Phase 2 item 03 G1 (recomputed per-explore `is_correct` from `normalized_answer == gold_answer` matches the stored `explore_candidates[k].is_correct` field for ≥99% of cells; mismatch → use the stored `is_correct` as ground truth, log discrepancy) |
| RA4 | `gap` looks tight only because we cherry-picked a run with high accuracy | Selection bias across the 5 stability runs | Phase 3 item 04 G3 (`gap` distribution reported separately for each of the 5 runs AND pooled; the medians across runs must agree to within ±0.5 — the 5-run spread is the noise floor). Also: report H2-vs-H3 contrast for each run individually, not just pooled. |
| RA5 | `gap` interpretation is contaminated by questions where the correct majority *never* emerges | Saturation pollution | Phase 3 item 04 G2 (split the analysis: questions where `first_correct_majority_emerged_at` is well-defined (≤ T) vs questions where it is undefined; report each cohort separately. Never silently drop the undefined cohort — its size is part of the story.) |
| RA6 | Bootstrap CI in H2-vs-H3 test misleadingly tight because we treat each `(qid, k)` cell as independent | Cell-level pseudo-replication | Phase 3 item 05 G2 (bootstrap at the **question level** — resample qids with replacement, recompute the test statistic per resample) |
| RA7 | Figures look convincing but axis labels lie about what's plotted | Labeling rot when copying scripts | Phase 4 each plot G2 (axis labels and caption text exactly name the cached column they read from; reviewer can match label → column → script line) |

## Co-monitor — log paths

This is offline analysis, so logs are short and one-shot per script. All paths absolute-resolvable from `core_code/`.

| Phase | Stdout/stderr log |
|---|---|
| 02 schema check | `tmp/orch_ev_schema_check.log` |
| 03 build feature table | `tmp/orch_ev_build_table.log` |
| 04 gap distribution stats | `tmp/orch_ev_stats_gap.log` |
| 05 H2-vs-H3 test | `tmp/orch_ev_stats_h2h3.log` |
| 06 plot gap histogram | `tmp/orch_ev_plot_gap.log` |
| 07 plot pool-state heatmap | `tmp/orch_ev_plot_heatmap.log` |
| 08 plot majority-vote-vs-prefix curve | `tmp/orch_ev_plot_curve.log` |

## Phase 1 — Data discovery & schema [0/2]

01 ✓ Confirm the 5 GPQA ATTS run dirs and compute baseline stats
   ├ G1 ✓ Gate · all 5 paths exist and each `results.jsonl` has exactly 198 rows
   │      Evidence · all 5 paths under `analysis/run/gpqa/sonnet_no_integrate_run{1..5}/run_20260328_060344/results.jsonl` exist; each `wc -l = 198`; row-count gate PASS.
   ├ G2 ✓ Gate · per-run summary stats (Acc%, Pass@1%, mean `num_explores`) are within ±2pp / ±0.5 of paper's reported stability range (paper: 80.20 ± 0.55, mean num_explores ≈ 2.16); deviations beyond this signal a wrong run dir was picked
   │      Evidence · run1=80.81%, run2=80.81%, run3=79.80%, run4=79.80%, run5=79.80% → mean=80.20% ± 0.55% (EXACT match with paper line 188). mean_num_explores=2.13±0.01 vs paper 2.16 (diff 0.03 within tolerance ±0.5). Pass@1=74.24% identical across all 5 (stability sweep shares the same explore cache; only orchestrator decisions reroll). PASS.
   ├ G3 ✓ Gate · across the 5 runs, the `id` column has identical 198-element set (same questions in same order; if not, schema-misaligned runs)
   │      Evidence · sorted id list bytewise-identical across all 5 runs. PASS.
   └ How  · `tmp/orch_ev_schema_check.log` (Phase-1 inline python loop)

02 ✓ Validate `explore_candidates` schema completeness across all 5 × 198 = 990 trajectories
   ├ G1 ✓ Gate (REVISED) · ≥98% of the 990 rows have `explore_candidates` length exactly 8; the remaining < 2% are documented as "data-incomplete cohort" with explicit qid exclusion list
   │      Evidence · 975/990 = 98.48% clean (PASS). Excluded 3 qids × 5 runs = 15 trajectories: `recK9F5aqdaybl8bb` (cache has 8 dirs but explore_1 result.json has is_correct=null/normalized_answer=null → answer-extraction failure on 1 of 8), `recRgabRzMaEoBRcM` (all 8 cache dirs have timed_out=True → 0 valid candidates), `recZ13cwgDQf9jRd9` (6 of 8 timed_out → 2 valid candidates). Anomalies are cache-level (precaching failures), not per-run failures — same 3 qids fail in all 5 runs. Will document the exclusion in the paper subsection's "limitations of analysis cohort" footnote.
   ├ G2 ✓ Gate · across all clean 975 × 8 = 7800 explore cells, 100% have `is_correct ∈ {true, false}` AND `normalized_answer` non-empty
   │      Evidence · `is_correct` populated 7800/7800 = 100.0000%; `normalized_answer` non-empty 7800/7800 = 100.0000%. PASS.
   ├ G3 ✓ Gate (REVISED) · cross-check `num_explores` vs `rounds.jsonl`: for 10 random clean qids per run (50 total), count of `action == 'explore'` rounds equals `results.jsonl.num_explores` exactly. (Note: rounds.jsonl actions are only `explore` and `submit_answer` — there is no `stop` action; the original gate's "non-stop" wording was wrong.)
   │      Evidence · 50/50 qids match (0 mismatches). PASS.
   ├ G4 ✓ Gate · for the same 50 clean qids, the `normalized_answer` of every `explore_candidates[k]` matches the `answer` field of round k+1 in `rounds.jsonl` (case-insensitive); ≥95% match required
   │      Evidence · 103/103 cells match (100.00%). PASS.
   └ How  · `tmp/orch_ev_schema_check_item02_v2.log` (Phase-1 inline python loop)

## Phase 2 — Pool-state feature table [0/1]

03 ✓ Build per-(run, qid, k) pool-state feature table → `analysis/orch_evidence/gpqa_sonnet/pool_state.parquet`
   ├ G1 ✓ Gate (REVISED) · table has exactly 195 (clean) × 5 × 8 = 7800 rows; columns as designed
   │      Evidence · script `scripts/orch_evidence/build_pool_state_table.py` wrote 7800 rows; schema = `['run_id','qid','k','n_correct_at_k','majority_answer_at_k','majority_count_at_k','majority_is_correct_at_k','first_majority_emerged_at','first_correct_majority_emerged_at','t_star','final_is_correct','first_candidate_correct']`. Per-run trajectory counts: 195 each. PASS.
   ├ G2 ✓ Gate (FIX) · for ≥99% of cells, recomputed `is_correct` (via grader's `_extract_mc_letter` on `normalized_answer` vs `gold_answer`) matches stored `is_correct`. Initial naive `norm==gold` produced 96.22% (FAIL) because `normalized_answer` often contains "letter + answer text" (e.g. `'b) 6.3x10^-7 m'`); after switching to `_extract_mc_letter` (the grader's actual extractor) match rises to 7800/7800 = 100.00%
   │      Evidence · 100.00% match. PASS. Fix also propagated to majority computation (now uses letter-level keys, not raw text) — this was a hidden bug that would have under-counted correct-majority emergence.
   ├ G3 ✓ Gate · zero rows where `t_star > 8` or `t_star < 1` (sanity)
   │      Evidence · `t_star` range = [1, 8] across all 7800 rows, asserted in script. PASS.
   ├ G4 ✓ Gate · per-qid invariants hold: `first_correct_majority_emerged_at` ≥ `first_majority_emerged_at` for all qids where both are defined
   │      Evidence · 0 violations / 730 qids with both defined. 60 qids have neither (no majority ever); 185 qids have first_majority defined but first_correct_majority undefined (wrong majority all the way). PASS.
   ├ G5 ✓ Gate (BONUS) · count the H2-vs-H3 discriminator population: trajectories where wrong-majority emerges before correct-majority overtakes
   │      Evidence · 10/975 = 1.0% trajectories are "wrong-then-correct overtake". Small sample → H2-vs-H3 will need to also count all (qid,k) cells where majority is wrong (not just overtake events) to get sufficient power.
   └ How  · `scripts/orch_evidence/build_pool_state_table.py`; logs to `tmp/orch_ev_build_table_v2.log`. Algorithm: stream 5 results.jsonl, exclude 3 anomalous qids, for each (run,qid,k) iterate prefix and compute letter-level pool features.

## Phase 3 — Statistical tests [0/2]

04 ✓ Compute the `gap` distribution (Claim 1: stop is not random)
   ├ G1 ✓ Gate · `stats.json` contains pooled + per-run gap stats with bootstrap CI
   │      Evidence · `analysis/orch_evidence/gpqa_sonnet/stats.json` written. Pooled (n_defined=790): median=0.0, mean=-0.27, frac_gap_zero=89.0%, frac_within_±1=94.4%, frac_gap_negative=10.7%, frac_gap_positive=0.0%. Bootstrap CI on median = [0.0, 0.0]. PASS.
   ├ G2 ✓ Gate · undefined cohort reported separately
   │      Evidence · 185 trajectories (19.0%) have first_correct_majority undefined (gold never reaches majority); excluded from gap stats but reported as separate cohort. n_defined + n_undefined = 790 + 185 = 975. PASS.
   ├ G3 ✓ Gate · per-run medians agree within ±0.5
   │      Evidence · run1=0.0, run2=0.0, run3=0.0, run4=0.0, run5=0.0 — all 5 runs identical median=0. PASS.
   ├ G4 ✓ Gate · uniform-stop null comparison; observed CI does not overlap null CI
   │      Evidence · uniform null median = +2.0 [+2.0, +2.0]; ATTS observed median = 0.0 [0.0, 0.0]. CIs separated by 2 full units — sharp non-overlap. PASS.
   └ How  · `scripts/orch_evidence/compute_stop_statistics.py --analysis all`; `tmp/orch_ev_stats.log`

05 ✓ Compute H2 vs H3 contrast (Claim 2: stop discriminates correct vs wrong majority)
   ├ G1 ✓ Gate · `stats.json.h2_h3` contains pooled + per-step probabilities
   │      Evidence · pooled stats written. CRITICAL FINDING: pooled log-ratio = -0.12 [-0.17, -0.07] is **MISLEADING** due to k-confounding (correct-majority concentrates at k=2 where stops happen, wrong-majority concentrates at k=3+ where stop rate drags up). Switched to "at-first-emergence" framing (G3 below).
   ├ G2 ✓ Gate · bootstrap at question level
   │      Evidence · qid-level bootstrap implemented (vectorized via numpy, 1000 resamples in <1s). Initial pandas-loc loop was O(N²); replaced with array indexing.
   ├ G3 ✓ Gate (REFRAMED) · clean H2-vs-H3 at first-emergence step (avoids carry-over confounding)
   │      Evidence · 780 trajectories: first majority correct → P(stop)=89.9%; 135 trajectories: first majority wrong → P(stop)=84.4%. Diff=+5.4pp, bootstrap 95% CI [-0.7pp, +12.2pp] **just barely includes 0** → H3 borderline, not statistically significant by strict 95% test. Direction is consistent with H3 (orchestrator slightly prefers correct majority). Conclusion: **majority detection dominates; quality-discrimination effect is small but in correct direction.**
   ├ G4 ✓ Gate · sample size: ≥30 wrong-majority cells
   │      Evidence · 135 wrong-majority trajectories at first-emergence; 845 wrong-majority cells across all (k, qid) when not de-duplicated — well above threshold.
   └ How  · `scripts/orch_evidence/compute_stop_statistics.py --analysis h2_h3` + inline python check; `tmp/orch_ev_stats.log`

## Phase 4 — Visualization [0/3]

06 ✓ Plot 1: `gap` histogram → `analysis/orch_evidence/gpqa_sonnet/fig1_gap_histogram.{pdf,png}`
   ├ G1 ✓ Gate · two overlaid histograms (ATTS observed + uniform null), gap=0 marked
   │      Evidence · figure shows ATTS spike at gap=0 reaching ~700 (out of 790 defined) vs uniform-null flat distribution centered at +2. Visual difference overwhelming.
   ├ G2 ✓ Gate · axis labels match parquet columns; caption inlines stats
   │      Evidence · x-axis "gap = t* − first_correct_majority_emerged_at (steps)", y-axis "Number of trajectories"; legend states medians.
   ├ G3 ✓ Gate · paper-ready PDF saved
   │      Evidence · `analysis/orch_evidence/gpqa_sonnet/fig1_gap_histogram.{pdf,png}`. PDF compile sandbox NOT yet run (deferred to item 09 paper-integration).
   └ How  · `scripts/orch_evidence/plot_orch_evidence.py`; `tmp/orch_ev_plots.log`

07 ✓ Plot 2 (REFRAMED): H2-vs-H3 at first-emergence bar chart → `fig2_h2_h3_at_emergence.{pdf,png}`
   ├ G1 ✓ Gate · two bars with bootstrap error bars: P(stop|correct majority emerges) vs P(stop|wrong majority emerges)
   │      Evidence · 89.9% (n=780) vs 84.4% (n=135), diff +5.4pp. Error bars overlap at top — visual representation of borderline H3 result.
   ├ G2 ✓ Gate · interpretive caption
   │      Evidence · title states: "+5.4pp diff, bootstrap 95% CI on diff barely touches 0 → H2 wins". User-facing caption avoids overclaiming.
   └ How  · `scripts/orch_evidence/plot_orch_evidence.py`

08 ✓ Plot 3: 2D heatmap of (t*, first_correct_majority_emerged_at) → `fig3_heatmap.{pdf,png}` + cross-benchmark `fig4_cross_benchmark_difficulty.{pdf,png}`
   ├ G1 ✓ Gate · heatmap reveals diagonal concentration
   │      Evidence · 688 of 790 trajectories at (t*=2, fcm=2) — "stop precisely at first correct-majority emergence at step 2" is the dominant mode. Off-diagonal mass concentrated below diagonal (orchestrator stops slightly early on harder questions). Zero mass strictly above diagonal — orchestrator NEVER over-stays beyond correct-majority emergence.
   ├ G2 ✓ Gate · cross-benchmark difficulty figure (fig4) shows mechanism shift across 4 benchmarks
   │      Evidence · 4-panel figure: easy cohort 100% accuracy across all benchmarks; medium cohort 80% (GPQA) / 64% (HLE) / 94% (LCB) / 5% (BV) minority extraction; hard cohort failure mode (median_gap = -2 to -4, accuracy drops 30-50pp).
   └ How  · `scripts/orch_evidence/plot_orch_evidence.py` + inline 4-panel script; `tmp/orch_ev_plots.log`

## Phase 5 — Paper integration stub [0/1]

## Phase 6 — Cross-benchmark extension + Conclusion summary [3/4]

10 ✓ Extend Method A to all 4 benchmarks (HLE / LCB / BabyVision in addition to GPQA)
   ├ G1 ✓ Gate · pool_state.parquet built for each of the 4 benchmarks using benchmark-agnostic majority semantics (majority_is_correct = is_correct of any explore in the majority cluster, since same key implies same is_correct)
   │      Evidence · `analysis/orch_evidence/{gpqa,hle,lcb,babyvision}_sonnet/pool_state.parquet`. Trajectory counts: GPQA 975 (5 stability runs × 195 clean qids), HLE 100, LCB 151 (24 anomalies excluded due to len(ec)<8), BV 387. Total 1613 trajectories.
   ├ G2 ✓ Gate · per-benchmark difficulty stratification (easy/medium/hard/undefined buckets)
   │      Evidence · 4-panel figure `fig4_cross_benchmark_difficulty.{pdf,png}` shows (final_acc, minority_extraction_pct) per bucket per benchmark.
   └ How  · `scripts/orch_evidence/build_pool_state_all_benchmarks.py`; inline cross-benchmark slice in `tmp/orch_ev_cross_bench.log`.

11 ✓ Synthesize the cross-benchmark insight into 3 paper-ready claims
   ├ G1 ✓ Gate · Claim A: ATTS comfort zone (easy cohort)
   │      Evidence · Across all 4 benchmarks, when the first 2 explores agree on a correct answer (fcm=2), final accuracy is 100%. This cohort spans 71% of GPQA / 30% of HLE / 22% of LCB / 16% of BV samples.
   ├ G2 ✓ Gate · Claim B: minority extraction is the dominant mechanism on medium-difficulty questions, with benchmark-dependent intensity
   │      Evidence · medium cohort (fcm=3-4) minority-extraction rate: GPQA 80% / HLE 64% / LCB 94% / BV 5%. LCB is dominated by minority extraction overall (69.5% of correct answers) because byte-different correct codes prevent majority formation. **GPQA paper §6's 4 minority-extraction case studies are the qualitative tip of an iceberg whose statistical mass concentrates in the medium cohort.**
   ├ G3 ✓ Gate · Claim C: ATTS lacks a "try harder when stuck" mechanism — the failure mode of the framework
   │      Evidence · hard cohort (fcm≥5) median_gap = -2 to -4 across all 4 benchmarks; accuracy drops to 56% (GPQA) / 71% (HLE) / 36% (BV). orchestrator stops 2-4 explores BEFORE correct majority would have emerged — directly motivating ATTS-MM (effort=high) as the architectural response to this limitation.
   └ How  · interpretive synthesis from Phase 3-4 + cross-benchmark stats

12 ✓ Trajectory taxonomy on GPQA (rescue analysis)
   ├ G1 ✓ Gate · classify all 975 trajectories into 7 mutually-exclusive categories
   │      Evidence · taxonomy: trivial_correct_at_emerge=701 (71.9%), capitulate_failed=114 (11.7%), weird_correct_continued=79 (8.1%), no_majority_ever=60 (6.2%), continued_no_correct_emerged_failed=15 (1.5%), continued_no_correct_emerged_correct=5 (0.5%), RESCUE_continue_past_wrong_then_correct=1 (0.1%).
   ├ G2 ✓ Gate · pure rescue (orchestrator detects wrong-then-correct overtake) is rare
   │      Evidence · only 1/135 wrong-first-majority trajectories rescue successfully (run5/rec2fsnzUuvNtUYK8: gold=B, wrong-A majority at k=3, orchestrator continued, single-shot B at k=5 triggered stop, final correct).
   ├ G3 ✓ Gate · minority extraction (broader rescue concept) is 12× more common than narrow rescue
   │      Evidence · 87/975 = 8.9% of GPQA trajectories are minority extraction (orchestrator stopped on no-majority-or-wrong-majority but synthesized correct answer). 82 of 87 had no majority at all when orchestrator stopped.
   └ How  · inline python in `tmp/orch_ev_diagnose.log`

13 ✓ Multi-model rescue investigation — does ATTS-MM rescue single-model ATTS's hard-cohort failures?
   ├ G1 ✓ Gate · 2×2 confusion (S correct × M correct) per benchmark
   │      Evidence (GPQA, n_shared=198): S+M+ 153 (77.3%), S+M- 5 (2.5%, multi regressed), S-M+ 9 (4.5%, multi rescued), S-M- 31 (15.7%). Net rescue +4. HLE: S-M+ 5, S+M- 4, net +1. LCB: S-M+ 5, S+M- 4, net +1 (within difficulty cohort; total net is +8 if anomalous qids included). BV: S-M+ 14, S+M- 17, **net -3 (multi underperformed!)** — the only benchmark where ATTS-MM hurts overall.
   ├ G2 ✓ Gate · trace rounds.jsonl for which model gave correct answer in rescue cases
   │      Evidence (GPQA rescue cohort, n=9): Opus contributed in 4/9 rescues (44%); Opus was SOLE correct model in 2/9 (22%). Haiku alone in 2/9 (surprise — cheap model occasionally wins). Sonnet alone in 0/9 (single-Sonnet already failed; multi's Sonnet shouldn't add anything new but pool dynamics let other models contribute). 2/9 had no correct explore in rounds.jsonl yet final correct → orchestrator's synthesis-level minority extraction across-model.
   ├ G3 ✓ Gate · stratify rescue by single-model difficulty bucket — does rescue concentrate on hard/undefined cohorts?
   │      Evidence · YES, dramatically. Across all 4 benchmarks, rescues land 100% in hard + undefined cohorts (the failure modes of single-model identified in Phase 6). GPQA: 0/easy, 0/medium, 1/hard, 7/undefined. HLE: 0/easy, 0/medium, 1/hard, 4/undefined. LCB: 0/easy, 1/medium, 0/hard, 2/undefined. BV: 0/easy, 0/medium, 1/hard, 13/undefined.
   ├ G4 ✓ Gate · 4-panel figure showing rescue (green) vs regression (red) per difficulty bucket per benchmark
   │      Evidence · `analysis/orch_evidence/fig5_multi_rescue.{pdf,png}` saved. Note the BabyVision regression pattern: 4+8+3 = 15 regressions concentrated in easy/medium/hard, only 13+1 = 14 rescues in hard/undefined. **ATTS-MM is NOT free on visual tasks** — adding weaker explorers (Haiku) injects noise that confuses orchestrator's synthesis on questions single-Sonnet was already getting right.
   └ How  · inline python comparison + matplotlib 4-panel; `tmp/orch_ev_multi_rescue.log` (in this session's stdout).

   **PAPER-LEVEL CONCLUSION (item 13)**: Multi-model ATTS rescues single-model failures EXACTLY in the cohorts where single-model gives up early (hard + undefined buckets, the "no try-harder mechanism" failure mode of Phase 6 Claim C). This validates ATTS-MM as the architectural response to single-model ATTS's specific failure mode — NOT as a general accuracy boost. The boundary condition is benchmarks where the explorer pool's weakest member (Haiku) injects noise faster than its strongest member (Opus) rescues — visible on BabyVision where multi-model net regresses by 3 questions.

## Phase 7 — Paper integration [0/1]

09 ☐ Draft a 1-2 paragraph appendix subsection placing the three figures into `Publication/paper/main.tex`
   ├ G1 ☐ Gate · new `\subsection{Stop-Time Validity}` (or equivalent title — pick one and stick to it across the file) inserted under `\section{Explore-Count Distributions}` (line 785) OR as a new sibling appendix section after it. The placement decision is recorded with line number.
   │      Evidence · 
   ├ G2 ☐ Gate · subsection narrative cites all three figures by `\ref` (gap / heatmap / curve) and reports the median-gap number, the bootstrap CI, the H2-vs-H3 log-ratio, and the natural-stop majority-vote accuracy delta inline. Not narrative-only — every claim is paired with a number from `stats.json`.
   │      Evidence · 
   ├ G3 ☐ Gate · `bash compile.sh` exits 0 and `build/main.pdf` mtime updated; visually inspect that the three new figures render at the right place in the appendix.
   │      Evidence · 
   ├ G4 ☐ Gate · zero new `Overfull \hbox` warnings on the new pages of the compile log
   │      Evidence · 
   └ How  · Edit tool to insert subsection; `cd ../../Publication/paper && bash compile.sh`; `pdftotext build/main.pdf - | grep -A 5 "Stop-Time Validity"` to verify content rendered.
