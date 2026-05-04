# TODO: Paper Anomaly Brainstorm — degradation mechanisms blocking submission

> Working dir: `/home/peijia/dr-claw/Explain/Experiment/core_code/`
> Paper: `../../Publication/paper/main.tex`
> Companion analysis (already done): `tmp/diff_atts_vs_baselines.md`, `todo_orch_evidence_A_pool_state.md`

## What this is

The current paper draft contains a number of numerical results that **violate the
naive monotonicity assumption** "more capability → better accuracy" or "more
compute → better accuracy". A reviewer who reads the tables in order will notice
these rows and ask "why?" — and the current narrative does not have a satisfying
answer for any of them. This file enumerates each anomaly with paper line
reference, hypothesizes the failure mechanism, and defines diagnostic Gates that
must pass before the row can stay (or be rewritten / dropped).

User directive (2026-05-04): "These anomalous values must be fixed before the
paper can be submitted. First brainstorm the full list of degradation
mechanisms; then for each one, define what evidence would explain the mechanism
and what the fix is."

The two starter items the user named:
1. **Augmentation that hurts** — when we add MORE to base ATTS (more models in
   the explorer pool, a separate integrator, higher effort), accuracy
   sometimes goes DOWN. Why?
2. **Effort that hurts** — `ATTS-MM (Low) → Med → High` is non-monotonic on
   3 of 4 benchmarks. Why?

Below: 14 anomalies organized into 6 phases. Each is a publication blocker
unless either (a) the mechanism is identified and the paper narrative is
rewritten to match, or (b) the row is dropped from the paper.

## Output target

This TODO produces three artifacts:
1. **Per-anomaly mechanism diagnosis** written to `tmp/anomaly_<id>.md`
   (one short markdown per anomaly with the diagnostic numbers).
2. **Paper edits** to `Publication/paper/main.tex` — every anomaly that
   stays gets a paragraph or table footnote that names its mechanism;
   every anomaly that is dropped gets removed cleanly with no orphan
   references.
3. **Aggregated supplementary appendix** `Publication/paper/appendix_anomalies.tex`
   — a single appendix subsection that collects the 14 mechanism findings
   into one place for reviewer consumption.

## Discipline

Every anomaly has Gates with checkboxes; flips ☐→✓ only after all Gates pass
AND each Gate's `Evidence ·` line is filled with concrete measurement (paper
line number, results.jsonl path + row count, qid + per-question delta — never
"looks fine"). No silent skipping. No narrative-only claims. No marking done
before Evidence is recorded.

## Anchors

- **Paper main results** — `Publication/paper/main.tex` line 200-350
  (`tab:main-results`). Six sub-tables.
- **Effort ablation** — `Publication/paper/main.tex` line 395-415
  (`tab:effort-ablation`).
- **Integrator ablation** — line 420-440 (`tab:integrator-ablation`).
- **Within-family orchestrator ablation** — line 458-478
  (`tab:orch-ablation`).
- **Cross-family orchestrator ablation** — line 521-541
  (`tab:orch-ablation-cross`).
- **Backbone ablation** — line 547-570 (`tab:backbone-ablation`).
- **Run dirs (verified)** — `Experiment/analysis/run/<bench>/<variant>/run_*/results.jsonl`.
- **Cache dirs** — `Experiment/analysis/cache/<bench>/<model>/`. Already paid for.

## Resume / restart procedure

All work in this TODO is offline analysis on existing results.jsonl files
EXCEPT items that require running a missing variant (currently only one:
BabyVision Opus orchestrator). For those: launch with `conda run -n explain`,
share PID + log path before continuing, monitor with 10-min heartbeat.

| Failure point | Recover by |
|---|---|
| Diagnostic script crash mid-analysis | Re-run; scripts are idempotent and read from results.jsonl only |
| Missing run that needs to be launched | Pre-flight verify cache dir exists; verify resume / cache-dirs banner shows N>0 already-cached; only then launch |
| Paper compile fails after edits | `cd Publication/paper && bash compile.sh` produces an error log; fix the syntax issue, re-compile; never commit a non-compiling main.tex |

## Risk register

| # | Failure | Root cause | Defense |
|---|---|---|---|
| R1 | Diagnostic finds no clear mechanism — anomaly is genuinely random / single-seed noise | Sample size too small for the effect | Item-level Gate requires running ≥1 additional seed before declaring "irreducible noise" |
| R2 | Mechanism is identified but the fix breaks another reported number | Cross-table coupling (e.g. fixing ATTS-MM stopping changes ATTS-MM Low/Med/High all together) | Each item lists which OTHER tables it touches; fix is gated on "all touched tables stay within ±1pp of current numbers" |
| R3 | We re-grade with a different judge and only some rows flip | Asymmetric judge upgrade (the SSR HLE case) | Item P6.01 forces uniform judge across all HLE methods; if any disagreement remains, all rows in that table re-graded together |
| R4 | Cost numbers shift after fix (e.g. fewer explores → less explorer cost charged) | Cost tied to consumption | Each item's Gate explicitly checks "$/q delta"; if Δ>5%, paper cost numbers must be updated |
| R5 | We exclude an anomaly row but forget to remove a `\ref` in narrative | Orphan reference | Item-level Gate runs `grep -n <ref>` after edit; zero references remain |

## Co-monitor — log paths

| Phase | Stdout/stderr |
|---|---|
| All offline analysis | `tmp/anom_<id>.log` |
| BV Opus-orch run (P1.04 only) | `tmp/anom_p1_04_bv_opus_orch.log` (PID + path shared on launch) |

---

## Phase 1 — Capability-monotonic violations [0/4]

01 ☐ **Anomaly: Opus-orch on GPQA underperforms Sonnet-orch by 4pp at SAME k\***
   Paper line 473-475 (`tab:orch-ablation`): GPQA-Diamond row says Opus 77.16% < Sonnet 80.81%.
   Verified from results.jsonl: 152/198 = 76.77% (paper rounds to 77.16; minor); k\* = 2.15 vs Sonnet 2.14 — same evidence pool.
   ├ G1 ☐ Gate · per-question diff between `gpqa/opus_orch/run_20260320_002145` and `gpqa/sonnet_no_integrate_run1/run_20260328_060344`. Count: ATTS-Sonnet correct, ATTS-Opus wrong on the SAME qid (Δwrong-side). Threshold: ≥6 such qids → mechanism is real, not single-row noise.
   │      Evidence · 
   ├ G2 ☐ Gate · For each Δwrong-side qid, classify the Opus prediction: (a) Opus picked a minority-cluster answer that no Sonnet explore endorsed, (b) Opus picked a wrong-majority answer that Sonnet correctly rejected, (c) Opus and Sonnet picked different members of the same cluster. Bucket sizes must be reported. Threshold: ≥50% in (a) or (b) → mechanism is "Opus over-reasons against the cached evidence".
   │      Evidence · 
   ├ G3 ☐ Gate · Run a second Opus-orch GPQA seed (same explore cache, different `seed:`). If second seed lands within ±2pp of 76.77 → effect is reproducible, NOT noise. If second seed lands at ≥80% → original was noise, retract the claim.
   │      Evidence · 
   ├ G4 ☐ Decision gate · Either (option A) keep the row and add a one-paragraph mechanism explanation citing the bucket sizes from G2; or (option B) drop the GPQA-Opus row from `tab:orch-ablation` and update the surrounding narrative (paper line 446-449 currently uses GPQA as the headline anomaly — text needs rewriting either way).
   │      Evidence · 
   └ How  · `tmp/anom_p1_01_gpqa_opus.py` for diff + bucketing; second-seed launch via `scripts/gpqa/opus_orch/...` reusing existing `cache_dir`.

02 ☐ **Anomaly: Haiku-orch on HLE BEATS Sonnet-orch (57% > 56%)**
   Paper line 473: Haiku 57 / Sonnet 56 / Opus 59.
   Cheaper model is BETTER than mid-tier. Reviewer will ask whether this is single-row noise or a real effect — we currently have no per-question evidence either way.
   ├ G1 ☐ Gate · per-question diff between `hle/haiku_orch/run_20260320_001806` and `hle/sonnet_no_integrate/run_20260319_003712`. Report Δ(Haiku correct, Sonnet wrong) and Δ(Sonnet correct, Haiku wrong). If both are 0-3 (small), the +1pp gap is noise; if Δwrong-side ≥6, the gap is real.
   │      Evidence · 
   ├ G2 ☐ Gate · Look at Haiku's k\* on HLE = 3.67 (vs Sonnet 2.39, Opus 2.73). Haiku over-explores. Test whether Haiku's gain comes from "compensating for low-capability single-shot synthesis by reading more cache". Concrete check: subset of HLE qids where k\*_haiku > k\*_sonnet AND Haiku correct AND Sonnet wrong. Threshold: ≥3 such qids → mechanism is "weak orchestrator + more reads beats strong orchestrator + fewer reads".
   │      Evidence · 
   ├ G3 ☐ Decision gate · If mechanism is real, paper line 446-449 should be expanded into a per-bench description of WHY the strongest model is not the best orchestrator. If noise, swap the row order so the table reads Haiku/Sonnet/Opus = monotonic 57/57/59 → row collapses cleanly.
   │      Evidence · 
   └ How  · `tmp/anom_p1_02_hle_haiku.py`; reuses cached cost data, no new run.

03 ☐ **Anomaly: ATTS-MM (Med) on BabyVision UNDERPERFORMS single-model ATTS by 0.78pp**
   Paper line 339, 345-346 (`tab:main-results d`): ATTS 23.20% / ATTS-MM 22.42%.
   Multi-model pool should be additive but is sub-additive on the visual benchmark.
   ├ G1 ☐ Gate · per-question diff between `babyvision/sonnet_no_integrate/run_20260319_021914` and `babyvision/multi_model_effort_medium/run_20260324_023252`. Report (ATTS correct, ATTS-MM wrong) count. Already done: tmp/diff_atts_vs_baselines.md shows **13 such qids**, with ATTS getting it right at k\*=2-5 and ATTS-MM tipping over after Haiku/Opus weaker visual answers were sampled.
   │      Evidence · 
   ├ G2 ☐ Gate · For each of the 13 qids, look at the multi-model trajectory: at the moment ATTS-MM committed to its (wrong) answer, what was the (Sonnet : Haiku : Opus) ratio of explores read? Threshold: ≥7/13 trajectories where ≥2 of the explores read were Haiku → mechanism confirmed as "weak-vision Haiku injection poisons synthesis".
   │      Evidence · 
   ├ G3 ☐ Gate · Cross-validate: re-run ATTS-MM on BV with the model_budgets shifted so Haiku is excluded (e.g. `{sonnet: 8, opus: 4, haiku: 0}`). Threshold: accuracy ≥ 23.20%. If yes → mechanism is causal.
   │      Evidence · 
   ├ G4 ☐ Decision gate · Either (option A) keep the row and ADD the "Haiku-injection-poisoning" mechanism narrative + recommend Haiku-excluded variant for BV in main results; or (option B) report ATTS-MM only on text benchmarks (HLE/LCB/GPQA) and drop the BV row from `tab:main-results d`.
   │      Evidence · 
   └ How  · `tmp/anom_p1_03_bv_attsmm.py`; budget-shifted re-run uses existing `cache_dirs`, no new explore cost.

04 ☐ **Anomaly (DATA GAP): BabyVision Opus-orch row is MISSING from `tab:orch-ablation`**
   Paper line 458-478: only HLE/LCB/GPQA are reported. BV row is blank.
   This is a hole, not a violation, but it leaves Figure 4's "Opus on BV?" question unanswered. Same explore cache that Sonnet uses; pure orchestrator-swap = $30-60 incremental cost.
   ├ G1 ☐ Gate · pre-flight: cache `analysis/cache/babyvision/sonnet/` exists with 388 question dirs each containing 8 explore_*/result.json files. Verify wc -l on a sample = 8.
   │      Evidence · 
   ├ G2 ☐ Gate · launch banner shows BOTH `Resuming ...: 0 rollouts already completed` (fresh) AND `Tasks: 388 to run, X already cached` with X close to 388×8 = 3104 cached explores reused.
   │      Evidence · 
   ├ G3 ☐ Gate · run completes with results.jsonl row count = 388 (no timeouts >5%) and total cost ≤ $80 (the projected upper bound; budget alarm).
   │      Evidence · 
   ├ G4 ☐ Gate · post-run: paper `tab:orch-ablation` updated with BV row (4 numbers: Haiku/Sonnet/Opus accuracy + $/q). Compile passes. Figure 4 PDF regenerated via `scripts/plot_all_methods.py`.
   │      Evidence · 
   └ How  · new yaml `scripts/babyvision/opus/babyvision_opus_orch.yaml` (orchestrator=Opus, explore cache=Sonnet's), launch via `python eval.py --config <yaml>`; log to `tmp/anom_p1_04_bv_opus_orch.log`.

---

## Phase 2 — Effort-monotonic violations [0/3]

05 ☐ **Anomaly: ATTS-MM (High) on BabyVision DROPS below Med (22.16 < 22.42)**
   Paper line 411: BV row reads `19.85 / 22.42 / 22.16` for Low/Med/High.
   Allocating MORE compute REDUCES accuracy. Direct contradiction of "more is better".
   ├ G1 ☐ Gate · per-question diff between `babyvision/multi_model_effort_medium/run_20260324_023252` and `babyvision/multi_model_effort_high/run_*`. Report (Med correct, High wrong) count. Threshold: ≥4 → effect is real.
   │      Evidence · 
   ├ G2 ☐ Gate · for each (Med correct, High wrong) qid, count num_explores at each effort. Pattern check: "High over-explored past Med's stop point AND added a wrong-answer cluster that synthesis flipped to". Threshold: ≥75% match.
   │      Evidence · 
   ├ G3 ☐ Gate · cross-bench: same Med→High delta on HLE (+3pp), LCB (0pp), GPQA (+0.5pp), BV (-0.26pp). Test: monotonic per-bench? If only BV regresses, mechanism is benchmark-specific.
   │      Evidence · 
   ├ G4 ☐ Decision gate · Either (A) keep the row and add a "BV is the only bench where Med→High regresses" paragraph linked to the Haiku-poisoning mechanism in P1.03, or (B) replace BV row with "Med (best) only" footnoted as "High effort regresses on visual reasoning; see appendix".
   │      Evidence · 
   └ How  · `tmp/anom_p2_05_effort_bv.py`. No new run needed.

06 ☐ **Anomaly: ATTS-MM (Med) on LCB EQUALS High (85.63 = 85.63) at half cost**
   Paper line 410: LCB row reads `77.01 / 85.63 / 85.63`. Med→High is FLAT but cost goes 0.69 → 1.09. This is anomalous because the same Med→High step is +3pp on HLE.
   ├ G1 ☐ Gate · per-question diff: how many qids did High get right that Med missed (call it `+`), and how many did High miss that Med got right (`-`)? `acc_diff = 0` could mean (a) `+ = - = 0` (deterministic stopping), or (b) `+ = -` (genuine reshuffle that nets zero). If (b), the equality is FRAGILE.
   │      Evidence · 
   ├ G2 ☐ Gate · if (b), the cost of going from Med to High buys nothing on net. Paper currently bolds Med as "best operating point" — the bolding must be reinforced with explicit "+ - = 0 reshuffling" language to preempt reviewer "why pay 60% more for nothing?" question.
   │      Evidence · 
   ├ G3 ☐ Decision gate · narrative rewrite vs row removal — option (A) add language acknowledging the reshuffle, (B) cut the High column for LCB only, marking it as "saturated" with a footnote.
   │      Evidence · 
   └ How  · `tmp/anom_p2_06_effort_lcb.py`.

07 ☐ **Anomaly: ATTS k\* ≈ 2.13 on GPQA — orchestrator uses 27% of budget, leaves 73% on the table**
   `todo_orch_evidence_A_pool_state.md` Phase 6 already established this. 82% of GPQA questions stop at exactly 2 explores. The cache contains 8 candidates per question; 6 are unread.
   This is the root cause underlying many of the per-question failures in `tmp/diff_atts_vs_baselines.md` ("premature-stop-saw-0-correct" pattern).
   ├ G1 ☐ Gate · simulate counterfactual: force `num_explores=4` on the same explore cache (re-run synthesis only via standalone-integrator with first 4 candidates). Measure GPQA accuracy at N=2/4/6/8. Threshold: monotonic curve with N=4 above N=2 by ≥2pp → "ATTS leaves accuracy on the table".
   │      Evidence · 
   ├ G2 ☐ Gate · compute the "oracle stop" upper bound: best achievable by picking the right stop point per question. This is the ceiling. Compare to ATTS's actual accuracy. Threshold: gap ≤ 5pp → ATTS is near-oracle and the "premature stop" framing is overstated; gap >10pp → there is real headroom.
   │      Evidence · 
   ├ G3 ☐ Decision gate · this is NOT a paper retraction; it is a paper *positioning* issue. Either (A) add "Limitations" paragraph acknowledging "ATTS leaves N=4-6 cache reads unused; future work: better stop policies", or (B) propose a fixed-num_explores=4 ATTS variant in main results to capture the headroom. Option B is the bigger paper claim and would need running.
   │      Evidence · 
   └ How  · `tmp/anom_p2_07_stop_headroom.py`; uses `methods/standalone_integrator.py` to fake the forced-stop synthesis.

---

## Phase 3 — Integration / synthesis violations [0/2]

08 ☐ **Anomaly: ATTS (+ Integrator) DROPS GPQA by 5pp (80.81 → 76.14) and LCB by 2pp (82.29 → 80.57)**
   Paper line 434-437 (`tab:integrator-ablation`). Adding a "more synthesis" step REDUCES accuracy on 3 of 4 benchmarks.
   ├ G1 ☐ Gate · per-question diff between `gpqa/sonnet_no_integrate_run1/run_20260328_060344` and `gpqa/sonnet_standalone_integrator/run_*`. Count (no-integrator correct, integrator wrong). Threshold: ≥10 qids on GPQA → mechanism is real.
   │      Evidence · 
   ├ G2 ☐ Gate · for each such qid, look at the integrator's prediction and the cached pool: is the integrator (a) overruling a correct majority (e.g. 5/8 cache says "B" → integrator says "A"), or (b) picking a minority answer that no other method picks? Threshold: ≥75% in pattern (a) → "integrator over-reasons and rejects valid majorities".
   │      Evidence · 
   ├ G3 ☐ Gate · cross-bench check: HLE +1pp, LCB -2pp, GPQA -5pp, BV +0pp. Per-bench mechanism difference. Threshold: at least 2 of 4 benchmarks show same dominant pattern from G2.
   │      Evidence · 
   ├ G4 ☐ Decision gate · the paper currently says the integrator "has less context" (line 444). G2's results either confirm or refute this. If confirmed, narrative stays; if refuted (e.g. mechanism is "integrator hallucinates new answers"), narrative must be rewritten with the new mechanism.
   │      Evidence · 
   └ How  · `tmp/anom_p3_08_integrator_diff.py`.

09 ☐ **Anomaly: Self-Refine on GPQA DROPS BELOW Pass@1 (73.10 < 74.24)**
   Paper line 311 + 366. SR's iterative refinement makes things WORSE than not refining.
   ├ G1 ☐ Gate · per-question diff between Self-Refine GPQA results and Pass@1 (= first cached Sonnet candidate). Count (Pass@1 correct, SR wrong) qids. Threshold: ≥5.
   │      Evidence · 
   ├ G2 ☐ Gate · for each such qid, is the SR final answer (a) the original draft revised AWAY from correct, or (b) a new wrong answer the feedback step introduced? Read the trajectory file in the SR run dir.
   │      Evidence · 
   ├ G3 ☐ Decision gate · paper line 366 already names this mechanism ("feedback model accepts initial draft, occasionally introduces errors"). Confirm with G2 numbers (e.g. "X of Y SR-failure cases were revisions away from a correct draft"). If confirmed, add the number to line 366.
   │      Evidence · 
   └ How  · `tmp/anom_p3_09_sr_gpqa.py`.

---

## Phase 4 — Tie-coincidence anomalies [0/1]

10 ☐ **Anomaly: BabyVision THREE-WAY TIE at 23.20% is COINCIDENTAL, not equivalence**
   Paper line 339, 344, 345: Majority Voting / LLM Selection / ATTS all = 23.20% on BV.
   `tmp/diff_atts_vs_baselines.md` already showed: agreement is 90/388 each, but the three methods disagree on **16 questions**. The equality is cancellation noise.
   ├ G1 ☐ Gate · build 3-way agreement matrix (qid × method ∈ {ATTS, MV, LLMSel}, cell = correct/wrong). Report: how many qids does ALL THREE methods agree on? Threshold: if ≥372/388 (96%), tie is robust; if ≤350/388 (90%), tie is fragile.
   │      Evidence · 
   ├ G2 ☐ Gate · run a second seed (different `seed:` value) for ATTS on BV. If accuracy lands in 21-25% range, tie is genuinely random; if it lands within ±0.3pp of 23.20%, tie is reproducibly coincidental.
   │      Evidence · 
   ├ G3 ☐ Decision gate · paper line 369 currently says "ties Majority Voting and LLM Selection at 23.20%". Either (A) add "though the three methods disagree on 16/388 individual questions; the equality at population scale is coincidental, not item-level equivalence", or (B) replace with "matches Majority Voting within sampling noise (±0.5pp)" — both honest, (A) is sharper.
   │      Evidence · 
   └ How  · `tmp/anom_p4_10_bv_tie.py`; second-seed run reuses cache, $0.

---

## Phase 5 — Grading / measurement consistency [0/2]

11 ☐ **Anomaly: HLE Socratic Self-Refine RE-GRADED 45 → 58, but other HLE methods may NOT have been**
   Paper line 226-244 (comments only): 13 of 100 SSR rows flipped wrong→correct under Haiku judge after the 2026-04-11 `judge_model=null` bug was restored. The same 13-row LaTeX-equivalence flip would also affect Self-Refine, Budget Forcing, Skywork, LLMSel, ATTS, ATTS-MM if their results.jsonl was graded under the bug period.
   ├ G1 ☐ Gate · for each method's HLE run dir, check `grade.json` files: read the `judge_model` field. Method must show `claude-haiku-4-5-20251001`, NOT `null`. Threshold: 100% of grade.json files report Haiku.
   │      Evidence · 
   ├ G2 ☐ Gate · for any method where ≥1% have null judge_model, force re-grade (the cache will auto-invalidate via the `_grade_with_cache` mechanism — `eval.py:45-73`). Re-run accuracy; report delta. Threshold: if delta >2pp on any method, paper number must be updated.
   │      Evidence · 
   ├ G3 ☐ Gate · post-fix consistency check: all HLE methods' results.jsonl re-graded under same Haiku judge. Acc% delta from current paper number reported per method. Threshold: ≤±3pp for all; if larger, ranking may change → all narrative paragraphs touching HLE need rereading.
   │      Evidence · 
   ├ G4 ☐ Decision gate · UPDATE every paper HLE row to the post-uniform-grade number. Re-run `tmp/diff_atts_vs_baselines.py`. Re-run `scripts/plot_all_methods.py` for the cost-vs-accuracy figure.
   │      Evidence · 
   └ How  · `tmp/anom_p5_11_hle_judge_uniformity.sh` — bash loop over HLE run dirs.

12 ☐ **Anomaly: LCB EMPTY-recovery rerun ONLY retried the 15 EMPTY rows; did NOT re-validate non-EMPTY rows**
   Paper line 498-513 (comments): 4/15 EMPTY rows from Qwen3.6 LCB run were retried at higher max_tokens; 3/4 flipped to correct. But the OTHER 160 already-graded rows from the original 32K run were not re-graded with the higher budget. Possible silent biases: short-context rows that produced syntactically OK but semantically incomplete answers may have been incorrectly graded as wrong.
   ├ G1 ☐ Gate · sample 10 random non-EMPTY rows from `lcb/qwen36_35b_a3b_fp8_temp/run_20260501_042951` where `is_correct=false`. Re-run with max_tokens=131,072. Report: how many flip to correct? Threshold: ≥2/10 flips → original LCB Qwen3.6 number is biased low; need full re-grade.
   │      Evidence · 
   ├ G2 ☐ Decision gate · either re-run the full 175 LCB Qwen3.6 row with the higher budget (cost: $80-150 estimate), or footnote the table to say "X% of non-EMPTY rows could potentially flip if budget were extended; reported number is therefore lower-bound".
   │      Evidence · 
   └ How  · `tmp/anom_p5_12_lcb_qwen_budget.py`; sample-based smoke before deciding full rerun.

---

## Phase 6 — Cross-backbone failures [0/2]

13 ☐ **Anomaly: Qwen3.6 backbone on BabyVision = 13.66% < Pass@1 19.59% (Gain -5.93)**
   Paper line 515-519 (comments only): the BV row was DROPPED from `tab:orch-ablation-cross` because it was negative-gain. This is currently HIDDEN — reviewer cannot see it.
   ├ G1 ☐ Gate · verify the dropped row is preserved at `archive/appendix_g_babyvision.tex` (mentioned in comments). If yes, content should be promoted to the main paper or to a new "Limitations" appendix; if no, the row is silently lost.
   │      Evidence · 
   ├ G2 ☐ Gate · paper line 515-518 explains: "explorer-cache binding constraint (only 36.1% of cached candidates contain a correct option) was out of scope". This is the mechanism — but it's hidden in a LaTeX comment. Promote to a paragraph in the main appendix.
   │      Evidence · 
   ├ G3 ☐ Decision gate · either (A) restore the BV row to `tab:orch-ablation-cross` with explicit "(BV: -5.93, see appendix for cache-binding analysis)" footnote, or (B) keep dropped but add a one-paragraph "Why BV was excluded from cross-backbone Qwen3.6 results" justification in the main text.
   │      Evidence · 
   └ How  · `tmp/anom_p6_13_qwen_bv.py` — extract the dropped numbers and compose the appendix paragraph.

14 ☐ **Anomaly: Backbone ablation only includes GPT-5.2 + Qwen3.6 + Gemma — DeepSeek / Kimi / Grok untested**
   Paper Table `tab:backbone-ablation` (line 547-570). The framework's claim is "transfers to other backbones". Three is a thin sample.
   ├ G1 ☐ Gate · CLAUDE.md project file lists which open-router providers are STRUCTURAL OK for tool_choice; the 2026-05-04 cheat-sheet says: gpt-oss-120b, gpt-oss-20b, grok-4.1-fast, deepseek-v4-pro, kimi-k2.6, gemini-3-flash-preview. At least one of these (gpt-oss-120b or grok-4.1-fast) should be runnable as orchestrator and would broaden the claim.
   │      Evidence · 
   ├ G2 ☐ Decision gate · either run +1 cross-family backbone before submission (cost: ~$50-100 per bench × 4 benches = $200-400), or restate the claim in the paper as "evaluated on 3 cross-family backbones" instead of the more general "framework transfers". The narrower claim is honest; the broader claim needs more rows.
   │      Evidence · 
   └ How  · only triggered if reviewer pressure demands more backbones; default is option B (narrow the claim).

---

## Run order

The 14 items naturally split into:

- **Cheap / offline / no-new-run** (1-2 hours each): 01-G1/G2, 02, 03-G1/G2, 05, 06, 07-G1, 08, 09, 10-G1, 11, 13. All read existing results.jsonl.
- **Cheap / new run / cache-reuse** ($30-100 each): 04 (BV Opus orch ~$30-60), 01-G3 (GPQA Opus seed-2), 03-G3 (BV ATTS-MM Haiku-excluded), 07-G2 (forced-stop synthesis ~$20), 10-G2 (BV ATTS seed-2), 12-G1 (LCB Qwen sample-10).
- **Defer / narrow-the-claim**: 14.

Suggested order:
1. Run all cheap/offline diagnostics in parallel — finishes within an afternoon, gives mechanism evidence for 11 of 14 items
2. Based on G2 findings, decide which $30-100 runs are worth firing
3. Sequence the new runs serially with monitoring (heartbeat 10-min)
4. Edit the paper one phase at a time, compile after each phase to catch ref breakage

## Open questions for the user before starting

a. The user wrote "augmentation" but the paper has no data augmentation. The starter items are interpreted in this TODO as "augmentation = adding more compute / more diversity / more synthesis on top of base ATTS" (covered by Phase 1 multi-model + Phase 2 effort + Phase 3 integrator). Confirm or redirect.
b. Are seeds for "second-seed" Gates (G3 of items 01, 10) different from the original seed=42? If so, which seed value should be used for reproducibility?
c. Budget cap for new runs in this TODO: total of $200-300 if all "$30-100 cheap new runs" fire. Confirm the cap before P1.04 is launched.
d. Is item 14 (more cross-family backbones) in scope for this submission cycle, or strictly future work?

---

# Appendix — Paper Story Registry & Evidence Audit

This appendix is the **contract** for the paper. Every story below must be
empirically defensible at submission time. Where evidence contradicts a
story, the **experiment is wrong, not the story** — fix the experiment to
align with the story, OR retract / soften the claim. Do NOT silently let
contradictory rows live in the paper.

User directive (2026-05-04): "数据中有些地方存在矛盾。这些 claim 是最弱的我们先去提升薄弱环节。"
Translation: "There are contradictions in the data. We will first strengthen
the weakest claims; the rest of the story can only land once those are fixed."

## The six stories

### S1 — Adaptive stopping beats fixed budget at lower cost

**Claim**: ATTS reaches accuracy comparable to or better than fixed-budget
test-time scaling baselines (Majority Voting, LLM Selection, Skywork-Reward,
etc.) while using a fraction of their cost.

**Predicted pattern**: For each (benchmark, baseline) pair,
`ATTS Acc ≥ baseline Acc - ε` AND `ATTS $/q ≤ 0.5 × baseline $/q`.

**Current evidence (table 1, paper line 200-350)**:

| Bench | ATTS Acc / $/q | Best baseline (Acc / $/q) | Aligned? |
|---|---|---|---|
| HLE | 56.00% / $1.59 | LLM Sel 58% / $3.71 | ✓ +2pp at 2.3× cost |
| LCB | 82.29% / $0.53 | LLM Sel 81.14% / $1.51 | ✓ ATTS higher AND cheaper |
| GPQA | 80.81% / $0.34 | LLM Sel 77.16% / $0.97 | ✓ ATTS higher AND cheaper |
| BV | 23.20% / $0.27 | MV / LLM Sel both 23.20% | △ tie, not win |

**Counter-evidence**:
- BV three-way tie at 23.20% is COINCIDENTAL — ATTS, MV, LLMSel each get
  90/388 right, but they disagree on 16 individual questions (8+8 cancel)
  → see `tmp/diff_atts_vs_baselines.md`. The equality at population scale
  hides item-level disagreement.

**Why not strong enough**: BV story is "matches at lower cost", not "wins".
Reviewer will ask "if it's a tie, why use ATTS?". The robustness argument
(stable under question reshuffle) is currently NOT in the paper.

**Fix**: TODO item P4.10 + add 16-question disagreement matrix to appendix
as ROBUSTNESS evidence rather than weakness.

---

### S2 — Stronger orchestrator → higher efficiency → lower total cost

**Claim**: A more capable orchestrator (Haiku → Sonnet → Opus) reasons more
effectively about when to stop and which answer to pick, so even though its
per-token price is higher, total $/q is lower because it wastes fewer
explorer calls.

**Predicted pattern**: Across Haiku / Sonnet / Opus on the SAME explore
cache, Acc and $/q should both move monotonically with capability tier:
Acc ↑, $/q ↓.

**Current evidence (Table 4, `tab:orch-ablation`, paper line 458-478)**:

| Bench | Haiku Acc / $/q | Sonnet Acc / $/q | Opus Acc / $/q | Acc monotonic? | $/q monotonic? |
|---|---|---|---|---|---|
| HLE | 57% / $1.83 | 56% / $1.59 | **59% / $1.50** | ✓ Opus best | **✓ Opus cheapest** |
| LCB | 79.43% / $0.71 | 81.71% / $0.53 | 83.43% / **$0.64** | ✓ | ✗ Sonnet cheapest, Opus 1.2× |
| GPQA | 80.20% / $0.36 | **80.81%** / $0.34 | **76.77%** / $0.34 | ✗ Opus REGRESSES | ≈ same |
| BV | -- | 23.20% / $0.27 | **MISSING** | -- | -- |

**Counter-evidence**:
1. GPQA: Opus 76.77% < Sonnet 80.81%, k\* identical (2.15 vs 2.14) → SAME
   evidence pool, Opus picks worse final answer. Synthesis-quality regression.
2. LCB: Opus k\* = 2.83 vs Sonnet 1.86 — Opus EXPLORES MORE rather than
   stopping more accurately, contradicting the "stops better" mechanism.
3. BV: data hole — cannot validate or refute on visual.

**Why not strong enough**: Story is verified on 1 of 4 benchmarks (HLE).
Reviewer will see the GPQA row immediately and question the universal claim.

**Fix path** (per user directive: experiment is wrong, not story):
- Run GPQA Opus second seed: if it stays at 76.77%, single-seed claim is
  retracted. If it lands at ≥80%, original was unlucky.
- Run BV Opus orch ($30-60): fills the missing cell.
- LCB Opus cost regression: investigate Opus's stop-criterion bias on code
  problems; if Opus systematically over-explores LCB, prompt-tune to commit
  earlier when 2 candidates pass tests.

**Mapped TODO items**: P1.01 (GPQA Opus diff + seed-2), P1.02 (HLE Haiku
reversal sanity), P1.04 (BV Opus orch run).

---

### S3 — Weakening orchestrator → worse performance

**Claim**: Constraints on the orchestrator (turning off extended thinking,
swapping to a smaller model, etc.) directly hurt performance because the
orchestrator's reasoning is what extracts the right answer from the candidate
pool.

**Predicted pattern**:
- Disable extended thinking → Acc drops, k\* unchanged → synthesis-quality
  regression (not stopping regression).
- Smaller model → Acc drops monotonically.

**Current evidence**:

| Weakening | Bench | Δ Acc | Δ k\* | Aligned? |
|---|---|---|---|---|
| Thinking off (Table 7, line 587-588) | GPQA | -3.5pp (82.83→79.29) | -0.07 (~same) | ✓ strong evidence |
| Sonnet→Haiku (Table 4) | GPQA | -0.6pp | +0.29 | ✓ small |
| Sonnet→Haiku | LCB | -2.28pp | +1.30 | ✓ |
| Sonnet→Haiku | HLE | **+1pp** | +1.28 | **✗ Haiku BEATS Sonnet** |
| MM Med→High effort (paper line 411) | BV | **-0.26pp** | (more) | **✗ more compute hurts** |

**Counter-evidence**:
1. HLE Haiku 57% > Sonnet 56%. Smaller model BEATS mid-tier on HLE.
2. ATTS-MM (Med→High) on BV: 22.42 → 22.16. Strengthening orchestrator's
   exploration budget makes BV WORSE.

**Why not strong enough**: Two reversals on visual / language extremes (HLE
free-form, BV image). The "weaker = worse" story holds on the structured
benchmarks (GPQA, LCB) but fails at both ends of the spectrum.

**Fix**:
- HLE Haiku reversal: per-question diff vs Sonnet — if the +1pp is 1-3
  questions, it's noise (sample stderr ≈ 5pp on n=100); add seed-2 if
  reviewer pressure demands.
- BV Med→High: P2.05 — diagnose 13 BV questions that ATTS-MM-High
  loses but ATTS-MM-Med gets. If they all involve weak Haiku visual answers
  consumed at higher k\*, the story is "Haiku noise floor saturated at Med
  effort"; rewrite the BV row with that narrative.

**Mapped TODO items**: P2.05 (effort-on-BV), P1.02 (HLE Haiku reversal).

---

### S4 — A separate integrator should not help

**Claim**: ATTS's stateful in-conversation synthesis is sufficient. Adding
a second model call (a "separate integrator" that re-reads the candidate
pool) does NOT improve accuracy and DOES increase cost.

**Predicted pattern**: For each benchmark, ATTS Acc ≥ ATTS+Integrator Acc,
AND $/q always increases.

**Current evidence (Table 5, `tab:integrator-ablation`, paper line 420-440)**:

| Bench | ATTS Acc / $/q | +Integrator Acc / $/q | Δ Acc | Aligned? |
|---|---|---|---|---|
| GPQA | 80.81% / $0.34 | 76.14% / $0.44 | **-4.67pp** | ✓ integrator HURTS |
| LCB | 82.29% / $0.53 | 80.57% / $0.72 | **-1.72pp** | ✓ integrator HURTS |
| BV | 23.20% / $0.27 | 23.20% / $0.38 | 0 | ✓ no help, costs 1.4× |
| HLE | 56.00% / $1.59 | 57.00% / $2.35 | **+1pp** | **✗ tiny gain** |

**Counter-evidence**: HLE +1pp at 1.5× cost. Single bench reversal.

**Why not strong enough**: One row of +1pp lets a reviewer say "integrator
sometimes helps, you should report when". The cleanest version of this
story would have HLE at +0pp or -1pp.

**Fix**:
- P3.08: per-question diff between HLE ATTS and HLE +Integrator. If the
  +1pp is 1 question that flips, frame as "within sampling noise (stderr
  ≈ 5pp on n=100); cost is 48% higher with no statistically significant
  Acc change". The story holds.
- If 2+ questions flip in the same direction, soften the universal claim
  to "no consistent improvement, occasional 1pp gain at 1.5× cost".

**Mapped TODO item**: P3.08.

---

### S5 — Framework transfers across model families

**Claim**: Same prompts and control logic, swap orchestrator backbone to a
non-Anthropic family (Qwen, Gemma, GPT) → still get positive Gain over
Pass@1 on the same explore cache.

**Predicted pattern**: For each (orchestrator backbone, benchmark), Gain
= Acc - Pass@1 ≥ 0.

**Current evidence (Tables 5, 6 — `tab:orch-ablation-cross`,
`tab:backbone-ablation`)**:

| Backbone | HLE Gain | LCB Gain | GPQA Gain | BV Gain |
|---|---|---|---|---|
| Within-family Anthropic | ✓ all 3 tiers | ✓ | △ Opus reversal | DATA HOLE |
| Qwen3.6-35B-A3B-FP8 | +8.0 | +1.15 | +8.58 | **-5.93 (HIDDEN)** |
| Gemma-4-26B-A4B-it | +4.0 | +2.86 | +0.51 | +2.32 |
| GPT-5.2 (low) | +1.0 | -- | -- | -- |
| GPT-5.2 (high) | +2.0 | -- | +X | -- |

**Counter-evidence**: Qwen3.6 on BV had Gain = -5.93. The row is preserved
at `archive/appendix_g_babyvision.tex` (paper line 515-519 comments) but is
NOT in the rendered PDF. A reviewer who finds the archive will see a
silent negative-result hidden.

**Why not strong enough**: Universal-transfer claim is undermined by 1
hidden negative. Either restore the BV row with mechanism explanation
("explorer-cache binding constraint: only 36.1% of cached candidates contain
a correct option, so any orchestrator is upper-bounded"), or weaken the
claim to "transfers to text/code; visual transfer is cache-bounded".

**Fix path**: P6.13 — promote BV row to main appendix with mechanism
narrative; OR rewrite line 481 claim from "robust to backbone substitution"
to "robust on text/code benchmarks; visual benchmark is bounded by explorer
cache quality, see appendix".

**Mapped TODO item**: P6.13.

---

### S6 — Tool space complexity → monotonic accuracy gain (NEW)

**Claim** (user directive 2026-05-04): The orchestrator's action space is
extensible. As we expand from a minimal action set to a richer one,
accuracy should improve monotonically.

Three planned stages:
- **Stage 1** = base ATTS — single action `explore` (orchestrator only
  decides when to STOP). Currently in paper.
- **Stage 2** = ATTS-MM — adds model selection (orchestrator picks WHICH
  explorer to dispatch from {Haiku, Sonnet, Opus}). Currently in paper.
- **Stage 3** (NOT YET IMPLEMENTED) — adds per-call customized prompt
  passed to the explorer (orchestrator can specify HOW to instruct the
  explorer for the next attempt). Future work.

**Predicted pattern**: For each benchmark,
`Acc(Stage 1) < Acc(Stage 2) < Acc(Stage 3)`. Stage 3 not testable yet,
but Stage 1 → Stage 2 monotonicity is required NOW.

**Current evidence (recomputed 2026-05-04 from results.jsonl)**:

| Bench | Stage 1 ATTS Acc / $/q / k\* | Stage 2 ATTS-MM Acc / $/q / k\* | Δ Acc | Aligned? |
|---|---|---|---|---|
| HLE | 56.00% / $1.90 / 2.39 | 57.00% / **$1.63** / 3.70 | **+1.00pp** | ✓ Acc UP, $/q DOWN |
| LCB | 81.71% / $0.53 / 1.86 | **85.63%** / $0.69 / 3.56 | **+3.92pp** | ✓ Acc UP |
| GPQA | 80.81% / $0.34 / 2.14 | 81.82% / $0.39 / 3.19 | **+1.01pp** | ✓ Acc UP |
| BV | 23.20% / $0.27 / 2.83 | 22.42% / $0.43 / 4.63 | **-0.77pp** | **✗ Acc DOWN** |

**Counter-evidence**:
1. **BV: ATTS-MM regresses by 0.77pp**. Adding model-selection action
   space DECREASES accuracy on visual reasoning. 13 of these regressions
   are documented in `tmp/diff_atts_vs_baselines.md` as
   "ATTS correct → ATTS-MM wrong, with multi-model pool injecting Haiku's
   weaker visual answer that flips the synthesis".
2. **Cost regresses on 3 of 4 benches** when going Stage 1 → Stage 2
   (LCB +$0.16, GPQA +$0.06, BV +$0.16). Only HLE gets Acc up AND cost
   down. This means S6's Stage 1 → Stage 2 transition is monotonic on
   accuracy (3/4) but NOT on cost (1/4) — separate from S2 which talks
   about orchestrator capability tier.

**Why not strong enough**:
- BV reversal is the same data point that breaks S3 (Med→High effort) and
  S4 (additional integrator does nothing). All three stories converge on
  "BV is the bench where adding capability fails". Cannot bury this in
  one paragraph; needs a unified mechanism explanation.
- Stage 3 has zero evidence. The monotonicity claim is built on n=2
  observations per benchmark. If a reviewer asks "why should it keep
  improving at Stage 3?", we have no extrapolation argument.

**Fix path**:
1. Document the BV regression mechanism (Haiku visual injection) — turn
   the failure into a finding: "Stage 2 monotonicity requires that the
   weakest member of the model pool be ≥ baseline; on visual benchmarks
   where Haiku is below baseline, expanding the action space without
   filtering hurts."
2. Validate the mechanism: re-run BV ATTS-MM with `model_budgets =
   {sonnet:8, opus:4, haiku:0}` (cache-zero $0). If Acc bounces back to
   ≥23.20%, mechanism confirmed.
3. Add a sentence about Stage 3 being future work, with the monotonic
   trend as the motivating signal — not a tested claim.

**Mapped TODO items**: P1.03 (BV ATTS-MM diff + Haiku-excluded re-run);
new item P7.15 below.

---

## Weakness ranking — fix the weakest links first

Each story's weakness is the worst counter-example or missing data point.
Ranked by "how badly a reviewer can break the story by pointing at one row":

| Rank | Story | Weakest data point | Why it breaks the story | Fix cost |
|---|---|---|---|---|
| **1 (worst)** | **S2** | GPQA Opus 76.77% < Sonnet 80.81% with same k\* | Direct contradiction of "stronger orch is more efficient"; reviewers will see the Table 4 row first | $0 diff + $30-50 seed-2 run |
| **2** | **S6** | BV ATTS-MM 22.42% < ATTS 23.20% | Direct contradiction of monotonic tool-space gain; same row also breaks S3 | $0 diff + $0 model_budgets re-run |
| **3** | **S5** | Qwen3.6 BV Gain = -5.93 hidden in archive | Reviewer who finds archive will accuse selective reporting | $0 promote-to-appendix |
| **4** | **S2** | LCB Opus $/q higher than Sonnet | Opus more expensive on LCB, undermines "stronger = cheaper" | $0 diff to identify k\* divergence reason |
| **5** | **S3** | HLE Haiku 57% > Sonnet 56% | Smaller model beats mid-tier — anti-monotonic | $0 diff (likely <3-question noise); $30 seed-2 if needed |
| **6** | **S1** | BV three-way tie 23.20% | "Wins at lower cost" downgrades to "matches at lower cost" | $0 add disagreement matrix as robustness |
| **7** | **S4** | HLE +Integrator +1pp | One bench tiny gain — cleanest story has all 4 ≤0 | $0 diff (likely <3-question noise) |
| **8** | **S6 Stage 3** | No evidence at all | Future-work claim, defensible | -- |

**Priority order** (do top of list first; each fix unlocks paper-narrative
rewrite for that story):

```
[P0]  Rank 1  S2 GPQA Opus regression  →  TODO P1.01
[P0]  Rank 2  S6 BV ATTS-MM regression →  TODO P1.03 + new P7.15
[P0]  Rank 3  S5 Qwen3.6 BV hidden     →  TODO P6.13
[P1]  Rank 4  S2 LCB Opus cost         →  TODO P1.01-G2 extension
[P1]  Rank 5  S3 HLE Haiku reversal    →  TODO P1.02
[P2]  Rank 6  S1 BV tie robustness     →  TODO P4.10
[P2]  Rank 7  S4 HLE integrator +1pp   →  TODO P3.08
[P3]  Rank 8  S6 Stage 3 future work   →  narrative only
```

---

## Phase 7 — S6 monotonicity validation (new) [0/1]

15 ☐ **Validate Stage 1 → Stage 2 monotonicity on BV by ablating the explorer pool**
   The S6 Stage 1 → Stage 2 reversal on BV is the second-worst weakness in the
   paper. Mechanism hypothesis (from `tmp/diff_atts_vs_baselines.md` 13-case
   study): adding Haiku to the explorer pool injects below-baseline visual
   answers that the orchestrator's synthesis cannot reliably reject.
   ├ G1 ☐ Gate · pre-flight: caches at `analysis/cache/babyvision/{haiku,sonnet,opus}/` exist; verify each has 388 question dirs and at least 4 explore_*/result.json each (Opus has budget=4 in MM).
   │      Evidence · 
   ├ G2 ☐ Gate · launch ATTS-MM on BV with `model_budgets: {haiku: 0, sonnet: 8, opus: 4}`. Resume + cache banner shows 0 new explorer calls (Haiku branch zeroed; Sonnet+Opus already cached). Run uses orchestrator-only $.
   │      Evidence · 
   ├ G3 ☐ Gate · post-run Acc ≥ 23.20% (= ATTS Stage 1 BV). If yes, the BV regression is causally attributable to Haiku injection → mechanism confirmed; S6 Stage 2 monotonicity restored under the constraint "weakest pool member ≥ baseline".
   │      Evidence · 
   ├ G4 ☐ Decision gate · either (A) update paper Table 1 BV row to use the Haiku-excluded ATTS-MM variant (state the constraint explicitly: "for benchmarks where Haiku < baseline, exclude Haiku from the model pool"), or (B) add a paragraph to Section 5.1 acknowledging the Stage-2 monotonicity has a precondition and citing the diagnostic experiment.
   │      Evidence · 
   └ How  · new yaml `scripts/babyvision/multi_model/babyvision_mm_no_haiku.yaml` with the budget override; launch via `python eval.py --config <yaml>`; log to `tmp/anom_p7_15_bv_mm_no_haiku.log`.
