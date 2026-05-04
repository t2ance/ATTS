# TODO: Unified action-space-complexity axis framing for ATTS paper

The paper will be re-framed around one continuous axis of orchestrator action-space complexity, with five labeled positions:

- P1 (left): static Aggregator (no decision; fixed-N + post-hoc aggregation)
- P2: + when-to-stop (current ATTS)
- P3: + which-model (current ATTS-MM)
- P4: + per-call sampling parameters (temperature / max_tokens / reasoning effort)
- P5 (right): + free-form directing prompt (explore is itself a sub-agent)

P1, P2, P3 are anchored by existing experiments. One additional experiment will be added at P4 or P5 to anchor the right side. Choice of P4 vs P5 is decided in Phase 3, after Phase 2 catalogues what existing data already covers.

## Phase 1 — Lock the framing [0/3]

01 [ ] Name the axis and the 5 anchored points
   - G1 [ ] Gate: one canonical axis name decided and written down here.
     Evidence:
   - G2 [ ] Gate: 5 point labels (P1..P5) finalized; one-line description for each, used identically in figure caption, intro, methodology.
     Evidence:
   - How: user picks; record in this file.

02 [ ] Decide the figure type and layout
   - G1 [ ] Gate: choose conceptual-only (no measurements on figure) OR semi-empirical (each anchored point carries a measured Acc / $/q).
     Evidence:
   - G2 [ ] Gate: layout sketched; gradient direction, label positions, presence/absence of y-axis confirmed.
     Evidence:
   - How: show 2-3 layout options as text mockups; user picks.

03 [ ] Decide insertion location in main.tex
   - G1 [ ] Gate: section confirmed (Section 1 Introduction OR Section 3 Methodology).
     Evidence:
   - G2 [ ] Gate: every paragraph that needs rewording with the new vocabulary is listed with line numbers.
     Evidence:
   - How: grep the current Methodology and Introduction sections; build the line-number list.

## Phase 2 — Catalogue existing experiments on the axis [0/3]

04 [ ] Tag every existing table row to one of P1..P5 or off-axis
   - G1 [ ] Gate: every row in tab:main-results tagged with rationale.
     Evidence:
   - G2 [ ] Gate: every row in tab:integrator-ablation, tab:effort-ablation, tab:orch-ablation, tab:orch-ablation-cross, tab:backbone-ablation, tab:thinking-ablation tagged with rationale.
     Evidence:
   - How: build a single tagging table (row -> position -> rationale); user reviews each line.

05 [ ] Test whether P4 (per-call sampling parameters) is already partially covered
   - G1 [ ] Gate: verdict on ATTS-MM effort=Low/Med/High (tab:effort-ablation): P4 anchor or off-axis, with rationale.
     Evidence:
   - G2 [ ] Gate: verdict on GPT-5.2 effort=low vs high (tab:backbone-ablation): P4 anchor or off-axis.
     Evidence:
   - G3 [ ] Gate: verdict on ATTS (no thinking) (tab:thinking-ablation): P4 anchor or off-axis.
     Evidence:
   - How: each verdict is 1-2 sentences citing the table label + the parameter being varied.

06 [ ] Test whether P5 (free-form directing prompt) has any existing data
   - G1 [ ] Gate: grep methods/ for the orchestrator-to-explorer call path; verdict on whether any free-text instruction beyond the question is currently passed.
     Evidence:
   - How: grep Experiment/core_code/methods/; cite the call site.

## Phase 3 — Decide and scope the new experiment [0/2]

07 [ ] Pick P4 or P5 for the new experiment
   - G1 [ ] Gate: choice recorded with one-paragraph rationale grounded in Phase 2 verdicts.
     Evidence:
   - G2 [ ] Gate: wall-time and dollar-cost estimate within overnight budget (one local GPU + OpenRouter limits).
     Evidence:
   - How: user decides after reviewing Phase 2 catalogue.

08 [ ] Write yaml + sh and pass smoke (N=10)
   - G1 [ ] Gate: results.jsonl schema matches existing main-results runs; zero Traceback in log.
     Evidence:
   - G2 [ ] Gate: cache_dir is new (no reuse of any existing sonnet / qwen / haiku cache).
     Evidence:
   - How: stage under Experiment/core_code/scripts/<bench>/ ; conda run -n explain python eval.py.

## Phase 4 — Full run [0/1]

09 [ ] Full evaluation on at least 2 of the 4 main benchmarks
   - G1 [ ] Gate: timed_out rate <=5% per benchmark.
     Evidence:
   - G2 [ ] Gate: Pass@1 baseline (first-explore-only) within +/-3pp of the corresponding row in tab:main-results, confirming the explorer cache is not silently corrupted.
     Evidence:
   - G3 [ ] Gate: final Acc and $/q recorded with sample size n.
     Evidence:
   - G4 [ ] Gate: result either supports the axis monotonicity claim, or non-monotonicity is documented in writing (no silent omission).
     Evidence:
   - How: background eval.py launch; 10-min heartbeat monitor.

## Phase 5 — Re-frame the paper [0/4]

10 [ ] Insert the unified-axis figure into main.tex
   - G1 [ ] Gate: figure renders in the compiled PDF; caption uses the canonical axis vocabulary from item 01.
     Evidence:
   - How: bash Publication/paper/compile.sh.

11 [ ] Rewrite Section 1 Introduction and Section 3 Methodology around the axis
   - G1 [ ] Gate: zero leftover phrasing that contradicts the new axis (e.g. "minimal action space", "extensible action space" used inconsistently).
     Evidence:
   - G2 [ ] Gate: ATTS labelled P2 and ATTS-MM labelled P3 explicitly in the methodology paragraph and the figure caption.
     Evidence:
   - How: edit main.tex; show diff to user before compile.

12 [ ] Add Section 4 subsection reporting the new P4/P5 experiment
   - G1 [ ] Gate: numbers in tex match results.jsonl exactly (line-by-line spot check at least 5 rows).
     Evidence:
   - G2 [ ] Gate: discussion paragraph ties the result back to the axis monotonicity claim, including any non-monotonic finding.
     Evidence:
   - How: edit Section 4.

13 [ ] Re-compile and spot-check PDF
   - G1 [ ] Gate: compile.sh exits 0; zero "??" tokens in body; zero missing-reference warnings.
     Evidence:
   - G2 [ ] Gate: page count within +/-2 of the pre-rewrite version.
     Evidence:
   - How: bash Publication/paper/compile.sh ; visual page flip.

## Phase 6 — User sign-off [0/1]

14 [ ] User reviews and approves
   - G1 [ ] Gate: explicit user approval recorded.
     Evidence:
   - How: send PDF page refs; wait for ack.
