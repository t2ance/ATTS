# Data Audit: Paper Numbers vs Experimental Data
Generated: 2026-03-26

## Definite Errors (must fix)

| # | Location | Paper Value | Correct Value | Source | Status |
|---|----------|-------------|---------------|--------|--------|
| D1 | Table 1a, HLE Pass@1 cost | $0.28 | $0.50 | delegated.log: $50.34/100 = $0.5034 | DONE |
| D2 | Table 1a, HLE Skywork cost | $2.22 | $4.41 | N=8 explore cost $440.73/100. Paper used best-of-4 cost by mistake | DONE |
| D3 | Table 1a, HLE LLM Selection acc | 58.00% | 58.00% (58/100) | Q100 completed correctly. Paper value is correct. | VERIFIED OK |
| D4 | Table 1c, GPQA Majority Voting acc | 74.24% | 73.74% (146/198) | Pass@1 value was copy-pasted into MV row | DONE |
| D5 | Table 1d, BV Majority Voting overall | 22.42% | 23.20% (90/388) | Per-subset numbers sum to 90/388=23.20%, not 87/388=22.42% | DONE |
| D6 | Table 3, BV +Integrator cost | $0.34 | $0.38 | delegated.log avg $0.3775/q | DONE |
| D7 | Sec 5 + Sec 6, "88% use 2 explores" | 88% | 82% (163/198) | 88% is from with-integrator run, default ATTS is 82% | DONE |

## Minor Rounding (fix for consistency)

| # | Location | Paper Value | Correct Value | Status |
|---|----------|-------------|---------------|--------|
| R1 | Table 1c, GPQA ATTS cost | $0.35 | $0.36 ($70.60/198) | DONE |
| R2 | Table 3, GPQA Default cost | $0.35 | $0.36 (same as R1) | DONE |
| R3 | Table 5, GPQA Sonnet orch cost | $0.35 | $0.36 (same as R1) | DONE |

## Questionable (need investigation)

| # | Location | Paper Value | Issue | Status |
|---|----------|-------------|-------|--------|
| Q1 | Table 1c, GPQA Pass@1 acc | 74.24% | Semantic diff: first_candidate_correct includes timeouts, best-of-1 skips them. 74.24% is the conservative value (timeout=wrong). Keeping as-is. | RESOLVED (keep) |
| Q2 | Table 6, AIME 2026 Pass@1 | 93.33% | Stale LLM-judge grading from old sonnet run. Correct value 96.67% (29/30). Fixed. | DONE |

## Figure-Table Inconsistency

| # | Location | Table Value | Figure Value | Status |
|---|----------|-------------|-------------|--------|
| F1 | GPQA effort High | Table 4: 82.32%, $0.40 (v1 run) | Plot script changed to v1 run. Table and figure now consistent. | DONE |

## Cascade Effects After Fixes

After fixing the numbers above, these text passages also need updating:

| # | Location | Current Text | Needs Update | Status |
|---|----------|-------------|--------------|--------|
| E1 | L253 HLE paragraph | "LLM Selection reaches 58.00%" | Update to match D3 final value | TODO |
| E2 | L253 HLE paragraph | "$4.53/q---a $2.4\times$ cost increase" | Recalculate ratio after D3 | TODO |
| E3 | L262 BV paragraph | "ATTS leads on BabyVision" | If BV MV=23.20%, then MV also ties. Fix wording | DONE |
| E4 | L262 BV paragraph | "ahead of Majority Voting (22.42%)" | Update to 23.20% | DONE |
| E5 | Abstract | "outperforms or matches all baselines" | May need softening depending on D3 | TODO |
| E6 | Table 1a bold marking | LLM Selection bolded at 58% | Re-check who is best after D3 | TODO |
| E7 | Table 1d bold marking | ATTS and LLM Selection bolded at 23.20% | MV also 23.20% after D5 fix, needs bold too | TODO |

## Verified Correct (no changes needed)

- All LCB numbers (Table 1b): all match
- All ATTS accuracy numbers across 4 main benchmarks: all match
- All multi-model Med numbers: all match
- All effort ablation numbers (Table 4): all match
- All orch ablation numbers (Table 5): all match (except GPQA cost rounding R3)
- All AIME numbers (Table 6): all match (except Q2 Pass@1 edge case)
- All Self-Refine numbers: all match
- All Budget Forcing numbers: all match
- Conclusion numbers (57.00% HLE multi-model, 85.63% LCB multi-model): all match
- Orch ablation in-text (Haiku 3.14 explores, Sonnet 1.85): all match
- Effort ablation in-text (+13 HLE, +8.6 LCB): all match
