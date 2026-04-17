# Step 3: Predictions vs Actuals

| Prediction | Predicted | Actual | Deviation? |
|-----------|-----------|--------|------------|
| P1: Initial reward | >0.4 | 0.326 (step 4) | Minor miss — lower than expected but within range. Step 4 is still early. |
| P2: Run health | HEALTHY or completed | CRASHED at step 4 | **MAJOR DEVIATION** — run crashed, not healthy. |
| P3: Step count | Mid-run | Step 4 of ~15 epochs worth | Match — early in run, confirmed mid-startup crash. |
| P4: Judge crash | Low probability if ongoing | Judge server still alive (GPU 2) | Partial: Judge server alive but its response caused the crash (truncation), not a dead-judge scenario. |
| P5: Memory pressure | No OOM | No OOM | Match — crash was AssertionError not OOM. |
| P6: Val accuracy at step 0 | >0.3 | val-core/acc/mean@4=0.260, best@4=0.425 | Match — baseline is reasonable for SFT init. |

## Critical Deviation Requiring Investigation

**P2 is wrong**: run is CRASHED, not healthy. The crash mechanism differs from the predicted "Judge dead" scenario:
- Predicted: Judge process dies → metric variance collapse
- Actual: Judge server alive but returned `finish_reason=length` on one grading call → `_judge_remote` assert fires → Ray exception → trainer abort

This is a **reward_fn assertion failure on judge output truncation**, not a judge crash. The assert at `reward_fn.py:106` has zero tolerance for truncation.
