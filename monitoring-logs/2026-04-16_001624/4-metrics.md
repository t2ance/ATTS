# Step 4: Metric Analysis

## Derived Judgment Criteria

From the training config:
- **Progress indicator**: `critic/rewards/mean` (primary), `val-core/atts_hle/acc/mean@4` (secondary)
- **KL health**: `actor/ppo_kl` < 0.1 (kl_ctrl.target_kl in config) = healthy
- **Clip ratio**: `actor/pg_clipfrac` < 0.3 = healthy
- **Total epochs**: 15, save_freq=3, so expected steps per epoch = train_set_size / 96

## 4-Step Metric History

| Step | rewards/mean | val/acc/mean@4 | grad_norm | ppo_kl | time/step |
|------|-------------|----------------|-----------|--------|-----------|
| 1 | 0.302 | 0.212 | 1.844 | - | 2244s |
| 2 | 0.318 | 0.240 | 1.999 | - | 2203s |
| 3 | 0.312 | 0.240 | 2.402 | - | 2348s |
| 4 | 0.326 | 0.260 | 1.970 | 0.000423 | 2162s |

*ppo_kl only in summaryMetrics at step 4 = 0.000423 (well below target 0.1 — no KL danger)*

## Training Health Assessment (before crash)

**Reward trend**: 0.302 → 0.318 → 0.312 → 0.326. Slight non-monotone dip at step 3 but overall upward trend over 4 steps. Not stalled.

**Val accuracy**: 0.212 → 0.240 → 0.240 → 0.260. Monotonically non-decreasing. Step 3 plateau followed by step 4 gain. Consistent with reward dip at step 3.

**KL divergence**: `actor/ppo_kl` = 0.000423 at step 4 << 0.1 target. No divergence.

**Clip fraction**: `actor/pg_clipfrac` = 0.00198 at step 4 << 0.3 threshold. Updates not too aggressive.

**Grad norm**: Spike at step 3 (2.40) then recovered to 1.97. Normal variation, no explosion.

**Entropy**: `actor/entropy` = 0.251 at step 4 (from summaryMetrics). Non-zero, no collapse.

**Step timing (step 4)**:
- Rollout gen: 318s (14.7%) — within healthy range 5-15%
- Old log prob: 286s (13.2%)
- Ref log prob: 412s (19.1%) — slightly slow relative to gen
- Update actor: 1140s (52.8%) — at top of healthy range 30-50%
- Checkpoint save: 145s (6.7%) — healthy
- Testing/val: 71s (3.3%) — lightweight

The update_actor time (52.8%) dominates. This is expected with ppo_mini_batch_size=16 and ppo_micro_batch_size_per_gpu=1 creating many micro-batches per step.

## Overall Status: CRITICAL (crashed)

Training signal was HEALTHY. The crash is not a training degradation — it is a code-level assertion failure in the reward pipeline. The assert at `reward_fn.py:106` fires when the Judge vllm returns `finish_reason=length` (truncated at 4096 tokens).

**Root cause**: The `_JUDGE_SCHEMA` requires a `reasoning` field. The judge model (Qwen3-8B) generates verbose LaTeX in the reasoning field, whose JSON-escaped form exceeds 4096 tokens. The assert treats this as a fatal error.

**Recurrence risk**: HIGH. This is deterministic — the same problematic question will be sampled again on resume, triggering the same crash. The question involves LaTeX-heavy math (normal cone representation: `{s ∈ ℝ³| s₃ ≥ 0}`).

**Key observation**: The `reasoning` field is extracted and logged but never used in score computation (`return 1.0 if result["correct"] else 0.0`). The schema unnecessarily forces the judge to generate verbose reasoning, consuming most of the 4096-token budget.
