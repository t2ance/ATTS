# Step 4: Metric Analysis

## Judgment criteria derived from this job's own artifacts

### Objective (from `reward_fn.py` + observed val metrics)
The run optimizes a reward = `score - penalty` aggregated across an ATTS agent rollout on HLE prompts. The score signal decomposes as: `has_answer` (parseable output), `acc` (correct answer), `num_explores`, `num_turns`. Step 0 validation showed:
- `acc = 0.0`, `has_answer = 0.3`, `penalty = 0.2775`, `reward = -0.2775`.

For GRPO to learn, either `acc` or `has_answer` must improve across steps. The **most tractable early signal** is `has_answer` rising (learning to emit structured output), then `acc` rising.

### Schedule (from script + dataloader print)
- `Size of train dataloader: 2`, `val dataloader: 1`. Epochs=30 → **total training steps = 60**.
- `test_freq=3` → val runs every 3 training steps → 20 validation passes over the whole run.
- `save_freq=6` → 10 checkpoints saved. Each 8B actor+optim state ≈ 50-70 GB → total ckpt budget 500-700 GB (but only 2 kept via `max_actor_ckpt_to_keep=2` → ~140 GB peak).

### Per-step wall-clock budget
- Observed: step 0 wallclock up to crash ≈ 1h 17m 50s, and training never even reached the optimizer update.
- Rollout dominates because the multi-turn tool agent has 1008 rollouts × mean 15.3 turns × up to ~1600 tokens/turn, plus tool-execution latency between turns.
- Extrapolated minimum per step (if ref_log_prob OOM is solved and no other regression): ≥1h rollout + 10-20 min log_prob + 5-10 min optimizer ≈ 1.5h/step.
- 60 steps × 1.5h = **≥90 hours wall clock** end-to-end. At 3 A100s, that is a significant compute commitment.

### Health criteria (if the run were live and progressing)
| Metric | Derivation | Healthy band |
|---|---|---|
| `val-core/atts_hle/acc/mean@1` | step-0 baseline = 0.0; reward_fn weights acc heavily | monotonic non-decreasing by step 3 or 6; positive slope by step 12 |
| `val-aux/atts_hle/has_answer/mean@1` | step-0 baseline = 0.3 | should rise faster than acc; target ≥0.8 by step 12 |
| `actor/kl_loss` | `kl_loss_coef=0.001`, `low_var_kl` | growing slowly, stay <1.0 for the duration |
| `actor/policy_loss` | GRPO vanilla, clip=0.2 | finite, non-NaN, no sign flips |
| `actor/pg_clipfrac` | clip_ratio=0.2 | <0.3 typical |
| rollout mean response length | 24576 cap | should **drop** from start (base rambles) as the model learns to terminate |
| `aborted_ratio` / overlong rollouts | `truncation=error` → any overlong crashes | must stay 0 |

### Current status vs criteria
- Only step 0 baseline data exists. No learning signal can be evaluated because **training never produced a single optimizer step**. Status cannot be "healthy" or "warning with signal" — it is **CRITICAL (crashed)**.

## Multi-GPU / distributed aspects (distributed-monitor heuristics)
- Ray driver is dead and no worker PIDs remain (`ps -ef | grep verl` empty). This is not a launcher-alive / workers-dead split; everything is gone. Consistent with a clean Ray-level shutdown after the OOM cascade.
- No NCCL timeout evidence. The crash is a pure CUDA OOM inside `F.linear(...)` on rank 0, surfaced before any collective op hung.
- GPU memory was not uneven between ranks (GPUs 0/1/2 got the same workload under FSDP ZeRO-2); the crash would happen identically on any rank that hit the same ref_log_prob path first.

## GRPO-specific anomalies (grpo-monitor heuristics)
- **Generation phase dominates step time** (skill reference says 5-15% of step time; here rollout likely ≥50-80%). The tool-agent loop inflates rollout time because tool latency is serialized per turn. This is not fixable without pipelining tool calls across sequences in a worker (out of scope for immediate remediation).
- **Reward distribution at step 0**: mean -0.2775, acc=0, has_answer=0.3. GRPO advantages are computed group-relative (per prompt, over n=16 samples). If within each group all 16 samples return 0 acc, the advantage is 0 across the group → **no gradient signal for that prompt**. The only non-zero advantages will come from groups where 1+ samples happen to produce a parseable answer. This is the pathology I flagged in P9/A1 for base-model init.
- **Reward hacking risk**: penalty is ~linear in num_turns (5.55 explores + 15.3 turns → big penalty share). Once training starts, one easy gradient is "make num_turns shorter". The reward could improve purely by shortening without accuracy improving, which would be reward hacking against `penalty`.

## Status: **CRITICAL**
Evidence: process dead, zero steps completed, explicit OOM traceback, no recoverable state.

## Anomalies listed for Step 7 (troubleshoot)
- **A-OOM** (primary): `_compute_ref_log_prob` OOM on GPU 0 → must be fixed before any further run. Falsifiable via a small re-run with one of the candidate fixes.
- **A-DISK** (secondary): `/data3` at 99% full. Blocks future checkpoint saves and risks Ray spill failure. Must be freed before a retry.
- **A-INIT** (strategic): base model init at 0% acc with num_turns hitting cap. GRPO may find no signal in 60 steps. SFT-initialized checkpoint (`sft_qwen3_8b_merged`) is already available and would be a stronger starting point.
- **A-TURNS** (minor): `num_turns/max=20` vs `max_assistant_turns=10` counter mismatch — diagnostic noise, not a crasher, flagged but non-blocking.
