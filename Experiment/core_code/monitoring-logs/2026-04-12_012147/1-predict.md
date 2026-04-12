# Step 1: Predictions (written BEFORE reading training evidence)

## Job identity
- Script: `scripts/training/train_grpo_vllm_8b_base_hle_3gpu.sh`
- Log: `tmp/grpo_8b_base_hle_3gpu.log`
- Experiment: `atts-grpo / 8b-base-hle-k16-3gpu`
- Session 1 (no prior monitoring state)

## Known spec (from script/config, not from training evidence)
- Model: **Qwen/Qwen3-8B (raw base, NOT SFT-initialized)** -- `sft_qwen3_8b_merged` exists but is not being used
- GPUs: 0,1,2 (GPU3 marked unusable); DP3 rollout, FSDP ZeRO-2 actor
- Rollout: vLLM, TP=1, gpu_memory_utilization=0.35, max_model_len=32768, response_length=24576, n=16, temperature=1.0
- Multi-turn: enabled, max_assistant_turns=10, Hermes format, tool_response_truncate=2048
- Training: train_batch=63 prompts/step → 1008 rollouts/step; mini_bs=63, micro_bs_per_gpu=2
- LR=1e-6, KL coef=0.001, low_var_kl, no KL in reward
- Data: train.parquet = 180 rows, val.parquet = 20 rows → ~2-3 steps/epoch (2 if any overlong filter kicks in); total epochs=30 → ~60-90 steps target
- save_freq=6, test_freq=3, val_before_train=True
- Logger: console + wandb; W&B project `atts-grpo`

## Predictions (what I expect when I look at evidence)

### Process / infra
- P1: main Python process alive (verl.trainer.main_ppo), child vLLM workers alive
- P2: GPUs 0/1/2 utilization > 70% during rollout phases; GPU 3 idle (0% util, ~0 MiB)
- P3: GPU memory ~55-75 GB/80 GB per GPU (vLLM 0.35 cache + FSDP ZeRO-2 8B weights + activations)

### Progress indicator
- P4: `val_before_train=True` should have produced an initial validation block at the very top of the log, BEFORE any "step 1" training.
- P5: If log shows step number, current step should be in [0, ~30] given this run is fresh (no previous atts-grpo/8b-base-hle checkpoint). Specifically:
  - step 0..1 if just started
  - step 6+ if at least one save has happened (there's no 8b-base-hle checkpoint dir yet, so I predict step < 6)
- P6: No saved checkpoint in `checkpoints/atts-grpo/` for this experiment name yet -- confirmed.

### Wall time
- P7: GRPO step wall time: 15-40 min/step (rollout-dominated: 1008 rollouts × multi-turn × up to 24576 tokens on 3 GPUs is expensive)
- P8: val_before_train wall time: 5-15 min (20 prompts, n=1, temperature=0, but multi-turn is still live)

### Metrics (rough priors)
- P9: **Reward low / near 0 at start** -- starting from **raw base** Qwen3-8B means the model has never seen Hermes tool-call format. I predict initial val reward is close to 0 (< 5%) and early-step rewards are similarly low. This is my biggest concern: starting GRPO from raw base, without SFT warmup, is usually a pathological setup because the policy emits ~0 tool calls in a valid format, so reward signal is ~uniformly 0 and GRPO has no gradient to climb.
- P10: policy_loss finite (not NaN), kl_loss small (< 0.01), grad_norm moderate (< 10), pg_clipfrac low
- P11: rollout response length trending toward max (base model likely rambles; truncation=error means any overlong will crash -- this is a risk flag)

### Anomalies I am actively looking for
- A1: **Zero or near-zero reward stuck across all steps** (GRPO dead signal -- expected if starting from base)
- A2: Truncation=error crashes if any response > 24576 tokens
- A3: NaN/Inf in loss/grad, NCCL timeouts, OOM
- A4: Validation % eating > 20% of wall time (test_freq=3 on 20-prompt val set could be heavy)
- A5: Ghost process / silent hang (last log line old; GPUs idle)

## Explicit uncertainties
- U1: Is this a fresh run or a resume? (No `@latest` checkpoint exists under `atts-grpo/8b-base-hle-k16-3gpu`, so I lean fresh.)
- U2: How long has it been running? Unknown until I read the log.
- U3: Why starting GRPO from base and not SFT? (User decision, possibly deliberate.)
