# Step 1: Predictions

## Job Identity

- Script: `train_grpo_vllm_8b_sft_2gpu.sh`
- Config: `/data3/peijia/dr-claw/Explain/Experiment/core_code/training/grpo`
- Model: `sft_qwen3_8b_merged` (SFT-initialized, not base)
- W&B run: `atts-grpo / 8b-sft-2gpu-bs96` (run ID: rilxxd8r)
- Checkpoint dir: `.../checkpoints/atts-grpo/8b-sft-2gpu-bs96`
- Log: `.../tmp/grpo_8b_sft_2gpu_bs96.log`

## Prior State

None. First monitoring session for this job.

## Predictions (written before reading any logs/metrics)

### P1: Initial reward level
SFT model achieved 45/50 (90%) pass rate on smoke test for tool-use format.
**Prediction**: train_reward/mean at early steps > 0.4, possibly 0.5+. Not starting from near-zero like base model.

### P2: Run health
Config is derived from 3-GPU variant with known-good 2-GPU memory disciplines (param_offload=False, gpu_memory_utilization=0.4, ppo_micro_batch_size_per_gpu=1). No prior crash incident recorded for this specific run.
**Prediction**: Run is HEALTHY or completed normally. Low probability of silent crash unless Judge process died (past incident pattern from memory).

### P3: Step count
total_epochs=15, train_batch_size=96, save_freq=3. No prior state to anchor on.
**Prediction**: Unknown. Could be anywhere from step 0 (just started) to epoch 15 (complete). Most likely mid-run if the user is asking to check it now.

### P4: Judge process
Past incident (memory: judge_crash_diagnosis.md) shows judge process crash causes metric equivalence (all rewards identical). 
**Prediction**: If run is ongoing, Judge is likely alive. If crashed, we expect train_reward variance to collapse.

### P5: Memory pressure
2 GPUs, max_model_len=12288, no param offload. gpu_memory_utilization=0.4 for vLLM is conservative.
**Prediction**: No OOM during training proper. vLLM resharding troughs possible but not indicative of crash.

### P6: Val accuracy
SFT init from high-quality checkpoint. Val accuracy should be reasonable from step 0 (val_before_train=True).
**Prediction**: val_accuracy/mean at step 0 > 0.3.

## Derived Judgment Criteria (to be confirmed from actual config/logs)

- **Progress indicator**: train_reward/mean (primary), val_accuracy/mean (secondary)
- **Health baseline**: step 0 val accuracy from val_before_train
- **Crash signal**: reward variance collapse (all-identical rewards = judge dead)
- **Completion signal**: trainer exits after epoch 15 OR latest_checkpointed_iteration.txt exists at final step
