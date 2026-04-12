# Step 2: Evidence

## Process / infra state (at inspection time 2026-04-12 01:23)
- `ps -ef | grep verl|main_ppo|grpo_vllm` → **zero matches**. Training process is gone.
- `nvidia-smi`:
  - GPU0: 0% util, 0 MiB used, 45C, 48 W
  - GPU1: 0% util, 0 MiB used, 45C, 50 W
  - GPU2: 0% util, 0 MiB used, 44C, 48 W
  - GPU3: 58% util, 36897 MiB, 58C, 180 W → **other user** (`pid=1887150`, user `namdo`, `python train_esm.py --cfg-path configs/genechat_stage3_reinforce.yaml`, etime 03:29:13). This GPU was excluded from our script (`CUDA_VISIBLE_DEVICES=0,1,2`).
- Log file `tmp/grpo_8b_base_hle_3gpu.log`:
  - Birth: 2026-04-11 23:52:34
  - Last modify: 2026-04-12 01:21:50
  - `wc -l` stable across reads → **not growing**
  - Size: 464 KB, 3117 lines
- Disk: `/data3 is 99% full`, 184 GB free of 18.5 TB. Ray logged this warning every ~10s throughout the run (line 16 onward).
- W&B run dir: `wandb/run-20260411_235632-qldsho5h/` (experiment 8b-base-hle-k16-3gpu)

## Training timeline (from log timestamps)
| Time (UTC) | Line | Event |
|---|---|---|
| 23:52:34 | birth | process started |
| 23:52:47 | 10 | Ray worker init |
| 23:53:12 | 11 | Ray local instance ready |
| 23:55:45 | 964-1012 | vLLM http server replicas up (3x, one per GPU) |
| (~00:0?) | 1082 | `test_gen_batch`: `global_steps=0, validate=True` -- val_before_train begins |
| (~00:??) | 1295-1301 | val_before_train complete: `step:0 - val-core/atts_hle/acc/mean@1:0.0` |
| same | 1302 | `Training Progress: 0%|  | 0/60 [00:00<?, ?it/s]` -- training loop enters |
| 01:21:39 | 2828+ | First `RayTaskError(OutOfMemoryError)` from `ref_compute_ref_log_prob` |
| 01:21:39 | 2984 | `Training Progress: 0%| | 0/60 [1:17:50<?, ?it/s]` -- confirms step 0 never completed |
| 01:21:39 | 3113 | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.87 GiB` |
| 01:21:39 | 3116 | vLLM `Engine core proc EngineCore_DP0 died` |
| 01:21:50 | 3117 | `conda run ... failed` (final line) |

- Wall clock: 1h 29min, **training steps completed: 0**
- Total training steps target: **60** (confirmed by `Training Progress: 0/60`. Train dataloader size=2, val=1, epochs=30 → 2×30=60.)

## val_before_train metrics (step 0 baseline)
From log line 1301 (only validation block ever produced):
```
val-core/atts_hle/acc/mean@1:           0.0
val-aux/atts_hle/reward/mean@1:        -0.2775
val-aux/atts_hle/score/mean@1:         -0.2775
val-aux/atts_hle/has_answer/mean@1:     0.3   (30% produced parseable answer)
val-aux/atts_hle/penalty/mean@1:        0.2775
val-aux/atts_hle/num_explores/mean@1:   5.55
val-aux/num_turns/min:                  2
val-aux/num_turns/max:                  20    (exceeds max_assistant_turns=10 set in config)
val-aux/num_turns/mean:                 15.3
```
- Base Qwen3-8B on HLE val (20 prompts): **0% accuracy**, net reward **-0.2775**. 70% of prompts produced no parseable answer. Mean 15.3 turns (confirmed cap-hit behavior).
- `num_turns` max=20 > configured `max_assistant_turns=10` → likely counts both assistant turns and tool responses, or verl's counter includes system/user turns. Not a crasher but worth noting.

## Crash root cause (log line 3113 + traceback)
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.87 GiB.
GPU 0 has a total capacity of 79.25 GiB of which 3.84 GiB is free.
Including non-PyTorch memory, this process has 74.09 GiB memory in use.
Process 2968307 has 1.27 GiB memory in use.
```
Call stack:
- `verl/trainer/ppo/ray_trainer.py:1440` → `self._compute_ref_log_prob(batch)`
- `ray_trainer.py:1128` → `self.ref_policy_wg.compute_ref_log_prob(batch)`
- `verl/workers/fsdp_workers.py:1164` → `compute_ref_log_prob`
- `verl/workers/actor/dp_actor.py:479` → `compute_log_prob` → `_forward_micro_batch`
- `transformers/models/qwen3/modeling_qwen3.py:520` → `self.lm_head(hidden_states[:, slice_indices, :])`
- `torch/nn/modules/linear.py:134` → `F.linear(input, self.weight, self.bias)`

**Interpretation**: OOM on the final `lm_head` projection while computing **reference policy log_prob**, not rollout and not actor backward. This is after rollout + actor log_prob succeeded, so the issue is specifically the ref policy pass.

Memory budget on GPU 0 at crash:
- vLLM engine KV cache / paged memory from `gpu_memory_utilization=0.35` → ~28 GB persistent (vLLM was alive the whole time, engine died only after OOM)
- FSDP ZeRO-2 with `reshard_after_forward=False` holds the full 8B bf16 weights gathered (~16 GB) during ref forward
- Activations for `ppo_micro_batch_size_per_gpu=2` at total seq up to `prompt 4096 + response 24576 = 28672`
- `lm_head` output logits `[2, 28672, 151936]` in bf16 = ~17.3 GB contiguous tensor → this is the 7.87 GiB bump the message references (allocator attempts at lm_head chunk)
- Total adds to 74 GB used / 80 GB available → 3.84 GB free → cannot fit the next alloc

**Why step 0 rollout+actor succeeded but ref OOM'd**: rollout is vLLM; actor log_prob happens under the actor's FSDP module; ref log_prob loads a **second** 8B model state (reference policy) and runs forward with the same long sequences, while vLLM still holds its KV reservation.

## Spec vs actuality (quick)
- Script total_epochs=30, n_gpus=3, batch=63, but dataloader size printed as `train=2, val=1` → after `filter_overlong_prompts=True` many prompts may have been dropped, or 180 prompts / 63 per step × 30 epochs yields 2 steps/epoch × 30 = 60. Matches the "0/60" progress bar.
- `val-aux/num_turns/max = 20` despite `max_assistant_turns=10` → counter semantics don't match config keyword. Not blocking.

## Summary of gate findings
1. Training is **DEAD** (OOM crash). Zero training steps completed.
2. ~1h 30min of 3-GPU wall time wasted; no checkpoint saved.
3. Root cause: ref policy log_prob forward OOM'd due to long-sequence `lm_head` projection sharing GPU with vLLM KV reservation.
4. Initial validation shows 0% accuracy on HLE (expected for raw base Qwen3-8B without SFT warmup) — GRPO signal would have been extremely sparse even if training had run.
5. Secondary: `/data3` at 99% full — would eventually block checkpoint save.
6. Secondary: GPU 3 is being used by another user (`namdo`), which is why the script was 3-GPU only.
