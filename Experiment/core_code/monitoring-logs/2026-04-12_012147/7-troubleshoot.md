# Step 7: Troubleshoot

## Anomaly A-OOM: CUDA OOM in `_compute_ref_log_prob`

### Observation (numbers, from the log)
- `torch.OutOfMemoryError: Tried to allocate 7.87 GiB. GPU 0 ... 3.84 GiB is free ... 74.09 GiB memory in use` (line 3113)
- Location: `verl/workers/fsdp_workers.py:1164 → dp_actor.py:479 → modeling_qwen3.py:520 → self.lm_head(hidden_states[:, slice_indices, :]) → F.linear`
- Phase: reference policy log_prob forward, after rollout + actor log_prob succeeded, before any optimizer update.

### Memory reconstruction on GPU 0 at crash (quantified)
| Component | Size | Source |
|---|---|---|
| vLLM KV cache reservation | ~28 GB | `rollout.gpu_memory_utilization = 0.35 × 80 GB` (vLLM engine was still alive when ref pass ran; engine died only after OOM) |
| Ref Qwen3-8B weights (bf16, gathered in forward) | ~16 GB | `ref.fsdp_config.reshard_after_forward=True` (confirmed line 428); but gather is still required for the forward itself |
| Forward activations (grad_ckpt=True helps but bounded) | few GB | `enable_gradient_checkpointing=True` |
| `lm_head` output logits `[2, 28672, 151936] bf16` | **~17.4 GB** contiguous | micro_bs=2, seq = prompt 4096 + response 24576 = 28672, vocab 151936, 2 bytes |
| Sum | **~63-65 GB of the 74 GB** | — |
| Free at allocation time | 3.84 GB | log message |
| Attempted alloc | 7.87 GB | log message |

**Conclusion**: the `lm_head` output tensor is the dominant variable. Every step that reduces its size or the headroom fixes this.

### Comparison to the working `8b-sft-3gpu` run (same hardware, same vLLM, same verl)
This comparison is the single strongest piece of evidence — the SFT run finished through at least step 78 without OOM, and the delta set is small enough to isolate the cause.

| Parameter | Working SFT run | Crashing base run | Effect on ref lm_head alloc |
|---|---|---|---|
| Model init | `sft_qwen3_8b_merged` | raw `Qwen/Qwen3-8B` | SFT is concise, base rambles to max cap |
| `max_response_length` | 4096 | **24576 (6×)** | seq: 8192 vs 28672 |
| `max_model_len` | 12288 | **32768** | allows longer KV cache |
| `rollout.n` | 8 | **16 (2×)** | affects rollout throughput, not per-seq memory |
| `max_assistant_turns` | 5 | **10 (2×)** | multi-turn → effective seq length bound |
| `ref.log_prob_micro_batch_size_per_gpu` | 4 | **2** | halved, but seq length dominates |
| `rollout.gpu_memory_utilization` | **0.5** | 0.35 | SFT had *more* vLLM memory and still didn't OOM |
| Worst ref `lm_head` tensor | `[4, 8192, 152K] bf16 ≈ 9.9 GB` | `[2, 28672, 152K] bf16 ≈ 17.4 GB` | **+7.5 GB** — matches the missing headroom exactly |

**Root cause**: the SFT → base change allows responses to stretch to 24576 tokens; the `lm_head` allocation at that seq length (even with micro_bs=2) pushes the alloc over the free headroom. The choice of 24576 + base model is the inciting factor; everything else compounds.

## Anomaly A-DISK: /data3 at 99% full

### Observation
- `df -h /data3 → 18T used, 184 GB free, 99%`. Ray warned continuously (`file_system_monitor.cc:116`) from line 16 through line 2900+.
- `checkpoints/atts-grpo/8b-sft-3gpu` = **458 GB**. Contains checkpoints at steps 6, 9, 12, 15, 18, 21, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78. `max_actor_ckpt_to_keep` was 2 but old ckpts were not cleaned because the save list includes more than `actor/` (optimizer + extras).
- No new checkpoint would fit at the first `save_freq=6` without freeing space.

### Observation (not a finding, but plausible angle)
- `rollout.free_cache_engine` / vLLM `enable_sleep_mode=True` setting was logged (line 967). In principle vLLM should release GPU memory before ref log_prob. Empirically GPU 0 had 74 GB used at ref time, which suggests either the sleep path didn't run or it doesn't release the full KV reservation. This is one of the candidate memory fixes below — it can be investigated without changing the effective config.

## Anomaly A-INIT: strategic — raw base as GRPO starting point

### Observation
- val_before_train: `acc=0.0, has_answer=0.3, num_turns_mean=15.3 (cap=10 configured)`. The raw base model produces few parseable answers and rambles to the turn cap.
- GRPO advantages are computed group-relative over `n=16` samples per prompt. If all 16 samples in a group have acc=0, their advantage is 0 → no gradient. With has_answer=0.3, most groups have 0 correct samples → **most prompts contribute no gradient**.
- A **working SFT-initialized checkpoint already exists** (`checkpoints/sft_qwen3_8b_merged`) and was used by the successful `8b-sft-3gpu` run. Starting from it would simultaneously:
  1. Give GRPO actual signal to climb (has_answer≫0.3)
  2. Shorten responses (directly fixes A-OOM without config changes)
  3. Copy a recipe already proven on this hardware
- If the user's research question is specifically *"does GRPO learn tool-calling from raw base"*, then this is a deliberate choice, not a bug. **This is a question for the user, not a thing I should decide.**

## Candidate fixes (ordered by confidence and blast radius)

### Fixes that address A-OOM alone (minimal change)
Each of the following, applied *singly*, would almost certainly resolve the OOM on the current config:

1. **`data.max_response_length: 24576 → 12288`** and **`rollout.response_length: 24576 → 12288`** and **`rollout.max_model_len: 32768 → 20480`**
   - Halves the ref lm_head tensor to ~8.7 GB, frees ~8.7 GB per GPU
   - Changes training semantics (shorter CoT budget for RL learning) — needs user consent
2. **`rollout.gpu_memory_utilization: 0.35 → 0.2`**
   - Frees ~12 GB per GPU during training phases
   - Reduces vLLM KV cache → slower rollout, but rollout was already dominating step time
3. **`actor_rollout_ref.ref.fsdp_config.param_offload: False → True`**
   - Offloads ref params to CPU between forwards, recovering ~16 GB per GPU
   - Slowest option but bulletproof
4. **`actor_rollout_ref.actor.use_dynamic_bsz: True` + `actor_rollout_ref.actor.ppo_max_token_len_per_gpu: 16384`** and the matching `rollout.log_prob_use_dynamic_bsz=True` with `log_prob_max_token_len_per_gpu`
   - Verl splits long sequences automatically so the ref forward never sees seq=28672 as a single batch
   - Better long-term hygiene than hardcoding micro_bs=2

### Combined fix that also resolves A-INIT (strategic)
5. **Use `MODEL_PATH=$PROJECT_DIR/checkpoints/sft_qwen3_8b_merged`** + tighten the above envelope to the `8b-sft-3gpu` recipe for the knobs known to work:
   - `max_response_length=4096` or `8192` instead of 24576
   - `n=8` instead of 16
   - `max_assistant_turns=5` instead of 10
   - `log_prob_micro_batch_size_per_gpu=4` instead of 2
   - This is the full "rebase on SFT recipe" option — very likely to run end-to-end, but it changes the **research question** (no longer "base model + GRPO").

### Disk fix for A-DISK
6. **Free `/data3`**: candidates in decreasing safety:
   - Drop old `8b-sft-3gpu` checkpoints that the user no longer needs for analysis (458 GB available)
   - Move old checkpoints to other storage
   - Separately: set `trainer.max_actor_ckpt_to_keep=1` and `save_freq=12` to reduce per-run pressure

## What I will NOT decide without the user
- Whether the research question is specifically "GRPO from base" (if yes, stay at raw base + fix memory only)
- Whether shortening `max_response_length` is acceptable (affects research design)
- Which checkpoints on `/data3` are safe to delete
- Whether to reduce `total_epochs=30 → 3` to match the working SFT run (30 epochs × ≥1.5h/step ≈ 90+ hours is a big commitment if the goal is exploratory)

These go to Step 8 as the user-facing decision menu.
