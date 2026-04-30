# Qwen3.6 GRPO Training — Strategy Pivot Log

**Status (2026-04-30):** ABANDON Qwen/Qwen3.6-35B-A3B (MoE). New target: `Qwen/Qwen3.6-27B`.
Reason: 2× A100-80GB local box has only 9 GiB residual workspace per card under
production token budget at 35B-A3B — every operating point either OOMs or
throttles GPU power to 27%. Switching to 27B raises residual workspace to
~24 GiB per card (2.7× expansion), which is enough to clear the power-gate ≥150 W.

---

## Why we are abandoning Qwen3.6-35B-A3B

### MoE has NO memory advantage at equal total params

vLLM must hold ALL 256 experts on GPU even though only 8 are active per token.
So the model footprint is dictated by total parameters, not active parameters.

For Qwen3.6-35B-A3B at TP=2:
- vLLM weights: 35.23 GB / card
- FSDP DP=2 shard: 35.23 GB / card
- Static co-residence: **70.46 GB / card → 9.5 GB residual workspace**

### Failure chain v7 → v20 (35B-A3B) — what we actually verified

| Ver | Key knob | Outcome | Lesson burned in |
|---|---|---|---|
| v7  | mem_util=0.70 | Init refused (free<utilization×total) | mem_util ceiling = (80 - FSDP shard) / 80 |
| v8  | mem_util=0.50 + max_num_seqs=64 | KV cache budget = 0 | profiler dummy forward eats from working budget |
| v9  | `param_offload=True` | 226 GiB CPU OOM at update_weights | offload pushes pressure to CPU; verl LoRA+sleep_level=1 path locks vLLM weights even when offloaded |
| v10 | max_num_seqs=8 + mem_util=0.5 | KV budget still 0 | dummy weight pool persists across profiler |
| v11 | mem_util=0.55 | cuda:0 init failed by 0.11 GiB | claude daemon 1.37 GB asymmetry between cards |
| v12 | mem_util=0.54 | update_weights GPU OOM (cuda:0 400 MB free) | tight static math; 0.01 mem_util margin not enough |
| v13 | enforce_eager=true + mem_util=0.55 | KV cache budget too small | save 5 GB CUDA graph but not enough |
| v14 | + chunked_prefill + max_num_batched_tokens=4096 | KV still tight | profiler peak = 4 GB cap |
| v15 | topology change → GPU 1+2 symmetric | update_weights OOM 1.57 GiB short | FSDP unshard transient = 35 GB summon-all peak |
| v16 | `lora.merge=True` | identical OOM | sleep level 1 keeps vLLM weights in memory regardless |
| v17 | **`layered_summon=True` + `load_format=safetensors`** | **update_weights PASS 0.88s** | layered_summon walks per-layer (440 MB transient) — THE update_weights solve |
| v18 | (same as v17, qwen-vl-utils installed) | rollout slow (27% power) | enforce_eager=true Python launch overhead caps SM utilization with max_num_seqs=4 |
| v19 | `mem_util=0.85 + param_offload=True + max_num_seqs=16` | rank 0 SIGKILL at update_weights | verl HYBRID + lora_as_adapter FORCES `sleep_level=1` (`vllm_async_server.py:636`) |
| v20 | revert to v18 baseline + max_num_seqs=8 only | CUDA OOM at rollout | 8 concurrent prefill peak (~4 GB activation) exceeds 8.77 GB working budget |

The 35B-A3B on 2×80 GB has only one stable operating point (v17/v18 baseline:
max_num_seqs=4, enforce_eager=true), and at that point GPU power is permanently
27% because the SMs are starved waiting for Python kernel launches. Any attempt
to increase concurrency or remove enforce_eager crosses the 9 GB working budget
ceiling and crashes.

---

## New target: Qwen/Qwen3.6-27B

### Architecture (verified via `https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json`, 2026-04-30)

| Field | Value |
|---|---|
| `architectures` | `["Qwen3_5ForConditionalGeneration"]` — same model class family as the failing 35B-A3B's `Qwen3_5MoeForConditionalGeneration` |
| `model_type` | `qwen3_5` |
| `language_model_only` | `false` (multimodal: includes vision encoder, `image_token_id`, `preprocessor_config.json`, `video_preprocessor_config.json`) |
| Total params | 27.78 B (BF16) → file sum 55.56 GB across 15 safetensors shards |
| Layers | 64 hybrid: 48× `linear_attention` (Gated DeltaNet / Mamba) + 16× `full_attention` (4-layer pattern × 16) |
| Hidden | 5120 |
| Attention heads | Q=24, KV=4, head_dim=256 |
| Linear (DeltaNet) heads | V=48, QK=16, head_dim=128 |
| FFN intermediate | 17408 |
| Vocab | 248320 |
| Native context | 262144 (extensible to 1.01 M with YaRN) |
| dtype | bf16 |

**Critical clarification (do NOT confuse with prior wrong claim):** 27B is NOT
"dense vanilla Qwen3 architecture". It is the same `Qwen3_5*ForConditionalGeneration`
family as 35B-A3B, just with linear_attention+full_attention layers replacing
the MoE expert routing. The bug class around vLLM PunicaWrapper LoRA support
(`visual.blocks.X.attn.qkv will be ignored` at 35B) may STILL surface here on
the linear_attention layers — to be verified at Phase 0.3 dummy load. We are
NOT bypassing the bug class; we are betting that the smaller working budget
will let us absorb whatever LoRA inefficiencies remain.

### Memory account (2× A100-80GB, TP=2 + DP=2, verified math)

| Item | Qwen3.6-35B-A3B (abandoned) | Qwen3.6-27B (new) |
|---|---|---|
| Safetensors total (bf16) | 70.46 GB | 55.56 GB |
| FSDP DP=2 shard | 35.23 GB / card | **27.78 GB / card** |
| vLLM TP=2 weights | 35.23 GB / card | **27.78 GB / card** |
| Static co-resident | 70.46 GB / card | **55.56 GB / card** |
| Free per card after FSDP wrap | ~9 GB | **~24 GB** |
| **vLLM mem_util ceiling** | (80 − 35.23) / 80 = 0.559 | **(80 − 27.78) / 80 = 0.652** |
| vLLM working budget at ceiling | 8.77 GB | **~24 GB (52.22 GB pool − 27.78 weights)** |
| Concurrency feasible | 4 (with enforce_eager) | **16-20 (without enforce_eager)** |
| Mamba SSM state per concurrent seq | n/a (35B is MoE) | 48 layers × 12 KB/tok ≈ 4.6 GB at 16×24K |
| Sustained GPU power expected | 27 % | **50–70 % (target)** |

### Knob delta from 35B-A3B v20 baseline → 27B v21

| Knob | 35B-A3B value | 27B value | Reason |
|---|---|---|---|
| `actor_rollout_ref.model.path` | `Qwen/Qwen3.6-35B-A3B` | `Qwen/Qwen3.6-27B` | new model |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.55 | **0.62** | new ceiling = (80-27.78)/80 = 0.652; leave 3% margin |
| `+actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager` | `true` | **STAY true** (REQUIRED workaround) | vllm#36372: enable_lora=True on Qwen3_5ForConditionalGeneration crashes at `column_parallel_linear.py:259 slice_lora_b` with IndexError during `profile_cudagraph_memory → maybe_dummy_run_with_lora`. Bug fires only when CUDA graph capture is on. With enforce_eager=True, that path is skipped. Phase 0.3 dummy load PASS confirmed 2026-04-30. |
| `+actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs` | 8 | **20** | Phase 0.3 measured `Available KV cache memory: 21.2 GiB` and `Maximum concurrency for 24,576 tokens per request: 25.31x`. Set 20 with 25% safety margin. |
| `++actor_rollout_ref.model.fused_kernel_options.impl_backend` | `triton` | **drop** | the triton override was a 35B-MoE-specific monkey-patch gate. Non-MoE Qwen3_5 path activates `use_fused_kernels=True` without it. |
| `++actor_rollout_ref.rollout.layered_summon` | `True` | **STAY True** | At update_weights: vLLM weights live (27.78) + FSDP shard (27.78) + summon-all transient (27.78) = 83.34 GB on 80 GB → still OOM by 3 GB. Same mechanism as 35B (just less tight). Per-layer summon transient is 27.78/64 ≈ 434 MB → total peak 56 GB. |
| `actor_rollout_ref.rollout.load_format` | `safetensors` | **STAY safetensors** | Coupled to layered_summon — vLLM must preload base from disk; only LoRA delta gets synced. |
| `+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce` | `true` | **STAY true** | A100 PCIe (no NVLink) — `custom_all_reduce` hits CUDA `invalid argument`. |
| `+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_chunked_prefill` | `true` | **STAY true** | universal best practice for long contexts |
| `+actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_batched_tokens` | 4096 | **STAY 4096** | caps profiler dummy + per-step compute |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | False | **STAY False** | for LoRA (frozen base, MB-scale adapters) offload buys nothing |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | False | **STAY False** | adapter-only optimizer state is trivial |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 2 | **STAY 2** | 27.78 GB doesn't fit one card after FSDP shard |

### Knobs that stay (proven baseline, NEVER regress)

- `model_dtype=bfloat16` (both actor and ref)
- `gradient_checkpointing=True`, `use_liger=True`, `use_fused_kernels=True`
- `use_remove_padding=True`, `entropy_from_logits_with_chunking=True`
- `use_dynamic_bsz=True`, `forward_prefetch=True`, `reshard_after_forward=True`
- Production token budget (NON-NEGOTIABLE):
  `max_prompt=8192`, `max_response=16384`, `max_model_len=24576`,
  `rollout.n=8`, `max_assistant_turns=9`, `format=hermes`,
  `kl_loss_coef=0.001`, `lora_rank=64`, `lora_alpha=128`,
  `target_modules=all-linear`
- `LD_LIBRARY_PATH` cu13 patch for bitsandbytes `libnvJitLink.so.13`
- Symmetric memory principle: GPU 1+2 (clean), judge on GPU 0
- Conda env: `explain-verl` (vllm 0.20.0, torch 2.11+cu130, verl 0.7.1, transformers 5.7.0, peft 0.18.0, flash-attn 2.8.3)

---

## TODO list — restart from scratch on Qwen3.6-27B

### Phase 0 — preparation

- [x] **0.1 Confirm GPU topology still holds**
  - GPU 0: judge (Qwen3-8B vLLM serve port 8000) + claude embedding daemon
  - GPU 1, 2: training (clean, 81 GB free each, symmetric)
  - GPU 3: namdo train_esm.py (still occupied, 56 GB)
- [x] **0.2 Pull Qwen3.6-27B safetensors weights**
  - 55.56 GB → /data1/peijia/hf_cache (chosen: /data2 was full at 20 GB; /data1 has 348 GB)
  - `huggingface_hub.snapshot_download(repo_id='Qwen/Qwen3.6-27B', cache_dir='/data1/peijia/hf_cache')`
  - `huggingface-cli download` does NOT work under `conda run --no-capture-output`; use the Python API
- [x] **0.3 Verify vLLM 0.20.0 supports Qwen3.6-27B dense load + LoRA — PASS (workaround discovered)**
  - **Finding**: vLLM 0.20.0 + `Qwen3_5ForConditionalGeneration` + `enable_lora=True` + `enforce_eager=False` triggers
    `IndexError: too many indices for tensor of dimension 1` at `column_parallel_linear.py:259 slice_lora_b` during
    `profile_cudagraph_memory → maybe_dummy_run_with_lora` (vllm#36372, related #36395 closed, #36603 still open).
  - **Workaround**: set `enforce_eager=True`. vLLM logs `"disabling torch.compile and CUDAGraphs"` and skips the
    buggy code path. Phase 0.3 dummy load `LORA_EAGER_PASS in 139.6s`, sample output correct.
  - **Why 35B v17/v18 worked silently**: 35B-A3B launcher had `enforce_eager=True` (because it needed the 5 GB
    saved to keep KV cache feasible at 9 GB workspace). It NEVER took the bug path. The "27B is simpler than 35B"
    intuition is misleading — both are `Qwen3_5*ForConditionalGeneration`, both have the bug, but 35B happened to
    have the workaround already because it was forced by memory constraints.
  - **Capacity measured**: `Available KV cache memory: 21.2 GiB`, `Maximum concurrency: 25.31x` at 24K tokens.
    27B's KV pool is 3.7× larger than 35B's 5.77 GB → max_num_seqs=20 is safe (vs 35B's cap of 8).
  - PunicaWrapper warnings on `visual.*` layers are non-fatal (vision encoder LoRA, we don't train vision).
- [x] **0.4 Save current launcher as `m6_grpo_smoke_qwen36_35b_a3b.sh.archive_v20`**

### Phase 1 — port launcher to Qwen3.6-27B

- [ ] **1.1 New launcher: `m6_grpo_smoke_qwen36_27b.sh`**
  - Apply the knob delta table above
  - Every override carries inline comment per `comment_on_config_overrides` rule
  - Separate `CKPT_DIR` (`tmp/grpo_smoke_qwen36_27b`) to avoid loading 35B state
- [x] **1.2 Verify training data still compatible** — same Hermes orchestrator format, 4240 train rows, schema intact
- [ ] **1.3 Smoke launch with `total_training_steps=1`, `val_before_train=False`**
  - Acceptance: no crash + finite loss + ckpt saved + sustained GPU power ≥ 50% (≥150 W on GPU 1+2)
  - Attach Monitor immediately on launch (per `feedback_monitor_tool_after_script_launch`)

### Phase 2 — performance validation

- [ ] **2.1 Confirm step 0 wall-clock** (expected 5–15 min for first rollout + actor update)
- [ ] **2.2 Confirm sustained power gate ≥ 150 W on GPU 1+2 during rollout**
  - Power monitor task `bl466q61q` (GATE_PASS_50pct state)
  - Failing this means the working-budget hypothesis was wrong → escalate
- [ ] **2.3 Confirm CUDA-graph capture is on** (no `enforce_eager` log message)
- [ ] **2.4 Capture PEFT trainable params count**
  - `print(model.print_trainable_parameters())` to verl model load hook
  - Validates LoRA all-linear actually applies; warn if Mamba layers silently skipped

### Phase 3 — production-scale smoke (3–5 steps)

- [ ] **3.1 Bump `total_training_steps=5`, run end-to-end**
- [ ] **3.2 Validate KL stays bounded, loss is finite each step**
- [ ] **3.3 Validate ckpt save + resume works**

### Phase 4 — full training (after smoke green)

- [ ] **4.1 Compute per-QID exposure target** (match prior peak step exposure)
- [ ] **4.2 Set `total_epochs` and `test_freq` accordingly**
- [ ] **4.3 Launch full training run** with W&B logging on
- [ ] **4.4 Set up checkpointed resume path** (HF Hub upload after every save_freq)

---

## Files / paths reference

- Old launcher (archived): `training/scripts/m6_grpo_smoke_qwen36_35b_a3b.sh.archive_v20`
- New launcher (Phase 1): `training/scripts/m6_grpo_smoke_qwen36_27b.sh`
- Judge launcher: `training/scripts/serve_judge_1gpu.sh` (unchanged)
- Main training log: `tmp/m6_grpo_smoke_verl.log`
- Power monitor task: `bl466q61q` (GPU 1+2, ≥150 W gate)
- Download log: `tmp/qwen36_27b_download.log`
- Project CLAUDE.md (perf rules + symmetric memory rule):
  `/home/peijia/dr-claw/Explain/CLAUDE.md`
- Conda env for training: `explain-verl`
- Conda env for judge: `verl` (separate, vllm-only)

---

## References

- [Qwen/Qwen3.6-27B HuggingFace](https://huggingface.co/Qwen/Qwen3.6-27B) — config.json verified 2026-04-30
- [vllm-project/vllm#40005](https://github.com/vllm-project/vllm/issues/40005) — Qwen3.5-MoE expert LoRA not loadable
- [vllm-project/vllm#35286](https://github.com/vllm-project/vllm/issues/35286) — Qwen3.5-MoE enable_lora IndexError
- [vllm-project/vllm#38085](https://github.com/vllm-project/vllm/issues/38085) — Qwen3.5 LoRA target modules
- [verl-project/verl#3516](https://github.com/volcengine/verl/issues/3516) — layered_summon vs sleep_level interaction
