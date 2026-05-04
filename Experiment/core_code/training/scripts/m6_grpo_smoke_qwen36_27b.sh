#!/usr/bin/env bash
set -ex

# M6 GRPO smoke for Qwen3.6-27B (hybrid linear_attention + full_attention).
# Goal: replace failed Qwen3.6-35B-A3B (MoE) v7→v20 with a smaller 27B target
# whose static co-residence (55.56 GB / card) leaves ~24 GB residual workspace
# per card vs the 9 GB of 35B-A3B — enough to clear power gate ≥150 W (≥50%).
#
# Architecture facts (verified from HF config.json 2026-04-30):
#   architectures: Qwen3_5ForConditionalGeneration  (same family as 35B-A3B's
#                                                    Qwen3_5MoeForConditionalGeneration)
#   layers: 64 hybrid (48 linear_attention / Mamba + 16 full_attention)
#   total params: 27.78 B (bf16 → 55.56 GB safetensors)
#   model_type: qwen3_5  (not "vanilla dense Qwen3" — multimodal VL with vision encoder)
#
# Topology (2026-05-03 — TP=1 DP=3 pivot, user directive: use GPUs 1+2+3 for training):
#   GPU 0 = judge Qwen3-8B vLLM serve (port 8000, mem_util=0.5 → 40 GB) + claude embedding daemon
#   GPU 1 = verl training rank 0 (FSDP shard 1/3 + full vLLM TP=1 weights)
#   GPU 2 = verl training rank 1 (FSDP shard 1/3 + full vLLM TP=1 weights)
#   GPU 3 = verl training rank 2 (FSDP shard 1/3 + full vLLM TP=1 weights)
# Why TP=1 DP=3 (not TP=3 DP=1):
#   Qwen3.6-27B text_config.num_key_value_heads=4. vLLM TP=N requires KV%N==0.
#   TP=3 → 4%3=1 → vLLM init refuses. Only TP=1 or TP=2 valid. With 3 cards, TP=1 + DP=3 is the
#   unique valid layout that uses all three GPUs.
# Memory account (verified by 2026-05-03 21:31:56 vLLM init crash, fixed):
#   vLLM weights/card  : 27.78 GB (TP=1 → full weights per card)
#   FSDP actor shard   : 27.78/3 ≈ 9.26 GB/card (DP=3 actor sharding)
#   FSDP ref shard     : 27.78/3 ≈ 9.26 GB/card (DP=3 ref sharding) — ref.fsdp_config.param_offload=False keeps ref on GPU
#   Static co-resident : 27.78 + 9.26 + 9.26 = 46.30 GB / card  (NOT 37.04 — actor + ref BOTH live)
#   Working budget     : 80 - 18.52 (FSDP actor+ref) - 27.78 (vLLM weights) = ~33 GB / card
#   mem_util ceiling   : (79.25 observed total - 18.52 FSDP)/79.25 = 0.766 → use 0.72 with 5% margin (set below)
# Symmetric memory: GPU 1+2+3 all start at 81.1 GB free → identical, vLLM init won't be capped by laggard.
# WHY 0.78 was wrong (2026-05-03): launcher v1 of TP=1 DP=3 only counted ACTOR FSDP shard (9.26 GB), missed REF FSDP shard (also 9.26).
# vLLM init refused with "Free memory 60.46/79.25 GiB < desired 0.78 (61.81 GiB), short by 1.35 GiB". Verified observation = both shards live.
export CUDA_VISIBLE_DEVICES=1,2
export RAY_ADDRESS=local
export RAY_memory_monitor_refresh_ms=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_DISABLE_TORCH_AO=1
export WANDB_MODE=disabled

# bitsandbytes loads libnvJitLink.so.13 from cu13 toolchain — NOT on default LD path.
# Solved in 35B v17; carried forward unchanged.
VERL_ENV="/home/peijia/miniconda3/envs/explain-verl"
export LD_LIBRARY_PATH="$VERL_ENV/lib/python3.11/site-packages/nvidia/cu13/lib:$VERL_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$VERL_ENV/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# HF cache redirect to /data1 (348 GB free). /data2 was full when 27B download started.
# Existing models on /data2 are still resolved by the symlink at ~/.cache/huggingface;
# only NEW Qwen3.6-27B blobs land in /data1. Other ATTS jobs unaffected.
export HF_HUB_CACHE="/data1/peijia/hf_cache"

PROJECT_DIR="/data3/peijia/dr-claw/Explain/Experiment/core_code"
CONFIG_PATH="$PROJECT_DIR/training/grpo"
DATA_DIR="$PROJECT_DIR/training/training_data/grpo"
TOOL_CONFIG="$CONFIG_PATH/tool_config.yaml"
REWARD_FN="$CONFIG_PATH/reward_fn.py"
# Separate ckpt dir from 35B to prevent accidental cross-arch checkpoint load
CKPT_DIR="$PROJECT_DIR/tmp/grpo_smoke_qwen36_27b"

export PYTHONPATH="$PROJECT_DIR/training/grpo/bootstrap:$PROJECT_DIR:${PYTHONPATH:-}"

LOG="$PROJECT_DIR/tmp/m6_grpo_smoke_verl.log"
echo "=== M6 GRPO smoke (Qwen3.6-27B) launched $(date -u) ===" > "$LOG"

conda run --no-capture-output -n explain-verl python3 -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='grpo_config' \
    algorithm.adv_estimator=grpo \
    `# train_batch_size=4 (2026-05-04 v11 TP=2 DP=1 pivot): DP=1 → no divisibility constraint.` \
    `# Total rollouts = 4 × n=8 = 32, all on single replica (concurrency capped by max_num_seqs=8 below).` \
    data.train_batch_size=4 \
    `# Production token budget (NON-NEGOTIABLE): policy quality knobs. Smoke uses production values.` \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    actor_rollout_ref.model.path=Qwen/Qwen3.6-27B \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    `# target_modules=all-linear: PEFT will wrap every nn.Linear including the linear_attention (Mamba)` \
    `# projections. vLLM PunicaWrapper may silently skip kernels for some of these layers (same warning class` \
    `# as 35B's "visual.blocks.X.attn.qkv will be ignored"). Confirm post-load via print_trainable_parameters().` \
    `# If linear_attention LoRA is silently dropped, narrow target_modules to attention-only modules in v22.` \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.model.use_liger=True \
    `# 35B v19 needed +fused_kernel_options.impl_backend=triton because the MoE-specific monkey-patch gate` \
    `# in monkey_patch.py:246 required impl_backend explicitly set. Non-MoE Qwen3_5 uses the default path,` \
    `# which activates use_fused_kernels=True without the override. INTENTIONALLY DROPPED FROM 35B v20.` \
    ++actor_rollout_ref.actor.use_fused_kernels=True \
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    `# ppo_mini_batch_size=4 (2026-05-04 v11 TP=2 DP=1 pivot): DP=1 → no divisibility constraint.` \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.actor.use_dynamic_bsz=True \
    ++actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    `# param_offload=False: for LoRA the base is frozen and the optimizer is over MB-scale adapters.` \
    `# Offload buys nothing and triggered v19's 226 GiB CPU OOM. STAY False — proven in 35B v18 baseline.` \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    `# forward_prefetch overlaps next-layer all-gather with current-layer compute — pure throughput win.` \
    ++actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.name=vllm \
    `# enable_sleep_mode=False + free_cache_engine=False (2026-05-03 fix v3):` \
    `# v3 (mem_util=0.74, max_num_seqs=8) passed KV cache check but VllmWorker-0 died at sleep_replicas() collective_rpc("sleep").` \
    `# Root cause: vllm 0.20 + Qwen3_5ForConditionalGeneration (hybrid Mamba+full-attention) + sleep_level=1 is broken.` \
    `# References:` \
    `#   - https://github.com/vllm-project/vllm/issues/32714 — "Sleep is broken since 0.14.0"` \
    `#   - https://github.com/vllm-project/vllm/issues/37749 — "Qwen 3.5 stops working after upgrade to v0.18.0"` \
    `#   - https://github.com/verl-project/verl/issues/3516 — verl HYBRID + lora_as_adapter forces sleep_level=1` \
    `# Fix: turn off sleep entirely. enable_sleep_mode=False at vLLM init + free_cache_engine=False to skip sleep() body.` \
    `# Tradeoff: vLLM weights stay live in GPU permanently (27.78 GB/card). FSDP step has tighter activation budget` \
    `# (~3 GiB per card free during update_weights). gradient_checkpointing=True + micro_batch_size_per_gpu=1 should fit.` \
    `# layered_summon=True (below) handles update_weights peak per-layer to stay safe.` \
    ++actor_rollout_ref.rollout.enable_sleep_mode=False \
    ++actor_rollout_ref.rollout.free_cache_engine=False \
    `# layered_summon=True STAYS at TP=1 DP=3 (universally safer, even though TP=1 DP=3 has more headroom):` \
    `#   At update_weights, vllm sleep_level=1 keeps weights live (27.78 GB/card with TP=1).` \
    `#   FSDP must summon-all to push 27.78 GB into vLLM weight buffer.` \
    `#   Per-card peak (no layered): vLLM (27.78) + FSDP shard (9.26) + summon-all transient (27.78) = 64.83 GB → fits in 80 GB but tight.` \
    `#   With layered_summon: per-layer transient = 27.78/64 ≈ 434 MB → total peak ~37.5 GB, very safe.` \
    `#   Verified mechanism: same as 35B v17 fix; do NOT drop just because TP=1 DP=3 looks loose.` \
    ++actor_rollout_ref.rollout.layered_summon=True \
    `# load_format=safetensors required by layered_summon: vLLM preloads base from disk so only LoRA delta is synced.` \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.response_length=16384 \
    `# gpu_memory_utilization=0.70 (2026-05-03 fix v8, was 0.74 in v6, 0.72 in v1, 0.78 in v0):` \
    `# v6 (0.74) DIED 23:05:08 mid-rollout with CUDA OOM: vLLM 58.70 GiB + FSDP shard 20.41 GiB = 79.11/79.25 GiB → 0.19 GiB headroom.` \
    `# Verl docs (https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) state:` \
    `#   "vLLM uses gpu_memory_utilization fraction as RESERVATION; the remaining (1-gpu_memory_utilization)` \
    `#    will ALSO be used during inference." → 0.74 was below the recommended 0.5-0.7 sweet spot.` \
    `# v8 fix: mem_util 0.74 → 0.70.` \
    `#   vLLM new budget: 0.70 × 79.25 = 55.48 GiB. Internal need: model 27.78 + Mamba 10.7 + KV (1.82×8=14.6) = 53.1 GiB.` \
    `#   Internal margin: 2.4 GiB. External: 79.25 - 55.48 - FSDP 20.41 = 3.36 GiB headroom (vs v6's 0.19 GiB → 18× margin).` \
    `# v0 (0.78) failed: 21:31:56 OOM at vLLM init "Free 60.46 GiB < desired 0.78 (61.81 GiB)".` \
    `# v1 (0.72) failed: 21:44:01 KV cache too small with max_num_seqs=20 (Mamba pool starved KV).` \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    `# enable_prefix_caching=False (2026-05-03 fix v9): v8 init failed at KV cache check —` \
    `# "1.82 GiB KV cache needed, 0.22 GiB available". Root: with prefix caching enabled, vLLM forces` \
    `# Mamba cache mode='align' for Qwen3_5ForConditionalGeneration (per vllm/config.py:367 warning),` \
    `# which pre-allocates ~27 GiB Mamba pool regardless of max_num_seqs. At mem_util=0.70 (55.48 GiB total),` \
    `# 27.78 (model) + 27 (Mamba pool) = 54.78 → leaves only 0.22 GiB for KV pool, far below 1.82 GiB needed.` \
    `# Empirical (v0 comment claimed Mamba scales linearly with max_num_seqs but v8 disproved this).` \
    `# Disabling prefix caching makes Mamba use 'eager' mode (small per-request allocation) → KV pool gets` \
    `# nearly all the working budget. Smoke loses prefix-caching speedup (~30% on multi-turn rollouts) but` \
    `# this is acceptable. Re-enable for production AFTER understanding Mamba pool sizing better.` \
    `# Sources: vllm/config.py:367; https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html` \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    `# tensor_model_parallel_size=1 BUMPED FROM 2 (2026-05-03 TP=1 DP=3 pivot):` \
    `# TP=3 forbidden by num_key_value_heads=4 (KV%TP must be 0 → 4%3=1 fails). Only TP=1 or TP=2 valid.` \
    `# TP=1 chosen so all 3 GPUs go to FSDP DP=3 (sharded 1/3 each). Each card holds full vLLM weights (27.78 GB)` \
    `# but FSDP shard is only 9.26 GB — net working budget 1.79× of TP=2 DP=2.` \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    `# disable_custom_all_reduce STAYS True for safety; with TP=1 there is no TP-level all_reduce so this is dead-code` \
    `# (no negative impact). Kept for forward-compat in case we ever switch back to TP=2.` \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true \
    `# enable_chunked_prefill caps activation peak per step. Default-on since vLLM 0.6 but explicit guarantees the cap.` \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_chunked_prefill=true \
    `# max_num_batched_tokens=16384 BUMPED FROM 4096 (P0 perf fix 2026-04-30 — Phase 1.3 GPU-power-low diagnosis):` \
    `# Phase 1.3 baseline (4096) ran 55 min at ~100W per A100 (35% util) with vLLM under-batched in eager mode.` \
    `# A100 prefill kernel saturates at 16K-32K token batch; 4096 was too small to amortize eager kernel-launch overhead.` \
    `# Risk math: activation peak grows ~4x with batch token count, from ~5 GiB to ~20 GiB. Stays under 21.8 GB working` \
    `# budget at gpu_memory_utilization=0.62 because KV pool (21.2 GiB) and activation peak don't co-peak in real steps.` \
    `# The original "30 GiB at 24576 breaks budget" warning targeted the cudagraph_profile path which is bypassed by` \
    `# enforce_eager=true (bug #36372 workaround) — only real-step prefill activation governs now.` \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_batched_tokens=16384 \
    `# max_num_seqs=4 (2026-05-03 fix v10, was 8 in v2-v9 — all v8/v9 init failed at KV check):` \
    `# v8/v9 evidence (mem_util=0.70): Available KV=0.22 GiB << 1.67 needed, even with prefix_caching=False.` \
    `# Diagnosis (vllm/v1/kv_cache_interface.py:540-547 + cache.py:132): Mamba 'none' mode still allocates` \
    `# page_size_bytes × max_num_seqs × num_linear_layers. Qwen3.5 has 48 linear layers + page_size ~70 MB` \
    `# → Mamba pool at max_num_seqs=8 = ~27.5 GB (empirical, derived from KV residual).` \
    `# At mem_util=0.70 (vLLM 55.48 GB), working budget = 55.48 - 27.78 (model) = 27.70 GB.` \
    `# 27.70 - 27.5 (Mamba) = 0.22 GiB residual for KV → fails 1.67 GiB KV need.` \
    `# Fix: halve max_num_seqs to 4. New Mamba ≈ 13.75 GB + KV (1.67×4=6.68) = 20.4 GB / 27.7 GB → 7.3 GB margin.` \
    `# Cost: rollout concurrency drops 8→4 per replica (24 total rollouts queue 2 batches). Smoke OK.` \
    `# Production may need to re-evaluate (revert to 8 with mem_util=0.74 OR shift to TP=2 to reduce per-card load).` \
    `# Prior v2 comment about "max_num_seqs 20→8 frees Mamba pool linearly" was WRONG — pool is roughly fixed` \
    `# given num_layers × max_num_seqs (linear in max_num_seqs but not in inverse direction predicted).` \
    `# Why 20 failed: qwen3_5 has 48 linear_attention layers (Gated DeltaNet/Mamba). vLLM with Mamba cache mode='align' (forced` \
    `# when prefix caching enabled) pre-allocates Mamba state for max_num_seqs concurrent seqs at full max_seq_len alignment.` \
    `# At max_num_seqs=20 × 48 layers, Mamba pool ate ~27 GiB of 29 GiB post-weights budget. KV pool starved to 1.78 GiB,` \
    `# but a single 24576-token request needs 1.82 GiB → vLLM init refused.` \
    `# Why 8 is correct: TP=1 DP=3, train_batch_size=3 × rollout.n=8 = 24 rollouts split across 3 replicas = 8 per replica.` \
    `# max_num_seqs=8 matches actual concurrency demand exactly. Mamba pool shrinks to 8/20 = 40% of v1, freeing ~16 GiB to KV pool.` \
    `# vLLM batch caps at actual concurrent demand → no throughput loss from this reduction.` \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=8 \
    `# enforce_eager=true REQUIRED workaround for vllm#36372 (LoRA on Qwen3.5 fails with` \
    `# "IndexError: too many indices for tensor of dimension 1" at column_parallel_linear.py:259 slice_lora_b).` \
    `# Bug fires during profile_cudagraph_memory → maybe_dummy_run_with_lora when CUDA graph capture is enabled.` \
    `# With enforce_eager=True, vLLM logs "disabling torch.compile and CUDAGraphs" → bug path skipped → init succeeds.` \
    `# Phase 0.3 dummy load confirmed: enable_lora=True + enforce_eager=True passes LLM_INIT_OK in 139.6s.` \
    `# Tradeoff: ~10-30% slower decode due to Python kernel-launch overhead (no CUDA graphs).` \
    `# Compensated by max_num_seqs=20 (vs 35B's 8) — 27B's 21.2 GB KV pool absorbs higher concurrency` \
    `# to keep GPU power gate ≥150 W satisfied.` \
    `# DO NOT DROP until vllm#36603 / #36395 / #35413 root-cause fix lands in a vLLM release.` \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=true \
    actor_rollout_ref.rollout.max_model_len=24576 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=9 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    `# rollout.agent.num_workers=8 BUMPED FROM 2 (P0 perf fix 2026-04-30 — same Phase 1.3 GPU-power-low diagnosis):` \
    `# 32 rollout sequences (train_batch_size=4 × rollout.n=8) with only 2 agent workers serialized 16 seqs/worker —` \
    `# vLLM batch was perpetually starved (~2-4 active vs max_num_seqs=20 ceiling). 8 workers × 4 seqs each` \
    `# keeps vLLM's 20-seq batch fully fed. Per-worker CPU-RAM ≈ 1-2 GB; 8 workers ≈ 16 GB, well under 125 GiB cap.` \
    `# reward.num_workers stays at 2 — reward path is post-rollout JUDGE_URL HTTP, not GPU-side, doesn't need bump.` \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    reward.num_workers=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    `# ref.param_offload=True (2026-05-03 v7 attempt, kept in v8 but PLACEBO):` \
    `# verl/workers/fsdp_workers.py:597 forces ref to use CPUOffload(offload_params=True) regardless of this flag` \
    `# ("# We force reference policy to use CPUOffload to save memory.") — so this flag has no effect on FSDP1.` \
    `# Kept True for FSDP2 path (offload_policy) and forward-compat. Real OOM fix is mem_util drop below.` \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FN" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name=qwen36-smoke \
    trainer.experiment_name=grpo-27b \
    `# n_gpus_per_node=3 BUMPED FROM 2 (2026-05-03 TP=1 DP=3 pivot — user directive 2026-05-03):` \
    `# 3 GPUs (1+2+3) for training, GPU 0 reserved for judge. World size = 3 × 1 nodes = 3 → DP=3, TP=1.` \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.resume_mode=disable \
    2>&1 | tee -a "$LOG"
