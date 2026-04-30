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
# Topology (2026-04-30 — symmetric memory principle):
#   GPU 0 = judge Qwen3-8B vLLM serve (port 8000) + claude embedding daemon
#   GPU 1 = verl training (clean, 81 GB free)
#   GPU 2 = verl training (clean, 81 GB free)  [symmetric with GPU 1]
#   GPU 3 = BLOCKED by another user (namdo train_esm.py, ~56 GB used)
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
    `# layered_summon=True STAYS at 27B. Mechanism (same as 35B v17 fix, just less tight at 27B):` \
    `#   At update_weights, vllm sleep_level=1 keeps weights live. FSDP must summon-all to push.` \
    `#   Per-card peak: vLLM (27.78) + FSDP shard (27.78) + summon-all transient (27.78) = 83.34 GB > 80 → OOM by 3.34 GB.` \
    `#   layered_summon walks per-layer: transient = 27.78/64 ≈ 434 MB → total peak ~56 GB, safe.` \
    `# DO NOT DROP. The 27B "dense" property does not save you 28 GB summon transient.` \
    ++actor_rollout_ref.rollout.layered_summon=True \
    `# load_format=safetensors required by layered_summon: vLLM preloads base from disk so only LoRA delta is synced.` \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.response_length=16384 \
    `# gpu_memory_utilization=0.62 — bumped from 35B v20's 0.55. Math: free per card after FSDP wrap = 80 - 27.78 = 52.22 GB.` \
    `# Ceiling = 52.22/80 = 0.652. Use 0.62 with 3% margin. Phase 0.3 dummy load measured Available KV cache memory: 21.2 GiB` \
    `# (no CUDA graph because enforce_eager=true — see below). 21.2 GiB / 600 MB per seq = 35 seqs theoretical, vLLM-reported` \
    `# Maximum concurrency 25.31x at 24K tokens → max_num_seqs=20 with 25% margin (set below).` \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.62 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    `# A100 PCIe (no NVLink) custom_all_reduce hits CUDA invalid argument; disable. Skip on H100/NVLink boxes.` \
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
    `# max_num_seqs=20 — bumped from 35B v20's 8. Phase 0.3 dummy load measured KV cache 21.2 GiB and reported` \
    `# "Maximum concurrency for 24,576 tokens per request: 25.31x" → 20 leaves 25% safety margin against KV exhaust.` \
    `# 35B v20 was capped at 8 by 5 GB KV; 27B's 21.2 GB KV (verified, not estimated) supports ~25 seqs concurrent.` \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=20 \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FN" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name=qwen36-smoke \
    trainer.experiment_name=grpo-27b \
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
