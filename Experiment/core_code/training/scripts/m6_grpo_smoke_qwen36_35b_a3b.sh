#!/usr/bin/env bash
set -ex

# M6 GRPO smoke for Qwen3.6-35B-A3B (MoE).
# Goal: verify whether GRPO can train on Qwen3.5-arch MoE in explain-verl env,
# given that M5 multi-turn SFT engine is blocked by verl#5944-class bug.
#
# Topology (corrected 2026-04-29 per user):
#   GPU 0,1 = verl hybrid (vllm TP=2 actor + FSDP shard across 2)
#   GPU 2   = judge Qwen3-8B vllm serve (separate process, port 8000)
#   GPU 3   = free
#
# Smoke params:
#   total_training_steps=1, val_before_train=False, save_freq=-1
#   rollout.n=2, max_assistant_turns=3, max_prompt=2048, max_response=4096
#   train_batch_size=4 (=2 prompts x rollout_n=2)
#   ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=1
#
# LoRA: r=64 alpha=128 all-linear; ref_in_actor auto-true.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

# Topology (2026-04-30 user directive — symmetric memory principle):
#   GPU 0 = judge (Qwen3-8B vLLM serve) + claude embedding daemon (1.37 GB)
#   GPU 1 = verl training (clean, 80 GB free)
#   GPU 2 = verl training (clean, 80 GB free)  [moved off judge to here]
#   GPU 3 = BLOCKED by another user (namdo train_esm.py, 54.9 GB)
# Why GPU 1+2 not 0+1: cuda:0 has the 1.37 GB daemon — asymmetric free memory across training ranks
# caps vLLM mem_util at the SMALLER card's free fraction (cuda:0 = 43.6 GB free instead of 45 GB on cuda:1).
# Symmetric clean cards (1+2 both at 45 GB free post-FSDP) raises mem_util ceiling by ~1.5 GB per card.
export CUDA_VISIBLE_DEVICES=1,2
export RAY_ADDRESS=local
export RAY_memory_monitor_refresh_ms=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_DISABLE_TORCH_AO=1
export WANDB_MODE=disabled

# Fix bitsandbytes loading libnvJitLink.so.13 — env has it under nvidia/cu13/lib
# (CUDA 13 toolchain) but it's not on the default LD path.
VERL_ENV="/home/peijia/miniconda3/envs/explain-verl"
export LD_LIBRARY_PATH="$VERL_ENV/lib/python3.11/site-packages/nvidia/cu13/lib:$VERL_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$VERL_ENV/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PROJECT_DIR="/data3/peijia/dr-claw/Explain/Experiment/core_code"
CONFIG_PATH="$PROJECT_DIR/training/grpo"
DATA_DIR="$PROJECT_DIR/training/training_data/grpo"
TOOL_CONFIG="$CONFIG_PATH/tool_config.yaml"
REWARD_FN="$CONFIG_PATH/reward_fn.py"
CKPT_DIR="$PROJECT_DIR/tmp/grpo_smoke_qwen36_35b_a3b"

export PYTHONPATH="$PROJECT_DIR/training/grpo/bootstrap:$PROJECT_DIR:${PYTHONPATH:-}"

LOG="$PROJECT_DIR/tmp/m6_grpo_smoke_verl.log"
echo "=== M6 GRPO smoke launched $(date -u) ===" > "$LOG"

conda run --no-capture-output -n explain-verl python3 -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='grpo_config' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=4 \
    `# Production token budget (NON-NEGOTIABLE 2026-04-29): multi-turn/response/n/max_model_len affect policy training quality.` \
    `# Even smoke must use production values - shrinking them produces worse-trained policies than the production target.` \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    actor_rollout_ref.model.path=Qwen/Qwen3.6-35B-A3B \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules=all-linear \
    `# v17 (2026-04-30) RE-DIAGNOSED root cause: v16 lora.merge=True still OOM identical 1.57 GiB short.` \
    `# Real bug is NOT sleep_level — at update_weights, FSDP must summon full params (35 GB transient) to push to vLLM.` \
    `# 35 GB FSDP shard + 35 GB summon transient + 35 GB vLLM weights live = 105 GB on 80 GB → OOM regardless of sleep level.` \
    `# Real fix: layered_summon=True (verl/utils/fsdp_utils.py:591) — walks submodules one at a time instead of summon-all.` \
    `# Peak transient = 1 layer (~440 MB) instead of 35 GB. Requires load_format=safetensors so vLLM preloads base & only LoRA delta is synced.` \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.model.use_liger=True \
    `# v19 (2026-04-30) FIX: v18 log showed 'Skipping monkey patch ... use_fused_kernels is False or fused_kernels_backend is torch'.` \
    `# Root cause: verl/workers/fsdp_workers.py:483-486 reads backend from model.fused_kernel_options.impl_backend (NOT actor.fused_kernels_backend).` \
    `# monkey_patch.py:246 gate is 'not use_fused_kernels OR backend not in [triton,torch]' — so backend=None (default) makes patch skip even with use_fused_kernels=True.` \
    `# Setting impl_backend=triton activates fused RMSNorm/RoPE/SwiGLU/CrossEntropy for Qwen3.5MoE actor.` \
    ++actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
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
    `# v20 (2026-04-30) REVERT v19's param_offload=True: it crashed rank 0 at update_weights, raylet log: 'Disconnecting client connection error code 2 EOF'.` \
    `# Mechanism: verl HYBRID + lora_as_adapter (vllm_async_server.py:636) FORCES sleep_level=1; vLLM weights stay 35GB; FSDP returning to GPU during update_weights = 70GB+ → CUDA OOM.` \
    `# layered_summon=True is also incompatible with sleep_level=2 (vllm_rollout.py:91-95). So we cannot lower vLLM static footprint during update — must keep FSDP coresident.` \
    `# v18 baseline (param_offload=False, mem_util=0.55) is the only proven-stable config through update_weights. Revert.` \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    `# optimizer_offload stays False — for LoRA the optimizer is only over adapter params (~MB), offloading it buys nothing.` \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    `# forward_prefetch overlaps next-layer all-gather with current-layer compute — pure throughput win, no math change.` \
    ++actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.name=vllm \
    `# v17 layered_summon: walk FSDP submodules one at a time during weight sync (verl/utils/fsdp_utils.py:591 layered_summon_lora_params).` \
    `# Without it: full unshard = 35 GB transient → OOM at update_weights. With it: per-layer ~440 MB peak.` \
    ++actor_rollout_ref.rollout.layered_summon=True \
    `# load_format=safetensors required by layered_summon: vLLM preloads base from disk so only LoRA delta gets synced.` \
    `# Default 'dummy' would not have base loaded; layered_summon would crash with "base-model is preloaded in vllm" assert.` \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.response_length=16384 \
    `# v20 (2026-04-30) REVERTED 0.85 → 0.55: v19's 0.85 caused rank 0 OOM at update_weights when FSDP shard returned to GPU.` \
    `# Real model: Qwen3.6-35B-A3B = 35.23 GB / card vLLM weights at TP=2. FSDP DP=2 shard = 35.23 GB / card.` \
    `# vLLM init pool ceiling = (80 - 35.23 FSDP) / 80 = 0.559. mem_util=0.55 sits just under that ceiling.` \
    `# At mem_util=0.55: pool=44 GB - vLLM weights 35.23 = 8.77 GB working - profiler activation 3 GB = 5.77 GB KV.` \
    `# Holds max_num_seqs=8 at avg seq len ~10K tok (each seq KV ~600 MB).` \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true \
    `# v20 (2026-04-30) RE-ADDED enforce_eager=true: at mem_util=0.55 we cannot afford the 5 GB CUDA graph capture (would crash KV budget to ~0).` \
    `# Tradeoff: ~10-30% slower decode due to Python kernel-launch overhead. v18 measured 27% sustained power.` \
    `# To reach the ≥50% gate without expanding mem budget, we instead bump max_num_seqs 4→8 (only feasible knob within the 5 GB KV pool).` \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=true \
    `# enable_chunked_prefill: split long prefills into chunks of max_num_batched_tokens. Default since vLLM 0.6 but explicit guarantees activation = batched_tokens * hidden, not max_model_len * hidden.` \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_chunked_prefill=true \
    `# max_num_batched_tokens=4096: caps profiler dummy forward + per-step compute. Default would be ~max_model_len(24576) → 30 GiB MoE activation peak.` \
    `# 4096 → ~4 GiB activation. Trade: long prompts (8192) prefill in 2 chunks (acceptable, prefill time dominated by GPU compute not launch overhead).` \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_batched_tokens=4096 \
    `# v20 (2026-04-30) max_num_seqs=4→8: ONLY feasible knob to lift power without breaking memory. v18 KV budget 5 GB.` \
    `# Each seq at avg ~10K tok decode ≈ 600 MB KV → 8 seqs ≈ 4.8 GB, within budget.` \
    `# 32 needed seqs (bs=4 × n=8) serialize in 4 waves instead of 8. Expected throughput ~2× = power 27%→~50%.` \
    `# This is the SOLE rollout-side change in v20. Everything else reverts to v18 baseline (proven stable through update_weights at 0.88s).` \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_model_len=24576 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=9 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.agent.num_workers=2 \
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
    trainer.experiment_name=grpo-35b-a3b \
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
