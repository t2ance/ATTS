set -x
ulimit -n 65535

# 2-GPU GRPO-from-SFT variant. Derived from train_grpo_vllm_8b_sft.sh (3-GPU)
# with 2-GPU memory disciplines borrowed from train_grpo_vllm_8b_base_hle_2gpu.sh:
#   CUDA_VISIBLE_DEVICES=0,1
#   n_gpus_per_node=2
#   train_batch_size=96, ppo_mini_batch_size=16 (96x8=768 rollouts/step; 6 mini-batch updates/step; 91% data/epoch)
#   ppo_micro_batch_size_per_gpu=1 (down from 4 to halve backward activations)
#   actor.fsdp_config.param_offload=False (keeps params on GPU to avoid CPU RAM OOM killing Judge)
#   ref.fsdp_config.param_offload=False (same reason)
#   model_dtype=bfloat16, reshard_after_forward=True, fsdp_size=-1
#   rollout.n=8 kept (GRPO group)
#
# Why SFT init instead of base: sft_qwen3_8b_merged smoke test showed
# 45/50 (90%) PASS on the canonical FullRenderer tool-return format, so the
# 'Action: Submit X' -> StructuredOutput mapping gap (atts_sft_gap) is closed.
#
# Token budget (measured from step-4 log + 160-sample FullRenderer profiling):
#   Sequence layout: original_prompt + response_sequence (assistant + tool tokens interleaved)
#   response_length cap applies to ALL tokens after the original prompt (assistant + tool combined).
#
#   Measured per trajectory (max_assistant_turns=5, ~4 explores + 1 StructuredOutput):
#     prompt tokens:          mean=2086  max=3528
#     response tokens total:  mean=1665  (tool: ~845 + assistant: ~820)
#     total sequence:         mean=3750  (well within old 12288 limit)
#
#   For 8 explores + 1 StructuredOutput (max_assistant_turns=9):
#     prompt:         mean=2086  max=3528
#     tool responses: 8 * 264(mean)=2112   8 * 478(max)=3824
#     assistant text: 9 * 195(mean)=1755   9 * 600(generous)=5400
#     total avg: ~6000   p95: ~10100   generous max: ~12700
#
#   32K budget chosen by user. Settings:
#     max_model_len=24576  (covers generous max with headroom; 12288 -> 2x)
#     max_prompt_length=8192 (covers p99 HLE questions; 4096 -> 2x)
#     response_length=16384  (= 24576 - 8192; gives 9 turns avg 1586 tokens each; 4096 -> 4x)
#     max_assistant_turns=9  (8 explores + 1 StructuredOutput; was 5 = 4 explores)
#     gpu_memory_utilization=0.55 (was 0.4; increased because KV cache scales with max_model_len)

export CUDA_VISIBLE_DEVICES=0,1
export RAY_ADDRESS=local
export RAY_memory_monitor_refresh_ms=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_INIT_TIMEOUT=300

VERL_ENV="/home/peijia/miniconda3/envs/grpo_vllm"
export LD_LIBRARY_PATH="$VERL_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$VERL_ENV/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PROJECT_DIR="/data3/peijia/dr-claw/Explain/Experiment/core_code"
CONFIG_PATH="$PROJECT_DIR/training/grpo"
DATA_DIR="$PROJECT_DIR/training_data/grpo"
TOOL_CONFIG="$CONFIG_PATH/tool_config.yaml"
REWARD_FN="$CONFIG_PATH/reward_fn.py"
MODEL_PATH="$PROJECT_DIR/checkpoints/sft_qwen3_8b_merged"
CKPT_DIR="$PROJECT_DIR/checkpoints/atts-grpo/8b-sft-2gpu-bs96"

export PYTHONPATH="$PROJECT_DIR/training/grpo/bootstrap:$PROJECT_DIR:$PYTHONPATH"

RESUME_ARGS=(
    trainer.default_local_dir="$CKPT_DIR"
)

if [[ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]]; then
    LAST_STEP="$(<"$CKPT_DIR/latest_checkpointed_iteration.txt")"
    if [[ -n "$LAST_STEP" && -d "$CKPT_DIR/global_step_$LAST_STEP" ]]; then
        RESUME_ARGS+=(
            trainer.resume_mode=resume_path
            trainer.resume_from_path="$CKPT_DIR/global_step_$LAST_STEP"
        )
    fi
fi

conda run --no-capture-output -n grpo_vllm python3 -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='grpo_config' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.actor.use_dynamic_bsz=True \
    ++actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    ++actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=24576 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=9 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FN" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=2 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=atts-grpo \
    trainer.experiment_name=8b-sft-2gpu-bs96 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.val_before_train=True \
    trainer.total_epochs=15 \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee "$PROJECT_DIR/tmp/grpo_8b_sft_2gpu_bs96.log"
