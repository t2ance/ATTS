set -x
ulimit -n 65535

# Stress test variant of train_grpo_vllm_8b_base_hle_3gpu.sh.
#
# Purpose: validate proposed config changes (log_prob micro batch bumps +
# reshard_after_forward=False i.e. real ZeRO-2) under the full production
# per-sample profile, reaching update_actor in minutes instead of an hour.
#
# The ONLY delta from production for speed is data.train_batch_size: 63 -> 3
# (smallest value divisible by dp_size=3). Everything else that determines
# per-sample compute / VRAM (micro batch sizes, response length, FSDP config,
# rollout.n, KV cache budget) is kept at production values or at the proposed
# new values we are testing.
#
# Expected rollout walltime: ~45min * (3/63) ~= ~2 min.
# Then old_log_prob -> ref -> adv -> update_actor executes normally.
# Watch peak VRAM during update_actor; that is the OOM boundary.
#
# Proposed config deltas under test (will stick in production after validation):
#   rollout.log_prob_micro_batch_size_per_gpu   2 -> 8    (forward-only headroom)
#   ref.log_prob_micro_batch_size_per_gpu       2 -> 8    (8 instead of 6 for divisibility)
#   actor.fsdp_config.reshard_after_forward  True -> False (ZeRO-2, matches script comment intent)
#
# Stress-test-only overrides (will NOT stick):
#   data.train_batch_size                     63 -> 3
#   actor.ppo_mini_batch_size                 63 -> 3
#   trainer.val_before_train               True -> False
#   trainer.save_freq                          6 -> 99999
#   trainer.test_freq                          3 -> 99999
#   trainer.total_epochs                      30 -> 1
#   trainer.experiment_name              <prod>-stress
#   rollout.trace.max_samples_per_step_per_worker 2 -> 0 (avoid weave overhead polluting timing)

export CUDA_VISIBLE_DEVICES=0,1
export RAY_ADDRESS=local
export RAY_memory_monitor_refresh_ms=0
# NOTE: PYTORCH_ALLOC_CONF=expandable_segments:True is INCOMPATIBLE with vLLM.
# vLLM hard-asserts against it in gpu_worker.py:334 because its custom CUDA
# memory pool (device_allocator/cumem.py) does not support expandable segments.
# Do NOT re-enable.

VERL_ENV="/home/peijia/miniconda3/envs/grpo_vllm"
export LD_LIBRARY_PATH="$VERL_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$VERL_ENV/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PROJECT_DIR="/data3/peijia/dr-claw/Explain/Experiment/core_code"
CONFIG_PATH="$PROJECT_DIR/training/grpo"
DATA_DIR="$PROJECT_DIR/training_data/grpo"
TOOL_CONFIG="$CONFIG_PATH/tool_config.yaml"
REWARD_FN="$CONFIG_PATH/reward_fn.py"
MODEL_PATH="Qwen/Qwen3-8B"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

conda run --no-capture-output -n grpo_vllm python3 -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='grpo_config' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=2 \
    data.max_prompt_length=4096 \
    data.max_response_length=24576 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=24576 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FN" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=0 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=atts-grpo \
    trainer.experiment_name=8b-base-hle-k16-3gpu-stress \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=99999 \
    trainer.test_freq=99999 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    2>&1 | tee "$PROJECT_DIR/tmp/grpo_8b_base_hle_3gpu_stress.log"
