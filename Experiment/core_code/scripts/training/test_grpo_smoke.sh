set -x
ulimit -n 65535

export CUDA_VISIBLE_DEVICES=1
export RAY_ADDRESS=local
export RAY_memory_monitor_refresh_ms=0

GRPO_ENV="/home/peijia/miniconda3/envs/grpo"
export LD_LIBRARY_PATH="$GRPO_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$GRPO_ENV/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PROJECT_DIR="/data3/peijia/dr-claw/Explain/Experiment/core_code"
CONFIG_PATH="$PROJECT_DIR/training/grpo"
DATA_DIR="$PROJECT_DIR/training_data/grpo"
TOOL_CONFIG="$CONFIG_PATH/tool_config.yaml"
REWARD_FN="$CONFIG_PATH/reward_fn.py"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

conda run -n grpo python3 -m training.grpo.run_grpo \
    --config-path="$CONFIG_PATH" \
    --config-name='grpo_config' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FN" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=atts-grpo-smoke \
    trainer.experiment_name=smoke-1.7b \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    2>&1 | tee "$PROJECT_DIR/tmp/grpo_smoke.log"
