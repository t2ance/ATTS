#!/usr/bin/env bash
set -ex

# M5 SFT smoke for Qwen3.6-35B-A3B (MoE) — verl SFT trainer fallback path.
# LLaMA-Factory 0.9.4 hard-pinned transformers<=4.57.1; explain-verl has 5.7.0,
# so we cannot use llamafactory-cli. verl 0.7.1 sft_trainer is the fallback.
#
# Topology: GPU 0 + GPU 2 = FSDP rank 0/1 (GPU 1 has external vllm from M3,
# GPU 3 occupied by another user). Per nvidia-smi at 18:38 UTC.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

# CUDA_VISIBLE_DEVICES=0,2 — see topology comment above.
export CUDA_VISIBLE_DEVICES=0,2
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Avoid prm-vllm's optional torchao import that fails under torch 2.11.0+cu130
export PYTORCH_DISABLE_TORCH_AO=1

# Hydra overrides
# - model.path: target model
# - model.trust_remote_code: required by HF for Qwen3_5MoE
# - model.lora_rank=64, lora_alpha=128, target=all-linear: production LoRA spec
# - model.use_liger=True: free-lunch fused kernels (Liger Qwen3MoE support TBD)
# - model.use_fused_kernels=True: verl-side fused ops
# - model.enable_gradient_checkpointing=True: free-lunch activation savings
# - data.train_files: 100-row smoke parquet
# - data.max_token_len_per_gpu=16384: matches production cutoff_len (8B SFT yaml)
# - data.train_batch_size=2 (= 1 micro × 2 grad_accum × 1 dp = trivial)
# - data.micro_batch_size_per_gpu=1
# - engine.model_dtype=bfloat16: production dtype
# - engine.fsdp_size=-1: auto FSDP across all visible GPUs (=2)
# - engine.reshard_after_forward=True: free-lunch peak memory savings
# - trainer.total_training_steps=5: smoke goal — 5 finite losses, no crash
# - trainer.logger=["console"]: skip wandb to avoid auth on smoke
LOG=/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/m5_sft_smoke_verl.log
echo "=== M5 SFT verl-trainer smoke launched $(date -u) ===" > "$LOG"

torchrun --standalone --nproc_per_node=2 \
    -m verl.trainer.sft_trainer \
    model.path=Qwen/Qwen3.6-35B-A3B \
    model.trust_remote_code=True \
    model.lora_rank=64 \
    model.lora_alpha=128 \
    model.target_modules=all-linear \
    model.use_liger=True \
    model.use_fused_kernels=True \
    model.enable_gradient_checkpointing=True \
    model.use_remove_padding=False \
    data.train_files=/data3/peijia/dr-claw/Explain/Experiment/core_code/training/training_data/sft_smoke_100.parquet \
    data.train_batch_size=2 \
    data.micro_batch_size_per_gpu=1 \
    data.max_token_len_per_gpu=16384 \
    data.use_dynamic_bsz=False \
    data.messages_key=messages \
    data.tools_key=tools \
    data.train_max_samples=100 \
    data.ignore_input_ids_mismatch=True \
    data.max_length=16384 \
    engine.model_dtype=bfloat16 \
    engine.fsdp_size=-1 \
    engine.reshard_after_forward=True \
    engine.param_offload=False \
    engine.optimizer_offload=False \
    optim.lr=2e-4 \
    trainer.project_name=qwen36-smoke \
    trainer.experiment_name=sft-35b-a3b \
    trainer.total_training_steps=5 \
    trainer.total_epochs=1 \
    trainer.logger='[console]' \
    trainer.default_local_dir=/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/sft_smoke_qwen36_35b_a3b_verl \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    2>&1 | tee -a "$LOG"
