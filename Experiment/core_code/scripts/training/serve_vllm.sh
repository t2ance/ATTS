#!/bin/bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
CUDA_VISIBLE_DEVICES=0 conda run -n verl vllm serve Qwen/Qwen3-8B \
    --enable-lora \
    --lora-modules atts-orch=checkpoints/sft_qwen3_8b \
    --max-model-len 16384 \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code
