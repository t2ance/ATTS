#!/usr/bin/env bash
set -euo pipefail

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

CUDA_VISIBLE_DEVICES=0,1,2 conda run --no-capture-output -n grpo_vllm \
    python3 -m vllm.entrypoints.openai.api_server \
        --model checkpoints/grpo_8b_step78_merged \
        --served-model-name grpo-8b \
        --max-model-len 12288 \
        --gpu-memory-utilization 0.9 \
        --data-parallel-size 3 \
        --port 8000 \
    2>&1 | tee tmp/vllm_serve_grpo_8b_dp3.log
