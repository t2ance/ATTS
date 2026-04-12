#!/usr/bin/env bash
set -euo pipefail

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n grpo_vllm \
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-8B \
        --served-model-name qwen3-8b-base \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.9 \
        --data-parallel-size 4 \
        --port 8000 \
    2>&1 | tee tmp/vllm_serve_qwen3_8b_base_dp4.log
