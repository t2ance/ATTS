#!/bin/bash
set -x
ulimit -n 65535

export CUDA_VISIBLE_DEVICES=2

/home/peijia/miniconda3/envs/verl/bin/vllm serve Qwen/Qwen3-8B \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --no-enable-log-requests \
    2>&1 | tee /data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/judge_1gpu.log
