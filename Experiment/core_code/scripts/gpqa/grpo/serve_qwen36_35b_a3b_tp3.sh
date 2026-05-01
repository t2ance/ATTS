#!/usr/bin/env bash
set -euo pipefail

# 3-card TP=3 serve attempt (2026-05-01 user directive). Earlier note in
# serve_qwen36_35b_a3b.sh archived that --data-parallel-size=3 hit fused_moe
# divisibility AssertionError; TP=3 may hit the same wall if intermediate_size
# is not divisible by 3. If TP=3 init asserts at fused_moe/layer.py:485 the
# launcher will exit non-zero in <90s and the caller falls back to TP=2.
cd /data3/peijia/dr-claw/Explain/Experiment/core_code

export HF_HUB_CACHE="/data1/peijia/hf_cache"

CUDA_VISIBLE_DEVICES=0,1,2 conda run --no-capture-output -n grpo_vllm \
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3.6-35B-A3B-FP8 \
        --served-model-name qwen36-35b-a3b-fp8 \
        --tensor-parallel-size 3 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 65536 \
        --trust-remote-code \
        --disable-custom-all-reduce \
        --port 8000 \
    2>&1 | tee tmp/vllm_serve_qwen36_35b_a3b_fp8_tp3.log
