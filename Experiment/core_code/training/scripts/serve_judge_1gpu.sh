#!/bin/bash
set -x
ulimit -n 65535

# Topology rule (2026-04-30 user directive): GPU 0 = judge, GPU 1+2 = verl training (symmetric clean cards),
# GPU 3 = blocked by another user. Judge on GPU 0 coexists with claude embedding daemon (1.37 GB) — judge
# 8B model uses ~20 GB at 0.5 mem_util, well under 80 GB. Symmetric memory principle: training cards must
# have IDENTICAL free memory at vLLM init time, otherwise the smaller card caps mem_util for both.
export CUDA_VISIBLE_DEVICES=0

/home/peijia/miniconda3/envs/verl/bin/vllm serve Qwen/Qwen3-8B \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --no-enable-log-requests \
    2>&1 | tee /data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/judge_1gpu.log
