#!/usr/bin/env bash
set -euo pipefail

# Single-process vLLM DP=4 serve (2026-05-02): one `vllm serve` process with
# --data-parallel-size 4 spans GPU 0/1/2/3 and exposes a single HTTP endpoint
# on port 8000. vllm internally runs 4 engine workers (one per GPU) and does
# request load-balancing + continuous batching across them.
#
# Why DP=4 over the prior 4-independent-replicas pattern: single endpoint
# (no client-side round-robin), shared scheduler / cross-worker batching,
# one process to monitor + restart. Same VRAM cost per card.
#
# Why not TP=4 / DP=2+TP=2 / DP=3: fused_moe intermediate_size=8192 must be
# divisible by tp_size and dp_size. 8192 % 3 != 0 (so DP=3 / TP=3 are out,
# verified by pydantic ValidationError 2026-05-01). 8192 % 4 == 0, but TP=4
# adds NCCL all-reduce on every token decode and is bandwidth-bound; pure
# DP=4 keeps each worker on one card with zero cross-card traffic on the
# decode hot path. Prior config history (3-replica, then 4-replica multi-
# process) is in git log under the prior names of this script.
#
# gpu-memory-utilization=0.85: each card holds full FP8 model (~35 GB) +
# ~30 GB KV cache pool. 4 workers in parallel give ~4x query throughput at
# num_workers saturation.
#
# max-model-len=131072 (128K) — bumped from 65536 on 2026-05-01 to absorb
# yaml max_tokens=65536 per turn. Multi-turn accumulation can put input
# alone at 30-50K on hard LCB/BV before the final response. KV memory
# roughly doubles versus 64K; vllm auto-lowers max_num_seqs if needed.
# Watch the log for the "Maximum concurrency for X tokens per request" line.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

export HF_HUB_CACHE="/data1/peijia/hf_cache"

LOG="tmp/vllm_serve_qwen36_dp4.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup conda run --no-capture-output -n grpo_vllm \
    vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
        --served-model-name qwen36-35b-a3b-fp8 \
        --tensor-parallel-size 1 \
        --data-parallel-size 4 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 131072 \
        --trust-remote-code \
        --disable-custom-all-reduce \
        --port 8000 \
    > $LOG 2>&1 &

echo "started DP=4 serve (PID $!)"
echo "log: tmp/vllm_serve_qwen36_dp4.log"
