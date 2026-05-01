#!/usr/bin/env bash
set -euo pipefail

# 3-replica serve: GPU 0/1/2 each runs an independent TP=1 vllm serve on
# ports 8000/8001/8002. ATTS client (backends/vllm.py:_get_client) round-robins
# across them via itertools.cycle. TP=3/DP=3 are categorically blocked by
# fused_moe intermediate_size=8192 not divisible by 3 (verified 2026-05-01
# 04:18 UTC: pydantic ValidationError "8192 is not divisible by 3"). Three
# independent replicas is the only path that uses all 3 cards.
#
# gpu-memory-utilization=0.85: each card holds full FP8 model (~35 GB) +
# ~30 GB KV cache pool. 3 replicas in parallel give ~3x query throughput at
# num_workers=32 saturation.
#
# max-model-len=131072 (128K) — bumped from 65536 on 2026-05-01 to absorb
# yaml max_tokens=65536 per turn. Need input + output capacity; multi-turn
# accumulation can put input alone at 30-50K on hard LCB/BV before the final
# response. KV cache memory roughly doubles versus the 64K config; vllm
# auto-lowers max_num_seqs if it would OOM. Watch the serve log for the
# "Maximum concurrency for X tokens per request" line on startup to verify
# the in-flight cap is still well above num_workers/3 ≈ 21.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

export HF_HUB_CACHE="/data1/peijia/hf_cache"

start_replica() {
    local gpu=$1
    local port=$2
    local log="tmp/vllm_serve_qwen36_3replica_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES=$gpu nohup conda run --no-capture-output -n grpo_vllm \
        python3 -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen3.6-35B-A3B-FP8 \
            --served-model-name qwen36-35b-a3b-fp8 \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.85 \
            --max-model-len 131072 \
            --trust-remote-code \
            --disable-custom-all-reduce \
            --port $port \
        > $log 2>&1 &
    echo "started GPU $gpu -> port $port (PID $!)"
}

start_replica 0 8000
start_replica 1 8001
start_replica 2 8002

echo
echo "All 3 replica launchers spawned. Tail any of:"
echo "  tmp/vllm_serve_qwen36_3replica_gpu0.log"
echo "  tmp/vllm_serve_qwen36_3replica_gpu1.log"
echo "  tmp/vllm_serve_qwen36_3replica_gpu2.log"
