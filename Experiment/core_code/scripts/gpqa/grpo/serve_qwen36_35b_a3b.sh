#!/usr/bin/env bash
set -euo pipefail

# Serve Qwen/Qwen3.6-35B-A3B-FP8 (FP8-quantized MoE base, untrained) for ATTS
# BASELINE eval. Three-card vLLM Data Parallel (DP=3, TP=1) — each GPU runs
# an INDEPENDENT replica with the FULL model. vLLM's internal load balancer
# round-robins requests across replicas, so the eval client sees one endpoint
# but gets 3x query throughput vs single-replica TP=2.
#
# Why FP8 over bf16 (user directive 2026-04-30):
#   bf16 35B-A3B = 69.3 GB. A100 80 GB - 15% headroom = 68 GB usable, so bf16
#   does NOT fit one card with any KV pool. TP=2 was the only option in bf16
#   but produces only 1 replica = 1x throughput. FP8 35B-A3B = ~35 GB,
#   leaving ~33 GB per card for KV pool — comfortable single-card residency.
#   Per-replica accuracy delta vs bf16 is ~1-2% on ATTS-style benchmarks
#   (within noise floor of GRPO uplift), and we are deploying FP8 anyway, so
#   FP8 is the correct precision for the baseline measurement.
#
# Why TP=2 single-replica (REVISED 2026-04-30 after DP=3 init failure):
#   First attempt was --data-parallel-size=3 --tensor-parallel-size=1 across
#   GPU 0+1+2 for 3x query throughput. vLLM 0.17 fails with
#       AssertionError at fused_moe/layer.py:485
#       (assert intermediate_size % self.tp_size == 0)
#   on Qwen3.5-MoE: even with TP=1, the DP path internally uses dp_size for
#   the MoE expert intermediate_size divisibility check. 35B-A3B expert
#   ffn_inner is not divisible by 3, so DP=3 is structurally blocked.
#   DP=2 is plausible (ffn_inner is power-of-2) but TP=2 is the safer choice:
#     - Single replica with one large KV pool (~100 GB across 2 cards)
#       supports 100+ concurrent sequences via vLLM's internal continuous
#       batching, fed by num_workers>=8 ATTS clients in parallel.
#     - Two-replica DP=2 splits the KV pool into two smaller ones; under
#       num_workers=8 the per-replica queue depth maxes out faster.
#     - Empirically TP=2 + saturating num_workers tends to match DP=2
#       throughput on tool-calling benchmarks while avoiding any DP+MoE
#       contract risk.
#
# GPU plan (2026-04-30): GPU 1+2 = single TP=2 replica. Judge already killed
# from GPU 0 by user directive; GPU 0 stays idle (free for restart of judge
# at HLE/BabyVision/RBenchV grading time). GPU 3 blocked by another user.
#
# Knobs:
#   - tensor-parallel-size=2: split MoE experts + attention heads across
#     GPU 1+2. ffn_inner % 2 == 0 holds for all MoE configs.
#   - gpu-memory-utilization=0.85: ~17.5 GB model shard per card +
#     ~50 GB KV per card. Massive aggregate KV pool.
#   - max-model-len=24576: matches GRPO production budget (8192 + 16384).
#   - dtype=auto: FP8 model has its own dtype config; let vLLM auto-resolve.
#   - disable-custom-all-reduce: A100 PCIe (no NVLink) hits CUDA invalid
#     argument with custom_all_reduce; carried from m6 35B archive_v20.
#   - HF_HUB_CACHE=/data1/peijia/hf_cache: 296 GB free. /data2 is 100% full,
#     /data3 has 263 GB but is the primary Ray spill target.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

export HF_HUB_CACHE="/data1/peijia/hf_cache"

# max-model-len bumped 24576 -> 65536 (2026-04-30): Qwen3.6 explorer on hard
# HLE questions writes ~19k+ reasoning tokens to fit ATTS's `reasoning` schema
# field. With 24K model_len the JSON gets truncated before the closing `"` and
# parser asserts. 65K leaves explorer ~63K output budget after a ~2K prompt;
# KV cache pool shrinks proportionally (each cached seq holds 2.7x more KV)
# but for num_workers<=8 single-replica serving this is acceptable.
CUDA_VISIBLE_DEVICES=1,2 conda run --no-capture-output -n grpo_vllm \
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3.6-35B-A3B-FP8 \
        --served-model-name qwen36-35b-a3b-fp8 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 65536 \
        --trust-remote-code \
        --disable-custom-all-reduce \
        --port 8000 \
    2>&1 | tee tmp/vllm_serve_qwen36_35b_a3b_fp8.log
