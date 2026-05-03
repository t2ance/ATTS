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
# max-model-len=65536 (64K). Sequence of values this session:
#   131072 -> 40960 (precache throughput optimization 2026-05-02)
#   40960  -> 131072 (eval orchestrator OOM-context fix 2026-05-02)
#   131072 -> 65536  (matched-budget design 2026-05-02 user-directed)
# Why 64K is the matched value:
#   1. Explorer/orchestrator decode at 50-80 tok/s under DP=4 saturation;
#      generating 64K tokens takes 800-1200s, which equals explore_timeout=1200s.
#      Anything generating past 64K won't finish under wall-clock budget anyway.
#   2. Throughput: KV pool / max-model-len = max in-flight per worker. At 64K
#      we get 2x more in-flight slots than 128K (~40 vs 20 per worker), which
#      lifts cluster cap from 83 to ~160 in-flight and increases per-step
#      batch -> higher SM utilization -> more power.
#   3. Context-overflow handling: when input + max_tokens > 65536 vllm raises
#      BadRequest with param="input_tokens"; backends/vllm.py soft-skips this
#      single explore (treats as timeout-equivalent) and the precache/eval
#      continues. This is wired in vllm.py's try/except around chat.completions.
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
        --max-model-len 65536 \
        --trust-remote-code \
        --disable-custom-all-reduce \
        --port 8000 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3xml \
    > $LOG 2>&1 &
        # --enable-auto-tool-choice + --tool-call-parser qwen3xml (added
        # 2026-05-02 to align with backends/vllm.py path-B refactor): vLLM
        # populates structured `message.tool_calls=[...]` so the client never
        # text-parses `<tool_call><function=NAME><parameter=K>V</parameter>
        # </function></tool_call>` itself. qwen3xml is the parser matching
        # Qwen3.6's chat_template tool-call format (XML body inside
        # <tool_call>...</tool_call>). Thread-safety race vllm#34932 fixed
        # in vllm#40059 (vllm 0.20.0), so DP=4 is safe.
        # NOT YET RE-VERIFIED with this serve flag set; next time Qwen is
        # served, smoke-test multi-turn before launching production eval.

echo "started DP=4 serve (PID $!)"
echo "log: tmp/vllm_serve_qwen36_dp4.log"
