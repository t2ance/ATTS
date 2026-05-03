#!/usr/bin/env bash
set -euo pipefail

# Serve Qwen/Qwen3.6-35B-A3B-FP8 on GPU 0 ONLY (TP=1, single replica) for the
# LCB EMPTY-row retry-with-4x-budget run. Created 2026-05-03 specifically for
# todo_lcb_empty_retry_4x.md Phase 1 Item 01.
#
# Why TP=1 + GPU 0: GPU 1, 2, 3 are fully occupied by other models / users;
#   GPU 0 is free (1.3 GB used at TODO write time). 35B-FP8 = ~35 GB weights,
#   fits one A100-80GB with ~45 GB headroom for KV cache.
# Why max-model-len 163840 (160K): the retry's max_tokens budget is 131072
#   (128K, 4x of original 32K). model_len = max_tokens + prompt + tool defs;
#   prompt+tools ~ 4-8K, so 160K leaves comfortable headroom and rounds nicely.
# Why port 8002: 8000 = gemma4 serve already running, 8001 = gpt-oss serve
#   already running (verified at TODO write time via `ps -ef | grep vllm`).
# Why disable-custom-all-reduce: A100 PCIe (no NVLink) can hit
#   "CUDA invalid argument" with custom_all_reduce on 35B-A3B; carried from
#   the original Table 9 serve script per archive_v20.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

export HF_HUB_CACHE="/data1/peijia/hf_cache"

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n grpo_vllm \
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3.6-35B-A3B-FP8 \
        --served-model-name qwen36-35b-a3b-fp8 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 163840 \
        --trust-remote-code \
        --disable-custom-all-reduce \
        --port 8002 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
    2>&1 | tee tmp/vllm_serve_qwen36_gpu0_128k.log
# Why --enable-auto-tool-choice + --tool-call-parser qwen3_xml (added 2026-05-03):
#   ATTS orchestrator sends `tool_choice="auto"` on chat-completions; vLLM
#   400s without these flags. qwen3_xml is the parser for Qwen3.x families
#   (per scripts/gpqa/grpo/serve_qwen36_35b_a3b_dp4.sh). Without this, the
#   first driver attempt crashed with:
#       openai.BadRequestError: '"auto" tool choice requires
#       --enable-auto-tool-choice and --tool-call-parser to be set'

