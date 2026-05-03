#!/usr/bin/env bash
set -euo pipefail

# gpt-oss-20b DP=2 vLLM serve. Topology specifics:
#   - HF id openai/gpt-oss-20b. ~12 GiB BF16 weights.
#   - max-model-len=131072 — gpt-oss native context window.
#   - alias `gptoss-20b` is what eval/precache YAMLs reference (not `gpt-oss-20b`
#     because hyphen-versus-underscore consistency with Qwen/Gemma aliases).
#   - GPU 2 + 3 only — placed on the cards Gemma freed when its DP went 4 -> 2
#     2026-05-03 per user directive "如果两个空出来了 你可以用那两个跑gpt oss".
#   - Port 8001 (NOT 8000) — port 8000 is held by the Gemma DP=2 serve on GPU 0+1.
#     Eval/precache YAMLs that target gpt-oss MUST point client.base_url to
#     http://localhost:8001/v1 instead of the default 8000. NOTE: backends/vllm.py
#     currently reads OPENAI_BASE_URL env or a hardcoded default; downstream
#     scripts will need to set OPENAI_BASE_URL=http://localhost:8001/v1 before
#     invoking eval.py / precache_explores.py against gpt-oss.
#   - NO `--reasoning-parser openai_gptoss`. Tested 2026-05-03 — adding this
#     flag made vLLM convert our function tools into harmony "functions" MCP
#     namespace. Tool calls came back as `type=mcp_call status=completed`
#     instead of `type=function_call`, and gpt-oss treated them as already
#     executed by an internal MCP server (status=completed semantics) so the
#     model never waited for our explore() result and exited every question
#     after the first dispatch (commit rate 0/10). Without the flag, items
#     return as plain `function_call` which our adapter dispatches correctly.
#     The forward leak (`explore<|channel|>commentary` etc., vllm#32587) is
#     handled via _HARMONY_LEAK_RE in backends/vllm.py instead.
#   - No `--structured-outputs-config` — gpt-oss does NOT exhibit the Gemma
#     vllm#40080 xgrammar repetition collapse on JSON-schema-constrained
#     decoding (per harmony format design). Add only if smoke tests show
#     repetition / parse-failure patterns analogous to Gemma's failure mode.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

export HF_HUB_CACHE="/data1/peijia/hf_cache"

LOG="tmp/vllm_serve_gptoss20b_dp2.log"

CUDA_VISIBLE_DEVICES=2,3 nohup conda run --no-capture-output -n grpo_vllm \
    vllm serve openai/gpt-oss-20b \
        --served-model-name gptoss-20b \
        --tensor-parallel-size 1 \
        --data-parallel-size 2 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 131072 \
        --trust-remote-code \
        --disable-custom-all-reduce \
        --port 8001 \
        --max-num-batched-tokens 8192 \
        --enable-auto-tool-choice \
        --tool-call-parser openai \
    > $LOG 2>&1 &

echo "started gpt-oss-20b DP=2 serve (PID $!)"
echo "log: tmp/vllm_serve_gptoss20b_dp2.log"
echo "endpoint: http://localhost:8001/v1 (alias=gptoss-20b)"
