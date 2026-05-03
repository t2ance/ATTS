#!/usr/bin/env bash
set -euo pipefail

# Gemma-4-26B-A4B-it DP=1 vLLM serve (debug-bench config). Topology specifics:
#   - HF id google/gemma-4-26B-A4B-it. BF16 only (no official FP8 release).
#     Per-card weight footprint ~27 GB; KV pool ~30 GB at gpu_mem_util=0.85.
#   - intermediate_size=2112, moe_intermediate_size=704, num_kv_heads=8.
#     2112%1=0 trivially → DP=1 (TP=1) divisibility OK.
#   - max-model-len=65536 matches the explore_timeout=1200s budget per the
#     Qwen archetype design (matched-budget).
#   - alias `gemma4-26b-a4b-it` (same as DP=2 variant) so eval/precache YAMLs
#     and `backends/vllm.py:MODEL_TO_BASE_URL` continue to route to port 8000.
#   - GPU 1 only (DP=2 → DP=1 dialed back 2026-05-03 per user directive
#     "把这两张都释放掉，然后重新开一个用一张卡的"): frees GPU 0 for the
#     embedding daemon and a 4th-card swap window. GPU 2+3 are taken by the
#     gpt-oss-20b serve (port 8001). Total topology = 1 Gemma + 2 gpt-oss = 3 cards.
#   - Purpose: debug bench for the Gemma4ReasoningParser fix (vllm#38855) — the
#     parser silently strips `<|channel>` tokens before reaching the budget
#     logits processor, so PR #20859's `thinking_token_budget` is dead on
#     Gemma4. The patch is a 4-line site-packages edit (add start_token_id /
#     end_token_id properties matching Qwen3ReasoningParser); we test it on
#     this single-card serve so a parser regression cannot pollute production
#     gpt-oss capacity.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

export HF_HUB_CACHE="/data1/peijia/hf_cache"

LOG="tmp/vllm_serve_gemma4_26b_a4b_dp1.log"

CUDA_VISIBLE_DEVICES=1 nohup conda run --no-capture-output -n grpo_vllm \
    vllm serve google/gemma-4-26B-A4B-it \
        --served-model-name gemma4-26b-a4b-it \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 65536 \
        --trust-remote-code \
        --disable-custom-all-reduce \
        --port 8000 \
        --max-num-batched-tokens 8192 \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --reasoning-parser gemma4 \
        --chat-template scripts/gpqa/grpo/tool_chat_template_gemma4.jinja \
        --structured-outputs-config '{"backend":"xgrammar"}' \
    > $LOG 2>&1 &
        # --chat-template scripts/gpqa/grpo/tool_chat_template_gemma4.jinja:
        # vendored from vllm/examples/tool_chat_template_gemma4.jinja per the
        # official Gemma 4 vLLM Recipes (docs.vllm.ai/projects/recipes/Google/
        # Gemma4.html). Static jinja rendering matches the model's default
        # tokenizer template, but vllm 0.20.0 chat completions endpoint goes
        # through a different code path when --chat-template is explicitly set;
        # this is the documented path for thinking_token_budget to fire on
        # Gemma4. Validated 2026-05-03 after raw /v1/completions confirmed
        # model reliably emits <|channel>thought\n... but /v1/chat/completions
        # 5/5 trials skipped channel mode at temp=1.0 budget=150.
        # --reasoning-parser gemma4 (top-level flag, NOT inlined in
        # --structured-outputs-config): vllm 0.20.0 only initializes
        # ReasoningConfig (and the per-request `thinking_token_budget` logits
        # processor that injects `<|/think|>`) when the TOP-LEVEL flag is set
        # (arg_utils.py:2332-2337). Without it any request carrying
        # thinking_token_budget gets HTTP 400. Required for Gemma4ReasoningParser
        # patch validation per todo_inf_gemma_thinking_budget.md.
        # --max-num-batched-tokens 8192 (override default 2048): Gemma's
        # multimodal-bidirectional attention forces --disable_chunked_mm_input,
        # so a single MM item (2496 tokens) cannot exceed max_num_batched_tokens.
        # --enable-auto-tool-choice + --tool-call-parser gemma4: server-side
        # tool-call parser populates structured `message.tool_calls=[...]`
        # (matching the DP=2 variant; backends/vllm.py:run_tool_conversation
        # reads that field directly).

echo "started Gemma DP=1 serve (PID $!)"
echo "log: tmp/vllm_serve_gemma4_26b_a4b_dp1.log"
echo "endpoint: http://localhost:8000/v1 (alias=gemma4-26b-a4b-it, GPU 1 only)"
