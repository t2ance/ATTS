#!/usr/bin/env bash
set -euo pipefail

# Gemma-4-26B-A4B-it DP=4 vLLM serve. Topology specifics:
#   - HF id google/gemma-4-26B-A4B-it. BF16 only (no official FP8 release).
#   - intermediate_size=2112, moe_intermediate_size=704, num_kv_heads=8.
#     2112%4=0, 704%4=0, 8%4=0  -> DP=4 (TP=1) divisibility OK.
#   - max-model-len=65536 matches the explore_timeout=1200s budget.
#   - alias `gemma4-26b-a4b-it` is what eval/precache YAMLs reference.
#   - Restored from DP=1 -> DP=4 on 2026-05-03 per user directive
#     "我允许你去释放目前 GPU 上 1、2、3 的负载, 然后重新启动一个数据并行为 4 的服务".
#     Quadruples nominal concurrency; per-card weight footprint ~27 GB,
#     KV pool larger because gpu-memory-utilization bumped to 0.95.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

export HF_HUB_CACHE="/data1/peijia/hf_cache"

LOG="tmp/vllm_serve_gemma4_26b_a4b_dp4.log"

# --gpu-memory-utilization 0.95 (override default 0.9, prior dp2 used 0.85):
#   bumped per user directive 2026-05-03 "你的 GPU Utilization (即 Memory 使用)
#   可以改成 0.95, 这样的话会更快一些". Larger KV cache = more in-flight
#   slots, drains the 800-explore HLE precache queue faster. Coupling: leaves
#   ~3.5 GiB free per 80 GiB card (vs ~12 GiB at 0.85), so any other process
#   that lands on 0/1/2/3 will OOM the engine -- DO NOT colocate.
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup conda run --no-capture-output -n grpo_vllm \
    vllm serve google/gemma-4-26B-A4B-it \
        --served-model-name gemma4-26b-a4b-it \
        --tensor-parallel-size 1 \
        --data-parallel-size 4 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 65536 \
        --trust-remote-code \
        --disable-custom-all-reduce \
        --port 8000 \
        --max-num-batched-tokens 8192 \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --reasoning-parser gemma4 \
        --structured-outputs-config '{"backend":"xgrammar"}' \
        --chat-template scripts/gpqa/grpo/tool_chat_template_gemma4_fixed.jinja \
    > $LOG 2>&1 &
        # --chat-template tool_chat_template_gemma4_fixed.jinja (added 2026-05-03):
        # LAYER-1 of the Gemma-4 thinking double-bug fix. HF default chat_template.jinja
        # leaves enable_thinking=true with bare `<|turn>model\n` end-of-prompt — IT-tuned
        # weights' first-token logit prefers a normal text token over `<|channel>`,
        # so the thinking channel never opens. The fixed jinja prefills `<|channel>thought\n`
        # (channel OPEN) when enable_thinking=true. Coupling: re-diff the fork against
        # HF default on every Gemma-4 model upgrade. Layer-2 (skip_special_tokens=False)
        # is patched on the client side in backends/vllm.py. Reference: vllm#39130 +
        # the "Gemma-4 thinking on vLLM 0.20.x" entry in vllm skill troubleshooting.md.
        # --reasoning-parser gemma4 (top-level): vllm 0.20.0 only initializes
        # ReasoningConfig (and the per-request `thinking_token_budget` logits
        # processor that injects `<|/think|>`) when the TOP-LEVEL flag is set
        # -- arg_utils.py:2332-2337 gates on self.reasoning_parser, not on
        # structured_outputs_config.reasoning_parser. Without this, requests
        # carrying thinking_token_budget get HTTP 400.
        # --structured-outputs-config '{"backend":"xgrammar"}': xgrammar paired
        # with reasoning-parser=gemma4 lets the parser separate thinking from
        # JSON enforcement, avoiding Gemma-4's repetition collapse on
        # JSON-schema-constrained generation (vllm#40080).
        # --max-num-batched-tokens 8192 (override default 2048): Gemma's
        # multimodal-bidirectional attention forces --disable_chunked_mm_input,
        # so a single MM item (2496 tokens) cannot exceed max_num_batched_tokens.
        # --enable-auto-tool-choice + --tool-call-parser gemma4: server-side
        # parser populates structured `tool_calls=[...]` in `message.tool_calls`
        # instead of bare text. Required by client path-B in backends/vllm.py.

echo "started Gemma DP=4 serve (PID $!)"
echo "log: tmp/vllm_serve_gemma4_26b_a4b_dp4.log"
