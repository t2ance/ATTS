#!/usr/bin/env bash
set -euo pipefail

# Gemma-4-26B-A4B-it DP=2 vLLM serve. Topology specifics:
#   - HF id google/gemma-4-26B-A4B-it. BF16 only (no official FP8 release).
#     Per-card weight footprint ~27 GB; KV pool ~30 GB at gpu_mem_util=0.85.
#   - intermediate_size=2112, moe_intermediate_size=704, num_kv_heads=8.
#     2112%2=0, 704%2=0, 8%2=0  -> DP=2 (TP=1) divisibility OK.
#   - max-model-len=65536 matches the explore_timeout=1200s budget per
#     serve_qwen36_35b_a3b_dp4.sh comment block; same matched-budget design.
#   - alias `gemma4-26b-a4b-it` is what eval/precache YAMLs reference.
#   - DP=4 -> DP=2 dialed back 2026-05-03 per user directive
#     "只使用前两个gpu进行推理". Halves nominal concurrency from 15.11x to ~7.5x
#     and roughly halves throughput; HLE explore_timeout=1200s budget unchanged
#     (per-request decode latency unchanged, only fewer in-flight slots).
#     Releases GPU 2 + GPU 3 for other workloads.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p tmp

export HF_HUB_CACHE="/data1/peijia/hf_cache"

LOG="tmp/vllm_serve_gemma4_26b_a4b_dp2.log"

CUDA_VISIBLE_DEVICES=0,1 nohup conda run --no-capture-output -n grpo_vllm \
    vllm serve google/gemma-4-26B-A4B-it \
        --served-model-name gemma4-26b-a4b-it \
        --tensor-parallel-size 1 \
        --data-parallel-size 2 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 65536 \
        --trust-remote-code \
        --disable-custom-all-reduce \
        --port 8000 \
        --max-num-batched-tokens 8192 \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --reasoning-parser gemma4 \
        --structured-outputs-config '{"backend":"xgrammar"}' \
    > $LOG 2>&1 &
        # --reasoning-parser gemma4 (added 2026-05-03; top-level flag, distinct
        # from the field formerly inlined into --structured-outputs-config):
        # vllm 0.20.0 only initializes ReasoningConfig (and thus the per-request
        # `thinking_token_budget` logits processor that injects `<|/think|>`)
        # when the TOP-LEVEL --reasoning-parser is set — see
        # vllm/engine/arg_utils.py:2332-2337 (_set_default_reasoning_config_args
        # gates on `self.reasoning_parser`, not on
        # structured_outputs_config.reasoning_parser). Without this flag, any
        # request carrying `thinking_token_budget` gets HTTP 400
        # "thinking_token_budget is set but reasoning_config is not configured.
        # Please set --reasoning-config to use thinking_token_budget"
        # (input_processor.py:101-109). The legacy CLI also auto-propagates
        # `self.reasoning_parser` INTO structured_outputs_config (arg_utils.py
        # 2063-2065), so we drop the duplicate `reasoning_parser` key from the
        # JSON to keep a single source of truth. Required by the
        # thinking_token_budget validation run targeting Gemma's 78% HLE
        # timeout rate (todo_inf_gemma_thinking_budget.md).
        # --structured-outputs-config '{"backend":"xgrammar"}' (current 2026-05-03):
        # The historical xgrammar -> outlines switch (2026-05-02 attempt to
        # avoid Gemma-4's repetition collapse on JSON-schema-constrained
        # generation, vllm#40080) was reverted because pairing xgrammar with
        # `--reasoning-parser gemma4` solves the same problem the simpler way:
        # the parser tells vLLM the thinking phase ends at `<channel|>`, so
        # xgrammar only enforces JSON on the post-think output and never sees
        # the deterministic-repetition-prone token-set restriction during
        # thinking. Required by `enable_thinking=true` flows and by the
        # thinking_token_budget logits processor (which needs ReasoningConfig
        # initialized by the top-level --reasoning-parser flag above).
        # --max-num-batched-tokens 8192 (override default 2048): Gemma's
        # multimodal-bidirectional attention forces --disable_chunked_mm_input,
        # so a single MM item (2496 tokens) cannot exceed max_num_batched_tokens.
        # 8192 gives 3.3x headroom for the 2496-token MM slot.
        # Set 2026-05-02 after vllm 0.20 upgrade.

        # --enable-auto-tool-choice + --tool-call-parser gemma4 (added 2026-05-02
        # for path-B verify): activates vLLM's server-side gemma4 tool-call
        # parser so the OpenAI response carries structured `tool_calls=[...]`
        # in `message.tool_calls` instead of bare text in `message.content`.
        # Eliminates the need for client-side `parse_tool_calls(text, model)`
        # branching. The thread-safety race (tokenizer.encode RuntimeError
        # "Already borrowed", vllm#34932) was fixed in vllm#40059 (merged
        # 2026-04-24, included in 0.20.0) — parsers now use vocab.get(...)
        # not tokenizer.encode(), so DP + concurrent requests are safe.
        # See backends/vllm.py for the matching client-side switch from
        # tool_choice="none"+text-parsing to tool_choice="auto"+structured.

echo "started Gemma DP=2 serve (PID $!)"
echo "log: tmp/vllm_serve_gemma4_26b_a4b_dp2.log"
