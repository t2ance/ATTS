#!/bin/bash
# Test the trained 9B orchestrator on 3 GPQA questions
# Requires vLLM server running (scripts/training/serve_vllm.sh)
cd /data3/peijia/dr-claw/Explain/Experiment/core_code

unset CLAUDECODE 2>/dev/null || true

python eval.py --benchmark gpqa \
    --backend vllm \
    --method tts-agent \
    --num 3 \
    --seed 42 \
    --num-explores 8 \
    --num-workers 1 \
    --log-dir ../analysis/run/gpqa/vllm_test \
    --orchestrator-model atts-orch \
    --explore-model claude-haiku-4-5-20251001 \
    --cache-dirs ../analysis/cache/gpqa/sonnet \
    --cache-only
