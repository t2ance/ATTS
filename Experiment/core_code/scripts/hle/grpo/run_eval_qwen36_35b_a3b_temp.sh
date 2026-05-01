#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

# HLE _temp variant: Qwen3.6 thinking-mode recipe (T=1.0/top_p=0.95/top_k=20/
# presence=1.5/enable_thinking=true/max_tokens=65536). Same Sonnet explore
# cache as _baseline; only orchestrator decoding differs. Prerequisite:
# scripts/gpqa/grpo/serve_qwen36_35b_a3b.sh up on :8000.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python eval.py \
	--config scripts/hle/grpo/hle_qwen36_35b_a3b_temp.yaml \
	2>&1 | tee tmp/eval_qwen36_35b_a3b_hle_temp.log
