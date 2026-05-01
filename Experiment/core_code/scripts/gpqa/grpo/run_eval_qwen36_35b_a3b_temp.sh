#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

# GPQA _temp variant: Qwen3.6 thinking-mode recipe. See hle/grpo/_temp.sh for context.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python eval.py \
	--config scripts/gpqa/grpo/gpqa_qwen36_35b_a3b_temp.yaml \
	2>&1 | tee tmp/eval_qwen36_35b_a3b_gpqa_temp.log
