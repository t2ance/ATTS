#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

# GPQA baseline eval against Qwen/Qwen3.6-35B-A3B BASE model
# (untrained, no LoRA). Prerequisite: serve_qwen36_35b_a3b.sh
# is running on http://127.0.0.1:8000 with served-model-name=qwen36-35b-a3b.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python eval.py \
	--config scripts/gpqa/grpo/gpqa_qwen36_35b_a3b.yaml \
	2>&1 | tee tmp/eval_qwen36_35b_a3b_gpqa.log
