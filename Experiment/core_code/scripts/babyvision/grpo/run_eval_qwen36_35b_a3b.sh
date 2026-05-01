#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

# BabyVision baseline eval against Qwen/Qwen3.6-35B-A3B-FP8 BASE (untrained).
# Multimodal: vLLM serve must route images through the vision encoder.
# Prerequisite: scripts/gpqa/grpo/serve_qwen36_35b_a3b.sh running on
# http://127.0.0.1:8000 with served-model-name=qwen36-35b-a3b-fp8.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python eval.py \
	--config scripts/babyvision/grpo/babyvision_qwen36_35b_a3b.yaml \
	2>&1 | tee tmp/eval_qwen36_35b_a3b_babyvision.log
