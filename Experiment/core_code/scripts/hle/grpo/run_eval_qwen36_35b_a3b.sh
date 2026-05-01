#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

# HLE-gold-text_only:100 baseline eval against Qwen/Qwen3.6-35B-A3B BASE
# model (untrained, no LoRA). Prerequisite: the same serve_qwen36_35b_a3b.sh
# launched from scripts/gpqa/grpo/ is running on http://127.0.0.1:8000 with
# served-model-name=qwen36-35b-a3b. One serve process feeds both benchmarks.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python eval.py \
	--config scripts/hle/grpo/hle_qwen36_35b_a3b.yaml \
	2>&1 | tee tmp/eval_qwen36_35b_a3b_hle.log
