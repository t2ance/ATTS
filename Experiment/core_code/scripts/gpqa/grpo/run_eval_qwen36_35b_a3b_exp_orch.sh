#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

# GPQA `_exp_orch` arm: BOTH explorer cache and orchestrator are Qwen3.6-35B-A3B-FP8.
# Prerequisites:
#   1. 3-replica serve up on :8000/:8001/:8002.
#   2. GPQA explorer cache populated by run_precache_qwen36_35b_a3b.sh.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python eval.py \
	--config scripts/gpqa/grpo/gpqa_qwen36_35b_a3b_exp_orch.yaml \
	2>&1 | tee tmp/eval_qwen36_35b_a3b_gpqa_exp_orch.log
