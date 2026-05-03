#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python eval.py \
	--config scripts/hle/grpo/hle_gemma4_26b_a4b_exp_orch.yaml \
	2>&1 | tee tmp/eval_gemma4_26b_a4b_hle_exp_orch.log
