#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

# GPQA explorer pre-cache for the `_exp_orch` ablation. Prerequisite:
# 3-replica serve up on :8000/:8001/:8002.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/gpqa/qwen36_35b_a3b_fp8/
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/gpqa/grpo/gpqa_qwen36_35b_a3b_precache.yaml \
	> ../analysis/run/gpqa/qwen36_35b_a3b_fp8/precache.log 2>&1 &
echo "GPQA precache PID $!"
echo "log: ../analysis/run/gpqa/qwen36_35b_a3b_fp8/precache.log"
