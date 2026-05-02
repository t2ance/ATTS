#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

# BabyVision explorer pre-cache for the `_exp_orch` ablation. Prerequisite:
# DP=4 vllm serve up on :8000 (single endpoint, 4 internal workers). Start
# via scripts/gpqa/grpo/serve_qwen36_35b_a3b_dp4.sh.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/babyvision/qwen36_35b_a3b_fp8/
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/babyvision/grpo/babyvision_qwen36_35b_a3b_precache.yaml \
	> ../analysis/run/babyvision/qwen36_35b_a3b_fp8/precache.log 2>&1 &
echo "BabyVision precache PID $!"
echo "log: ../analysis/run/babyvision/qwen36_35b_a3b_fp8/precache.log"
