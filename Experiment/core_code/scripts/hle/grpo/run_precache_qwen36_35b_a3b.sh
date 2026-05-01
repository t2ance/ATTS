#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

# HLE explorer pre-cache for the `_exp_orch` ablation: Qwen3.6-35B-A3B-FP8 as
# explorer (matches orchestrator). Prerequisite: 3-replica serve up on
# :8000/:8001/:8002 (scripts/gpqa/grpo/serve_qwen36_35b_a3b_3replica.sh).

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/hle/qwen36_35b_a3b_fp8/
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/hle/grpo/hle_qwen36_35b_a3b_precache.yaml \
	> ../analysis/run/hle/qwen36_35b_a3b_fp8/precache.log 2>&1 &
echo "HLE precache PID $!"
echo "log: ../analysis/run/hle/qwen36_35b_a3b_fp8/precache.log"
