#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

# LCB `_exp_orch` arm: BOTH explorer cache and orchestrator are Qwen3.6-35B-A3B-FP8.
# Prerequisites:
#   1. DP=4 vllm serve up on :8000 (single endpoint). Start via
#      scripts/gpqa/grpo/serve_qwen36_35b_a3b_dp4.sh.
#   2. LCB explorer cache populated by run_precache_qwen36_35b_a3b.sh.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# PYTHONPATH override carried over from `_temp`: explain env's editable install
# of livecodebench points to a stale /data1 path. Prepending the real
# code_references location lets `import lcb_runner.evaluation` resolve cleanly
# so the LCB grader (grade_code -> check_correctness) loads.
export PYTHONPATH="/data3/peijia/dr-claw/Explain/Experiment/code_references/LiveCodeBench:${PYTHONPATH:-}"
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python eval.py \
	--config scripts/lcb/grpo/lcb_qwen36_35b_a3b_exp_orch.yaml \
	2>&1 | tee tmp/eval_qwen36_35b_a3b_lcb_exp_orch.log
