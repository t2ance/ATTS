#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

# LCB baseline eval against Qwen/Qwen3.6-35B-A3B-FP8 BASE model (untrained).
# Prerequisite: scripts/gpqa/grpo/serve_qwen36_35b_a3b.sh running on
# http://127.0.0.1:8000 with served-model-name=qwen36-35b-a3b-fp8.

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# PYTHONPATH override: explain env's editable install of livecodebench points
# to a stale /data1 path (project relocated to /data3 — finder MAPPING never
# refreshed). Prepending the real code_references location lets `import
# lcb_runner.evaluation` resolve via PYTHONPATH before the broken editable
# finder is consulted, so the LCB grader (grade_code -> check_correctness)
# loads cleanly. Permanent fix would be `pip install -e .` from the new
# LiveCodeBench dir, but that mutates the conda env; PYTHONPATH is reversible
# and scoped to this launcher.
export PYTHONPATH="/data3/peijia/dr-claw/Explain/Experiment/code_references/LiveCodeBench:${PYTHONPATH:-}"
PYTHONUNBUFFERED=1 conda run -n explain --no-capture-output python eval.py \
	--config scripts/lcb/grpo/lcb_qwen36_35b_a3b.yaml \
	2>&1 | tee tmp/eval_qwen36_35b_a3b_lcb.log
