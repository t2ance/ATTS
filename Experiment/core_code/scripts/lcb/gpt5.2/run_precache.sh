#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate explain
export PYTHONPATH="../code_references/LiveCodeBench:${PYTHONPATH:-}"
mkdir -p ../analysis/run/lcb/gpt5.2_no_integrate_high
PYTHONUNBUFFERED=1 nohup python precache_explores.py \
	--config configs/lcb_gpt5.2_precache.yaml \
	> ../analysis/run/lcb/gpt5.2_no_integrate_high/precache.log 2>&1 &
