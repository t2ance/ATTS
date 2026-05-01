#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate explain
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/lcb/sonnet/lcb_sonnet_delegated.yaml \
	> ../analysis/run/lcb/sonnet/delegated.log 2>&1 &
