#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate explain
# --resume <RUN_DIR> stays as a CLI override (per-launch, not per-config) via -o.
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/lcb/sonnet/lcb_sonnet_delegated.yaml \
	-o resume=../analysis/run/lcb/sonnet/run_20260308_230222 \
	> ../analysis/run/lcb/sonnet/delegated.log 2>&1 &
