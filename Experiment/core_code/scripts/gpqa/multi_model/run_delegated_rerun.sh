#!/usr/bin/env bash
set -euo pipefail
unset CLAUDECODE 2>/dev/null || true
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/gpqa/multi_model/gpqa_multi_delegated.yaml \
	> ../analysis/run/gpqa/multi_model/delegated_rerun.log 2>&1 &
