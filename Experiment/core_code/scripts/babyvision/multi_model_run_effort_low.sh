#!/usr/bin/env bash
set -euo pipefail
unset CLAUDECODE 2>/dev/null || true
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
        --config scripts/babyvision/babyvision_multi_effort_low.yaml \
        > ../analysis/run/babyvision/multi_model_effort_low/delegated.log 2>&1 &
