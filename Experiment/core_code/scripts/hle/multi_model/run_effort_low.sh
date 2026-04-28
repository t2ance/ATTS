#!/usr/bin/env bash
set -euo pipefail
unset CLAUDECODE 2>/dev/null || true
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py \
        --benchmark hle \
        --config configs/hle_multi_effort_low.yaml \
        > ../analysis/run/hle/multi_model_effort_low/delegated.log 2>&1 &
