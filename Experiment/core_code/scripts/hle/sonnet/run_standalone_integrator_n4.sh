#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/hle/sonnet_standalone_integrator_n4
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/hle/sonnet/hle_sonnet_standalone_integrator_n4.yaml \
	> ../analysis/run/hle/sonnet_standalone_integrator_n4/run.log 2>&1 &
echo "PID=$!"
echo "log=/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_standalone_integrator_n4/run.log"
