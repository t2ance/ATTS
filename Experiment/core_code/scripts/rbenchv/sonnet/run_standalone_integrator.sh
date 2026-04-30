#!/usr/bin/env bash
set -euo pipefail

# Standalone integrator over 8 cached explores per question. This is the
# "LLM Selection (N=8)" row in main.tex Table tab:lb-rbenchv: a single LLM
# call reads all 8 cached explores and selects/synthesizes the answer.
# Depends on a fully-precached explore set; run after all 4 family precaches
# (physics, math, counting, game) finish.

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/rbenchv/sonnet_standalone_integrator
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
	--config scripts/rbenchv/sonnet/rbenchv_sonnet_standalone_integrator.yaml \
	> ../analysis/run/rbenchv/sonnet_standalone_integrator/standalone_integrator.log 2>&1 &

echo "Launched. PID=$!"
echo "Tail: tail -f /data3/peijia/dr-claw/Explain/Experiment/analysis/run/rbenchv/sonnet_standalone_integrator/standalone_integrator.log"
