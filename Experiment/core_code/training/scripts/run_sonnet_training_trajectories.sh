#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

PYTHONUNBUFFERED=1 nohup python eval.py --benchmark hle \
	--backend claude \
	--subset gold \
	--skip 100 \
	--num 200 \
	--seed 42 \
	--num-explores 8 \
	--num-workers 4 \
	--text-only \
	--log-dir ../analysis/run/hle/sonnet_training \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-haiku-4-5-20251001 \
	--integrate-model claude-sonnet-4-6 \
	--cache-dirs ../analysis/cache/hle/haiku/gold \
	--no-integrate \
	--max-output-chars 3500 \
	> ./tmp/sonnet_training_trajectories.log 2>&1 &

echo "Started. PID: $!"
echo "Log: ./tmp/sonnet_training_trajectories.log"
