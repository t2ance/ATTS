#!/usr/bin/env bash
set -euo pipefail
unset CLAUDECODE 2>/dev/null || true
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark babyvision \
	--backend claude --seed 42 --num-explores 16 --num-workers 8 \
	--log-dir ../analysis/run/babyvision/multi_model_effort_high \
	--method tts-agent-multi \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 --integrate-model claude-sonnet-4-6 \
	--cache-dirs "haiku:../analysis/cache/babyvision/haiku,sonnet:../analysis/cache/babyvision/sonnet,opus:../analysis/cache/babyvision/opus" \
	--model-budgets "haiku:8,sonnet:8,opus:2" \
	--no-integrate --exploration-effort high \
	> ../analysis/run/babyvision/multi_model_effort_high/delegated.log 2>&1 &
