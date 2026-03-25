#!/usr/bin/env bash
set -euo pipefail
unset CLAUDECODE 2>/dev/null || true
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark gpqa \
	--backend claude --seed 42 --num-explores 16 --num-workers 16 \
	--log-dir ../analysis/run/gpqa/multi_model_effort_medium \
	--method tts-agent-multi \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 --integrate-model claude-sonnet-4-6 \
	--cache-dirs "haiku:../analysis/cache/gpqa/haiku,sonnet:../analysis/cache/gpqa/sonnet,opus:../analysis/cache/gpqa/opus" \
	--model-budgets "haiku:8,sonnet:8,opus:4" \
	--no-integrate --exploration-effort medium \
	> ../analysis/run/gpqa/multi_model_effort_medium/delegated.log 2>&1 &
