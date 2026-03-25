#!/usr/bin/env bash
set -euo pipefail
unset CLAUDECODE 2>/dev/null || true
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark hle \
	--backend claude --subset gold --num 100 --seed 42 \
	--num-explores 16 --num-workers 8 --text-only \
	--log-dir ../analysis/run/hle/multi_model_effort_low \
	--method tts-agent-multi \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 --integrate-model claude-sonnet-4-6 \
	--cache-dirs "haiku:../analysis/cache/hle/haiku/gold,sonnet:../analysis/cache/hle/sonnet/gold,opus:../analysis/cache/hle/opus/gold" \
	--model-budgets "haiku:8,sonnet:8,opus:4" \
	--no-integrate --exploration-effort low \
	> ../analysis/run/hle/multi_model_effort_low/delegated.log 2>&1 &
