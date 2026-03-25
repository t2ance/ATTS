#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark hle \
	--backend claude \
	--method self-refine \
	--subset gold \
	--num 100 \
	--text-only \
	--seed 42 \
	--num-explores 8 \
	--num-workers 16 \
	--log-dir ../analysis/run/hle/sonnet_self_refine \
	--orchestrator-model claude-sonnet-4-6 \
	--explore-model claude-sonnet-4-6 \
	--integrate-model claude-sonnet-4-6 \
	--no-cache-only \
	--cache-dirs ../analysis/cache/hle/sonnet_self_refine/gold \
	> ../analysis/run/hle/sonnet_self_refine/self_refine.log 2>&1 &
