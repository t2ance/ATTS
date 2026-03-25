#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark hle \
	--backend codex \
	--subset gold \
	--num 200 \
	--seed 42 \
	--num-explores 8 \
	--num-workers 8 \
	--text-only \
	--log-dir ../analysis/run/hle/gpt5.2_low \
	--orchestrator-model gpt-5.2 \
	--explore-model gpt-5.2 \
	--integrate-model gpt-5.2 \
	--cache-dirs ../analysis/cache/hle/gpt5.2_low/gold \
	--effort low \
	> ../analysis/run/hle/gpt5.2_low/gold_delegated.log 2>&1 &
