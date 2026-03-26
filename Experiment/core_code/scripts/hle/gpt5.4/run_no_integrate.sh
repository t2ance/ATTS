#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/hle/gpt5.4_no_integrate
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark hle \
	--backend codex \
	--subset gold \
	--num 100 \
	--seed 42 \
	--num-explores 8 \
	--num-workers 4 \
	--text-only \
	--no-integrate \
	--log-dir ../analysis/run/hle/gpt5.4_no_integrate \
	--orchestrator-model gpt-5.4 \
	--explore-model gpt-5.4 \
	--integrate-model gpt-5.4 \
	--cache-dirs ../analysis/cache/hle/gpt5.4/gold \
	> ../analysis/run/hle/gpt5.4_no_integrate/delegated.log 2>&1 &
