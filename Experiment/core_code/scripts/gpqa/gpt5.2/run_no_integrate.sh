#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
mkdir -p ../analysis/run/gpqa/gpt5.2_no_integrate
PYTHONUNBUFFERED=1 nohup python eval.py --benchmark gpqa \
	--backend codex \
	--seed 42 \
	--num-explores 8 \
	--num-workers 4 \
	--log-dir ../analysis/run/gpqa/gpt5.2_no_integrate \
	--orchestrator-model gpt-5.2 \
	--explore-model gpt-5.2 \
	--integrate-model gpt-5.2 \
	--cache-dirs ../analysis/cache/gpqa/gpt5.2 \
	--no-integrate \
	--effort low \
	> ../analysis/run/gpqa/gpt5.2_no_integrate/delegated.log 2>&1 &
