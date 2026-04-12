#!/usr/bin/env bash
set -euo pipefail

unset CLAUDECODE 2>/dev/null || true

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 python eval.py --benchmark hle \
	--backend vllm \
	--seed 42 \
	--num-explores 8 \
	--num-workers 3 \
	--skip 100 \
	--num 200 \
	--subset gold \
	--text-only \
	--log-dir ../analysis/run/hle/qwen3_8b_base \
	--orchestrator-model qwen3-8b-base \
	--explore-model qwen3-8b-base \
	--integrate-model qwen3-8b-base \
	--cache-dirs ../analysis/cache/hle/haiku/gold \
	--no-integrate \
	2>&1 | tee tmp/eval_qwen3_8b_base_q101_300.log
