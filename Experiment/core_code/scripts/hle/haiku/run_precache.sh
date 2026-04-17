#!/usr/bin/env bash
set -euo pipefail

NUM=400  # process rows[:NUM]; rows[:SKIP] are eval set, rows[SKIP:NUM] are training pool

CACHE_DIR="/data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/haiku/gold"
LOG="/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/haiku/precache_num${NUM}.log"
mkdir -p "$(dirname "$LOG")"

PYTHONUNBUFFERED=1 nohup python /data3/peijia/dr-claw/Explain/Experiment/core_code/precache_explores.py \
	--backend claude \
	--cache-dirs "$CACHE_DIR" \
	--subset gold \
	--num "$NUM" \
	--num-explores 8 \
	--num-workers 8 \
	--seed 42 \
	--text-only \
	--explore-model claude-haiku-4-5-20251001 \
	> "$LOG" 2>&1 &
echo "PID $! — log: $LOG"
