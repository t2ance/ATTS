#!/usr/bin/env bash
set -euo pipefail

NUM=400  # process rows[:NUM]; rows[:SKIP] are eval set, rows[SKIP:NUM] are training pool
LOG="/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/haiku/precache_num${NUM}.log"
mkdir -p "$(dirname "$LOG")"

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
# -o num overrides YAML when NUM is changed above (default in YAML is 400).
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/hle/haiku/hle_haiku_precache.yaml \
	-o num="$NUM" \
	> "$LOG" 2>&1 &
echo "PID $! — log: $LOG"
