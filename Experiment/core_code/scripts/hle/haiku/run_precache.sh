#!/usr/bin/env bash
set -euo pipefail

# To change how many rows are processed, edit `num:` in the YAML (currently 400);
# rows[:SKIP] are eval set, rows[SKIP:num] are training pool.
LOG="/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/haiku/precache_num400.log"
mkdir -p "$(dirname "$LOG")"

cd /data3/peijia/dr-claw/Explain/Experiment/core_code
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py \
	--config scripts/hle/haiku/hle_haiku_precache.yaml \
	> "$LOG" 2>&1 &
echo "PID $! — log: $LOG"
