#!/bin/bash
set -u
TRAIN_LOG=/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/grpo_8b_sft_2gpu_bs96.log
SERVE_SCRIPT=/data3/peijia/dr-claw/Explain/Experiment/core_code/training/scripts/serve_judge_1gpu.sh
SWAP_LOG=/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/judge_swap.log
JUDGE_LOG=/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/judge_1gpu.log
TRIGGER="local_global_step_folder.*global_step_20"
NEW_UTIL=0.5

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SWAP_LOG"; }

log "monitor start, watching for: $TRIGGER"

tail -n 0 -F "$TRAIN_LOG" | while IFS= read -r line; do
    if echo "$line" | grep -qE "$TRIGGER"; then
        log "TRIGGER MATCHED: $line"
        log "window: save_checkpoint ongoing, ~150s until testing"
        break
    fi
done

log "step 1/5: snapshot current judge PIDs"
OLD_PIDS=$(pgrep -f "vllm serve Qwen/Qwen3-8B.*--port 8000" || true)
log "old PIDs: $OLD_PIDS"

log "step 2/5: edit serve script util 0.9 -> $NEW_UTIL"
sed -i 's/--gpu-memory-utilization 0\.9/--gpu-memory-utilization '"$NEW_UTIL"'/' "$SERVE_SCRIPT"
grep "gpu-memory-utilization" "$SERVE_SCRIPT" | tee -a "$SWAP_LOG"

log "step 3/5: kill old Judge"
for pid in $OLD_PIDS; do kill -TERM "$pid" 2>&1 | tee -a "$SWAP_LOG" || true; done
# wait for exit
for i in $(seq 1 30); do
    if ! pgrep -f "vllm serve Qwen/Qwen3-8B.*--port 8000" >/dev/null; then
        log "old judge exited after ${i}s"
        break
    fi
    sleep 1
done
if pgrep -f "vllm serve Qwen/Qwen3-8B.*--port 8000" >/dev/null; then
    log "WARN: old judge still alive after 30s, SIGKILL"
    pkill -9 -f "vllm serve Qwen/Qwen3-8B.*--port 8000" || true
    sleep 2
fi

log "step 4/5: relaunch judge via serve script"
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
nohup bash "$SERVE_SCRIPT" > "$JUDGE_LOG" 2>&1 &
NEW_BASH_PID=$!
log "new bash wrapper PID: $NEW_BASH_PID"
disown || true

log "step 5/5: health poll /v1/models (max 120s)"
for i in $(seq 1 120); do
    if curl -sf -o /dev/null -m 2 http://127.0.0.1:8000/v1/models 2>/dev/null; then
        log "judge READY after ${i}s"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$SWAP_LOG"
        exit 0
    fi
    sleep 1
done
log "ERROR: judge not ready after 120s"
exit 1
