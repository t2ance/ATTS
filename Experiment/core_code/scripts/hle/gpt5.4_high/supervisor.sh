#!/usr/bin/env bash
# Supervisor for HLE 5.4-high precache: launches precache, watches process life
# via kill -0 (NOT pgrep pattern), restarts on death after 30min cooldown,
# cleans timeout placeholders each cycle.
#
# Pattern-match approach (pgrep -f) was abandoned because the supervisor's
# command line itself contains the pattern literal, causing self-matches.
# A standalone .sh file plus tracked-PID via kill -0 is robust against this.
set +e
CACHE_DIR=/data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/gpt5.4_high/gold
LOG=/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/gpt5.4_high/precache.log
CFG=/data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/hle/gpt5.4_high/hle_gpt5.4_high_precache.yaml
PIDFILE=/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/gpt5.4_high/precache.pid
TARGET=800
COOLDOWN=1800
TICK=1800
STALL_LIMIT=3

cd /data3/peijia/dr-claw/Explain/Experiment/core_code

clean_placeholders() {
  local files n=0
  files=$(find $CACHE_DIR -name result.json 2>/dev/null | xargs grep -l '"timed_out": true' 2>/dev/null)
  if [ -n "$files" ]; then
    for f in $files; do
      rm -f "$f"
      rmdir "$(dirname $f)" 2>/dev/null
      n=$((n+1))
    done
    echo "$(date -Is) CLEANUP: removed $n timeout placeholder(s)"
  fi
}

real_count() {
  local total to
  total=$(find $CACHE_DIR -name result.json 2>/dev/null | wc -l)
  to=$(find $CACHE_DIR -name result.json 2>/dev/null | xargs grep -l '"timed_out": true' 2>/dev/null | wc -l)
  echo $((total - to))
}
ok_total()   { local n; n=$(grep -c '"HTTP/1.1 200 OK"' $LOG 2>/dev/null); echo "${n:-0}"; }
err_total()  { local n; n=$(grep -c '"HTTP/1.1 429' $LOG 2>/dev/null); echo "${n:-0}"; }

no_real_progress=0
prev_real=$(real_count)
echo "$(date -Is) START: cache=$prev_real target=$TARGET cfg=$CFG"

while true; do
  cur=$(real_count)
  if [ "$cur" -ge "$TARGET" ]; then
    echo "$(date -Is) DONE: cache=$cur"
    break
  fi
  clean_placeholders

  # Launch with setsid so the whole tree (conda wrapper + python child) is its own
  # process group, pgid == pypid. Group kill (kill -- -$pgid) propagates to children
  # — fixes orphan-python-child bug where killing wrapper alone left workers running.
  PYTHONUNBUFFERED=1 setsid nohup conda run -n explain --no-capture-output python precache_explores.py --config $CFG >> $LOG 2>&1 &
  pypid=$!
  echo "$pypid" > $PIDFILE
  pre_ok=$(ok_total)
  # Reset stall counter at every relaunch — each Round gets its own 90min patience
  no_real_progress=0
  echo "$(date -Is) LAUNCHED: pid=$pypid cache=$cur ok_total_pre=$pre_ok"

  # Inner loop: tick every TICK seconds while ANY process in the group is alive.
  # `kill -0 -- -$pypid` checks the group, not just the wrapper.
  while kill -0 -- -$pypid 2>/dev/null; do
    sleep $TICK
    clean_placeholders
    new=$(real_count)
    delta=$((new - prev_real))
    ok_now=$(ok_total)
    err_now=$(err_total)
    if [ "$delta" -eq 0 ]; then
      no_real_progress=$((no_real_progress + 1))
      echo "$(date -Is) STALL: pid=$pypid cache=$new (no progress ${no_real_progress}×30min) ok_total=$ok_now err_total=$err_now"
      if [ "$no_real_progress" -ge "$STALL_LIMIT" ]; then
        echo "$(date -Is) FORCED_KILL: 90min no real progress, killing pgid=$pypid for fresh cooldown"
        kill -- -$pypid 2>/dev/null
        sleep 5
        kill -9 -- -$pypid 2>/dev/null
        break
      fi
    else
      no_real_progress=0
      echo "$(date -Is) PROGRESS: pid=$pypid cache=$new delta=+$delta ok_total=$ok_now err_total=$err_now"
    fi
    prev_real=$new
    if [ "$new" -ge "$TARGET" ]; then break; fi
  done

  cur=$(real_count)
  if [ "$cur" -ge "$TARGET" ]; then
    echo "$(date -Is) DONE: cache=$cur"
    break
  fi
  echo "$(date -Is) DEAD_OR_KILLED: cache=$cur — entering ${COOLDOWN}s cooldown"
  sleep $COOLDOWN
  prev_real=$(real_count)
done
