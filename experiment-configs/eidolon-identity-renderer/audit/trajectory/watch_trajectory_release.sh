#!/bin/bash
# Safety net for the operator's GPU handover.
#
# The trajectory runner releases the 4090 itself when it finishes. But the scheduler has no
# "renew" subcommand, and this job's 2h reservation window nominally ends mid-run (it was
# requested at 08:01 for 2h). If the scheduler enforces that and kills the runner, the runner's
# own release never fires and a stale claim would sit on the card - the exact pathology that
# idled it for 3.5h yesterday. The operator wants the GPU next, so guarantee the handover.
#
# Polls until eir-trajectory.service is gone; if the claim is still listed as current, release it
# once and exit. If the runner exited normally the claim is already gone and this is a no-op.
set -u

PY=/home/tim/source/activity/prx-tg/.venv/bin/python
SCHED=/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py
JOB=prx-tg-eir-trajectory-v1
LOG=/home/tim/.hermes/profiles/prx-tg/cache/scratch/trajectory_watchdog.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "=== watchdog start ==="
for i in $(seq 1 120); do                      # 10 h cap at 5-min granularity
  if systemctl --user is-active --quiet eir-trajectory.service; then
    sleep 300
    continue
  fi
  if $PY "$SCHED" status 2>/dev/null | grep -q "current_job: *$JOB"; then
    $PY "$SCHED" release --gpu 4090 --job-id "$JOB" --status completed >>"$LOG" 2>&1
    log "runner gone but claim still held -> RELEASED $JOB for the operator"
  else
    log "runner gone, no claim held -> nothing to do (normal exit)"
  fi
  exit 0
done
log "watchdog gave up after 10h - inspect manually"
