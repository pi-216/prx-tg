#!/bin/bash
# Safety net for the operator's GPU handover -- guidance sweep.
#
# scratch/run_guidance_sweep.sh releases the 4090 itself on every exit path (EXIT trap
# plus TERM/INT/HUP traps, so the systemd RuntimeMaxSec SIGTERM also lands on a
# release). This watchdog covers the one hole the runner cannot: SIGKILL, which no
# trap can catch -- e.g. systemd's SIGKILL after TimeoutStopSec, or a dead NFS mount
# stalling the release. A stale claim left on the card is the exact pathology that
# idled it for 3.5h; the operator wants the GPU next, so the handover is guaranteed
# from two independent places.
#
# RUN IT BEFORE (or at the same time as) the runner, one per launch:
#   systemd-run --user --unit=guidance-sweep-watchdog --collect \
#     --property=RuntimeMaxSec=3h \
#     /home/tim/source/activity/prx-tg/experiment-configs/eidolon-identity-renderer/audit/probe_runners/watch_guidance_sweep_release.sh
#
# It polls until the runner is gone, then releases whatever guidance-sweep claim is
# still held, once, and exits. On a normal run it is a no-op (the claim is already
# gone). The claim is read from the scheduler's state via `status` -- claim truth is
# state.json, never the job file, which can read 'queued' while actually claimed.
set -u

PY=/home/tim/source/activity/prx-tg/.venv/bin/python
SCHED=/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py
GPU=4090
UNIT=${GS_UNIT:-guidance-sweep.service}
# Full path, not a basename: the fallback check below must be able to tell the runner
# apart from this watchdog's own process, and a bare name can match a stale copy left
# in scratch/ (which is where the runner lived for the 2026-09-28 run).
RUNNER=/home/tim/source/activity/prx-tg/experiment-configs/eidolon-identity-renderer/audit/probe_runners/run_guidance_sweep.sh
JOB_BASE=${GS_JOB:-prx-tg-guidance-sweep-v1}
SCRATCH=/home/tim/source/activity/prx-tg/scratch
JOBID_FILE=${GS_JOBID_FILE:-$SCRATCH/guidance_sweep_job.id}
LOG=${GS_WATCHDOG_LOG:-$SCRATCH/guidance_sweep_watchdog.log}
TRIES=${GS_WATCH_TRIES:-36}          # x 300s = 3h cap
SLEEP_S=${GS_WATCH_SLEEP:-300}
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# Which job does the scheduler believe it is holding for this sweep? Prefer the id
# the runner wrote out (a unique id when the base id was already taken); fall back to
# whatever current_job starts with the base prefix, so a stale/absent id file cannot
# turn the handover into a no-op.
resolve_held_job() {
  jid=""
  if [ -s "$JOBID_FILE" ]; then
    jid=$(head -1 "$JOBID_FILE" | tr -d '[:space:]')
    if [ -n "$jid" ] && $PY "$SCHED" status 2>/dev/null | grep -q "current_job: *$jid"; then
      printf '%s\n' "$jid"
      return 0
    fi
  fi
  $PY "$SCHED" status 2>/dev/null \
    | sed -n 's/^ *current_job: *//p' \
    | grep "^$JOB_BASE" | head -1
}

log "=== guidance-sweep watchdog start (unit $UNIT, job base $JOB_BASE) ==="
for i in $(seq 1 "$TRIES"); do
  # Runner still alive: as a systemd unit, as a plain process, or (worst case) both
  # checks miss because the unit was launched under another name -- the process check
  # makes the watchdog safe to run even then.
  if systemctl --user is-active --quiet "$UNIT"; then
    sleep "$SLEEP_S"
    continue
  fi
  if pgrep -f "$RUNNER" >/dev/null 2>&1; then
    sleep "$SLEEP_S"
    continue
  fi

  JOB=$(resolve_held_job)
  if [ -n "$JOB" ]; then
    out=$($PY "$SCHED" release --gpu "$GPU" --job-id "$JOB" --status completed 2>&1)
    log "runner gone but claim $JOB still held -> RELEASED for the operator: $out"
  else
    log "runner gone, no guidance-sweep claim held -> nothing to do (normal exit)"
  fi
  exit 0
done
log "watchdog gave up after $((TRIES * SLEEP_S))s - inspect manually"
