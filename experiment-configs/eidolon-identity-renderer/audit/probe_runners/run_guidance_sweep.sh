#!/bin/bash
# prx-tg -- guidance-sweep probe runner (scheduler-safe).
#
# WHAT IT DOES, IN ORDER
#   1. takes an exclusive lock (two runners must never double-request the card)
#   2. refuses to claim the GPU unless the probe AND its checkpoint exist
#   3. requests the 4090 through the file-based GPU scheduler, polls, activates
#   4. runs scripts/diag_render_guidance_sweep.py (one probe, minutes)
#   5. RELEASES the claim unconditionally, on every exit path, with status
#      completed|failed matching the probe's exit code
#
# WHY IT IS WRITTEN THIS WAY (each of these has cost GPU time before)
#   * NEVER `set -e`. A crashing probe must not abort the script before the release
#     runs; we capture the probe's exit code ourselves and release with the
#     matching status.
#   * The release lives in ONE function, called from an EXIT trap, so it cannot be
#     skipped by an early `exit`, a `set -u` unbound-variable abort, a crash, or a
#     timeout. A missed release idles the 4090 indefinitely -- that pathology
#     already cost 3.5h.
#   * TERM/INT/HUP are trapped and converted into `exit 143`, which fires the EXIT
#     trap. That is the path systemd takes when RuntimeMaxSec expires. SIGKILL can
#     never be trapped, so the launch line also sets TimeoutStopSec=120 (room for
#     the release to land) and scratch/watch_guidance_sweep_release.sh is the last
#     resort.
#   * The reservation is short (`--duration 1h`) on purpose. Project rule: reserve
#     the GPU only to the next natural break, never for a whole multi-day run.
#     This probe is minutes long; the card goes back the moment it lands.
#   * The probe is pinned to ONE checkpoint, resolved and written to the log BEFORE
#     the claim. If a pinned checkpoint is missing the runner aborts WITHOUT
#     claiming -- it never silently substitutes a different step (skip vs
#     substitute; a substituted checkpoint invalidates the sweep).
#   * Optional flags are passed as a bash array, so an unset value never becomes a
#     bare `--cells` with a missing argument (flag-arity mistake).
#   * Claim truth lives in the scheduler's state.json, not in the job file: a job
#     file can read 'queued' while the job is actually claimed. All decisions here
#     go through the scheduler CLI, never through job files.
#
# This script does not touch the long EIR training units (eir-launcher.service /
# eir-heartbeat.service) and never starts a systemd unit itself.
#
# Env overrides (all optional):
#   GS_CKPT       pin an exact checkpoint path (must exist; never substituted)
#   GS_CELLS      value(s) for the probe's optional --cells flag, space-separated
#   GS_JOB        base job id        (default prx-tg-guidance-sweep-v1)
#   GS_VRAM       GB to request      (default 10, the ask the EIR trajectory probe used)
#   GS_DURATION   reservation length (default 1h)
#   GS_OUT        probe output dir   (default <scratch>/guidance_sweep_out)
#   GS_LOG        log file           (default <scratch>/guidance_sweep_runner.log)
#   GS_POLL_TRIES / GS_POLL_SLEEP    queue-wait budget (default 60 x 30s = 30 min)
set -u

REPO=/home/tim/source/activity/prx-tg
PY=$REPO/.venv/bin/python
SCHED=/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py
GPU=4090
PROBE=$REPO/scripts/diag_render_guidance_sweep.py
SCRATCH=$REPO/scratch
CKPT_DIR=/mnt/nas-ai-models/training-data/prx-tg/eidolon-identity-renderer/runs/2026-09-25_0734/checkpoints

JOB_BASE=${GS_JOB:-prx-tg-guidance-sweep-v1}
JOB=$JOB_BASE
VRAM=${GS_VRAM:-10}
DURATION=${GS_DURATION:-1h}
OUT=${GS_OUT:-$SCRATCH/guidance_sweep_out}
LOG=${GS_LOG:-$SCRATCH/guidance_sweep_runner.log}
JOBID_FILE=$SCRATCH/guidance_sweep_job.id
LOCK=$SCRATCH/run_guidance_sweep.lock
POLL_TRIES=${GS_POLL_TRIES:-60}
POLL_SLEEP=${GS_POLL_SLEEP:-30}

mkdir -p "$SCRATCH" "$OUT" 2>/dev/null
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# ---- one runner at a time: a second request would double-book the queue -----
exec 9>"$LOCK" || { log "cannot open lock $LOCK"; exit 1; }
if ! flock -n 9; then
  log "another run_guidance_sweep already holds $LOCK - refusing to double-request"
  exit 1
fi

CLAIMED=0
RELEASED=0

# Only a clean probe exit is recorded as `completed`; everything else is `failed`.
status_for_rc() { if [ "$1" -eq 0 ]; then echo completed; else echo failed; fi; }

# THE single release path -- called from the EXIT trap only, so every exit route
# (normal, error, `set -u` abort, signal, timeout) ends here. Never releases a
# claim we do not own.
release_claim() {
  rc_status=$1
  if [ "$RELEASED" = "1" ]; then return 0; fi
  if [ "$CLAIMED" != "1" ]; then
    log "release skipped: nothing was claimed, nothing held"
    return 0
  fi
  RELEASED=1
  out=$($PY "$SCHED" release --gpu "$GPU" --job-id "$JOB" --status "$rc_status" 2>&1)
  log "RELEASE $JOB --status $rc_status -> $out"
  # Read-back: a release is only believable if the scheduler agrees.
  if $PY "$SCHED" status 2>/dev/null | grep -q "current_job: *$JOB"; then
    log "WARNING: $JOB still listed as current_job after release - retrying once"
    out=$($PY "$SCHED" release --gpu "$GPU" --job-id "$JOB" --status "$rc_status" 2>&1)
    log "RELEASE retry -> $out"
  fi
}

on_signal() {
  log "signal received - unwinding so the claim is released"
  exit 143
}

on_exit() {
  rc=$?
  release_claim "$(status_for_rc "$rc")"
  log "=== guidance-sweep runner exit rc=$rc (release fired=$RELEASED) ==="
  exit "$rc"
}

trap on_exit EXIT
trap on_signal TERM INT HUP

cd "$REPO" 2>/dev/null || { log "cannot cd $REPO"; exit 1; }
log "=== guidance-sweep runner start (pid $$, host $(hostname), gpu $GPU) ==="
log "git commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown), dirty files: $(git status --porcelain 2>/dev/null | wc -l)"

# ---- pre-claim gates: never claim the card for a run that cannot happen -----
if [ ! -f "$PROBE" ]; then
  log "ABORT: probe $PROBE does not exist - nothing requested, nothing held"
  exit 1
fi

if [ -n "${GS_CKPT:-}" ]; then
  CKPT=${GS_CKPT}
  if [ ! -f "$CKPT" ]; then
    log "ABORT: GS_CKPT=$CKPT is pinned but missing - refusing to substitute another step - nothing requested, nothing held"
    exit 1
  fi
  log "checkpoint PINNED by GS_CKPT: $CKPT"
else
  # Default object of study: the newest published step checkpoint, resolved by step
  # NUMBER (zero-padded names, so lexical order == numeric) rather than mtime, which
  # a NAS rsync can rewrite. The resolved absolute path is logged, so the sweep's
  # artifact is recorded even though it was not pinned by hand.
  CKPT=$(ls "$CKPT_DIR"/checkpoint_step*.pt 2>/dev/null | sort | tail -1)
  if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    log "ABORT: no checkpoint_step*.pt under $CKPT_DIR - nothing requested, nothing held"
    exit 1
  fi
  log "checkpoint RESOLVED to newest step: $CKPT"
fi

CELL_ARGS=()
if [ -n "${GS_CELLS:-}" ]; then
  CELL_ARGS=(--cells ${GS_CELLS})
  log "cells from GS_CELLS: ${CELL_ARGS[*]:-}"
else
  log "cells: probe default (--cells not passed)"
fi

# ---- queue, then poll for our turn -----------------------------------------
req_out=$($PY "$SCHED" request --gpu "$GPU" --project prx-tg --vram "$VRAM" --duration "$DURATION" --job-id "$JOB" 2>&1)
req_rc=$?
if [ "$req_rc" -ne 0 ]; then
  # A finished job file from an earlier sweep still occupies the id (request_job
  # raises FileExistsError on reuse). Fall back to a unique id rather than polling a
  # job the scheduler considers finished: poll() would answer `not_queued` forever
  # while our entry sat at the head of the queue, idling the card behind a dead
  # queue head.
  log "request $JOB refused (rc=$req_rc): $req_out - retrying with a unique job id"
  JOB="$JOB_BASE-$(date +%Y%m%d-%H%M%S)"
  req_out=$($PY "$SCHED" request --gpu "$GPU" --project prx-tg --vram "$VRAM" --duration "$DURATION" --job-id "$JOB" 2>&1)
  req_rc=$?
fi
log "request $JOB -> rc=$req_rc: $req_out"
if [ "$req_rc" -ne 0 ]; then
  log "ABORT: could not queue the job twice - nothing claimed, nothing held"
  exit 1
fi
# Hand the resolved id to the watchdog, which has no other way to know it.
printf '%s\n' "$JOB" > "$JOBID_FILE"

R=""
for i in $(seq 1 "$POLL_TRIES"); do
  R=$($PY "$SCHED" poll --gpu "$GPU" --job-id "$JOB" 2>&1 | tail -1)
  log "poll $i/$POLL_TRIES -> $R"
  echo "$R" | grep -qi "claimed" && break
  sleep "$POLL_SLEEP"
done
if ! echo "$R" | grep -qi "claimed"; then
  log "NEVER GRANTED within $((POLL_TRIES * POLL_SLEEP))s - exiting; nothing claimed, nothing held"
  exit 1
fi
# `poll` performs the claim itself, so the claim is held from here on: everything
# below is covered by the EXIT trap.
CLAIMED=1
log "granted (claim held, this is now the exit-trap's responsibility)"

act=$($PY "$SCHED" activate --gpu "$GPU" --job-id "$JOB" 2>&1)
log "activate -> $act"
case "$act" in
  *not_owner*|*already_released*)
    log "ABORT: activate says we no longer own $GPU - not touching the GPU"
    exit 1
    ;;
esac

# ---- the probe -------------------------------------------------------------
log "=== probe start: $PROBE --checkpoint $CKPT --out-dir $OUT ==="
$PY "$PROBE" --checkpoint "$CKPT" --out-dir "$OUT" ${CELL_ARGS[@]+"${CELL_ARGS[@]}"} >>"$LOG" 2>&1
probe_rc=$?
log "=== probe exit=$probe_rc -> will release as $(status_for_rc "$probe_rc") ==="
exit "$probe_rc"
