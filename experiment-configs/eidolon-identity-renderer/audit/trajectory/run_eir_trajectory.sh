#!/bin/bash
# Identity trajectory across checkpoints, for eidolon-identity-renderer.
# Answers: does identity move toward its goal as training progresses?
# ONE FIXED PROTOCOL at every checkpoint (same pool, same seeds, same index size) - the
# intra-sample temporal tracking the operator prefers over cross-sample aggregation.
#
# Per checkpoint, two curves:
#   1. diag_identity_render_to_render.py -> render-to-render R@1 (quality-invariant) and
#      render-to-real R@1 (absolute, expected chance-floored by P1 grain - diagnostic only)
#   2. diag_eidolon_zg_dilution.py (default sweep) -> the geometry response, already known
#      to be sensitive (7-9 px at 5000 -> 46-93 px at 7000), so it is the positive control
#      that the instrument can see change at all across this trajectory.
#
# Never sets -e: a missing checkpoint or a crashing probe must not strand the claim.
set -u

REPO=/home/tim/source/activity/prx-tg
PY=$REPO/.venv/bin/python
SCHED=/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py
JOB=prx-tg-eir-trajectory-v1
RUN=/mnt/nas-ai-models/training-data/prx-tg/eidolon-identity-renderer/runs/2026-09-25_0734
CK=$RUN/checkpoints
BASE=/home/tim/.hermes/profiles/prx-tg/cache/scratch
LOG=$BASE/trajectory_runner.log
OUT=$BASE/eir_trajectory
CFG=$REPO/experiment-configs/eidolon-identity-renderer/config.yaml

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
cd "$REPO" || exit 1
mkdir -p "$OUT"
log "=== trajectory runner start (pid $$) ==="

$PY "$SCHED" request --gpu 4090 --project prx-tg --vram 10 --duration 2h --job-id "$JOB" >>"$LOG" 2>&1
log "requested $JOB"

R=""
for i in $(seq 1 720); do
  R=$($PY "$SCHED" poll --gpu 4090 --job-id "$JOB" 2>&1 | tail -1)
  log "poll -> $R"
  echo "$R" | grep -qi "claimed" && break
  sleep 60
done
if ! echo "$R" | grep -qi "claimed"; then
  log "NEVER GRANTED after 12h - exiting, nothing claimed, nothing held"
  exit 1
fi
$PY "$SCHED" activate --gpu 4090 --job-id "$JOB" >>"$LOG" 2>&1
log "granted + activated"

for STEP in 1000 3000 5000 7000 10000; do
  ck=$(ls "$CK"/checkpoint_step$(printf '%07d' "$STEP").pt 2>/dev/null | head -1)
  if [ -z "$ck" ] || [ ! -f "$ck" ]; then
    log "step $STEP: checkpoint MISSING - skipped (recorded, not substituted)"
    continue
  fi
  log "=== step $STEP: $ck ==="
  if [ -f scripts/diag_identity_render_to_render.py ]; then
    $PY scripts/diag_identity_render_to_render.py --config "$CFG" --checkpoint "$ck" \
        --samples 40 --out-dir "$OUT/step${STEP}_identity" >>"$LOG" 2>&1
    log "step $STEP identity probe exit=$?"
  else
    log "step $STEP: identity probe MISSING"
  fi
  if [ -f scripts/diag_eidolon_zg_dilution.py ]; then
    $PY scripts/diag_eidolon_zg_dilution.py --config "$CFG" --checkpoint "$ck" \
        --out-dir "$OUT/step${STEP}_geometry" >>"$LOG" 2>&1
    log "step $STEP geometry probe exit=$?"
  else
    log "step $STEP: geometry probe MISSING"
  fi
done

$PY "$SCHED" release --gpu 4090 --job-id "$JOB" --status completed >>"$LOG" 2>&1
log "released; trajectory runner done"
