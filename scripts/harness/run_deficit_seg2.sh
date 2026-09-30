#!/bin/bash
# prx-tg deficit report -- stratum2 seg2 for the MASKED G0 gates (render side).
#
# WHY: G0a/G0c are SKIN-region gates (they need a seg mask), and the renders have no seg
# masks -- so on generated images the frozen suite falls back to whole-frame statistics
# (the `_full` variants) which every existing arm verdict was scored on. To measure the
# masked gates on renders we must SEGMENT THE RENDERS. That is what stratum-hq's Sapiens2
# seg2 pass is for (operator steer, 2026-09-28).
#
# WHAT IT SEGMENTS
#   178 EIR renders  (160 `reconstruction/` = model generating from a real sample's own
#                     identity+geometry conditioning; 18 render-guidance-sweep panels)
#   +12 REAL FFHQ images, the SAME ids as the deficit report's `real` leg -- these are a
#       TAXONOMY CHECK, not a mask source. The frozen masked bands were calibrated with the
#       stratum tree's v1 seg (Face_Neck = 2); seg2 is DOME_CLASSES_29 (Face_Neck = 3).
#       Segmenting the same faces both ways lets us validate the substitution (IoU) BEFORE
#       any render mask is compared against the frozen band.
#
# SAFETY (each rule paid for by an earlier incident)
#   * NEVER `set -e`: a failing pass must not skip the release.
#   * The release lives in ONE function called from an EXIT trap, so no exit route (error,
#     unbound var, signal, timeout) can orphan the claim. A stale claim idles the card.
#   * A FRESH, uniquely-suffixed job id every launch: scheduler job ids are single-use
#     (request with an existing id raises FileExistsError), so a fixed id silently forces
#     the unique-id fallback -- name it correctly up front instead.
#   * Staging + pre-flight gates run BEFORE the request: never claim the card for a run
#     that cannot happen.
#   * 5B seg needs ~22GB of the 24GB card; one 5B model at a time on the 4090, and the
#     input list must be readable on the executing machine (it is on the shared NAS).
#
# USAGE: systemd-run --user --unit=deficit-seg2 --collect --property=TimeoutStopSec=120 \
#          --property=RuntimeMaxSec=5400 /path/to/this/script
set -u

REPO=/home/tim/source/activity/prx-tg
PY=$REPO/.venv/bin/python
SCHED=/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py
SHQ=/home/tim/source/activity/stratum-hq
GPU=4090
ROOT=/mnt/nas-ai-models/training-data/prx-tg/diagnostics/deficit-2026-09-28
IN=$ROOT/render_in
OUT=$ROOT/seg2
LIST=$ROOT/seg2_list.txt
LEGS=$REPO/research/avenues/gates/deficit-2026-09-28
STRATUM=/mnt/nas-ai-models/training-data/ffhq/stratum
RAW=/mnt/nas-ai-models/training-data/ffhq/raw
LOG=$REPO/scratch/deficit_seg2.log
JOB="prx-tg-deficit-seg2-$(date +%Y%m%d-%H%M%S)"
VRAM=23
CLAIMED=0
RELEASED=0

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

status_for_rc() { if [ "$1" -eq 0 ]; then echo completed; else echo failed; fi; }

release_claim() {
  rc_status=$1
  [ "$RELEASED" = "1" ] && return 0
  if [ "$CLAIMED" != "1" ]; then log "release skipped: nothing was claimed"; return 0; fi
  RELEASED=1
  out=$($PY "$SCHED" release --gpu "$GPU" --job-id "$JOB" --status "$rc_status" 2>&1)
  log "RELEASE $JOB --status $rc_status -> $out"
  if $PY "$SCHED" status 2>/dev/null | grep -q "current_job: *$JOB"; then
    log "WARNING: still current_job after release - retrying once"
    log "RELEASE retry -> $($PY "$SCHED" release --gpu "$GPU" --job-id "$JOB" --status "$rc_status" 2>&1)"
  fi
}
on_signal() { log "signal received - unwinding so the claim is released"; exit 143; }
on_exit() { rc=$?; release_claim "$(status_for_rc "$rc")"; log "=== seg2 runner exit rc=$rc (release fired=$RELEASED) ==="; exit "$rc"; }
trap on_exit EXIT
trap on_signal TERM INT HUP

log "=== deficit seg2 runner start (pid $$, host $(hostname)) ==="

# ---- stage inputs (no GPU) ---------------------------------------------------
# COPY, do not symlink. stratum2 resolves each input path and asserts it lies INSIDE the input
# dir ("error: '<path>' is not in the subpath of '<input dir>'"), which a symlink pointing at
# the NAS or at a leg dir violates. The first attempt symlinked: the 5B model loaded (4 min)
# and then the run died on image 1 with 0 artifacts. Real files are the only safe staging.
mkdir -p "$IN" "$OUT"
$PY - "$LEGS" "$IN" >> "$LOG" 2>&1 <<'PYEOF'
import json, os, shutil, sys

legs, root = sys.argv[1], sys.argv[2]
ids = json.load(open(f"{legs}/real/sampled.json"))["kept"][:12]
raw = "/mnt/nas-ai-models/training-data/ffhq/raw"
srcs = []
for leg in ("eir-recon", "eir-sweep"):
    d = f"{legs}/{leg}/images"
    if os.path.isdir(d):
        srcs += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")]
# the taxonomy-check sample: SAME real ids as the deficit report's `real` leg
srcs += [f"{raw}/{i}.png" for i in ids]
n = 0
for s in srcs:
    if not os.path.isfile(s):          # isfile() follows symlinks -> the leg dirs are symlinks
        continue
    dest = os.path.join(root, os.path.basename(s))
    if os.path.lexists(dest):
        os.unlink(dest)                # clears the stale symlink left by the failed first run
    shutil.copy2(os.path.realpath(s), dest)
    n += 1
print(f"staged {n} files (copied); taxonomy sample ids: {ids}")
PYEOF
ls "$IN" | grep -E '\.png$' > "$LIST"
n_in=$(wc -l < "$LIST")
log "staged $n_in images -> $IN; list -> $LIST"

# ---- pre-flight gates: never claim the card for a run that cannot happen ------
[ -s "$LIST" ] || { log "ABORT: empty image list"; exit 1; }
[ -x "$SHQ/.venv/bin/stratum2" ] || { log "ABORT: stratum2 CLI missing"; exit 1; }
N_RECON=$(grep -c "recon" "$LIST" || true); N_SWEEP=$(grep -c '^s[0-9]' "$LIST" || true)
N_REAL=$(grep -cE '^[0-9]{5}\.png$' "$LIST" || true)   # 5-digit FFHQ ids; NOT '^0[0-9]{4}'
log "list composition: recon=$N_RECON sweep=$N_SWEEP real-check=$N_REAL"
[ "$N_REAL" -ge 10 ] || log "WARNING: fewer than 10 real taxonomy-check images staged"

# ---- queue, poll, activate ---------------------------------------------------
req=$($PY "$SCHED" request --gpu "$GPU" --project prx-tg --vram "$VRAM" --duration 2h --job-id "$JOB" 2>&1)
log "request $JOB -> $req"
R=""
for i in $(seq 1 40); do
  R=$($PY "$SCHED" poll --gpu "$GPU" --job-id "$JOB" 2>&1 | tail -1)
  log "poll $i/40 -> $R"
  echo "$R" | grep -qi "claimed" && break
  sleep 30
done
echo "$R" | grep -qi "claimed" || { log "NEVER GRANTED - nothing claimed, nothing held"; exit 1; }
CLAIMED=1
log "granted (claim held; the EXIT trap owns it from here)"
act=$($PY "$SCHED" activate --gpu "$GPU" --job-id "$JOB" 2>&1)
log "activate -> $act"
case "$act" in *not_owner*|*already_released*) log "ABORT: lost ownership"; exit 1;; esac

# ---- the pass ----------------------------------------------------------------
cd "$SHQ" || { log "ABORT: cannot cd $SHQ"; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
export SAPIENS2_SIZE=5b
export SAPIENS2_CACHE_DIR=/mnt/nas-ai-models/sapiens2
export HF_HOME=/mnt/nas-ai-models/huggingface-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
log "=== stratum2 seg2 start: in=$IN out=$OUT n=$n_in SAPIENS2_SIZE=$SAPIENS2_SIZE ==="
stratum2 process "$IN" --output "$OUT" --image-list "$LIST" --passes seg2 --device cuda >> "$LOG" 2>&1
rc=$?
log "=== stratum2 seg2 exit=$rc ==="
# A per-image OOM does NOT kill stratum2 (it warns and continues), so also count artifacts.
n_seg=$(ls "$OUT"/*/seg2.npy 2>/dev/null | wc -l)
log "seg2 artifacts written: $n_seg / $n_in"
exit "$rc"
