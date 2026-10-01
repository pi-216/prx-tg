# ECR reboot-resume runbook — 2026-10-01

**Why this file exists:** box `game` is being rebooted while Arm ECR (`eir-cosine-restart`)
was training. This is the deterministic record of what state the reboot left behind and the
exact steps that resume it, so a later session does not have to re-derive them.

## State left behind (all verified before the reboot)

| Fact | Value |
|---|---|
| Arm | `eir-cosine-restart` (ECR), branch `arm/eir-cosine-restart`, mode `confirmatory` |
| Run dir | `experiments/eir-cosine-restart/runs/2026-09-30_0950/` (NAS) |
| Stop | clean SIGTERM via `/tmp/ecr_trainer.pid` at 2026-10-01 08:24:08 EDT; trainer exited ~08:24 |
| Last step | **14,922** (last `training_log.jsonl` row) |
| Resume artifact | `checkpoints/checkpoint_interrupt.pt`, 2026-10-01 08:24:08, 2,823,458,912 bytes |
| Segment | 14,002 → 15,000 (boundary 15,000 not yet reached; its checkpoint/validation will be written by the next segment) |
| Timers | `ecr-launcher.timer`, `ecr-heartbeat.timer` **stopped** (still `enabled`, so they re-arm at boot) |
| GPU claim | `prx-tg-eir-cosine-restart-v5` **deliberately HELD** (state.json → 4090 `busy`), boot_id `16ab89ef-095b-4434-8862-9d71fc393984` |

**Why the claim was held instead of released:** with the box down, a released 4090 would be
claimable by turbo-carnival's queued ComfyUI job (`comfyui-1790848031-3866338-1`, requested
`2026-10-01T09:47:11Z`) — which the reboot would then kill, leaving *their* zombie claim.
Holding ours means the card stays marked busy for the whole outage and nobody starts work
that cannot finish. ECR-v5 (`requested_at 2026-10-01T09:01:25Z`) is *earlier* than that
ComfyUI job, so re-arming v5 also keeps ECR's original FIFO position.

## Steps after boot, in order

1. **Health first.** `nvidia-smi` must answer, and
   `python3 /mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py status` must not report
   `gpu_unhealthy`. A NVML userspace/kernel mismatch has previously failed this probe
   *closed* and denied every claim for ~6h — fix that before anything else.
2. **Confirm the run survived.** Newest checkpoint in
   `experiments/eir-cosine-restart/runs/2026-09-30_0950/checkpoints/` is
   `checkpoint_interrupt.pt` (08:24, 2,823,458,912 bytes); last logged step 14,922.
3. **Release the held claim:**
   ```
   python3 /mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py release \
     --gpu 4090 --job-id prx-tg-eir-cosine-restart-v5 --status completed
   ```
   Expect `released`; state.json 4090 → `idle`. If a poll's `detect_stale` already reaped it
   (heartbeat older than 30 min), that is fine — just continue.
4. **Re-arm the v5 slot** so the launcher reuses it (`next_job_id()` picks the earliest
   *queued* job file of ours) instead of creating a fresh id that would order *behind*
   ComfyUI: set `jobs/prx-tg-eir-cosine-restart-v5.json` → `"status": "queued"`.
   That is the normal post-grant state of a granted job file, and it restores the slot ECR
   already held; it does not move anyone else's place in line.
5. **Resume:**
   ```
   systemctl --user start ecr-launcher.service
   ```
   (or wait for the next :00/:30 tick). Watch `/tmp/ecr_launcher.log` for
   `poll prx-tg-eir-cosine-restart-v5 -> claimed` and
   `training launched in tmux 'ecr-full' ... resuming checkpoint_interrupt.pt`.
   Gate: ≥12 GB free VRAM (`LF_MIN_FREE_VRAM_GB=12`); otherwise it defers to the next tick.
6. **Confirm the timers are live again:** `systemctl --user list-timers --all | grep ecr-`
   → both with a `NEXT`.
7. **Verify the resume is real:** `training_log.jsonl` grows past step 14,922;
   `validation/step0015000/` appears when the boundary is crossed; state.json
   `last_heartbeat` advances.

**No provenance edit is needed** for the reboot: the arm stays `active`, and the run's soft
evidence (log, tensorboard) lives on the NAS and appends continuously.
