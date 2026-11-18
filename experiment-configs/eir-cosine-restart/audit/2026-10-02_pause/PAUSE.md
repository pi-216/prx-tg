# Arm ECR — deliberate pause at step 16,916

**When:** 2026-10-02 07:46:40 EDT · **Arm:** `eir-cosine-restart` (ECR), mode `confirmatory`
**Run:** `experiments/eir-cosine-restart/runs/2026-09-30_0950/`
**Trigger:** operator directive — stop the running segment now instead of letting it reach the 17000
boundary; no further 1k segments are to be scheduled until the general research harness is cleaned.

**This is neither a failure nor a verdict.** The pre-registered gate remains uncomputed and no
`verdict` is claimed. The directional mid-run read stands as recorded in `../15k_review/RESULTS.md`.

---

## What was done, in order

| # | Action | Evidence |
|---|---|---|
| 1 | `ecr-launcher.timer` stopped **and disabled** | `is-active` → inactive, `is-enabled` → disabled |
| 2 | `ecr-heartbeat.timer` stopped **and disabled** | same — done *first* so the heartbeat could not race the release and stamp it `failed` |
| 3 | `SIGTERM` to the trainer's own pid file (`3009138`) | process gone after ~40 s |
| 4 | Trainer took its graceful path | log: `⚠️ Training interrupted by user (SIGINT/SIGTERM)` → `Saving checkpoint` → `Saved checkpoint to …/checkpoint_interrupt.pt` |
| 5 | Claim released | `release --gpu 4090 --job-id prx-tg-eir-cosine-restart-v7 --status completed` → `released` |
| 6 | Launcher marker rewritten | `/tmp/ecr_launcher_state.json` → `phase: paused` + resume hint |

## State at the pause

- **Step 16,916** — 84 steps short of the 17000 boundary target.
- **Checkpoint:** `experiments/eir-cosine-restart/runs/2026-09-30_0950/checkpoints/checkpoint_interrupt.pt`
  — 2,823,458,912 bytes, mtime 2026-10-02 07:46:40 EDT. Written by the SIGTERM handler, so the
  segment's work is preserved and the arm is resumable from exactly here.
- **Scheduler:** `state.json` gpus.4090 → `status: idle`, `current_job: null`, `consecutive_failures: 0`,
  `queue: []`. Job file `prx-tg-eir-cosine-restart-v7.json` → `status: completed`.
- **GPU:** 11,219 MiB used, 2 % util — that is `image_embed` (6,894 MiB) plus a co-resident
  `ollama llama-server` (3,858 MiB) only. No ECR process remains.
- **No relaunch possible:** both timers are disabled, so the 30-minute launcher tick cannot request a
  new segment.

## Why `completed` and not `failed`

The scheduler CLI accepts only `completed` or `failed` (`release --help`). The stop was deliberate and
is *provable by an artifact* — the interrupt checkpoint written by the graceful handler — which is the
same evidence standard the heartbeat itself uses to justify `completed`. `failed` would assert a
failure that did not occur and would write an unearned failure against this project. The truth
(deliberate stop, 84 steps short of the boundary) is recorded in `provenance.yaml: pause_record`.

## Resume

```bash
systemctl --user enable --now ecr-launcher.timer ecr-heartbeat.timer
```

The launcher's `latest_checkpoint()` resolves `checkpoint_interrupt.pt` and resumes around step 16,916;
`next_job_id()` allocates a fresh `prx-tg-eir-cosine-restart-v8` because v7 is no longer `queued`.

## Two harness notes surfaced by this stop (for the pending cleanup)

1. **The launcher's docstring is stale.** It claims *"a released job's FILE stays 'queued' forever —
   the scheduler only writes state.json."* `gpu_scheduler.py:704-711` in fact rewrites the job file's
   status on release, which is why v1–v6 all read `completed`. The comment drove a resume-design
   assumption (`next_job_id()` reuse) that is not what the code does.
2. **The launcher's `train_alive()` false-positive warning is real, and was reproduced here by hand.**
   A `pgrep -af "train_production.*eir-cosine-restart"` run *after* the trainer died still printed a
   match — the agent's own shell, whose argv contained the pattern. The PID-file + `/proc/<pid>/cmdline`
   guard is the correct check and should not be relaxed.
