# Project Status — prx-tg

**Last updated:** 2026-11-18
**Governance:** `pi-216/prx-tg` — Board-driven under `founders-harness` (`AGENTS.md`, `bin/verify`). This file is a read-only orientation summary, never a work queue.

## Goal & Downstream Consumers

- **Renderer for `eidolon`:** Turn a point in the disentangled identity space (`64-d AuraFace-LDA`) plus a geometry/pose vector (`50-d z_g`) into a high-fidelity portrait on 24GB consumer hardware (`RTX 4090`) where the same identity vector renders consistently across seeds and poses.
- **Public Research Output (`lawrenz.com`):** Every concluded milestone produces a committed blog brief in `docs/blog/` and a published article on `lawrenz.com` (`moves:artefacts-published`).

## Current Empirical State

- **Mechanism tier (`eidolon-identity-renderer`, Arm EIR, 10k steps) — `CONCLUDED — PIVOT`:**
  - `z_g` dim0 controls yaw monotonically with the correct sign (`|r| = 0.942`, rank `1/50`).
  - Cross-render identity consistency passes (`render-to-render R@1 = 1.0000`, 6-seed cosine `0.9943`).
  - Defect: **facial anatomy is absent/distorted** ("coarse texture over a distorted armature"; `21.2%` face/skin detection attrition vs. `0.0%` on real FFHQ).
- **Schedule re-heat (`eir-cosine-restart`, Arm ECR, paused at step 16,916) — `PIVOT` (PRD-0001 T1 closing):**
  - Re-heating the cosine LR from `1e-6 → 1.56e-4` over steps `10,000 → 16,916` shifted renders from *"smudged half right"* to *"high detail total wrong — face anatomy is absent"*.
  - Both pairwise Elo (`SNR = 1.33 / 1.39`) and `G0` spectral/noise bands were proven blind to absent facial structure.
- **Active Study (`PRD-0001`):** Close Arm ECR at step 16,916 (`brief-07`) and calibrate the **Structural Face-Anatomy Gate** before `research-agent` proposes the next renderer architecture/training arm.
