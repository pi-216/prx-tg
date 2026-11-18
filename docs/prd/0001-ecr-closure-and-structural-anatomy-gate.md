---
prd: "0001"
title: "Arm ECR Closure & Structural Face-Anatomy Gate"
status: ruled
ruled: "2026-11-18"
intent: "moves:artefacts-published"
---

# PRD-0001 — Arm ECR Closure & Structural Face-Anatomy Gate

## 1. Problem & Context

1. **Arm `eir-cosine-restart` (ECR) stopped at step 16,916 (`2026-10-02`) after the 15k decision point proved both the intervention and the instruments failed:**
   - **Intervention:** Re-heating the cosine learning rate from `1e-6 → 1.56e-4` over steps `10,000 → 16,916` on `eidolon-identity-renderer` (EIR) did not fix the underlying defect. Operator visual inspection of the 10k vs. 15k validation grid (`experiment-configs/eir-cosine-restart/audit/15k_review/RESULTS.md` Addendum) established: *"The images turn from one way of terrible to another way of terrible. They get from smudged half right to high detail total wrong. Face anatomy is absent."*
   - **Instrument blindness:** Pairwise intra-sample Elo (`SNR = 1.33 / 1.39`, `13/30` and `17/30` late-vs-early inversions) cannot rank "smudged half right" against "high detail total wrong" because they fail on orthogonal axes. The automatic `G0` spectral/noise bands also oscillate because the renders contain high-frequency texture energy while lacking coherent facial structure.
2. **What already distinguishes the defect on disk:**
   - The `2026-09-28` photorealism deficit report (`docs/DEFICIT_REPORT_2026-09-28.md`) measured **21.2% face/skin detection attrition** on EIR renders versus **0.0%** on real FFHQ and the FLUX-AE VAE floor — a direct, automatic structural measurement of "face anatomy is absent" that was never wired into a gate.

## 2. Goal & Pre-Registered Gate

Before any new GPU training arm is launched on `prx-tg`:
1. **Close Arm ECR honestly at step 16,916 (`CONCLUDED — PIVOT`)** in `experiment-configs/eir-cosine-restart/provenance.yaml`, `docs/EXPERIMENT_TREE.md`, and `docs/EXPERIMENTS_AND_RESULTS.md`, and draft the corresponding `docs/blog/brief-07-eir-ecr-anatomy-wall.md` progress brief (`moves:artefacts-published`).
2. **Build and calibrate the Structural Face-Anatomy Gate (`scripts/harness/structural_anatomy_gate.py` + unit tests):**
   - Evaluates a checkpoint's validation render directory (`16` canonical validation identities) on three deterministic structural criteria:
     - **S1 — Face/Landmark Detection Attrition:** fraction of validation renders where the face/landmark detector (`AuraFace` / `Sapiens2` seg2 skin+face mask) fails to detect a coherent face (`real FFHQ = 0.0%`, `VAE floor = 0.0%`, `EIR 10k = 21.2%`).
     - **S2 — Anatomical Landmark Plausibility / Symmetry Residual:** inter-ocular / nose / mouth geometric ratio residual against the real FFHQ `[p5, p95]` band (catching "eyes + lips floating in a textured smudge").
     - **S3 — Regression pins preserved:** `z_g` dim0 yaw monotonicity (`Spearman > 0.85`, correct sign) and cross-seed identity consistency (`mean pairwise cosine >= 0.90`).
   - **Calibration / Sensitivity Gate (must pass in `bin/verify` / Ticket 2):** Running `structural_anatomy_gate.py` on existing on-disk renders (`real FFHQ`, `VAE ceiling`, `EIR step 10k`, `ECR step 15k`) must **PASS** `real FFHQ` and `VAE ceiling` (`attrition == 0.0`, anatomy in-band) and **FAIL** both `EIR step 10k` and `ECR step 15k` (`attrition > 0.10` or landmark residual out-of-band).

## 3. Atomic Ticket Breakdown

- **T1 (`Task` — Close Arm ECR at 16,916 steps & write Ledger + Blog Brief 07):**
  - Update `experiment-configs/eir-cosine-restart/provenance.yaml` (`status: concluded`, `verdict: PIVOT`, `tags: [arm/eir-cosine-restart/concluded-PIVOT]`, `blog_brief: docs/blog/brief-07-eir-ecr-anatomy-wall.md`).
  - Record the full ECR 15k/16.9k ledger entry in `docs/EXPERIMENTS_AND_RESULTS.md` and move ECR from `## Active` to `## Concluded` in `docs/EXPERIMENT_TREE.md`.
  - Write `docs/blog/brief-07-eir-ecr-anatomy-wall.md` summarizing the EIR → ECR trajectory, why pairwise Elo and G0 spectral bands missed absent facial anatomy, and why `scripts/check_arm_records.py --allow-missing` passes with zero errors on `eir-cosine-restart`.
- **T2 (`Task` — Build & calibrate `structural_anatomy_gate.py` on Real FFHQ / VAE Floor vs. EIR 10k / ECR 15k):**
  - `Depends on: #<T1>`
  - Implement `scripts/harness/structural_anatomy_gate.py` + unit tests in `tests/test_structural_anatomy_gate.py` (including a deliberate-break test on scrambled/merged facial landmarks).
  - Run the gate on the existing on-disk validation renders (`real FFHQ`, `VAE ceiling`, `EIR 10k`, `ECR 15k`), commit the calibration summary JSON + ledger calibration table, and verify it cleanly separates real/VAE (`PASS`) from EIR/ECR (`FAIL`).
