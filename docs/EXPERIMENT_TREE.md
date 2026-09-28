# Experiment Tree — prx-tg

Living map of all active, planned, and concluded workstreams. Each entry links to the experiment arm directory under `experiments/{slug}/`. Status tags: `[ACTIVE]`, `[CONCLUDED]`, `[TBD]`, `[DISCONTINUED]`.

Last updated: 2026-09-28

---

## Goal (root node)

**prx-tg is the renderer for the product vision of `eidolon`.** The vision belongs to
eidolon; prx-tg is the vehicle that must deliver it. Every `tree_path` in this project
starts here — an arm whose `tree_path[0]` is anything else is off-goal by construction.

**The vision (eidolon):**
1. I can pick a random spot in an identity space — *an* identity space, not necessarily AuraFace.
2. I can move dials or toggle switches that have an impact on that spot in the identity space.
3. I can render an image of that identity.
4. I can move dials or toggle switches that have an impact on pose.

**Deferred sub-products (named here so they are not mistaken for the current work):**
* A **text → point-in-identity-space** helper: "a blonde woman with blue eyes and a round
  face" resolves *somewhere near the valid centroid* of that description in the identity space.
* A **visualizer**, which should/could be text-to-image (ponytail, no makeup, wearing glasses, …).

**Current focus — the hurdle under this goal:** **high-fidelity** image generation, based on
identity-space and pose definition. Fidelity is the operative term and it is a *product*
criterion (is the rendered person a high-quality, convincing image of that identity?), not a
recogniser score.

**The carriers are hypotheses, not settled decisions.** The current concept is that identity
space is *somehow connected to the LDA* (64-d AuraFace-LDA today) and that pose — possibly
including **expression** — is *somehow connected to `z_g`* (50-d today). Either is the *current*
proposed encoding. Neither is a requirement, and a result that only holds for this pair of
carriers has not been shown to hold for the vision.

> **Why this section exists.** Arm EIR wrote its return edge (provenance `serves:` /
> `tree_path:` / `return_question:`) but wrote the goal node as *"render a chosen identity at a
> chosen pose"* — **fidelity absent** — and asked, as its return question, whether a real
> *recogniser* agrees. That is how an arm passes its mechanism gate while sitting nowhere near
> the product: no measurement was ever pointed at the missing node, so render quality could only
> ever be discovered late, as an anomaly, and generate an endless tail of explanatory probes.
> The root now carries the term that was missing.

---

## Active

* **dip-conv-head** (`experiments/dip-conv-head/`) `[CONCLUDED — GO-with-caveat]`
  * DiP-style conv U-Net output head vs linear 3x3. 10k steps, resumed from 5k.
  * **Criteria a/b/d PASS**: LPIPS 0.7251 (≤0.746), loss ~0.011 (≤0.0135),
    blind 10k-vs-5k win-rate 0.846 (CI LB 0.5776 > 0.5), calibration 1.0. FaceConf 0.488→0.707.
  * **Criterion (c) G0 NOT_VALIDATED → strike 1/3** (g0a/g0d p50/p95 out-of-band
    vs frozen real-FFHQ band; no 5k G0 baseline). Improved but not yet photorealistic;
    champion candidate vs 5k baseline, arm still `active` in registry.

* **Arm EIR — `eidolon-identity-renderer`** (`experiment-configs/eidolon-identity-renderer/`) `[CONCLUDED — PIVOT]` (2026-09-27; run 2026-09-25_0734 complete at 10k; verdict + scorecard in the ledger; entry still sits under the Active heading — move it)
  * Branch `arm/eidolon-identity-renderer`. From-scratch **FLUX-VAE latent** DiT (16ch, 128²,
    patch 2 — P1-shaped) carrying the Eidolon adapter: 64-d AuraFace-LDA identity via adaLN
    + z_g geometry via cross-attention, per-dim basis. **No text**, no REPA (hegre ships no
    `dinov3_patches`), no warm-start, no PP stage, no weight donor.
  * Question: does the two-source LDA stream (FFHQ per-image vectors + hegre persona
    centroids) produce a **generalizable** identity mapping, or per-image lookup?
  * **Gate (frozen before the first run):** held-out identity retrieval, rank-based R@1 with a
    bootstrap CI LB > 0.5 on all three probes — FFHQ unseen-identity fidelity (index 6,996,
    ref ~1.0), hegre unseen-persona generalization (index 32, ref **0.9504**), hegre
    persistence (index 313, ref **0.8180**). The ceiling is **index-size dependent**, so each
    number is read against the ceiling measured at its own index size.
  * **Holdout locked:** `experiment-configs/eidolon-identity-renderer/holdout.json`, seed `20260925` —
    6,996 FFHQ identities (sha256 `2967fafac71f2a75`) + 32 hegre personas (sha256
    `705936d1c0f3b805`), identities whose vectors are never fed in training, so the gate
    measures generalization rather than recall.
  * Instrument, measured before the arm exists: `research/results/eidolon-identity-instrument/`.
  * Pre-registration: `experiment-configs/eidolon-identity-renderer/provenance.yaml`;
    `config.yaml` there is **PROVISIONAL** (dropout profile + architecture freeze outstanding).

* **[ACTIVE] Phase 1b Research Loop** — photorealism gates + blind review + avenue registry
  (`.hermes/plans/2026-08-30_photorealism-gates-blind-review.md`). M1 shipped 2026-08-30;
  M2 (calibration) + M3 (tick state machine) next. No training arms active.

---

## Concluded

### Geometry Adapter for Pretrained SD 1.5 Series (Arms P, P2, P3, P4) — `[CONCLUDED — KILL]`

Four-arm series testing whether a lightweight geometry adapter (~630K params) can steer face yaw in a pretrained SD 1.5 UNet by injecting z_g geometry tokens alongside CLIP text embeddings. All four arms failed — zero controlled yaw across 40K+ cumulative steps.

* **Arm P — `geometry-adapter-sd15`** (`experiments/geometry-adapter-sd15/`) `[CONCLUDED — FAIL]`
  * Vanilla SD1.5, frozen attention. 10K steps. Adapter learns (differential output) but produces artifacts, not yaw. Bug 1 (basis leak) contaminated training. Code review found 8 bugs.
* **Arm P2 — `geo-adapter-bugfix-restart`** (`experiments/geo-adapter-bugfix-restart/`) `[CONCLUDED — FAIL]`
  * Frozen attention + bug fixes + pose-stripped captions. 10K steps. Same artifacts, no yaw control.
* **Arm P3 — `geo-adapter-absreal`** (`experiments/geo-adapter-absreal/`) `[CONCLUDED — KILL]`
  * Frozen attention, AbsoluteReality v1.8.1. Killed at step 2340 (budget-justified: P/P2 exhausted 10K identically). Garbled output.
* **Arm P4 — `geo-adapter-lora-crossattn`** (`experiments/geo-adapter-lora-crossattn/`) `[CONCLUDED — KILL]`
  * LoRA-UNFROZEN cross-attention (rank-8 K,V, ~5M params). 10K steps on vanilla SD1.5. Loss diverged +48% from step 2K→10K; output collapsed to pure noise. Falsified the "frozen attention" root cause — unfreezing attention made it *worse*. ⚠️ Cross-model eval confound (trained vanilla, 4/6 sweeps on AbsoluteReality).

**Refined root cause (supersedes earlier "frozen attention can't attend"):** The ε-prediction denoising loss provides no gradient that rewards geometry control — the frozen text/CLIP path already denoises the face, so the adapter is redundant to the loss. Trainable weights either stay inert (frozen attention: P/P2/P3) or random-walk into destruction (LoRA: P4). Not data-dependent, not base-model-quality-dependent, not attention-freeze-dependent. **Thread-level KILL. Project → PIVOT to from-scratch DiT (Arm O).**


### Ablation Series (Arms A–D) — `[CONCLUDED — GO]`

Foundational 4-arm sweep isolating TREAD, Muon, and REPA contributions on 7k FFHQ subset. All 5,000 steps on same quad-GPU Vast.ai node.

* **Arm A — `minimal-baseline`**: AdamW only, no TREAD, no REPA. Baseline reference.
* **Arm B — `tread-adamw`**: TREAD + AdamW. **Reconstruction collapsed at step 3500** — TREAD's token sparsity incompatible with AdamW's per-parameter adaptive rates.
* **Arm C — `tread-muon`**: TREAD + Muon. **Recovered stability.** Muon's orthogonalized updates prevent learning-rate inflation under sparse gradients. Best text controllability (Text Manip delta 0.546).
* **Arm D — `full-stack-baseline`**: TREAD + Muon + REPA. **Best overall quality** — Recon LPIPS 0.927, Text LPIPS 0.922. REPA loss scale not comparable to A/B/C.
  * 📄 Detailed writeup: `docs/ablation_a_through_d/ablation_study.md`
  * ⚠️ Note: Arm D directories (`full-stack-baseline`, `minimal-baseline`, `tread-adamw`, `tread-muon`) not yet created under `experiments/` — artifacts live in legacy timestamp-named dirs.

### Scaling & Optimization Arms (E–J)

* **Arm E — `seg-weight-spatial`** `[CONCLUDED — FAIL (CONFOUNDED)]`
  * Seg weight map (face 2×, bg 0.5×) vs Arm D. Confounded variables: effective batch size 128 vs 256, REPA warmdown not matched.
  * 📄 Compliance review: `docs/part-e-compliance-review.md`

* **Arm F — `seg-weight-clean-ablation`** `[CONCLUDED — PAUSED]`
  * Clean re-run of Arm E with batch size and REPA warmdown matched. ⏸️ Paused.

* **Arm G — `asym-flow-ablation`** `[CONCLUDED — GO]`
  * Asymmetric Flow Matching (rank 8). Recon LPIPS 0.938, Text-only LPIPS 0.914. Convergence acceleration verified. No quality regression vs baseline.

* **Arm H — `shared-adaln-lora`** `[CONCLUDED — KILL]`
  * Shared adaLN + per-block LoRA (rank 8). **Failed to converge at 5k steps.** Recon LPIPS 0.982, Text-only 1.006. Visually noise/dithering. Per-block modulation is load-bearing.
  * 💀 Tombstone: `docs/DISCONTINUATION_NOTICE.md`

* **Arm I — `fp8-native`** (`experiments/arm_i_fp8/`) `[CONCLUDED — GO]`
  * Native FP8 via torchao. **2×+ speedup (26h vs 56h BF16)** with negligible quality loss (Recon LPIPS 0.906 vs 0.900 BF16). Text controllability actually improved (Text Manip Diff 0.504 vs 0.485).

* **Arm J — `faces70k-fp8`** (`experiments/faces70k-fp8/`) `[CONCLUDED — INCOMPLETE]`
  * Scaled FP8 to 70k portraits (40k steps w/ REPA linear decay). 🔄 Stopped at 34k steps.

### Quality Metric Arms (K–M) — `[CONCLUDED — GO]`

3-arm series testing DINO patch token contribution to generation quality. All use quality metrics (CLIP + aesthetic + DWPose face confidence) alongside LPIPS.

* **Arm K — `spatial-window-baseline`** (`experiments/spatial-window-baseline/`) `[CONCLUDED]`
  * Full DINO patches. Best Recon LPIPS (0.937) but worst generation quality metrics.

* **Arm L — `spatial-window-2`** (`experiments/spatial-window-2/`) `[CONCLUDED — GO]`
  * DINO patch spatial window (r=2) via 2×2 AvgPool2d. **Degraded reconstruction (0.999 vs 0.937) without matching the quality of full removal.** 155× FLOP reduction doesn't justify quality trade-off.

* **Arm M — `no-dino-patch-ablation`** (`experiments/no-dino-patch-ablation/`) `[CONCLUDED — GO]`
  * DINO patch cross-attention disabled entirely. **Best generation quality on 4 of 5 metrics** (CLIP 0.204, Aesthetic 4.44, Face Conf 0.790, Text-only LPIPS 0.907).
  * 🏆 Key finding: DINO patches buy reconstruction fidelity (+0.043 LPIPS) at the cost of CLIP score, aesthetics, face confidence, and text-following. DINO CLS token alone is sufficient style conditioning.

### Eidolon Conditioning (branch `exp/eidolon-conditioning`)

* **Eidolon Conditioning baseline** (`experiments/eidolon-conditioning/`) `[CONCLUDED — GO]`
  * Neutral DiT + `EidolonAdapter` (AuraFace-LDA identity via adaLN + z_g geometry via cross-attention). T5 text + DINO patches dropped. Renders all faces frontal (root cause identified: z_g tokens are permutation-symmetric).

* **Arm O — `zg-token-basis-cfg-guard`** (`experiments/zg-token-basis-cfg-guard/`) `[CONCLUDED — PASS]` ✅
  * CFG-guarded per-dim basis. **PASS on pre-registered gate:** monotonic dim0→yaw at steps 3000 AND 5000, zero collapse through 5,000 steps, loss 0.0099, recon LPIPS 0.819. First arm to hold controlled yaw without collapse — validates the Arm N collapse diagnosis. Re-scoring under the G-gate suite pending (M2/M3).
  * Release artifact: `release/prx_tg_armo_cfgguard_step5000_bf16.safetensors`.

* **Hegre Geometry** (`experiments/hegre-geometry/`) `[CONCLUDED — GO]`
  * Added hegre corpus data (w1.0) + geometry-only CFG dropout (p=0.30). Shared MLP (no per-dim basis). Still renders all faces frontal. Proved data augmentation alone doesn't fix the architectural bug.

* **Arm N — `zg-token-basis`** (`experiments/zg-token-basis/`) `[CONCLUDED — KILL]`
  * **Per-dim learned embedding** so cross-attention can bind z_g dim0 → head yaw. ✅ **Proved yaw binding**: dim0 controlled head yaw at steps 2000/2500 (first time ever). ❌ **Mode collapse at step 3500**: output darkness, no gradient/loss spike.
  * Root cause: basis added on all steps (~30% CFG-dropped), making it a pure noise injection (~1500 events) that accumulated and pushed output into clipping.
  * 💀 Tombstone: `docs/DISCONTINUATION_NOTICE.md`

---

## TBD

* **[TBD] Single-Stream DiT** — Remove cross-attention, concatenate T5 text + DINO CLS into self-attention sequence. Validated by Ideogram 4 at 9.3B scale. Good arm after eidolon branch stabilizes.
* **[TBD] iREPA Upgrade** — Replace `nn.Linear` REPA projection with Conv2d + spatial normalization (per Singh et al.). ~4 lines each, proven FID gains.
* **[TBD] Resolution-Aware Logit-Normal Schedule** — Adjust mu per bucket resolution (per Ideogram 4). 3-line change in `train.py`.
* **[TBD] DINO-Patch-Free Architecture** — Redesign conditioning around DINO CLS + DWPose + T5 text only, dropping DINO patches permanently per Arms K–M findings.
* **[TBD] Structured Face Attribute Captions** — Convert caption pipeline to JSON attribute templates for denser supervision (per Ideogram 4).
* **[TBD] SpectraReward RL Post-Training** — Freeze MLLM reward model, score rollout groups, apply advantage-weighted matching for face attribute binding.
* **[TBD] QK-RMSNorm** — Add RMSNorm to Q and K separately for attention stability (per Ideogram 4 / Llama 3).
* **[TBD] Sapiens2 0.8B Backbone** — Scale up backbone after winning config found (from breast-tissue-net notes).
