# Brief 07 — the anatomy wall: how the renders got detailed and stayed wrong

**Step:** `eir-cosine-restart` (Arm ECR) — the learning-rate re-heat continuation of `eidolon-identity-renderer` (Arm EIR)
**Date:** arm run 2026-09-30 → deliberately stopped 2026-10-02 at step 16,916; closed 2026-11-18
**Verdict:** `CONCLUDED — PIVOT` — and, just as important, a **measured instrument failure**: both automatic adjudicators this project trusted are now proven blind to the defect
**Source:** `experiment-configs/eir-cosine-restart/audit/15k_review/RESULTS.md` (+ Addendum), `audit/2026-10-02_pause/PAUSE.md`, `docs/DEFICIT_REPORT_2026-09-28.md`, `docs/EXPERIMENTS_AND_RESULTS.md` (ECR closure entry)
**Suggested tags:** `research`, `prx-tg`, `diffusion`, `evaluation`, `machine-learning`
**Suggested thumbnail:** **an actual face** — the step-15,000 render of validation identity 0 beside the same identity at step 10,000 (paths in the Visuals table in `README.md`). One pair shows the whole post: sharper, richer, and anatomically gone.

---

## Where this sits

prx-tg is the renderer for the group's *"Getting a grip on human identity"* work: pick a point in an identity space, move the pose dials, render an image of that identity. The previous prx-tg post ended with Arm EIR concluded **PIVOT**: the renderer could bind an identity and obey a pose vector — the mechanism tier passed — but the pictures themselves were the open problem.

This post closes the direct follow-up experiment and reports the more uncomfortable result of the two. The follow-up tested one clean hypothesis — that EIR's quality plateau was the **learning-rate schedule** running out. It wasn't. And the instrument we built to answer "is this picture better?" turned out to be unable to see the actual defect. The post is the EIR → ECR arc, the anatomy wall, and the measurement that finally points at it.

## Plain-English summary (write this first, before any metric)

We thought our face renderer had stalled because its training recipe had run its course: the learning rate — how big the model's correction steps are — had decayed to its floor. So we ran the obvious experiment: take the model exactly where it stopped, turn the learning rate back up, and train another 6,916 steps. One variable, nothing else changed.

Two things came out of it. First, the model did not get better in any way we could measure. Second — and this is the one that stung — when we put the before and after renders side by side, they weren't two points on one quality scale. They were *different kinds of wrong*. The earlier renders were blurry but roughly right; the new ones are richly detailed and anatomically impossible. Eyes with real texture, attached to a face with no nose, no cheekbones, no jaw.

Then we checked the machines we use to judge quality. A panel of human raters ranking render pairs could not separate "before" from "after" — 13 and 17 preference reversals out of 30 head-to-heads. Our automatic photo-quality gates — the spectral and noise fingerprints that separate real faces from fakes — bounced in and out of the real-face band with no trend. Neither instrument can see whether a face has facial anatomy. One measures *preference*; the other measures *texture energy*. Anatomy is neither.

The defect has a name now: **facial anatomy absent, texture present.** And there is a measurement that sees it directly — face detectors that simply fail to find a face in 21.2% of our renders, versus 0.0% of real photographs and 0.0% of what the image compressor alone can explain. Making that detector a hard gate, before any new training run, is the whole next step.

---

## Block 1 — What the renderer had already proven it could do

- **Hook** — Before the bad news: the mechanism works, and the evidence for it is strong.
- **Setup** — Arm EIR trained a from-scratch latent diffusion transformer conditioned on two vectors: a 64-d identity vector and a 50-d pose vector.
- **Evidence** — Identity *consistency*: the same identity point rendered across six different noise seeds agrees at mean cosine 0.9943, and a render retrieves its own identity first-of-16 at R@1 = 1.00. Pose *control*: the yaw dial (`z_g` dim 0) moves the rendered head monotonically with the correct sign at 10,000 steps — a face turned is a face turned.
- **Turn** — Every one of those wins is a *relationship between renders*. None of them is a claim that a render looks like a face.
- **Meaning** — Mechanism tier: passed. Fidelity tier: never measured. That gap is exactly what the next experiment had to walk into.
- **Handoff** — So why did the pictures stop improving?

## Block 2 — One variable: turn the learning rate back up

- **Hook** — The plateau had an obvious suspect: a learning-rate schedule that had decayed to its floor.
- **Setup** — Arm ECR resumed EIR's step-10,000 weights *with optimizer state*, raised the training horizon 10,000 → 20,000, and let the cosine schedule re-heat the learning rate from 1e-6 back to 1.56e-4. Config diff against the parent: exactly one substantive change. Everything else — data, identity holdout, model, optimizer, sampling — byte-identical, so the before/after renders stay comparable.
- **Evidence** — 500-step render grid from 10,000 to 20,000; two identity pools (validation indices 0 and 4) with a blind pairwise human ranking pre-registered against a null of about 1 in 350,000.
- **Turn** — The run never reached its registered finish line: the operator stopped it deliberately at step 16,916 to clean the research harness first — a pause, not a failure.
- **Meaning** — What survives is the pre-registered decision point at 15,000 steps and the evidence it produced.
- **Handoff** — Which is where the flat line begins.

## Block 3 — The flat line, and the human read that explained it

- **Hook** — Two identity pools, every pair of checkpoints judged by a human, 110 games each. Did the curve rise?
- **Setup** — Blind intra-sample Elo over checkpoints 10.0k…15.0k, pools exhausted, every image inside its trust radius.
- **Evidence** — Ordering-vs-training-step rank correlation +0.127 and +0.173 (near zero); late half above early half in 0 of 2 pools on an even split, with 13/30 and 17/30 preference reversals; best-minus-worst render gaps of 332 and 349 Elo at z = 1.85 and 1.99 — inside noise. The automatic quality number (recon LPIPS, n = 16 per point) oscillated 0.79–0.85 with no direction.
- **Turn** — Then the operator looked at the renders themselves, and described what no metric had said: *"The images turn from one way of terrible to another way of terrible. They get from smudged half right to high detail total wrong. Face anatomy is absent."*
- **Meaning** — The endpoints of the ranking task weren't two levels of quality on one scale. They were different failure modes on different axes — and a single preference axis cannot rank those. The flat line was not "no change"; it was "changed kind."
- **Handoff** — Which raises the sharper question: why did *both* instruments miss it?

## Block 4 — Why the human Elo could not see it

- **Hook** — The Elo panel was this project's first instrument that could answer "is this picture better?" It turned out to be answering a different question.
- **Setup** — Check the instrument's own vitals before blaming it: the same protocol, run on EIR's earlier trajectory, resolved a real quality trend — 6 of 6 pools with positive rank correlation, 4 of 6 separating perfectly (p ≈ 3.7e-9), the step-1000 render ranked last in all six.
- **Evidence** — So the Elo is alive. On ECR it flatlined because the comparands were incomparable: reversals near coin-flip, the two pools disagreeing about their own winner (11.0k vs 15.0k), signal-to-noise 1.33 and 1.39 against a 5.0 "good enough" line.
- **Turn** — This is the distinction the project insists on: "the instrument could not resolve" is **not** "there is no difference." Collapsing the two is how flat lines get quietly misread as null results.
- **Meaning** — A preference Elo is the wrong instrument for a structural defect. You cannot vote your way out of a missing nose.
- **Handoff** — And the automatic gates were not any better.

## Block 5 — Why the spectral gates could not see it either

- **Hook** — We have mask-free photo-quality gates: noise floor, spectral slope, texture energy, local contrast, all calibrated against real FFHQ photographs. Didn't they scream?
- **Setup** — Re-measured (no GPU) across every 500-step checkpoint of the continuation, 16 identities each, against the frozen real-FFHQ band.
- **Evidence** — The noise-floor gate sits *above* the real band in 6 of 10 continuation steps — the renders carry **more** high-frequency energy than real faces — and local contrast is out of band in 7 of 10. But nothing trends with training, and the texture gate oscillates in and out.
- **Turn** — Read carefully, that out-of-band high noise floor *is* the quantitative shadow of "high detail." The gates measured the texture explosion faithfully — and are constitutionally blind to whether the texture is organized into a face.
- **Meaning** — Spectral statistics describe the distribution of light. Anatomy is arrangement. The old gates will never see the defect, no matter how many checkpoints they sweep.
- **Handoff** — So what does see it?

## Block 6 — The number that sees the defect: 21.2% vs 0.0%

- **Hook** — Somewhere in the harness sits a measurement that asks, in effect: "is there even a face in this?"
- **Setup** — Detection attrition: run the face/skin detector over every panel; count how many renders yield a usable face.
- **Evidence** — Real photographs: 0 of 150 undetectable (0.0%). The image-compressor floor — the *best any model in this architecture can do* — also 0.0%. Our renders: 34 of 160 undetectable, **21.2%**, concentrated early but persisting at every later checkpoint.
- **Turn** — This number was measured on 2026-09-28, in the deficit report that named the problem — and was never wired into a gate. Every verdict since was taken without it.
- **Meaning** — The next rule for this project, now written down: **no new training arm without a pre-registered structural gate** — detection attrition plus landmark-plausibility geometry (eyes too far apart, nose floating) with real-face bands, plus regression pins so the yaw dial and identity consistency that already work are never quietly lost.
- **Handoff** — The wall has a door. Next post: building it.

---

## Caveats to carry into the post — read these before drafting

- **The pre-registered statistical gate is INCONCLUSIVE, not failed.** The run stopped at 16,916 of a registered 20,000; the 21-checkpoint ranking grid never existed, so the registered null was never computed. The 15k read used 11-item pools — 32,000× weaker null — and is directional only. Say "the instrument could not adjudicate," never "the hypothesis was tested and rejected."
- **The PIVOT verdict rests on the structural record** (the operator's direct render read + the measured blindness of both instruments), and the ledger entry says so explicitly. Do not present it as an Elo result.
- **No photorealism claim survives this post.** The mechanism wins in Block 1 (identity consistency, yaw control) are render-to-render properties; the product criterion — a convincing face — is unmet and named.
- **n with every number:** 110 games per pool, 16 identities per G0 step, 160 recon panels for attrition, 150 panels per real/floor leg. Sub-noise moves (the ±23/−6 Elo means against a ~250 floor) are stated as noise.
- **The 21.2% attrition figure is measured on the parent arm (EIR) renders** — the deficit report predates the ECR stop. The continuation renders show the same defect by direct inspection; the detector number quoted is the one actually on disk. Don't silently re-attribute it to ECR.
- **Real-FFHQ attribution rules apply** to any figure pairing a render with a real photograph (CC BY-NC-SA 4.0 by NVIDIA, full image IDs, per the compliance section in `README.md`). Never name the proprietary multi-shoot dataset in public material.
- **Lead with the render pair, not the Elo table.** The one figure that carries this post is the same identity at 10k and 15k.

## Artifacts

- Provenance: `experiment-configs/eir-cosine-restart/provenance.yaml` (verdict PIVOT, 5 conclusions, return answer PIVOTED)
- 15k decision point + Addendum: `experiment-configs/eir-cosine-restart/audit/15k_review/RESULTS.md` (payloads `project_11_i00.json`, `project_12_i04.json`, `summary.json`, `g0_trajectory_full.json`)
- Pause record: `experiment-configs/eir-cosine-restart/audit/2026-10-02_pause/PAUSE.md`
- Ledger: `docs/EXPERIMENTS_AND_RESULTS.md` — "eir-cosine-restart closure at 16,916"
- Deficit report (attrition table): `docs/DEFICIT_REPORT_2026-09-28.md`
- Run (NAS): `experiments/eir-cosine-restart/runs/2026-09-30_0950/`
