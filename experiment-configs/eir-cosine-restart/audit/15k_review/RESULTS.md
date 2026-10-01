# Arm ECR — 15k review (pre-registered decision point)

**Date:** 2026-10-01 · **Arm:** `eir-cosine-restart` (ECR), mode `confirmatory`
**Run:** `experiments/eir-cosine-restart/runs/2026-09-30_0950/`
**Grid:** 10.0k, 10.5k … 15.0k (500-step spacing) · **Checkpoint under review:** `checkpoint_step0015000.pt`
**Instrument:** elo-picker (blind intra-sample Elo), one pool per validation index
**Raw payloads:** `project_11_i00.json`, `project_12_i04.json`, `summary.json` (this directory)

---

## What this read is, and what it is not

This is the arm's declared decision point (README: *"the pre-registered decision point is 15k"*), run
half-way through the 10,000-step continuation. At 15k only **11 of the 21 pre-registered items exist**.

**The pre-registered PASS/FAIL test is NOT computable here.** It partitions *late = 15.0k–20.0k*
(11 items, of which we have **one**) against *early = 10.0k–14.5k* (10 items, of which we have all),
at a null of **1/C(21,11) = 2.8e-6**. The analogous read on 11 items has a null of **1/11 = 0.091** —
**32,000× weaker**. Nothing in this document is gate evidence. It is a directional mid-run read of the
same instrument on the same axis.

## Pool state — the instrument resolved, and both pools are exhausted

| | `ecr-traj-i00` (project 11) | `ecr-traj-i04` (project 12) |
|---|---|---|
| Images | 11 | 11 |
| Comparisons | **exhausted** — 110 games = 2 × C(11,2) | **exhausted** — 110 games |
| Trusted | 11/11 (RD 120.9–128.3, all ≤ 150) | 11/11 (RD 121.1–131.5, all ≤ 150) |
| Phase | 3 (distance) | 3 (distance) |
| SNR | 1.334 | 1.395 |
| Reliability | 64.0% | 66.0% |
| Spread | 332 Elo | 349 Elo |
| Per-claim floor (2·RD) | 249 Elo | 250 Elo |

Neither pool can retreat behind *"the instrument could not resolve"*: every pair was judged and every
image reached trust. But the resolution is **coarse** — the per-claim floor (≈250 Elo) is most of the
field width (≈330–350 Elo), so no single adjacent pair in these pools is a decisive claim. What carries
power is the **aggregate ordering**, which is what is reported below.

SNR sits at 1.33/1.39, far below the SNR ≥ 5 "good-enough for training labels" line. Per the arm's own
pre-registration this is **not a gate input** (SNR ≥ 5 is structurally unreachable for a within-run
field this wide — it would need ~130 games/image against a maximum of 10 here), but it does mean the
aggregate below is *weak* evidence, not strong evidence.

## Ordering by Elo

**`i00` (project 11)** — Spearman(checkpoint, Elo) = **+0.127**

| rank | checkpoint | Elo | RD |
|---|---|---|---|
| 1 | 11000 | 8142 | 126.0 |
| 2 | 12000 | 8126 | 124.6 |
| 3 | 13500 | 8079 | 123.9 |
| 4 | **15000** | 8047 | 120.9 |
| 5 | 10000 | 7999 | 128.3 |
| 6 | 14000 | 7985 | 122.8 |
| 7 | 14500 | 7984 | 121.4 |
| 8 | 13000 | 7969 | 123.5 |
| 9 | 12500 | 7969 | 127.4 |
| 10 | 11500 | 7889 | 123.6 |
| 11 | 10500 | 7810 | 127.4 |

**`i04` (project 12)** — Spearman(checkpoint, Elo) = **+0.173**

| rank | checkpoint | Elo | RD |
|---|---|---|---|
| 1 | **15000** | 8174 | 122.9 |
| 2 | 14000 | 8142 | 126.8 |
| 3 | 11500 | 8063 | 127.1 |
| 4 | 12500 | 8047 | 122.4 |
| 5 | 12000 | 8016 | 123.5 |
| 6 | 10500 | 8016 | 131.5 |
| 7 | 10000 | 7968 | 125.1 |
| 8 | 13500 | 7953 | 121.1 |
| 9 | 11000 | 7906 | 126.8 |
| 10 | 14500 | 7889 | 123.2 |
| 11 | 13000 | 7826 | 124.2 |

## Split tests

| Test | `i00` | `i04` | Pools perfect |
|---|---|---|---|
| Pre-registered **form** (late = 15.0k vs early = 10.0k–14.5k) | 15.0k ranks **4/11** — not perfect | 15.0k ranks **1/11** — PERFECT | **1/2** (null 1/11 = 0.091) |
| Equal halves (early 10.0k–12.5k vs late 13.0k–15.0k) | max(early) 8142 > min(late) 7969 — **not perfect**, 13/30 inversions | max(early) 8063 > min(late) 7826 — **not perfect**, 17/30 inversions | **0/2** (null 1/462 = 0.00216) |

Late-vs-early mean Elo: i00 **+23** Elo (7989 → 8013), i04 **−6** Elo (8003 → 7997). Both deltas are an
order of magnitude below the 250-Elo per-claim floor — they are noise, not a measured gain.

Top-vs-bottom: i00 gap 332 Elo, z = 1.85; i04 gap 349 Elo, z = 1.99. Neither extreme pair is separable.

## Cross-pool agreement (the robustness check)

The two pools **disagree about the order**, which is itself the finding:

- `i00` puts its top at **11.0k** and 15.0k 4th; `i04` puts **15.0k** 1st. Only 15.0k and 14.0k sit in
  the top three of both pools.
- Spearman is positive in both, but only just: **+0.127 / +0.173**.
- The bottom is unstable too: 10.5k is last in `i00`, 13.0k last in `i04`.

## Automatic (secondary) trajectory — reported, not gating

Recon mean LPIPS over the same grid (16 samples each): 0.829, 0.809, 0.833, 0.797, 0.809, 0.824, 0.825,
0.794, 0.850, 0.821 (10.5k → 15k). Oscillating in a 0.79–0.85 band with **no monotone movement**.

## Reading

**Directional: no resolvable gain.** Over the first 5,000 steps of the re-heated LR schedule the render
trajectory is flat on both identity pools — Spearman barely positive, late-vs-early within noise, and
0/2 pools showing the late half above the early half on an equal split. The lone "PERFECT" (i04, 15.0k
rank 1) comes from a single item clearing a 1/11 null and is **not** evidence at the gate's standard;
it is not even directionally corroborated, since i00 ranks the same checkpoint 4th.

This direction matters for the arm's `falsified_if`, whose second leg — *"the automatic trajectory at
the same 500-step grid also flattens"* — is currently **met**. If it survives to 20k, the arm resolves
**PIVOTED**: the LR schedule was not the brake.

**But the confirmatory verdict is not reachable at 15k**, and must not be recorded from this read. The
pre-registered gate requires the full 21-item pools at exhaustion. On the current evidence the honest
statement is:

> At the 15k decision point the re-heated schedule shows **no resolvable improvement** in blind
> intra-sample render quality on either identity pool. The pre-registered split test remains
> **INCOMPUTABLE** until the 20k items exist; this is a directional read, not a verdict.

Two things would change it before 20k: a verdict requires the run to reach 20k (it is currently
stopped at step 15,006), and the pools must be rebuilt over the full 21-item grid so the real
1/C(21,11) null applies.

---

# Addendum — the human read, and what the automatic replacement sees (2026-10-01)

## The product criterion, read directly

The arm's primary is *"human, blind, product criterion"*; the operator's read of the 10k→15k
renders is the product criterion being applied, and it is a first-class datum:

> *"The images turn from one way of terrible to another way of terrible. They get from smudged
> half right to high detail total wrong. Face anatomy is absent."*

Inspecting the renders directly (validation indices 0, 8, 16 at 10.0k and 15.0k) corroborates it,
and it is **index-dependent and catastrophic**:

- **index 0 @ 15.0k** — no coherent face at all: the features are merged into a rough, fibrous
  mass; the eye sockets read as hollows, the nose as a textured smudge.
- **index 8 / 16 @ 15.0k** — recognisable as a face, but "high detail, anatomically wrong": real
  structure survives only in the eyes and (faintly) the lips and teeth; the nose, cheekbones,
  jawline and ears are textured smudges with no underlying form.
- **index 0 @ 10.0k (EIR origin)** — a different failure mode: coarse, sculpted, high-frequency
  texture over a distorted armature.

This is the defect Arm EIR was closed on: **structure absent, texture present.** It never left.

## Why the Elo could not see it

The Elo ranks on a single preference axis. When the two endpoints are *different kinds of wrong*
rather than points on one quality scale, the operator's preference is unstable — and that is
exactly what the data shows:

- 13/30 and 17/30 late-vs-early inversions;
- the two pools disagree on their own top (i00 → 11.0k, i04 → 15.0k);
- top-vs-bottom z = 1.85 / 1.99 — **not separable**, i.e. the ordering as a whole is inside the
  noise;
- SNR 1.33 / 1.39 against the 5.0 "good enough" line, reliability 64% / 66%.

So the flat Spearman is **two facts at once**, and they must not be collapsed: the trajectory does
not rise measurably, *and* the instrument is too weak to have resolved it if it had. This is the
"the instrument could not resolve" case, not a null result — and it is compounded by the axis
mismatch above.

## The automatic replacement is blind too (measured, no GPU)

Reusing the frozen primitives (`run_gates.measure_image` → `calibrate_gates`), mask-free gates over
all 16 indices × the 11 checkpoints. Bands are the real-FFHQ [p5, p95] from
`gates_calibration.json`; **bold/fail** marks out-of-band:

| step | n | `g0a_noise_floor_full` | `g0b_spectral_slope` | `g0c_texture_energy_full` | `g0d_contrast_p50` |
|---|---|---|---|---|---|
| band | | [-1.5360, -1.2225] | [-1.6085, -1.0926] | [0.0029, 0.0165] | [0.0460, 0.1136] |
| 10000 | 16 | -1.2837 | -1.2901 | 0.0109 | 0.0989 |
| 10500 | 16 | -1.2324 | -1.4478 | 0.0138 | 0.1177 ❌ |
| 11000 | 16 | -1.2386 | -1.5832 | 0.0101 | 0.1236 ❌ |
| 11500 | 16 | **-1.1762 ❌** | -1.3102 | 0.0154 | 0.1162 ❌ |
| 12000 | 16 | -1.3156 | -1.5008 | 0.0072 | 0.1172 ❌ |
| 12500 | 16 | **-1.1631 ❌** | -1.4165 | **0.0169 ❌** | 0.1353 ❌ |
| 13000 | 16 | **-1.1612 ❌** | -1.3518 | **0.0196 ❌** | 0.1419 ❌ |
| 13500 | 16 | **-1.2025 ❌** | -1.2615 | 0.0138 | 0.1089 |
| 14000 | 16 | -1.2586 | -1.2868 | 0.0102 | 0.1039 |
| 14500 | 16 | **-1.1276 ❌** | **-1.0043 ❌** | **0.0202 ❌** | 0.1117 |
| 15000 | 16 | **-1.2191 ❌** | -1.1734 | 0.0133 | 0.1058 |

Reading: the noise-floor gate sits **above the top of the real band in 6 of 10 continuation steps**
— the renders carry *more* high-frequency energy than real faces, which is the quantitative shadow
of "high detail". But nothing is monotone: the excursion count does not rise or fall with training,
and `g0c` oscillates in and out of band. **The frozen G0 instrument is as flat as the Elo** — which
reproduces the 2026-09-28 deficit report's own finding that G0 does not see this defect.

## What this means for the arm

1. **Direction is unchanged and now better supported:** the re-heat did not resume gain. But the
   decisive evidence is the operator's structural read, not the Elo.
2. **The arm's primary instrument cannot adjudicate its own question.** A full 21-item pool at 20k
   would return a reading at SNR far below the good-enough line; the gate's own text makes an
   untrusted or unexhausted pool INCONCLUSIVE, and this arm's pre-registration additionally
   excludes SNR as a gate input. Completing the run to 20k will not produce a PASS/FAIL — it will
   produce a tighter version of this same unresolved ordering.
3. **The next arm needs a structure metric, not a preference Elo.** The already-validated
   structural signal in the harness is *detection attrition* (deficit report: **21.2%** of EIR
   recon renders had no usable skin mask vs **0.0%** for real faces) — a direct "anatomy absent"
   measurement that the current gate does not use as a criterion.

