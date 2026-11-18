# Arm ECR — `eir-cosine-restart`

**One variable: the learning-rate schedule.**

| | |
|---|---|
| Arm | `eir-cosine-restart` (ECR) |
| Mode | `confirmatory` (gate frozen before the first run) |
| Parent | `eidolon-identity-renderer` (EIR) — resumed from its `checkpoint_step0010000.pt` |
| Budget | 10,000 further steps (global horizon 10,000 → 20,000) ≈ 45.6 h on the 4090 |
| Host | `game`, RTX 4090, under the 1k-SEGMENT scheduler policy |
| Status | `concluded` — **PIVOT** (closed 2026-11-18, PRD-0001 T1; deliberately stopped 2026-10-02 at step 16,916; Elo gate recorded INCONCLUSIVE — instrument blind; see `provenance.yaml` + ledger) |

## Hypothesis

The render-quality plateau observed on Arm EIR between ~6k and 10k steps was caused by the
**learning-rate schedule** decaying to its 1e-6 floor — not by exhausted headroom. Re-heating the
LR on EIR's step-10000 weights (same model, same data, same optimizer, same dropout profile;
horizon raised 10000 → 20000 so the LR resumes at 1.56e-4) resumes a **measurable gain in blind
intra-sample human preference**.

## Falsified if

The blind intra-sample Elo ordering over the continuation's checkpoints (10.0k … 20.0k at 500-step
spacing) shows no resolvable improvement: the pre-registered split test separates the late half
above the early half in **0 of 2** identity pools, **and** the automatic trajectory at the same
500-step grid (recon LPIPS, loss, detection attrition) also flattens.

## Pre-registered gate

> PASS if the blind intra-sample Elo ranking separates **late (15.0k–20.0k) above early
> (10.0k–14.5k) perfectly in ≥1 of 2 identity pools** (null p = 1/C(21,11) = **2.8e-6** per pool)
> **with the other pool directionally consistent** (Spearman(checkpoint, Elo) > 0), **and** both
> pools are fully exhausted (every pair judged — an untrusted or unexhausted pool is
> INCONCLUSIVE, not FAIL), **and** no regression in the yaw response (`z_g` dim0 → yaw, monotone,
> correct sign) or in cross-render identity consistency.

**Pre-registered null:** a fixed 11-item set holds the top 11 of 21 ranks with probability
1/C(21,11) per pool; the pooled Spearman has mean 0. Both are computed *before* the run, per the
harness plan's rule.

**Deliberately NOT gated on the tool's SNR meter:** SNR ≥ 5 is structurally unreachable for a
within-run field ~300 Elo wide (it would need avg RD ≤ 30, i.e. ~130 games per image against a
maximum of 20). See `../eidolon-identity-renderer/audit/elo_trajectory/INTERPRETATION_2026-09-30.md`.

## How it differs from Arm EIR — the whole list

| Line | EIR | ECR | Why |
|---|---|---|---|
| `training.total_steps` | 10000 | **20000** | **the intervention** — re-heats the LR at the resume step |
| `checkpoint.save_every` | 1000 | 500 | instrumentation: the 500-step render grid (validation fires from this branch) |
| `checkpoint.keep_last_n` | 10 | 20 | 20 boundaries × ~2.8 GB ≈ 56 GB; NAS holds 12 TB free |
| `validation.interval_steps` | 1000 | 500 | cosmetic — a *dead knob* kept truthful |

Everything else is byte-identical (verified by diff): adapter + dropout profile, model geometry,
TREAD, Muon with peak 3e-4 / floor 1e-6, both data sources, the locked identity-level holdout, and
the whole `sampling:` block. That is what keeps the continuation's renders comparable to the EIR
pool at the same validation indices.

## What gets measured

1. **Primary — human, blind, product criterion:** Elo over intra-sample pools, **2 identities
   (validation indices 0 and 4)** — chosen because the EIR pools already cover those indices, so
   each pool's 10.0k origin item already exists. 21 items each (10.0k, 10.5k … 20.0k), exhaustive
   (210 comparisons/pool). Method: `scripts/harness/elo_results.py` (reads `/projects/{id}.json`).
2. **Secondary, automatic, 500-step grid:** recon LPIPS, loss, G0 gates, detection attrition —
   **reported, not gating** (G0 is proven blind to this defect; attrition is not a validated proxy,
   Spearman +0.50 against the human ranking).
3. **Regression clauses:** yaw response (`z_g` dim0 → yaw) and cross-render identity consistency —
   the root node's own criterion.
4. **Not measured / not claimed:** absolute quality ("how far from good" — outside this
   instrument), resemblance to real people (off-goal by the root), R@1 (retired).

## Operational notes

* **Decision points:** the run is granted in 1k blocks, so **every 1k step is a natural check-in**
  (and a resumable state). The pre-registered decision point is **15k**.
* **Risk to watch:** step 10000 ≫ warmup 500, so the re-heat is *immediate* — the first resumed
  step runs at 1.56e-4 with the parent's Muon/Adam state loaded. Watch loss and `grad_norm` for
  the first ~100 steps (AGENTS.md NaN/divergence guard).
* **Foundation:** EIR's mechanism tier is PASS (pose controllable, identity consistent), so this
  arm builds on a *gated* foundation. EIR's **fidelity tier was never measured** — that is what
  this arm measures — so **ECR inherits no fidelity claim** and must not be described as building
  on one.
* **Hard data:** run dir `experiments/eir-cosine-restart/runs/<ts>/`, seeded with a copy of EIR's
  step-10000 checkpoint so the resume has a valid parent structure, and so the parent arm's
  evidence stays frozen and unambiguous.
