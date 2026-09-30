#!/usr/bin/env python3
"""Deficit report — G0 photorealism gates across real / VAE-ceiling / EIR legs.

WHY THIS EXISTS
    `research/avenues/gates/` holds G0 runs for `arm-j`, `dip-conv-head` and
    `gamma2-noise-scale` — and nothing for the EIDOLON line. So EIR's "missing structure"
    finding lives only in bespoke numbers (LPIPS ~0.80, render HF energy 446.7 vs clean real
    191.7) that no other arm shares, and Phase 1's promised "first photorealism deficit
    report" was never produced. Without it, choosing the next arm is a guess.

WHAT IT MEASURES (legs)
    real          the target: real FFHQ `pixel.npy`, a deterministic slice DISJOINT from the
                  calibration's own sample (different seed).
    vae-ceiling   decode(TRUE `flux_latent.npy`) for the SAME dirs — what the FLUX AE decoder
                  can produce from a perfect latent. This is the "ceiling of the VAE": if this
                  leg is in-band and the model's renders are not, the decoder is NOT the
                  limiter and the shortfall is the model's.
    eir-recon     `validation/step*/reconstruction/*.png` — EIR generating from a real
                  sample's OWN identity+geometry conditioning (the product use case).
    eir-sweep     the render guidance/step sweep panels (settings held at the control cell).
    context       existing gates.jsonl for arm-j / dip-conv-head / gamma2-noise-scale.

INSTRUMENT, NOT REIMPLEMENTED
    Measurement is delegated to `scripts/harness/run_gates.py` against the FROZEN
    `research/avenues/gates_calibration.json` (n=995 real FFHQ, seed 42). Nothing here
    recomputes a gate; this script only materializes inputs and joins outputs.

    A real discrepancy found while building this: `calibrate_gates.py:process_dir` ERODES the
    skin mask (`erode(mask, r=1)`) before measuring, but `run_gates.py --masks` does not. So a
    raw `seg==2` mask does not reproduce the frozen MASKED bands. This script writes masks that
    are already downsampled AND eroded with the shared primitives, so `--masks` reproduces the
    calibration exactly. The `_full` (mask-free) variants were never affected — which is
    fortunate, because every existing arm verdict was scored on those.

USAGE
    .venv/bin/python scripts/harness/deficit_report.py build   [--n 150] [--seed 20260928]
    .venv/bin/python scripts/harness/deficit_report.py measure
    .venv/bin/python scripts/harness/deficit_report.py report
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))                      # for `production.*` imports
sys.path.insert(0, str(REPO / "scripts" / "harness"))
from calibrate_gates import (  # noqa: E402
    FACE_SKIN_V1, MIN_SKIN_FRAC, REF_SIZE, downsample, erode, sample_dirs,
)

STRATUM = Path("/mnt/nas-ai-models/training-data/ffhq/stratum")
EIR_RUN = REPO / "experiments" / "eidolon-identity-renderer" / "runs" / "2026-09-25_0734"
CALIB = REPO / "research" / "avenues" / "gates_calibration.json"
OUT = REPO / "research" / "avenues" / "gates" / "deficit-2026-09-28"
RUN_GATES = REPO / "scripts" / "harness" / "run_gates.py"

LEGS = ("real", "vae-ceiling", "eir-recon", "eir-sweep")
CONTEXT_ARMS = {  # existing runs, read as-is (not re-measured)
    "arm-j": REPO / "research" / "avenues" / "gates" / "arm-j" / "step0010000" / "gates.jsonl",
    "dip-conv-head": REPO / "research" / "avenues" / "gates" / "dip-conv-head" / "step0010000" / "gates.jsonl",
    "gamma2-noise-scale": REPO / "research" / "avenues" / "gates" / "gamma2-noise-scale" / "10000" / "gates.jsonl",
}
# stratum2 (Sapiens2 5B) seg2 outputs for the renders + the real taxonomy-check sample.
# Written by scratch/run_deficit_seg2.sh; consumed by `masks`/`taxonomy`.
SEG2_OUT = Path("/mnt/nas-ai-models/training-data/prx-tg/diagnostics/deficit-2026-09-28/seg2")
SKIN_V2 = 3                      # DOME_CLASSES_29 (seg2): Background0 Apparel1 Eyeglass2 Face_Neck3
V2_LEGS = ("eir-recon", "eir-sweep")   # legs whose masks must come from seg2, not the stratum


# --------------------------------------------------------------------------- build
def _pool() -> list[str]:
    """Readdir-only pool listing (per-dir stat over 70k NFS entries is minutes)."""
    pool = sorted(os.listdir(STRATUM))
    return [p for p in pool if not p.startswith(".")]


def _save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 3:      # channels-first
        a = a.transpose(1, 2, 0)
    # `stratum/*/pixel.npy` is float16 in [0,1]. The PNG must round-trip to the SAME values
    # that run_gates() recovers after its own /255 (it measures on [0,1] luminance), so a
    # [0,1] source is scaled x255. Detect the scale rather than assume it: clipping a [0,1]
    # float image to [0,255] yields an ALL-BLACK png -- which is what the first build
    # produced, and `verify` caught it (real leg mean=0.000).
    if a.size and float(a.max()) <= 1.001:
        a = a * 255.0
    a = np.clip(a, 0.0, 255.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a.astype(np.uint8)).convert("RGB").save(path)


def _skin_mask(seg: np.ndarray, skin_id: int = FACE_SKIN_V1) -> np.ndarray | None:
    """Reproduce calibrate_gates' mask EXACTLY: downsample(seg==skin, 512) then erode r=1.

    `skin_id` differs by taxonomy: the stratum tree's v1 seg has Face_Neck = 2 (what the
    frozen masked bands were calibrated with), while stratum2's seg2 is DOME_CLASSES_29 with
    Face_Neck = 3. Passing the wrong id yields an empty or wrong mask, so the id is explicit.
    """
    m = downsample((seg == skin_id), REF_SIZE)
    m = erode(m, r=1)
    if m.sum() < MIN_SKIN_FRAC * REF_SIZE * REF_SIZE:
        return None
    return m


def _clear(d: Path) -> None:
    """Empty a leg dir before rebuilding: a different --n samples DIFFERENT dirs, so stale
    files from a smaller run would silently contaminate the leg's statistics."""
    if not d.is_dir():
        return
    for pat in ("*.png", "*.npy"):
        for f in d.glob(pat):
            f.unlink()


def build_real(n: int, seed: int) -> list[str]:
    """Materialize real FFHQ images + pre-eroded masks; returns the sampled dir names."""
    pool = _pool()
    sel = sample_dirs(pool, n, seed)
    img_dir, mask_dir = OUT / "real" / "images", OUT / "real" / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    _clear(img_dir)
    _clear(mask_dir)
    kept, skipped = [], 0
    for name in sel:
        d = STRATUM / name
        try:
            pix = np.load(d / "pixel.npy", mmap_mode="r")
            seg = np.load(d / "seg.npy", mmap_mode="r")
        except FileNotFoundError:
            skipped += 1
            continue
        m = _skin_mask(np.asarray(seg))
        _save_png(np.asarray(pix), img_dir / f"{name}.png")
        if m is not None:
            np.save(mask_dir / f"{name}.npy", m)
        kept.append(name)
    (OUT / "real" / "sampled.json").write_text(json.dumps(
        {"pool": len(pool), "requested": n, "seed": seed, "kept": kept,
         "skipped": skipped, "with_mask": len(list(mask_dir.glob("*.npy")))}, indent=2))
    print(f"[build] real: {len(kept)} images ({skipped} skipped) -> {img_dir}", file=sys.stderr)
    return kept


def build_roundtrip(device: str = "cpu", batch: int = 4) -> None:
    """Decode TRUE flux latents -> PNG. Pairwise with `real`: same stems, same masks."""
    import torch
    from production.sample import decode_latents, load_vae_decoder, tensor_to_pil

    names = json.loads((OUT / "real" / "sampled.json").read_text())["kept"]
    img_dir = OUT / "vae-ceiling" / "images"
    mask_dir = OUT / "vae-ceiling" / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    _clear(img_dir)
    _clear(mask_dir)

    # load_vae_decoder returns a single AutoencoderKL (flux_ae.py:96) and is already
    # eval()'d; decode_latents() moves latents to the VAE's own device AND dtype, so we
    # must NOT pre-cast here (casting to a mismatched dtype is the documented failure
    # mode that made an earlier from-HF path produce garbage).
    vae = load_vae_decoder(device=device)

    t0 = time.time()
    done = 0
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        lats = []
        for name in chunk:
            lat = np.load(STRATUM / name / "flux_latent.npy").astype(np.float32)
            lats.append(torch.from_numpy(lat).unsqueeze(0))
            src_mask = OUT / "real" / "masks" / f"{name}.npy"
            if src_mask.is_file():
                np.save(mask_dir / f"{name}.npy", np.load(src_mask))
        x = torch.cat(lats, dim=0)
        with torch.no_grad():
            imgs = decode_latents(vae, x)
        for j, name in enumerate(chunk):
            tensor_to_pil(imgs[j].float().cpu()).save(img_dir / f"{name}.png")
        done += len(chunk)
        if i == 0:
            t0 = time.time()
        print(f"[build] vae-ceiling {done}/{len(names)} "
              f"({(time.time() - t0) / max(1, done):.2f}s/img)", file=sys.stderr)
    print(f"[build] vae-ceiling: {done} images -> {img_dir}", file=sys.stderr)


def build_eir() -> None:
    """Symlink EIR renders in (no 100MB+ NFS copy): recon per boundary, plus sweep panels."""
    recon_dir = OUT / "eir-recon" / "images"
    sweep_dir = OUT / "eir-sweep" / "images"
    recon_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    n_recon = 0
    for step_dir in sorted((EIR_RUN / "validation").glob("step*")):
        rec = step_dir / "reconstruction"
        for png in sorted(rec.glob("*.png")):
            link = recon_dir / f"{step_dir.name}_{png.name}"
            if not link.exists():
                os.symlink(png.resolve(), link)
            n_recon += 1

    sweep_src = EIR_RUN / "guidance_sweep"
    n_sweep = 0
    for png in sorted(sweep_src.glob("s[0-9]*.png")):   # per-panel only (excludes sheet_*.png)
        link = sweep_dir / png.name
        if not link.exists():
            os.symlink(png.resolve(), link)
        n_sweep += 1
    print(f"[build] eir-recon: {n_recon} symlinks; eir-sweep: {n_sweep} symlinks", file=sys.stderr)


# ------------------------------------------------------------------------- measure
def measure() -> None:
    for leg in LEGS:
        img_dir = OUT / leg / "images"
        if not img_dir.is_dir() or not list(img_dir.glob("*.png")):
            print(f"[measure] SKIP {leg}: no images", file=sys.stderr)
            continue
        cmd = [sys.executable, str(RUN_GATES), "--images", str(img_dir),
               "--calibration", str(CALIB), "--out", str(OUT / leg / "gates.jsonl")]
        mask_dir = OUT / leg / "masks"
        if mask_dir.is_dir() and list(mask_dir.glob("*.npy")):
            cmd += ["--masks", str(mask_dir)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[measure] FAILED {leg} rc={r.returncode}\n{r.stderr[-2000:]}", file=sys.stderr)
            continue
        n = len(list(img_dir.glob("*.png")))
        print(f"[measure] {leg}: n={n} -> {OUT / leg / 'gates.jsonl'}", file=sys.stderr)


# -------------------------------------------------------------------------- report
def _read_gates(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["gate_id"]] = rec
    return out


def _gate_bands(bands: dict) -> dict:
    """Calibration `gates` map, minus non-gate entries (`__meta__`) and anything without a band."""
    return {k: v for k, v in bands.items()
            if not k.startswith("_") and isinstance(v, dict) and "p5" in v and "p95" in v}


def masks_from_seg2() -> int:
    """Build eroded skin masks for the RENDER legs from stratum2 seg2 outputs.

    G0a/G0c are skin-region gates; renders ship no seg mask, so on generated images the
    frozen suite can only compute the whole-frame `_full` variants -- which every existing
    arm verdict was scored on, and which may not rank renders by the property that matters.
    Segmenting the renders (stratum-hq / Sapiens2 5B, operator steer 2026-09-28) makes the
    masked gates available for them.

    Masks are written pre-downsampled AND pre-eroded so `run_gates.py --masks` reproduces the
    frozen calibration exactly (run_gates does not erode; calibrate_gates does).
    """
    made, missing = {}, []
    for leg in V2_LEGS:
        img_dir, mask_dir = OUT / leg / "images", OUT / leg / "masks"
        if not img_dir.is_dir():
            continue
        mask_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for png in sorted(img_dir.glob("*.png")):
            seg_path = SEG2_OUT / png.stem / "seg2.npy"
            if not seg_path.is_file():
                missing.append(f"{leg}/{png.stem}")
                continue
            m = _skin_mask(np.load(seg_path), skin_id=SKIN_V2)
            if m is None:
                missing.append(f"{leg}/{png.stem} (no usable skin)")
                continue
            np.save(mask_dir / f"{png.stem}.npy", m)
            n += 1
        made[leg] = n
        print(f"[masks] {leg}: {n} masks from seg2 -> {mask_dir}", file=sys.stderr)
    if missing:
        print(f"[masks] no mask for {len(missing)} images (first 5: {missing[:5]})", file=sys.stderr)
    return 0 if made else 1


def taxonomy_check(min_iou: float = 0.85) -> int:
    """Validate the taxonomy substitution BEFORE any masked render number is believed.

    The frozen masked bands describe v1 masks (stratum `seg.npy`, Face_Neck=2). Render masks
    come from seg2 (DOME_CLASSES_29, Face_Neck=3). Both aim at the same anatomy, but that is
    an ASSUMPTION until measured: segment the SAME real faces both ways and compare the
    resulting skin masks by IoU. Without this, a masked render number could be an artifact of
    a different class boundary rather than a real measurement.
    """
    import json as _json
    ids = _json.loads((OUT / "real" / "sampled.json").read_text())["kept"][:12]
    ious, bad = [], []
    for i in ids:
        v1 = np.load(STRATUM / i / "seg.npy", mmap_mode="r")
        v2_path = SEG2_OUT / i / "seg2.npy"
        if not v2_path.is_file():
            bad.append((i, "no seg2"))
            continue
        a = _skin_mask(np.asarray(v1), skin_id=FACE_SKIN_V1)
        b = _skin_mask(np.load(v2_path), skin_id=SKIN_V2)
        if a is None or b is None:
            bad.append((i, "empty mask"))
            continue
        inter = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        ious.append(inter / union if union else 0.0)
    mean_iou = float(np.mean(ious)) if ious else float("nan")
    print(f"[taxonomy] v1(Face_Neck=2) vs seg2(Face_Neck=3) skin-mask IoU: "
          f"n={len(ious)} mean={mean_iou:.3f} min={min(ious) if ious else float('nan'):.3f} "
          f"bar={min_iou}", file=sys.stderr)
    if bad:
        print(f"[taxonomy] unusable: {bad[:5]}", file=sys.stderr)
    ok = bool(ious) and mean_iou >= min_iou and not bad
    print("[taxonomy]", "PASS" if ok else "FAIL", file=sys.stderr)
    return 0 if ok else 1


def verify(tol_mse: float = 0.002) -> int:
    """Self-check the legs: the VAE round-trip must reproduce its source image.

    The whole `vae-ceiling` claim rests on the decode being faithful, so assert it against
    the source `pixel.npy`-derived PNG (same protocol flux_ae.py's own self-test uses) rather
    than trusting that it "looked fine". Also flags a blank/degenerate leg by mean/std.
    """
    from PIL import Image
    names = json.loads((OUT / "real" / "sampled.json").read_text())["kept"]
    rt_dir, real_dir = OUT / "vae-ceiling" / "images", OUT / "real" / "images"

    def arr(p: Path):
        a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
        return a

    mses, bad = [], []
    for name in names:
        a, b = real_dir / f"{name}.png", rt_dir / f"{name}.png"
        if not (a.is_file() and b.is_file()):
            bad.append((name, "missing png"))
            continue
        x, y = arr(a), arr(b)
        if x.shape != y.shape:
            bad.append((name, f"shape {x.shape} vs {y.shape}"))
            continue
        mses.append(float(np.mean((x - y) ** 2)))
    m = float(np.mean(mses)) if mses else float("nan")
    print(f"[verify] paired real vs vae-ceiling: n={len(mses)} mean mse={m:.5f} "
          f"(max {max(mses):.5f}) tol={tol_mse}", file=sys.stderr)
    ok = bool(mses) and m < tol_mse and not bad
    if bad:
        print(f"[verify] PROBLEM pairs: {bad[:5]}", file=sys.stderr)
    for leg in LEGS:
        d = OUT / leg / "images"
        if not d.is_dir():
            continue
        a = arr(sorted(d.glob("*.png"))[0]) if list(d.glob("*.png")) else None
        if a is not None:
            print(f"[verify] {leg}: first-image mean={a.mean():.3f} std={a.std():.3f} "
                  f"({'PLAUSIBLE' if 0.01 < a.std() < 0.4 else 'DEGENERATE?'})", file=sys.stderr)
    print("[verify]", "OK" if ok else "FAILED", file=sys.stderr)
    return 0 if ok else 1


def report() -> None:
    calib = json.loads(CALIB.read_text())
    bands = _gate_bands(calib["gates"])
    legs = {leg: _read_gates(OUT / leg / "gates.jsonl") for leg in LEGS}
    ctx = {arm: _read_gates(p) for arm, p in CONTEXT_ARMS.items()}

    gate_ids = sorted(set(bands) | {g for d in list(legs.values()) + list(ctx.values()) for g in d})

    md = ["# Photorealism deficit report — 2026-09-28", "",
          "Instrument: `scripts/harness/run_gates.py` against the frozen",
          f"`research/avenues/gates_calibration.json` (n={calib['reference'].get('n_sampled')} real FFHQ,",
          f"seed {calib['reference'].get('seed')}). Bands are [p5, p95] of the real reference.", "",
          "| gate | band [p5, p95] | real (A) | **VAE ceiling (C)** | EIR recon (B) | EIR sweep | arm-j | dip-conv-head |",
          "|---|---|---|---|---|---|---|---|"]

    def cell(rec: dict | None, band: dict | None) -> str:
        if rec is None:
            return "—"
        v = rec["value"]
        mark = ""
        if band is not None:
            mark = " ✅" if band["p5"] <= v <= band["p95"] else " ❌"
        return f"{v:.4f}{mark}"

    for g in gate_ids:
        b = bands.get(g)
        btxt = f"[{b['p5']:.4f}, {b['p95']:.4f}]" if b else "—"
        md.append("| `" + g + "` | " + btxt + " | "
                  + " | ".join([cell(legs["real"].get(g), b),
                                cell(legs["vae-ceiling"].get(g), b),
                                cell(legs["eir-recon"].get(g), b),
                                cell(legs["eir-sweep"].get(g), b),
                                cell(ctx["arm-j"].get(g), b),
                                cell(ctx["dip-conv-head"].get(g), b)]) + " |")

    md += ["", "## Counts in band", "",
           "| leg | in band | out of band | of total |", "|---|---|---|---|"]
    summary = {"generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
               "calibration": str(CALIB.relative_to(REPO)), "legs": {}}
    for label, d in [("real (A)", legs["real"]), ("vae-ceiling (C)", legs["vae-ceiling"]),
                     ("eir-recon (B)", legs["eir-recon"]), ("eir-sweep", legs["eir-sweep"]),
                     ("arm-j", ctx["arm-j"]), ("dip-conv-head", ctx["dip-conv-head"]),
                     ("gamma2-noise-scale", ctx["gamma2-noise-scale"])]:
        n_in = sum(1 for g, r in d.items() if g in bands and r.get("in_band"))
        n_out = sum(1 for g, r in d.items() if g in bands and r.get("in_band") is False)
        n_tot = n_in + n_out
        summary["legs"][label] = {"in_band": n_in, "out_of_band": n_out, "scored": n_tot,
                                  "gates": {g: r["value"] for g, r in d.items()}}
        md.append(f"| {label} | {n_in} | {n_out} | {n_tot} |")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (REPO / "docs" / "DEFICIT_REPORT_2026-09-28.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-14:]), file=sys.stderr)
    print(f"\n[report] wrote {OUT / 'summary.json'} and docs/DEFICIT_REPORT_2026-09-28.md",
          file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "build-roundtrip", "measure", "verify", "masks",
                                    "taxonomy", "report", "all"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd in ("build", "all"):
        build_real(args.n, args.seed)
        build_eir()
    if args.cmd in ("build-roundtrip", "all"):
        build_roundtrip(device=args.device)
    if args.cmd in ("measure", "all"):
        measure()
    if args.cmd == "masks":
        return masks_from_seg2()
    if args.cmd == "taxonomy":
        return taxonomy_check()
    if args.cmd == "verify":
        return verify()
    if args.cmd in ("report", "all"):
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
