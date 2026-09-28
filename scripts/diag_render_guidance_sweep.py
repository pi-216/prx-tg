#!/usr/bin/env python3
"""DIAGNOSTIC: is the garbled look of Arm EIR's renders an artifact of our OWN
sampling settings (guidance scales + step count) rather than of the model?

Context
-------
`eidolon-identity-renderer` (Arm EIR) concluded PIVOT. Every image judged in that
review was rendered by the arm's default three-pass dual-CFG sampling at identity
guidance 3.0 x geometry guidance 2.0, 50 steps, and the diagnostic ladders that
accompanied it pushed GEOMETRY guidance to 8x and 10x
(experiment-configs/eidolon-identity-renderer/config.yaml -> `sampling:`;
scripts/diag_eidolon_zg_dilution.py CFG_CONFIGS). High dual-CFG scales are a
well-known source of saturated / oversharp / "garbled" output, so part of the
observed garble may be a property of the sampler settings we chose, not of the
checkpoint.

Hypothesis under test
---------------------
Part of the observed garbling is an artifact of our own sampling settings.
Falsified if the low-guidance and longer-schedule cells are indistinguishable
from the default control.

What this script does
---------------------
It renders images and nothing else: a FIXED identity conditioning, a FIXED
geometry conditioning and a FIXED initial noise draw, at a single checkpoint,
across a grid of sampling settings. The ONLY things that vary between panels are

    (a) identity guidance scale   (= EulerSampler.sample(text_scale=...))
    (b) geometry guidance scale   (= EulerSampler.sample(dino_scale=...))
    (c) the number of Euler steps (= EulerSampler(num_steps=...))

Cells (default; --cells overrides, space separated, each "ID,GEO,STEPS")

    3.0,2.0,50     <- CONTROL: the arm config default, matches every image
                      judged so far
    1.0,1.0,50     <- no guidance at all, same schedule
    1.5,1.0,50
    2.0,1.0,50
    1.0,1.0,100    <- no guidance, double schedule
    3.0,2.0,100    <- default guidance, double schedule

Sample indices are the project's fixed evaluation indices [10, 30, 50] (same list
as scripts/diag_pose_cfg_sweep.py and validate.py's
EIDOLON_GEOMETRY_SWEEP_INDICES), the SAME indices in every cell, so every
comparison in this run is intra-sample: same persona, same geometry, same noise,
only guidance/steps move.

Noise matching
--------------
`torch.manual_seed(SEED)` is reset immediately before EVERY generation and no
`seed=` argument is passed to the sampler, exactly as
scripts/diag_pose_cfg_sweep.py does it. The initial latent draw is therefore
byte-identical in every cell of the whole run -- including across sample indices
-- so a panel difference is attributable to guidance / step count only.
(The in-train validation sweep takes the other documented path: it passes an
explicit `seed=` so the draw comes from a local generator and the global RNG
stream is never perturbed. That path is used for in-run renders because they
execute INLINE in the training loop; it is not needed here.)

Face detection
--------------
A no-face count is reported, reusing the project's own CPU DWPose ONNX detector
(scripts/dwpose_onnx.py) exactly as scripts/diag_eidolon_zg_dilution.py does
(which is also where the detector call convention and the fail-soft behaviour
come from). An image the detector cannot parse at all is counted as no-face, not
silently dropped. The yaw proxy of that script is deliberately NOT computed here:
z_g is fixed in this probe, so a yaw reading would have no attributable cause and
would invite a conclusion the design cannot support. Detection confidence and
keypoint count are recorded per panel as supporting evidence. `--no-face-detect`
skips detection entirely (CPU ONNX is not free at 1024px).

Outputs (under --out-dir)
-------------------------
    s{idx:02d}_{cell}.png                    one panel per (cell, sample index)
    sheet_{cell}_cols=samples[...].png        one contact sheet per setting:
                                             the sample indices side by side
    comparison_rows=samples[...]_cols=[...].png
                                            the cross-setting comparison sheet:
                                            one ROW per sample index, one COLUMN
                                            per cell, in --cells order
    results.json                             grid, per-cell + per-panel paths,
                                             seed, checkpoint, git HEAD, command
                                             line, no-face counts

Provenance of the copied API calls
----------------------------------
* Model construction + checkpoint load + validation-dataloader conditioning
  extraction + EulerSampler invocation style + the `grid()` panel sheet:
      scripts/diag_pose_cfg_sweep.py  (the source of truth this file copies)
* Latent-vs-pixel shape selection, VAE decode and the identity/geometry basis
  guards (this arm is FLUX-AE LATENT space, which diag_pose_cfg_sweep.py never
  handled -- it only covers pixel-space arms):
      production/validate.py (ValidationSampler.generate / run_validation) and
      production/sample.py (EulerSampler.sample, decode_latents,
      load_vae_decoder, tensor_to_pil)
* DWPose detector setup + fail-soft detector call:
      scripts/diag_eidolon_zg_dilution.py
* `--sweep-values`-style CLI (nargs="+", space separated, strict defaults):
      scripts/diag_eidolon_zg_dilution.py

Deliberate deviations from scripts/diag_pose_cfg_sweep.py (all documented here,
none silent):
  1. `EulerSampler(..., gamma=gamma)` is passed explicitly. The sibling resolves
     the gamma and then constructs the sampler without it (EulerSampler's own
     default is 1.0). For THIS arm gamma resolves to 1.0, so the call is a no-op
     here, but passing it keeps the sampler matched to the training schedule if a
     noise_schedule block is ever enabled (production/config_loader.py documents
     that an unmatched sampler produced dark-green noise).
  2. The checkpoint is named explicitly (checkpoint_final.pt) rather than
     auto-discovered. The sibling's mtime-based discovery would pick
     checkpoint_step0010000.pt here, i.e. a different artefact than the one this
     probe is asked to measure. The equivalent alternative is passed with
     --checkpoint.
  3. Panels are moved to CPU for storage (the sibling keeps them on the GPU); a
     six-cell grid holds 18 1024px panels, and this keeps VRAM flat.
  4. DWPoseDetector is imported lazily inside build_detector() so that --help and
     module import do not depend on onnxruntime being present. The detector still
     fails loudly when detection is actually requested.
  5. Optional text labels on the sheets (the sibling's `grid()` draws none), so
     the cross-setting comparison sheet identifies its own rows/columns.

This script does NOT train, does not write to the checkpoint, and does not modify
anything under production/ or any existing script.

GPU ARBITRATION
---------------
This script allocates the GPU. The shared 4090 is arbitrated by the file-based
scheduler at /mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py (see the
gpu-resource-scheduler skill): acquire a claim before running it and release it
afterwards. One cell at 50 steps is a few seconds per panel; the default grid is
18 panels.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Repo root on sys.path, exactly as scripts/diag_pose_cfg_sweep.py does.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from production.config_loader import (  # noqa: E402
    load_config,
    resolve_latent_space,
    resolve_sampling_gamma,
)
from production.model import NanoDiT  # noqa: E402
from production.sample import (  # noqa: E402
    EulerSampler,
    decode_latents,
    load_vae_decoder,
    tensor_to_pil,
)
from production.data import get_deterministic_validation_dataloader  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed experiment parameters
# ---------------------------------------------------------------------------
CONFIG_PATH = (
    REPO / "experiment-configs" / "eidolon-identity-renderer" / "config.yaml"
)
# Named explicitly: the artefact under test. `experiments` is a git symlink to the
# NAS tree, so this is the same file as
# /mnt/nas-ai-models/training-data/prx-tg/eidolon-identity-renderer/runs/
# 2026-09-25_0734/checkpoints/checkpoint_final.pt
DEFAULT_CHECKPOINT = (
    REPO / "experiments" / "eidolon-identity-renderer" / "runs"
    / "2026-09-25_0734" / "checkpoints" / "checkpoint_final.pt"
)
DEFAULT_OUT_DIR = (
    REPO / "experiment-configs" / "eidolon-identity-renderer" / "audit"
    / "guidance_sweep"
)

SAMPLE_INDICES = [10, 30, 50]
SEED = 1234                 # reset before EVERY generation (noise-matched grid)

# (identity_scale, geometry_scale, num_steps) -- passed to sampler.sample() as
# text_scale / dino_scale respectively. For the eidolon adapter the sampler's
# "text" branch IS identity-only and its "dino" branch IS geometry-only
# (production/sample.py dual-CFG block; documented in the arm config under
# `sampling:`), so these two numbers are identity and geometry guidance.
DEFAULT_CELLS = [
    "3.0,2.0,50",     # CONTROL: arm config default == every image judged so far
    "1.0,1.0,50",     # no guidance, same schedule
    "1.5,1.0,50",
    "2.0,1.0,50",
    "1.0,1.0,100",    # no guidance, double schedule
    "3.0,2.0,100",    # default guidance, double schedule
]
CONTROL_CELL = (3.0, 2.0, 50)

# Fail-closed bounds for --cells (a typo must not silently render nonsense).
MAX_SCALE = 50.0
MAX_STEPS = 1000


# ---------------------------------------------------------------------------
# CLI parsing helpers (strict; every invalid value exits non-zero)
# ---------------------------------------------------------------------------
def parse_cell_spec(spec: str):
    """'ID,GEO,STEPS' -> (identity_scale, geometry_scale, num_steps).

    Used as an argparse `type=`, so a malformed or out-of-bounds value becomes a
    usage error and a non-zero exit (argparse converts ArgumentTypeError).
    """
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"cell {spec!r}: expected exactly 3 comma-separated fields "
            f"'IDENTITY,GEO,TRIALS' i.e. 'IDENTITY,GEO,STEPS', got {len(parts)}"
        )
    try:
        ids, gs, steps_f = (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"cell {spec!r}: all three fields must be numeric "
            f"(identity_scale, geometry_scale, num_steps)"
        )
    for label, val in (("identity_scale", ids), ("geometry_scale", gs),
                       ("num_steps", steps_f)):
        if not math.isfinite(val):
            raise argparse.ArgumentTypeError(f"cell {spec!r}: {label} is not finite")
    if steps_f != int(steps_f):
        raise argparse.ArgumentTypeError(
            f"cell {spec!r}: num_steps must be a whole number of Euler steps"
        )
    steps = int(steps_f)
    if not (0.0 <= ids <= MAX_SCALE):
        raise argparse.ArgumentTypeError(
            f"cell {spec!r}: identity_scale must be in [0, {MAX_SCALE:g}]"
        )
    if not (0.0 <= gs <= MAX_SCALE):
        raise argparse.ArgumentTypeError(
            f"cell {spec!r}: geometry_scale must be in [0, {MAX_SCALE:g}]"
        )
    if not (1 <= steps <= MAX_STEPS):
        raise argparse.ArgumentTypeError(
            f"cell {spec!r}: num_steps must be in [1, {MAX_STEPS}]"
        )
    return (ids, gs, steps)


def fmt_scale(value: float) -> str:
    """3.0 -> '3', 1.5 -> '1.5' (compact, filesystem safe)."""
    return f"{value:g}"


def cell_name(identity_scale: float, geometry_scale: float, num_steps: int) -> str:
    return (f"id{fmt_scale(identity_scale)}_geo{fmt_scale(geometry_scale)}"
            f"_steps{int(num_steps)}")


def cell_spec_str(cell):
    """(3.0, 2.0, 50) -> '3.0,2.0,50', for CLI and provenance messages."""
    ids, gs, steps = cell
    return f"{fmt_scale(ids)},{fmt_scale(gs)},{int(steps)}"


def build_cells(specs):
    """Turn parsed (identity, geometry, steps) tuples into cell records.

    Fails closed on duplicates: identical settings would collide on panel
    filenames and silently drop a setting from the comparison sheet.
    """
    if not specs:
        raise SystemExit("--cells must contain at least one 'ID,GEO,STEPS' cell")
    cells = []
    seen = set()
    for cell in specs:
        ids, gs, steps = cell
        name = cell_name(ids, gs, steps)
        if name in seen:
            raise SystemExit(
                f"duplicate cell {name!r} (identity {fmt_scale(ids)}, geometry "
                f"{fmt_scale(gs)}, steps {int(steps)}) was listed more than once: "
                f"identical settings would collide on panel filenames and drop a "
                f"setting from the comparison sheet; pass distinct cells"
            )
        seen.add(name)
        cells.append({
            "name": name,
            "identity_scale": ids,
            "geometry_scale": gs,
            "num_steps": steps,
            "is_control": (ids, gs, steps) == CONTROL_CELL,
        })
    return cells


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------
def _git(args):
    try:
        r = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True,
                           text=True, check=False)
    except OSError:
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def git_provenance():
    """git HEAD + dirty flag, or a loud failure before any GPU work happens."""
    head = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain"])
    if head is None or status is None:
        raise SystemExit(
            f"cannot read git provenance from {REPO}: results.json must record "
            f"the commit that produced the images (AGENTS.md experiment rules)"
        )
    return {
        "git_head": head,
        "git_dirty": bool(status),
        "git_dirty_entries": len(status.splitlines()) if status else 0,
    }


def rel_to_repo(path: Path) -> str:
    """Repo-relative path when possible (the repo's own convention for citing
    artefacts), absolute otherwise. Never a symlink-resolved NAS path by accident:
    the unresolved form is what a reader can re-open from the repo root."""
    p = Path(path)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Model construction -- copied faithfully from scripts/diag_pose_cfg_sweep.py
# ---------------------------------------------------------------------------
def build_model(config, ckpt_path: Path, device: torch.device):
    kw = dict(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        use_gradient_checkpointing=False,
    )
    ad = getattr(config, "adapter", None)
    if ad:
        kw["adapter_kwargs"] = {
            "name": ad.name,
            "identity_dim": getattr(ad, "identity_dim", 64),
            "z_g_dim": getattr(ad, "z_g_dim", 50),
            "geometry_token_basis": getattr(ad, "geometry_token_basis", False),
        }
    model = NanoDiT(**kw)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Training weights, NOT ema (matches diag_pose_cfg_sweep.py / visual_debug).
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Validation samples -- copied faithfully from scripts/diag_pose_cfg_sweep.py
# ---------------------------------------------------------------------------
def load_samples(config, latent_space: bool):
    """Grab SAMPLE_INDICES from the deterministic validation dataloader.

    diag_pose_cfg_sweep.py passes target_latent_size=config.model.input_size and
    does not forward the identity-basis guard. For a LATENT-space arm,
    production/validate.py uses input_size * 8 (a 1024px image -> 128x128 FLUX
    latents), so this script follows validate.py for that one argument only. The
    value is ignored for conditioning extraction (only identity_emb /
    geometry_emb are used), and production/data_stratum.py documents that
    target_latent_size is IGNORED in latent mode anyway.
    """
    target_latent_size = (
        config.model.input_size * 8 if latent_space else config.model.input_size
    )
    loader = get_deterministic_validation_dataloader(
        shard_dir=None,
        batch_size=1,
        target_latent_size=target_latent_size,
        source="stratum",
        stratum_dir=config.data.stratum_dir,
        adapter_name="eidolon",
        # Forward the guard values the arm ships (production/train_production.py
        # does this for the validation loader) so the basis guard does not reject
        # the root and silently yield zero samples.
        expected_basis_fingerprint=getattr(config.data, "basis_fingerprint", None),
        allow_unstamped_identity=getattr(config.data, "allow_unstamped_identity",
                                         False),
    )

    maxi = max(SAMPLE_INDICES)
    samples = []
    it = iter(loader)
    cnt = 0
    while cnt <= maxi:
        b = next(it)
        n = (b["image_data"].shape[0] if "image_data" in b
             else b["identity_emb"].shape[0])
        for i in range(n):
            if cnt <= maxi:
                samples.append({
                    "identity_emb": b["identity_emb"][i],
                    "geometry_emb": b["geometry_emb"][i],
                    "image_id": b["image_ids"][i],
                })
                cnt += 1
    if len(samples) <= maxi:
        raise RuntimeError(
            f"Validation loader yielded only {len(samples)} samples; need index {maxi}"
        )
    return samples


# ---------------------------------------------------------------------------
# Sheet builder -- copied from scripts/diag_pose_cfg_sweep.py (grid()); the
# optional `labels` argument is this script's addition (deviation 5 above).
# rows: list of rows, each a list of (3,H,W) tensors in [-1,1]. One COLUMN per
# cell / sample, one ROW per grouping.
# ---------------------------------------------------------------------------
def _label_font(size=28):
    """Best-effort label font. load_default(size=) exists on Pillow >= 10.1."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _label_strip(text, width, font, pad=6):
    """White strip with black text, sized to `width` and the font's own metrics."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font)
    h = (b - t) + 2 * pad
    strip = Image.new("RGB", (width, h), (255, 255, 255))
    ImageDraw.Draw(strip).text((pad, pad - t), text, font=font, fill=(0, 0, 0))
    return strip


def grid(rows, spacing=8, labels=None, font=None):
    """Build a sheet image.

    rows: list of list of (3,H,W) tensors in [-1,1].
    labels: optional list of list of str, same shape as rows; each label is drawn
            on a white strip above its own panel.
    """
    font = font or _label_font()
    pil_rows = []
    for ri, r in enumerate(rows):
        imgs = []
        for ci, img in enumerate(r):
            a = (img.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            panel = Image.fromarray((a.transpose(1, 2, 0) * 255).astype(np.uint8))
            if labels is not None:
                text = labels[ri][ci]
                if text:
                    strip = _label_strip(text, panel.width, font)
                    labelled = Image.new(
                        "RGB", (panel.width, panel.height + strip.height),
                        (255, 255, 255),
                    )
                    labelled.paste(strip, (0, 0))
                    labelled.paste(panel, (0, strip.height))
                    panel = labelled
            imgs.append(panel)
        pil_rows.append(imgs)
    h = pil_rows[0][0].height
    w = pil_rows[0][0].width
    ncol = max(len(r) for r in pil_rows)
    nrow = len(pil_rows)
    W = ncol * w + (ncol - 1) * spacing
    H = nrow * h + (nrow - 1) * spacing
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    for ri, r in enumerate(pil_rows):
        for ci, im in enumerate(r):
            canvas.paste(im, (ci * (w + spacing), ri * (h + spacing)))
    return canvas


# ---------------------------------------------------------------------------
# Face detection -- DWPose setup + fail-soft call copied from
# scripts/diag_eidolon_zg_dilution.py (yaw proxy intentionally NOT computed: z_g
# is fixed in this probe, so a yaw number would have no attributable cause).
# ---------------------------------------------------------------------------
def build_detector():
    from scripts.dwpose_onnx import DWPoseDetector
    print("loading DWPose (ONNX, CPU)...", flush=True)
    return DWPoseDetector(device="cpu")


def detect_face(det, panel_path: Path):
    """Return (det_conf, n_keypoints) or None when no face was detected.

    A panel the detector cannot read, or that yields no keypoints, returns None
    and is counted as no-face -- never substituted. Detector exceptions are
    reported and also count as no-face, matching the sibling's fail-soft call.
    """
    img = cv2.imread(str(panel_path))
    if img is None:
        print(f"    [warn] unreadable panel: {panel_path.name}")
        return None
    try:
        kp, sc, _bb = det(img, single_person=True)
    except Exception as e:
        print(f"    [warn] detector error on {panel_path.name}: "
              f"{type(e).__name__}: {e}")
        return None
    if kp is None or len(kp) == 0:
        return None
    kp = np.asarray(kp[0] if kp.ndim == 3 else kp)
    sc = (np.asarray(sc[0] if sc is not None and np.asarray(sc).ndim == 2 else sc)
          if sc is not None else None)
    conf = float(np.mean(sc)) if sc is not None and len(sc) else float("nan")
    return conf, len(kp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # The `global` declaration must precede the parser build below: argparse reads
    # SAMPLE_INDICES for the --sample-indices default, and load_samples() reads the
    # module-level value, so the CLI override has to land on the module global.
    global SAMPLE_INDICES

    ap = argparse.ArgumentParser(
        description="Render one fixed identity + fixed geometry sample across a "
                    "grid of guidance/step settings at a single checkpoint, to "
                    "test whether Arm EIR's garbled renders are an artifact of "
                    "the sampling settings we chose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=str, default=str(CONFIG_PATH),
                    help="arm config; supplies model shape, prediction_type, "
                         "data roots and the latent-space/gamma discriminators")
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                    help="checkpoint .pt (checkpoint_step0010000.pt is the "
                         "equivalent alternative in the same directory)")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                    help="output directory (panels, sheets, results.json)")
    ap.add_argument("--cells", type=parse_cell_spec, nargs="+",
                    default=[parse_cell_spec(s) for s in DEFAULT_CELLS],
                    metavar="ID,GEO,STEPS",
                    help="sampling settings to render, space separated, each "
                         "'identity_scale,geometry_scale,num_steps'. Every cell "
                         "renders the same samples with the same initial noise. "
                         "Defaults as (identity,geometry,steps) tuples: "
                         "%(default)s")
    ap.add_argument("--sample-indices", type=int, nargs="+",
                    default=list(SAMPLE_INDICES),
                    help="validation sample indices to render in every cell "
                         "(the same indices across cells is the point: the "
                         "comparison is intra-sample)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="seed reset immediately before EVERY generation "
                         "(noise-matched grid)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no-face-detect", action="store_true",
                    help="skip the DWPose no-face count (CPU ONNX, not free at "
                         "1024px); results.json then records no face data rather "
                         "than zeros")
    args = ap.parse_args()

    cells = build_cells(args.cells)

    sample_indices = list(args.sample_indices)
    if not sample_indices:
        raise SystemExit("--sample-indices must contain at least one index")
    if len(set(sample_indices)) != len(sample_indices):
        raise SystemExit(f"--sample-indices has duplicates: {sample_indices}")
    if any(i < 0 for i in sample_indices):
        raise SystemExit(f"--sample-indices must be non-negative: {sample_indices}")
    if any(i != int(i) for i in sample_indices):
        raise SystemExit(f"--sample-indices must be integers: {sample_indices}")
    SAMPLE_INDICES = sample_indices

    out = Path(args.out_dir)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    provenance = git_provenance()
    command_line = " ".join(sys.argv)

    # Out-dir is created only after the pre-flight gates pass, so a failed
    # pre-flight leaves no empty audit directory behind.
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EIDOLON RENDER GUIDANCE / STEP SWEEP (sampling settings vs garble)")
    print("=" * 72)
    print(f"  config     : {args.config}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  out-dir    : {out}")
    print(f"  samples    : {SAMPLE_INDICES}   seed {args.seed} "
          f"(reset before every generation)")
    print(f"  git        : {provenance['git_head']} "
          f"dirty={provenance['git_dirty']}")
    for c in cells:
        tag = "  <- CONTROL" if c["is_control"] else ""
        print(f"  cell       : {c['name']:22s} identity_scale={c['identity_scale']}"
              f"  geometry_scale={c['geometry_scale']}  steps={c['num_steps']}{tag}")

    config = load_config(str(args.config))
    latent_space = resolve_latent_space(config)
    gamma = resolve_sampling_gamma(config)
    prediction_type = config.model.prediction_type
    device = torch.device(args.device)
    print(f"  device={device} latent_space={latent_space} gamma={gamma} "
          f"prediction_type={prediction_type}")

    model = build_model(config, ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} params")

    vae = load_vae_decoder(device=device) if latent_space else None

    samples = load_samples(config, latent_space)
    print(f"  loaded {len(samples)} validation samples (need index "
          f"{max(SAMPLE_INDICES)})")

    det = None if args.no_face_detect else build_detector()

    # Generation shape: pixel-space arms sample RGB at full res; this arm is
    # latent-space, so latents are sampled at the model grid and VAE-decoded
    # (production/sample.py ValidationSampler.generate, production/validate.py).
    if latent_space:
        in_channels, spatial = 16, config.model.input_size
    else:
        in_channels, spatial = 3, 1024
    shape = (1, in_channels, spatial, spatial)

    results = {
        "script": rel_to_repo(Path(__file__)),
        "repo": str(REPO),
        "command_line": command_line,
        "config": str(args.config),
        "checkpoint": str(ckpt_path),
        "checkpoint_size_bytes": ckpt_path.stat().st_size,
        "checkpoint_mtime_ns": ckpt_path.stat().st_mtime_ns,
        "git_head": provenance["git_head"],
        "git_dirty": provenance["git_dirty"],
        "git_dirty_entries": provenance["git_dirty_entries"],
        "device": str(device),
        "seed": args.seed,
        "noise": ("torch.manual_seed(seed) immediately before every "
                  "generation, no seed= passed to the sampler "
                  "(diag_pose_cfg_sweep.py convention); the initial latent draw "
                  "is identical in every cell of this run"),
        "sample_indices": SAMPLE_INDICES,
        "generation_shape": list(shape),
        "latent_space": latent_space,
        "gamma": gamma,
        "prediction_type": prediction_type,
        "model_params": n_params,
        "control_cell": cell_name(*CONTROL_CELL),
        "face_detector": (None if det is None
                          else "scripts/dwpose_onnx.DWPoseDetector(device=cpu)"),
        "cells": [],
        "panels_total": 0,
        "no_face_panels_total": 0,
        "notes": [],
    }

    grid_tensors = []          # [cell_idx][sample_idx] -> CPU tensor, shared by
    grid_labels = []           # the sheets below; built once, not re-rendered.
    panels_total = 0
    no_face_total = 0

    for cell in cells:
        cname = cell["name"]
        print(f"\n######## CELL {cname}  identity_scale="
              f"{cell['identity_scale']} geometry_scale="
              f"{cell['geometry_scale']} steps={cell['num_steps']} ########")
        # gamma passed explicitly (deviation 1 in the module docstring): a no-op
        # for this arm (gamma resolves to 1.0 == the sampler's own default) but it
        # keeps the sampler matched to the training schedule if one is enabled.
        sampler = EulerSampler(num_steps=cell["num_steps"], gamma=gamma)

        cell_entry = dict(cell)
        cell_entry["samples"] = []
        row_tensors = []
        row_labels = []

        for si in SAMPLE_INDICES:
            s = samples[si]
            idv = s["identity_emb"].unsqueeze(0).to(device)
            geo = s["geometry_emb"].unsqueeze(0).to(device)
            torch.manual_seed(args.seed)     # noise-matched: same draw in every
            with torch.no_grad():             # cell AND every sample index
                o = sampler.sample(
                    model=model,
                    shape=shape,
                    identity_emb=idv,
                    geometry_emb=geo,
                    device=device,
                    text_scale=cell["identity_scale"],
                    dino_scale=cell["geometry_scale"],
                    prediction_type=prediction_type,
                )
            if latent_space:
                img_t = decode_latents(vae, o)[0]        # (3, H*8, W*8) [-1,1]
            else:
                img_t = o[0].clamp(0, 1) * 2 - 1          # (3, H, W) [-1,1]

            panel_path = out / f"s{si:02d}_{cname}.png"
            tensor_to_pil(img_t).save(panel_path)
            panels_total += 1

            face_entry = {"face_detected": None, "det_conf": None,
                          "n_keypoints": None}
            if det is not None:
                m = detect_face(det, panel_path)
                if m is None:
                    no_face_total += 1
                    face_entry["face_detected"] = False
                    print(f"  s{si:02d}: no face detected "
                          f"({panel_path.name})", flush=True)
                else:
                    conf, n_kp = m
                    face_entry["face_detected"] = True
                    face_entry["det_conf"] = conf
                    face_entry["n_keypoints"] = n_kp
                    print(f"  s{si:02d}: face conf={conf:.3f} kp={n_kp} "
                          f"({panel_path.name})", flush=True)
            else:
                print(f"  s{si:02d}: rendered ({panel_path.name})", flush=True)

            cell_entry["samples"].append({
                "sample_idx": si,
                "image_id": str(s["image_id"]),
                "panel_path": rel_to_repo(panel_path),
                **face_entry,
            })
            # CPU storage (deviation 3): keeps VRAM flat across a 6-cell grid.
            row_tensors.append(img_t.detach().cpu())
            row_labels.append(f"s{si} {cname}")

        cell_entry["no_face_panels"] = (
            None if det is None else sum(
                1 for e in cell_entry["samples"] if e["face_detected"] is False
            )
        )
        cell_entry["panels"] = len(cell_entry["samples"])

        # One contact sheet per setting: the sample indices side by side.
        sheet = grid([row_tensors], labels=[row_labels])
        sheet_path = out / (f"sheet_{cname}"
                            f"_cols=samples[{','.join(str(i) for i in SAMPLE_INDICES)}]"
                            f".png")
        sheet.save(sheet_path)
        cell_entry["sheet_path"] = rel_to_repo(sheet_path)
        print(f"  SHEET -> {sheet_path}")

        if det is not None:
            print(f"  no-face panels: {cell_entry['no_face_panels']}"
                  f"/{cell_entry['panels']}")

        results["cells"].append(cell_entry)
        grid_tensors.append(row_tensors)
        grid_labels.append(row_labels)
        del sampler, row_tensors
        torch.cuda.empty_cache()

    # Cross-setting comparison sheet: one row per sample index, one column per
    # cell, in --cells order (the same order used by every filename above).
    comparison_rows = [
        [grid_tensors[ci][ri] for ci in range(len(cells))]
        for ri in range(len(SAMPLE_INDICES))
    ]
    comparison_labels = [
        [grid_labels[ci][ri] for ci in range(len(cells))]
        for ri in range(len(SAMPLE_INDICES))
    ]
    comparison = grid(comparison_rows, labels=comparison_labels)
    comparison_path = out / (
        f"comparison_rows=samples[{','.join(str(i) for i in SAMPLE_INDICES)}]"
        f"_cols=[{','.join(c['name'] for c in cells)}].png"
    )
    comparison.save(comparison_path)
    results["comparison_sheet"] = rel_to_repo(comparison_path)
    print(f"\nCOMPARISON SHEET -> {comparison_path}")

    results["panels_total"] = panels_total
    results["no_face_panels_total"] = no_face_total
    if det is None:
        results["notes"].append(
            "--no-face-detect: no face detection was run, so per-panel face data "
            "is null (not zero) and no no-face count is claimed."
        )
    elif no_face_total:
        results["notes"].append(
            f"{no_face_total}/{panels_total} panels had no detectable face. This "
            f"is a detector reading on a 1024px render, not a quality metric on "
            f"its own; it is recorded per panel and no value was substituted."
        )
    if latent_space:
        results["notes"].append(
            f"latent-space arm: latents sampled at {in_channels}ch x {spatial}x"
            f"{spatial} and VAE-decoded to RGB via production/sample.py "
            f"decode_latents."
        )
    results["notes"].append(
        "Every panel in this run shares one initial latent draw "
        f"(seed {args.seed}); cells differ only in identity guidance, geometry "
        "guidance and step count."
    )

    # ---------------- summary table ----------------
    print("\n" + "=" * 72)
    print(f"SUMMARY -- guidance/steps grid, samples {SAMPLE_INDICES}, "
          f"seed {args.seed}")
    print("=" * 72)
    print(f"{'cell':24s} {'id':>5s} {'geo':>5s} {'steps':>5s} "
          f"{'panels':>6s} {'noface':>7s} {'sheet':>8s}")
    for cell_entry in results["cells"]:
        nf = ("n/a" if cell_entry["no_face_panels"] is None
              else (f"{cell_entry['no_face_panels']}"
                    f"/{cell_entry['panels']}"))
        print(f"{cell_entry['name']:24s} {cell_entry['identity_scale']:>5g} "
              f"{cell_entry['geometry_scale']:>5g} {cell_entry['num_steps']:>5d} "
              f"{cell_entry['panels']:>6d} {str(nf):>7s} "
              f"{'yes' if cell_entry.get('sheet_path') else 'no':>8s}")
    print(f"\ncontrol cell: {results['control_cell']} (arm config default; the "
          f"setting every previously judged image used)")
    print(f"panels={panels_total}  no_face={no_face_total if det is not None else 'n/a'}")

    results_path = out / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results.json -> {results_path}")
    print("\nDONE. This probe renders only; the readout is visual judgment plus "
          "the recorded no-face counts.")


if __name__ == "__main__":
    main()
