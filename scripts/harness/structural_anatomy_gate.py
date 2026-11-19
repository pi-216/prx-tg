#!/usr/bin/env python3
"""Structural face-anatomy gate: deterministic detection of absent/distorted
facial anatomy that pairwise Elo and the G0 spectral bands miss.

Evaluates one render set, in order of preference:

    full path    — a directory with `images/` PNGs and `masks/` bool `.npy`
                   skin masks of matching basenames (measured directly)
    derived path — a directory with only the tracked `gates.jsonl` aggregate:
                   detection_attrition = 1 - n_masked / n_full, the ratio of
                   mask-gated to full-frame gate populations

Three structural criteria:

    S1  detection_attrition    fraction of renders with NO usable detection
    S2  skin_mask_pass_rate    fraction of renders with a non-degenerate mask
                               (coverage >= min_mask_coverage)   [full path only]
    S3  geometry_pass_rate     fraction of renders whose mask has a plausible
                               bounding-box fill ratio (min_bbox_fill..max_bbox_fill)
                               [full path only]

A set PASSES when every evaluated axis is within its threshold; a FAIL result
names the failed axes.

USAGE:
    .venv/bin/python3 scripts/harness/structural_anatomy_gate.py \
        --set research/avenues/gates/deficit-2026-09-28/eir-recon [--out gate.json]

    .venv/bin/python3 scripts/harness/structural_anatomy_gate.py --calibrate \
        [--root research/avenues/gates/deficit-2026-09-28] \
        [--out research/avenues/structural_anatomy_calibration.json]

EXIT CODES:
    0  set mode: verdict PASS · calibrate mode: separation proven
       (real + vae-ceiling PASS, eir-recon FAIL)
    1  set mode: verdict FAIL · calibrate mode: separation NOT proven
    2  usage / missing input
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEFICIT_ROOT = REPO_ROOT / "research" / "avenues" / "gates" / "deficit-2026-09-28"
REQUIRED_LEGS = ("real", "vae-ceiling", "eir-recon")


@dataclass(frozen=True)
class Thresholds:
    max_detection_attrition: float = 0.05
    min_skin_mask_pass_rate: float = 0.90
    min_geometry_pass_rate: float = 0.90
    min_mask_coverage: float = 0.002
    min_bbox_fill: float = 0.20
    max_bbox_fill: float = 0.90


def _align_mask(mask: np.ndarray, shape: tuple) -> np.ndarray:
    if mask.shape[:2] == shape:
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    resampling = getattr(Image, "Resampling", Image).NEAREST
    img = img.resize((shape[1], shape[0]), resampling)
    return np.asarray(img) > 127


def _image_paths(set_dir: Path) -> list:
    return sorted((set_dir / "images").glob("*.png"))


FULL_GATES = (
    "g0a_sensor_noise_floor_full",
    "g0c_skin_texture_energy_full",
    "g0d_local_contrast_full_p5",
    "g0d_local_contrast_full_p50",
    "g0d_local_contrast_full_p95",
)
MASKED_GATES = ("g0a_sensor_noise_floor", "g0c_skin_texture_energy", "g0d_local_contrast_p5")


def _derive_attrition(gates_path: Path) -> tuple:
    n_full = 0
    n_masked = 0
    with gates_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gate = row.get("gate_id")
            n = int(row.get("n", 0))
            if gate in FULL_GATES:
                n_full = max(n_full, n)
            elif gate in MASKED_GATES:
                n_masked = max(n_masked, n)
    if n_full == 0:
        raise ValueError(f"{gates_path}: no full-frame gate population found")
    return n_full, n_masked


def _evaluate_derived(set_dir: Path, thresholds: Thresholds) -> dict:
    n_images, n_masks = _derive_attrition(set_dir / "gates.jsonl")
    metrics = {
        "n_images": n_images,
        "n_masks": n_masks,
        "detection_attrition": (n_images - n_masks) / n_images,
        "skin_mask_pass_rate": None,
        "geometry_pass_rate": None,
        "mask_coverage": None,
        "bbox_fill": None,
    }
    return _finalize(set_dir, "derived", thresholds, metrics)


def _finalize(set_dir: Path, path_mode: str, thresholds: Thresholds, metrics: dict) -> dict:
    failed = []
    attrition = metrics["detection_attrition"]
    if attrition > thresholds.max_detection_attrition:
        failed.append("detection_attrition")
    skin = metrics["skin_mask_pass_rate"]
    if skin is not None and skin < thresholds.min_skin_mask_pass_rate:
        failed.append("skin_mask_pass_rate")
    geometry = metrics["geometry_pass_rate"]
    if geometry is not None and geometry < thresholds.min_geometry_pass_rate:
        failed.append("geometry_pass_rate")
    return {
        "set": _recorded_root(set_dir),
        "path": path_mode,
        "thresholds": asdict(thresholds),
        "metrics": metrics,
        "failed_axes": failed,
        "verdict": "PASS" if not failed else "FAIL",
    }


def evaluate_set(set_dir: Path, thresholds: Thresholds = Thresholds()) -> dict:
    set_dir = Path(set_dir)
    if (set_dir / "images").is_dir() and (set_dir / "masks").is_dir():
        return _evaluate_full(set_dir, thresholds)
    if (set_dir / "gates.jsonl").is_file():
        return _evaluate_derived(set_dir, thresholds)
    raise FileNotFoundError(
        f"{set_dir}: needs images/ + masks/ or a gates.jsonl aggregate"
    )


def _evaluate_full(set_dir: Path, thresholds: Thresholds) -> dict:
    images = _image_paths(set_dir)
    if not images:
        raise FileNotFoundError(f"{set_dir}: images/ contains no PNGs")

    n = len(images)
    missing = 0
    skin_ok = 0
    geometry_ok = 0
    coverages = []
    fills = []
    for path in images:
        mask_path = set_dir / "masks" / f"{path.stem}.npy"
        if not mask_path.is_file():
            missing += 1
            continue
        mask = np.load(mask_path).astype(bool)
        with Image.open(path) as handle:
            height, width = handle.size[1], handle.size[0]
        mask = _align_mask(mask, (height, width))
        pixels = int(mask.sum())
        coverage = pixels / mask.size
        coverages.append(coverage)
        if coverage < thresholds.min_mask_coverage or pixels == 0:
            continue
        skin_ok += 1
        ys, xs = np.where(mask)
        bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        fill = pixels / bbox_area
        fills.append(fill)
        if thresholds.min_bbox_fill <= fill <= thresholds.max_bbox_fill:
            geometry_ok += 1

    def dist(values):
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return {
            "min": float(arr.min()),
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    metrics = {
        "n_images": n,
        "n_masks": n - missing,
        "detection_attrition": missing / n,
        "skin_mask_pass_rate": skin_ok / n,
        "geometry_pass_rate": geometry_ok / n,
        "mask_coverage": dist(coverages),
        "bbox_fill": dist(fills),
    }
    return _finalize(set_dir, "full", thresholds, metrics)


def _recorded_root(root: Path) -> str:
    try:
        return str(Path(root).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(root)


def separation_proven(legs: dict) -> bool:
    present = all(leg in legs for leg in REQUIRED_LEGS)
    if not present:
        return False
    return (
        legs["real"]["verdict"] == "PASS"
        and legs["vae-ceiling"]["verdict"] == "PASS"
        and legs["eir-recon"]["verdict"] == "FAIL"
    )


def calibrate(root: Path, thresholds: Thresholds = Thresholds()) -> dict:
    root = Path(root)
    legs = {}
    for leg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        has_renders = (leg_dir / "images").is_dir() and (leg_dir / "masks").is_dir()
        if not has_renders and not (leg_dir / "gates.jsonl").is_file():
            continue
        legs[leg_dir.name] = evaluate_set(leg_dir, thresholds)
    return {
        "gate": "structural_anatomy_gate",
        "root": _recorded_root(root),
        "thresholds": asdict(thresholds),
        "required_legs": list(REQUIRED_LEGS),
        "separation_proven": separation_proven(legs),
        "legs": [dict(name=name, **result) for name, result in sorted(legs.items())],
    }


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_dir", type=Path, help="render set with images/ and masks/")
    ap.add_argument("--calibrate", action="store_true", help="evaluate the on-disk reference legs and assert separation")
    ap.add_argument("--root", type=Path, default=DEFAULT_DEFICIT_ROOT, help="reference root for --calibrate")
    ap.add_argument("--out", type=Path, help="write the result JSON here")
    args = ap.parse_args(argv)

    if args.set_dir is None and not args.calibrate:
        ap.error("one of --set or --calibrate is required")

    try:
        if args.calibrate:
            payload = calibrate(args.root)
            ok = payload["separation_proven"]
        else:
            payload = evaluate_set(args.set_dir)
            ok = payload["verdict"] == "PASS"
    except (FileNotFoundError, OSError) as exc:
        print(f"structural_anatomy_gate: {exc}", file=sys.stderr)
        return 2

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    verdict = payload.get("verdict") or (
        "SEPARATION PROVEN" if payload.get("separation_proven") else "SEPARATION NOT PROVEN"
    )
    print(f"structural_anatomy_gate: {verdict}")
    for line in payload.get("failed_axes", []):
        print(f"  failed axis: {line}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
