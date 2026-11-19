"""Unit and sensitivity tests for the structural face-anatomy gate (PRD-0001 T2)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "harness"))
import structural_anatomy_gate as sag  # noqa: E402

DEFICIT = ROOT / "research" / "avenues" / "gates" / "deficit-2026-09-28"
CALIBRATION = ROOT / "research" / "avenues" / "structural_anatomy_calibration.json"


def test_real_ffhq_passes():
    result = sag.evaluate_set(DEFICIT / "real")
    assert result["verdict"] == "PASS"
    assert result["metrics"]["detection_attrition"] == 0.0
    assert result["failed_axes"] == []


def test_vae_ceiling_passes():
    result = sag.evaluate_set(DEFICIT / "vae-ceiling")
    assert result["verdict"] == "PASS"
    assert result["metrics"]["detection_attrition"] == 0.0


def test_eir_recon_fails_on_detection_attrition():
    result = sag.evaluate_set(DEFICIT / "eir-recon")
    assert result["verdict"] == "FAIL"
    assert result["metrics"]["detection_attrition"] > 0.05
    assert "detection_attrition" in result["failed_axes"]


def test_derived_path_from_gates_jsonl(tmp_path):
    leg = tmp_path / "eir-recon"
    leg.mkdir()
    shutil.copy(DEFICIT / "eir-recon" / "gates.jsonl", leg / "gates.jsonl")
    result = sag.evaluate_set(leg)
    assert result["path"] == "derived"
    assert result["verdict"] == "FAIL"
    assert result["metrics"]["n_images"] == 160
    assert result["metrics"]["n_masks"] == 126
    assert abs(result["metrics"]["detection_attrition"] - 34 / 160) < 1e-9
    assert "detection_attrition" in result["failed_axes"]


def test_calibration_separates_real_and_vae_from_eir():
    assert CALIBRATION.is_file()
    calib = json.loads(CALIBRATION.read_text())
    verdicts = {leg["name"]: leg["verdict"] for leg in calib["legs"]}
    assert verdicts["real"] == "PASS"
    assert verdicts["vae-ceiling"] == "PASS"
    assert verdicts["eir-recon"] == "FAIL"
    eir = next(leg for leg in calib["legs"] if leg["name"] == "eir-recon")
    assert eir["metrics"]["detection_attrition"] > 0.05


def _ellipse_mask(size, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:size, 0:size]
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0


def _mixed_mask_fn(bad_positions, bad_fn):
    def fn():
        fn.index += 1
        if fn.index in bad_positions:
            return bad_fn()
        return _ellipse_mask(64, 32, 32, 14, 18)

    fn.index = -1
    return fn


def _write_set(tmp_path, n, mask_fn=None, missing=()):
    size = 64
    mask_fn = mask_fn or (lambda: _ellipse_mask(size, 32, 32, 14, 18))
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = np.full((size, size, 3), 120, dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(images / f"img{i:03d}.png")
        if i not in missing:
            np.save(masks / f"img{i:03d}.npy", mask_fn().astype(np.float32))
    return tmp_path


def test_clean_fixture_passes(tmp_path):
    result = sag.evaluate_set(_write_set(tmp_path, 20))
    assert result["verdict"] == "PASS"
    assert result["metrics"]["detection_attrition"] == 0.0


def test_attrition_above_five_percent_flips_fixture_to_fail(tmp_path):
    result = sag.evaluate_set(_write_set(tmp_path, 20, missing=(0, 1)))
    assert result["verdict"] == "FAIL"
    assert result["metrics"]["detection_attrition"] > 0.05
    assert "detection_attrition" in result["failed_axes"]


def test_collapsed_skin_coverage_flips_fixture_to_fail(tmp_path):
    def collapsed():
        m = np.zeros((64, 64), dtype=bool)
        m[30, 30] = True
        m[31, 31] = True
        m[32, 32] = True
        return m

    fn = _mixed_mask_fn((2, 7, 12, 17), collapsed)
    result = sag.evaluate_set(_write_set(tmp_path, 20, mask_fn=fn))
    assert result["verdict"] == "FAIL"
    assert "skin_mask_pass_rate" in result["failed_axes"]
    assert "detection_attrition" not in result["failed_axes"]


def test_degenerate_geometry_flips_fixture_to_fail(tmp_path):
    def scattered():
        m = np.zeros((64, 64), dtype=bool)
        for y in range(6, 58, 3):
            m[y, 6] = True
            m[y, 57] = True
        m[6, 30] = True
        return m

    fn = _mixed_mask_fn((2, 7, 12), scattered)
    result = sag.evaluate_set(_write_set(tmp_path, 20, mask_fn=fn))
    assert result["verdict"] == "FAIL"
    assert "geometry_pass_rate" in result["failed_axes"]
    assert "detection_attrition" not in result["failed_axes"]
    assert "skin_mask_pass_rate" not in result["failed_axes"]


def test_cli_set_mode_writes_verdict_json(tmp_path):
    set_dir = _write_set(tmp_path / "set", 10)
    out = tmp_path / "gate_out.json"
    rc = sag.main(["--set", str(set_dir), "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["verdict"] == "PASS"


def test_cli_calibrate_proves_separation(tmp_path):
    out = tmp_path / "calib.json"
    rc = sag.main(["--calibrate", "--root", str(DEFICIT), "--out", str(out)])
    assert rc == 0
    calib = json.loads(out.read_text())
    verdicts = {leg["name"]: leg["verdict"] for leg in calib["legs"]}
    assert verdicts["real"] == "PASS"
    assert verdicts["vae-ceiling"] == "PASS"
    assert verdicts["eir-recon"] == "FAIL"


def test_cli_calibrate_fails_when_separation_absent(tmp_path):
    partial = tmp_path / "only_real"
    partial.mkdir()
    (partial / "real").symlink_to(DEFICIT / "real", target_is_directory=True)
    rc = sag.main(["--calibrate", "--root", str(partial), "--out", str(tmp_path / "c.json")])
    assert rc != 0


def test_cli_calibrate_fails_when_all_legs_pass(tmp_path):
    root = tmp_path / "all_pass"
    root.mkdir()
    for leg in sag.REQUIRED_LEGS:
        source = DEFICIT / ("vae-ceiling" if leg == "eir-recon" else leg)
        target = root / leg
        target.mkdir()
        shutil.copy(source / "gates.jsonl", target / "gates.jsonl")
    rc = sag.main(["--calibrate", "--root", str(root), "--out", str(tmp_path / "c.json")])
    assert rc != 0
    calib = json.loads((tmp_path / "c.json").read_text())
    assert calib["separation_proven"] is False
    assert {leg["name"] for leg in calib["legs"]} == set(sag.REQUIRED_LEGS)
