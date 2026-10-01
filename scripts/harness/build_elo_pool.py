#!/usr/bin/env python3
"""Build Elo-picker pools from a trajectory arm's renders -- ONE POOL PER VALIDATION INDEX.

WHY ONE POOL PER INDEX
    `validation/step*/reconstruction/recon_<idx>.png` is the SAME validation sample (same
    identity, same conditioning, same noise) rendered at every checkpoint. Pooling several
    indices into one pool (or using sampled renders from different identities) makes Elo rank
    partly *which person looks better* rather than *which checkpoint renders better* -- an
    identity confound that would silently corrupt the ordering. One index = one pool, so every
    comparison inside a pool is intra-sample, and repeated indices give independent rankings of
    the SAME checkpoints (robustness across identities).

    The elo-picker has no cross-project leaderboard by design (ELO is pool-relative), so the
    per-index pools are separate projects; agreement ACROSS pools is the robustness evidence.

MULTIPLE RUN DIRS
    A continuation arm (ECR) resumes a parent arm (EIR), so its trajectory spans two run dirs:
    the origin checkpoint's renders live in the PARENT's run dir, the continuation's in its own.
    Pass every run dir with `--run-dirs`; each checkpoint resolves from the first dir that holds
    its render. The manifest records which dir each item came from.

USAGE
    # parent arm (single run dir)
    .venv/bin/python scripts/harness/build_elo_pool.py \\
        --prefix eir-traj --run-dirs experiments/eidolon-identity-renderer/runs/2026-09-25_0734 \\
        --indices 0 4 8 --checkpoints 1000 2000 ... 10000
    # continuation arm (parent origin + own segments)
    .venv/bin/python scripts/harness/build_elo_pool.py \\
        --prefix ecr-traj \\
        --run-dirs experiments/eidolon-identity-renderer/runs/2026-09-25_0734 \\
                   experiments/eir-cosine-restart/runs/2026-09-30_0950 \\
        --indices 0 4 --checkpoints 10000 10500 ... 15000
    then ingest each printed folder in the elo-picker (Project + IngestImages).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS = [REPO / "experiments" / "eidolon-identity-renderer" / "runs" / "2026-09-25_0734"]
DEFAULT_PREFIX = "eir-traj"
OUT_ROOT = Path("/mnt/nas-ai-models/training-data/prx-tg/diagnostics/"
                "deficit-2026-09-28/elo_pools")
DEFAULT_CHECKPOINTS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
DEFAULT_INDICES = [0, 4, 8]


def resolve(index: int, checkpoint: int, runs: list[Path]) -> Path | None:
    """First run dir that holds this (index, checkpoint) render, else None."""
    for run in runs:
        src = run / "validation" / f"step{checkpoint:07d}" / "reconstruction" / f"recon_{index:05d}.png"
        if src.is_file():
            return src
    return None


def build(index: int, checkpoints: list[int], runs: list[Path], prefix: str) -> dict:
    """Copy one index's render across checkpoints into a self-describing pool folder."""
    pool = f"{prefix}-i{index:02d}"
    folder = OUT_ROOT / pool
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for ck in checkpoints:
        src = resolve(index, ck, runs)
        if src is None:
            print(f"  MISSING i{index:02d} step{ck} in any of {len(runs)} run dir(s)")
            continue
        # Name encodes index AND checkpoint so the elo-picker leaderboard is readable, and so
        # a ranked image can be traced back to its checkpoint without a side table.
        dest = folder / f"i{index:02d}_step{ck:05d}.png"
        shutil.copy2(src, dest)
        entries.append({"pool_file": dest.name, "checkpoint": ck,
                        "source": str(src), "recon_index": index})
    # The pool must contain exactly the checkpoints asked for, or the trajectory has a hole and
    # every aggregate (Spearman, split test) is computed over a different grid than declared.
    if len(entries) != len(checkpoints):
        raise SystemExit(f"ABORT {pool}: wanted {len(checkpoints)} items, resolved {len(entries)}")
    # Stale files from a previous, wider grid would silently join the pool (ingest globs the
    # folder), so refuse to leave any render in the pool dir that is not a declared entry.
    declared = {e["pool_file"] for e in entries}
    stale = [p.name for p in folder.glob("*.png") if p.name not in declared]
    if stale:
        raise SystemExit(f"ABORT {pool}: stale renders present, remove by hand: {sorted(stale)}")
    manifest = {
        "pool": pool,
        "run_dirs": [str(r) for r in runs],
        "recon_index": index,
        "checkpoints": checkpoints,
        "note": ("One validation index, all checkpoints. Every pair in this pool is "
                 "INTRA-SAMPLE: same identity, same conditioning, same initial noise -- so the "
                 "Elo order is about render quality, not identity preference."),
        "n_images": len(entries),
        "entries": entries,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  {pool}: {len(entries)} images -> {folder}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", default=DEFAULT_INDICES)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--run-dirs", type=Path, nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = ap.parse_args()
    runs = [p if p.is_absolute() else (REPO / p) for p in args.run_dirs]
    missing = [str(r) for r in runs if not r.is_dir()]
    if missing:
        print(f"ABORT: run dir(s) missing: {missing}")
        return 1
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"building {len(args.indices)} pool(s) x {len(args.checkpoints)} checkpoints "
          f"from {len(runs)} run dir(s)")
    for idx in args.indices:
        build(idx, args.checkpoints, runs, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
