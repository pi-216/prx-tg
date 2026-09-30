#!/usr/bin/env python3
"""Build Elo-picker pools from Arm EIR's trajectory renders -- ONE POOL PER VALIDATION INDEX.

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

USAGE
    .venv/bin/python scripts/harness/build_elo_pool.py [--indices 0 4 8] [--checkpoints 1000 ...]
    then ingest each printed folder in the elo-picker (Project + IngestImages).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RUN = REPO / "experiments" / "eidolon-identity-renderer" / "runs" / "2026-09-25_0734"
OUT_ROOT = Path("/mnt/nas-ai-models/training-data/prx-tg/diagnostics/"
                "deficit-2026-09-28/elo_pools")
DEFAULT_CHECKPOINTS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
DEFAULT_INDICES = [0, 4, 8]


def build(index: int, checkpoints: list[int]) -> dict:
    """Copy one index's render across checkpoints into a self-describing pool folder."""
    pool = f"eir-traj-i{index:02d}"
    folder = OUT_ROOT / pool
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for ck in checkpoints:
        src = RUN / "validation" / f"step{ck:07d}" / "reconstruction" / f"recon_{index:05d}.png"
        if not src.is_file():
            print(f"  MISSING {src}")
            continue
        # Name encodes index AND checkpoint so the elo-picker leaderboard is readable, and so
        # a ranked image can be traced back to its checkpoint without a side table.
        dest = folder / f"i{index:02d}_step{ck:05d}.png"
        shutil.copy2(src, dest)
        entries.append({"pool_file": dest.name, "checkpoint": ck,
                        "source": str(src), "recon_index": index})
    manifest = {
        "pool": pool,
        "run": str(RUN),
        "recon_index": index,
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
    args = ap.parse_args()
    if not RUN.is_dir():
        print(f"ABORT: run dir missing: {RUN}")
        return 1
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"building {len(args.indices)} pool(s) x {len(args.checkpoints)} checkpoints")
    for idx in args.indices:
        build(idx, args.checkpoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
