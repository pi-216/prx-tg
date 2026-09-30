#!/usr/bin/env python3
"""Read the EIR-trajectory Elo pools from the elo-picker and analyse the checkpoint ordering.

The question: does render quality improve with training, in a way a human can see?
Each pool is ONE validation index (one identity) at all 10 checkpoints, so every pair inside a
pool is intra-sample and the Elo order is about render quality rather than identity preference.

WHAT THIS REPORTS, per pool
  - the checkpoint ordering by Elo, with each image's RD and games
  - Spearman rho between checkpoint step and Elo (does quality rise with training?)
  - separability of the extremes: z = (elo_top - elo_bottom) / sqrt(rd_top^2 + rd_bottom^2).
    Elo POINTS are not evidence; a gap smaller than the combined RD is not separable.
  - the tool's own meters (phase / trusted / snr / reliability), read from /projects/{id}.json

And across pools: the rank each checkpoint holds in each pool. Agreement across identities is
the robustness evidence; disagreement means the effect does not generalise.

INTERPRETATION RULE (kept explicit, because it is the trap this project keeps hitting):
a pool that is untrusted or low-SNR means the INSTRUMENT could not resolve a difference -- not
that no difference exists. The two are different findings and are labelled differently.

USAGE
    .venv/bin/python scripts/harness/elo_results.py [--port 3010] [--host 192.168.86.49]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "experiment-configs" / "eidolon-identity-renderer" / "audit" / "elo_trajectory"
POOLS = {5: "i00", 6: "i04", 7: "i08", 8: "i12", 9: "i16", 10: "i20"}
STEP_RE = re.compile(r"step(\d+)\.png$")


def fetch(host: str, port: int, project_id: int) -> dict:
    url = f"http://{host}:{port}/projects/{project_id}.json"
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode())


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, computed directly (no scipy)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                      # average ranks for ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.86.49")
    ap.add_argument("--port", type=int, default=3010)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {"source": f"elo-picker :{args.port}", "pools": {}, "ranks_by_checkpoint": {}}
    md = ["# EIR trajectory — human Elo ranking of the checkpoints", "",
          "Source: elo-picker pools, one validation index per pool (intra-sample: every pair is",
          "the same identity at two checkpoints). Raw payloads saved beside this file.", ""]

    per_pool_ranks: dict[str, dict[int, int]] = {}
    for pid, slug in sorted(POOLS.items()):
        data = fetch(args.host, args.port, pid)
        (OUT_DIR / f"project_{pid}_{slug}.json").write_text(json.dumps(data, indent=2))

        rows = []
        for r in data.get("image_rows", []):
            name = Path(r["path"]).name
            m = STEP_RE.search(name)
            if not m:
                continue
            rows.append({"file": name, "checkpoint": int(m.group(1)),
                         "elo": r["elo"], "rd": r["rd"], "games": r["games_played"]})
        rows.sort(key=lambda x: -x["elo"])
        spec = data.get("spectrum", {})
        ranks = {r["checkpoint"]: i + 1 for i, r in enumerate(rows)}
        per_pool_ranks[slug] = ranks

        top, bot = rows[0], rows[-1]
        gap = top["elo"] - bot["elo"]
        zsep = gap / ((top["rd"] ** 2 + bot["rd"] ** 2) ** 0.5) if rows else float("nan")
        rho = spearman([r["checkpoint"] for r in rows], [r["elo"] for r in rows])
        total_games = sum(r["games"] for r in rows) // 2

        summary["pools"][slug] = {
            "project_id": pid, "n": len(rows), "comparisons": total_games,
            "spectrum": spec, "order_top_to_bottom": [r["checkpoint"] for r in rows],
            "spearman_ckpt_vs_elo": rho, "gap_top_bottom": gap, "z_separable": zsep,
            "rows": rows,
        }
        md += [f"## Pool `{slug}` (project {pid}) — {len(rows)} images, {total_games} comparisons",
               "",
               f"- tool meters: phase {spec.get('phase')} · covered {spec.get('covered')}/"
               f"{spec.get('total')} · trusted {spec.get('trusted')}/{spec.get('total')} · "
               f"snr {spec.get('snr')} · reliability {spec.get('reliability')}",
               f"- Spearman(checkpoint, Elo) = **{rho:+.2f}**  (positive = quality rises with training)",
               f"- top-vs-bottom gap {gap:.0f} Elo, **z = {zsep:.2f}** "
               f"({'separable' if zsep >= 2 else 'NOT separable — gap is within the RDs'})",
               "",
               "| rank | checkpoint | Elo | RD | games |", "|---|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            md.append(f"| {i} | {r['checkpoint']} | {r['elo']:.0f} | {r['rd']:.0f} | {r['games']} |")
        md.append("")

    # cross-pool: where does each checkpoint place, in each identity?
    md += ["## Cross-pool agreement", "",
           "Rank of each checkpoint in each identity's pool (lower = better render).", "",
           "| checkpoint | " + " | ".join(per_pool_ranks) + " | mean rank |",
           "|---" * (len(per_pool_ranks) + 2) + "|"]
    all_ckpts = sorted({c for r in per_pool_ranks.values() for c in r})
    for c in all_ckpts:
        ranks = [per_pool_ranks[s].get(c) for s in per_pool_ranks]
        got = [x for x in ranks if x]
        mean_rank = sum(got) / len(got) if got else float("nan")
        summary["ranks_by_checkpoint"][c] = {"ranks": ranks, "mean_rank": mean_rank}
        md.append(f"| {c} | " + " | ".join(str(x) if x else "—" for x in ranks)
                  + f" | {mean_rank:.1f} |")
    md.append("")

    # EXACT TEST on the split. A per-pool ranking of 10 items is a weak instrument on its own
    # (each pool's SNR is ~1.2 against the tool's 5.0 gate), but the pools are INDEPENDENT
    # replicates of the same 10 checkpoints on different identities, so agreement across them is
    # the real evidence -- and the extreme version of "agreement" is exactly computable. If the
    # early checkpoints really render worse, they should occupy the BOTTOM 5 ranks of every
    # pool; a random permutation puts a given 5-item set in the bottom 5 with probability
    # 1/C(10,5) = 1/252.
    split = 5000
    early = sorted(c for c in all_ckpts if c <= split)
    late = sorted(c for c in all_ckpts if c > split)
    if early and late:
        perfect = 0
        md += [f"## Split test: early (≤{split}) vs late (>{split})", "",
               "For each identity pool: how many (early, late) pairs are inverted (early ranked",
               "better). Perfect separation = the early checkpoints hold the bottom ranks.", "",
               "| pool | inversions | early ranks | late ranks | separation |",
               "|---|---|---|---|---|"]
        for slug, ranks in per_pool_ranks.items():
            e = sorted(ranks[c] for c in early if c in ranks)
            l = sorted(ranks[c] for c in late if c in ranks)
            inv = sum(1 for a in e for b in l if a < b)   # early ranked BETTER than late
            # Lower rank number = better, so separation means every early rank is NUMERICALLY
            # GREATER than every late rank. (An earlier version wrote max(e) < min(l) and
            # reported 0/6 perfect on pools whose ranks were [6..10] vs [1..5] -- the check
            # contradicted its own inversion count, which is what caught it.)
            is_perfect = bool(e and l) and min(e) > max(l)
            assert is_perfect == (inv == 0), (slug, e, l, inv)
            perfect += int(is_perfect)
            md.append(f"| {slug} | {inv} / {len(e) * len(l)} | {e} | {l} | "
                      f"{'PERFECT' if is_perfect else 'overlap'} |")
        n_pools = len(per_pool_ranks)
        k = len(early)
        # P(a fixed 5-subset holds the bottom 5 of 10) = 1 / C(10,5)
        p_one = 1.0 / (math.comb(len(all_ckpts), k) if all_ckpts else 1)
        p_atleast = sum(math.comb(n_pools, j) * p_one ** j * (1 - p_one) ** (n_pools - j)
                        for j in range(perfect, n_pools + 1))
        md += ["", f"**{perfect} of {n_pools} pools separate perfectly.** Under the null (random "
                   f"ordering) a fixed {k}-item set lands in the bottom {k} with p = 1/{int(1/p_one)} "
                   f"per pool, so ≥{perfect}/{n_pools} perfect gives **p ≈ {p_atleast:.2g}**."]
        summary["split_test"] = {"split": split, "early": early, "late": late,
                                 "perfect_pools": perfect, "n_pools": n_pools,
                                 "p_at_least_this_many": p_atleast}

    # EXPLORATORY: does our one automatic candidate -- detection attrition (renders where
    # Sapiens2 finds no face+neck at all) -- track the HUMAN ranking? This is a lead, not
    # evidence: 10 checkpoints, one metric, one run. It answers "do we have a cheap proxy?"
    # with a number instead of a hope.
    gate = REPO / "research" / "avenues" / "gates" / "deficit-2026-09-28" / "eir-recon"
    if (gate / "images").is_dir():
        panels: dict[int, int] = {}
        masks: dict[int, int] = {}
        for p in (gate / "images").glob("step*.png"):
            c = int(p.name.split("_")[0].replace("step", ""))
            panels[c] = panels.get(c, 0) + 1
        for p in (gate / "masks").glob("step*.npy"):
            c = int(p.name.split("_")[0].replace("step", ""))
            masks[c] = masks.get(c, 0) + 1
        steps = sorted(panels)
        att = [panels[s] - masks.get(s, 0) for s in steps]
        ranks = [summary["ranks_by_checkpoint"].get(s, {}).get("mean_rank") for s in steps]
        if all(r is not None for r in ranks):
            rho_att = spearman([float(a) for a in att], [float(r) for r in ranks])
            md += ["", "## Exploratory: does detection attrition track the human ranking?", "",
                   "Attrition = renders of that checkpoint in which Sapiens2 seg2 finds no usable",
                   "skin mask (no face+neck). Lower rank = better render, so a USEFUL proxy needs a",
                   "POSITIVE correlation with mean rank (more attrition -> worse).", "",
                   "| checkpoint | panels | with mask | attrition | human mean rank |",
                   "|---|---|---|---|---|"]
            for s, a, r in zip(steps, att, ranks):
                md.append(f"| {s} | {panels[s]} | {masks.get(s, 0)} | {a} | {r:.1f} |")
            md += ["", f"Spearman(attrition, human mean rank) = **{rho_att:+.2f}** "
                       f"over {len(steps)} checkpoints. EXPLORATORY: n=10, one run, one metric -- "
                       f"a lead to test on the next run, not a validated proxy."]
            summary["attrition_vs_human"] = {"spearman": rho_att, "attrition": att,
                                             "mean_rank": ranks, "steps": steps}

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[elo_results] wrote {OUT_DIR/'RESULTS.md'}, summary.json, and the raw payloads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
