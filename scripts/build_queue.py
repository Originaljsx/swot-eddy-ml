"""Build a front-sorted labeling queue from the feature table. Read-only inputs.

Unlabeled kind=keep detections, likely-frontal first: 80% of the budget = eddies within
--near-km of the ADT front (frontal eddies hug the line: nearest 25% of detections held
59% of hand-labeled frontals), taken by WHOLE PASSES densest-first so each ~15 s
pass-load in the labeler yields many eddies (~10/load vs ~1.4 if sorted purely by
distance). The other 20% = random draws from everything else (keeps coastal/open-ocean
labels and seasonal coverage growing; unbiased audit sample).

  out/aoi5/queue/queue_NNN.csv   cycle, pass, kind, eddy, dist_front_adt_km, queue_reason

Usage:
  .venv/bin/python scripts/build_queue.py [--size 400] [--seed 0]
Label with:
  CatalogLabeler(cfg, queue="out/aoi5/queue/queue_001.csv").show()
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swot_eddy.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=400,
                    help="queue budget; 0 = ALL near-front eddies, no audit sample")
    ap.add_argument("--near-km", type=float, default=50.0,
                    help="|front dist| defining the near-front slice")
    ap.add_argument("--cyclonic", action="store_true",
                    help="near-front slice: cyclonic eddies only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    root = Path(cfg["out_root"]) / "aoi5"
    feat = pd.read_csv(root / "features" / "features.csv", dtype={"cycle": str, "pass": str})
    feat = feat[feat["kind"] == "keep"]

    labels_csv = root / "labels" / "eddy_labels_aoi5.csv"
    if labels_csv.exists():
        lab = pd.read_csv(labels_csv, dtype={"cycle": str, "pass": str})
        done = set(zip(lab["cycle"], lab["pass"], lab["kind"], lab["eddy"]))
        feat = feat[[k not in done for k in
                     zip(feat["cycle"], feat["pass"], feat["kind"], feat["eddy"])]]
    print(f"{len(feat)} unlabeled keep-detections available")

    feat = feat.copy()
    feat["abs_d"] = feat["dist_front_adt_km"].abs()
    # Near-front slice by WHOLE PASSES, densest first: each pass costs one detect+load in
    # the labeler (~15 s), so 10 near-front eddies from one pass beat 10 from 10 passes.
    near = feat[feat["abs_d"] < args.near_km]
    if args.cyclonic:
        near = near[near["polarity"] == "cyclonic"]
    if args.size == 0:                                 # everything near-front, no audit
        front, rand = near, near.iloc[:0]
    else:
        order = near.groupby(["cycle", "pass"]).size().sort_values(ascending=False)
        budget = int(args.size * 0.8)
        picked, n_front = [], 0
        for (cyc, pnum), cnt in order.items():
            if n_front >= budget:
                break
            picked.append((cyc, pnum))
            n_front += cnt
        front = near.merge(pd.DataFrame(picked, columns=["cycle", "pass"]), on=["cycle", "pass"])
        rest = feat.drop(front.index, errors="ignore")
        n_rand = min(args.size - len(front), len(rest))
        rand = rest.sample(n=n_rand, random_state=args.seed) if n_rand > 0 else rest.iloc[:0]

    q = pd.concat([front.assign(queue_reason="near_front"),
                   rand.assign(queue_reason="audit")])
    q = q.sort_values(["cycle", "pass", "kind", "abs_d"])
    q = q[["cycle", "pass", "kind", "eddy", "dist_front_adt_km", "queue_reason"]]

    out_dir = root / "queue"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 1 + max((int(p.stem.split("_")[1]) for p in out_dir.glob("queue_*.csv")), default=0)
    out = out_dir / f"queue_{n:03d}.csv"
    q.to_csv(out, index=False)
    print(f"wrote {out}: {len(q)} eddies over {q.groupby(['cycle','pass']).ngroups} passes "
          f"({(q.queue_reason == 'near_front').sum()} near-front + "
          f"{(q.queue_reason == 'audit').sum()} audit), "
          f"median |front dist| of near-front slice: "
          f"{q.loc[q.queue_reason == 'near_front', 'dist_front_adt_km'].abs().median():.0f} km")


if __name__ == "__main__":
    main()
