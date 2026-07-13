"""Build contextual features for every detection in the AOI5 catalog. Resumable.

Reads out/aoi5/catalog/detections.csv (read-only), reconstructs the per-(cycle,pass,kind)
eddy index (cumcount -- verified to match the labels CSV keying 805/805), then per pass
opens the granule once and computes the front/ADT features (src/swot_eddy/features.py).
Appends to:
  out/aoi5/features/features.csv        one row per detection (catalog key + features)
  out/aoi5/features/features_done.csv   per-pass manifest (resume marker)

Usage:
  .venv/bin/python scripts/build_features.py                # everything not yet done
  .venv/bin/python scripts/build_features.py --cycles 1 2   # subset
  .venv/bin/python scripts/build_features.py --redo         # rebuild from scratch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swot_eddy.config import load_config
from swot_eddy import features as ft

KEY_COLS = ["cycle", "pass", "kind", "eddy"]
PASS_COLS = ["date", "polarity", "lon", "lat", "radius_km", "radius_eff_km",
             "amplitude_m", "gap_frac", "nadir_frac", "center_in_nadir"]


def _append(path: Path, rows: list, cols: list):
    if not rows:
        return
    pd.DataFrame(rows, columns=cols).to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", nargs="*", type=int)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg["out_root"]) / "aoi5" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_csv, manifest = out_dir / "features.csv", out_dir / "features_done.csv"
    if args.redo:
        for p in (feat_csv, manifest):
            p.unlink(missing_ok=True)

    cat = pd.read_csv(Path(cfg["out_root"]) / "aoi5" / "catalog" / "detections.csv",
                      dtype={"cycle": str, "pass": str})
    cat["eddy"] = cat.groupby(["cycle", "pass", "kind"]).cumcount()
    if args.cycles:
        want = {f"cycle_{n:03d}" for n in args.cycles}
        cat = cat[cat["cycle"].isin(want)]

    done = set()
    if manifest.exists():
        m = pd.read_csv(manifest, dtype={"cycle": str, "pass": str})
        done = set(zip(m["cycle"], m["pass"]))

    groups = list(cat.groupby(["cycle", "pass"], sort=True))
    todo = [(k, g) for k, g in groups if k not in done]
    print(f"{len(cat)} detections over {len(groups)} passes; {len(done)} passes done, {len(todo)} left")

    cols = KEY_COLS + PASS_COLS + ft.FEATURE_COLS
    data_root = Path(cfg["data_root"])
    for n, ((cyc, pnum), grp) in enumerate(todo, 1):
        f = ft.find_granule(data_root, cyc, pnum)
        if f is None:
            _append(manifest, [{"cycle": cyc, "pass": pnum, "n_rows": 0,
                                "status": "no_granule"}],
                    ["cycle", "pass", "n_rows", "status"])
            print(f"  WARN {cyc}/{pnum}: granule not found")
            continue
        try:
            frows = ft.pass_features(grp, f, cfg)
        except Exception as e:
            _append(manifest, [{"cycle": cyc, "pass": pnum, "n_rows": 0,
                                "status": f"error:{type(e).__name__}"}],
                    ["cycle", "pass", "n_rows", "status"])
            print(f"  ERROR {cyc}/{pnum}: {type(e).__name__}: {e}")
            continue
        rows = [{**det, **fr} for det, fr in
                zip(grp[KEY_COLS + PASS_COLS].to_dict("records"), frows)]
        _append(feat_csv, rows, cols)
        _append(manifest, [{"cycle": cyc, "pass": pnum, "n_rows": len(rows), "status": "ok"}],
                ["cycle", "pass", "n_rows", "status"])
        if n % 25 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {cyc}/{pnum}: {len(rows)} rows")

    print(f"done -> {feat_csv}")


if __name__ == "__main__":
    main()
