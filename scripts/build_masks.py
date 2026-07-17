"""Build U-Net training samples: SWOT pass -> input channels + frontal-eddy mask. Resumable.

One .npz per labeled pass in out/aoi5/training/:
  x     float16 (3, L, 128)  inpainted SSHA [m], inpainted ADT [m], |grad ADT| [m/km]
        (same preprocessing as detection, so inference on a raw granule matches training)
  y     uint8   (L, 128)     1 inside the PET speed contour of every hand-CONFIRMED
                             frontal eddy (labels CSV x contours.csv), else 0
  inp   uint8   (L, 128)     1 where the SSHA channel was biharmonic-inpainted
                             (optional loss down-weighting; invented pixels)
Manifest training_manifest.csv: cycle, pass, split, n_frontal, n_lines, status.
Temporal split (eddies persist across adjacent 21-day cycles -> no random split):
  train = cycles 1-30 (fit), val = 31-35 (model selection), cal = 36-42 (conformal
  calibration -- untouched by training or selection), test = 43-50 (final eval).

Known label caveat (documented, accepted for v1): PET-missed frontal eddies appear as
background in y. See docs/2026-07-12-frontal-prior-methods.md Sec. 5.

Usage:
  .venv/bin/python scripts/build_masks.py [--cycles 1 2] [--redo]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*partition.*will ignore the 'mask'.*")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swot_eddy.config import load_config
from swot_eddy import preprocess as pp
from swot_eddy.features import find_granule, grad_mag_per_km

MAN_COLS = ["cycle", "pass", "split", "n_frontal", "n_lines", "status"]


def split_of(cycle: str) -> str:
    n = int(cycle.split("_")[1])
    if n <= 30:
        return "train"
    if n <= 35:
        return "val"
    if n <= 42:
        return "cal"
    return "test"


def stitch(tiles, n_lines):
    C = tiles[0].data.shape[0]
    full = np.full((C, n_lines, 128), np.nan)
    inp = np.zeros((n_lines, 128), dtype=bool)
    for t in tiles:
        h = t.data.shape[1]
        e = min(t.line_start + h, n_lines)
        full[:, t.line_start:e, :] = t.data[:, :e - t.line_start, :]
        inp[t.line_start:e, :] |= t.inpaint_mask[0, :e - t.line_start, :]
    return full, inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", nargs="*", type=int)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(cfg["out_root"]) / "aoi5"
    out_dir = root / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "training_manifest.csv"
    if args.redo:
        manifest.unlink(missing_ok=True)

    lab = pd.read_csv(root / "labels" / "eddy_labels_aoi5.csv",
                      dtype={"cycle": str, "pass": str})
    frontal = lab[lab["label"] == "frontal"]
    verts = pd.read_csv(root / "catalog" / "contours.csv",
                        dtype={"cycle": str, "pass": str})
    fkeys = set(zip(frontal["cycle"], frontal["pass"], frontal["kind"], frontal["eddy"]))
    verts = verts[[k in fkeys for k in
                   zip(verts["cycle"], verts["pass"], verts["kind"], verts["eddy"])]]

    passes = lab[["cycle", "pass"]].drop_duplicates()
    if args.cycles:
        want = {f"cycle_{n:03d}" for n in args.cycles}
        passes = passes[passes["cycle"].isin(want)]

    done = set()
    if manifest.exists():
        m = pd.read_csv(manifest, dtype={"cycle": str, "pass": str})
        done = set(zip(m["cycle"], m["pass"]))
    todo = [t for t in passes.itertuples(index=False) if (t[0], t[1]) not in done]
    print(f"{len(passes)} labeled passes; {len(done)} done, {len(todo)} left")

    from skimage.draw import polygon as sk_polygon
    for n, (cyc, pnum) in enumerate(todo, 1):
        f = find_granule(cfg["data_root"], cyc, pnum)
        row = {"cycle": cyc, "pass": pnum, "split": split_of(cyc),
               "n_frontal": 0, "n_lines": 0, "status": "ok"}
        if f is None:
            row["status"] = "no_granule"
            pd.DataFrame([row], columns=MAN_COLS).to_csv(
                manifest, mode="a", header=not manifest.exists(), index=False)
            continue
        try:
            tiles, summ, _, _ = pp.preprocess_pass(str(f), cfg)
            full, inp = stitch(tiles, summ["n_lines_cropped"])
            x = np.concatenate([full, grad_mag_per_km(full[1])[None]], axis=0)

            y = np.zeros(full.shape[1:], dtype=np.uint8)
            vv = verts[(verts["cycle"] == cyc) & (verts["pass"] == pnum)]
            for _, g in vv.groupby(["kind", "eddy"]):
                rr, cc = sk_polygon(g["line"].values, g["col"].values, shape=y.shape)
                y[rr, cc] = 1
                row["n_frontal"] += 1
            row["n_lines"] = int(full.shape[1])

            np.savez_compressed(out_dir / f"{cyc}_{pnum}.npz",
                                x=x.astype(np.float16), y=y,
                                inp=inp.astype(np.uint8))
        except Exception as e:
            row["status"] = f"error:{type(e).__name__}"
            print(f"  ERROR {cyc}/{pnum}: {type(e).__name__}: {e}")
        pd.DataFrame([row], columns=MAN_COLS).to_csv(
            manifest, mode="a", header=not manifest.exists(), index=False)
        if n % 25 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {cyc}/{pnum}: {row['n_frontal']} frontal")

    m = pd.read_csv(manifest)
    print(m.groupby("split")[["n_frontal"]].agg(["count", "sum"]))
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
