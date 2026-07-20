"""End-to-end HYBRID frontal-eddy detection: py-eddy-tracker x U-Net.

PET supplies eddy-ness (closed SSH contours, shape/amplitude criteria, de Marez recipe);
the v2 U-Net supplies frontal-ness (probability field from raw native channels). A pass's
frontal eddies = PET detections whose contour contains U-Net probability > tau.

Test-set performance (labeled PET detections, cycles 43-50):
  precision 0.80 / recall 0.96 / F1 0.875  at tau=0.9
  (U-Net alone: 0.41/0.96/0.58; ADT-distance rule: 0.84/0.72/0.78)

Usage:
  .venv/bin/python scripts/predict_hybrid.py GRANULE.nc [--tau 0.9] [--report]

Outputs in out/aoi5/predictions/:
  <stem>_hybrid.csv        one row per PET eddy: lon, lat, radius_km, amplitude_m,
                           polarity, frontal_score, is_frontal
  <stem>_hybrid.geojson    frontal eddies as polygons (PET contours) for MapLibre
  <stem>_hybrid.png        report card (--report)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*partition.*will ignore the 'mask'.*")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swot_eddy.config import load_config
from swot_eddy.aoi import load_aoi
from swot_eddy import preprocess as pp
from swot_eddy.detect import detect_pass_split
from swot_eddy.features import raw_channels
from swot_eddy import unet as U


def unet_prob(x, model):
    import torch
    L = x.shape[1]
    prob = np.zeros((L, 128), np.float32)
    hits = np.zeros((L, 128), np.float32)
    with torch.no_grad():
        for s, xt in U.pass_tiles(x):
            p = torch.sigmoid(model(torch.from_numpy(U.normalize(xt))[None]))[0, 0].numpy()
            e = min(s + U.TILE, L)
            prob[s:e] += p[:e - s]
            hits[s:e] += 1
    return prob / np.maximum(hits, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("granule")
    ap.add_argument("--tau", type=float, default=0.9,
                    help="frontal threshold on max U-Net prob inside the PET contour")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    import datetime as dt
    import re
    import torch
    from skimage.draw import polygon as sk_polygon

    cfg = load_config()
    aoi = load_aoi(cfg)
    root = Path(__file__).resolve().parents[1]
    wpath = Path(args.weights) if args.weights else root / "out/aoi5/model/unet_frontal_v2.pt"
    model = U.build_unet(in_ch=4)
    model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
    model.eval()

    # PET branch (inpainted tiles, faithful recipe) + U-Net branch (raw channels)
    m = re.search(r"_(\d{8})T(\d{6})_", Path(args.granule).name)
    date = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else dt.datetime(2023, 1, 1)
    tiles, summ, clon, clat = pp.preprocess_pass(str(args.granule), cfg)
    kept, _, _ = detect_pass_split(tiles, cfg, clon, clat, date)
    kept = [d for d in kept if aoi.contains(d.center_lon, d.center_lat)]
    x, _ = raw_channels(args.granule, cfg)
    prob = unet_prob(x.astype(np.float32), model)

    rows, feats = [], []
    for i, d in enumerate(kept):
        mask = np.zeros(prob.shape, bool)
        if d.contour_pix:
            cp = np.array(d.contour_pix)
            rr, cc = sk_polygon(cp[:, 0], cp[:, 1], shape=prob.shape)
            mask[rr, cc] = True
        score = float(prob[mask].max()) if mask.any() else \
            float(prob[d.center_line, d.center_col])
        frontal = score > args.tau
        lon180 = None if d.center_lon is None else ((d.center_lon + 180) % 360) - 180
        rows.append({"id": i, "lon": None if lon180 is None else round(lon180, 4),
                     "lat": None if d.center_lat is None else round(d.center_lat, 4),
                     "radius_km": round(d.radius_km, 1),
                     "amplitude_m": round(d.amplitude_m, 4), "polarity": d.polarity,
                     "frontal_score": round(score, 3), "is_frontal": frontal})
        if frontal and d.contour_lonlat:
            ring = [[round(((p[0] + 180) % 360) - 180, 4), round(p[1], 4)]
                    for p in d.contour_lonlat if p is not None]
            if len(ring) >= 4:
                ring.append(ring[0])
                feats.append({"type": "Feature",
                              "geometry": {"type": "Polygon", "coordinates": [ring]},
                              "properties": {**{k: rows[-1][k] for k in
                                                ("id", "lon", "lat", "radius_km",
                                                 "polarity")},
                                             "max_p": rows[-1]["frontal_score"]}})

    import pandas as pd
    out_dir = root / "out/aoi5/predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.granule).stem
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{stem}_hybrid.csv", index=False)
    (out_dir / f"{stem}_hybrid.geojson").write_text(json.dumps(
        {"type": "FeatureCollection",
         "properties": {"method": "PET x U-Net hybrid", "tau": args.tau,
                        "model": wpath.name},
         "features": feats}))
    nf_ = int(df.is_frontal.sum())
    print(f"{stem}: {len(df)} PET eddies, {nf_} frontal (tau={args.tau})")
    if len(df):
        print(df[df.is_frontal][["id", "lon", "lat", "radius_km", "polarity",
                                 "frontal_score"]].to_string(index=False))

    if args.report:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        L = summ["n_lines_cropped"]
        full = np.full((L, 128), np.nan)
        for t in tiles:
            h = t.data.shape[1]
            e = min(t.line_start + h, L)
            full[t.line_start:e, :] = t.data[0, :e - t.line_start, :] * 100
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 10))
        axes[0].imshow(full, cmap="RdBu_r", vmin=-10, vmax=30, origin="lower", aspect="auto")
        axes[0].set_title("SSHA + PET eddies (violet=frontal, gray=other)", fontsize=9)
        axes[1].imshow(prob, cmap="magma", vmin=0, vmax=1, origin="lower", aspect="auto")
        axes[1].set_title("U-Net P(frontal)", fontsize=9)
        for d, r in zip(kept, rows):
            col = "#7b3294" if r["is_frontal"] else "0.6"
            if d.contour_pix:
                cp = np.array(d.contour_pix)
                for a in axes:
                    a.plot(cp[:, 1], cp[:, 0], color=col,
                           lw=1.8 if r["is_frontal"] else 0.9)
        for a in axes:
            a.axis("off")
        fig.suptitle(f"{stem}\nhybrid: {nf_} frontal of {len(df)} PET eddies", fontsize=9)
        fig.savefig(out_dir / f"{stem}_hybrid.png", dpi=130, bbox_inches="tight")
        print(f"report -> {out_dir / (stem + '_hybrid.png')}")


if __name__ == "__main__":
    main()
