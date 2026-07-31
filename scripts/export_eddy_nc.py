"""Hybrid detections exported back onto the ORIGINAL granule grid.

Runs the same hybrid pipeline as predict_hybrid.py (PET proposes closed contours,
v2 U-Net scores frontal-ness), then writes a copy of the original NetCDF with two
new variables on the native (num_lines, num_pixels) grid:

  eddy_flag  uint8   0 = background, 1 = inside a non-frontal PET eddy contour,
                     2 = inside a frontal eddy contour (filled interior, not just
                     the 1-px ring — the exact ring is the mask boundary)
  eddy_id    int16   -1 = background, else the eddy id matching *_hybrid.csv

Also writes a geographic swath map (native lon/lat, no inpainting, no padding):
SSHA with the detected contours drawn on top.

Usage:
  .venv/bin/python scripts/export_eddy_nc.py GRANULE.nc [--tau 0.9] [--weights W]

Outputs in out/aoi5/predictions/:
  <stem>_hybrid_eddies.nc       full original granule + eddy_flag + eddy_id
  <stem>_hybrid.csv             every PET eddy: lat/lon, radius, amplitude,
                                polarity, frontal_score, is_frontal
  <stem>_hybrid_frontal.csv     confirmed frontal eddies only (score > tau)
  <stem>_hybrid_eddies_map.png  swath map in native geometry
"""
from __future__ import annotations

import argparse
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
from swot_eddy.features import raw_channels, native_fields, front_vertices
from swot_eddy import unet as U

C_FRONTAL, C_OTHER, C_FRONT = "#7b3294", "#666666", "#B8860B"


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
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    import datetime as dt
    import re
    import torch
    import xarray as xr
    from skimage.draw import polygon as sk_polygon

    cfg = load_config()
    aoi = load_aoi(cfg)
    root = Path(__file__).resolve().parents[1]
    wpath = Path(args.weights) if args.weights else root / "out/aoi5/model/unet_frontal_v2.pt"
    model = U.build_unet(in_ch=4)
    model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
    model.eval()

    m = re.search(r"_(\d{8})T(\d{6})_", Path(args.granule).name)
    date = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else dt.datetime(2023, 1, 1)
    tiles, summ, clon, clat = pp.preprocess_pass(str(args.granule), cfg)
    kept, _, _ = detect_pass_split(tiles, cfg, clon, clat, date)
    kept = [d for d in kept if aoi.contains(d.center_lon, d.center_lat)]
    x, _ = raw_channels(args.granule, cfg)
    prob = unet_prob(x.astype(np.float32), model)

    # per-eddy filled interior in the cropped/padded frame, then score
    masks, scores = [], []
    for d in kept:
        mask = np.zeros(prob.shape, bool)
        if d.contour_pix:
            cp = np.array(d.contour_pix)
            rr, cc = sk_polygon(cp[:, 0], cp[:, 1], shape=prob.shape)
            mask[rr, cc] = True
        masks.append(mask)
        scores.append(float(prob[mask].max()) if mask.any() else
                      float(prob[d.center_line, d.center_col]))

    # map cropped/padded frame -> original granule grid
    line0 = summ["line_slice"][0]
    with xr.open_dataset(args.granule) as ds0:
        n_lines = ds0.sizes["num_lines"]
        n_pix = ds0.sizes["num_pixels"]
    pad_left = (128 - n_pix) // 2

    flag = np.zeros((n_lines, n_pix), np.uint8)
    eid = np.full((n_lines, n_pix), -1, np.int16)
    # draw non-frontal first so frontal wins any overlap
    order = sorted(range(len(kept)), key=lambda i: scores[i] > args.tau)
    for i in order:
        rr, cc = np.nonzero(masks[i])
        rr = rr + line0
        cc = cc - pad_left
        ok = (cc >= 0) & (cc < n_pix) & (rr >= 0) & (rr < n_lines)
        rr, cc = rr[ok], cc[ok]
        flag[rr, cc] = 2 if scores[i] > args.tau else 1
        eid[rr, cc] = i

    out_dir = root / "out/aoi5/predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.granule).stem

    # CSVs: all PET eddies + frontal-only (same rows/ids as predict_hybrid.py)
    import pandas as pd
    rows = []
    for i, (d, s) in enumerate(zip(kept, scores)):
        lon180 = None if d.center_lon is None else ((d.center_lon + 180) % 360) - 180
        rows.append({"id": i, "lon": None if lon180 is None else round(lon180, 4),
                     "lat": None if d.center_lat is None else round(d.center_lat, 4),
                     "radius_km": round(d.radius_km, 1),
                     "amplitude_m": round(d.amplitude_m, 4), "polarity": d.polarity,
                     "frontal_score": round(s, 3), "is_frontal": s > args.tau})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{stem}_hybrid.csv", index=False)
    df[df.is_frontal].to_csv(out_dir / f"{stem}_hybrid_frontal.csv", index=False)

    ds = xr.open_dataset(args.granule)
    ds["eddy_flag"] = (("num_lines", "num_pixels"), flag)
    ds["eddy_flag"].attrs = {
        "long_name": "hybrid frontal-eddy detection mask (filled contour interior)",
        "flag_values": np.array([0, 1, 2], np.uint8),
        "flag_meanings": "background eddy_interior frontal_eddy_interior",
        "comment": (f"py-eddy-tracker closed contour x U-Net frontal score "
                    f"(tau={args.tau}, weights={wpath.name}); interior fill of the "
                    f"PET contour, ring = mask boundary"),
    }
    ds["eddy_id"] = (("num_lines", "num_pixels"), eid)
    ds["eddy_id"].attrs = {
        "long_name": "eddy id matching the *_hybrid.csv rows",
        "comment": "-1 = background",
    }
    ds.attrs["history"] = (ds.attrs.get("history", "") +
                           f" | eddy_flag/eddy_id added by export_eddy_nc.py "
                           f"({date:%Y-%m-%d} pass)").strip(" |")
    enc = {v: {"zlib": True, "complevel": 4} for v in ("eddy_flag", "eddy_id")}
    nc_out = out_dir / f"{stem}_hybrid_eddies.nc"
    ds.to_netcdf(nc_out, encoding=enc)
    ds.close()

    n_frontal = sum(s > args.tau for s in scores)
    print(f"{stem}: {len(kept)} PET eddies, {n_frontal} frontal (tau={args.tau})")
    print(f"eddy_flag: {int((flag == 2).sum())} frontal px, "
          f"{int((flag == 1).sum())} other px of {flag.size}")
    print(f"nc -> {nc_out}")

    # swath map in native geometry
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
    try:
        import cartopy.crs as ccrs
        PC = ccrs.PlateCarree()
    except Exception:
        PC = None

    nf = native_fields(args.granule, cfg)
    front = front_vertices(nf["adt"], nf["lon"], nf["lat"],
                           cfg["features"]["front_level_m"], 20)
    ssha_cm = x[0, :, pad_left:pad_left + n_pix] * 100
    v1, v2 = np.nanpercentile(ssha_cm, 2), np.nanpercentile(ssha_cm, 98)
    kw = {"transform": PC} if PC else {}

    fig = plt.figure(figsize=(5.2, 6.4))
    if PC is not None:
        ax = fig.add_axes([0.08, 0.10, 0.86, 0.82], projection=PC)
        ax.coastlines("10m", linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.85", linestyle=":")
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
    else:
        ax = fig.add_axes([0.08, 0.10, 0.86, 0.82])
    pm = ax.pcolormesh(nf["lon"], nf["lat"], ssha_cm, cmap="RdBu_r",
                       vmin=v1, vmax=v2, shading="nearest", **kw)
    if len(front):
        ax.plot(front[:, 0], front[:, 1], ".", color=C_FRONT, ms=1.0, **kw)
    for d, s in zip(kept, scores):
        pts = [p for p in (d.contour_lonlat or []) if p is not None]
        if not pts:
            continue
        arr = np.array(pts)
        lon = ((arr[:, 0] + 180) % 360) - 180
        frontal = s > args.tau
        ax.plot(lon, arr[:, 1], color=C_FRONTAL if frontal else C_OTHER,
                lw=1.8 if frontal else 0.8, **kw)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.07, shrink=0.85,
                      aspect=35)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label("SSHA (cm)", fontsize=7)
    ax.set_title(f"{stem}\n{len(kept)} PET eddies, {n_frontal} frontal "
                 f"(violet; gray = non-frontal; gold = {cfg['features']['front_level_m']:.2f} m ADT front)",
                 fontsize=8)
    png_out = out_dir / f"{stem}_hybrid_eddies_map.png"
    fig.savefig(png_out, dpi=150, bbox_inches="tight")
    print(f"map -> {png_out}")


if __name__ == "__main__":
    main()
