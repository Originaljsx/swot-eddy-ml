"""Side-by-side comparison: L3 swath detections vs L2-zarr gridded detections.

(a) L3: SSHA from the *_hybrid_eddies.nc written by export_eddy_nc.py, with its
    eddy outlines, bboxes and id/score labels
(b) L2 zarr: SSHA window from the SWOT_L2Z_* _hybrid_eddies.nc written by
    predict_hybrid_gridded.py, same overlay style

Identical extent and color scale so the only differences are the data and the
detections. Violet = frontal, gray = non-frontal.

Usage:
  .venv/bin/python scripts/make_compare_l3_l2z.py --cycle 29 --pass 91

Writes out/aoi5/predictions/compare_l3_l2z_{CCC}_{PPP}.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from swot_eddy.config import load_config
from swot_eddy import preprocess as pp

C_FRONTAL, C_OTHER = "#7b3294", "#666666"
BOX_MARGIN_DEG = 0.15


def style_axis(ax, label, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    try:
        ax.coastlines("10m", linewidth=0.6, color="k")
    except Exception as exc:
        warnings.warn(f"coastlines unavailable: {exc}")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(0.02, 0.985, f"({label})", transform=ax.transAxes, fontsize=16,
            va="top", ha="left",
            bbox=dict(facecolor="w", edgecolor="none", alpha=0.7, pad=1.5))


def draw_dets(ax, lon, lat, eid, det, pc):
    for i in sorted(det.index):
        mask = eid == i
        if not mask.any():
            continue
        frontal = bool(det.loc[i, "is_frontal"])
        color = C_FRONTAL if frontal else C_OTHER
        ax.contour(lon, lat, mask.astype(float), levels=[0.5], colors=[color],
                   linewidths=1.8 if frontal else 0.9, transform=pc, zorder=9)
        lo0 = lon[mask].min() - BOX_MARGIN_DEG
        lo1 = lon[mask].max() + BOX_MARGIN_DEG
        la0 = lat[mask].min() - BOX_MARGIN_DEG
        la1 = lat[mask].max() + BOX_MARGIN_DEG
        ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0],
                color=color, lw=1.2 if frontal else 0.7, ls="--", transform=pc,
                zorder=9)
        ax.text(lo0, la1 + 0.03, f"{i}  {det.loc[i, 'frontal_score']:.2f}",
                color=color, fontsize=7,
                fontweight="bold" if frontal else "normal",
                ha="left", va="bottom", transform=pc, zorder=10,
                bbox=dict(facecolor="w", edgecolor="none", alpha=0.6, pad=0.8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--pass", dest="pass_num", type=int, required=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    pred = root / "out/aoi5/predictions"
    tag = f"{args.cycle:03d}_{args.pass_num:03d}"

    l3_nc = sorted(pred.glob(f"SWOT_L3_LR_SSH_Expert_{tag}_*_hybrid_eddies.nc"))
    l2_nc = sorted(pred.glob(f"SWOT_L2Z_Expert_{tag}_*_hybrid_eddies.nc"))
    if not l3_nc or not l2_nc:
        sys.exit(f"missing outputs for {tag}: L3={len(l3_nc)} L2Z={len(l2_nc)} "
                 "(run export_eddy_nc.py and predict_hybrid_gridded.py first)")
    l3_nc, l2_nc = l3_nc[0], l2_nc[0]
    l3_det = pd.read_csv(str(l3_nc).replace("_hybrid_eddies.nc", "_hybrid.csv")
                         ).set_index("id")
    l2_det = pd.read_csv(str(l2_nc).replace("_hybrid_eddies.nc", "_hybrid.csv")
                         ).set_index("id")

    cfg = load_config()
    with xr.open_dataset(l3_nc) as ds:
        sub, _ = pp.crop_to_region(ds, cfg["region"])
        lat3 = sub["latitude"].values
        lon3 = ((sub["longitude"].values + 180) % 360) - 180
        ssha3 = sub["ssha_filtered"].values * 100.0
        eid3 = sub["eddy_id"].values
        t3 = str(sub["time"].values[len(sub["time"]) // 2])[:19]
    with xr.open_dataset(l2_nc) as ds:
        lat1 = ds["latitude"].values
        lon1 = ds["longitude"].values
        ssha2 = ds["ssha"].values * 100.0
        eid2 = ds["eddy_id"].values
        t2 = str(ds.attrs.get("slice_time", ""))[:19]
    lon2, lat2 = np.meshgrid(lon1, lat1)

    has = np.isfinite(ssha3)
    extent = [np.nanmin(lon3[has]) - 0.3, np.nanmax(lon3[has]) + 0.3,
              np.nanmin(lat3[has]) - 0.2, np.nanmax(lat3[has]) + 0.2]
    lim = np.ceil(np.nanpercentile(np.abs(ssha3), 99) / 5.0) * 5.0
    lim = float(np.clip(lim, 15.0, 40.0))

    pc = ccrs.PlateCarree()
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.6})
    panel_w = 6.0
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(2 * panel_w + 1.0, panel_w * aspect + 2.0), facecolor="w",
        constrained_layout=True, subplot_kw={"projection": pc})

    n3 = int(l3_det["is_frontal"].sum())
    n2 = int(l2_det["is_frontal"].sum())
    for label, ax, lon, lat, fld, eid, det, ttl in (
            ("a", ax_a, lon3, lat3, ssha3, eid3, l3_det,
             f"L3 AVISO swath — {len(l3_det)} eddies, {n3} frontal"),
            ("b", ax_b, lon2, lat2, ssha2, eid2, l2_det,
             f"L2 JPL zarr (0.02° grid) — {len(l2_det)} eddies, {n2} frontal")):
        pm = ax.pcolormesh(lon, lat, np.ma.masked_invalid(fld), cmap="RdBu_r",
                           vmin=-lim, vmax=lim, shading="auto", transform=pc,
                           zorder=3)
        draw_dets(ax, lon, lat, eid, det, pc)
        style_axis(ax, label, extent)
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.02,
                     shrink=0.9).set_label("height anomaly (cm)")

    fig.suptitle(f"SWOT pass {args.pass_num:03d}  cycle {args.cycle:03d}   "
                 f"L3 {t3.replace('T', ' ')} | L2Z {t2.replace('T', ' ')} UTC",
                 fontsize=13)
    out = pred / f"compare_l3_l2z_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
