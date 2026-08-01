"""Per-eddy zoom gallery: ADT + total geostrophic currents around each frontal eddy.

One figure per detected frontal eddy, for fast visual review of hundreds of
detections:

  (a) main: ~1x1 deg ADT (Spectral_r, per-window percentile clim) with absolute
      geostrophic velocity quiver (ugos/vgos_filtered), the eddy's PET contour in
      violet, other detections in view in gray. Eddy centered in LATITUDE; the
      swath nadir gap centered in LONGITUDE (so most of the swath width shows).
  (b) locator strip: full cropped pass ADT with coastline, all detections as gray
      dots, this eddy as violet star + zoom-box outline.

Input: the *_hybrid_eddies.nc written by export_eddy_nc.py (full granule copy +
eddy_id) and its sibling *_hybrid.csv.

Usage:
  .venv/bin/python scripts/make_eddy_zoom_figs.py out/aoi5/predictions/<stem>_hybrid_eddies.nc
  ... [--ids 13 9] [--size 1.0]

Writes out/aoi5/eddy_zoom/eddy_{cycle}_{pass}_{id:02d}.png
"""
from __future__ import annotations

import argparse
import re
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swot_eddy.config import load_config
from swot_eddy import preprocess as pp

C_FRONTAL, C_OTHER = "#7b3294", "#666666"
NADIR_COL = 34               # center pixel of the 69-px swath
QUIVER_N = 2                 # every 2nd pixel = 4 km spacing
QUIVER_SCALE = 12            # m/s per unit length (zoomed view, longer arrows)
QUIVER_KEY = 0.5             # m/s


def geo_ax(fig, rect, labels=True):
    ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
    ax.set_anchor("N")      # aspect-shrunk maps (tall panel c) align at the TOP

    try:
        ax.coastlines("10m", linewidth=0.6, color="k")
    except Exception as exc:
        warnings.warn(f"coastlines unavailable: {exc}")
    gl = ax.gridlines(draw_labels=labels, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle=":")
    if labels:
        gl.top_labels = gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = gl.ylabel_style = {"size": 7}
    return ax


def one_eddy(d, i, size, out_dir):
    row = d["det"].loc[i]
    lat_c = float(row["lat"])
    lat0, lat1 = lat_c - size / 2, lat_c + size / 2

    # longitude center = swath nadir gap at the eddy's latitude
    rows = np.where((d["lat"][:, NADIR_COL] >= lat0)
                    & (d["lat"][:, NADIR_COL] <= lat1))[0]
    nadir_lon = d["lon"][rows, NADIR_COL] if rows.size else np.array([np.nan])
    lon_c = float(np.nanmean(nadir_lon))
    if not np.isfinite(lon_c):
        lon_c = float(row["lon"])
    lon0, lon1 = lon_c - size / 2, lon_c + size / 2

    inwin = ((d["lon"] >= lon0) & (d["lon"] <= lon1)
             & (d["lat"] >= lat0) & (d["lat"] <= lat1))
    adt_win = d["adt"][inwin & np.isfinite(d["adt"])]
    if adt_win.size < 50:
        print(f"  eddy {i}: window nearly empty, skipped")
        return
    vmin, vmax = np.percentile(adt_win, 2), np.percentile(adt_win, 98)

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13.2, 6.8), facecolor="w")

    def frontal_contours(ax, this_id):
        for j in d["det"].index:
            if not bool(d["det"].loc[j, "is_frontal"]):
                continue
            mask = d["eid"] == j
            if not mask.any() or not (mask & inwin).any():
                continue
            ax.contour(d["lon"], d["lat"], mask.astype(float), levels=[0.5],
                       colors=[C_FRONTAL],
                       linewidths=2.2 if j == this_id else 1.2,
                       transform=pc, zorder=7)

    # (a) zoom: ADT + absolute geostrophic quiver
    ax = geo_ax(fig, [0.045, 0.13, 0.37, 0.74])
    ax.set_extent([lon0, lon1, lat0, lat1], crs=pc)
    pm = ax.pcolormesh(d["lon"], d["lat"], np.ma.masked_invalid(d["adt"]),
                       cmap="Spectral_r", vmin=vmin, vmax=vmax,
                       shading="auto", transform=pc, zorder=3)
    n = QUIVER_N
    q = ax.quiver(d["lon"][::n, ::n], d["lat"][::n, ::n],
                  d["ugos"][::n, ::n], d["vgos"][::n, ::n],
                  color="k", scale=QUIVER_SCALE, width=0.0022,
                  transform=pc, zorder=6)
    ax.quiverkey(q, 0.93, 1.06, QUIVER_KEY, f"{QUIVER_KEY} m s$^{{-1}}$",
                 labelpos="E", coordinates="axes", fontproperties={"size": 8})
    frontal_contours(ax, i)
    ax.set_title("(a) ADT + absolute geostrophic velocity", fontsize=9)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.06,
                      shrink=0.85, aspect=35)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("ADT (cm)", fontsize=8)

    # (b) zoom, same size: SSHA + geostrophic velocity ANOMALY quiver
    axs = geo_ax(fig, [0.46, 0.13, 0.37, 0.74])
    axs.set_extent([lon0, lon1, lat0, lat1], crs=pc)
    ssha_win = d["ssha"][inwin & np.isfinite(d["ssha"])]
    sv1, sv2 = (np.percentile(ssha_win, [2, 99]) if ssha_win.size
                else (-10.0, 10.0))
    pms = axs.pcolormesh(d["lon"], d["lat"], np.ma.masked_invalid(d["ssha"]),
                         cmap="RdBu_r", vmin=sv1, vmax=sv2,
                         shading="auto", transform=pc, zorder=3)
    qa = axs.quiver(d["lon"][::n, ::n], d["lat"][::n, ::n],
                    d["ugosa"][::n, ::n], d["vgosa"][::n, ::n],
                    color="k", scale=QUIVER_SCALE, width=0.0022,
                    transform=pc, zorder=6)
    axs.quiverkey(qa, 0.93, 1.06, QUIVER_KEY, f"{QUIVER_KEY} m s$^{{-1}}$",
                  labelpos="E", coordinates="axes", fontproperties={"size": 8})
    frontal_contours(axs, i)
    axs.set_title("(b) SSHA + geostrophic velocity anomaly", fontsize=9)
    cbs = fig.colorbar(pms, ax=axs, orientation="horizontal", pad=0.06,
                       shrink=0.85, aspect=35)
    cbs.ax.tick_params(labelsize=7)
    cbs.set_label("SSHA (cm)", fontsize=8)

    # (c) locator: full-pass ADT + quiver + detections (pair_adt panel-b style)
    axb = geo_ax(fig, [0.87, 0.13, 0.12, 0.74], labels=False)
    has = np.isfinite(d["adt"])
    axb.set_extent([np.nanmin(d["lon"][has]) - 0.4, np.nanmax(d["lon"][has]) + 0.4,
                    np.nanmin(d["lat"][has]) - 0.3, np.nanmax(d["lat"][has]) + 0.3],
                   crs=pc)
    fv1 = float(np.floor(np.percentile(d["adt"][has], 2)))
    fv2 = float(np.ceil(np.percentile(d["adt"][has], 98)))
    axb.pcolormesh(d["lon"], d["lat"], np.ma.masked_invalid(d["adt"]),
                   cmap="Spectral_r", vmin=fv1, vmax=fv2, shading="auto",
                   transform=pc, zorder=3)
    colnan = np.isnan(d["adt"]).all(axis=0)
    valid = np.where(~colnan)[0]
    if valid.size:
        left, right = int(valid.min()), int(valid.max())
        gap = [c for c in range(left, right + 1) if colnan[c]]
        cols = [left, right] + ([min(gap) - 1, max(gap) + 1] if gap else [])
        for col in sorted(set(c for c in cols if 0 <= c < d["adt"].shape[1])):
            lon_c_ = np.where(np.isfinite(d["adt"][:, col]), d["lon"][:, col], np.nan)
            axb.plot(lon_c_, d["lat"][:, col], "k", linewidth=0.9, transform=pc,
                     zorder=7)
    nb = 6
    axb.quiver(d["lon"][::nb, ::nb], d["lat"][::nb, ::nb],
               d["ugos"][::nb, ::nb], d["vgos"][::nb, ::nb],
               color="k", scale=20, width=0.0018, transform=pc, zorder=6)
    for j in d["det"].index:
        if not bool(d["det"].loc[j, "is_frontal"]):
            continue
        mask = d["eid"] == j
        if not mask.any():
            continue
        axb.contour(d["lon"], d["lat"], mask.astype(float), levels=[0.5],
                    colors=[C_FRONTAL], linewidths=1.6, transform=pc, zorder=8)
    axb.plot(float(row["lon"]), lat_c, "o", ms=22, mfc="none", mec="k",
             mew=2.2, transform=pc, zorder=9)
    axb.set_title("(c) full pass", fontsize=9)

    fig.suptitle(f"cycle {d['cycle']}  pass {d['pas']}  "
                 f"{d['time'].replace('T', ' ')} UTC   —   "
                 f"eddy {i}  score {row['frontal_score']:.2f}  "
                 f"R {row['radius_km']:.1f} km  {row['polarity']}",
                 fontsize=11, y=0.96)
    out = out_dir / f"eddy_{d['cycle']}_{d['pas']}_{i:02d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eddies_nc", help="*_hybrid_eddies.nc from export_eddy_nc.py")
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="eddy ids (default: all frontal)")
    ap.add_argument("--size", type=float, default=1.3, help="window size, deg")
    args = ap.parse_args()

    nc_path = Path(args.eddies_nc)
    stem = nc_path.name.replace("_hybrid_eddies.nc", "")
    det = pd.read_csv(nc_path.parent / f"{stem}_hybrid.csv").set_index("id")

    cfg = load_config()
    with xr.open_dataset(nc_path) as ds:
        sub, _ = pp.crop_to_region(ds, cfg["region"])
        lat = sub["latitude"].values
        lon = ((sub["longitude"].values + 180) % 360) - 180
        adt = (sub["ssha_unedited"].values + sub["mdt"].values) * 100.0
        ssha = sub["ssha_filtered"].values * 100.0
        ugos = sub["ugos_filtered"].values
        vgos = sub["vgos_filtered"].values
        ugosa = sub["ugosa_filtered"].values
        vgosa = sub["vgosa_filtered"].values
        eid = sub["eddy_id"].values
        t = str(sub["time"].values[len(sub["time"]) // 2])[:16]

    m = re.search(r"_(\d{3})_(\d{3})_\d{8}T", nc_path.name)
    cycle, pas = (m.group(1), m.group(2)) if m else ("xxx", "xxx")
    d = dict(lon=lon, lat=lat, adt=adt, ssha=ssha, ugos=ugos, vgos=vgos,
             ugosa=ugosa, vgosa=vgosa, eid=eid, det=det, time=t,
             cycle=cycle, pas=pas)

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.6})
    out_dir = Path(__file__).resolve().parents[1] / "out/aoi5/eddy_zoom"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.ids if args.ids else list(det[det["is_frontal"]].index)
    print(f"{stem}: {len(ids)} eddies")
    for i in ids:
        one_eddy(d, i, args.size, out_dir)


if __name__ == "__main__":
    main()
