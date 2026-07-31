"""Side-by-side pair figures: (a) field, (b) same field + hybrid eddy detections.

Three versions per granule, F1_v15 styling:

  *_hybrid_pair_ssha.png    (a) SSHA                          (b) SSHA + detections
  *_hybrid_pair_speed.png   (a) geostrophic speed             (b) speed + detections
  *_hybrid_pair_adt.png     (a) ADT + geostrophic quiver      (b) same + detections

Panels a and b are rendered identically (same field, same color scale) so the
only difference is the detection overlay: eddy outline from eddy_flag, dashed
bounding box, id + frontal score (violet = frontal, gray = non-frontal).

Input is the *_hybrid_eddies.nc from export_eddy_nc.py; scores come from the
sibling *_hybrid.csv.

Usage:
  .venv/bin/python scripts/make_pair_eddy_figs.py out/aoi5/predictions/<stem>_hybrid_eddies.nc
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
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

SSHA_CLIM_MIN, SSHA_CLIM_MAX = 15.0, 40.0     # cm
SSHA_CMAP = "RdBu_r"
SPEED_CMAP, SPEED_CLIM = "Spectral_r", (0.0, 1.5)
ADT_CMAP = "Spectral_r"
GEOS_SUBSAMPLE = 6            # 2 km grid -> 12 km spacing
GEOS_SCALE, GEOS_WIDTH, GEOS_KEY_MS = 20, 0.0015, 0.5
BOX_MARGIN_DEG = 0.15


def _wrap180(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def ssha_clim(ssha_cm):
    finite = ssha_cm[np.isfinite(ssha_cm)]
    if finite.size == 0:
        return SSHA_CLIM_MIN
    lim = np.ceil(np.percentile(np.abs(finite), 99) / 5.0) * 5.0
    return float(np.clip(lim, SSHA_CLIM_MIN, SSHA_CLIM_MAX))


def adt_clim(adt_cm, pct=(2, 98)):
    finite = adt_cm[np.isfinite(adt_cm)]
    if finite.size == 0:
        return (0.0, 1.0)
    a = np.percentile(finite, list(pct))
    return (float(np.floor(a[0])), float(np.ceil(a[1])))


def swath_boundary_cols(field2d):
    colnan = np.isnan(field2d).all(axis=0)
    valid = np.where(~colnan)[0]
    if valid.size == 0:
        return []
    left, right = int(valid.min()), int(valid.max())
    gap = [c for c in range(left, right + 1) if colnan[c]]
    cols = [left, right]
    if gap:
        cols += [int(min(gap)) - 1, int(max(gap)) + 1]
    return sorted(set(c for c in cols if 0 <= c < field2d.shape[1]))


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


def draw_detections(ax, d, pc):
    """Eddy outlines + dashed bboxes + id/score labels onto one axis."""
    for i in sorted(d["det"].index):
        mask = d["eid"] == i
        if not mask.any():
            continue
        frontal = bool(d["det"].loc[i, "is_frontal"])
        color = C_FRONTAL if frontal else C_OTHER
        ax.contour(d["lon"], d["lat"], mask.astype(float), levels=[0.5],
                   colors=[color], linewidths=1.8 if frontal else 0.9,
                   transform=pc, zorder=9)
        lo0 = d["lon"][mask].min() - BOX_MARGIN_DEG
        lo1 = d["lon"][mask].max() + BOX_MARGIN_DEG
        la0 = d["lat"][mask].min() - BOX_MARGIN_DEG
        la1 = d["lat"][mask].max() + BOX_MARGIN_DEG
        ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0],
                color=color, lw=1.2 if frontal else 0.7, ls="--", transform=pc,
                zorder=9)
        ax.text(lo0, la1 + 0.03, f"{i}  {d['det'].loc[i, 'frontal_score']:.2f}",
                color=color, fontsize=8,
                fontweight="bold" if frontal else "normal",
                ha="left", va="bottom", transform=pc, zorder=10,
                bbox=dict(facecolor="w", edgecolor="none", alpha=0.6, pad=0.8))


def draw_adt_extras(ax, d, pc):
    for col in swath_boundary_cols(d["adt"]):
        lon_c = np.where(np.isfinite(d["adt"][:, col]), d["lon"][:, col], np.nan)
        ax.plot(lon_c, d["lat"][:, col], "k", linewidth=1.4, transform=pc,
                zorder=7)
    n = GEOS_SUBSAMPLE
    q = ax.quiver(d["lon"][::n, ::n], d["lat"][::n, ::n],
                  d["ugos"][::n, ::n], d["vgos"][::n, ::n],
                  color="k", scale=GEOS_SCALE, width=GEOS_WIDTH, transform=pc,
                  zorder=8)
    ax.quiverkey(q, 0.86, 0.04, GEOS_KEY_MS, f"{GEOS_KEY_MS} m s$^{{-1}}$",
                 labelpos="E", coordinates="axes", fontproperties={"size": 9})


def pair_figure(d, mode, out_path):
    pc = ccrs.PlateCarree()
    extent = d["extent"]
    panel_w = 6.0
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    figsize = (2 * panel_w + 1.0, panel_w * aspect + 2.0)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize, facecolor="w",
                                     constrained_layout=True,
                                     subplot_kw={"projection": pc})

    if mode == "ssha":
        clim = ssha_clim(d["ssha"])
        field, cmap, vmin, vmax = d["ssha"], SSHA_CMAP, -clim, clim
        title, cb_label = "Sea surface height anomaly", "height anomaly (cm)"
    elif mode == "speed":
        field, cmap = d["spd"], SPEED_CMAP
        vmin, vmax = SPEED_CLIM
        title, cb_label = "Geostrophic speed", r"geostrophic speed (m s$^{-1}$)"
    else:  # adt
        vmin, vmax = adt_clim(d["adt"])
        field, cmap = d["adt"], ADT_CMAP
        title = "Absolute dynamic topography and geostrophic velocity"
        cb_label = "absolute dynamic topography (cm)"

    for label, ax in (("a", ax_a), ("b", ax_b)):
        pm = ax.pcolormesh(d["lon"], d["lat"], np.ma.masked_invalid(field),
                           cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
                           transform=pc, zorder=3)
        if mode == "adt":
            draw_adt_extras(ax, d, pc)
        if label == "b":
            draw_detections(ax, d, pc)
        style_axis(ax, label, extent)
        n_frontal = int(d["det"]["is_frontal"].sum())
        ax.set_title(title if label == "a" else
                     f"{title} + detections ({n_frontal} frontal of {len(d['det'])})",
                     fontsize=10)
        fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.02,
                     shrink=0.9).set_label(cb_label)

    fig.suptitle(f"SWOT L3 Expert   pass {d['pas']}   cycle {d['cycle']}   "
                 f"{d['time'].replace('T', ' ')} UTC", fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eddies_nc", help="*_hybrid_eddies.nc from export_eddy_nc.py")
    args = ap.parse_args()

    nc_path = Path(args.eddies_nc)
    stem = nc_path.name.replace("_hybrid_eddies.nc", "")
    import pandas as pd
    det = pd.read_csv(nc_path.parent / f"{stem}_hybrid.csv").set_index("id")

    cfg = load_config()
    with xr.open_dataset(nc_path) as ds:
        sub, _ = pp.crop_to_region(ds, cfg["region"])
        lat = sub["latitude"].values
        lon = _wrap180(sub["longitude"].values)
        ssha_cm = sub["ssha_filtered"].values * 100.0
        adt_cm = (sub["ssha_unedited"].values + sub["mdt"].values) * 100.0
        ugos = sub["ugos_filtered"].values
        vgos = sub["vgos_filtered"].values
        eid = sub["eddy_id"].values
        t = str(sub["time"].values[len(sub["time"]) // 2])[:19]

    m = re.search(r"_(\d{3})_(\d{3})_\d{8}T", nc_path.name)
    cycle, pas = (m.group(1), m.group(2)) if m else ("?", "?")
    has = np.isfinite(ssha_cm) | np.isfinite(adt_cm)
    d = dict(lon=lon, lat=lat, ssha=ssha_cm, adt=adt_cm, ugos=ugos, vgos=vgos,
             spd=np.hypot(ugos, vgos), eid=eid, det=det, time=t,
             cycle=cycle, pas=pas,
             extent=[np.nanmin(lon[has]) - 0.3, np.nanmax(lon[has]) + 0.3,
                     np.nanmin(lat[has]) - 0.2, np.nanmax(lat[has]) + 0.2])

    for mode in ("ssha", "speed", "adt"):
        pair_figure(d, mode, nc_path.parent / f"{stem}_hybrid_pair_{mode}.png")


if __name__ == "__main__":
    main()
