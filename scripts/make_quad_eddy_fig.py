"""Quad figure in the F1_v15 style, panel (b) = hybrid eddy detections.

Layout follows Figures/lee_waves/v15_unsmoothed_pass369_quad/F1_v15.py:

  (a) sea surface height anomaly (ssha_filtered), cm
  (b) SSHA + hybrid eddy detections: filled-contour outline from eddy_flag,
      dashed bounding box per detection, id + frontal score label
      (violet = frontal, gray = non-frontal PET eddy)
  (c) geostrophic speed = hypot(ugos_filtered, vgos_filtered), m/s
  (d) absolute dynamic topography (ssha_unedited + mdt) + geostrophic quiver

Input is the *_hybrid_eddies.nc written by export_eddy_nc.py (original Expert
granule + eddy_flag/eddy_id); frontal scores come from the sibling *_hybrid.csv
written by predict_hybrid.py.

Usage:
  .venv/bin/python scripts/make_quad_eddy_fig.py out/aoi5/predictions/<stem>_hybrid_eddies.nc

Writes <stem>_hybrid_quad.png next to the input.
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
GEOS_SUBSAMPLE = 6            # 2 km grid -> 12 km spacing (extent ~3x the F1 box)
GEOS_SCALE, GEOS_WIDTH, GEOS_KEY_MS = 20, 0.0015, 0.5
BOX_MARGIN_DEG = 0.15         # bbox padding around each eddy


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eddies_nc", help="*_hybrid_eddies.nc from export_eddy_nc.py")
    args = ap.parse_args()

    nc_path = Path(args.eddies_nc)
    stem = nc_path.name.replace("_hybrid_eddies.nc", "")
    csv_path = nc_path.parent / f"{stem}_hybrid.csv"
    import pandas as pd
    det = pd.read_csv(csv_path).set_index("id")

    cfg = load_config()
    with xr.open_dataset(nc_path) as ds:
        sub, _ = pp.crop_to_region(ds, cfg["region"])
        lat = sub["latitude"].values
        lon = _wrap180(sub["longitude"].values)
        ssha_cm = sub["ssha_filtered"].values * 100.0
        adt_cm = (sub["ssha_unedited"].values + sub["mdt"].values) * 100.0
        ugos = sub["ugos_filtered"].values
        vgos = sub["vgos_filtered"].values
        flag = sub["eddy_flag"].values
        eid = sub["eddy_id"].values
        t = str(sub["time"].values[len(sub["time"]) // 2])[:19]
    spd = np.hypot(ugos, vgos)

    m = re.search(r"_(\d{3})_(\d{3})_\d{8}T", nc_path.name)
    cycle, pas = (m.group(1), m.group(2)) if m else ("?", "?")

    # frame only where the swath actually has data (crop band runs further north)
    has = np.isfinite(ssha_cm) | np.isfinite(adt_cm)
    extent = [np.nanmin(lon[has]) - 0.3, np.nanmax(lon[has]) + 0.3,
              np.nanmin(lat[has]) - 0.2, np.nanmax(lat[has]) + 0.2]
    pc = ccrs.PlateCarree()
    panel_w = 6.0
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    figsize = (2 * panel_w + 1.0, 2 * (panel_w * aspect + 2.0))
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor="w",
                             constrained_layout=True,
                             subplot_kw={"projection": pc})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    clim = ssha_clim(ssha_cm)
    pa = ax_a.pcolormesh(lon, lat, np.ma.masked_invalid(ssha_cm), cmap=SSHA_CMAP,
                         vmin=-clim, vmax=clim, shading="auto", transform=pc)
    style_axis(ax_a, "a", extent)
    ax_a.set_title("Sea surface height anomaly")
    fig.colorbar(pa, ax=ax_a, orientation="horizontal", pad=0.02,
                 shrink=0.9).set_label("height anomaly (cm)")

    # (b) SSHA backdrop + eddy outlines from eddy_flag + per-eddy dashed bbox
    pb = ax_b.pcolormesh(lon, lat, np.ma.masked_invalid(ssha_cm), cmap=SSHA_CMAP,
                         vmin=-clim, vmax=clim, alpha=0.55, shading="auto",
                         transform=pc)
    n_frontal = 0
    for i in sorted(det.index):
        mask = eid == i
        if not mask.any():
            continue
        frontal = bool(det.loc[i, "is_frontal"])
        n_frontal += frontal
        color = C_FRONTAL if frontal else C_OTHER
        ax_b.contour(lon, lat, mask.astype(float), levels=[0.5], colors=[color],
                     linewidths=1.8 if frontal else 0.9, transform=pc)
        lo0, lo1 = lon[mask].min() - BOX_MARGIN_DEG, lon[mask].max() + BOX_MARGIN_DEG
        la0, la1 = lat[mask].min() - BOX_MARGIN_DEG, lat[mask].max() + BOX_MARGIN_DEG
        ax_b.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0],
                  color=color, lw=1.2 if frontal else 0.7, ls="--", transform=pc)
        ax_b.text(lo0, la1 + 0.03, f"{i}  {det.loc[i, 'frontal_score']:.2f}",
                  color=color, fontsize=8, fontweight="bold" if frontal else "normal",
                  ha="left", va="bottom", transform=pc,
                  bbox=dict(facecolor="w", edgecolor="none", alpha=0.6, pad=0.8))
    style_axis(ax_b, "b", extent)
    ax_b.set_title(f"Hybrid eddy detections ({n_frontal} frontal of {len(det)})")
    fig.colorbar(pb, ax=ax_b, orientation="horizontal", pad=0.02,
                 shrink=0.9).set_label("height anomaly (cm)")

    pcm = ax_c.pcolormesh(lon, lat, np.ma.masked_invalid(spd), cmap=SPEED_CMAP,
                          vmin=SPEED_CLIM[0], vmax=SPEED_CLIM[1], shading="auto",
                          transform=pc, zorder=3)
    style_axis(ax_c, "c", extent)
    ax_c.set_title("Geostrophic speed")
    fig.colorbar(pcm, ax=ax_c, orientation="horizontal", pad=0.02,
                 shrink=0.9).set_label(r"geostrophic speed (m s$^{-1}$)")

    aclim = adt_clim(adt_cm)
    pdm = ax_d.pcolormesh(lon, lat, np.ma.masked_invalid(adt_cm), cmap=ADT_CMAP,
                          vmin=aclim[0], vmax=aclim[1], shading="auto",
                          transform=pc, zorder=3)
    for col in swath_boundary_cols(adt_cm):
        # break the edge line where the column has no data (crop band > swath)
        lon_c = np.where(np.isfinite(adt_cm[:, col]), lon[:, col], np.nan)
        ax_d.plot(lon_c, lat[:, col], "k", linewidth=1.4, transform=pc, zorder=7)
    n = GEOS_SUBSAMPLE
    q = ax_d.quiver(lon[::n, ::n], lat[::n, ::n], ugos[::n, ::n], vgos[::n, ::n],
                    color="k", scale=GEOS_SCALE, width=GEOS_WIDTH, transform=pc,
                    zorder=8)
    ax_d.quiverkey(q, 0.86, 0.04, GEOS_KEY_MS, f"{GEOS_KEY_MS} m s$^{{-1}}$",
                   labelpos="E", coordinates="axes", fontproperties={"size": 9})
    style_axis(ax_d, "d", extent)
    ax_d.set_title("Absolute dynamic topography and geostrophic velocity")
    fig.colorbar(pdm, ax=ax_d, orientation="horizontal", pad=0.02,
                 shrink=0.9).set_label("absolute dynamic topography (cm)")

    fig.suptitle(f"SWOT L3 Expert   pass {pas}   cycle {cycle}   "
                 f"{t.replace('T', ' ')} UTC", fontsize=14)
    out = nc_path.parent / f"{stem}_hybrid_quad.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
