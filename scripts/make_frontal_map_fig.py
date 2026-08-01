"""All frontal eddies (all 50 cycles, every pass) on the study-region map.

Same styling as paper fig study_region_aoi5_labels_and_front (AOI5 polygon dashed,
example swath ADT, gold 0.35 m ADT front line) but frontal only.

Two sources:
  default            hand-confirmed labels (out/aoi5/release/labeled_eddies_aoi5.csv)
                     -> frontal_eddies_all_cycles.(png|pdf)
  --detections CSV   hybrid-model detections from run_hybrid_all.py
                     (rows with is_frontal == True)
                     -> frontal_eddies_hybrid_all_cycles.(png|pdf)

Usage:
  .venv/bin/python scripts/make_frontal_map_fig.py
  .venv/bin/python scripts/make_frontal_map_fig.py --detections out/aoi5/batch/hybrid_detections.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from swot_eddy.config import load_config
from swot_eddy.aoi import load_aoi
from swot_eddy import features as ft

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
    "axes.titlesize": 8.5, "legend.fontsize": 7, "figure.dpi": 120,
})
C_FRONTAL, C_FRONT = "#D55E00", "#B8860B"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/aoi5/paper_figs"
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
aoi = load_aoi(cfg)
LEVEL = cfg["features"]["front_level_m"]

try:
    import cartopy.crs as ccrs
    PC = ccrs.PlateCarree()
except Exception:
    PC = None

ap = argparse.ArgumentParser()
ap.add_argument("--detections", default=None,
                help="hybrid_detections.csv from run_hybrid_all.py")
args = ap.parse_args()

if args.detections:
    rel = pd.read_csv(args.detections)
    fro = rel[rel.is_frontal].dropna(subset=["lon", "lat"])
    src_desc = f"hybrid detections (PET x U-Net, tau=0.9), n={len(fro)}"
    out_name = "frontal_eddies_hybrid_all_cycles"
else:
    rel = pd.read_csv(ROOT / "out/aoi5/release/labeled_eddies_aoi5.csv")
    fro = rel[rel.label == "frontal"]
    src_desc = f"hand-confirmed frontal eddies, n={len(fro)}"
    out_name = "frontal_eddies_all_cycles"

cyc, pnum = "cycle_001", "341"
nf = ft.native_fields(ft.find_granule(cfg["data_root"], cyc, pnum), cfg)
front = ft.front_vertices(nf["adt"], nf["lon"], nf["lat"], LEVEL, 20)

kw = {"transform": PC} if PC else {}
fig = plt.figure(figsize=(5.2, 5.4))
if PC is not None:
    ax = fig.add_axes([0.1, 0.08, 0.85, 0.88], projection=PC)
    ax.coastlines("10m", linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.8", linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
else:
    ax = fig.add_axes([0.1, 0.08, 0.85, 0.88])
pm = ax.pcolormesh(nf["lon"], nf["lat"], nf["adt"], cmap="Spectral_r",
                   vmin=-0.1, vmax=0.8, shading="nearest", **kw)
geoms = aoi.geom.geoms if aoi.geom.geom_type == "MultiPolygon" else [aoi.geom]
for g in geoms:
    xs, ys = g.exterior.xy
    ax.plot(xs, ys, color="k", lw=1.0, ls="--", **kw)
ax.plot(front[:, 0], front[:, 1], ".", color=C_FRONT, ms=1.2, **kw)
ax.plot(fro.lon, fro.lat, "o", color=C_FRONTAL, ms=2.2, mew=0, alpha=0.65,
        linestyle="none", label=f"frontal eddies (n={len(fro)})", **kw)
if PC:
    env = aoi.envelope_region()          # full corrected AOI, incl. eastern Gulf
    ax.set_extent([env["lon_min"] - 0.5, env["lon_max"] + 0.5,
                   env["lat_min"] - 0.5, env["lat_max"] + 0.5], crs=PC)
ax.legend(loc="lower right", markerscale=2.5, framealpha=0.9)
fig.colorbar(pm, ax=ax, shrink=0.65, pad=0.02).set_label("ADT (m)")
ax.set_title(f"Frontal eddies, all 50 cycles / every pass — {src_desc}\n"
             f"GS_AOI5 (dashed), example swath ADT ({cyc}/{pnum}), "
             f"ADT {LEVEL:.2f} m front (gold)")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"{out_name}.{ext}", dpi=300, bbox_inches="tight")
print("wrote", OUT / f"{out_name}.png")
