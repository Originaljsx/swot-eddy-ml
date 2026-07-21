"""Render real-data PNG assets for the pipeline video (run with the repo .venv).

Produces in pipeline-video/assets/:
  adt_full.png, ssha_full.png, grad_adt.png    example pass, dark-themed, no axes
  front_overlay.png                            ADT + 0.35 m front line + labeled frontal rings
  tile_before.png, tile_after.png              real gappy tile pre/post biharmonic inpaint
  eddy_zoom_1..3.png                           confirmed frontal eddies, SSHA zoom + PET contour
  meta.json                                    lon/lat extents + eddy positions for manim
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from swot_eddy.config import load_config
from swot_eddy import features as ft
from swot_eddy import preprocess as pp

OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)
BG = "#111418"
GOLD = "#e3b341"

cfg = load_config()
lab = pd.read_csv(REPO / "out/aoi5/labels/eddy_labels_aoi5.csv", dtype={"cycle": str, "pass": str})
geom = pd.read_csv(REPO / "out/aoi5/catalog/contours_geom.csv", dtype={"cycle": str, "pass": str})
verts = pd.read_csv(REPO / "out/aoi5/catalog/contours.csv", dtype={"cycle": str, "pass": str})
frontal = lab[lab.label == "frontal"].merge(geom, on=["cycle", "pass", "kind", "eddy"])

# example pass: labeled pass with >=3 frontal eddies AND best data coverage
cands = (frontal.groupby(["cycle", "pass"]).size().sort_values(ascending=False)
         .loc[lambda s: s >= 3].head(8))
best, best_cov = None, -1.0
for (c, p), n in cands.items():
    g = ft.find_granule(cfg["data_root"], c, p)
    if g is None:
        continue
    cov = float(np.isfinite(ft.native_fields(g, cfg)["adt"]).mean())
    if cov > best_cov:
        best, best_cov = (c, p), cov
cyc, pnum = best
fr_pass = frontal[(frontal.cycle == cyc) & (frontal["pass"] == pnum)]
print(f"example pass {cyc}/{pnum}: {len(fr_pass)} frontal eddies, coverage {best_cov:.0%}")

granule = ft.find_granule(cfg["data_root"], cyc, pnum)
nf = ft.native_fields(granule, cfg)
# crop all fields to the finite-data bounding box (kills black margins)
fin = np.isfinite(nf["adt"])
rr, cc = np.where(fin)
r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
for k in ("lon", "lat", "adt", "mdt", "grad_adt"):
    nf[k] = nf[k][r0:r1, c0:c1]
level = cfg["features"]["front_level_m"]
front = ft.front_vertices(nf["adt"], nf["lon"], nf["lat"], level,
                          cfg["features"]["front_min_vertices"])


def save_field(field, cmap, vmin, vmax, name, overlay=None):
    fig, ax = plt.subplots(figsize=(4.2, 7), facecolor=BG)
    ax.pcolormesh(nf["lon"], nf["lat"], field, cmap=cmap, vmin=vmin, vmax=vmax,
                  shading="nearest")
    if overlay == "front":
        ax.plot(front[:, 0], front[:, 1], ".", color=GOLD, ms=1.5)
    if overlay == "front+eddies":
        ax.plot(front[:, 0], front[:, 1], ".", color=GOLD, ms=1.5)
        for e in fr_pass.itertuples():
            ax.plot(e.lon, e.lat, "o", color=GOLD, ms=14, mfc="none", mew=2.5)
    ax.set_aspect(1 / np.cos(np.deg2rad(float(np.nanmean(nf["lat"])))))
    ax.axis("off")
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor=BG)
    plt.close(fig)
    print("wrote", name)


ssha = None
import xarray as xr
var_map = cfg["product"]["variables"][cfg["product"]["product_version"]]
with xr.open_dataset(str(granule)) as ds:
    dc, _ = pp.crop_to_region(ds, cfg["region"])
    ssha = dc[var_map["ssha"]].values[r0:r1, c0:c1] * 100.0

save_field(nf["adt"], "Spectral_r", -0.1, 0.8, "adt_full.png")
save_field(ssha, "RdBu_r", -10, 30, "ssha_full.png")
save_field(nf["grad_adt"], "turbo", 0, 0.02, "grad_adt.png", overlay="front")
save_field(nf["adt"], "Spectral_r", -0.1, 0.8, "front_overlay.png", overlay="front+eddies")

# --- real tile before/after inpaint -----------------------------------------
tiles, summ, clon, clat = pp.preprocess_pass(str(granule), cfg)
# most-gappy tile that's still <60% missing (visually interesting, not empty)
gap_fracs = [(t.inpaint_mask[0].mean(), i) for i, t in enumerate(tiles)]
gf, ti = max((g, i) for g, i in gap_fracs if g < 0.6)
t = tiles[ti]
before = np.where(t.inpaint_mask[0], np.nan, t.data[0]) * 100
after = t.data[0] * 100
for arr, name in ((before, "tile_before.png"), (after, "tile_after.png")):
    fig, ax = plt.subplots(figsize=(4, 4), facecolor=BG)
    ax.imshow(arr, cmap="RdBu_r", vmin=-10, vmax=30, origin="lower")
    ax.axis("off")
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor=BG)
    plt.close(fig)
print(f"tile {ti}: {gf:.0%} inpainted -> tile_before/after.png")

# --- eddy zooms: 3 confirmed frontal eddies with PET contour ----------------
full = np.full((summ["n_lines_cropped"], 128), np.nan)
for tt in tiles:
    h = tt.data.shape[1]
    e = min(tt.line_start + h, full.shape[0])
    full[tt.line_start:e, :] = tt.data[0, :e - tt.line_start, :] * 100

# biggest confirmed frontal eddies dataset-wide (one granule load each)
picks = frontal.sort_values("radius_km", ascending=False).head(3)
for k, e in enumerate(picks.to_dict("records"), 1):
    g = ft.find_granule(cfg["data_root"], e["cycle"], e["pass"])
    tls, smm, _, _ = pp.preprocess_pass(str(g), cfg)
    fl = np.full((smm["n_lines_cropped"], 128), np.nan)
    for tt in tls:
        h = tt.data.shape[1]
        en = min(tt.line_start + h, fl.shape[0])
        fl[tt.line_start:en, :] = tt.data[0, :en - tt.line_start, :] * 100
    vv = verts[(verts.cycle == e["cycle"]) & (verts["pass"] == e["pass"]) &
               (verts.kind == e["kind"]) & (verts.eddy == e["eddy"])]
    r = max(30, int(e["radius_km"]))
    i0, i1 = max(0, e["center_line"] - r), min(fl.shape[0], e["center_line"] + r)
    j0, j1 = max(0, e["center_col"] - r), min(128, e["center_col"] + r)
    win = fl[i0:i1, j0:j1]
    vmin, vmax = np.nanpercentile(win, 2), np.nanpercentile(win, 98)
    col = "#58a6ff" if e["polarity"] == "cyclonic" else "#f85149"
    fig, ax = plt.subplots(figsize=(3.6, 3.6), facecolor=BG)
    ax.imshow(win, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower",
              extent=[j0, j1, i0, i1])
    ax.plot(vv.col, vv.line, color=col, lw=2.5)
    ax.plot(e["center_col"], e["center_line"], "o", color=col, ms=8, mfc="none", mew=2)
    ax.axis("off")
    fig.savefig(OUT / f"eddy_zoom_{k}.png", dpi=150, bbox_inches="tight",
                pad_inches=0.02, facecolor=BG)
    plt.close(fig)
    print(f"eddy_zoom_{k}.png: {e['cycle']}/{e['pass']} R={e['radius_km']:.0f}km "
          f"amp={e['amplitude_m']*100:.1f}cm {e['polarity']}")

meta = {"pass": f"{cyc}/{pnum}", "level_m": level,
        "n_frontal": len(fr_pass), "tile_gap_frac": round(float(gf), 2)}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print("meta:", meta)
