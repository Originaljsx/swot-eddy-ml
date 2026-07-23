"""Per-pass 3-panel gallery: SSHA | ADT + front line | SSHA + hybrid detections.

Geographic axes with coastline, one PNG per pass, publication styling. Detections come
from the HYBRID (PET proposes closed contours, v2 U-Net scores frontal-ness, tau=0.9):
violet = frontal, gray = other PET eddies, both drawn as their true PET contours.

Usage:
  .venv/bin/python scripts/make_pass_gallery.py cycle_049:091 cycle_048:132 ...
  .venv/bin/python scripts/make_pass_gallery.py --default   # 12 test-cycle passes

Writes out/aoi5/pass_gallery/pass_<cycle>_<pass>.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*partition.*will ignore the 'mask'.*")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import datetime as dt
import re

import matplotlib.pyplot as plt
import torch
from skimage.draw import polygon as sk_polygon

from swot_eddy.config import load_config
from swot_eddy.aoi import load_aoi
from swot_eddy import preprocess as pp
from swot_eddy.detect import detect_pass_split
from swot_eddy.features import find_granule, native_fields, front_vertices, raw_channels
from swot_eddy import unet as U

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
C_FRONTAL, C_OTHER, C_FRONT = "#7b3294", "#666666", "#B8860B"
TAU = 0.9

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/aoi5/pass_gallery"
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
aoi = load_aoi(cfg)
LEVEL = cfg["features"]["front_level_m"]

try:
    import cartopy.crs as ccrs
    PC = ccrs.PlateCarree()
except Exception:
    PC = None

model = U.build_unet(in_ch=4)
model.load_state_dict(torch.load(ROOT / "out/aoi5/model/unet_frontal_v2.pt",
                                 map_location="cpu", weights_only=True))
model.eval()

DEFAULT = ["cycle_043:091", "cycle_043:326", "cycle_044:188", "cycle_044:410",
           "cycle_045:369", "cycle_046:091", "cycle_047:354", "cycle_048:132",
           "cycle_048:369", "cycle_049:091", "cycle_049:020", "cycle_050:035"]


def geo_ax(fig, rect):
    if PC is not None:
        ax = fig.add_axes(rect, projection=PC)
        ax.coastlines("10m", linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.85", linestyle=":")
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
        return ax
    return fig.add_axes(rect)


def one_pass(cyc, pnum):
    g = find_granule(cfg["data_root"], cyc, pnum)
    if g is None:
        print(f"  no granule {cyc}/{pnum}")
        return
    m = re.search(r"_(\d{8})T(\d{6})_", g.name)
    date = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

    tiles, summ, clon, clat = pp.preprocess_pass(str(g), cfg)
    kept, _, _ = detect_pass_split(tiles, cfg, clon, clat, date)
    kept = [d for d in kept if aoi.contains(d.center_lon, d.center_lat)]
    x, _ = raw_channels(g, cfg)
    x = x.astype(np.float32)
    L = x.shape[1]
    prob = np.zeros((L, 128), np.float32)
    hits = np.zeros((L, 128), np.float32)
    with torch.no_grad():
        for s, xt in U.pass_tiles(x):
            p = torch.sigmoid(model(torch.from_numpy(U.normalize(xt))[None]))[0, 0].numpy()
            e = min(s + U.TILE, L)
            prob[s:e] += p[:e - s]
            hits[s:e] += 1
    prob /= np.maximum(hits, 1)
    scores = []
    for d in kept:
        mask = np.zeros(prob.shape, bool)
        if d.contour_pix:
            cp = np.array(d.contour_pix)
            rr, cc = sk_polygon(cp[:, 0], cp[:, 1], shape=prob.shape)
            mask[rr, cc] = True
        scores.append(float(prob[mask].max()) if mask.any() else 0.0)

    nf = native_fields(g, cfg)
    front = front_vertices(nf["adt"], nf["lon"], nf["lat"], LEVEL, 20)
    ssha_n = nf["adt"] * np.nan
    pad_left = (128 - nf["lon"].shape[1]) // 2
    ssha_n = x[0, :, pad_left:pad_left + nf["lon"].shape[1]] * 100
    v1, v2 = np.nanpercentile(ssha_n, 2), np.nanpercentile(ssha_n, 98)
    kw = {"transform": PC} if PC else {}

    def det_lonlat(d):
        pts = [p for p in (d.contour_lonlat or []) if p is not None]
        if not pts:
            return None
        arr = np.array(pts)
        return ((arr[:, 0] + 180) % 360) - 180, arr[:, 1]

    n_frontal = sum(s > TAU for s in scores)
    fig = plt.figure(figsize=(8.6, 6.2))
    panels = [(ssha_n, "RdBu_r", v1, v2, "SSHA (cm)"),
              (nf["adt"], "Spectral_r", -0.1, 0.8, f"ADT (m) + {LEVEL:.2f} m front"),
              (ssha_n, "RdBu_r", v1, v2, f"hybrid detections ({n_frontal} frontal)")]
    for j, (fld, cmap, vmin, vmax, ttl) in enumerate(panels):
        ax = geo_ax(fig, [0.05 + j * 0.32, 0.09, 0.28, 0.80])
        pm = ax.pcolormesh(nf["lon"], nf["lat"], fld, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="nearest", **kw)
        if j == 1 and len(front):
            ax.plot(front[:, 0], front[:, 1], ".", color=C_FRONT, ms=1.0, **kw)
        if j == 2:
            for d, s in zip(kept, scores):
                ll = det_lonlat(d)
                if ll is None:
                    continue
                frontal = s > TAU
                ax.plot(ll[0], ll[1], color=C_FRONTAL if frontal else C_OTHER,
                        lw=1.8 if frontal else 0.8, **kw)
        ax.set_title(f"({'abc'[j]}) {ttl}", fontsize=8)
        cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.06, shrink=0.9,
                          aspect=30)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"{cyc.replace('cycle_', 'cycle ')} / pass {pnum} — "
                 f"{date:%Y-%m-%d} — {len(kept)} PET eddies, {n_frontal} frontal "
                 f"(violet; gray = non-frontal)", fontsize=9, y=0.97)
    out = OUT / f"pass_{cyc}_{pnum}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {cyc}/{pnum}: {len(kept)} eddies, {n_frontal} frontal -> {out.name}")


def main():
    args = sys.argv[1:]
    targets = DEFAULT if (not args or args[0] == "--default") else args
    for t in targets:
        cyc, pnum = t.split(":")
        one_pass(cyc, pnum)


if __name__ == "__main__":
    main()
