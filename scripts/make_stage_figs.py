"""Visualize EVERY stage of the pipeline on one real granule: input .nc -> final output.

Renders the actual intermediate arrays (not schematics) for both branches and the fusion:

  step01_input_granule.png       raw ssha_filtered on the real swath (gaps white)
  step02_crop_band.png           cropped along-track band, native 69-px width
  step03_pad_and_tiles.png       128-px padded frame with the 128x128/64 tile grid
  step04_inpaint.png             biharmonic fill: gappy tile before/after + full field
  step05_pet_contours.png        PET closed contours: kept vs auto-rejected (gap/nadir)
  step06_pet_candidates.png      after trust rejection + AOI filter: the PET candidates
  step07_unet_inputs.png         the U-Net's 4 raw channels (no inpainting)
  step08_unet_probability.png    stitched P(frontal) field + conformal threshold
  step09_hybrid_scoring.png      PET contours colored by U-Net score
  step10_final_output.png        the answer: frontal (violet) vs other eddies (gray)
  pipeline_storyboard.png        all ten steps in one strip

Usage:  .venv/bin/python scripts/make_stage_figs.py [GRANULE.nc]
Writes to out/aoi5/stage_figs/. Default granule: cycle_049 pass 091 (held-out test).
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
import matplotlib.pyplot as plt
import torch
import xarray as xr
from matplotlib.patches import Rectangle
from skimage.draw import polygon as sk_polygon

from swot_eddy.config import load_config
from swot_eddy.aoi import load_aoi
from swot_eddy import preprocess as pp
from swot_eddy.detect import detect_pass_split
from swot_eddy.features import find_granule, raw_channels
from swot_eddy import unet as U

BG = "white"
C_KEEP, C_REJ, C_FRONTAL, C_OTHER = "#0072B2", "#999999", "#7b3294", "#777777"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.facecolor": BG})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/aoi5/stage_figs"
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
aoi = load_aoi(cfg)
TAU = 0.9

granule = Path(sys.argv[1]) if len(sys.argv) > 1 else find_granule(
    cfg["data_root"], "cycle_049", "091")
print("granule:", granule.name)
var_map = cfg["product"]["variables"][cfg["product"]["product_version"]]

steps = []          # (path, caption) for the storyboard


def save(fig, name, caption):
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    steps.append((p, caption))
    print("wrote", name)


# ---- step 1: raw granule ------------------------------------------------------
with xr.open_dataset(str(granule)) as ds:
    lon_all = ((ds.longitude.values + 180) % 360) - 180
    lat_all = ds.latitude.values
    ssha_all = ds[var_map["ssha"]].values * 100
fig, ax = plt.subplots(figsize=(4.2, 6))
sub = slice(None, None, 6)                     # decimate global pass for plotting
ax.scatter(lon_all[sub], lat_all[sub], c=ssha_all[sub], s=1, cmap="RdBu_r",
           vmin=-15, vmax=30)
ax.set_title(f"1. input granule (global pass)\n{granule.name.split('_2')[0]}")
ax.set_xlabel("lon"); ax.set_ylabel("lat")
save(fig, "step01_input_granule",
     "1 · Input .nc — one SWOT half-orbit, pole to pole; the study region is a small band of it.")

# ---- step 2: crop -------------------------------------------------------------
with xr.open_dataset(str(granule)) as ds:
    dc, sl = pp.crop_to_region(ds, cfg["region"])
    ssha_c = dc[var_map["ssha"]].values * 100
fig, ax = plt.subplots(figsize=(3.2, 7))
ax.imshow(ssha_c, cmap="RdBu_r", vmin=-15, vmax=30, origin="lower", aspect="auto")
ax.set_title(f"2. crop to region band\n{ssha_c.shape[0]} lines x 69 px (white = gaps)")
ax.set_xlabel("across-track px"); ax.set_ylabel("along-track line")
save(fig, "step02_crop_band",
     "2 · Crop — the along-track band intersecting the study region; native 69-pixel swath, NaN gaps kept.")

# ---- step 3: pad + tiles ------------------------------------------------------
stack, names, _ = pp.compute_channels(dc, var_map, None)
padded, pad_left = pp.pad_across_track(stack, 128)
starts = pp.tile_starts(padded.shape[1], 128, 64)
fig, ax = plt.subplots(figsize=(3.6, 7))
ax.imshow(padded[0] * 100, cmap="RdBu_r", vmin=-15, vmax=30, origin="lower", aspect="auto")
for s in starts:
    ax.add_patch(Rectangle((0, s), 128, 128, fill=False, edgecolor="k", lw=0.8))
ax.axvline(pad_left, color="k", ls=":", lw=0.8)
ax.axvline(pad_left + 69, color="k", ls=":", lw=0.8)
ax.set_title(f"3. pad 69->128 + tile\n{len(starts)} tiles, 64-px overlap")
save(fig, "step03_pad_and_tiles",
     "3 · Pad & tile — NaN-pad across-track to 128, cut 128x128 windows with 50% overlap so every eddy is whole in some tile.")

# ---- step 4: inpaint ----------------------------------------------------------
tiles, summ, clon, clat = pp.preprocess_pass(str(granule), cfg)
L = summ["n_lines_cropped"]
gaps = [(t.inpaint_mask[0].mean(), i) for i, t in enumerate(tiles)]
gfrac, ti = max((g, i) for g, i in gaps if g < 0.7)
t = tiles[ti]
before = np.where(t.inpaint_mask[0], np.nan, t.data[0]) * 100
full_inp = np.full((L, 128), np.nan)
for tt in tiles:
    h = tt.data.shape[1]
    e = min(tt.line_start + h, L)
    full_inp[tt.line_start:e, :] = tt.data[0, :e - tt.line_start, :] * 100
fig, axes = plt.subplots(1, 3, figsize=(8.5, 5))
axes[0].imshow(before, cmap="RdBu_r", vmin=-15, vmax=30, origin="lower")
axes[0].set_title(f"tile {ti}: {gfrac:.0%} missing")
axes[1].imshow(t.data[0] * 100, cmap="RdBu_r", vmin=-15, vmax=30, origin="lower")
axes[1].set_title("after biharmonic fill\n$\\nabla^4 u = 0$ in gaps")
axes[2].imshow(full_inp, cmap="RdBu_r", vmin=-15, vmax=30, origin="lower", aspect="auto")
axes[2].set_title("full inpainted pass")
for a in axes:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("4. biharmonic inpainting (PET needs closed contours -> no gaps allowed)")
save(fig, "step04_inpaint",
     "4 · Inpaint — every gap filled by solving the biharmonic equation; PET cannot close contours across holes. Filled pixels are remembered.")

# ---- step 5+6: PET ------------------------------------------------------------
import datetime as dt
import re
m = re.search(r"_(\d{8})T(\d{6})_", granule.name)
date = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
kept_all, rejected_all, dsum = detect_pass_split(tiles, cfg, clon, clat, date)

fig, ax = plt.subplots(figsize=(3.6, 7))
ax.imshow(full_inp, cmap="RdBu_r", vmin=-15, vmax=30, origin="lower", aspect="auto")
for d in rejected_all:
    if d.contour_pix:
        cp = np.array(d.contour_pix)
        ax.plot(cp[:, 1], cp[:, 0], color=C_REJ, lw=1.0, ls="--")
for d in kept_all:
    if d.contour_pix:
        cp = np.array(d.contour_pix)
        ax.plot(cp[:, 1], cp[:, 0], color=C_KEEP, lw=1.4)
ax.set_title(f"5. PET closed contours\nblue kept={len(kept_all)}, "
             f"gray dashed auto-rejected={len(rejected_all)}")
save(fig, "step05_pet_contours",
     "5 · PET identification — closed SSH contours passing shape/amplitude/radius tests. Dashed gray: contours living mostly on inpainted or nadir pixels, auto-rejected.")

kept = [d for d in kept_all if aoi.contains(d.center_lon, d.center_lat)]
fig, ax = plt.subplots(figsize=(3.6, 7))
ax.imshow(full_inp, cmap="RdBu_r", vmin=-15, vmax=30, origin="lower", aspect="auto")
for d in kept:
    if d.contour_pix:
        cp = np.array(d.contour_pix)
        col = "#d62728" if d.polarity == "anticyclonic" else C_KEEP
        ax.plot(cp[:, 1], cp[:, 0], color=col, lw=1.5)
        ax.plot(d.center_col, d.center_line, "o", color=col, ms=4, mfc="none")
ax.set_title(f"6. PET candidates after AOI filter\n{len(kept)} eddies "
             "(blue cyclonic / red anticyclonic)")
save(fig, "step06_pet_candidates",
     "6 · PET candidates — deduplicated across tile overlaps, filtered to the study polygon. These are ALL closed-contour eddies; frontal-ness is still unknown.")

# ---- step 7: U-Net inputs -----------------------------------------------------
x_raw, _ = raw_channels(granule, cfg)
x_raw = x_raw.astype(np.float32)
fig, axes = plt.subplots(1, 4, figsize=(9, 5.5))
for a, (fld, cmap, vmin, vmax, ttl) in zip(axes, (
        (x_raw[0] * 100, "RdBu_r", -15, 30, "SSHA (cm)"),
        (x_raw[1], "Spectral_r", -0.1, 0.8, "ADT (m)"),
        (x_raw[2] * 1000, "magma", 0, 20, "|∇ADT| (mm/km)"),
        (x_raw[3], "gray", 0, 1, "validity"))):
    a.imshow(fld, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    a.set_title(ttl); a.set_xticks([]); a.set_yticks([])
fig.suptitle("7. U-Net input: 4 RAW channels — no inpainting, gaps stay gaps")
save(fig, "step07_unet_inputs",
     "7 · U-Net branch input — native fields plus a validity mask; the network learns to work around the gaps instead of trusting invented pixels.")

# ---- step 8: probability ------------------------------------------------------
model = U.build_unet(in_ch=4)
model.load_state_dict(torch.load(ROOT / "out/aoi5/model/unet_frontal_v2.pt",
                                 map_location="cpu", weights_only=True))
model.eval()
LAM = float(np.load(ROOT / "out/aoi5/model/lam_hat_v2.npy")[0])
prob = np.zeros((L, 128), np.float32)
hits = np.zeros((L, 128), np.float32)
with torch.no_grad():
    for s, xt in U.pass_tiles(x_raw):
        p = torch.sigmoid(model(torch.from_numpy(U.normalize(xt))[None]))[0, 0].numpy()
        e = min(s + U.TILE, L)
        prob[s:e] += p[:e - s]
        hits[s:e] += 1
prob /= np.maximum(hits, 1)
fig, axes = plt.subplots(1, 2, figsize=(6, 6.5))
pm = axes[0].imshow(np.where(x_raw[3] > 0, prob, np.nan), cmap="magma", vmin=0, vmax=1,
                    origin="lower", aspect="auto")
axes[0].set_title("P(frontal)")
axes[1].imshow(np.where(x_raw[3] > 0, prob > LAM, np.nan), cmap="gray", vmin=0, vmax=1,
               origin="lower", aspect="auto")
axes[1].set_title(f"P > λ̂ = {LAM:.2f} (conformal)")
for a in axes:
    a.set_xticks([]); a.set_yticks([])
fig.colorbar(pm, ax=axes[0], shrink=0.6)
fig.suptitle("8. U-Net output: frontal probability per pixel (128x128 tiles, stitched)")
save(fig, "step08_unet_probability",
     "8 · U-Net output — per-pixel frontal probability, tiles averaged over overlaps; the conformal threshold carries a ≤10% miss-rate guarantee.")

# ---- step 9: hybrid scoring ---------------------------------------------------
scores = []
for d in kept:
    mask = np.zeros(prob.shape, bool)
    if d.contour_pix:
        cp = np.array(d.contour_pix)
        rr, cc = sk_polygon(cp[:, 0], cp[:, 1], shape=prob.shape)
        mask[rr, cc] = True
    scores.append(float(prob[mask].max()) if mask.any() else 0.0)
fig, ax = plt.subplots(figsize=(3.9, 7))
ax.imshow(full_inp, cmap="RdBu_r", vmin=-15, vmax=30, origin="lower", aspect="auto",
          alpha=0.55)
sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
for d, s in zip(kept, scores):
    if d.contour_pix:
        cp = np.array(d.contour_pix)
        ax.plot(cp[:, 1], cp[:, 0], color=sm.to_rgba(s), lw=2.0)
fig.colorbar(sm, ax=ax, shrink=0.6).set_label("U-Net score inside contour")
ax.set_title("9. hybrid scoring\neach PET contour gets max P(frontal) inside it")
save(fig, "step09_hybrid_scoring",
     "9 · Fusion — each PET contour is scored with the U-Net probability inside it. The distribution is bimodal: frontal eddies ≈1, everything else ≈0.")

# ---- step 10: final -----------------------------------------------------------
nfrontal = sum(s > TAU for s in scores)
fig, ax = plt.subplots(figsize=(3.9, 7))
ax.imshow(full_inp, cmap="RdBu_r", vmin=-15, vmax=30, origin="lower", aspect="auto")
for d, s in zip(kept, scores):
    if d.contour_pix:
        cp = np.array(d.contour_pix)
        frontal = s > TAU
        ax.plot(cp[:, 1], cp[:, 0], color=C_FRONTAL if frontal else C_OTHER,
                lw=2.2 if frontal else 1.0)
ax.set_title(f"10. OUTPUT: {nfrontal} frontal (violet) of {len(kept)} eddies\n"
             "-> CSV + GeoJSON + this map")
save(fig, "step10_final_output",
     f"10 · Final output — {nfrontal} frontal eddies (violet, τ={TAU}) among {len(kept)} closed-contour eddies; written as per-eddy CSV and MapLibre GeoJSON.")

# ---- storyboard ----------------------------------------------------------------
from matplotlib.image import imread
fig = plt.figure(figsize=(22, 7))
n = len(steps)
for i, (p, cap) in enumerate(steps):
    ax = fig.add_axes([i / n + 0.003, 0.05, 1 / n - 0.006, 0.88])
    ax.imshow(imread(p))
    ax.axis("off")
    if i < n - 1:
        fig.text((i + 1) / n, 0.5, "→", ha="center", fontsize=18)
fig.suptitle(f"SWOT frontal-eddy pipeline, every stage on one real pass — {granule.stem}",
             fontsize=13, y=0.99)
fig.savefig(OUT / "pipeline_storyboard.png", dpi=110, bbox_inches="tight", facecolor=BG)
print("wrote pipeline_storyboard")
print("all ->", OUT)
