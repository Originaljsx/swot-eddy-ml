"""Publication figures for the frontal-eddy pipeline + v2 U-Net. Read-only inputs.

Writes out/aoi5/paper_figs/*.(png|pdf), 300 dpi, serif, panel letters:
  study_region_aoi5_labels_and_front            AOI5 polygon, swath ADT, front, labels
  unet_input_channels_and_frontal_mask_target   training sample: channels + target
  adt_front_prior_validation_distance_and_auc   signed-distance by class + AUC sweep
  unet_predictions_vs_labels_test_passes        two held-out passes, truth vs prediction
  precision_recall_vs_threshold_and_confidence  P/R/F1 sweep + confidence distributions
  raw_pass_ssha_adt_and_detections              raw pass: SSHA / ADT+front / detections

Conventions: Okabe-Ito colorblind-safe class colors; predictions violet, confirmed
labels green throughout; all model panels from TEST cycles (43-50) only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy import ndimage

from swot_eddy.config import load_config
from swot_eddy.aoi import load_aoi
from swot_eddy import features as ft
from swot_eddy.unet import build_unet, normalize, pass_tiles, TILE

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "figure.dpi": 120,
})
C_FRONTAL, C_COASTAL, C_OPEN = "#D55E00", "#009E73", "#0072B2"   # Okabe-Ito
C_TRUTH, C_PRED, C_FRONT = "#1a9641", "#7b3294", "#B8860B"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/aoi5/paper_figs"
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
aoi = load_aoi(cfg)
LEVEL = cfg["features"]["front_level_m"]

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    PC = ccrs.PlateCarree()
except Exception:
    PC = None


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def label_panel(ax, s):
    ax.text(0.02, 0.985, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))


def geo_ax(fig, rect):
    if PC is not None:
        ax = fig.add_axes(rect, projection=PC)
        ax.coastlines("10m", linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.8", linestyle=":")
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 7}
        return ax
    ax = fig.add_axes(rect)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    return ax


# ---------------------------------------------------------------- fig 1: region
lab = pd.read_csv(ROOT / "out/aoi5/labels/eddy_labels_aoi5.csv",
                  dtype={"cycle": str, "pass": str})
cyc, pnum = "cycle_001", "341"
nf = ft.native_fields(ft.find_granule(cfg["data_root"], cyc, pnum), cfg)
front = ft.front_vertices(nf["adt"], nf["lon"], nf["lat"], LEVEL, 20)

fig = plt.figure(figsize=(5.2, 5.4))
ax = geo_ax(fig, [0.1, 0.08, 0.85, 0.88])
pm = ax.pcolormesh(nf["lon"], nf["lat"], nf["adt"], cmap="Spectral_r",
                   vmin=-0.1, vmax=0.8, shading="nearest",
                   **({"transform": PC} if PC else {}))
geoms = aoi.geom.geoms if aoi.geom.geom_type == "MultiPolygon" else [aoi.geom]
for g in geoms:
    xs, ys = g.exterior.xy
    ax.plot(xs, ys, color="k", lw=1.0, ls="--", **({"transform": PC} if PC else {}))
ax.plot(front[:, 0], front[:, 1], ".", color=C_FRONT, ms=1.2,
        **({"transform": PC} if PC else {}))
for cls, col in (("frontal", C_FRONTAL), ("coastal", C_COASTAL), ("open ocean", C_OPEN)):
    s = lab[lab.label == cls]
    ax.plot(s.lon, s.lat, "o", color=col, ms=1.6, mew=0, alpha=0.6, linestyle="none",
            label=f"{cls} (n={len(s)})", **({"transform": PC} if PC else {}))
if PC:
    ax.set_extent([-82.5, -68, 23, 42], crs=PC)
ax.legend(loc="lower right", markerscale=3, framealpha=0.9)
fig.colorbar(pm, ax=ax, shrink=0.65, pad=0.02).set_label("ADT (m)")
ax.set_title(f"Study region: GS_AOI5 (dashed), example swath ADT ({cyc}/{pnum}),\n"
             f"ADT {LEVEL:.2f} m front line (gold), hand-labeled eddies (all cycles)")
save(fig, "study_region_aoi5_labels_and_front")

# ------------------------------------------------------- fig 2: training sample
with np.load(ROOT / "out/aoi5/training_v2" / f"{cyc}_{pnum}.npz") as d:
    x2, y2 = d["x"].astype(np.float32), d["y"]
fig, axes = plt.subplots(1, 4, figsize=(6.8, 5.2))
panels = [(x2[0] * 100, "RdBu_r", -10, 30, "SSHA (cm)"),
          (x2[1], "Spectral_r", -0.1, 0.8, "ADT (m)"),
          (x2[2] * 1000, "magma", 0, 20, r"$|\nabla \mathrm{ADT}|$ (mm km$^{-1}$)"),
          (x2[0] * 100, "RdBu_r", -10, 30, "target mask")]
for i, (ax, (fld, cmap, vmin, vmax, ttl)) in enumerate(zip(axes, panels)):
    im = ax.imshow(fld, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax.set_title(ttl)
    ax.set_xticks([]); ax.set_yticks([])
    label_panel(ax, f"({'abcd'[i]})")
    if i < 3:
        fig.colorbar(im, ax=ax, shrink=0.5, pad=0.03)
axes[3].contour(y2, levels=[0.5], colors=C_TRUTH, linewidths=1.1)
axes[0].set_ylabel("along-track pixel (2 km)")
fig.suptitle("U-Net training sample: native input channels and confirmed-frontal mask",
             y=0.98)
save(fig, "unet_input_channels_and_frontal_mask_target")

# ------------------------------------------------- fig 3: front-prior validation
fd = pd.read_csv(ROOT / "out/aoi5/features/front_dists_labeled.csv")
sweep = pd.read_csv(ROOT / "out/aoi5/features/front_level_sweep.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
bins = np.arange(-400, 401, 25)
for cls, col in (("frontal", C_FRONTAL), ("coastal", C_COASTAL), ("open ocean", C_OPEN)):
    axes[0].hist(fd.loc[fd.label == cls, "dist_front_adt_km"], bins=bins, alpha=0.6,
                 color=col, label=cls)
axes[0].axvline(0, color="k", lw=0.7)
axes[0].set_xlabel("signed distance to ADT front (km)")
axes[0].set_ylabel("labeled eddies")
axes[0].legend(frameon=False)
label_panel(axes[0], "(a)")
for field, col, mk in (("adt", "k", "o"), ("mdt", "0.55", "s")):
    s = sweep[sweep.field == field]
    axes[1].plot(s.level_m, s.auc_frontal_vs_open, marker=mk, ms=3, lw=1, color=col,
                 label=f"{field.upper()} front")
axes[1].axvline(LEVEL, color=C_FRONT, lw=0.9, ls="--")
axes[1].set_xlabel("contour level (m)")
axes[1].set_ylabel("AUC (frontal vs open ocean)")
axes[1].set_ylim(0.55, 1.0)
axes[1].legend(frameon=False)
label_panel(axes[1], "(b)")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
save(fig, "adt_front_prior_validation_distance_and_auc")

# ------------------------------------------------------------- fig 4/5: model
model = build_unet(in_ch=4)
model.load_state_dict(torch.load(ROOT / "out/aoi5/model/unet_frontal_v2.pt",
                                 map_location="cpu", weights_only=True))
model.eval()
LAM = float(np.load(ROOT / "out/aoi5/model/lam_hat_v2.npy")[0])


def run(x):
    L = x.shape[1]
    prob = np.zeros((L, 128), np.float32); hits = np.zeros((L, 128), np.float32)
    with torch.no_grad():
        for s, xt in pass_tiles(x):
            p = torch.sigmoid(model(torch.from_numpy(normalize(xt))[None]))[0, 0].numpy()
            e = min(s + TILE, L); prob[s:e] += p[:e - s]; hits[s:e] += 1
    return prob / np.maximum(hits, 1)


PAD_LEFT = (128 - 69) // 2
show = [("cycle_043", "091"), ("cycle_048", "369")]
fig = plt.figure(figsize=(6.8, 6.4))
for i, (c, p) in enumerate(show):
    with np.load(ROOT / "out/aoi5/training_v2" / f"{c}_{p}.npz") as d:
        x, y = d["x"].astype(np.float32), d["y"]
    prob = run(x)
    nfp = ft.native_fields(ft.find_granule(cfg["data_root"], c, p), cfg)
    nat = slice(PAD_LEFT, PAD_LEFT + nfp["lon"].shape[1])
    ssha_n = x[0, :, nat] * 100
    prob_n = np.where(x[3, :, nat] > 0, prob[:, nat], np.nan)   # hide no-data
    v1_, v2_ = np.nanpercentile(ssha_n, 2), np.nanpercentile(ssha_n, 98)
    for j, (fld, cmap, vmin, vmax, ttl) in enumerate((
            (ssha_n, "RdBu_r", v1_, v2_, "SSHA (cm)"),
            (prob_n, "magma", 0, 1, "P(frontal)"))):
        ax = geo_ax(fig, [0.07 + (2 * i + j) * 0.23, 0.06, 0.21, 0.86])
        pm = ax.pcolormesh(nfp["lon"], nfp["lat"], fld, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="nearest", **({"transform": PC} if PC else {}))
        for arr, col, ls in ((y[:, nat], C_TRUTH, "-"), (prob[:, nat] > 0.5, C_PRED, "--")):
            ll, ln = ndimage.label(arr)
            for k in range(1, ln + 1):
                m = ll == k
                ys_, xs_ = np.where(m)
                ax.plot(np.take(nfp["lon"], np.ravel_multi_index((ys_, xs_), nfp["lon"].shape)),
                        np.take(nfp["lat"], np.ravel_multi_index((ys_, xs_), nfp["lat"].shape)),
                        ".", color=col, ms=0.8, **({"transform": PC} if PC else {}))
        label_panel(ax, f"({'abcd'[2 * i + j]})")
        ax.set_title(f"{c.replace('cycle_', 'c')}/{p}\n{ttl}", fontsize=7.5)
fig.suptitle("v2 U-Net on held-out passes — green: confirmed frontal, violet: prediction",
             y=0.985, fontsize=9)
save(fig, "unet_predictions_vs_labels_test_passes")

# fig 5: quantitative
man = pd.read_csv(ROOT / "out/aoi5/training_v2/training_manifest.csv",
                  dtype={"cycle": str, "pass": str})
test = man[(man.split == "test") & (man.status == "ok")]
passes = []
for c, p in zip(test.cycle, test["pass"]):
    with np.load(ROOT / "out/aoi5/training_v2" / f"{c}_{p}.npz") as d:
        x, y = d["x"].astype(np.float32), d["y"]
    passes.append((y, run(x)))

taus = np.linspace(0.05, 0.99, 40)
P, R = [], []
ms, es = [], []
for y, prob in passes:
    pred = prob > 0.5
    pl, pn = ndimage.label(pred)
    hp = set(np.unique(pl[y > 0])) - {0}
    for k in range(1, pn + 1):
        (ms if k in hp else es).append(float(prob[pl == k].max()))
for tau in taus:
    TP = FP = FN = 0
    for y, prob in passes:
        pred = prob > tau
        tl, tn = ndimage.label(y); pl, pn = ndimage.label(pred)
        ht = set(np.unique(tl[pred])) - {0}; hp = set(np.unique(pl[y > 0])) - {0}
        TP += len(ht); FN += tn - len(ht); FP += pn - len(hp)
    P.append(TP / max(TP + FP, 1)); R.append(TP / max(TP + FN, 1))

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
axes[0].plot(taus, P, color=C_FRONTAL, lw=1.4, label="precision")
axes[0].plot(taus, R, color=C_OPEN, lw=1.4, label="recall")
axes[0].plot(taus, [2 * p * r / max(p + r, 1e-9) for p, r in zip(P, R)],
             color="0.4", lw=1.1, ls=":", label="F1")
axes[0].axvline(LAM, color=C_FRONT, ls="--", lw=0.9)
axes[0].text(LAM + 0.015, 0.08, rf"$\hat\lambda$={LAM:.2f}", fontsize=7, color=C_FRONT)
axes[0].set_xlabel("detection threshold")
axes[0].set_ylabel("eddy-level score")
axes[0].set_ylim(0, 1.02)
axes[0].legend(frameon=False, loc="center left")
label_panel(axes[0], "(a)")
bins = np.linspace(0, 1, 31)
axes[1].hist(ms, bins=bins, alpha=0.65, color=C_OPEN, label=f"matched (n={len(ms)})")
axes[1].hist(es, bins=bins, alpha=0.65, color=C_FRONTAL, label=f"unmatched (n={len(es)})")
axes[1].set_xlabel("detection confidence (blob max probability)")
axes[1].set_ylabel("detections")
axes[1].legend(frameon=False, loc="upper left")
label_panel(axes[1], "(b)")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
save(fig, "precision_recall_vs_threshold_and_confidence")

# --------------------------- fig 6: raw pass -> SSHA / ADT / detections
C6, P6 = "cycle_049", "091"
g6 = ft.find_granule(cfg["data_root"], C6, P6)
nf6 = ft.native_fields(g6, cfg)
front6 = ft.front_vertices(nf6["adt"], nf6["lon"], nf6["lat"], LEVEL, 20)
x6, _ = ft.raw_channels(g6, cfg)
prob6 = run(x6.astype(np.float32))
nat6 = slice(PAD_LEFT, PAD_LEFT + nf6["lon"].shape[1])
mask6 = prob6[:, nat6] > LAM
lbl6, n6 = ndimage.label(mask6)
ssha6 = nf6["adt"] * 0 + x6[0, :, nat6] * 100      # native-grid SSHA in cm
v1_, v2_ = np.nanpercentile(ssha6, 2), np.nanpercentile(ssha6, 98)

fig = plt.figure(figsize=(6.8, 5.6))
panels = [(ssha6, "RdBu_r", v1_, v2_, "SSHA (cm)", "SSHA (cm)"),
          (nf6["adt"], "Spectral_r", -0.1, 0.8, "ADT (m)", "ADT (m)"),
          (ssha6, "RdBu_r", v1_, v2_, "SSHA (cm)", "detected frontal eddies")]
for j, (fld, cmap, vmin, vmax, cblab, ttl) in enumerate(panels):
    ax = geo_ax(fig, [0.06 + j * 0.31, 0.10, 0.27, 0.80])
    pm = ax.pcolormesh(nf6["lon"], nf6["lat"], fld, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="nearest", **({"transform": PC} if PC else {}))
    if j == 1 and len(front6):
        ax.plot(front6[:, 0], front6[:, 1], ".", color=C_FRONT, ms=1.0,
                **({"transform": PC} if PC else {}))
    if j == 2:
        for k in range(1, n6 + 1):
            ys_, xs_ = np.where(lbl6 == k)
            ax.plot(nf6["lon"][ys_, xs_], nf6["lat"][ys_, xs_], ".", color=C_PRED,
                    ms=1.0, **({"transform": PC} if PC else {}))
            cy, cx = (int(v) for v in ndimage.center_of_mass(lbl6 == k))
            ax.annotate(str(k), (nf6["lon"][cy, cx], nf6["lat"][cy, cx]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7,
                        **({"transform": PC} if PC else {}))
    label_panel(ax, f"({'abc'[j]})")
    ax.set_title(ttl, fontsize=8.5)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.07, shrink=0.9,
                      aspect=28)
    cb.set_label(cblab, fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
fig.suptitle(f"Raw SWOT pass {C6.replace('cycle_', 'cycle ')}/{P6}: input fields and "
             f"U-Net frontal-eddy detections (threshold $\\hat\\lambda$={LAM:.2f})",
             y=0.985, fontsize=9)
save(fig, "raw_pass_ssha_adt_and_detections")
print("done ->", OUT)
