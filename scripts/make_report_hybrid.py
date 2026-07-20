"""Aggregate visual report for the HYBRID (PET x U-Net) detector on test cycles 43-50.

Scores every labeled PET detection in the test cycles with the v2 U-Net (max prob inside
the PET speed contour) and produces graphs in out/aoi5/report_hybrid/:

  method_comparison.png   P/R/F1 bars: ADT-distance rule vs U-Net alone vs hybrid
  score_by_class.png      U-Net score distribution of PET detections per hand label
  pr_vs_tau.png           hybrid precision/recall/F1 over the tau sweep
  class_outcome.png       what tau=0.9 keeps vs rejects, per label class
  map_test_frontals.png   geographic map of hybrid frontal detections (score-colored)
  radius_amplitude.png    eddy-scale distributions: hybrid frontals vs all PET eddies
  report_hybrid.json      the numbers
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from skimage.draw import polygon as sk_polygon

from swot_eddy.unet import build_unet, normalize, pass_tiles, TILE

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
C_FRONTAL, C_COASTAL, C_OPEN, C_REJ = "#D55E00", "#009E73", "#0072B2", "#8b949e"
CLASS_C = {"frontal": C_FRONTAL, "coastal": C_COASTAL, "open ocean": C_OPEN,
           "reject": C_REJ}
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/aoi5/report_hybrid"
OUT.mkdir(parents=True, exist_ok=True)
TAU = 0.9

model = build_unet(in_ch=4)
model.load_state_dict(torch.load(ROOT / "out/aoi5/model/unet_frontal_v2.pt",
                                 map_location="cpu", weights_only=True))
model.eval()


def run(x):
    L = x.shape[1]
    prob = np.zeros((L, 128), np.float32)
    hits = np.zeros((L, 128), np.float32)
    with torch.no_grad():
        for s, xt in pass_tiles(x):
            p = torch.sigmoid(model(torch.from_numpy(normalize(xt))[None]))[0, 0].numpy()
            e = min(s + TILE, L)
            prob[s:e] += p[:e - s]
            hits[s:e] += 1
    return prob / np.maximum(hits, 1)


man = pd.read_csv(ROOT / "out/aoi5/training_v2/training_manifest.csv",
                  dtype={"cycle": str, "pass": str})
test = man[(man.split == "test") & (man.status == "ok")]
lab = pd.read_csv(ROOT / "out/aoi5/labels/eddy_labels_aoi5.csv",
                  dtype={"cycle": str, "pass": str})
lab = lab[lab.cycle.isin(set(test.cycle)) & (lab.kind == "keep") & (lab.label != "skip")]
verts = pd.read_csv(ROOT / "out/aoi5/catalog/contours.csv",
                    dtype={"cycle": str, "pass": str})
cat = pd.read_csv(ROOT / "out/aoi5/catalog/detections.csv",
                  dtype={"cycle": str, "pass": str})
cat["eddy"] = cat.groupby(["cycle", "pass", "kind"]).cumcount()

rows = []
for (c, p), sub in lab.groupby(["cycle", "pass"]):
    f = ROOT / "out/aoi5/training_v2" / f"{c}_{p}.npz"
    if not f.exists():
        continue
    with np.load(f) as d:
        x = d["x"].astype(np.float32)
    prob = run(x)
    vv = verts[(verts.cycle == c) & (verts["pass"] == p) & (verts.kind == "keep")]
    cc_ = cat[(cat.cycle == c) & (cat["pass"] == p) & (cat.kind == "keep")]
    for e in sub.to_dict("records"):
        g = vv[vv.eddy == e["eddy"]]
        meta = cc_[cc_.eddy == e["eddy"]]
        if not len(g) or not len(meta):
            continue
        m = np.zeros(prob.shape, bool)
        rr, ccp = sk_polygon(g["line"].values, g["col"].values, shape=prob.shape)
        m[rr, ccp] = True
        if not m.any():
            continue
        meta = meta.iloc[0]
        rows.append({"cycle": c, "pass": p, "label": e["label"],
                     "score": float(prob[m].max()),
                     "lon": float(meta.lon), "lat": float(meta.lat),
                     "radius_km": float(meta.radius_km),
                     "amplitude_m": float(meta.amplitude_m),
                     "polarity": meta.polarity})
df = pd.DataFrame(rows)
df.to_csv(OUT / "hybrid_scored_detections.csv", index=False)
print(f"{len(df)} labeled PET detections scored; "
      f"{(df.label == 'frontal').sum()} frontal")

is_f = df.label == "frontal"

# ---- method comparison --------------------------------------------------------
methods = {
    "ADT rule\n(cyclonic, <50 km)": (0.84, 0.72),
    "U-Net alone": (0.414, 0.959),
    f"HYBRID PET x U-Net\n(tau={TAU})": None,
}
sel = df[df.score > TAU]
P = (sel.label == "frontal").mean()
R = (sel.label == "frontal").sum() / is_f.sum()
methods[f"HYBRID PET x U-Net\n(tau={TAU})"] = (P, R)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
xpos = np.arange(len(methods))
for i, (metric, off, col) in enumerate((("precision", -0.22, C_FRONTAL),
                                        ("recall", 0.0, C_OPEN),
                                        ("F1", 0.22, "#555555"))):
    vals = []
    for name, (p_, r_) in methods.items():
        vals.append({"precision": p_, "recall": r_,
                     "F1": 2 * p_ * r_ / (p_ + r_)}[metric])
    b = ax.bar(xpos + off, vals, width=0.2, color=col, label=metric)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", fontsize=8)
ax.set_xticks(xpos, methods.keys())
ax.set_ylim(0, 1.1)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncols=3, loc="upper left")
ax.set_title("Frontal-eddy detection: method comparison (test cycles 43–50)")
fig.tight_layout()
fig.savefig(OUT / "method_comparison.png", dpi=150)
plt.close(fig)

# ---- score by class -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.6))
bins = np.linspace(0, 1, 41)
for cls in ("frontal", "coastal", "open ocean", "reject"):
    s = df.loc[df.label == cls, "score"]
    if len(s):
        ax.hist(s, bins=bins, alpha=0.6, color=CLASS_C[cls],
                label=f"{cls} (n={len(s)})")
ax.axvline(TAU, color="k", ls="--", lw=1)
ax.text(TAU - 0.02, ax.get_ylim()[1] * 0.95, f"tau={TAU}", ha="right", fontsize=8)
ax.set_xlabel("U-Net max probability inside PET contour")
ax.set_ylabel("PET detections")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
ax.set_title("What the U-Net says about each PET detection, by hand label")
fig.tight_layout()
fig.savefig(OUT / "score_by_class.png", dpi=150)
plt.close(fig)

# ---- PR vs tau ----------------------------------------------------------------
taus = np.linspace(0.02, 0.995, 60)
Ps, Rs = [], []
for t in taus:
    s = df[df.score > t]
    Ps.append((s.label == "frontal").mean() if len(s) else 1.0)
    Rs.append((s.label == "frontal").sum() / is_f.sum())
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(taus, Ps, color=C_FRONTAL, lw=1.8, label="precision")
ax.plot(taus, Rs, color=C_OPEN, lw=1.8, label="recall")
ax.plot(taus, [2 * p * r / max(p + r, 1e-9) for p, r in zip(Ps, Rs)],
        color="#555555", lw=1.4, ls=":", label="F1")
ax.axvline(TAU, color="k", ls="--", lw=0.9)
ax.set_xlabel("tau (frontal threshold on U-Net score)")
ax.set_ylim(0, 1.03)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower left")
ax.set_title("Hybrid performance vs threshold")
fig.tight_layout()
fig.savefig(OUT / "pr_vs_tau.png", dpi=150)
plt.close(fig)

# ---- class outcome at tau -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.4))
classes = ["frontal", "coastal", "open ocean", "reject"]
kept = [int(((df.label == c) & (df.score > TAU)).sum()) for c in classes]
rej = [int(((df.label == c) & (df.score <= TAU)).sum()) for c in classes]
xpos = np.arange(len(classes))
ax.bar(xpos, kept, 0.55, color=[CLASS_C[c] for c in classes], label="kept as frontal")
ax.bar(xpos, rej, 0.55, bottom=kept, color=[CLASS_C[c] for c in classes], alpha=0.25,
       label="rejected")
for i, (k, r) in enumerate(zip(kept, rej)):
    ax.text(i, k + r + 1.5, f"{k}/{k + r}", ha="center", fontsize=8)
ax.set_xticks(xpos, classes)
ax.set_ylabel("PET detections")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
ax.set_title(f"Hybrid decision at tau={TAU}: solid = called frontal (want: only orange solid)")
fig.tight_layout()
fig.savefig(OUT / "class_outcome.png", dpi=150)
plt.close(fig)

# ---- geographic map of hybrid frontals ---------------------------------------
try:
    import cartopy.crs as ccrs
    PC = ccrs.PlateCarree()
except Exception:
    PC = None
fig = plt.figure(figsize=(7, 7))
if PC is not None:
    ax = fig.add_axes([0.08, 0.06, 0.86, 0.88], projection=PC)
    ax.coastlines("10m", linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.85", linestyle=":")
    gl.top_labels = gl.right_labels = False
else:
    ax = fig.add_axes([0.08, 0.06, 0.86, 0.88])
kw = {"transform": PC} if PC else {}
hyb = df[df.score > TAU]
other = df[df.score <= TAU]
ax.scatter(other.lon, other.lat, s=10, c="0.75", label="PET eddy, not frontal", **kw)
sc = ax.scatter(hyb.lon, hyb.lat, s=26, c=hyb.score, cmap="plasma", vmin=TAU, vmax=1.0,
                edgecolors="k", linewidths=0.3, label="hybrid frontal", **kw)
if PC:
    ax.set_extent([-82.5, -68, 23, 42], crs=PC)
fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02).set_label("U-Net score")
ax.legend(loc="lower right")
ax.set_title(f"Hybrid frontal eddies, test cycles 43–50 (n={len(hyb)})")
fig.savefig(OUT / "map_test_frontals.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- radius/amplitude ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
axes[0].hist(df.radius_km, bins=np.arange(5, 45, 2.5), alpha=0.4, color="0.5",
             label="all PET eddies")
axes[0].hist(hyb.radius_km, bins=np.arange(5, 45, 2.5), alpha=0.75, color=C_FRONTAL,
             label="hybrid frontal")
axes[0].set_xlabel("speed radius (km)")
axes[0].set_ylabel("eddies")
axes[1].hist(df.amplitude_m * 100, bins=np.arange(0, 25, 1.25), alpha=0.4, color="0.5")
axes[1].hist(hyb.amplitude_m * 100, bins=np.arange(0, 25, 1.25), alpha=0.75,
             color=C_FRONTAL)
axes[1].set_xlabel("amplitude (cm)")
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)
fig.suptitle("Scale of detected frontal eddies")
fig.tight_layout()
fig.savefig(OUT / "radius_amplitude.png", dpi=150)
plt.close(fig)

F1 = 2 * P * R / (P + R)
data = {"n_scored": len(df), "n_frontal_true": int(is_f.sum()),
        "tau": TAU, "precision": round(float(P), 3), "recall": round(float(R), 3),
        "f1": round(float(F1), 3), "n_hybrid_frontal": len(hyb),
        "kept_by_class": dict(zip(classes, kept)),
        "polarity_of_hybrid": dict(hyb.polarity.value_counts())}
(OUT / "report_hybrid.json").write_text(json.dumps(data, indent=1, default=int))
print(json.dumps(data, indent=1, default=int))
