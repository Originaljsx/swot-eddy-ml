"""Visual report of U-Net results on the held-out TEST cycles (43-50). Read-only inputs.

Writes to out/aoi5/report/:
  pass_<cycle>_<pass>.png   4-panel: SSHA+labels | ADT | P(frontal) | prediction vs labels
  gallery_tp.png / gallery_fp.png / gallery_fn.png   zoomed eddy crops by outcome
  scores_hist.png           per-eddy max-prob distributions (cal vs test) + thresholds
  report_data.json          metrics + per-pass stats for the HTML report

"Labels" = hand-confirmed frontal eddies from the PET+ADT bootstrap — treated as a
lower bound on truth, NOT ground truth (FP may be real frontals PET missed).
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
from scipy import ndimage

from swot_eddy.unet import build_unet, normalize, pass_tiles, TILE

BG = "#111418"
TRUTH_C = "#2ea043"     # green  = hand-confirmed frontal
PRED_C = "#d2a8ff"      # violet = U-Net prediction
ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "out/aoi5/training"
OUT = ROOT / "out/aoi5/report"
OUT.mkdir(parents=True, exist_ok=True)

model = build_unet()
model.load_state_dict(torch.load(ROOT / "out/aoi5/model/unet_frontal.pt",
                                 map_location="cpu", weights_only=True))
model.eval()
lam_hat = float(np.load(ROOT / "out/aoi5/model/lam_hat.npy")[0])


def predict(x):
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


man = pd.read_csv(TRAIN / "training_manifest.csv", dtype={"cycle": str, "pass": str})
splits = {}
for split in ("cal", "test"):
    rows = []
    for c, p in zip(*(lambda s: (s.cycle, s["pass"]))(man[(man.split == split) & (man.status == "ok")])):
        with np.load(TRAIN / f"{c}_{p}.npz") as d:
            x, y = d["x"].astype(np.float32), d["y"]
        rows.append((c, p, x, y, predict(x)))
    splits[split] = rows

# ---- per-eddy outcomes on TEST ------------------------------------------------
crops = {"tp": [], "fp": [], "fn": []}
scores = {"cal": [], "test": []}
pass_stats = []
for split, rows in splits.items():
    for c, p, x, y, prob in rows:
        pred = prob > 0.5
        tl, tn = ndimage.label(y)
        for k in range(1, tn + 1):
            scores[split].append(float(prob[tl == k].max()))
        if split != "test":
            continue
        pl, pn = ndimage.label(pred)
        hit_t = set(np.unique(tl[pred])) - {0}
        hit_p = set(np.unique(pl[y > 0])) - {0}
        pass_stats.append({"cycle": c, "pass": p, "n_true": int(tn), "n_pred": int(pn),
                           "tp": len(hit_t), "fn": int(tn - len(hit_t)),
                           "fp": int(pn - len(hit_p))})
        def crop(cy, cx, tag, score):
            r = 26
            i0, i1 = max(0, cy - r), min(y.shape[0], cy + r)
            j0, j1 = max(0, cx - r), min(128, cx + r)
            crops[tag].append((x[0, i0:i1, j0:j1] * 100, y[i0:i1, j0:j1],
                               (prob[i0:i1, j0:j1] > 0.5), score, f"{c[-3:]}/{p}"))
        for k in range(1, tn + 1):
            cy, cx = (int(v) for v in ndimage.center_of_mass(tl == k))
            s = float(prob[tl == k].max())
            crop(cy, cx, "tp" if k in hit_t else "fn", s)
        for k in range(1, pn + 1):
            if k not in hit_p:
                cy, cx = (int(v) for v in ndimage.center_of_mass(pl == k))
                crop(cy, cx, "fp", float(prob[pl == k].max()))

# ---- figures ------------------------------------------------------------------
def draw_contours(ax, y, pred):
    if y.any():
        ax.contour(y, levels=[0.5], colors=TRUTH_C, linewidths=1.8)
    if pred.any():
        ax.contour(pred, levels=[0.5], colors=PRED_C, linewidths=1.4, linestyles="--")


best = sorted(pass_stats, key=lambda r: -(r["tp"] + r["fp"]))[:6]
for r in best:
    c, p = r["cycle"], r["pass"]
    x, y, prob = next((x, y, pr) for cc, pp, x, y, pr in splits["test"] if cc == c and pp == p)
    pred = prob > 0.5
    fig, axes = plt.subplots(1, 4, figsize=(13, 8), facecolor=BG)
    panels = [(x[0] * 100, "RdBu_r", -10, 30, "SSHA (cm) + confirmed frontal"),
              (x[1], "Spectral_r", -0.1, 0.8, "ADT (m)"),
              (prob, "magma", 0, 1, "P(frontal)"),
              (x[0] * 100, "RdBu_r", -10, 30, "prediction vs labels")]
    for ax, (fld, cmap, vmin, vmax, ttl) in zip(axes, panels):
        ax.imshow(fld, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
        ax.set_title(ttl, fontsize=9, color="w")
        ax.axis("off")
    draw_contours(axes[0], y, np.zeros_like(y))
    draw_contours(axes[3], y, pred)
    axes[2].contour(pred, levels=[0.5], colors=PRED_C, linewidths=1.0)
    fig.suptitle(f"TEST {c}/{p} — {r['tp']} matched, {r['fp']} extra predictions, "
                 f"{r['fn']} missed  (green=hand label, violet=U-Net)",
                 color="w", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"pass_{c}_{p}.png", dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

rng = np.random.default_rng(0)
for tag, title in (("tp", "detected confirmed frontals (TP)"),
                   ("fp", "extra predictions (FP — possibly PET-missed frontals)"),
                   ("fn", "missed confirmed frontals (FN)")):
    items = crops[tag]
    if not items:
        continue
    take = [items[i] for i in rng.permutation(len(items))[:16]]
    n = len(take)
    cols = 4
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(11, 2.9 * rows_n), facecolor=BG)
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (ssha, y, pred, s, tag2) in zip(np.ravel(axes), take):
        vmin, vmax = np.nanpercentile(ssha, 2), np.nanpercentile(ssha, 98)
        ax.imshow(ssha, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        draw_contours(ax, y, pred)
        ax.set_title(f"{tag2}  p={s:.2f}", fontsize=8, color="w")
    fig.suptitle(f"{title} — {len(items)} total, showing {n}", color="w", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"gallery_{tag}.png", dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
bins = np.linspace(0, 1, 41)
ax.hist(scores["cal"], bins=bins, alpha=0.65, label=f"cal eddies (n={len(scores['cal'])})",
        color="#6e40c9")
ax.hist(scores["test"], bins=bins, alpha=0.65, label=f"test eddies (n={len(scores['test'])})",
        color="#1f6feb")
ax.axvline(0.5, color="#57606a", ls="--", lw=1.2)
ax.text(0.51, ax.get_ylim()[1] * 0.9, "0.5 mask threshold", fontsize=8, color="#57606a")
ax.axvline(max(lam_hat, 0.004), color="#cf222e", ls="--", lw=1.2)
ax.text(max(lam_hat, 0.004) + 0.01, ax.get_ylim()[1] * 0.75,
        f"conformal lam_hat={lam_hat:.3f}\n(degenerate: weak-eddy tail)", fontsize=8,
        color="#cf222e")
ax.set_xlabel("max P(frontal) inside each confirmed frontal eddy")
ax.set_ylabel("eddies")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "scores_hist.png", dpi=130, bbox_inches="tight")
plt.close(fig)

TP = sum(r["tp"] for r in pass_stats)
FP = sum(r["fp"] for r in pass_stats)
FN = sum(r["fn"] for r in pass_stats)
data = {"tp": TP, "fp": FP, "fn": FN,
        "precision": round(TP / (TP + FP), 3), "recall": round(TP / (TP + FN), 3),
        "lam_hat": lam_hat, "n_cal": len(scores["cal"]), "n_test": len(scores["test"]),
        "weak_cal": int(np.sum(np.array(scores["cal"]) < 0.1)),
        "best_passes": [f"{r['cycle']}_{r['pass']}" for r in best]}
(OUT / "report_data.json").write_text(json.dumps(data, indent=1))
print(json.dumps(data, indent=1))
print("figures ->", OUT)
