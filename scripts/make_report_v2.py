"""Deep visual report for the v2 (raw-input) U-Net on held-out test cycles 43-50.

Focus: SHOW what the model sees and does, not metrics.
Writes to out/aoi5/report_v2/:
  inputs_<pass>.png    the 4 channels the net ingests (native SSHA/ADT/|gradADT|/validity)
                       + the actual normalized tensor
  outputs_<pass>.png   probability map, mask @0.5, mask @lam_hat, overlay vs labels
  extras_anatomy_[12].png  8 extra (unmatched) detections, one row each:
                       SSHA | ADT | |gradADT| | P(frontal)  -- what fooled it
  misses_anatomy.png   every missed confirmed frontal, same row layout
  confidence_hist.png  blob max-prob: matched vs extra -- can a threshold separate them?
  pr_vs_threshold.png  eddy-level precision & recall over the threshold sweep
  saliency.png         input-gradient saliency for 2 matched + 2 extra detections
  report_v2.json       metrics + counts
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
TRUTH_C = "#3fb950"
PRED_C = "#d2a8ff"
ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "out/aoi5/training_v2"
OUT = ROOT / "out/aoi5/report_v2"
OUT.mkdir(parents=True, exist_ok=True)

model = build_unet(in_ch=4)
model.load_state_dict(torch.load(ROOT / "out/aoi5/model/unet_frontal_v2.pt",
                                 map_location="cpu", weights_only=True))
model.eval()
LAM = float(np.load(ROOT / "out/aoi5/model/lam_hat_v2.npy")[0])


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
test = man[(man.split == "test") & (man.status == "ok")]
passes = []
for c, p in zip(test.cycle, test["pass"]):
    with np.load(TRAIN / f"{c}_{p}.npz") as d:
        x, y = d["x"].astype(np.float32), d["y"]
    passes.append((c, p, x, y, predict(x)))

# ---- blob bookkeeping ---------------------------------------------------------
matched, extra, missed = [], [], []   # (cyc, p, x, y, prob, blobmask, score)
for c, p, x, y, prob in passes:
    pred = prob > 0.5
    tl, tn = ndimage.label(y)
    pl, pn = ndimage.label(pred)
    hit_t = set(np.unique(tl[pred])) - {0}
    hit_p = set(np.unique(pl[y > 0])) - {0}
    for k in range(1, pn + 1):
        blob = pl == k
        rec = (c, p, x, y, prob, blob, float(prob[blob].max()))
        (matched if k in hit_p else extra).append(rec)
    for k in range(1, tn + 1):
        if k not in hit_t:
            missed.append((c, p, x, y, prob, tl == k, float(prob[tl == k].max())))
print(f"blobs: matched {len(matched)}, extra {len(extra)}, missed frontals {len(missed)}")

CH = [("SSHA (cm)", "RdBu_r", -10, 30, lambda x: x[0] * 100),
      ("ADT (m)", "Spectral_r", -0.1, 0.8, lambda x: x[1]),
      ("|∇ADT| (m/km)", "turbo", 0, 0.02, lambda x: x[2]),
      ("validity", "gray", 0, 1, lambda x: x[3])]

# ---- per-pass INPUT figures ---------------------------------------------------
show = [("cycle_043", "091"), ("cycle_048", "369"), ("cycle_049", "091")]
for cyc, pn_ in show:
    c, p, x, y, prob = next(t for t in passes if t[0] == cyc and t[1] == pn_)
    xt_norm = normalize(x)
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 8), facecolor=BG)
    for ax, (ttl, cmap, vmin, vmax, fn) in zip(axes, CH):
        ax.imshow(fn(x), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
        ax.set_title(ttl, fontsize=9, color="w")
        ax.axis("off")
    axes[4].imshow(xt_norm[0], cmap="RdBu_r", vmin=-3, vmax=3, origin="lower", aspect="auto")
    axes[4].set_title("normalized ch0\n(the actual tensor)", fontsize=9, color="w")
    axes[4].axis("off")
    fig.suptitle(f"WHAT THE MODEL SEES — {cyc}/{pn_} (native fields, gaps kept)",
                 color="w", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"inputs_{cyc}_{pn_}.png", dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 8), facecolor=BG)
    panels = [(prob, "magma", 0, 1, "P(frontal)"),
              ((prob > 0.5).astype(float), "gray", 0, 1, "mask @ 0.5"),
              ((prob > LAM).astype(float), "gray", 0, 1, f"mask @ λ̂={LAM:.2f} (conformal)"),
              (x[0] * 100, "RdBu_r", -10, 30, "prediction vs labels")]
    for ax, (fld, cmap, vmin, vmax, ttl) in zip(axes, panels):
        ax.imshow(fld, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
        ax.set_title(ttl, fontsize=9, color="w")
        ax.axis("off")
    axes[3].contour(y, levels=[0.5], colors=TRUTH_C, linewidths=1.6)
    axes[3].contour(prob > 0.5, levels=[0.5], colors=PRED_C, linewidths=1.2, linestyles="--")
    fig.suptitle(f"WHAT THE MODEL DOES — {cyc}/{pn_} (green=confirmed, violet=prediction)",
                 color="w", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"outputs_{cyc}_{pn_}.png", dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

# ---- anatomy rows -------------------------------------------------------------
def anatomy(items, path, title, n=8, seed=0):
    rng = np.random.default_rng(seed)
    take = [items[i] for i in rng.permutation(len(items))[:n]]
    fig, axes = plt.subplots(len(take), 4, figsize=(11, 2.6 * len(take)), facecolor=BG)
    axes = np.atleast_2d(axes)
    for row, (c, p, x, y, prob, blob, s) in zip(axes, take):
        cy, cx = (int(v) for v in ndimage.center_of_mass(blob))
        r = 26
        i0, i1 = max(0, cy - r), min(x.shape[1], cy + r)
        j0, j1 = max(0, cx - r), min(128, cx + r)
        sub = lambda a: a[i0:i1, j0:j1]
        for ax, (ttl, cmap, vmin, vmax, fn) in zip(row[:3], CH[:3]):
            fld = fn(x)
            if ttl.startswith("SSHA"):
                vmin, vmax = np.nanpercentile(sub(fld), 2), np.nanpercentile(sub(fld), 98)
            ax.imshow(sub(fld), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        row[3].imshow(sub(prob), cmap="magma", vmin=0, vmax=1, origin="lower")
        for ax in row:
            ax.contour(sub(y), levels=[0.5], colors=TRUTH_C, linewidths=1.4)
            ax.contour(sub(blob.astype(float)), levels=[0.5], colors=PRED_C,
                       linewidths=1.2, linestyles="--")
            ax.axis("off")
        row[0].set_title(f"{c[-3:]}/{p}  p={s:.2f}", fontsize=8, color="w", loc="left")
    for ax, (ttl, *_r) in zip(axes[0], CH[:3] + [("P(frontal)", 0, 0, 0, 0)]):
        ax.set_title((ax.get_title() + "   " if ax.get_title() else "") + ttl,
                     fontsize=8, color="w", loc="left")
    fig.suptitle(title, color="w", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

anatomy(extra, OUT / "extras_anatomy_1.png",
        f"EXTRA detections (not confirmed) — what fools the model  [1/2 of {len(extra)}]",
        seed=1)
anatomy(extra, OUT / "extras_anatomy_2.png",
        f"EXTRA detections (not confirmed) — what fools the model  [2/2 of {len(extra)}]",
        seed=2)
anatomy(missed, OUT / "misses_anatomy.png",
        f"MISSED confirmed frontals — all {len(missed)}", n=len(missed))

# ---- confidence separation + threshold sweep ---------------------------------
ms = np.array([t[6] for t in matched])
es = np.array([t[6] for t in extra])
fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
bins = np.linspace(0, 1, 41)
ax.hist(ms, bins=bins, alpha=0.65, color="#1f6feb", label=f"matched blobs (n={len(ms)})")
ax.hist(es, bins=bins, alpha=0.65, color="#cf222e", label=f"extra blobs (n={len(es)})")
ax.set_xlabel("blob max P(frontal)")
ax.set_ylabel("blobs")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
ax.set_title("Can a threshold separate extras from matches?", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "confidence_hist.png", dpi=130, bbox_inches="tight")
plt.close(fig)

taus = np.linspace(0.05, 0.99, 40)
P, Rc = [], []
for tau in taus:
    TP = FP = FN = 0
    for c, p, x, y, prob in passes:
        pred = prob > tau
        tl, tn = ndimage.label(y)
        pl, pn = ndimage.label(pred)
        ht = set(np.unique(tl[pred])) - {0}
        hp = set(np.unique(pl[y > 0])) - {0}
        TP += len(ht)
        FN += tn - len(ht)
        FP += pn - len(hp)
    P.append(TP / max(TP + FP, 1))
    Rc.append(TP / max(TP + FN, 1))
fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
ax.plot(taus, P, color="#cf222e", lw=2, label="precision (vs confirmed labels)")
ax.plot(taus, Rc, color="#1f6feb", lw=2, label="recall")
ax.axvline(0.5, color="#57606a", ls=":", lw=1)
ax.axvline(LAM, color="#9a6700", ls="--", lw=1.2)
ax.text(LAM + 0.01, 0.1, f"λ̂={LAM:.2f}", fontsize=8, color="#9a6700")
ax.set_xlabel("mask threshold")
ax.set_ylim(0, 1.02)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower left")
ax.set_title("Eddy-level precision / recall vs threshold (test cycles)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "pr_vs_threshold.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- saliency: what pixels drive the score -----------------------------------
def saliency(c, p, x, blob):
    cy, cx = (int(v) for v in ndimage.center_of_mass(blob))
    s0 = min(max(cy - TILE // 2, 0), x.shape[1] - TILE)
    xt = normalize(x[:, s0:s0 + TILE, :])
    t = torch.from_numpy(xt)[None].requires_grad_(True)
    logit = model(t)[0, 0, cy - s0, cx]
    logit.backward()
    g = t.grad[0].abs().sum(0).numpy()
    return x[0, s0:s0 + TILE, :] * 100, g, (cy - s0, cx)

picks = [("matched", matched[int(i)]) for i in [0, len(matched) // 2]] + \
        [("extra", extra[int(i)]) for i in [0, len(extra) // 2]]
fig, axes = plt.subplots(2, 4, figsize=(11, 6), facecolor=BG)
for col, (tag, (c, p, x, y, prob, blob, s)) in enumerate(picks):
    ssha, g, (cy, cx) = saliency(c, p, x, blob)
    axes[0, col].imshow(ssha, cmap="RdBu_r", vmin=np.nanpercentile(ssha, 2),
                        vmax=np.nanpercentile(ssha, 98), origin="lower")
    axes[0, col].plot(cx, cy, "o", mfc="none", mec=PRED_C, ms=10, mew=1.6)
    axes[0, col].set_title(f"{tag}  {c[-3:]}/{p}  p={s:.2f}", fontsize=8, color="w")
    axes[1, col].imshow(g, cmap="inferno", origin="lower",
                        vmax=np.percentile(g, 99.5))
    axes[1, col].set_title("|∂logit/∂input|", fontsize=8, color="w")
    for ax in axes[:, col]:
        ax.axis("off")
fig.suptitle("Saliency — which input pixels drive each detection", color="w", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "saliency.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.close(fig)

data = {"matched": len(ms), "extra": len(es), "missed": len(missed), "lam_hat": LAM,
        "extra_above_09": int((es > 0.9).sum()), "show_passes": [f"{c}_{p}" for c, p in show]}
(OUT / "report_v2.json").write_text(json.dumps(data, indent=1))
print(json.dumps(data, indent=1))
