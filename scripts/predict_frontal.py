"""Frontal-eddy mask from a raw SWOT L3 Expert granule: netcdf in -> mask out.

Runs the SAME preprocessing as training (crop -> pad -> tile -> inpaint -> 3 channels),
tiles the pass, runs the trained U-Net, and stitches tile probabilities back to a
full-pass frontal-eddy probability map (overlaps averaged). Requires torch + trained
weights (out/aoi5/model/unet_frontal.pt, produced by notebooks/04_train_unet.ipynb).

Usage:
  python scripts/predict_frontal.py PATH/TO/GRANULE.nc [--weights W.pt] [--thresh 0.5]
Outputs next to nothing else (read-only inputs):
  out/aoi5/predictions/<granule-stem>_frontal.npz   prob float16 (L,128), mask uint8
  out/aoi5/predictions/<granule-stem>_frontal.png   SSHA + predicted contours overlay
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*partition.*will ignore the 'mask'.*")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swot_eddy.config import load_config
from swot_eddy import preprocess as pp
from swot_eddy.features import grad_mag_per_km
from swot_eddy import unet as U


def predict_pass(granule, cfg, model, device="cpu", stride=64, raw=False):
    import torch
    if raw:                                  # v2: native channels, no inpaint (fast path)
        from swot_eddy.features import raw_channels
        x, L = raw_channels(granule, cfg)
        clon = clat = None
    else:                                    # v1: same inpainted stack as detection
        tiles, summ, clon, clat = pp.preprocess_pass(str(granule), cfg)
        L = summ["n_lines_cropped"]
        full = np.full((2, L, 128), np.nan)
        for t in tiles:
            h = t.data.shape[1]
            e = min(t.line_start + h, L)
            full[:, t.line_start:e, :] = t.data[:, :e - t.line_start, :]
        x = np.concatenate([full, grad_mag_per_km(full[1])[None]], axis=0)

    prob = np.zeros((L, 128), dtype=np.float32)
    hits = np.zeros((L, 128), dtype=np.float32)
    batch, starts = [], []
    for s, xt in U.pass_tiles(x, stride=stride):
        batch.append(U.normalize(xt))
        starts.append(s)
    with torch.no_grad():
        xb = torch.from_numpy(np.stack(batch)).to(device)
        pb = torch.sigmoid(model(xb)).cpu().numpy()[:, 0]
    for s, p in zip(starts, pb):
        e = min(s + U.TILE, L)
        prob[s:e, :] += p[:e - s, :]
        hits[s:e, :] += 1
    prob /= np.maximum(hits, 1)
    return prob, x, clon, clat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("granule")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--conformal", action="store_true",
                    help="use the conformally calibrated threshold lam_hat "
                         "(out/aoi5/model/lam_hat[_v2].npy; miss rate <= alpha guarantee)")
    ap.add_argument("--v2", action="store_true",
                    help="v2 model: raw native channels + validity, no inpainting")
    args = ap.parse_args()

    import torch
    cfg = load_config()
    ver = "_v2" if args.v2 else ""
    wpath = Path(args.weights) if args.weights else \
        Path(cfg["out_root"]) / "aoi5" / "model" / f"unet_frontal{ver}.pt"
    model = U.build_unet(in_ch=4 if args.v2 else 3)
    model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
    model.eval()

    thresh = args.thresh
    if args.conformal:
        lam = np.load(wpath.parent / f"lam_hat{ver}.npy")
        thresh = float(lam[0])
        print(f"conformal threshold lam_hat={thresh:.3f} (alpha={lam[1]:.2f})")

    prob, x, _, _ = predict_pass(args.granule, cfg, model, raw=args.v2)
    mask = (prob >= thresh).astype(np.uint8)

    out_dir = Path(cfg["out_root"]) / "aoi5" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.granule).stem
    np.savez_compressed(out_dir / f"{stem}_frontal.npz",
                        prob=prob.astype(np.float16), mask=mask)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7, 10))
    axes[0].imshow(x[0] * 100, cmap="RdBu_r", vmin=-10, vmax=30, origin="lower", aspect="auto")
    axes[0].set_title("SSHA (cm)", fontsize=9)
    axes[1].imshow(prob, cmap="magma", vmin=0, vmax=1, origin="lower", aspect="auto")
    axes[1].set_title(f"P(frontal), thresh {thresh:.3f}", fontsize=9)
    for a in axes:
        a.contour(mask, levels=[0.5], colors="lime", linewidths=1)
        a.axis("off")
    fig.savefig(out_dir / f"{stem}_frontal.png", dpi=120, bbox_inches="tight")
    n_blobs = int(__import__("scipy.ndimage", fromlist=["label"]).label(mask)[1])
    print(f"{stem}: {n_blobs} predicted frontal eddies, "
          f"{mask.sum()} px -> {out_dir / (stem + '_frontal.npz')}")


if __name__ == "__main__":
    main()
