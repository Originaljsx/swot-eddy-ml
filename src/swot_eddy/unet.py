"""Small U-Net for frontal-eddy segmentation on SWOT swaths, plus the tile dataset.

Torch imports are kept inside functions/classes so the rest of the package works
without torch installed (torch is only needed for training/inference).

Input : (3, 128, 128) tiles — inpainted SSHA [m], inpainted ADT [m], |grad ADT| [m/km]
Target: (1, 128, 128) binary mask (1 = inside a confirmed frontal eddy's PET contour)
Built from the per-pass .npz files produced by scripts/build_masks.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Channels are first CLIPPED to physical bounds, then standardized. The clip is
# load-bearing: biharmonic inpainting explodes inside large gaps (ADT p99 was +120 m
# before clipping) and ssha_unedited carries raw outliers; bounds turn blowups into
# flat "no-data-like" patches. Stats measured on the clipped training distribution
# (cycles 1-35 sample) and FROZEN so inference scales identically.
CHANNEL_CLIP = np.array([[-1.0, 1.5], [-0.6, 1.6], [0.0, 0.05]], dtype=np.float32)
CHANNEL_MEAN = np.array([0.167, 0.606, 0.0089], dtype=np.float32)
CHANNEL_STD = np.array([0.234, 0.582, 0.0146], dtype=np.float32)
# v2 (raw, 4-channel): native fields, NaN gaps, 4th channel = validity (passed through
# unscaled). Same clip bounds (ssha_unedited still has raw outliers); stats re-measured
# on the raw training distribution. Selected automatically by channel count.
CHANNEL_MEAN_V2 = np.array([0.164, 0.607, 0.0056], dtype=np.float32)
CHANNEL_STD_V2 = np.array([0.150, 0.366, 0.0076], dtype=np.float32)
TILE = 128


def build_unet(base: int = 32, in_ch: int = 3):
    """3-level U-Net (~1.9M params at base=32). in_ch=3 (v1 inpainted) or 4 (v2 raw+validity).
    Returns nn.Module: (B,in_ch,128,128)->(B,1,128,128) logits."""
    import torch
    import torch.nn as nn

    def block(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = base
            self.e1, self.e2, self.e3 = block(in_ch, b), block(b, 2 * b), block(2 * b, 4 * b)
            self.mid = block(4 * b, 8 * b)
            self.pool = nn.MaxPool2d(2)
            self.u3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2)
            self.d3 = block(8 * b, 4 * b)
            self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
            self.d2 = block(4 * b, 2 * b)
            self.u1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
            self.d1 = block(2 * b, b)
            self.head = nn.Conv2d(b, 1, 1)

        def forward(self, x):
            import torch
            s1 = self.e1(x)
            s2 = self.e2(self.pool(s1))
            s3 = self.e3(self.pool(s2))
            m = self.mid(self.pool(s3))
            d = self.d3(torch.cat([self.u3(m), s3], 1))
            d = self.d2(torch.cat([self.u2(d), s2], 1))
            d = self.d1(torch.cat([self.u1(d), s1], 1))
            return self.head(d)

    return UNet()


def normalize(x: np.ndarray) -> np.ndarray:
    """(3,H,W) v1 or (4,H,W) v2 -> clip to physical bounds, standardize, NaN -> 0.
    v2's 4th (validity) channel passes through unscaled."""
    x = x.astype(np.float32)
    if x.shape[0] == 4:
        f = np.clip(x[:3], CHANNEL_CLIP[:, 0, None, None], CHANNEL_CLIP[:, 1, None, None])
        f = (f - CHANNEL_MEAN_V2[:, None, None]) / CHANNEL_STD_V2[:, None, None]
        return np.concatenate([np.nan_to_num(f, nan=0.0), x[3:4]])
    x = np.clip(x, CHANNEL_CLIP[:, 0, None, None], CHANNEL_CLIP[:, 1, None, None])
    out = (x - CHANNEL_MEAN[:, None, None]) / CHANNEL_STD[:, None, None]
    return np.nan_to_num(out, nan=0.0)


def pass_tiles(x, y=None, stride: int = 64):
    """Cut a stitched pass (3,L,128) [+ mask (L,128)] into (start, x_tile[, y_tile]) 128-tiles,
    same tiling scheme as detection (final window snapped to the end)."""
    L = x.shape[1]
    starts = list(range(0, max(L - TILE, 0) + 1, stride)) or [0]
    if starts[-1] != max(L - TILE, 0):
        starts.append(max(L - TILE, 0))
    out = []
    for s in starts:
        xt = x[:, s:s + TILE, :]
        if xt.shape[1] < TILE:
            xt = np.pad(xt, ((0, 0), (0, TILE - xt.shape[1]), (0, 0)))
        if y is None:
            out.append((s, xt))
        else:
            yt = y[s:s + TILE, :]
            if yt.shape[0] < TILE:
                yt = np.pad(yt, ((0, TILE - yt.shape[0]), (0, 0)))
            out.append((s, xt, yt))
    return out


class TileDataset:
    """Torch dataset over build_masks .npz files for one split.

    pos_frac: fraction of samples guaranteed to contain mask pixels (positive tiles are
    rare — ~1 in 8 — so training batches are rebalanced; validation uses pos_frac=None
    = natural distribution, every tile enumerated once).
    """

    def __init__(self, training_dir, split, pos_frac=0.5, seed=0, augment=False):
        import pandas as pd
        self.dir = Path(training_dir)
        man = pd.read_csv(self.dir / "training_manifest.csv", dtype={"cycle": str, "pass": str})
        man = man[(man.split == split) & (man.status == "ok")]
        self.files = [self.dir / f"{c}_{p}.npz" for c, p in zip(man.cycle, man["pass"])]
        self.files = [f for f in self.files if f.exists()]
        self.pos_frac = pos_frac
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.index = []          # (file_i, start, has_pos)
        for i, f in enumerate(self.files):
            with np.load(f) as d:
                y = d["y"]
                for s, _, yt in pass_tiles(np.zeros((3,) + y.shape, np.float16), y):
                    self.index.append((i, s, bool(yt.any())))
        self.pos = [t for t in self.index if t[2]]
        self.neg = [t for t in self.index if not t[2]]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        import torch
        if self.pos_frac is not None:
            pool = self.pos if self.rng.random() < self.pos_frac and self.pos else self.neg
            i, s, _ = pool[self.rng.integers(len(pool))]
        else:
            i, s, _ = self.index[k]
        with np.load(self.files[i]) as d:
            x = d["x"][:, s:s + TILE, :].astype(np.float32)
            y = d["y"][s:s + TILE, :].astype(np.float32)
        if x.shape[1] < TILE:
            x = np.pad(x, ((0, 0), (0, TILE - x.shape[1]), (0, 0)))
            y = np.pad(y, ((0, TILE - y.shape[0]), (0, 0)))
        x = normalize(x)
        if self.augment:
            if self.rng.random() < 0.5:          # along-track flip
                x, y = x[:, ::-1, :].copy(), y[::-1, :].copy()
            if self.rng.random() < 0.5:          # across-track flip
                x, y = x[:, :, ::-1].copy(), y[:, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(y[None])


def dice_bce_loss(logits, target, bce_weight: float = 0.5):
    import torch
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    return bce_weight * bce + (1 - bce_weight) * dice.mean()
