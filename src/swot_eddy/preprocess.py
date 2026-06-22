"""Shared SWOT preprocessing core: crop -> channels -> pad -> tile -> inpaint.

Channel definitions (see config.yaml product.variables):
- channel 0 "ssha" = ssha_filtered (denoised; the field PET detects on)
- channel 1 "adt"  = ssha_unedited + mdt (carried for viewing; matches reference figures)
Pixels with quality_flag > quality_max are set to NaN on BOTH channels, then
biharmonic-inpaint fills every NaN so closed contours can form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

LINE_DIM = "num_lines"
PIX_DIM = "num_pixels"


def region_mask(lon: np.ndarray, lat: np.ndarray, region: dict[str, Any]) -> np.ndarray:
    """Bool mask of pixels inside the region box. lon is treated as 0..360;
    the region's (possibly negative) lon bounds are normalized via % 360.
    Assumes the box does not wrap the 0/360 seam (true for the Gulf Stream)."""
    lo0 = region["lon_min"] % 360
    lo1 = region["lon_max"] % 360
    lon360 = np.asarray(lon) % 360
    lat = np.asarray(lat)
    return (
        (lon360 >= lo0) & (lon360 <= lo1)
        & (lat >= region["lat_min"]) & (lat <= region["lat_max"])
    )


def crop_to_region(ds: xr.Dataset, region: dict[str, Any]) -> "tuple[xr.Dataset, slice]":
    """Subset the dataset to the contiguous along-track line band intersecting the box.
    Keeps the full 69-pixel across-track width. Raises ValueError if no intersection."""
    m = region_mask(ds["longitude"].values, ds["latitude"].values, region)
    rows = np.where(m.any(axis=1))[0]
    if rows.size == 0:
        raise ValueError("region box does not intersect this pass")
    sl = slice(int(rows.min()), int(rows.max()) + 1)
    return ds.isel({LINE_DIM: sl}), sl


def compute_channels(ds: xr.Dataset, var_map: dict[str, str],
                     quality_max: "int | None" = None) -> "tuple[np.ndarray, list[str], float]":
    """Build the [SSHA, ADT] channel stack.
    SSHA = var_map['ssha'] (e.g. ssha_filtered); ADT = var_map['ssha_adt_base'] + MDT
    (e.g. ssha_unedited + mdt), NaN where MDT missing. If quality_max is not None,
    pixels with quality_flag > quality_max are set to NaN on both channels.
    Returns (stack (2,L,P) float64, ['ssha','adt'], mdt_coverage_fraction)."""
    ssha = ds[var_map["ssha"]].values.astype("float64")
    adt_base = ds[var_map.get("ssha_adt_base", var_map["ssha"])].values.astype("float64")
    mdt = ds[var_map["mdt"]].values.astype("float64")
    adt = adt_base + mdt  # NaN propagates where mdt is NaN

    if quality_max is not None and var_map.get("quality_flag") in ds.variables:
        qf = ds[var_map["quality_flag"]].values
        bad = np.asarray(qf) > quality_max
        ssha = np.where(bad, np.nan, ssha)
        adt = np.where(bad, np.nan, adt)

    stack = np.stack([ssha, adt], axis=0)
    mdt_coverage = float(np.isfinite(mdt).mean())
    return stack, ["ssha", "adt"], mdt_coverage


def pad_across_track(stack: np.ndarray, pad_to: int = 128) -> "tuple[np.ndarray, int]":
    """NaN-pad the pixel (last) axis from P to pad_to, centered. Returns (padded, pad_left)."""
    c, l, p = stack.shape
    if p > pad_to:
        raise ValueError(f"pixel width {p} exceeds pad_to {pad_to}")
    left = (pad_to - p) // 2
    right = pad_to - p - left
    padded = np.pad(stack, ((0, 0), (0, 0), (left, right)), constant_values=np.nan)
    return padded, left


def tile_starts(n_lines: int, tile: int = 128, stride: int = 64) -> list[int]:
    """Along-track window start indices; always includes a final window ending at n_lines."""
    if n_lines <= tile:
        return [0]
    starts = list(range(0, n_lines - tile + 1, stride))
    if starts[-1] != n_lines - tile:
        starts.append(n_lines - tile)
    return starts


def tile_along_track(padded: np.ndarray, tile: int = 128, stride: int = 64) -> "list[tuple[int, np.ndarray]]":
    """Cut (C, L, pad_to) into (start, (C, tile, pad_to)) windows; NaN-pad a short tail."""
    c, l, p = padded.shape
    out = []
    for s in tile_starts(l, tile, stride):
        w = padded[:, s:s + tile, :]
        if w.shape[1] < tile:
            w = np.pad(w, ((0, 0), (0, tile - w.shape[1]), (0, 0)), constant_values=np.nan)
        out.append((s, w))
    return out


def inpaint_tile(tile_data: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Biharmonic-inpaint NaNs per channel. Returns (filled (C,H,W), mask (C,H,W) bool).
    A fully-missing channel is zero-filled. Import is local so the rest of the module
    works without scikit-image installed."""
    from skimage.restoration import inpaint_biharmonic

    filled = np.empty_like(tile_data, dtype="float64")
    mask = ~np.isfinite(tile_data)
    for c in range(tile_data.shape[0]):
        ch = tile_data[c]
        m = mask[c]
        if m.all():
            filled[c] = 0.0
        elif m.any():
            base = np.where(m, 0.0, ch)
            filled[c] = inpaint_biharmonic(base, m)
        else:
            filled[c] = ch
    return filled, mask


@dataclass(frozen=True)
class Tile:
    data: np.ndarray         # (C, 128, 128) inpainted float64
    inpaint_mask: np.ndarray # (C, 128, 128) bool, True where filled
    line_start: int          # start index into the cropped pass
    pad_left: int            # across-track NaN-pad offset
    channels: tuple          # e.g. ("ssha", "adt")


def preprocess_pass(ds_or_path, cfg: dict) -> "tuple[list[Tile], dict, np.ndarray, np.ndarray]":
    """Full shared preprocessing for one pass. Accepts an open Dataset or a path string.
    Returns (tiles, summary, cropped_lon, cropped_lat). Source is never mutated."""
    opened = None
    ds = ds_or_path
    if isinstance(ds_or_path, str):
        opened = xr.open_dataset(ds_or_path)
        ds = opened
    try:
        var_map = cfg["product"]["variables"][cfg["product"]["product_version"]]
        quality_max = cfg["product"].get("quality_max")
        ds_c, sl = crop_to_region(ds, cfg["region"])
        stack, names, mdt_cov = compute_channels(ds_c, var_map, quality_max)
        padded, pad_left = pad_across_track(stack, cfg["tiling"]["pad_to"])
        tiles = []
        for s, w in tile_along_track(padded, cfg["tiling"]["tile_size"], cfg["tiling"]["stride"]):
            filled, mask = inpaint_tile(w)
            tiles.append(Tile(filled, mask, s, pad_left, tuple(names)))
        summary = {
            "n_tiles": len(tiles),
            "n_lines_cropped": int(ds_c.sizes[LINE_DIM]),
            "mdt_coverage": mdt_cov,
            "quality_max": quality_max,
            "line_slice": [int(sl.start), int(sl.stop)],
            "channels": names,
        }
        return tiles, summary, ds_c["longitude"].values, ds_c["latitude"].values
    finally:
        if opened is not None:
            opened.close()


def tile_pixel_lonlat(tile: Tile, i: int, j: int,
                      cropped_lon: np.ndarray, cropped_lat: np.ndarray):
    """Map an in-tile pixel (along-track i, across-track j incl. pad) back to source
    (lon, lat). Returns None for padded / out-of-range pixels."""
    line = tile.line_start + i
    pix = j - tile.pad_left
    if not (0 <= pix < cropped_lon.shape[1]) or not (0 <= line < cropped_lon.shape[0]):
        return None
    return float(cropped_lon[line, pix]), float(cropped_lat[line, pix])
