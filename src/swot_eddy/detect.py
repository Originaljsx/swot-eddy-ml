"""First py-eddy-tracker (PET) detection pass over preprocessed SWOT tiles.

Runs PET on the SSHA channel (channel 0) of each inpainted 128x128 Tile, extracts
eddies (polarity, center, contour, radius, amplitude), maps them back to full-pass
pixel coordinates, de-duplicates across the 64-px tile overlaps, and flags
detections that sit on heavily inpainted / nadir-gap pixels.

Tuned for HIGH RECALL (the user rejects false positives by hand): pixel_limit min
lowered 4->3, shape_error kept high (config 85), no high-pass filter.

PET API (verified against installed source, py_eddy_tracker 0+untagged.754.g229543f):
- RegularGridDataset.with_array(coordinates, datas, **kwargs)  grid.py:1232  (data ordered (X,Y))
- RegularGridDataset.add_uv(grid_height, uname, vname)          grid.py:1835
- GridDataset.eddy_identification(grid_height, uname, vname, date, step, shape_error,
      pixel_limit) -> [anticyclones, cyclones]                  grid.py:627
- eddy fields: lon, lat, radius_e (m), radius_s (m), amplitude (m),
      contour_lon_e, contour_lat_e                              observations/observation.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 2 km KaRIn pixels; degree-per-2km metrics for the local equirectangular tile grid.
KM_PER_PIXEL = 2.0
_DEG_PER_KM_LAT = 1.0 / 110.574


@dataclass
class Detection:
    polarity: str                              # "cyclonic" | "anticyclonic"
    center_line: int                           # full-pass along-track line index
    center_col: int                            # across-track pixel (0..127, tile col incl. pad)
    center_lon: "float | None"                 # 0..360 from tile_pixel_lonlat, None if padded
    center_lat: "float | None"
    radius_km: float                           # R_max = speed radius radius_s/1000 (paper primary)
    radius_eff_km: float                       # effective radius radius_e/1000 (outermost contour)
    amplitude_m: float
    contour_pix: list                          # [(line, col), ...] SPEED (max-velocity) contour
    contour_lonlat: list                       # [(lon, lat) | None, ...] speed contour geo
    gap_frac: float                            # fraction of speed-contour pixels that were inpainted
    gap_reject: bool                           # paper criterion: gap_frac > thresh (0.40, 0.60 at nadir)
    nadir_frac: float                          # fraction of enclosed area sitting in the nadir gap columns
    nadir_reject: bool                         # nadir_frac >= nadir_reject_frac -> biharmonic-invented eddy, DROP
    center_in_nadir: bool                      # center sits in the central nadir gap band
    source_tile: int                           # tile index it was detected in (pre-dedup)
    edge_dist: int                             # min pixel distance of center to its tile edge
    channel: str = "ssha"


def _pet_params(cfg: dict) -> dict:
    """Map config pet.params -> eddy_identification kwargs + post-filter thresholds.
    Follows de Marez et al. (2026); pixel_limit configurable (paper-standard = (4,1000))."""
    p = cfg["pet"]["params"]
    return {
        "step": p["min_amplitude_m"] / 2.0,    # nb_step_min=2 -> amplitude floor = min_amplitude_m
        "shape_error": p["shape_error"],       # 85
        "pixel_limit": (p.get("pixel_limit_min", 4), p.get("pixel_limit_max", 1000)),
        "mle": p.get("mle", 1),                # max local extrema per contour (PET default 1)
        "min_radius_km": p["min_radius_km"],   # post-filter on R_max (radius_s)
        "gap_reject_frac": p["gap_reject_frac"],
        "gap_reject_frac_nadir": p["gap_reject_frac_nadir"],
        "nadir_reject_frac": p.get("nadir_reject_frac", 0.70),  # drop if >= this much of area in nadir
    }


def _tile_anchor(tile, cropped_lon: np.ndarray, cropped_lat: np.ndarray) -> "tuple[float, float]":
    """Representative (lon0, lat0) for the tile from its real (non-padded) source pixels.
    lat0 sets the local-grid metric + Coriolis term; lon0 is an arbitrary self-consistent base."""
    H = tile.data.shape[1]
    line0 = tile.line_start
    line1 = min(line0 + H, cropped_lat.shape[0])
    lat = cropped_lat[line0:line1, :]
    lon = cropped_lon[line0:line1, :]
    lat0 = float(np.nanmean(lat)) if np.isfinite(lat).any() else 30.0
    lon0 = float(np.nanmin(lon % 360)) if np.isfinite(lon).any() else 280.0
    return lon0, lat0


def build_tile_grid(tile, channel_idx: int, cropped_lon: np.ndarray, cropped_lat: np.ndarray):
    """Build a RegularGridDataset from one inpainted tile channel on a local equirectangular
    degree grid. Axis convention: x = along-track row i, y = across-track col j, so
    datas[name].shape == (128, 128) == (i, j). Returns (grid, x_c, y_c, dx_deg, dy_deg)."""
    from py_eddy_tracker.dataset.grid import RegularGridDataset

    field2d = np.asarray(tile.data[channel_idx], dtype="f8")
    nx, ny = field2d.shape                                   # (along-track, across-track)
    lon0, lat0 = _tile_anchor(tile, cropped_lon, cropped_lat)
    dy_deg = KM_PER_PIXEL * _DEG_PER_KM_LAT                  # ~0.0181 deg / 2 km
    dx_deg = KM_PER_PIXEL / (111.320 * np.cos(np.deg2rad(lat0)))
    x_c = lon0 + dx_deg * np.arange(nx)                      # along-track axis (x)
    y_c = lat0 + dy_deg * np.arange(ny)                      # across-track axis (y), latitude-like

    # Tiles have NO NaNs (inpainted); use an all-False mask so PET rejects nothing here.
    masked = np.ma.array(field2d, mask=np.zeros_like(field2d, dtype=bool))
    grid = RegularGridDataset.with_array(
        coordinates=("x", "y"),
        datas={"x": x_c, "y": y_c, "ssha": masked},
        variables_description={"ssha": {"units": "m"}},   # avoids reading units from a file
        centered=True,
    )
    return grid, x_c, y_c, dx_deg, dy_deg


def _contour_gap_frac(contour_ij: list, mask_ssha: np.ndarray) -> float:
    """Fraction of pixels enclosed by a contour (tile-local (i, j) verts) that were inpainted.
    This is the paper's 'overlap of the eddy (speed) contour with gaps in the SLA field'."""
    from matplotlib.path import Path

    if len(contour_ij) < 3:
        return 0.0
    ij = np.asarray(contour_ij, dtype="f8")          # (N, 2) = (i row, j col)
    i0, i1 = int(np.floor(ij[:, 0].min())), int(np.ceil(ij[:, 0].max())) + 1
    j0, j1 = int(np.floor(ij[:, 1].min())), int(np.ceil(ij[:, 1].max())) + 1
    h, w = mask_ssha.shape
    i0, i1 = max(0, i0), min(h, i1)
    j0, j1 = max(0, j0), min(w, j1)
    if i1 <= i0 or j1 <= j0:
        return 0.0
    rr, cc = np.mgrid[i0:i1, j0:j1]
    pts = np.column_stack([cc.ravel(), rr.ravel()])  # Path wants (x=col, y=row)
    path = Path(np.column_stack([ij[:, 1], ij[:, 0]]))
    inside = path.contains_points(pts)
    if not inside.any():
        return 0.0
    enclosed = pts[inside]
    return float(mask_ssha[enclosed[:, 1], enclosed[:, 0]].mean())


def _contour_nadir_frac(contour_ij: list, nadir_cols: set, h: int, w: int) -> float:
    """Fraction of pixels enclosed by a contour that sit in the central nadir-gap columns.
    Biharmonic inpainting fabricates smooth blobs in the nadir; an eddy whose area is mostly
    inside the nadir is invented, not observed -> drop it."""
    from matplotlib.path import Path

    if len(contour_ij) < 3 or not nadir_cols:
        return 0.0
    ij = np.asarray(contour_ij, dtype="f8")          # (N, 2) = (i row, j col)
    i0, i1 = int(np.floor(ij[:, 0].min())), int(np.ceil(ij[:, 0].max())) + 1
    j0, j1 = int(np.floor(ij[:, 1].min())), int(np.ceil(ij[:, 1].max())) + 1
    i0, i1 = max(0, i0), min(h, i1)
    j0, j1 = max(0, j0), min(w, j1)
    if i1 <= i0 or j1 <= j0:
        return 0.0
    rr, cc = np.mgrid[i0:i1, j0:j1]
    pts = np.column_stack([cc.ravel(), rr.ravel()])  # (x=col, y=row)
    path = Path(np.column_stack([ij[:, 1], ij[:, 0]]))
    inside = path.contains_points(pts)
    if not inside.any():
        return 0.0
    cols = pts[inside][:, 0]                          # enclosed pixel columns
    return float(np.isin(cols, np.fromiter(nadir_cols, dtype=int)).mean())


def _nadir_cols(inpaint_mask_ssha: np.ndarray, pad_left: int) -> set:
    """Columns of the central nadir gap: the ~20 km dead band at swath CENTER that is
    inpainted on (almost) every row. Restricted to a central window so a mostly-NaN tile
    (whole swath inpainted) does NOT get every column flagged as nadir."""
    h, w = inpaint_mask_ssha.shape
    center = pad_left + 34                                   # middle of the 69-px swath
    band = range(max(0, center - 12), min(w, center + 13))  # only the central region can be nadir
    return {j for j in band if inpaint_mask_ssha[:, j].mean() > 0.9}


def detect_tile(tile, tile_index: int, cfg: dict,
                cropped_lon: np.ndarray, cropped_lat: np.ndarray,
                date: datetime, channel: str = "ssha") -> list:
    """Run PET on one tile's SSHA channel; return full-pass-mapped, flagged Detections."""
    from swot_eddy.preprocess import tile_pixel_lonlat

    params = _pet_params(cfg)
    ch_idx = list(tile.channels).index(channel)

    # A near-constant tile (e.g. fully quality-masked -> zero-filled) yields no contours
    # and crashes PET's Contours(); skip it -- it carries no eddy signal anyway.
    field2d = np.asarray(tile.data[ch_idx], dtype="f8")
    if np.ptp(field2d) < 2.0 * params["step"]:
        return []

    grid, x_c, y_c, dx_deg, dy_deg = build_tile_grid(tile, ch_idx, cropped_lon, cropped_lat)
    grid.add_uv("ssha")
    # force_*_unit avoids GridDataset.units() reading from the (non-existent) source file
    # for an in-memory array grid.
    try:
        anticyc, cyc = grid.eddy_identification(
            "ssha", "u", "v", date,
            step=params["step"], shape_error=params["shape_error"],
            pixel_limit=params["pixel_limit"], mle=params["mle"],
            force_height_unit="m", force_speed_unit="m/s",
        )
    except (UnboundLocalError, ValueError) as exc:
        # PET can fail on pathological contour sets; skip the tile rather than the whole pass.
        logger.warning("PET identification failed on tile %d: %s", tile_index, exc)
        return []

    h, w = tile.data.shape[1], tile.data.shape[2]
    mask_ssha = tile.inpaint_mask[0]
    nadir = _nadir_cols(mask_ssha, tile.pad_left)
    x0, y0 = x_c[0], y_c[0]

    def to_pix(lon_val, lat_val):
        i = int(round((lon_val - x0) / dx_deg))             # along-track (x)
        j = int(round((lat_val - y0) / dy_deg))             # across-track (y)
        return min(max(i, 0), h - 1), min(max(j, 0), w - 1)

    out = []
    for polarity, eddies in (("anticyclonic", anticyc), ("cyclonic", cyc)):
        for k in range(len(eddies)):
            # R_max = speed (max-velocity) radius -> the paper's primary radius
            radius_km = float(eddies["radius_s"][k]) / 1000.0
            if radius_km < params["min_radius_km"]:
                continue
            ci, cj = to_pix(float(eddies["lon"][k]), float(eddies["lat"][k]))

            # SPEED contour (region of maximum azimuthal velocity) -> tile-local + full-pass + geo
            clon = np.asarray(eddies["contour_lon_s"][k])
            clat = np.asarray(eddies["contour_lat_s"][k])
            contour_ij, contour_pix, contour_lonlat = [], [], []
            for lo, la in zip(clon, clat):
                pi, pj = to_pix(float(lo), float(la))
                contour_ij.append((pi, pj))
                contour_pix.append((tile.line_start + pi, pj))
                contour_lonlat.append(tile_pixel_lonlat(tile, pi, pj, cropped_lon, cropped_lat))

            # Paper gap criterion: reject if >40% of the speed contour overlaps gaps,
            # relaxed to 60% when the center sits in the central nadir band.
            gap_frac = _contour_gap_frac(contour_ij, mask_ssha)
            center_in_nadir = cj in nadir
            thresh = params["gap_reject_frac_nadir"] if center_in_nadir else params["gap_reject_frac"]
            gap_reject = gap_frac > thresh

            # Nadir criterion (ours, stricter than the paper's relaxed nadir tolerance): if the
            # eddy area is mostly inside the nadir dead band it is a biharmonic-invented blob -> drop.
            nadir_frac = _contour_nadir_frac(contour_ij, nadir, h, w)
            nadir_reject = nadir_frac >= params["nadir_reject_frac"]

            geo = tile_pixel_lonlat(tile, ci, cj, cropped_lon, cropped_lat)
            out.append(Detection(
                polarity=polarity,
                center_line=tile.line_start + ci,
                center_col=cj,
                center_lon=None if geo is None else geo[0],
                center_lat=None if geo is None else geo[1],
                radius_km=radius_km,
                radius_eff_km=float(eddies["radius_e"][k]) / 1000.0,
                amplitude_m=float(eddies["amplitude"][k]),
                contour_pix=contour_pix,
                contour_lonlat=contour_lonlat,
                gap_frac=gap_frac,
                nadir_frac=nadir_frac,
                nadir_reject=nadir_reject,
                gap_reject=gap_reject,
                center_in_nadir=center_in_nadir,
                source_tile=tile_index,
                edge_dist=int(min(ci, h - 1 - ci, cj, w - 1 - cj)),
                channel=channel,
            ))
    return out


def dedup_detections(dets: list, cfg: dict) -> list:
    """Merge duplicates across 64-px tile overlaps. Same polarity + center distance (in
    full-pass pixels) < min(R1,R2) -> keep the most interior (largest edge_dist), tie-break
    by larger amplitude."""
    kept: list = []
    for polarity in ("anticyclonic", "cyclonic"):
        group = [d for d in dets if d.polarity == polarity]
        # most-interior first so it wins ties and absorbs edge-clipped neighbours
        group.sort(key=lambda d: (d.edge_dist, d.amplitude_m), reverse=True)
        chosen: list = []
        for d in group:
            r_d = max(1.0, d.radius_km / KM_PER_PIXEL)
            dup = False
            for c in chosen:
                dist = np.hypot(d.center_line - c.center_line, d.center_col - c.center_col)
                r_c = max(1.0, c.radius_km / KM_PER_PIXEL)
                if dist < min(r_d, r_c):
                    dup = True
                    break
            if not dup:
                chosen.append(d)
        kept.extend(chosen)
    return kept


def detect_pass(tiles: list, cfg: dict, cropped_lon: np.ndarray, cropped_lat: np.ndarray,
                date: datetime, channel: str = "ssha") -> "tuple[list, dict]":
    """Detect over all tiles of one pass, dedup overlaps, return (detections, summary)."""
    kept, _rejected, summary = detect_pass_split(tiles, cfg, cropped_lon, cropped_lat, date, channel)
    return kept, summary


def detect_pass_split(tiles: list, cfg: dict, cropped_lon: np.ndarray, cropped_lat: np.ndarray,
                      date: datetime, channel: str = "ssha") -> "tuple[list, list, dict]":
    """Like detect_pass but ALSO returns the auto-rejected detections (kept, rejected, summary).
    rejected = the nadir-blob / gap-overlap detections detect_pass drops -- useful as ML
    negatives. Both lists are deduped across tile overlaps."""
    raw: list = []
    for idx, t in enumerate(tiles):
        raw.extend(detect_tile(t, idx, cfg, cropped_lon, cropped_lat, date, channel))
    n_total = len(raw)
    n_nadir_reject = sum(d.nadir_reject for d in raw)
    n_gap_reject = sum(d.gap_reject and not d.nadir_reject for d in raw)
    rejected_raw = [d for d in raw if d.nadir_reject or d.gap_reject]
    kept_raw = [d for d in raw if not (d.nadir_reject or d.gap_reject)]
    kept = dedup_detections(kept_raw, cfg)
    rejected = dedup_detections(rejected_raw, cfg)
    summary = {
        "n_total": n_total,
        "n_nadir_reject": n_nadir_reject,                      # dropped: mostly in nadir gap
        "n_gap_reject": n_gap_reject,                          # dropped: >40%/60% contour on gaps
        "n_raw": len(kept_raw),                                # kept survivors before dedup
        "n_dedup": len(kept),
        "n_kept": len(kept),
        "n_rejected": len(rejected),                           # deduped auto-rejected (negatives)
        "n_anticyclonic": sum(d.polarity == "anticyclonic" for d in kept),
        "n_cyclonic": sum(d.polarity == "cyclonic" for d in kept),
        "n_nadir_center": sum(d.center_in_nadir for d in kept),
        "channel": channel,
    }
    return kept, rejected, summary
