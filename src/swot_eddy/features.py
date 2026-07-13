"""Per-detection contextual features from the native SWOT swath (no external data).

The Gulf Stream separates low ADT (shelf/slope side) from high ADT (Sargasso side),
so a single mid-level contour of native ADT (= ssha_unedited + mdt) traces the jet's
dividing line through the swath. Front distances are SIGNED by the ADT value at the
eddy center relative to that level: negative = inshore/low-ADT side (where frontal
eddies live), positive = offshore/high-ADT side. Two front proxies per pass:
- adt front: instantaneous line; may be absent when the swath misses the jet
  (front_adt_missing flag, distance NaN)
- mdt front: mean-state line from the smooth MDT field; almost always present

All lon handled in -180..180 (catalog convention); granule 0..360 is normalized here.
Pixel spacing is ~2 km; contour vertices are taken at nearest-pixel lon/lat, which is
plenty for the ~tens-of-km distance scales these features feed.
"""
from __future__ import annotations

import numpy as np

KM_PER_DEG_LAT = 110.574
KM_PER_PIXEL = 2.0


def _lon180(lon):
    return ((np.asarray(lon, dtype="float64") + 180.0) % 360.0) - 180.0


def _local_xy(lon, lat, lat0):
    """Equirectangular km coords about reference latitude lat0 (fine at basin scale)."""
    x = np.asarray(lon, dtype="float64") * KM_PER_DEG_LAT * np.cos(np.deg2rad(lat0))
    y = np.asarray(lat, dtype="float64") * KM_PER_DEG_LAT
    return x, y


def front_vertices(field, lon2d, lat2d, level, min_vertices=20):
    """Front line(s) = iso-contours of `field` at `level`, as (M, 2) lon/lat vertices.

    Uses skimage.measure.find_contours with mask=isfinite so the nadir gap and swath
    edges simply break the contours. Contours shorter than min_vertices (~2 km/px, so
    20 px ~ 40 km) are dropped as closed-eddy rings / noise, keeping the jet line.
    Returns an empty (0, 2) array when nothing survives.
    """
    from skimage.measure import find_contours

    field = np.asarray(field, dtype="float64")
    mask = np.isfinite(field)
    if not mask.any():
        return np.empty((0, 2))
    lon2d = _lon180(lon2d)
    lat2d = np.asarray(lat2d, dtype="float64")
    verts = []
    for c in find_contours(field, level, mask=mask):
        if len(c) < min_vertices:
            continue
        r = np.clip(np.rint(c[:, 0]).astype(int), 0, field.shape[0] - 1)
        p = np.clip(np.rint(c[:, 1]).astype(int), 0, field.shape[1] - 1)
        verts.append(np.column_stack([lon2d[r, p], lat2d[r, p]]))
    return np.concatenate(verts) if verts else np.empty((0, 2))


def signed_front_dist_km(lons, lats, center_vals, level, front_lonlat):
    """Signed distance [km] from each point to the nearest front vertex.

    Magnitude = nearest-vertex distance in local km; sign = sign(center_val - level)
    (+ offshore/high side, - inshore/low side), which is immune to contour
    fragmentation/orientation. Returns (signed_dist, missing) where missing=True and
    dist=NaN when the front line is empty or the center value is NaN.
    """
    from scipy.spatial import cKDTree

    lons = np.atleast_1d(_lon180(lons))
    lats = np.atleast_1d(np.asarray(lats, dtype="float64"))
    center_vals = np.atleast_1d(np.asarray(center_vals, dtype="float64"))
    out = np.full(lons.shape, np.nan)
    if len(front_lonlat) == 0:
        return out, np.ones(lons.shape, dtype=bool)
    lat0 = float(np.nanmean(front_lonlat[:, 1]))
    fx, fy = _local_xy(front_lonlat[:, 0], front_lonlat[:, 1], lat0)
    px, py = _local_xy(lons, lats, lat0)
    dist, _ = cKDTree(np.column_stack([fx, fy])).query(np.column_stack([px, py]))
    sign = np.sign(center_vals - level)
    missing = ~np.isfinite(center_vals)
    out = np.where(missing, np.nan, dist * np.where(sign == 0, 1.0, sign))
    return out, missing


def nearest_pixel(lon2d, lat2d, lon, lat):
    """(line, col) of the grid pixel nearest to (lon, lat), cos-lat-scaled."""
    lon2d = _lon180(lon2d)
    dlon = (lon2d - _lon180(lon)) * np.cos(np.deg2rad(lat))
    dlat = np.asarray(lat2d, dtype="float64") - lat
    return np.unravel_index(np.nanargmin(dlon * dlon + dlat * dlat), lon2d.shape)


def sample_nearest_finite(field, line, col, max_px=10):
    """field[line, col], falling back to the nearest finite pixel within max_px
    (needed for centers in the nadir gap). Returns (value, dist_px); (NaN, NaN)
    when nothing finite is in range."""
    field = np.asarray(field, dtype="float64")
    if np.isfinite(field[line, col]):
        return float(field[line, col]), 0.0
    l0, l1 = max(0, line - max_px), min(field.shape[0], line + max_px + 1)
    c0, c1 = max(0, col - max_px), min(field.shape[1], col + max_px + 1)
    win = field[l0:l1, c0:c1]
    fin = np.isfinite(win)
    if not fin.any():
        return np.nan, np.nan
    ll, cc = np.nonzero(fin)
    d2 = (ll + l0 - line) ** 2 + (cc + c0 - col) ** 2
    k = int(np.argmin(d2))
    return float(win[ll[k], cc[k]]), float(np.sqrt(d2[k]))


def grad_mag_per_km(field):
    """|grad field| in field-units/km on the ~2 km pixel grid. NaN propagates
    (nadir gap and edges stay NaN) -- no smoothing, no inpainting."""
    gy, gx = np.gradient(np.asarray(field, dtype="float64"))
    return np.hypot(gy, gx) / KM_PER_PIXEL


def find_granule(data_root, cycle, pnum):
    """Granule path for (cycle_dir, pass_number), or None. Matches the CYCLE_PASS field
    pair in the filename (SWOT_L3_..._CCC_PPP_...): a bare `*_{pass}_*` glob also hits the
    cycle field, returning the wrong pass when a pass number equals the cycle number
    (e.g. cycle_048 pass 048)."""
    from pathlib import Path

    cnum = str(cycle).split("_")[-1]
    fs = sorted((Path(data_root) / str(cycle)).glob(f"*_{cnum}_{pnum}_*.nc"))
    return fs[0] if fs else None


def native_fields(granule_path, cfg):
    """Open one granule, crop to the region band, and return the NATIVE (non-inpainted)
    fields the features need: dict with lon, lat (2-D, lon in -180..180), adt, mdt,
    grad_adt. ADT = ssha_adt_base + mdt exactly as compute_channels defines it."""
    import xarray as xr
    from swot_eddy import preprocess as pp

    var_map = cfg["product"]["variables"][cfg["product"]["product_version"]]
    with xr.open_dataset(str(granule_path)) as ds:
        ds_c, _ = pp.crop_to_region(ds, cfg["region"])
        lon = _lon180(ds_c["longitude"].values)
        lat = ds_c["latitude"].values.astype("float64")
        mdt = ds_c[var_map["mdt"]].values.astype("float64")
        adt = ds_c[var_map["ssha_adt_base"]].values.astype("float64") + mdt
    return {"lon": lon, "lat": lat, "adt": adt, "mdt": mdt, "grad_adt": grad_mag_per_km(adt)}


FEATURE_COLS = [
    "dist_front_adt_km", "front_adt_missing", "dist_front_mdt_km", "front_mdt_missing",
    "adt_center_m", "mdt_center_m", "center_sample_dist_px",
    "grad_adt_center", "grad_adt_max_2r",
]


def pass_features(dets, granule_path, cfg):
    """Front/ADT features for all detections of one pass. `dets` is a DataFrame with
    lon, lat, radius_km (catalog rows, lon in -180..180). Returns a list of dicts
    aligned with dets' rows (FEATURE_COLS)."""
    fcfg = cfg.get("features", {})
    level = float(fcfg.get("front_level_m", 0.45))
    min_v = int(fcfg.get("front_min_vertices", 20))

    nf = native_fields(granule_path, cfg)
    front_adt = front_vertices(nf["adt"], nf["lon"], nf["lat"], level, min_v)
    front_mdt = front_vertices(nf["mdt"], nf["lon"], nf["lat"], level, min_v)

    rows = []
    for det in dets.itertuples():
        line, col = nearest_pixel(nf["lon"], nf["lat"], det.lon, det.lat)
        adt_c, samp_px = sample_nearest_finite(nf["adt"], line, col)
        mdt_c, _ = sample_nearest_finite(nf["mdt"], line, col)
        grad_c, _ = sample_nearest_finite(nf["grad_adt"], line, col)
        d_adt, miss_adt = signed_front_dist_km(det.lon, det.lat, adt_c, level, front_adt)
        d_mdt, miss_mdt = signed_front_dist_km(det.lon, det.lat, mdt_c, level, front_mdt)

        r_px = max(int(round(det.radius_km / KM_PER_PIXEL)), 1) * 2
        l0, l1 = max(0, line - r_px), min(nf["adt"].shape[0], line + r_px + 1)
        c0, c1 = max(0, col - r_px), min(nf["adt"].shape[1], col + r_px + 1)
        win = nf["grad_adt"][l0:l1, c0:c1]
        grad_max = float(np.nanmax(win)) if np.isfinite(win).any() else np.nan

        rows.append({
            "dist_front_adt_km": round(float(d_adt[0]), 2) if np.isfinite(d_adt[0]) else None,
            "front_adt_missing": bool(miss_adt[0] or len(front_adt) == 0),
            "dist_front_mdt_km": round(float(d_mdt[0]), 2) if np.isfinite(d_mdt[0]) else None,
            "front_mdt_missing": bool(miss_mdt[0] or len(front_mdt) == 0),
            "adt_center_m": round(adt_c, 4) if np.isfinite(adt_c) else None,
            "mdt_center_m": round(mdt_c, 4) if np.isfinite(mdt_c) else None,
            "center_sample_dist_px": samp_px if np.isfinite(samp_px) else None,
            "grad_adt_center": round(grad_c, 5) if np.isfinite(grad_c) else None,
            "grad_adt_max_2r": round(grad_max, 5) if np.isfinite(grad_max) else None,
        })
    return rows
