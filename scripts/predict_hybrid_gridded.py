"""HYBRID frontal-eddy detection on the regridded L2 GeoZarr (gridded, no swath frame).

Same two signals as the L3 chain, adapted to a regular lat/lon window:
py-eddy-tracker proposes closed contours (canonical RegularGridDataset mode, real
degree axes) on the biharmonic-inpainted swath footprint; the v2 U-Net scores
frontal-ness from raw (non-inpainted) channels, tiled 128x128 in BOTH dimensions.
frontal = max U-Net probability inside the PET speed contour > tau.

The window comes from scripts/fetch_zarr_window.py (auto-invoked under the
geozarr venv, cfg["l2zarr"]["fetch_python"]). The L3 swath chain is untouched.

Differences vs the swath recipe, carried knowingly:
- PET runs once on the whole window (no 128-px tiling / dedup needed)
- nadir rejection generalized: the nadir gap is a tilted stripe here, so the
  column-band rule becomes "drop if >= nadir_reject_frac of the enclosed AREA
  was inpainted" (same biharmonic-invented-blob failure mode)
- store ssha is unfiltered (crossover-corrected); tau carried over uncalibrated

Usage:
  .venv/bin/python scripts/predict_hybrid_gridded.py --cycle 29 --pass 91 --report
  .venv/bin/python scripts/predict_hybrid_gridded.py WINDOW.nc --report

Outputs in out/aoi5/predictions/ under the SWOT_L2Z_* stem:
  _hybrid.csv, _hybrid_frontal.csv, _hybrid.geojson,
  _hybrid_eddies.nc (window grid + eddy_flag/eddy_id), _hybrid_map.png (--report)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*partition.*will ignore the 'mask'.*")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swot_eddy.config import load_config
from swot_eddy.aoi import load_aoi
from swot_eddy import unet as U

C_FRONTAL, C_OTHER, C_FRONT = "#7b3294", "#666666", "#B8860B"
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQ = 111.320


def fetch_window(cfg, cycle, pass_num, refetch):
    root = Path(__file__).resolve().parents[1]
    lz = cfg["l2zarr"]
    out_dir = root / lz["out_dir"]
    pat = f"SWOT_L2Z_Expert_{cycle:03d}_{pass_num:03d}_*.nc"
    have = sorted(out_dir.glob(pat))
    if have and not refetch:
        return have[0]
    env = load_aoi(cfg).envelope_region()
    m = 0.5
    cmd = [lz["fetch_python"], str(root / "scripts/fetch_zarr_window.py"),
           "--store", lz["store"], "--s3-region", lz["s3_region"],
           "--cycle", str(cycle), "--pass", str(pass_num),
           "--bbox", str(env["lon_min"] - m), str(env["lon_max"] + m),
           str(env["lat_min"] - m), str(env["lat_max"] + m),
           "--out", str(out_dir)]
    subprocess.run(cmd, check=True)
    have = sorted(out_dir.glob(pat))
    if not have:
        sys.exit("fetch reported success but no window file found")
    return have[0]


def swath_footprint(finite: np.ndarray) -> np.ndarray:
    """Swath coverage incl. nadir/rain gaps: morphological closing (16 km) of the
    finite-pixel mask. Off-swath ocean stays outside."""
    from skimage.morphology import binary_closing, disk, remove_small_objects

    fp = binary_closing(finite, disk(8))
    return remove_small_objects(fp, 64)


def contour_pixel_mask(clon, clat, lon360, lat1d, shape):
    """Boolean window mask of pixels enclosed by a (lon, lat) contour."""
    from matplotlib.path import Path as MplPath

    dlon = lon360[1] - lon360[0]
    dlat = lat1d[1] - lat1d[0]
    cols = (np.asarray(clon) - lon360[0]) / dlon
    rows = (np.asarray(clat) - lat1d[0]) / dlat
    r0, r1 = int(np.floor(rows.min())), int(np.ceil(rows.max())) + 1
    c0, c1 = int(np.floor(cols.min())), int(np.ceil(cols.max())) + 1
    r0, r1 = max(0, r0), min(shape[0], r1)
    c0, c1 = max(0, c0), min(shape[1], c1)
    mask = np.zeros(shape, bool)
    if r1 <= r0 or c1 <= c0:
        return mask
    rr, cc = np.mgrid[r0:r1, c0:c1]
    pts = np.column_stack([cc.ravel(), rr.ravel()])
    inside = MplPath(np.column_stack([cols, rows])).contains_points(pts)
    sel = pts[inside]
    mask[sel[:, 1], sel[:, 0]] = True
    return mask


def grad_mag_per_km(field, lat1d, dlon_deg, dlat_deg):
    """|grad| in per-km units on the geographic grid (dx shrinks with cos lat)."""
    dy_km = dlat_deg * KM_PER_DEG_LAT
    dx_km = dlon_deg * KM_PER_DEG_LON_EQ * np.cos(np.deg2rad(lat1d))[:, None]
    gy = np.gradient(field, axis=0) / dy_km
    gx = np.gradient(field, axis=1) / dx_km
    return np.hypot(gx, gy)


def unet_prob_2d(x, model):
    """Average sigmoid probability over 128x128 tiles, stride 64 in both dims."""
    import torch

    _, H, W = x.shape
    T, S = U.TILE, 64

    def starts(n):
        s = list(range(0, max(n - T, 0) + 1, S)) or [0]
        if s[-1] != max(n - T, 0):
            s.append(max(n - T, 0))
        return s

    prob = np.zeros((H, W), np.float32)
    hits = np.zeros((H, W), np.float32)
    with torch.no_grad():
        for r in starts(H):
            for c in starts(W):
                xt = x[:, r:r + T, c:c + T]
                ph, pw = T - xt.shape[1], T - xt.shape[2]
                if ph or pw:
                    xt = np.pad(xt, ((0, 0), (0, ph), (0, pw)))
                p = torch.sigmoid(model(
                    torch.from_numpy(U.normalize(xt))[None]))[0, 0].numpy()
                he, we = min(r + T, H), min(c + T, W)
                prob[r:he, c:we] += p[:he - r, :we - c]
                hits[r:he, c:we] += 1
    return prob / np.maximum(hits, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("window", nargs="?", help="window .nc (else use --cycle/--pass)")
    ap.add_argument("--cycle", type=int)
    ap.add_argument("--pass", dest="pass_num", type=int)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--min-radius", type=float, default=None,
                    help="km, on PET speed radius; default = config pet.params."
                         "min_radius_km (5). The regridder's quality masking clips "
                         "speed contours, so gridded radii run smaller than L3.")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()

    import torch
    import xarray as xr
    from skimage.restoration import inpaint_biharmonic

    cfg = load_config()
    aoi = load_aoi(cfg)
    root = Path(__file__).resolve().parents[1]

    if args.window:
        wpath_nc = Path(args.window)
    else:
        if args.cycle is None or args.pass_num is None:
            sys.exit("give a window file or --cycle and --pass")
        wpath_nc = fetch_window(cfg, args.cycle, args.pass_num, args.refetch)
    print(f"window: {wpath_nc.name}")

    with xr.open_dataset(wpath_nc) as w:
        ssha = w["ssha"].values.astype("f8")
        adt = w["adt"].values.astype("f8")
        lat1d = w["latitude"].values.astype("f8")
        lon1d = w["longitude"].values.astype("f8")
        date = dt.datetime.fromisoformat(w.attrs["slice_time"])
        cycle = int(w.attrs["cycle"])
        pass_num = int(w.attrs["pass"])
    lon360 = lon1d % 360
    dlon = float(np.mean(np.diff(lon360)))
    dlat = float(np.mean(np.diff(lat1d)))

    # 1. footprint + biharmonic inpaint of the detection field (recipe algorithm)
    finite = np.isfinite(ssha)
    fp = swath_footprint(finite)
    inpaint_mask = fp & ~finite
    field = np.where(finite, ssha, 0.0)
    if inpaint_mask.any():
        field = inpaint_biharmonic(field, inpaint_mask)
    print(f"footprint {int(fp.sum())} px, inpainted {int(inpaint_mask.sum())} px "
          f"({inpaint_mask.sum() / max(fp.sum(), 1):.0%} of footprint)")

    # 2. PET, canonical regular-grid mode with real degree axes
    from py_eddy_tracker.dataset.grid import RegularGridDataset

    p = cfg["pet"]["params"]
    masked = np.ma.array(field.T, mask=~fp.T)          # (x=lon, y=lat) order
    grid = RegularGridDataset.with_array(
        coordinates=("x", "y"),
        datas={"x": lon360, "y": lat1d, "ssha": masked},
        variables_description={"ssha": {"units": "m"}},
        centered=True,
    )
    grid.add_uv("ssha")
    anticyc, cyc_ = grid.eddy_identification(
        "ssha", "u", "v", date,
        step=p["min_amplitude_m"] / 2.0, shape_error=p["shape_error"],
        pixel_limit=(p.get("pixel_limit_min", 4), p.get("pixel_limit_max", 1000)),
        force_height_unit="m", force_speed_unit="m/s",
    )

    # 3. post-filters (same thresholds as the swath recipe)
    min_radius = args.min_radius if args.min_radius is not None else p["min_radius_km"]
    dets = []
    for polarity, eddies in (("anticyclonic", anticyc), ("cyclonic", cyc_)):
        for k in range(len(eddies)):
            radius_km = float(eddies["radius_s"][k]) / 1000.0
            if radius_km < min_radius:
                continue
            elon, elat = float(eddies["lon"][k]), float(eddies["lat"][k])
            if not aoi.contains(elon, elat):
                continue
            clon = np.asarray(eddies["contour_lon_s"][k], dtype="f8") % 360
            clat = np.asarray(eddies["contour_lat_s"][k], dtype="f8")
            emask = contour_pixel_mask(clon, clat, lon360, lat1d, ssha.shape)
            if not emask.any():
                continue
            gap_frac = float(inpaint_mask[emask].mean())
            if gap_frac > p["gap_reject_frac"]:
                continue
            if gap_frac >= p.get("nadir_reject_frac", 0.70):
                continue
            dets.append({
                "polarity": polarity, "lon": elon, "lat": elat,
                "radius_km": radius_km,
                "radius_eff_km": float(eddies["radius_e"][k]) / 1000.0,
                "amplitude_m": float(eddies["amplitude"][k]),
                "gap_frac": gap_frac, "clon": clon, "clat": clat, "mask": emask,
            })

    # 4. U-Net scoring on raw channels, 2-D tiling
    wts = Path(args.weights) if args.weights else root / "out/aoi5/model/unet_frontal_v2.pt"
    model = U.build_unet(in_ch=4)
    model.load_state_dict(torch.load(wts, map_location="cpu", weights_only=True))
    model.eval()
    grad = grad_mag_per_km(adt, lat1d, dlon, dlat)
    x = np.stack([ssha, adt, grad, finite.astype("f8")]).astype(np.float32)
    prob = unet_prob_2d(x, model)
    for d in dets:
        d["frontal_score"] = float(prob[d["mask"]].max())
        d["is_frontal"] = d["frontal_score"] > args.tau

    # 5. outputs
    import pandas as pd
    out_dir = root / "out/aoi5/predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = wpath_nc.stem
    rows = [{"id": i,
             "lon": round(((d["lon"] + 180) % 360) - 180, 4),
             "lat": round(d["lat"], 4),
             "radius_km": round(d["radius_km"], 1),
             "amplitude_m": round(d["amplitude_m"], 4),
             "polarity": d["polarity"],
             "frontal_score": round(d["frontal_score"], 3),
             "is_frontal": d["is_frontal"]} for i, d in enumerate(dets)]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{stem}_hybrid.csv", index=False)
    if len(df):
        df[df.is_frontal].to_csv(out_dir / f"{stem}_hybrid_frontal.csv", index=False)
    else:
        df.to_csv(out_dir / f"{stem}_hybrid_frontal.csv", index=False)

    feats = []
    for i, d in enumerate(dets):
        if not d["is_frontal"]:
            continue
        ring = [[round(((lo + 180) % 360) - 180, 4), round(la, 4)]
                for lo, la in zip(d["clon"], d["clat"])]
        if len(ring) >= 4:
            ring.append(ring[0])
            feats.append({"type": "Feature",
                          "geometry": {"type": "Polygon", "coordinates": [ring]},
                          "properties": {**{k: rows[i][k] for k in
                                            ("id", "lon", "lat", "radius_km",
                                             "polarity")},
                                         "max_p": rows[i]["frontal_score"]}})
    (out_dir / f"{stem}_hybrid.geojson").write_text(json.dumps(
        {"type": "FeatureCollection",
         "properties": {"method": "PET x U-Net hybrid (gridded L2 zarr)",
                        "tau": args.tau, "model": wts.name,
                        "cycle": cycle, "pass": pass_num},
         "features": feats}))

    flag = np.zeros(ssha.shape, np.uint8)
    eid = np.full(ssha.shape, -1, np.int16)
    for i in sorted(range(len(dets)), key=lambda i: dets[i]["is_frontal"]):
        flag[dets[i]["mask"]] = 2 if dets[i]["is_frontal"] else 1
        eid[dets[i]["mask"]] = i
    out_nc = xr.Dataset(
        {"ssha": (("latitude", "longitude"), ssha.astype("f4")),
         "adt": (("latitude", "longitude"), adt.astype("f4")),
         "eddy_flag": (("latitude", "longitude"), flag),
         "eddy_id": (("latitude", "longitude"), eid)},
        coords={"latitude": lat1d, "longitude": lon1d},
        attrs={"method": "PET x U-Net hybrid on regridded L2 GeoZarr window",
               "tau": args.tau, "weights": wts.name, "cycle": cycle,
               "pass": pass_num, "slice_time": str(date),
               "source_window": wpath_nc.name})
    out_nc["eddy_flag"].attrs = {
        "long_name": "hybrid frontal-eddy detection mask (filled contour interior)",
        "flag_values": np.array([0, 1, 2], np.uint8),
        "flag_meanings": "background eddy_interior frontal_eddy_interior"}
    out_nc["eddy_id"].attrs = {"long_name": "eddy id matching *_hybrid.csv",
                               "comment": "-1 = background"}
    out_nc.to_netcdf(out_dir / f"{stem}_hybrid_eddies.nc",
                     encoding={v: {"zlib": True, "complevel": 4}
                               for v in out_nc.data_vars})

    nf = sum(d["is_frontal"] for d in dets)
    print(f"{stem}: {len(dets)} PET eddies, {nf} frontal (tau={args.tau})")
    if len(df):
        print(df[df.is_frontal][["id", "lon", "lat", "radius_km", "polarity",
                                 "frontal_score"]].to_string(index=False))

    if args.report:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from swot_eddy.features import front_vertices
        plt.rcParams.update({"font.family": "serif", "font.size": 8,
                             "axes.linewidth": 0.6})
        try:
            import cartopy.crs as ccrs
            PC = ccrs.PlateCarree()
        except Exception:
            PC = None
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        front = front_vertices(adt, lon2d, lat2d,
                               cfg["features"]["front_level_m"], 20)
        ssha_cm = ssha * 100
        v1, v2 = np.nanpercentile(ssha_cm, 2), np.nanpercentile(ssha_cm, 98)
        kw = {"transform": PC} if PC else {}
        fig = plt.figure(figsize=(5.2, 6.4))
        if PC is not None:
            ax = fig.add_axes([0.08, 0.10, 0.86, 0.82], projection=PC)
            ax.coastlines("10m", linewidth=0.5)
            gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.85",
                              linestyle=":")
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
        else:
            ax = fig.add_axes([0.08, 0.10, 0.86, 0.82])
        pm = ax.pcolormesh(lon1d, lat1d, ssha_cm, cmap="RdBu_r", vmin=v1, vmax=v2,
                           shading="nearest", **kw)
        if len(front):
            ax.plot(front[:, 0], front[:, 1], ".", color=C_FRONT, ms=1.0, **kw)
        for d in dets:
            lo = ((d["clon"] + 180) % 360) - 180
            ax.plot(lo, d["clat"],
                    color=C_FRONTAL if d["is_frontal"] else C_OTHER,
                    lw=1.8 if d["is_frontal"] else 0.8, **kw)
        cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.07,
                          shrink=0.85, aspect=35)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label("SSHA (cm)", fontsize=7)
        ax.set_title(f"{stem} (L2 zarr, gridded)\n{len(dets)} PET eddies, "
                     f"{nf} frontal (violet)", fontsize=8)
        fig.savefig(out_dir / f"{stem}_hybrid_map.png", dpi=150,
                    bbox_inches="tight")
        print(f"map -> {out_dir / (stem + '_hybrid_map.png')}")


if __name__ == "__main__":
    main()
