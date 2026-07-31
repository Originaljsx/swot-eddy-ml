"""Fetch one (cycle, pass) AOI window from the regridded L2 icechunk store.

Runs under the swot-250m-geozarr venv (icechunk 2 + zarr 3), NOT the eddy-ml venv
(which pins zarr<3 for py-eddy-tracker). scripts/predict_hybrid_gridded.py invokes
this via subprocess using cfg["l2zarr"]["fetch_python"].

Dependencies: icechunk, zarr, xarray, numpy only.

Usage:
  <geozarr-venv-python> scripts/fetch_zarr_window.py \
      --store s3://swot-zarr/ice/c001_002_expert.icechunk --s3-region us-west-2 \
      --cycle 29 --pass 91 --bbox -83.0 -76.0 23.0 43.2 --out out/aoi5/l2zarr

Writes <out>/SWOT_L2Z_Expert_{CCC}_{PPP}_{YYYYMMDD}T{HHMMSS}.nc with 2-D ssha,
adt, time_offset on 1-D latitude/longitude coords (the native level-0 lattice).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--s3-region", default="us-west-2")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--pass", dest="pass_num", type=int, required=True)
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("LON0", "LON1", "LAT0", "LAT1"))
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    import icechunk
    import xarray as xr

    assert args.store.startswith("s3://")
    bucket, _, prefix = args.store[5:].partition("/")
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix,
                                  region=args.s3_region, from_env=True)
    repo = icechunk.Repository.open(storage)
    ds = xr.open_zarr(repo.readonly_session("main").store, group="0",
                      consolidated=False)

    cyc = ds["cycle"].values
    pas = ds["pass"].values
    hit = np.where((cyc == args.cycle) & (pas == args.pass_num))[0]
    if hit.size == 0:
        have = sorted(int(c) for c in np.unique(cyc[pas == args.pass_num]))
        sys.exit(f"no slice for cycle {args.cycle} pass {args.pass_num}; "
                 f"cycles with pass {args.pass_num}: {have}")
    i = int(hit[0])

    lon0, lon1, lat0, lat1 = args.bbox
    sub = ds[["ssha", "adt", "time_offset"]].isel(time=i).sel(
        latitude=slice(lat0, lat1), longitude=slice(lon0, lon1))
    sub = sub.load()

    t = np.datetime64(ds["time"].values[i], "s")
    tstr = str(t).replace("-", "").replace(":", "")
    fname = f"SWOT_L2Z_Expert_{args.cycle:03d}_{args.pass_num:03d}_{tstr}.nc"

    sub.attrs = {}                # store root attrs carry non-HDF5 types (lists of dicts)
    for v in sub.variables:
        sub[v].attrs = {k: a for k, a in sub[v].attrs.items()
                        if isinstance(a, (str, int, float, np.ndarray))}
    sub.attrs.update({
        "source_store": args.store,
        "source_group": "0",
        "source_time_index": i,
        "slice_time": str(np.datetime64(ds["time"].values[i], "s")),
        "cycle": args.cycle,
        "pass": args.pass_num,
        "source_filename": str(ds["filename"].values[i])
        if "filename" in ds else "",
        "note": "AOI window of the regridded SWOT JPL L2 Expert GeoZarr level 0 "
                "(equirectangular 0.02 deg); ssha = ssha_karin + height_cor_xover, "
                "adt = ssha + MDT CNES-CLS",
    })
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / fname
    enc = {v: {"zlib": True, "complevel": 4} for v in sub.data_vars}
    tmp = out.with_suffix(".nc.tmp")
    sub.to_netcdf(tmp, encoding=enc)      # atomic: a crashed write never leaves a bad .nc
    tmp.replace(out)
    n_fin = int(np.isfinite(sub["ssha"].values).sum())
    print(f"{out} shape={dict(sub.sizes)} finite_ssha_px={n_fin}")


if __name__ == "__main__":
    main()
