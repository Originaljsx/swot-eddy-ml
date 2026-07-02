"""Detect eddies over ALL cycles/passes -> a master catalog CSV. Resumable.

For each pass: preprocess -> PET detect -> drop nadir/gap (in detect_pass) -> keep in-box.
Appends one row per kept eddy to out/catalog/detections.csv, and one row per pass to
out/catalog/passes_done.csv (so 0-eddy passes are still marked done). Re-running skips passes
already in the manifest, so you can stop/resume freely.

Usage:
  python scripts/run_all.py                  # all 50 cycles, catalog only (fast)
  python scripts/run_all.py --png            # also write per-pass review PNGs (slow, ~1 GB)
  python scripts/run_all.py --cycles 4 43    # only cycle_004 + cycle_043
  python scripts/run_all.py --redo           # ignore the manifest, reprocess everything

Tip: long run -> launch in the background and tail the log:
  nohup .venv/bin/python scripts/run_all.py > out/run_all.log 2>&1 &
  tail -f out/run_all.log
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

# Silence PET's noisy "'partition' will ignore the 'mask'" UserWarning (harmless: our tiles
# are fully inpainted with an all-False mask) so the progress log stays readable.
warnings.filterwarnings("ignore", message=r".*partition.*will ignore the 'mask'.*")

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swot_eddy.config import load_config
from swot_eddy.aoi import load_aoi
from swot_eddy import preprocess as pp
from swot_eddy.detect import detect_pass_split
from run_cycle import in_box, pass_date, render          # reuse the per-pass logic + figure

CATALOG_COLS = [
    "cycle", "pass", "date", "kind", "polarity", "lon", "lat", "radius_km", "radius_eff_km",
    "amplitude_m", "gap_frac", "nadir_frac", "center_in_nadir", "source_tile",
]
MANIFEST_COLS = ["cycle", "pass", "date", "n_detected", "n_kept", "n_rejected", "n_dropped", "status"]


def _append(path: Path, rows: list, cols: list):
    """Append rows to a CSV, writing the header only when the file is new."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def pass_number(name: str) -> str:
    m = re.search(r"_(\d{3})_(\d{3})_", name)            # ..._<cycle>_<pass>_...
    return m.group(2) if m else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true", help="also write a per-pass review PNG")
    ap.add_argument("--cycles", nargs="*", type=int, help="only these cycle numbers (e.g. 4 43)")
    ap.add_argument("--redo", action="store_true", help="ignore the manifest, reprocess all")
    args = ap.parse_args()

    cfg = load_config()
    region = cfg["region"]                            # AOI5 envelope (crop band); polygon filter below
    aoi = load_aoi(cfg)
    var_map = cfg["product"]["variables"][cfg["product"]["product_version"]]
    out = Path(cfg["out_root"]) / "aoi5" / "catalog"
    out.mkdir(parents=True, exist_ok=True)
    catalog, manifest = out / "detections.csv", out / "passes_done.csv"

    done = set()
    if manifest.exists() and not args.redo:
        m = pd.read_csv(manifest, dtype={"cycle": str, "pass": str})
        done = set(zip(m["cycle"], m["pass"]))
    if args.redo:
        for p in (catalog, manifest):
            p.unlink(missing_ok=True)

    cycle_dirs = sorted((Path(cfg["data_root"])).glob("cycle_*"))
    if args.cycles:
        want = {f"cycle_{n:03d}" for n in args.cycles}
        cycle_dirs = [d for d in cycle_dirs if d.name in want]

    n_all = sum(len(list(cd.glob("*.nc"))) for cd in cycle_dirs)   # total passes (denominator)
    print(f"resuming: {len(done)} passes already done, {n_all} total ({n_all - len(done)} left)")
    n_pass = n_eddies = n_skip = n_err = 0
    for cd in cycle_dirs:
        cycle = cd.name
        png_dir = Path(cfg["out_root"]) / "aoi5" / f"{cycle}_review"
        for f in sorted(cd.glob("*.nc")):
            pnum = pass_number(f.name)
            if (cycle, pnum) in done:
                n_skip += 1
                continue
            date = pass_date(f.name)
            try:
                tiles, summ, clon, clat = pp.preprocess_pass(str(f), cfg)
                kept_all, rejected_all, dsum = detect_pass_split(tiles, cfg, clon, clat, date)
                kept = [d for d in kept_all if in_box(d, aoi)]
                rejected = [d for d in rejected_all if in_box(d, aoi)]
            except ValueError:                           # pass does not intersect the box
                _append(manifest, [{"cycle": cycle, "pass": pnum, "date": date.isoformat(),
                                    "n_detected": 0, "n_kept": 0, "n_rejected": 0, "n_dropped": 0,
                                    "status": "no_box_intersection"}], MANIFEST_COLS)
                done.add((cycle, pnum)); n_skip += 1
                continue
            except Exception as e:                       # PET / IO failure -> log, keep going
                print(f"  ERROR {cycle}/{pnum}: {type(e).__name__}: {e}")
                _append(manifest, [{"cycle": cycle, "pass": pnum, "date": date.isoformat(),
                                    "n_detected": 0, "n_kept": 0, "n_rejected": 0, "n_dropped": 0,
                                    "status": f"error:{type(e).__name__}"}], MANIFEST_COLS)
                done.add((cycle, pnum)); n_err += 1
                continue

            def _row(d, kind):
                return {
                    "cycle": cycle, "pass": pnum, "date": date.isoformat(), "kind": kind,
                    "polarity": d.polarity,
                    "lon": None if d.center_lon is None else round(((d.center_lon + 180) % 360) - 180, 4),
                    "lat": None if d.center_lat is None else round(d.center_lat, 4),
                    "radius_km": round(d.radius_km, 2), "radius_eff_km": round(d.radius_eff_km, 2),
                    "amplitude_m": round(d.amplitude_m, 4), "gap_frac": round(d.gap_frac, 3),
                    "nadir_frac": round(d.nadir_frac, 3), "center_in_nadir": bool(d.center_in_nadir),
                    "source_tile": d.source_tile,
                }
            _append(catalog, [_row(d, "keep") for d in kept]
                    + [_row(d, "reject") for d in rejected], CATALOG_COLS)
            dropped = dsum.get("n_nadir_reject", 0) + dsum.get("n_gap_reject", 0)
            _append(manifest, [{"cycle": cycle, "pass": pnum, "date": date.isoformat(),
                                "n_detected": dsum.get("n_total", 0), "n_kept": len(kept),
                                "n_rejected": len(rejected), "n_dropped": dropped,
                                "status": "ok"}], MANIFEST_COLS)
            done.add((cycle, pnum))
            n_pass += 1; n_eddies += len(kept)

            if args.png:
                png_dir.mkdir(parents=True, exist_ok=True)
                with xr.open_dataset(str(f)) as dsr:
                    ds_c, _ = pp.crop_to_region(dsr, region)
                    native, _, _ = pp.compute_channels(ds_c, var_map, quality_max=None)
                    ua, va = pp.geostrophic_anomaly_uv(ds_c, var_map)
                    edge_cols = pp.swath_edge_cols(ds_c)
                render(native, np.hypot(ua, va), (ua, va), kept, tiles[0].pad_left, edge_cols,
                       f"{f.name}\n{len(kept)} in-box ({dsum['n_total']} detected, {dropped} dropped)",
                       png_dir / (f.stem + ".png"))
            print(f"  {cycle}/{pnum}: detected={dsum.get('n_total',0):2d} kept={len(kept):2d}  "
                  f"[DONE {len(done)}/{n_all} passes  (+{n_pass} this session, {n_eddies} eddies)]")

    print(f"\nDONE. processed {n_pass} passes ({n_eddies} kept eddies), skipped {n_skip}, errors {n_err}.")
    print(f"catalog : {catalog}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
