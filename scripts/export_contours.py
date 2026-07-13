"""Export PET eddy CONTOURS (speed-contour polygons) for every catalog detection. Resumable.

The catalog CSV keeps only scalars (center, radius, amplitude); the contour shape PET drew
lives on the Detection object and dies with the session. This re-runs detection per pass
(deterministic -- same order as run_all, so (cycle, pass, kind, eddy=list-index) matches the
catalog/labels key) and writes:

  out/aoi5/catalog/contours.csv       long format, one row per contour vertex:
                                      cycle, pass, kind, eddy, v, line, col, lon, lat
                                      (line/col = full-pass pixel incl. pad; lon in -180..180,
                                       empty where the vertex sits on padding)
  out/aoi5/catalog/contours_geom.csv  one row per eddy: center_line, center_col, n_vertices,
                                      match_catalog (n eddies found == catalog rows for the pass)
  out/aoi5/catalog/contours_done.csv  per-pass manifest (resume marker)

Existing catalog/label CSVs are never touched. Full run re-detects all ~1,280 passes
(~4-5 h) -- launch in the background:
  nohup .venv/bin/python scripts/export_contours.py > out/export_contours.log 2>&1 &
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*partition.*will ignore the 'mask'.*")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swot_eddy.config import load_config
from swot_eddy.aoi import load_aoi
from swot_eddy import preprocess as pp
from swot_eddy.detect import detect_pass_split
from swot_eddy.features import find_granule

VERT_COLS = ["cycle", "pass", "kind", "eddy", "v", "line", "col", "lon", "lat"]
GEOM_COLS = ["cycle", "pass", "kind", "eddy", "center_line", "center_col",
             "n_vertices", "match_catalog"]


def _append(path: Path, rows: list, cols: list):
    if not rows:
        return
    pd.DataFrame(rows, columns=cols).to_csv(path, mode="a", header=not path.exists(), index=False)


def _pass_date(name: str):
    import datetime as dt
    import re
    m = re.search(r"_(\d{8})T(\d{6})_", name)
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else dt.datetime(2023, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", nargs="*", type=int)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    aoi = load_aoi(cfg)
    out = Path(cfg["out_root"]) / "aoi5" / "catalog"
    verts_csv, geom_csv, manifest = out / "contours.csv", out / "contours_geom.csv", out / "contours_done.csv"
    if args.redo:
        for p in (verts_csv, geom_csv, manifest):
            p.unlink(missing_ok=True)

    cat = pd.read_csv(out / "detections.csv", dtype={"cycle": str, "pass": str})
    counts = cat.groupby(["cycle", "pass", "kind"]).size()
    passes = cat[["cycle", "pass"]].drop_duplicates().itertuples(index=False)
    passes = [(c, p) for c, p in passes
              if not args.cycles or int(c.split("_")[1]) in args.cycles]

    done = set()
    if manifest.exists():
        m = pd.read_csv(manifest, dtype={"cycle": str, "pass": str})
        done = set(zip(m["cycle"], m["pass"]))
    todo = [(c, p) for c, p in passes if (c, p) not in done]
    print(f"{len(passes)} catalog passes; {len(done)} done, {len(todo)} left")

    for n, (cyc, pnum) in enumerate(todo, 1):
        f = find_granule(cfg["data_root"], cyc, pnum)
        if f is None:
            _append(manifest, [{"cycle": cyc, "pass": pnum, "status": "no_granule"}],
                    ["cycle", "pass", "status"])
            continue
        try:
            tiles, summ, clon, clat = pp.preprocess_pass(str(f), cfg)
            kept, rejected, _ = detect_pass_split(tiles, cfg, clon, clat, _pass_date(f.name))
            kept = [d for d in kept if aoi.contains(d.center_lon, d.center_lat)]
            rejected = [d for d in rejected if aoi.contains(d.center_lon, d.center_lat)]
        except Exception as e:
            _append(manifest, [{"cycle": cyc, "pass": pnum,
                                "status": f"error:{type(e).__name__}"}],
                    ["cycle", "pass", "status"])
            print(f"  ERROR {cyc}/{pnum}: {type(e).__name__}: {e}")
            continue

        vrows, grows = [], []
        for kind, dets in (("keep", kept), ("reject", rejected)):
            match = len(dets) == int(counts.get((cyc, pnum, kind), 0))
            for eddy, d in enumerate(dets):
                pix = d.contour_pix or []
                geo = d.contour_lonlat or []
                for v, (lc, ll) in enumerate(zip(pix, list(geo) + [None] * (len(pix) - len(geo)))):
                    lon = None if ll is None else round(((ll[0] + 180) % 360) - 180, 4)
                    lat = None if ll is None else round(ll[1], 4)
                    vrows.append({"cycle": cyc, "pass": pnum, "kind": kind, "eddy": eddy,
                                  "v": v, "line": int(lc[0]), "col": int(lc[1]),
                                  "lon": lon, "lat": lat})
                grows.append({"cycle": cyc, "pass": pnum, "kind": kind, "eddy": eddy,
                              "center_line": d.center_line, "center_col": d.center_col,
                              "n_vertices": len(pix), "match_catalog": match})
        _append(verts_csv, vrows, VERT_COLS)
        _append(geom_csv, grows, GEOM_COLS)
        _append(manifest, [{"cycle": cyc, "pass": pnum, "status": "ok"}],
                ["cycle", "pass", "status"])
        if n % 20 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {cyc}/{pnum}: {len(grows)} eddies, {len(vrows)} vertices")

    print(f"done -> {verts_csv}")


if __name__ == "__main__":
    main()
