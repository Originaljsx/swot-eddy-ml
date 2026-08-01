"""Batch hybrid detection over the whole archive + per-frontal-eddy zoom figures.

Runs the PET x U-Net hybrid (same recipe as predict_hybrid.py, model loaded once)
on every L3 Expert granule, accumulates a single detection catalog, and renders
the approved eddy-zoom figure (make_eddy_zoom_figs.one_eddy) for every frontal
detection. No per-pass NetCDF copies (that is export_eddy_nc.py's job) — figures
are built from in-memory fields, which keeps the full 1,288-pass run ~8 h instead
of ~100 h on /mnt/c.

Resumable: passes listed in the manifest are skipped on rerun (--redo to force).

Usage:
  .venv/bin/python scripts/run_hybrid_all.py                 # everything
  .venv/bin/python scripts/run_hybrid_all.py --cycles 42 43  # subset
  .venv/bin/python scripts/run_hybrid_all.py --no-figs       # catalog only

Outputs:
  out/aoi5/batch/hybrid_detections.csv   every PET eddy, all passes, with scores
  out/aoi5/batch/passes_done.csv         manifest (resume + error log)
  out/aoi5/eddy_zoom_all/eddy_{cycle}_{pass}_{id:02d}.png   frontal eddies only
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*partition.*will ignore the 'mask'.*")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from swot_eddy.config import load_config
from swot_eddy.aoi import load_aoi
from swot_eddy import preprocess as pp
from swot_eddy.detect import detect_pass_split
from swot_eddy.features import raw_channels, native_fields
from swot_eddy import unet as U

TAU = 0.9
CAT_COLS = ["cycle", "pass", "date", "id", "lon", "lat", "radius_km",
            "radius_eff_km", "amplitude_m", "polarity", "frontal_score",
            "is_frontal"]
MAN_COLS = ["cycle", "pass", "granule", "status", "n_pet", "n_frontal"]


def unet_prob(x, model):
    import torch
    L = x.shape[1]
    prob = np.zeros((L, 128), np.float32)
    hits = np.zeros((L, 128), np.float32)
    with torch.no_grad():
        for s, xt in U.pass_tiles(x):
            p = torch.sigmoid(model(torch.from_numpy(U.normalize(xt))[None]))[0, 0].numpy()
            e = min(s + U.TILE, L)
            prob[s:e] += p[:e - s]
            hits[s:e] += 1
    return prob / np.maximum(hits, 1)


def process_pass(granule, cfg, aoi, model, make_figs, fig_dir):
    import pandas as pd
    from skimage.draw import polygon as sk_polygon

    name = granule.name
    m = re.search(r"_(\d{3})_(\d{3})_(\d{8})T(\d{6})_", name)
    cycle, pas = m.group(1), m.group(2)
    date = dt.datetime.strptime(m.group(3) + m.group(4), "%Y%m%d%H%M%S")

    tiles, summ, clon, clat = pp.preprocess_pass(str(granule), cfg)
    kept, _, _ = detect_pass_split(tiles, cfg, clon, clat, date)
    kept = [d for d in kept if aoi.contains(d.center_lon, d.center_lat)]
    x, _ = raw_channels(granule, cfg)
    prob = unet_prob(x.astype(np.float32), model)

    rows, masks = [], []
    for i, d in enumerate(kept):
        mask = np.zeros(prob.shape, bool)
        if d.contour_pix:
            cp = np.array(d.contour_pix)
            rr, cc = sk_polygon(cp[:, 0], cp[:, 1], shape=prob.shape)
            mask[rr, cc] = True
        masks.append(mask)
        score = float(prob[mask].max()) if mask.any() else \
            float(prob[d.center_line, d.center_col])
        lon180 = None if d.center_lon is None else ((d.center_lon + 180) % 360) - 180
        rows.append({"cycle": cycle, "pass": pas, "date": date.isoformat(),
                     "id": i,
                     "lon": None if lon180 is None else round(lon180, 4),
                     "lat": None if d.center_lat is None else round(d.center_lat, 4),
                     "radius_km": round(d.radius_km, 1),
                     "radius_eff_km": round(d.radius_eff_km, 1),
                     "amplitude_m": round(d.amplitude_m, 4),
                     "polarity": d.polarity,
                     "frontal_score": round(score, 3),
                     "is_frontal": score > TAU})

    n_frontal = sum(r["is_frontal"] for r in rows)
    if make_figs and n_frontal:
        from make_eddy_zoom_figs import one_eddy
        fig_dir = fig_dir / f"cycle_{cycle}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        nf = native_fields(granule, cfg)
        W = nf["lon"].shape[1]
        pad_left = (128 - W) // 2
        with __import__("xarray").open_dataset(granule) as ds0:
            ds_c, _ = pp.crop_to_region(ds0, cfg["region"])
            ssha = ds_c["ssha_filtered"].values * 100.0
            ugos = ds_c["ugos_filtered"].values
            vgos = ds_c["vgos_filtered"].values
            ugosa = ds_c["ugosa_filtered"].values
            vgosa = ds_c["vgosa_filtered"].values
        eid = np.full(nf["lon"].shape, -1, np.int16)
        for i, mask in enumerate(masks):
            if rows[i]["is_frontal"]:
                eid[mask[:, pad_left:pad_left + W]] = i
        det = pd.DataFrame(rows).set_index("id")
        d = dict(lon=nf["lon"], lat=nf["lat"], adt=nf["adt"] * 100.0,
                 mdt=nf["mdt"] * 100.0, ssha=ssha, ugos=ugos, vgos=vgos,
                 ugosa=ugosa, vgosa=vgosa, eid=eid, det=det,
                 time=date.isoformat()[:16], cycle=cycle, pas=pas)
        for i in det[det["is_frontal"]].index:
            try:
                one_eddy(d, i, 1.3, fig_dir)
            except Exception as exc:
                print(f"  fig eddy {i} failed: {exc}")

    return rows, n_frontal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, nargs="*", default=None)
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.6})

    cfg = load_config()
    aoi = load_aoi(cfg)
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "out/aoi5/batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = root / "out/aoi5/eddy_zoom_all"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cat_path = out_dir / "hybrid_detections.csv"
    man_path = out_dir / "passes_done.csv"

    model = U.build_unet(in_ch=4)
    model.load_state_dict(torch.load(root / "out/aoi5/model/unet_frontal_v2.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()

    done = set()
    if man_path.exists() and not args.redo:
        import pandas as pd
        man = pd.read_csv(man_path, dtype={"cycle": str, "pass": str})
        done = set(zip(man["cycle"], man["pass"]))
    for path, cols in ((cat_path, CAT_COLS), (man_path, MAN_COLS)):
        if not path.exists() or args.redo:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(cols)
    if args.redo:
        done = set()

    granules = sorted(Path(cfg["data_root"]).glob(
        "cycle_*/SWOT_L3_LR_SSH_Expert_*.nc"))
    if args.cycles:
        want = {f"{c:03d}" for c in args.cycles}
        granules = [g for g in granules
                    if re.search(r"_(\d{3})_\d{3}_", g.name).group(1) in want]
    print(f"{len(granules)} granules, {len(done)} already done")

    n_run = 0
    for g in granules:
        m = re.search(r"_(\d{3})_(\d{3})_", g.name)
        key = (m.group(1), m.group(2))
        if key in done:
            continue
        try:
            rows, n_frontal = process_pass(g, cfg, aoi, model,
                                           not args.no_figs, fig_dir)
            status = "ok"
        except Exception as exc:
            rows, n_frontal, status = [], 0, f"error: {exc}"
        with open(cat_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CAT_COLS)
            for r in rows:
                w.writerow(r)
        with open(man_path, "a", newline="") as f:
            csv.writer(f).writerow([key[0], key[1], g.name, status,
                                    len(rows), n_frontal])
        n_run += 1
        print(f"[{n_run}] {key[0]}/{key[1]}: {len(rows)} eddies, "
              f"{n_frontal} frontal ({status})", flush=True)
    print("done")


if __name__ == "__main__":
    main()
