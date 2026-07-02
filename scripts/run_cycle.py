"""Batch-detect one SWOT cycle and render a native-pass review figure per pass.

Runs the shared preprocess->PET pipeline over every .nc in a cycle dir, filters
detections to the study box (by center lon/lat), and writes one PNG per pass:
native (raw, no-inpaint) ssha_filtered + ADT with detection contours overlaid.

Usage: python scripts/run_cycle.py [cycle_004]
"""
from __future__ import annotations

import re
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swot_eddy.config import load_config
from swot_eddy.aoi import load_aoi
from swot_eddy import preprocess as pp
from swot_eddy.detect import detect_pass


def in_box(d, aoi) -> bool:
    """Detection center inside the GS_AOI5 polygon (point-in-polygon, NOT the bbox).
    aoi.contains() handles None centers + the 0..360 -> -180..180 lon normalization."""
    return aoi.contains(d.center_lon, d.center_lat)


def pass_date(name: str) -> dt.datetime:
    m = re.search(r"_(\d{8})T(\d{6})_", name)
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else dt.datetime(2023, 1, 1)


def render(native, speed_a, uv, dets, pad_left, edge_cols, title, out_path):
    ua, va = uv
    fig, ax = plt.subplots(1, 3, figsize=(10, 22))
    def overlay(a, fld, ttl, cmap, vmin, vmax, scale, label, quiver=False):
        # aspect="equal" -> native 2-km pixel proportions (true swath shape, undistorted)
        im = a.imshow(fld * scale, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
        for c in edge_cols:                         # 4 SWOT swath-edge lines (outer + nadir-gap)
            a.axvline(c, color="k", lw=0.8)
        if quiver:                                  # anomaly velocity vectors (real obs only)
            s = 6
            XX, YY = np.meshgrid(np.arange(0, native.shape[2], s), np.arange(0, native.shape[1], s))
            a.quiver(XX, YY, ua[::s, ::s], va[::s, ::s], color="k",
                     angles="xy", scale_units="xy", scale=0.06, width=0.0015, pivot="mid")
        for d in dets:
            c = "red" if d.polarity == "anticyclonic" else "blue"
            ls = "--" if d.gap_reject else "-"
            if d.contour_pix:
                cl = np.array(d.contour_pix, dtype="f8"); cl[:, 1] -= pad_left
                a.plot(cl[:, 1], cl[:, 0], color=c, lw=1.2, ls=ls)
            mk = "x" if d.center_in_nadir else ("s" if d.gap_reject else "o")
            a.plot(d.center_col - pad_left, d.center_line, mk, color=c, ms=5, mfc="none", mew=1.4)
        a.set_xlim(-0.5, native.shape[2] - 0.5)
        a.set_title(ttl); a.set_xlabel("across-track px (native 0..68)"); a.set_ylabel("along-track line")
        fig.colorbar(im, ax=a, shrink=0.8, pad=0.02).set_label(label)

    overlay(ax[0], native[0], "(a) ssha_filtered (SLA)", "RdBu_r", -10, 30, 100, "SLA (cm)")
    overlay(ax[1], native[1], "(b) native ADT", "Spectral_r", -0.1, 0.8, 1.0, "ADT (m)")
    overlay(ax[2], speed_a, "(c) geos. velocity anomaly", "turbo", 0, 0.6, 1.0, "|u'| (m/s)", quiver=True)
    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    cycle = sys.argv[1] if len(sys.argv) > 1 else "cycle_004"
    cfg = load_config()
    region = cfg["region"]                            # AOI5 envelope (crop band); polygon filter below
    aoi = load_aoi(cfg)
    var_map = cfg["product"]["variables"][cfg["product"]["product_version"]]
    cyc_dir = Path(cfg["data_root"]) / cycle
    out_dir = Path(cfg["out_root"]) / "aoi5" / f"{cycle}_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cyc_dir.glob("*.nc"))
    print(f"{cycle}: {len(files)} passes -> {out_dir}")
    rows = []
    for f in files:
        date = pass_date(f.name)
        try:
            tiles, summary, clon, clat = pp.preprocess_pass(str(f), cfg)
        except ValueError as e:                       # pass does not intersect box
            print(f"  SKIP {f.name}: {e}")
            continue
        dets, dsum = detect_pass(tiles, cfg, clon, clat, date)
        # gap_reject + nadir blobs already dropped in detect_pass; keep = inside AOI5 polygon.
        kept = [d for d in dets if in_box(d, aoi)]
        with xr.open_dataset(str(f)) as ds_raw:
            ds_c, _ = pp.crop_to_region(ds_raw, region)
            native, _, _ = pp.compute_channels(ds_c, var_map, quality_max=None)
            ua, va = pp.geostrophic_anomaly_uv(ds_c, var_map)
            edge_cols = pp.swath_edge_cols(ds_c)
        speed_a = np.hypot(ua, va)
        pad_left = tiles[0].pad_left
        out_png = out_dir / (f.stem + ".png")
        nnad, ngapd = dsum.get("n_nadir_reject", 0), dsum.get("n_gap_reject", 0)
        render(native, speed_a, (ua, va), kept, pad_left, edge_cols,
               f"{f.name}\n{len(kept)} in-box ({dsum['n_total']} detected, {nnad} nadir + {ngapd} gap dropped)", out_png)
        na = sum(d.polarity == "anticyclonic" for d in kept)
        nc = len(kept) - na
        rows.append((f.name[:46], summary["n_lines_cropped"], dsum["n_total"], len(kept), na, nc, nnad + ngapd))
        print(f"  {f.name[:46]}  lines={summary['n_lines_cropped']:4d}  detected={dsum['n_total']:2d}  "
              f"inbox={len(kept):2d} (A{na}/C{nc}) dropped(nadir+gap)={nnad + ngapd}")

    print("\nsummary  pass | lines | raw_dedup | inbox | A | C | nadir_dropped")
    for r in rows:
        print(" ", r)
    print(f"\nPNGs: {out_dir}")


if __name__ == "__main__":
    main()
