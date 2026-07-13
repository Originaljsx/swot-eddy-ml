"""Validate the ADT-front constraint against the hand labels (cycles 1-7).

For every labeled pass: extract the front line at a sweep of candidate ADT levels and
compute each labeled eddy's signed front distance. Then per level, report how well
|distance| separates frontal from open-ocean/coastal labels (the make-or-break check
for using front proximity as a labeling prior). Writes:

  out/aoi5/features/front_level_sweep.csv   per-level separation stats (adt + mdt front)
  out/aoi5/features/front_validation.png    signed-distance histograms by label (best level)
  out/aoi5/features/front_dists_labeled.csv per-eddy signed distances at the best level

Usage:  .venv/bin/python scripts/validate_front.py [--levels 0.2 0.25 ...]
Read-only w.r.t. all existing catalog/label CSVs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swot_eddy.config import load_config
from swot_eddy import features as ft

CLASSES = ["frontal", "coastal", "open ocean"]  # reject/skip excluded: junk sits anywhere


def auc_abs_dist(d_frontal, d_other):
    """AUC of -|dist| ranking frontal above other (1.0 = |dist| separates perfectly)."""
    a = -np.abs(d_frontal[np.isfinite(d_frontal)])
    b = -np.abs(d_other[np.isfinite(d_other)])
    if len(a) == 0 or len(b) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([a, b])).rank().values
    return (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", type=float,
                    default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    args = ap.parse_args()

    cfg = load_config()
    out_root = Path(cfg["out_root"]) / "aoi5"
    out_dir = out_root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv(out_root / "labels" / "eddy_labels_aoi5.csv",
                      dtype={"cycle": str, "pass": str})
    lab = lab[lab["label"].isin(CLASSES)].reset_index(drop=True)
    print(f"{len(lab)} class-labeled eddies over "
          f"{lab.groupby(['cycle', 'pass']).ngroups} passes; sweeping {args.levels}")

    min_v = int(cfg.get("features", {}).get("front_min_vertices", 20))
    data_root = Path(cfg["data_root"])
    # dists[(field, level)] = per-row signed distance aligned with lab
    dists = {(f, lv): np.full(len(lab), np.nan) for f in ("adt", "mdt") for lv in args.levels}
    missing = {k: np.zeros(len(lab), dtype=bool) for k in dists}

    for (cyc, pnum), grp in lab.groupby(["cycle", "pass"]):
        f = ft.find_granule(data_root, cyc, pnum)
        if f is None:
            print(f"  WARN no granule for {cyc}/{pnum}, skipping {len(grp)} labels")
            continue
        nf = ft.native_fields(f, cfg)
        idx = grp.index.values
        for field in ("adt", "mdt"):
            centers = np.array([
                ft.sample_nearest_finite(
                    nf[field], *ft.nearest_pixel(nf["lon"], nf["lat"], lo, la))[0]
                for lo, la in zip(grp.lon, grp.lat)])
            for lv in args.levels:
                fv = ft.front_vertices(nf[field], nf["lon"], nf["lat"], lv, min_v)
                d, miss = ft.signed_front_dist_km(grp.lon.values, grp.lat.values, centers, lv, fv)
                dists[(field, lv)][idx] = d
                missing[(field, lv)][idx] = miss | (len(fv) == 0)

    rows = []
    is_f = (lab["label"] == "frontal").values
    is_o = (lab["label"] == "open ocean").values
    is_c = (lab["label"] == "coastal").values
    for (field, lv), d in sorted(dists.items()):
        ad = np.abs(d)
        rows.append({
            "field": field, "level_m": lv,
            "missing_frac": round(float(missing[(field, lv)].mean()), 3),
            "frontal_med_km": round(float(np.nanmedian(ad[is_f])), 1),
            "open_med_km": round(float(np.nanmedian(ad[is_o])), 1),
            "coastal_med_km": round(float(np.nanmedian(ad[is_c])), 1),
            "frontal_within50": round(float(np.nanmean(ad[is_f] < 50)), 3),
            "open_within50": round(float(np.nanmean(ad[is_o] < 50)), 3),
            "auc_frontal_vs_open": round(float(auc_abs_dist(d[is_f], d[is_o])), 3),
            "auc_frontal_vs_rest": round(float(auc_abs_dist(d[is_f], d[is_o | is_c])), 3),
        })
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / "front_level_sweep.csv", index=False)
    print("\n", sweep.to_string(index=False))

    best = sweep[sweep.field == "adt"].sort_values("auc_frontal_vs_open").iloc[-1]
    lv = float(best.level_m)
    print(f"\nbest adt level by frontal-vs-open AUC: {lv:.2f} m "
          f"(AUC {best.auc_frontal_vs_open}, frontal within 50 km: {best.frontal_within50:.0%})")

    per_eddy = lab[["cycle", "pass", "kind", "eddy", "label", "lon", "lat"]].copy()
    per_eddy["dist_front_adt_km"] = dists[("adt", lv)]
    per_eddy["dist_front_mdt_km"] = dists[("mdt", lv)]
    per_eddy.to_csv(out_dir / "front_dists_labeled.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    bins = np.arange(-400, 401, 20)
    for ax, field in zip(axes, ("adt", "mdt")):
        d = dists[(field, lv)]
        for cls, color in zip(CLASSES, ("tab:red", "tab:green", "tab:blue")):
            ax.hist(d[(lab["label"] == cls).values], bins=bins, alpha=0.55,
                    label=cls, color=color)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel(f"signed dist to {field} front @ {lv:.2f} m [km]  (- inshore, + offshore)")
        ax.legend()
    axes[0].set_ylabel("labeled eddies")
    fig.suptitle("Front-distance by hand label (cycles 1-7)")
    fig.tight_layout()
    fig.savefig(out_dir / "front_validation.png", dpi=130)
    print(f"\nwrote {out_dir/'front_level_sweep.csv'}\n      {out_dir/'front_dists_labeled.csv'}"
          f"\n      {out_dir/'front_validation.png'}")


if __name__ == "__main__":
    main()
