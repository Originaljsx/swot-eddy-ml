"""Export the labeled eddy dataset: hand labels joined to the full PET catalog.

Joins out/aoi5/labels/eddy_labels_aoi5.csv to out/aoi5/catalog/detections.csv on
the deterministic key (cycle, pass, kind, eddy=cumcount) and writes one release
CSV carrying every py-eddy-tracker variable (radius, effective radius, amplitude,
gap/nadir fractions, ...) plus the human label, and a second CSV with the PET
contour vertices for the labeled eddies.

Usage:
  .venv/bin/python scripts/export_labeled_dataset.py

Writes:
  out/aoi5/release/labeled_eddies_aoi5.csv
  out/aoi5/release/labeled_eddies_contours.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["cycle", "pass", "kind", "eddy"]

cat = pd.read_csv(ROOT / "out/aoi5/catalog/detections.csv")
cat["eddy"] = cat.groupby(["cycle", "pass", "kind"]).cumcount()
lab = pd.read_csv(ROOT / "out/aoi5/labels/eddy_labels_aoi5.csv")

merged = lab[KEY + ["label", "ts"]].merge(cat, on=KEY, how="left", validate="1:1")
n_miss = merged["lon"].isna().sum()
assert n_miss == 0, f"{n_miss} labels failed to join the catalog"

# cross-check: label rows carry their own lon/lat snapshot — must agree exactly
chk = lab.merge(cat, on=KEY, suffixes=("_lab", "_cat"))
assert np.allclose(chk["lon_lab"], chk["lon_cat"], atol=1e-4)
assert np.allclose(chk["lat_lab"], chk["lat_cat"], atol=1e-4)

cols = KEY + ["date", "polarity", "lon", "lat", "radius_km", "radius_eff_km",
              "amplitude_m", "gap_frac", "nadir_frac", "center_in_nadir",
              "source_tile", "label", "ts"]
out_dir = ROOT / "out/aoi5/release"
out_dir.mkdir(parents=True, exist_ok=True)
merged[cols].to_csv(out_dir / "labeled_eddies_aoi5.csv", index=False)

con = pd.read_csv(ROOT / "out/aoi5/catalog/contours.csv")
con_lab = con.merge(merged[KEY], on=KEY, how="inner")
con_lab.to_csv(out_dir / "labeled_eddies_contours.csv", index=False)

n_con = con_lab.groupby(KEY).ngroups
print(f"labeled_eddies_aoi5.csv:      {len(merged)} rows "
      f"({merged['label'].value_counts().to_dict()})")
print(f"labeled_eddies_contours.csv:  {len(con_lab)} vertices, {n_con} eddies")
