# SWOT Gulf Stream Eddy Detection

[![AI-assisted: written by a human with help from AI tools](https://img.shields.io/static/v1?label=&message=AI-assisted&color=gold)](https://nasa-ammos.github.io/slim/?search=Badges)
[![Best Practices from SLIM: this project follows SLIM best practices](https://img.shields.io/badge/Best%20Practices%20from-SLIM-blue)](https://nasa-ammos.github.io/slim/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

Machine-learning detection of frontal eddies in [SWOT](https://podaac.jpl.nasa.gov/SWOT)
altimetry over the Gulf Stream. The production detector is a hybrid:
py-eddy-tracker proposes closed-contour eddies on an inpainted pass, and a small
U-Net trained on raw (non-inpainted) fields scores each one for frontal-ness.
One command turns a raw SWOT Level-3 LR SSH Expert granule into every
closed-contour eddy in the pass, frontal ones flagged with a confidence score.

**`scripts/predict_hybrid.py` is the only supported inference entrypoint** — it
takes a raw NetCDF granule and the versioned weights, nothing else. Everything
under `scripts/experimental/` (regridded-zarr predictor, L2-zarr fetcher, L3-vs-L2
comparison) is a research spike on a different, unreleased data source, needs a
separate zarr>=3 environment, and is not meant to be served — see that
directory's own docstrings before touching it.

`docs/PIPELINE.md` is the full end-to-end record of how the project got from raw
granules to the trained model. This README is the short version and the setup
guide.

## How the detector was built

The model is trained on labels bootstrapped from a classical eddy detector, sorted
by a physical front prior:

1. **Data.** SWOT Level-3 LR SSH Expert v3.0 granules over the Gulf Stream — 50
   repeat cycles, 1,288 granules. Real passes are gappy (a blind nadir strip plus
   rain and quality gaps remove 34–70 % of the cropped swath).
2. **Bootstrap detection.** [py-eddy-tracker](https://github.com/AntSimi/py-eddy-tracker)
   is run on every pass following de Marez et al. (2026), producing 15,641
   candidate detections.
3. **Front prior.** A single iso-contour of absolute dynamic topography (ADT, at
   0.35 m) traces the Gulf Stream jet through each pass; frontal eddies hug that
   line on its inshore side. Signed distance to the front, validated against hand
   labels (AUC ≈ 0.92), turns labeling into a short front-sorted queue instead of
   an exhaustive walk.
4. **Labels.** 2,124 manual labels across all 50 cycles, of which 902 are confirmed
   frontal eddies (versioned in the repo).
5. **Model.** A ~1.9 M-parameter 3-level U-Net (`src/swot_eddy/unet.py`) trained on
   128×128 tiles of native SSHA, ADT, and |∇ADT| plus a validity channel — no
   inpainting. Split temporally, since eddies persist across adjacent cycles.
6. **Conformal calibration.** A reserved calibration split fixes a probability
   threshold with a distribution-free guarantee: a new frontal eddy is missed with
   probability ≤ 10 % (λ̂ = 0.348, 97.3 % test coverage).

The U-Net alone over-fires (test precision 0.41 at recall 0.96): it sees only
height fields, has no rotation cue, and so marks any cold-core pattern near the
front. The hybrid fixes this by requiring both signals — a closed SSH contour
(py-eddy-tracker) *and* high frontal probability inside it (U-Net, threshold
τ = 0.9). On the held-out test cycles (43–50): **precision 0.80, recall 0.96,
F1 0.875**, with an almost perfectly bimodal score separation. All metrics are
conditional on py-eddy-tracker's proposals; the open question of eddies PET
itself misses is documented in `docs/2026-07-20-recall-problem-writeup.md`,
written for external review.

## Setup

Verified on WSL / Linux with Python 3.12, no conda:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# py-eddy-tracker must come from GitHub master, not PyPI. The PyPI release (3.6.1)
# pins numpy<1.23 / numba<0.56 / netCDF4<1.6 and cannot install on Python >=3.12
# (it forces a source build of an ancient netCDF4 that needs HDF5).
python -m pip install "git+https://github.com/AntSimi/py-eddy-tracker.git" -e . pytest
```

This resolves to an all-wheel stack (numpy is auto-downgraded to satisfy numba):
`pyEddyTracker (master) · numpy 1.26.4 · numba 0.65.1 · netCDF4 1.7.4 ·
matplotlib 3.7.5 · xarray 2026.4.0 · scipy · zarr<3 · opencv · pint`.

On native Windows (Python 3.12) the commands are the same but with
`py -3.12 -m venv .venv` and `.venv\Scripts\activate`. Colab and conda users can
build the environment from `environment.yml`, which also installs py-eddy-tracker
from GitHub.

`config.yaml` auto-detects the platform (windows / wsl / colab) and resolves data
paths; override the detection with `SWOT_EDDY_ENV=wsl|windows|colab`. **Edit the
`paths:` block in `config.yaml`** so `data_root` points at your granule directory and
`out_root` at this repo's `out/` — the defaults are the author's machine.

## Get the data

The input product is **SWOT Level-3 LR SSH v3.0, *Expert* variant** (AVISO/CNES),
one NetCDF file per half-orbit pass. It is free but requires an
[AVISO+ account](https://www.aviso.altimetry.fr/en/data/data-access.html); once
registered, download it via the AVISO+ FTP or THREDDS server (SWOT L3 products,
`l3_lr_ssh`, *Expert* variant). Granule filenames look like

```
SWOT_L3_LR_SSH_Expert_002_007_20230811T072733_20230811T081900_v3.0.nc
```

(`002` = 21-day repeat cycle, `007` = pass number). A single granule is enough to run
the detector. The 27 passes crossing the study area (any cycle) are:

```
007 020 035 048 063 076 091 104 119 132 160 188 270 285
298 313 326 341 354 369 382 397 410 425 438 466 576
```

The batch scripts (`scripts/run_all.py`, `scripts/run_hybrid_all.py`)
expect granules laid out as `<data_root>/cycle_002/SWOT_L3_LR_SSH_Expert_*.nc`;
`scripts/predict_hybrid.py` takes any explicit path and ignores `data_root`.

## Verify the install

```bash
python -c "import py_eddy_tracker; print('PET OK')"          # imports cleanly
python -m swot_eddy.pet_version                                # pins pet_version into config.yaml
pytest -q                                                     # all tests green
# with at least one granule under data_root:
python -m swot_eddy.inspect_nc --out out/inspect_002_091.json # dumps a real granule
```

## Running the detector

Point the hybrid at any raw L3 Expert granule (the trained weights
`out/aoi5/model/unet_frontal_v2.pt` are versioned in the repo):

```bash
python scripts/predict_hybrid.py GRANULE.nc --report
```

This writes, next to the granule name under `out/aoi5/predictions/`:

- `*_hybrid.csv` — every PET eddy in the pass with its contour, radius,
  amplitude, `frontal_score`, and `is_frontal` (τ = 0.9; override with `--tau`)
- `*_hybrid.geojson` — the frontal eddies' true PET contours, for map overlays
- `*_hybrid.png` — annotated map: SSHA, ADT + front line, and the detections

Runs on CPU; a pass takes ~20 s (PET + U-Net). For a U-Net-only pass (no
py-eddy-tracker, no inpainting, ~15 s, conformal miss-rate guarantee but lower
precision) use `scripts/predict_frontal.py GRANULE.nc --v2 --conformal --report`.

## The labeled dataset

`out/aoi5/release/labeled_eddies_aoi5.csv` — the 2,217 hand-labeled eddies
(903 frontal / 1,045 reject / 168 open ocean / 95 coastal / 6 skip) joined to the
full py-eddy-tracker catalog: date, polarity, lon, lat, speed radius, effective
radius, **amplitude**, gap and nadir fractions, and the human label with its
timestamp. `labeled_eddies_contours.csv` carries the PET contour vertices
(lon/lat + pixel coords) for every labeled eddy. Both are regenerated from the
raw catalog + labels by `scripts/export_labeled_dataset.py`, which asserts a
1:1 join and exact lon/lat agreement. The unjoined sources are also versioned:
`out/aoi5/catalog/detections.csv` (all 15,641 PET detections) and
`out/aoi5/labels/eddy_labels_aoi5.csv`.

## Repository map

| Stage | Code | Output |
|---|---|---|
| Preprocess + PET | `src/swot_eddy/{preprocess,detect,aoi}.py`, `scripts/run_all.py` | `out/aoi5/catalog/detections.csv` |
| Labeling UI | `src/swot_eddy/review.py`, notebooks 02–03 | `out/aoi5/labels/eddy_labels_aoi5.csv` |
| Front prior | `src/swot_eddy/features.py`, `scripts/{validate_front,build_features,build_queue}.py` | `out/aoi5/features/` |
| Contours | `scripts/export_contours.py` | `out/aoi5/catalog/contours*.csv` |
| Training data | `scripts/build_masks.py` | `out/aoi5/training[_v2]/` |
| Model | `src/swot_eddy/unet.py`, notebooks 04–05 | `out/aoi5/model/unet_frontal[_v2].pt` |
| Inference (hybrid, supported) | `scripts/predict_hybrid.py` | `out/aoi5/predictions/*_hybrid.{csv,geojson,png}` |
| Inference (U-Net only) | `scripts/predict_frontal.py` | `out/aoi5/predictions/` |
| Experimental (L2-zarr, not served) | `scripts/experimental/{predict_zarr,predict_hybrid_gridded,fetch_zarr_window,make_compare_l3_l2z}.py` | `out/aoi5/predictions/`, `out/aoi5/l2zarr/` |
| Figures | `scripts/make_{paper_figs,report_hybrid,pass_gallery,stage_figs}.py` | `out/aoi5/{paper_figs,report_hybrid,pass_gallery,stage_figs}/` |

## Further reading

- `docs/PIPELINE.md` — the full end-to-end record, from raw granules to the trained model.
- `docs/2026-07-12-frontal-prior-methods.md` — the ADT front prior in detail.
- `docs/2026-07-20-recall-problem-writeup.md` — the open recall problem, written for external review.
