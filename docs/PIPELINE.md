# From Raw SWOT Data to a Frontal-Eddy Detector

End-to-end record of how this project went from raw SWOT altimetry granules to a trained
U-Net that takes any SWOT pass and returns frontal-eddy detections with coordinates and a
calibrated confidence guarantee. Written 2026-07-17, at model v2.

**One-line summary:** classical eddy detection (py-eddy-tracker) plus a physical front
prior (one ADT contour) bootstrapped a labeled dataset of ~900 confirmed Gulf Stream
frontal eddies; a small U-Net trained on those labels now detects frontal eddies directly
from raw passes — no eddy tracker, no inpainting, ~15 s per pass on a CPU.

---

## 1. The data

- **Product:** SWOT Level-3 LR SSH *Expert* v3.0 granules (AVISO/CNES), one file per
  half-orbit pass. 50 repeat cycles (21-day), 1,288 granules, ~13 GB, covering the Gulf
  Stream study region.
- **Swath geometry:** 69 across-track pixels at ~2 km spacing (~120 km wide), with a
  permanently blind nadir strip in the middle. Real passes are heavily gappy — 34–70 % of
  the cropped swath can be missing (nadir, rain, quality editing).
- **Variables used:** `ssha_filtered` (denoised sea-level anomaly — the detection field),
  `ssha_unedited + mdt` = **ADT** (absolute dynamic topography — carries the Gulf Stream
  itself), `mdt`, and geostrophic velocities (visual vetting only, so far).
- **No quality-flag masking** (`quality_max: null`): faithful to de Marez et al. (2026) —
  fields are used as the product ships them; only the product's own NaN gaps are treated
  as missing.
- **Study area:** the `GS_AOI5` polygon (concave; keeps the Gulf Stream corridor and
  shelf, clips the southeast open ocean). Fields are cropped to the polygon's rectangular
  envelope; only detection *centers* are filtered by true point-in-polygon, so no fake
  data edges are introduced before interpolation.

## 2. Bootstrap detection: py-eddy-tracker on every pass

Goal at this stage: get a large set of *candidate* eddies cheaply, knowing the method
under-detects. Recipe follows de Marez et al. (2026) exactly (`src/swot_eddy/preprocess.py`,
`src/swot_eddy/detect.py`):

1. Crop pass to the AOI envelope (contiguous along-track band, full 69-px width).
2. Pad across-track 69 → 128 pixels (NaN wings, centered).
3. Cut 128×128 tiles along-track with 64-px overlap (every eddy fully inside ≥1 tile).
4. **Biharmonic inpainting** (`∇⁴u = 0` over each gap, matching value and slope at gap
   boundaries) — PET requires closed contours, so gaps must be filled.
5. PET `eddy_identification` per tile on SSHA: closed contours around extrema, shape
   error ≤ 85 %, amplitude ≥ 1 cm, speed-radius ≥ 5 km, 4–1000 px area. Polarity from
   SLA sign.
6. Rejection of untrustworthy detections: speed contour > 40 % on inpainted pixels
   (60 % if centered in nadir), or ≥ 70 % of area inside the nadir strip (biharmonic
   invents smooth blobs there). Duplicates across tile overlaps deduplicated (10 km).

Run over all 50 cycles (`scripts/run_all.py`): **15,641 detections**
(`out/aoi5/catalog/detections.csv`; kept + auto-rejected kinds, the latter as ML
negatives). Detection order per (cycle, pass, kind) is deterministic, which makes
`(cycle, pass, kind, row-index)` a stable eddy key — everything downstream joins on it.

## 3. Manual labeling, phase 1: unbiased walk

A Jupyter labeling UI (`src/swot_eddy/review.py`, notebook 02) shows each detection in
full context (SSHA/ADT zooms, geostrophic velocity quivers, geographic panels with
coastline) with 4 classes: **frontal / coastal / open ocean / reject**. Labels go to a
resumable, atomically-written CSV keyed on the eddy key.

Cycles 1–7 were labeled exhaustively (~800 labels). Findings: 61 % of PET-kept
detections are junk (`reject`), and rejects are *not* separable from real eddies by any
scalar flag (gap fraction, amplitude, radius) — junk vetting genuinely needs eyes.
At ~115 labels/cycle, labeling all 50 cycles would have taken 24+ hours — motivating the
front prior.

## 4. The ADT front prior (the key accelerator)

Physical idea (user's): the Gulf Stream separates low ADT (shelf/slope side) from high
ADT (Sargasso side), so **a single iso-contour of native ADT traces the jet** through
every pass, and frontal eddies hug that line on its inshore side.

Implementation (`src/swot_eddy/features.py`):
- Front = *open* contours of native (non-inpainted) ADT at a fixed level, extracted with
  `skimage.find_contours` masked to finite data (contours break naturally at the nadir
  gap and swath edges). Closed rings at that level are eddies themselves and are dropped
  (audited: including them barely changed the result, 0.956 → 0.948 AUC, but exclusion is
  the clean definition).
- Per detection: signed distance to the nearest front vertex; sign from
  `sign(ADT(center) − level)` (− inshore, + offshore) — immune to contour fragmentation.

**Validation against the hand labels** (`scripts/validate_front.py`): level swept
0.20–0.60 m; AUC (|distance| ranking frontal vs open-ocean) plateaus at 0.30–0.35 m;
**0.35 m chosen** (0 % missing-front rate). Frontal eddies: median 22 km from the line,
75 % within 50 km. Open-ocean: median ~216 km. A gradient-ridge front (max |∇ADT|) was
tested and lost decisively (AUC 0.82 vs 0.92) — eddy rims are themselves strong
gradients, so the ridge fires everywhere; the level-line is the robust definition.

## 5. Manual labeling, phase 2: front-sorted queue

With the prior validated, labeling switched to a queue (`scripts/build_queue.py`,
notebook 03): every unlabeled **cyclonic** detection within **50 km** of the front —
1,409 eddies, batched by whole passes (one ~15 s pass load serves many eddies; the
labeler also prefetches the next pass in a background thread). Rule-predicted frontal
fraction 53 %; realized **57.9 %** — the prior validated out-of-sample.

Final label set: **2,124 labels; 902 confirmed frontal eddies across all 50 cycles**
(`out/aoi5/labels/eddy_labels_aoi5.csv`, versioned in git). Two documented sampling
regimes: cycles 1–7 = unbiased full walk; the rest = curated near-front cyclonic (a
provenance column is planned for the public release; a 200-eddy random audit queue
exists for unbiased class rates and is partially labeled).

Every detection's PET speed-contour polygon was also re-derived deterministically and
saved (`scripts/export_contours.py` → `out/aoi5/catalog/contours.csv`) — the catalog CSV
had only scalars; the polygons become segmentation targets.

## 6. Training data for the U-Net

`scripts/build_masks.py` builds one sample per labeled pass (884 passes):

- **Targets:** binary mask = union of the PET contour polygons of hand-*confirmed*
  frontal eddies. (Labels are a bootstrap, not ground truth: PET misses real frontal
  eddies, which therefore sit in the background class — a documented limitation.)
- **Inputs, v1:** the inpainted 3-channel stack (SSHA, ADT, |∇ADT|) — identical to what
  PET saw.
- **Inputs, v2 (`--raw`):** native fields, *no inpainting* — SSHA, ADT, |∇ADT| with NaN
  gaps kept, plus a **validity channel** (1 = real data). Inpainting is a PET
  requirement, not a U-Net one; it was also found to *explode* inside large gaps
  (ADT reaching +120 m), which had to be clipped in v1.
- **Split (temporal — eddies persist across adjacent 21-day cycles, so no random
  splits):** train = cycles 1–30, val = 31–35, **cal = 36–42** (reserved for conformal
  calibration, untouched by training or model selection), test = 43–50.

## 7. The model

`src/swot_eddy/unet.py`: a small 3-level U-Net (~1.9 M parameters), 128×128 tiles,
dice + BCE loss, positive-tile rebalancing (frontal tiles are ~1 in 8), flip
augmentation, per-channel standardization with physical clip bounds frozen from the
training distribution. Trained ~30 epochs on Colab (L4 GPU), best-val-IoU checkpoint
kept (`notebooks/05_train_unet_v2_colab.ipynb`, self-contained).

**Conformal calibration:** on the cal split, each true frontal eddy gets a score
(max predicted probability inside its mask); the split-conformal α-quantile gives a
threshold λ̂ with a distribution-free guarantee: *a new frontal eddy is missed with
probability ≤ 10 %*. v1's λ̂ was degenerate (~0) because inpaint artifacts blinded the
model to ~10 % of eddies; with raw inputs **λ̂ = 0.348** and test coverage is 97.3 %.

## 8. Results (test cycles 43–50, never seen in training)

| Eddy-level, threshold 0.5 | v1 (inpainted) | v2 (raw) |
|---|---|---|
| Precision | 0.53 | 0.41 |
| Recall | 0.90 | **0.96** |
| F1 | 0.67 | 0.58 |
| TP / FP / FN | 133 / 117 / 15 | 142 / 201 / 6 |
| Conformal λ̂ | degenerate | **0.348 (97 % coverage)** |

Honest reading: metrics are measured against bootstrap labels. Manual review of the v2
extras found the *vast majority are not real frontal eddies* — so precision is honest,
recall is a lower-bound statement about labeled eddies only. Failure-mode analysis
(saliency + case anatomy): the model fires on any cold-core-near-front pattern because
it has **no rotation cue** — humans used velocity quivers, PET used closed contours, the
net sees only height fields. Thresholding cannot fix it (169/201 extras at p > 0.9).

## 9. The tool

```
python scripts/predict_frontal.py GRANULE.nc --v2 --conformal --report
```

Any raw L3 Expert granule → annotated geographic map (SSHA + numbered eddies, ADT +
front line, probability) and a per-eddy CSV (id, lon, lat, confidence, radius, area).
No PET, no inpainting, ~15 s per pass on CPU. Publication figures:
`scripts/make_paper_figs.py` → `out/aoi5/paper_figs/fig1–fig6` (PNG + PDF); visual
reports: `scripts/make_report_figs.py`, `scripts/make_report_v2.py`.

## 10. Known limitations and next steps

1. **No rotation information** → planned v3: add native `ugos/vgos` (or vorticity)
   channels; expected to close most of the precision gap.
2. **No hard negatives in training** → v3: the 969 hand-rejected detections and the
   reviewed false positives become explicitly weighted negatives.
3. **Bootstrap-label ceiling:** true recall against *all* frontal eddies is unmeasurable
   until ~20 exhaustively annotated gold passes exist.
4. **Sampling provenance** column needed before the dataset release (two labeling
   regimes); random audit queue to be finished for unbiased class rates.
5. Longer term: cross-pass eddy tracking (lifetimes, trajectories); transfer to other
   western-boundary currents — nothing in the method is Gulf-Stream-specific except the
   front level, which is recalibrated per region by the same sweep.

## Repository map

| Stage | Code | Output |
|---|---|---|
| Preprocess + PET | `src/swot_eddy/{preprocess,detect,aoi}.py`, `scripts/run_all.py` | `out/aoi5/catalog/detections.csv` |
| Labeling UI | `src/swot_eddy/review.py`, notebooks 02–03 | `out/aoi5/labels/eddy_labels_aoi5.csv` |
| Front prior | `src/swot_eddy/features.py`, `scripts/{validate_front,build_features,build_queue}.py` | `out/aoi5/features/` |
| Contours | `scripts/export_contours.py` | `out/aoi5/catalog/contours*.csv` |
| Training data | `scripts/build_masks.py` | `out/aoi5/training[_v2]/` |
| Model | `src/swot_eddy/unet.py`, notebooks 04–05 | `out/aoi5/model/unet_frontal[_v2].pt` |
| Inference | `scripts/predict_frontal.py` | `out/aoi5/predictions/` |
| Figures | `scripts/make_{paper_figs,report_figs,report_v2}.py` | `out/aoi5/{paper_figs,report,report_v2}/` |

Method details for the front prior: `docs/2026-07-12-frontal-prior-methods.md`.
