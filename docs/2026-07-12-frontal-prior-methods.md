# Frontal-eddy prior: PET detections + per-swath ADT front distance

**Status:** validated 2026-07-12 against 822 hand labels. Code: `src/swot_eddy/features.py`,
`scripts/validate_front.py`, `scripts/build_features.py`, `scripts/build_queue.py`.

## 1. What this is

A physically-grounded prior that ranks py-eddy-tracker (PET) detections by probability of
being a **Gulf Stream frontal eddy**, using only data inside each SWOT L3 Expert pass — no
external products. It is used to (a) pre-sort the manual labeling queue and (b) serve as a
classification baseline. It does not fix PET's recall (eddies PET never detects stay
invisible to it).

Pipeline context: SWOT L3 pass → de Marez et al. (2026) preprocessing (crop → 69→128 pad →
128×128/64 tiles → biharmonic inpaint) → PET eddy identification (shape_error 85, min R
5 km, min amp 0.01 m, gap/nadir rejection) → AOI5 polygon clip → **this prior** per
detection.

## 2. Front extraction (per pass)

Done independently for every pass, on the **native** (non-inpainted, non-tiled) fields of
that granule (`features.native_fields`):

- ADT = `ssha_unedited + mdt` (same definition as the viewing channel; NaN where MDT is
  missing). Never inpainted — gaps stay gaps so the front cannot be fabricated.
- The Gulf Stream separates low ADT (shelf/slope water) from high ADT (Sargasso side), so
  one mid-level iso-contour of ADT traces the jet's dividing line through the swath.
- Extraction = `skimage.measure.find_contours(adt, level, mask=isfinite(adt))`
  (`features.front_vertices`). The mask makes contours terminate at the nadir gap and
  swath edges instead of wrapping around them.
- **Only OPEN contours are kept** (first vertex ≠ last). A closed ring at the front level
  is an eddy's own ADT anomaly; keeping it would put that eddy at distance ~0 from "the
  front" (self-labeling). The jet line always terminates on a data edge, i.e. is open.
- Fragments shorter than 20 vertices (~40 km at the 2 km pixel) are dropped as noise.
- Contour vertices are mapped to lon/lat by nearest pixel (≤2 km quantization, negligible
  against the 50 km decision scale).
- A pass where no contour survives gets `front_adt_missing = 1` (swath missed the jet);
  distances are NaN there.

The same extraction at the same level on the **MDT** field gives a mean-state front
(`dist_front_mdt_km`) as a backup feature; it validates worse than the instantaneous ADT
front (Sec. 4) and is not used by the queue.

## 3. Signed distance per detection

For each catalog detection (center lon/lat from PET):

- **Magnitude** = distance to the nearest front vertex, in km, on a local equirectangular
  projection about the front's mean latitude (`features.signed_front_dist_km`, cKDTree).
- **Sign** = `sign(ADT_at_center − level)`: **+ offshore/Sargasso side, − inshore/coastal
  side**. Sign from the field value, not contour geometry, so it is immune to contour
  fragmentation and orientation.
- ADT at center is sampled at the nearest pixel, falling back to the nearest *finite*
  pixel within 10 px for centers in the nadir gap (`center_sample_dist_px` records the
  fallback distance for auditability).

Features are built for all 15,641 catalog detections by `scripts/build_features.py`
(resumable; output `out/aoi5/features/features.csv`, keyed `(cycle, pass, kind, eddy)` —
the key is the row order of `detections.csv` via `groupby(...).cumcount()`, verified to
match the hand-label keying 805/805 with exact lon/lat agreement).

## 4. Validation against hand labels

`scripts/validate_front.py`; labels = `out/aoi5/labels/eddy_labels_aoi5.csv`, cycles 1–7,
4 classes (frontal / coastal / open ocean / reject), labeled blind to this prior.

**Metrics** (class-labeled eddies only, i.e. frontal/coastal/open-ocean; rejects handled
separately in Sec. 5):

- `auc_frontal_vs_open` — Mann-Whitney AUC of **−|distance|** ranking frontal above
  open-ocean labels (rank-based with average ties; NaN-distance rows excluded). 1.0 =
  |distance| alone separates the classes perfectly.
- medians of |distance| per class; fraction of each class within 50 km; missing-front rate.

**Sweep** over levels 0.20–0.60 m (open-contours-only, 822 labels):

| level (m) | AUC frontal-vs-open | frontal med \|d\| | open med \|d\| | missing |
|---|---|---|---|---|
| 0.25 | 0.929 | 31 km | 225 km | 2.5% |
| **0.30** | **0.948** | 22 km | 218 km | 1.2% |
| **0.35** | **0.952** | 22 km | 216 km | 0.0% |
| 0.40 | 0.907 | 24 km | 197 km | 0.0% |
| 0.50 | 0.869 | 31 km | 134 km | 0.0% |

AUC plateaus at 0.30–0.35; **0.35 m chosen** (config `features.front_level_m`) for the 0%
missing-front rate. MDT front peaks at AUC 0.890 (level 0.20) — strictly worse, kept only
as a feature. Signed histograms (`out/aoi5/features/front_validation.png`): frontal labels
pile up just **inshore (−)** of the line, open ocean far offshore (+), coastal far inshore.

**Robustness audit** (2026-07-12): closed eddy-rings were 111/625 contours on the labeled
passes; including them changes AUC only 0.956 → 0.948, so the headline result is not a
self-labeling artifact. They are excluded anyway (Sec. 2).

## 5. The rule baseline and its honest limits

Two-term rule **cyclonic ∧ |dist_front_adt| < 50 km**, evaluated on all 802 usable labels:

| population | precision (frontal) | recall (frontal) |
|---|---|---|
| real eddies only (rejects excluded) | 0.84 | 0.72 |
| everything PET kept (rejects included) | 0.53 | 0.72 |

Known limits, in order of impact:

1. **PET junk.** 61% of hand labels on PET-kept detections are `reject`, they do NOT
   separate on gap_frac / nadir_frac / amplitude thresholds, and ~40% of them sit within
   50 km of the front. This is what drops end-to-end precision to 0.53 and why the queue
   is manually approved.
2. **Recall ceiling ~72–75%.** ~28% of labeled frontals sit >50 km from the contour
   (detached / meander cases). Widening the window trades precision.
3. **PET's own misses** are invisible to any catalog-level metric; measuring them needs
   exhaustively annotated gold passes (planned, not yet built).
4. Frontal labels are 89% cyclonic (114/128 at the 805-label snapshot); the cyclonic term
   assumes the 14 anticyclonic frontal labels are edge cases — pending user review.
5. Labels cover cycles 1–7 only (~one season). Seasonal generalization unmeasured until
   queue labels (all 50 cycles) accumulate.

## 6. Labeling queue

`scripts/build_queue.py --size 0 --cyclonic`: every unlabeled kind=keep detection matching
the rule (1,409 at build time), grouped by whole passes densest-first (one ~15 s pass load
in the labeler serves many eddies), labeled in `notebooks/03_frontal_queue_aoi5.ipynb` via
`CatalogLabeler(cfg, queue=...)`. Expected composition from the labeled sample: ~53%
frontal, ~38% junk, remainder coastal/open-ocean. Queue clicks feed the same labels CSV
and double as validation data for the next iteration of this prior.

## 7. Contour export

`scripts/export_contours.py` re-runs detection deterministically (same eddy keys) and
persists each detection's PET speed-contour polygon (`out/aoi5/catalog/contours.csv`, one
row per vertex, pixel + lon/lat; `contours_geom.csv` per-eddy center pixel + vertex count +
catalog-match flag). Detections themselves are unchanged — this only saves shapes the
catalog CSV had discarded, for downstream use (segmentation masks, shape features, SSL
targets).
