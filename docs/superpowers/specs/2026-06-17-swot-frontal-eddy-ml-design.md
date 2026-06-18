# SWOT Gulf Stream Eddy Detection — Baseline Design Spec

**Status:** Approved scope — simplified to a grounded baseline (supersedes the earlier
over-scoped draft).
**Repo:** private, local git, kept simple.

---

## 1. Goal

Prove a **simple, grounded baseline end-to-end** before adding anything else:

```
SWOT passes (region) → preprocess → PET → boxes → YOLO → detect on new pass → measure
```

Give a new SWOT pass, detect eddies in it, with calibrated uncertainty, and report how
eddy frequency varies by season. This is the *whole* first project. Everything fancier is
deferred (§11) until this baseline shows it's needed.

This is the proven combination: **de Marez** runs PET on native SWOT swaths (exact product
family, L3 Expert v3.0); **Bardi** trains YOLO on SWOT SSH tiles. Nothing here is invented.

**Region** (config): `lat 23..42 N, lon -82..-69 W`.
**Data:** 50 cycles, 1088 passes, ~11 GB, `SWOT_L3_LR_SSH Expert v3.0`.

---

## 2. Scope

**IN (baseline):** region crop · de Marez preprocessing · PET labeling · 2 input channels
(SSHA, ADT) · 3 classes · YOLOv8 · by-cycle train/cal/test splits · conformal-prediction
UQ · inference to a detection product · seasonal frequency analysis · honest metrics.

**DEFERRED (§11), only if the baseline justifies:** segmentation masks · self-supervised
/ JEPA · active-learning loop · napari labeling tool · extra channels (high-pass,
vorticity, Okubo-Weiss) · "beyond-PET" recall recovery.

---

## 3. Decisions

| Decision | Choice |
|---|---|
| First deliverable | Baseline PET→YOLO detector, measured end-to-end |
| Input channels | **2: SSHA (anomaly) + ADT (= SSHA + MDT)** — ADT shows the front |
| Output geometry | **Bounding boxes** (YOLO) |
| Classes | **3: anticyclonic eddy · cyclonic eddy · frontal cyclonic eddy** |
| "Frontal" assignment | **Provisional front-proximity tag** (§5) — coarse, tunable, revisable |
| Labeler | **PET** (de Marez params), per inpainted tile; polarity → class for free |
| Model | **YOLOv8** (Ultralytics), Colab GPU, checkpoint→Drive |
| Splits | **By cycle: train / calibration / test** (calibration needed for conformal) |
| Uncertainty | **Conformal prediction** on the calibration split (coverage guarantee) |
| Analysis | **Seasonal eddy-frequency** (counts per class, per season, coverage-normalized) |
| Validation | held-out **test cycles**: precision/recall/mAP per class |

---

## 4. Data + preprocessing (required for PET — not optional complexity)

A single shared preprocessing function, used identically by labeling, training, inference.
PET cannot run on a raw SWOT swath (nadir gap + NaNs break contour-finding), so the de
Marez steps are the price of using PET:

1. Read v3.0 (version-aware var map: `ssha_filtered`, `mdt`). **Never mutate source.**
2. Crop to region box.
3. Build channels: **SSHA** (`ssha_filtered`) and **ADT = SSHA + MDT**.
4. **NaN-pad** across-track 69 → 128.
5. **Tile** along-track **128×128, stride 64** (50% overlap).
6. **Biharmonic inpaint** each tile (`skimage.restoration.inpaint_biharmonic`); keep an
   `inpaint_mask` so inpainted pixels are traceable.

> MDT coverage: `mdt` has gaps in the raw product; verify region coverage in P1. If MDT is
> sparse, ADT falls back to SSHA + a smoothed/climatological MDT.

---

## 5. PET labeling → 3 classes

PET (Mason 2014, via `pyEddyTracker`) runs **per inpainted tile**, de Marez params:
`shape_error 85%`, `min_radius 5 km`, `min_amplitude 0.01 m`, gap-overlap reject >40%
(60% if center in nadir gap). Dedup across overlapping tiles + passes.

PET returns polarity directly:
- **anticyclonic eddy** ← PET anticyclonic.
- **cyclonic eddy** ← PET cyclonic, not flagged frontal.
- **frontal cyclonic eddy** ← PET cyclonic + frontal tag below.

**Frontal tag (provisional, light-touch):** geostrophic speed
`speed = √(ugos² + vgos²)` comes directly from the product's `ugos_filtered` /
`vgos_filtered` (no SSH differentiation). Flag a cyclonic eddy `frontal cyclonic` if its
center sits near the high-speed jet (within ~N km of the speed-max ridge). This is a
coarse provisional split for convenience, not a claim about frontal dynamics — `N` and the
speed threshold are tuned in P2, and the tag can be revisited or hand-corrected later.
Speed is used *only* for this tag, not as a model input channel.

PET boxes (axis-aligned bounding box of each eddy contour) + class → YOLO labels.

> PET is the baseline labeler. Its known low recall on front-attached eddies is *measured*
> by this baseline, not engineered around. If recall is poor, that motivates the deferred
> work (§11) — with evidence.

---

## 6. Label store + YOLO dataset

- **Master table** (parquet) = single source of truth: one row per detection — box (geo +
  pixel), class, polarity, PET attributes (radius, amplitude), front-distance, cycle,
  pass, temporality, provenance (`pet_version`, params, git SHA).
- YOLO `.txt` label files are **always regenerated** from the master table, never hand-
  edited.

---

## 7. Splits (by cycle)

Split **whole cycles** (never random windows) into **train / calibration / test**:
- **train** — fit YOLO.
- **calibration** — held out, used only for conformal prediction.
- **test** — frozen, used only for final metrics.

Record `temporality` (forward/reproc; handbook 5.8 calibration seam) per cycle; report
metrics per side rather than blindly mixing.

---

## 8. Model — YOLOv8

- Ultralytics YOLOv8 (start `m` or `l`), **2-channel input (SSHA, ADT)** — adapt the
  first conv stem to 2 channels (init from pretrained RGB weights by averaging/slicing).
- Train on Colab GPU; **checkpoint to Drive + resume** (Colab times out).
- Augment: rotations/flips (ocean has no canonical orientation).
- Raise YOLO `patience` well above default (Bardi: it early-stops ~epoch 30 otherwise).

---

## 9. Conformal prediction (UQ)

After training, use the **calibration split** to conformalize:
- per-class **confidence calibration** → a score threshold per class giving a guaranteed
  coverage level (e.g. 90%), or conformal **box margins** (prediction intervals on box
  corners). Method (e.g. conformal risk control on false-negative rate, or split-conformal
  on objectness) chosen in P5; the load-bearing requirement is the dedicated
  calibration split, which §7 provides.
- Output per detection: class, box, score, **and a calibrated confidence / coverage
  statement**.

---

## 10. Inference + outputs

New pass `.nc` → shared preprocess → tiles → YOLO → boxes → stitch to swath geometry
(merge overlaps, dedup across seams) → **new detection product** (source never mutated):
- detections table: center lat/lon, box, class, score, calibrated confidence;
- provenance attrs: model version, source `product_version` (3.0), `pet_version`,
  preprocess params, channels, git SHA, date.

---

## 11. Seasonal frequency analysis

Each pass is timestamped. Bin detections by season (and/or month), count per class,
**normalize by observational coverage** (number/area of passes per season) so frequency is
comparable. Answers: *how often do eddies (and frontal cyclonic eddies specifically) occur
by season?* Simple table + plot.

---

## 12. Evaluation

On frozen **test cycles**: per-class precision / recall / mAP@IoU; conformal coverage
check (does realized coverage match the target on test?); report per temporality side.

---

## 13. Cross-cutting rules (hygiene, cheap, kept)

1. **One** shared preprocess fn, pinned to `product_version`.
2. Master table = source of truth; YOLO labels always regenerated, never hand-edited.
3. Split by cycle; freeze test; never random-split windows.
4. reproc/forward seam awareness.
5. Never mutate source `.nc`; outputs are new files with provenance (model version, source
   `product_version`, params, git SHA).
6. `config.yaml` drives data/out roots (Windows ↔ Colab) + pins `pet_version`/params.
7. Private repo, kept simple — no speculative features.

---

## 14. Phasing

| Phase | Deliverable | Done when |
|---|---|---|
| **P0** | Repo + `config.yaml` + env (pin `pyEddyTracker`→`pet_version`); inspection script | PET imports, versions pinned, inspection prints for the real file |
| **P1** | Shared preprocessing (crop, SSHA+ADT, pad, tile, inpaint) | one pass → clean 2-channel tiles, geo round-trip verified; MDT coverage checked |
| **P2** | PET labeling + front heuristic → master table + YOLO labels | 3-class labels written with provenance; front-distance + N swept |
| **P3** | By-cycle train/cal/test splits | splits frozen, temporality-tagged |
| **P4** | YOLOv8 training on Colab | trains, checkpoints/resumes, detects on a tile |
| **P5** | Conformal prediction on calibration split | per-class calibrated thresholds / coverage |
| **P6** | Inference → detection product + provenance | new pass → product, source untouched |
| **P7** | Seasonal frequency analysis | per-class per-season frequency, coverage-normalized |
| **P8** | Evaluation on frozen test | precision/recall/mAP + conformal coverage, per temporality side |

Build phase by phase; stop at each "Done when" for confirmation.

---

## 15. Deferred (revisit only if baseline metrics justify)

Segmentation masks · multi-channel stack (high-pass SSHA, geostrophic speed, ζ/f,
Okubo-Weiss) · self-supervised / JEPA pretraining · human-in-the-loop active-learning loop
· napari labeling tool · "beyond-PET" recall recovery · SST/chlorophyll corroboration ·
eddy tracking across cycles.

---

## 16. References (what each grounded)

- **de Marez et al. (Ocean Science 2026; + subpolar brief comm.)** — swath-native SWOT
  preprocessing (NaN-pad 69→128, 128×128/64 tiling, biharmonic inpaint, gap reject 40/60%)
  and PET params (`shape_error 85%`, `min_radius 5 km`, `min_amp 0.01 m`).
- **Fu et al. (Comms Earth Env 2026)** — uses the exact product (L3 LR Expert v3.0); SWOT
  resolves cyclonic eddies down to ~2–25 km via closed-contour detection.
- **Bardi et al. (Sci. Data)** — YOLO on SWOT SSH tiles works; raise `patience`; (avoid its
  Fourier de-striping / grayscale histogram-eq — they destroy SSH magnitude).
- **Archer et al. (Nature 2025)** — fronts/frontal eddies show in ADT/gradient; don't
  blanket-drop `quality_flag` (deletes eddy cores); internal tides are the main false
  positive. *Informs the deferred multi-channel work if the baseline needs more recall.*
- **py-eddy-tracker docs** — closed-SSH-contour identification; identification module only
  is needed for labels.
