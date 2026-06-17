# SWOT Gulf Stream Frontal-Eddy ML Detection — Design Spec

**Date:** 2026-06-17
**Status:** Approved (brainstorming → ready for implementation planning)
**Repo:** private, local git, kept simple (no speculative features)

---

## 1. Goal

Automatically detect **frontal eddies** in SWOT L3 LR SSH Expert v3.0 passes over the
Gulf Stream. Given a new pass file, output where the frontal eddies are (per-pixel
footprint + a catalog of centers/attributes).

Frontal eddies here = cyclonic, cold-core, front-attached features (Gulf Stream
"shingles"), typically ~10–50 km, riding the shoreward edge of the front — strongest in
the South Atlantic Bight (23–35N) portion of the study box.

**Study region** (config): `lat_min=23, lat_max=42, lon_min=-82, lon_max=-69`.
**Data on hand:** 50 cycles, 1088 passes, ~11 GB, product `SWOT_L3_LR_SSH Expert v3.0`.

### Central finding that shapes the whole design
py-eddy-tracker (PET) and all closed-SSH-contour detectors (Mason 2014; Ni 2020 used by
Fu; de Marez) are **closed-contour, geostrophic, roundness-gated mesoscale detectors**.
Frontal eddies break all three assumptions: they are **front-attached** (often no closed
SSH contour), **elongated** (fail the roundness gate), and **high-Rossby-number, Ro~1**
(gradient-wind balanced, not geostrophic). Therefore **PET is high-precision but
low-recall for frontal eddies** — it finds the detached/classic ones and systematically
misses the front-attached ones, which are the actual target.

→ The project target is **"Beyond PET"**: PET seeds high-precision labels; a
human-in-the-loop active-learning loop expands recall to the eddies PET misses; the ML
model learns the front-attached signature from physics-informed channels (not raw SSHA).

---

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Project target | **Beyond PET** (PET = seed + benchmark, not predictor) | PET structurally misses front-attached eddies; active learning recovers them |
| Classes | **Cyclonic frontal eddy** + **ring/large eddy** (2 classes) | Ring labels near-free from PET; stops model conflating 120 km ring with 30 km frontal eddy |
| Output geometry | **Segmentation masks** (semantic + connected-components instances) | Faithful to comma-shaped eddies; converts directly from PET/traced polygons; *is* the `eddy_flag` field |
| Model strategy | **Staged: supervised baseline → JEPA**, compare | Baseline is certain + Colab-cheap + the proposer the loop needs; JEPA is a layered experiment given small data |
| Input channels | **Multi-channel, SSHA-primary**: `SSHA` (main), `ADT=SSHA+MDT`, `high-pass SSHA @50 km`, `geostrophic speed √(u²+v²)` | Raw SSHA hides front-attached eddies (Archer); derived channels free from `ugos/vgos`; vorticity/Okubo-Weiss optional sweep-in |
| Front removal | High-pass SSHA @50 km + naturally front-insensitive velocity channels | Gulf Stream front dominates raw SSHA; de Marez needed no detrend (flat Labrador Sea) — we do |
| Labeling tool | **napari** (Windows) | Native multi-channel float arrays, editable mask layers, clean GeoParquet round-trip |
| Label store | **Master GeoParquet** = single source of truth | Masks/tiles always regenerated; never hand-edited |
| Splits | **By cycle**, freeze test cycles, reproc/forward seam aware | Never random-split windows; respect handbook 5.8 calibration seam |
| Validation | **Manual hold-out only** (frozen test cycles): precision/recall/IoU; headline = **frontal-eddy recall vs PET** | SST/chlorophyll corroboration noted as optional future work, not core |

---

## 3. Architecture & data flow

A **single shared preprocessing core** feeds three consumers — PET label-generation,
model training, and inference — so they cannot drift. PET + napari corrections build the
master GeoParquet. A staged segmentation model (baseline → JEPA) learns the eddies PET
misses via the active-learning loop. Inference emits a provenance-stamped `eddy_flag`
NetCDF. **PET is never in the inference path.**

```
                    ┌─────────────────────────────────────────┐
 source .nc  ──────▶│  shared preprocess core (version-pinned) │
 (never mutated)    │  crop→QC→NaN-pad 69→128→tile 128×128/64   │
                    │  →biharmonic inpaint→channel stack        │
                    └───────────────┬───────────────┬──────────┘
                                    │               │
                         ┌──────────▼──────┐   ┌─────▼──────────┐
                         │ PET (retuned,   │   │ model train /  │
                         │ cyclonic+ring)  │   │ inference      │
                         └────────┬────────┘   └─────┬──────────┘
                                  │ seed masks        │ proposed masks
                                  ▼                   ▼
                    ┌─────────────────────────────────────────┐
                    │  napari: confirm / reject / edit          │◀── expert
                    └───────────────────┬───────────────────────┘
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │  MASTER GeoParquet  (single source of     │
                    │  truth: polygons, class, cycle, pass,     │
                    │  temporality, provenance)                 │
                    └───────────────────┬───────────────────────┘
                       split BY CYCLE   │  (rasterize on demand)
                    ┌────────┬──────────┼───────────┐
                    ▼        ▼          ▼           ▼
                  train     val    FROZEN test   active-learning
                                   (held out)    proposals → loop
```

### Repo skeleton

```
swot-eddy-ml/
  config.yaml              # data_root, out_root, region box, channels,
                           #   product_version map, pet_version, pet_params, tile params
  src/swot_eddy/
    config.py              # load config; Windows↔Colab path switch
    io.py                  # open .nc, version-aware variable-name map, never mutate
    preprocess.py          # THE shared core (crop, QC, pad, tile, inpaint, channels)
    channels.py            # SSHA / ADT / high-pass50 / geos-speed (+optional ζ, OW)
    label_pet.py           # PET identification, retuned; → candidate polygons
    labels_store.py        # master GeoParquet read/write; rasterize polygons↔masks
    napari_tool.py         # interactive confirm/reject/edit app
    splits.py              # by-cycle split, freeze test, reproc/forward aware
    datasets.py            # tile dataset from GeoParquet (regenerated each run)
    model_baseline.py      # U-Net / SegFormer seg head
    model_jepa.py          # I-JEPA-style SSL pretrain + seg fine-tune
    train.py / infer.py    # Colab-aware, checkpoint→Drive, resume
    active_loop.py         # propose→triage→napari→retrain orchestration
    eval.py                # precision/recall/IoU, recall-vs-PET on frozen test
    eddy_flag.py           # write provenance-stamped output NetCDF
  notebooks/               # Colab training/JEPA notebooks
  docs/superpowers/specs/  # this design
```

---

## 4. Components

### 4.1 Shared preprocessing core (the no-drift guarantee)
Used identically by labeling, training, inference. Pinned to `product_version`.
1. Read v3.0 via version-aware var map (`ssha_filtered`, `ugos/vgos_filtered`, `mdt`).
   **Never mutate source.**
2. Crop to region box.
3. **Feature-aware QC** — do *not* blanket-drop `quality_flag != 0` (deletes eddy cores,
   per Archer Fig 4). Mask only clearly-bad (land, hard nadir gap). Carry a
   `real_vs_inpainted` mask.
4. **NaN-pad** across-track 69 → 128.
5. **Tile** along-track **128×128, stride 64** (50% overlap) — de Marez.
6. **Biharmonic inpaint** each tile (`skimage.restoration.inpaint_biharmonic`) → gap-free
   tiles; keep the inpaint mask.
7. **Channels**: `SSHA` (main) · `ADT=SSHA+MDT` · `high-pass SSHA @50 km` (the *input*
   channel high-pass — fixed at 50 km) · `geostrophic speed √(u²+v²)`. Per-channel z-score
   with **train-only stats, pinned**. Optional sweep-in: `ζ/f`, Okubo-Weiss (from
   `ugos/vgos`).

   > Note: this 50 km **input-channel** high-pass is distinct from PET's own internal
   > **detection** high-pass (§4.2), which is a separate, sweepable filter applied inside
   > the labeler before contouring.

### 4.2 PET labeling (retuned for frontal eddies)
PET runs **per inpainted tile** (de Marez method, no separate regridding). Tuning moves
away from mesoscale defaults toward small/front-attached:
- high-pass **~50–100 km** (not 700) · `step ~0.002 m` (lower amplitude floor) ·
  `shape_error 85%` (permissive; keep) · `min_radius 5 km` · lower `pixel_limit` min.
- **gap-overlap reject** >40% (60% if center in nadir gap).
- **cyclonic → frontal-eddy candidate; large/round + anticyclonic → ring** (radius +
  amplitude split proposes class; expert confirms in napari).
- Dedup across overlapping tiles + passes (center sep < max(Rmax, threshold)).
- Output candidate polygons → GeoParquet, stamped `pet_version` + params + git SHA.

PET stays low-recall on frontal eddies by design; the active loop closes that gap.

### 4.3 Master GeoParquet (single source of truth)
One row per eddy instance: `geometry` (lon/lat polygon), `class`, `status`
(candidate/confirmed/rejected/edited), `source` (pet/model/manual), `cycle`, `pass`,
`temporality`, `confidence`, provenance. **All masks/tiles rasterized from this on
demand** — nothing downstream is hand-edited.

### 4.4 napari labeling tool (Windows)
Load multi-channel tile (SSHA default view) + candidate polygons as editable Labels layer
→ **confirm / reject / edit / add** → write status back to GeoParquet. Supports
Bardi-style confidence triage on model proposals: auto-accept >0.9, review 0.7–0.9, drop
<0.7 (tunable).

### 4.5 Splits
**By cycle, never random window.** Freeze whole **test cycles**. Record `temporality`
(forward/reproc; handbook 5.8 seam) per file; report metrics per side rather than mixing.

### 4.6 Baseline segmentation model (the proposer)
- **U-Net or SegFormer-B0** (small, Colab-cheap). In = 4-channel tile. Out = **semantic
  seg** (background / frontal-eddy / ring); instances via **connected components**
  (no Mask R-CNN).
- Loss = class-weighted Dice + cross-entropy; **mask out inpainted/gap pixels** via the
  inpaint mask.
- Augment = rotations/flips (ocean orientation-free).
- Colab T4, ~15–60 min, **checkpoint→Drive + resume**.

### 4.7 Active-learning loop ("Beyond PET")
```
baseline → propose masks+confidence on un-reviewed tiles
        → triage: >0.9 auto-accept · 0.7–0.9 napari review · <0.7 drop
        → confirmed/edited → GeoParquet → retrain → repeat
```
Focus on the frontal-eddy class (rings already covered by PET). Stop when newly-confirmed
plateaus (Bardi converged in ~4 rounds).

### 4.8 JEPA (layered experiment)
- **I-JEPA-style**: ViT-Small encoder, latent mask-and-predict, **self-supervised on all
  ~20k tiles** (no labels), heavy rot/flip aug, checkpoint→Drive.
- Attach the **same segmentation head**, fine-tune on expanded labels.
- **Head-to-head vs baseline** on frozen test (recall/IoU). If SSL doesn't win on ~20k
  tiles, that is a documented result — baseline still ships. JEPA never blocks a working
  detector.
- **Risk:** ~20k tiles is small for SSL from scratch (SSL favors 10^5–10^6). GPU *hours*
  are not the blocker (feasible on Colab Pro w/ checkpointing); data *size* is. Hence
  staged.

### 4.9 Inference
New pass `.nc` → shared preprocess core → tiles → model → per-tile masks → **stitch back
to swath geometry** (merge overlaps, dedup instances across seams). Output = **new
`eddy_flag` NetCDF** (source never mutated):
- per-pixel class field (background / frontal-eddy / ring),
- **centers + attributes** table (center lat/lon, equiv-radius, class, confidence),
- **provenance attrs**: model version, source `product_version` (3.0), `pet_version`,
  preprocess params, channel set, git SHA, date.

Resolves the plan's `[DECIDE]` (footprint vs center): **emit both** — footprint mask as
the field, centers as the catalog.

### 4.10 Evaluation
On frozen test cycles: per-class precision/recall/IoU; instance detection matched by IoU
threshold; **headline = frontal-eddy recall vs PET**; reported per temporality side.

---

## 5. Phasing (each → its own implementation plan)

| Phase | Deliverable | Done when |
|---|---|---|
| **P0** | Repo + `config.yaml` + env (pin `pyEddyTracker`→`pet_version`); inspection script | PET imports cleanly, versions pinned, inspection prints for the real file |
| **P1** | Shared preprocessing core + channels + tiling/inpaint + QC | one pass → clean multi-channel tiles, geo round-trip verified |
| **P2** | PET labeling (retuned) → master GeoParquet + ring labels | candidate polygons written with provenance for a set of cycles |
| **P3** | napari tool + GeoParquet round-trip + by-cycle split (freeze test) | can confirm/reject/edit; splits frozen and temporality-tagged |
| **P4** | Supervised baseline segmentation on Colab | trains + checkpoints/resumes; produces proposals with confidence |
| **P5** | Human-in-the-loop active-learning loop | ≥1 full round expands confirmed frontal-eddy labels |
| **P6** | JEPA SSL pretrain → fine-tune → compare | head-to-head metrics vs baseline on frozen test |
| **P7** | Inference → `eddy_flag` NetCDF with provenance | new pass → output product, source untouched |
| **P8** | Evaluation on frozen hold-out | precision/recall/IoU + recall-vs-PET, per temporality side |

Build **phase by phase**; stop at each "Done when" for confirmation before continuing.

---

## 6. Cross-cutting rules (pinned everywhere)
1. **One** shared preprocess fn, pinned to `product_version`.
2. Master GeoParquet = source of truth; masks/tiles always regenerated, never hand-edited.
3. Split by cycle; freeze test; never random-split windows.
4. reproc/forward seam (handbook 5.8) awareness throughout.
5. NaN pad · tile 128×128/64 · biharmonic inpaint · PET retuned (high-pass ~50–100 km,
   `step ~0.002 m`, `shape_error 85%`, `min_radius 5 km`, gap-reject 40/60%).
6. Never mutate source `.nc`; `eddy_flag` is a new NetCDF with provenance (model version,
   source `product_version`, params, git SHA).
7. `config.yaml` drives data/out roots (Windows ↔ Colab) + pins `pet_version`/params.
8. Private repo, kept simple.

---

## 7. Sweepable / open parameters (decide empirically, log every choice)
- Input-channel high-pass = 50 km (fixed, user decision). PET *detection* high-pass = separate, sweep 50–150 km.
- PET `step`, `min_radius`, `min_amplitude`, `pixel_limit` for frontal-eddy recall.
- Tile size (128 committed for PET-compat + front context; consider 64 for small eddies).
- Channel set: add `ζ/f` / Okubo-Weiss if they improve recall.
- Active-loop confidence thresholds (0.7 / 0.9 starting point).
- frontal-vs-ring split heuristic (radius/amplitude/location).
- `internal_tide` subtraction (available in product) to suppress internal-wave false eddies.

---

## 8. Key references (and what each contributed)
- **de Marez et al. (high-latitude eddy statistics, Ocean Science 2026)** — gold-standard
  swath-native SWOT preprocessing: NaN-pad 69→128, 128×128/64 tiling, biharmonic inpaint,
  gap-overlap reject 40/60%, PET `shape_error 85% / min_radius 5 km / min_amp 0.01 m`.
  Caveat: flat Labrador background → no front removal (we must add it).
- **de Marez et al. (subpolar hotspots, brief comm.)** — same pipeline applied; native
  swath preserves sub-50 km eddies.
- **Fu et al. (Arctic small mesoscale, Comms Earth Env 2026)** — uses the exact product
  (**L3 LR Expert v3.0**); SWOT resolves cyclonic eddies down to ~2–25 km; closed-contour
  detector; weak-gradient regime (no front removal needed).
- **Archer et al. (global submesoscale, Nature 2025)** — frontal eddies are visible in
  SWOT but in **gradient-magnitude / vorticity**, not raw SSHA; Kuroshio analogue to Gulf
  Stream; high-Ro → gradient-wind not geostrophic; quality_flag deletes eddy cores;
  internal tides/solitons are the dominant false-positive contaminant; ~8.5 km velocity
  noise floor.
- **Bardi et al. (solitons via ML, Sci. Data)** — the transferable workflow: SSH→tiles→
  detector→**confidence-triaged human-in-the-loop incremental retraining** (the cure for
  PET's low recall). Avoid: Fourier de-striping, grayscale histogram-eq, mean-fill — they
  destroy the SSH magnitude/gradient signal eddies need.
- **py-eddy-tracker docs** — closed-SSH-contour identification (Mason 2014); needs
  regular grid or per-tile run; identification vs tracking separate; only identification
  needed for labels; retune high-pass/step/shape_error for small features.

---

## 9. Risks
- **Small data for SSL** (~20k tiles) → JEPA payoff uncertain; mitigated by staging
  (baseline ships regardless).
- **PET recall on frontal eddies may be very low** → active loop + expert time is the
  load-bearing effort; budget for it.
- **Internal tides / solitons** masquerade as eddies → subtract `internal_tide`, use
  vorticity/Ro and shape as discriminators.
- **Calibration seam (reproc/forward)** could bias train/test → split- and report-aware.
- **Front removal tuning** (50 km high-pass) may clip larger frontal eddies → sweep.
