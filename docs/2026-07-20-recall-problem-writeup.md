# Detecting Gulf Stream Frontal Eddies in SWOT Altimetry — Project Briefing and the Open Recall Problem

*Prepared 2026-07-20 for external review. Self-contained: no prior knowledge of the
project assumed. The specific question we want help with is in Section 8.*

---

## 1. Executive summary

We built a detector that takes a raw SWOT wide-swath altimetry pass over the Gulf
Stream and returns the frontal eddies in it. It works in three stages, developed in
sequence: (1) a classical eddy detector (py-eddy-tracker, "PET") plus a one-parameter
physical front prior bootstrapped a hand-verified label set of ~900 frontal eddies at
~10× the labeling speed of exhaustive review; (2) a small U-Net trained on those labels
detects frontal eddies directly from raw fields; (3) a hybrid — PET proposes eddies, the
U-Net classifies frontal-ness — achieves precision 0.80 / recall 0.96 / F1 0.875 on
held-out cycles.

**The open problem: every label, and therefore every metric, is conditioned on PET
having detected the eddy in the first place.** PET is known to under-detect in this
region. Our recall numbers are recall *given PET support*; recall against the true
population of frontal eddies is currently unidentifiable from our data. Section 8
states the open questions precisely, Section 9 lists the assets and evidence in hand,
and Section 10 lists practical constraints. We are deliberately not prescribing a
direction — we want your read on how you would attack this, including whether the
current architecture is the right one at all.

## 2. Scientific goal

Gulf Stream frontal eddies — cyclonic, cold-core eddies of roughly 5–40 km radius that
form and propagate along the shoreward edge of the jet — matter for cross-shelf
exchange, nutrient supply, and larval transport. They are small enough that
conventional gridded altimetry mostly cannot resolve them. SWOT's 2 km-resolution,
120 km-wide swath can. The end goal is an automated census: give the system any SWOT
pass, get every frontal eddy with position, size, and confidence.

## 3. Data

- **Product:** SWOT Level-3 LR SSH *Expert* v3.0 (AVISO/CNES), one netCDF granule per
  half-orbit pass. 50 repeat cycles (21-day repeat), 1,288 granules over the study
  region (Florida Straits to ~40°N; the `GS_AOI5` polygon).
- **Swath geometry:** 69 across-track pixels at ~2 km (~120 km wide), with a permanently
  blind nadir strip down the middle. Passes are gappy: 34–70% of a cropped swath can be
  missing (nadir, rain, editing).
- **Fields used:** `ssha_filtered` (denoised sea-level anomaly; detection field);
  ADT = `ssha_unedited + mdt` (absolute dynamic topography; carries the Gulf Stream
  front); geostrophic velocities (so far only for human vetting, not the model).
- No quality-flag masking beyond the product's own editing (faithful to the reference
  method below).

## 4. Stage 1 — bootstrap labeling

### 4.1 PET on every pass — exact configuration

py-eddy-tracker (git master, `0+untagged.754.g229543f`; PyPI release is uninstallable
on modern Python) is used purely as a **per-pass detector** — no tracking. The recipe
follows de Marez et al. (2026) Sect. 3.1, with each step spelled out because it defines
the support of all downstream labels:

1. **Field:** detection runs on `ssha_filtered` exactly as the L3 product ships it —
   no extra quality-flag masking, no high-pass/detrending (deviations we tried and
   reverted for fidelity).
2. **Geometry:** crop the pass to the study region's bounding band (full 69-px width);
   NaN-pad across-track 69 → 128 (centered); cut 128×128 along-track tiles with 64-px
   overlap so every eddy is fully interior to at least one tile.
3. **Inpainting:** biharmonic fill (`skimage inpaint_biharmonic`) of every NaN gap per
   tile — PET cannot close contours across gaps. Fill locations are remembered.
4. **Identification:** each tile becomes a local equirectangular `RegularGridDataset`
   (~2 km spacing; forced units m and m/s); `eddy_identification` scans SSH levels for
   closed contours around extrema with: shape error ≤ **85%**, amplitude ≥ **0.01 m**,
   speed radius R_max ≥ **5 km**, contour area **4–1000 pixels** (PET-standard; caps
   effective radius near ~36 km on this grid). Polarity from SLA sign. Near-constant
   tiles are skipped (PET contour crash guard).
5. **Trust rejection (drops, not flags):** a detection is discarded when > **40%** of
   its speed-contour-enclosed area lies on inpainted pixels (relaxed to **60%** when the
   center sits in the nadir band), or ≥ **70%** of its area lies inside the nadir strip
   (biharmonic invents smooth blobs there). These rules protect precision and directly
   cost recall near gaps — failure mode 2 of Section 8.
6. **Dedup:** detections from overlapping tiles within 10 km keep one copy; centers are
   then filtered to the study polygon (point-in-polygon; pixels are never masked before
   inpainting, to avoid fabricated edges).

Result: **15,641 detections over 50 cycles** (kept + auto-rejected kinds, the latter
retained as ML negatives), with deterministic per-pass ordering that serves as a stable
join key for everything below. In the final hybrid detector (Section 6) this same
pipeline runs again at inference time as the proposal stage.

### 4.2 Hand labeling with a physical accelerator

A Jupyter reviewer shows each detection in context (SSHA/ADT zooms, geostrophic
velocity quivers — the human checks closed rotation — and geographic panels). Classes:
frontal / coastal / open ocean / reject.

Exhaustive labeling of cycles 1–7 (~800 labels) established two facts: 61% of PET's
kept detections are junk, and junk is not separable by any scalar quality flag. At the
observed pace, labeling all 50 cycles would take 24+ hours — so we built a prior:

**The ADT front prior.** The Gulf Stream separates low ADT (shelf side) from high ADT
(Sargasso side); a single iso-contour of native (non-inpainted) ADT traces the jet
through every pass. We keep only *open* contours (closed rings at the level are eddies
themselves — audited: excluding them changes AUC by 0.008). Each detection gets a
signed distance to this line (sign from `ADT(center) − level`; negative = shoreward).
Validated against the hand labels: sweeping the level, AUC(|distance| ranking frontal
vs open-ocean) plateaus at 0.95 for levels 0.30–0.35 m; we fixed **0.35 m**. Frontal
eddies sit at median 22 km from the line (75% within 50 km); open-ocean at ~216 km.
A gradient-ridge front (max |∇ADT|) was tested and clearly lost (AUC 0.82 vs 0.92) —
eddy rims are themselves strong gradients.

Labeling then proceeded through a queue of all unlabeled *cyclonic* detections within
50 km of the line (predicted 53% frontal; realized 57.9% — out-of-sample validation of
the prior). Final: **2,124 labels, 902 confirmed frontal eddies, all 50 cycles.** Two
documented sampling regimes (cycles 1–7 exhaustive; the rest curated near-front
cyclonic). A 200-eddy random audit queue for unbiased class rates is partially labeled.

## 5. Stage 2 — U-Net

- **Targets:** per-pass binary masks = union of PET speed-contour polygons of confirmed
  frontal eddies (polygons re-derived deterministically and archived).
- **Inputs (v2, current):** native SSHA, native ADT, |∇ADT|, and a validity channel
  (1 = real data), NaN gaps preserved — *no inpainting*. (v1 used the inpainted stack;
  inpainting turned out to explode inside large gaps — ADT reaching +120 m — and caused
  a pathological tail discussed below.)
- **Split (temporal; eddies persist across adjacent 21-day cycles, so random splits
  leak):** train cycles 1–30, val 31–35, **calibration 36–42** (untouched by training or
  model selection), test 43–50.
- **Model:** 3-level U-Net, 1.9M parameters, dice+BCE, positive-tile rebalancing, flip
  augmentation. Trained on Colab (L4).
- **Conformal calibration:** per confirmed eddy, score = max predicted probability
  inside its mask; the split-conformal α-quantile over the calibration cycles gives a
  threshold λ̂ with the distribution-free guarantee "a new frontal eddy is missed with
  probability ≤ 10%". v1's λ̂ degenerated to ~0 (≈10% of calibration eddies were
  invisible to the model — traced to inpaint artifacts); v2 gives a healthy
  **λ̂ = 0.348** with 97.3% empirical test coverage. *Note: this guarantee is also
  conditional on the label distribution — see Section 8.*

**v2 test results (eddy-level, threshold 0.5):** precision 0.41, recall 0.96, F1 0.58
(142 TP / 201 FP / 6 FN).

**Failure analysis** (saliency maps + per-case anatomy + manual review of all 201
extras): the U-Net fires on any cold-core-near-front pattern — front meander troughs,
cold streamers, diffuse lows — because it sees only height fields. It has no
closed-circulation cue: the human vetter used velocity quivers, PET used closed
contours; the net has neither. Manual review determined the large majority of the 201
extras are *not* real frontal eddies (importantly: they are also not "PET misses" —
they are non-eddies). 169/201 extras score p > 0.9, so thresholding cannot rescue
precision; the deficiency is structural.

## 6. Stage 3 — the hybrid (current best)

PET supplies eddy-ness; the U-Net supplies frontal-ness. A pass's frontal eddies = PET
detections whose contour contains U-Net probability > τ (τ = 0.9; performance is flat
in τ because the score distribution is almost perfectly bimodal).

**Test results (244 labeled PET detections, cycles 43–50):**

| method | precision | recall* | F1 |
|---|---|---|---|
| ADT-distance rule (cyclonic, <50 km) | 0.84 | 0.72 | 0.78 |
| U-Net alone | 0.41 | 0.96 | 0.58 |
| **hybrid PET × U-Net (τ=0.9)** | **0.80** | **0.96** | **0.875** |

Decision breakdown at τ=0.9: 144/150 confirmed frontals kept; 0/8 open-ocean and 2/4
coastal leak through; 33/82 hand-rejected PET artifacts survive (the residual precision
loss — these fool both methods and are the target of planned hard-negative retraining).
An emergent check: 100% of hybrid frontals are cyclonic; polarity was never enforced.

\* Recall *with respect to labeled PET detections* — which is exactly the problem:

## 7. What the numbers can and cannot say

Everything above is honest *conditional on PET support*:

- Precision (0.80) is essentially trustworthy: a manual review of the U-Net's
  non-PET detections found the vast majority are not real, so false positives are
  measured against a human standard, not just PET.
- Recall (0.96) is **recall of confirmed-frontal PET detections**. Since every label
  is by construction a PET detection, an eddy PET never surfaced appears in no
  training target, no calibration score, no test metric. The 10% conformal miss-rate
  guarantee inherits the same conditioning: it promises coverage over eddies
  *exchangeable with the calibration set*, i.e., PET-visible ones.

## 8. THE QUESTION — measuring or bounding recall beyond the bootstrap detector

**Setup.** Let F be the true population of frontal eddies present in the observed
swaths. PET detects a subset F_PET ⊂ F (and much junk, which labeling handles). All
2,124 labels — hence training masks, calibration scores, and test truth — are drawn
from PET's output. Our detector's recall estimate R̂ = 0.96 estimates
P(detected | e ∈ F_PET ∩ labeled), not P(detected | e ∈ F).

**Known PET failure modes in this region** (mechanistic, not quantified):
1. Eddies embedded in the strong front gradient that never close an SSH contour within
   the swath (the reference method's own authors note their high-latitude test region
   has small, wholly-within-swath eddies; the Gulf Stream does not cooperate — a
   front-hugging eddy's contour merges with the jet's).
2. Eddies clipped by the swath edge or straddling the nadir gap (rejection rules
   discard contours living on inpainted pixels — correctly for precision, at recall's
   expense).
3. Small (< ~5 km radius) or weak (< 1 cm amplitude) eddies below PET's thresholds.
4. Eddies visible in the *unfiltered* field but smoothed away in `ssha_filtered`.

**The open questions — deliberately unconstrained:**

1. Given the assets in hand (Section 9), how would you measure or bound
   P(e ∈ F_PET | e ∈ F) — the fraction of real frontal eddies the bootstrap detector
   ever surfaced?
2. More broadly: knowing the goal (every frontal eddy in a pass, with confidence) and
   the data (raw swaths, 902 bootstrap-confirmed positives, unused velocity fields,
   repeat-pass geometry), **what approach / model / architecture would you pursue?**
   Nothing about the current PET → U-Net → hybrid design should be treated as fixed;
   it is one path through the problem, not the answer.

## 9. Constraints and partial evidence already in hand

- The U-Net runs on raw fields and is *not* structurally limited to PET's support: it
  could in principle find PET-missed eddies. Empirically, its 201 non-PET detections on
  test cycles were reviewed by the domain expert and are overwhelmingly *not* real
  eddies — a few may be, but the yield is low. So the one non-PET channel we have says
  the easily-visible missed-eddy population is not large, but this is weak evidence:
  the U-Net was trained on PET-shaped targets and plausibly inherited PET's blind
  spots (it learned "what PET-detectable frontal eddies look like").
- Velocity fields (`ugos/vgos`) exist in every granule and are unused by any model so
  far. Closed rotation is visible in them even when the SSH contour cannot close —
  potentially a detection channel with a *different* miss profile than height contours.
- The 21-day repeat gives ~2 revisits/cycle over the region; a persistent eddy missed
  in one pass geometry may be caught in another. No cross-pass tracking exists yet.
- Manual capacity is real but finite: the expert can label a few hours' worth; a
  ~20-pass exhaustive gold set (~every frontal eddy in those passes, PET or not) is
  planned and feasible, but 20 passes may be a small sample for a rate estimate,
  and "exhaustive human annotation" has its own recall question.

## 10. Practical notes for whatever you propose

Deliberately *not* a list of our preferred solutions — just constraints that shape
feasibility:

- Compute: laptop CPU for inference-scale work; Colab GPUs (L4/A100) for training.
  Full-archive reprocessing (all 1,288 granules) costs hours, not days.
- Expert labeling capacity exists but is scarce (a few hours at a time); an exhaustive
  ~20-pass "gold" annotation set is feasible and already contemplated but not built.
- All per-detection artifacts are joinable on a stable key: raw fields, PET contours,
  hand labels, front distances, model scores.
- The 21-day cycle gives ~2 swath revisits of any point per cycle in this region;
  no cross-pass eddy tracking has been built.
- External data (other sensors, ocean-model output, gridded altimetry) is on the table
  — nothing restricts the answer to SWOT-only.

## 11. Numbers at a glance

| quantity | value |
|---|---|
| SWOT granules processed | 1,288 (50 cycles) |
| PET detections in study polygon | 15,641 |
| Hand labels / confirmed frontal | 2,124 / 902 |
| Front prior | ADT 0.35 m open contour; AUC 0.95 (frontal vs open-ocean) |
| U-Net v2 (test) | P 0.41 / R* 0.96 / F1 0.58; conformal λ̂ 0.348, coverage 97.3% |
| Hybrid PET×U-Net (test) | **P 0.80 / R* 0.96 / F1 0.875**; 100% cyclonic |
| Residual hybrid FPs | 33/82 PET artifacts (fool both methods) |
| \* recall conditional on PET support | the subject of Section 8 |

## 12. Materials

- Repository: `github.com/Originaljsx/swot-eddy-ml` (private), branch `aoi5-frontal`.
- Full pipeline record: `docs/PIPELINE.md`; front-prior methods:
  `docs/2026-07-12-frontal-prior-methods.md`.
- Publication figures: `out/aoi5/paper_figs/` (region/labels, training sample, prior
  validation, test predictions, metric sweeps, raw-pass detections).
- Visual reports (inputs/outputs, failure anatomy, saliency, hybrid graphs, raw-pass
  galleries): `out/aoi5/report_v2/`, `out/aoi5/report_hybrid/`.
- Reference: de Marez et al. (2026), Ocean Science 22:1515 — the PET-on-SWOT recipe
  followed for the bootstrap.
