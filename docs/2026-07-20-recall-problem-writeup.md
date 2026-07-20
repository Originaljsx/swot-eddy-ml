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
population of frontal eddies is currently unidentifiable from our data. We want a
principled way to measure, bound, or mitigate this — Section 8 states the question
precisely, Section 9 lists what we know that constrains it, and Section 10 lists the
approaches we've considered with their weaknesses.

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

### 4.1 PET on every pass

Following de Marez et al. (2026) exactly: crop to region → pad 69→128 across-track →
128×128 tiles with 64-px overlap → biharmonic inpainting of gaps (PET requires closed
contours) → PET `eddy_identification` per tile (closed SSH contours around extrema;
shape error ≤ 85%, amplitude ≥ 1 cm, speed radius ≥ 5 km) → rejection of detections
whose contours live mostly on inpainted or nadir pixels (inpainting invents smooth
blobs) → cross-tile deduplication. Result: **15,641 detections over 50 cycles**, with
deterministic per-pass ordering that serves as a stable join key for everything below.

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

**What we want:** a defensible estimate or bound on P(e ∈ F_PET | e ∈ F) for frontal
eddies — equivalently, a way to either (a) measure the detector's true recall, (b)
train/evaluate in a way that is not conditioned on PET, or (c) argue rigorously about
the magnitude and character of the unobserved stratum. What would you do?

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

## 10. Approaches considered, with their problems

1. **Gold passes** (exhaustive expert annotation of ~20 passes, stratified by season
   and front state). Direct estimate of both PET's and the hybrid's true recall.
   Problems: small n → wide confidence intervals on a possibly small miss rate; human
   recall is itself unverified; selection of "annotatable" passes may bias toward easy
   conditions.
2. **Two-tier output** (ship hybrid detections as tier 1, high-probability non-PET
   U-Net blobs as flagged tier-2 candidates for human review). Surfaces misses
   operationally; doesn't measure anything, and tier 2 is precision-poor (~0.4).
3. **Loosen PET** (drop amplitude/shape thresholds, accept junk, let the U-Net gate
   filter). Recovers threshold-type misses (mode 3) only; does nothing for modes 1–2,
   which we believe dominate; junk explosion burdens the gate with inputs unlike its
   training distribution.
4. **Velocity-based detection branch** (add `ugos/vgos` or vorticity channels; or a
   separate winding-number/closed-streamline detector). Attacks mode 1 directly with a
   different physical signature. Problem: its training labels would come from... the
   same PET bootstrap — circularity again, unless combined with (1) or (5).
5. **OSSE / model-truth evaluation:** take a high-resolution ocean simulation of the
   Gulf Stream (e.g., ~1 km regional model) where *all* eddies are knowable from the
   full 3-D state; sample it with the real SWOT orbit/swath geometry, noise, and gap
   masks; run PET and the hybrid on the synthetic swaths; measure true recall by
   construction. Problems: simulation realism (does the model produce frontal eddies
   with the right statistics?), SWOT error-spectrum realism, and domain gap for any
   *trained* component (fair for evaluating PET; shakier for evaluating the U-Net).
6. **Independent sensor cross-validation:** frontal eddies are visible in SST (GOES /
   VIIRS) and ocean-color imagery as backward-breaking waves along the north wall.
   Match cloud-free SST scenes to SWOT passes; count SST-visible frontal eddies in the
   swath footprint; compare with detections. Problems: cloud cover selection bias, SST
   expression ≠ SSH expression (partially different eddy populations), matching
   tolerance in space/time, and it needs its own (manual or algorithmic) SST detector.
7. **Capture–recapture across repeat passes:** treat successive swath samplings of the
   same region as independent "captures" of a persistent eddy population and estimate
   the undetected fraction Lincoln–Petersen-style. Problems: eddy lifetimes (days to
   weeks) vs 10.5-day revisit alias; detection probabilities are heterogeneous
   (size/position-dependent), violating the classical assumptions; needs tracking.

Our current inclination: (1) + (2) as the immediate practical pair, (5) as the
rigorous backbone if the simulation realism holds, (4) as the most promising way to
*reduce* (not just measure) the miss rate. We would value a view on whether (7) can be
made assumption-honest, whether (5)'s domain gap invalidates it for the learned
component, and whether there is a smarter estimator we haven't considered.

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
