# From Satellite Strip to Frontal Eddies — the SWOT → PET → ADT pipeline

## Overview
- **Topic**: How raw SWOT altimetry becomes labeled frontal-eddy detections: de Marez
  preprocessing (crop → pad → tile → biharmonic inpaint), py-eddy-tracker contour
  detection, and the ADT front line used to classify frontal eddies.
- **Hook**: "One 120-km-wide strip of sea-surface height. Somewhere in it: frontal
  eddies spinning along the Gulf Stream. How do you find them?"
- **Target Audience**: oceanography / data-science colleagues; comfortable with basic
  calculus, no altimetry background needed.
- **Estimated Length**: ~5 minutes
- **Key Insight**: two iso-contour tricks do all the work — closed SSH contours find
  eddies, one OPEN ADT contour finds the Gulf Stream; signed distance between them
  classifies frontal eddies (AUC 0.95).
- **Resolution**: 480p (default)
- **Aspect Ratio**: 16:9

## Narrative Arc
A satellite swath looks like a noisy, gap-riddled ribbon of numbers. Step by step the
pipeline tames it: gaps are filled by solving a PDE, closed height contours become
eddies, and one magic ADT contour traces the Gulf Stream so each eddy's distance to it
reveals its type. Real detections from the Gulf Stream dataset appear throughout.

---

## Scene 1: Scene1_Hook (~30 s)
**Purpose**: pose the problem with real data.
### Visual Elements
- Real SWOT pass ADT image (asset `adt_full.png`), slides in as a tilted ribbon
- Title text; question text
### Content
Show the real swath. Zoom-feel by scaling image up slightly. Ask: where are the frontal
eddies? Flash 3 gold rings (positions from real labels) then fade — "by the end you'll
know how these were found."
### Voiceover (subcaptions)
- "This is one pass of the SWOT satellite over the Gulf Stream…"
- "Somewhere in here are frontal eddies. This is how we find them."

## Scene 2: Scene2_Swath (~45 s)
**Purpose**: anatomy of the data + the two channels.
### Visual Elements
- Pixel-grid diagram of swath cross-section: 69 across-track pixels @ 2 km, nadir gap
  in the middle (gray strip), rain/quality holes
- Real SSHA image (`ssha_full.png`) beside ADT image (`adt_full.png`)
- MathTex: SSHA = ssha_filtered (detection channel); ADT = ssha_unedited + MDT
### Content
Grid builds up; nadir gap highlighted — the satellite can't see directly below itself.
Two channels appear with formulas: SSHA (anomaly, what eddies are detected on) and ADT
(absolute height, carries the Gulf Stream itself).
### Technical Notes
- `VGroup` of small squares for grid; `ImageMobject` w/ nearest resampling for real data.

## Scene 3: Scene3_PadTile (~35 s)
**Purpose**: 69→128 pad + 128×128 tiling with 64-px overlap.
### Visual Elements
- Long thin rectangle (swath, 69 px wide) → NaN-pad animation to 128 (gray wings)
- Sliding 128×128 window stepping by 64, overlap region flashing
### Content
Networks and PET both want square tiles. Pad across-track with NaN, slide a 128×128
window along-track with 50% overlap so every eddy is fully inside at least one tile.
### Voiceover
- "Pad the 69 pixels to 128… then cut overlapping square tiles."

## Scene 4: Scene4_Inpaint (~50 s)
**Purpose**: THE interpolation — biharmonic inpainting.
### Visual Elements
- Real tile pair before/after (`tile_before.png`, `tile_after.png`)
- MathTex: ∇⁴u = 0 on Ω (gaps), u = data on ∂Ω; expanded
  u_xxxx + 2 u_xxyy + u_yyyy = 0
- Smoothness metaphor: 1-D curve with a hole; straight-line fill (∇²) vs spline-like
  fill (∇⁴) — biharmonic matches value AND slope at the hole edges
### Content
PET needs CLOSED contours; NaN holes break contours. Fill every gap by solving the
biharmonic equation — the "thin-plate" interpolant: smoothest surface agreeing with the
data at the hole boundary. Caution beat: inpainting can invent smooth blobs inside the
nadir gap — flagged later.
### Voiceover
- "The gaps are filled by solving a PDE — the biharmonic equation…"
- "…the same equation as a thin bent plate: smooth in value and slope."

## Scene 5: Scene5_PET (~55 s)
**Purpose**: py-eddy-tracker detection on the inpainted SLA.
### Visual Elements
- Contour "topo map" of SLA: nested closed contours growing around extrema
- Geostrophy MathTex: u_g = -(g/f) ∂η/∂y ,  v_g = (g/f) ∂η/∂x (height gradients spin
  water — SSH bumps ARE eddies)
- Criteria checklist: amplitude ≥ 1 cm, R_max ≥ 5 km, shape error ≤ 85 %, 4–1000 px
- Speed contour + R_max annotation on a real eddy zoom (`eddy_zoom_1.png`)
- Real examples row: `eddy_zoom_1/2/3.png` (confirmed frontal eddies w/ PET contour)
### Content
PET scans SLA levels, keeps closed contours around extrema passing the de Marez
criteria; the speed contour (max mean geostrophic speed) defines the eddy edge R_max.
Sign of SLA gives polarity: lows = cyclonic (blue), highs = anticyclonic (red).
### Voiceover
- "Closed height contours are eddies — geostrophy makes water orbit height extrema."
- "These are real detections from the dataset."

## Scene 6: Scene6_Reject (~30 s)
**Purpose**: honesty step — dedup + gap/nadir rejection.
### Visual Elements
- Two overlapping tiles both drawing same eddy → merge (dedup, 10-km rule)
- Eddy whose contour lies mostly in gray inpainted zone → red X (gap_frac > 40 %)
- Eddy with ≥70 % of area in nadir strip → red X
### Content
Overlap tiles find the same eddy twice — dedup keeps one. Contours living mostly on
invented (inpainted) pixels are rejected; so are nadir-gap blobs.
### Voiceover
- "If most of an eddy stands on invented pixels, we don't trust it."

## Scene 7: Scene7_ADTFront (~60 s)
**Purpose**: the ADT contour/gradient trick for FRONTAL classification.
### Visual Elements
- 1-D ADT profile across the swath: low (shelf) → steep jump (jet) → high (Sargasso);
  animated dot walking the tanh curve
- Horizontal line at 0.35 m intersecting the jump → the front point; extend to 2-D:
  iso-contour ADT = 0.35 m drawn gold on real ADT image (`front_overlay.png`)
- |∇ADT| image (`grad_adt.png`) — bright ridge = jet; gold line rides the ridge
- MathTex: F = { x : ADT(x) = 0.35 m } (open contours only);
  d(e) = sign(ADT(x_e) − 0.35) · min_{p∈F} ‖x_e − p‖
- Histogram bars (manim-native): frontal median 22 km vs open-ocean 216 km; AUC 0.95
### Content
ADT steps from low to high across the Gulf Stream — so ONE level-line traces the jet
through every pass. Closed rings at that level are eddies themselves and are dropped;
the open curve is the front. Signed distance to it separates frontal (hug the line,
inshore side) from open-ocean eddies almost perfectly.
### Voiceover
- "The Gulf Stream is a step in absolute height — one contour line finds it."
- "Frontal eddies live within ~50 km of this line, on the coastal side."

## Scene 8: Scene8_Pipeline (~25 s)
**Purpose**: recap chain + real payoff.
### Visual Elements
- Pipeline chain: SWOT pass → crop/pad/tile → ∇⁴u=0 → PET contours → ADT front
  distance → frontal candidates
- Real gallery image (`front_overlay.png`) with gold rings = final candidates
- Metrics text: "84 % precision / 72 % recall among real eddies (n = 802 labels)"
### Voiceover
- "From raw strip to labeled frontal eddies — two contour tricks and one PDE."

---

## Transitions & Flow
Ribbon/swath motif recurs (Scenes 1, 2, 3, 7, 8). Gold = front/frontal throughout.
Each scene FadeOut-clean before next.

## Shared Elements
- Real-pass images from assets/ (generated from actual dataset by assets.py)
- Blue = cyclonic / cold; Red = anticyclonic / warm; Gold = front + frontal eddies
- Gray = NaN / inpainted / untrusted

## Color Palette
- Background: #111418
- Primary: #58a6ff (cyclonic blue)
- Secondary: #f85149 (anticyclonic red)
- Accent: #e3b341 (gold — front line, frontal eddies)
- Muted: #8b949e (gaps/inpaint gray)
