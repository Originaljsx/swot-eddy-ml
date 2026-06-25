# AOI5 polygon clip + 4-class frontal relabel — design

**Date:** 2026-06-25
**Status:** Approved (design), pending implementation plan
**Branch target:** `aoi5-frontal` (current `master` tagged `pipeline-aoi3-bbox` first)

## Goal

Refocus the SWOT Gulf Stream eddy pipeline on **frontal eddies**. Two changes:

1. **Spatial:** replace the AOI3 rectangular bbox (which sweeps a large block of
   open ocean in the SE) with the new **GS_AOI5 polygon**, which clips the SE open
   ocean and concentrates on the Gulf Stream + inland.
2. **Labeling:** restart labeling from scratch with **4 classes** —
   `frontal`, `coastal`, `open ocean`, `reject`. Cyclonic/anticyclonic polarity
   stays **auto** (PET, from SLA sign) and is not hand-labeled.

The current (AOI3 bbox) pipeline is preserved as a reference, not deleted.

## Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Polygon role | **Filter detections only.** Crop/tile/inpaint stay on the polygon's envelope band; polygon gates detection *centers* by point-in-polygon. No new inpaint-edge artifacts. |
| `coastal` class | **Hand-label** (4th button), part of a full relabel restart. |
| Old pipeline preservation | **Git tag** `pipeline-aoi3-bbox` on `master`, then build forward on branch `aoi5-frontal`. |
| AOI source file | **GeoJSON** `C:\Users\jspier\Bbox\GS_AOI5.json`, read via **shapely** (already present — ships with cartopy; shapely 2.1.2 confirmed). |
| config envelope | **Auto-derive** the crop bbox from the polygon at runtime. Polygon is the single source of truth; no hand-kept `region:` numbers to drift. |
| Re-run scope | Plan includes a **smoke-test** (`run_cycle` on one cycle) before any full `run_all`. |
| Output location | **Separate `out/aoi5/`** subtree so AOI3 reference outputs stay intact for A/B. |

## Polygon facts (GS_AOI5.json)

- `GeometryCollection` with one `Polygon`, 10 vertices (closed ring).
- Lon range ≈ -85.48 .. -68.41 (WGS84, -180..180 convention).
- Lat range ≈ 23.32 .. 40.79.
- Concave on the SE side — that concavity is exactly the open-ocean cut, so a
  point-in-polygon test (not the envelope) is required for the detection filter.
- **Convention bridge:** SWOT source lon is 0..360; polygon coords are -180..180.
  All `contains()` checks normalize detection lon to -180..180 before testing.

## Architecture

### New module: `src/swot_eddy/aoi.py`

Single owner of the polygon. Keeps geometry logic out of preprocess/detect/scripts.

```
load_aoi(cfg) -> AOI            # reads cfg aoi.geojson (per-platform path), caches
class AOI:
    contains(lon, lat) -> bool  # lon any convention -> normalized to -180..180, shapely point-in-poly
    envelope_region() -> dict   # {lon_min, lat_min, lon_max, lat_max} from polygon.bounds
```

- `contains` accepts scalars; a vectorized helper may be added only if a call site
  needs arrays (none currently do — all 4 filter sites loop detection-by-detection).
- `envelope_region()` returns the same dict shape `crop_to_region`/`region_mask`
  already consume, so **preprocess is untouched** beyond receiving an
  envelope-derived region instead of a config one.

### config.yaml

- Add an `aoi:` block with per-platform geojson paths (windows / wsl / colab),
  mirroring the existing `paths:` structure:
  ```yaml
  aoi:
    windows: "C:/Users/jspier/Bbox/GS_AOI5.json"
    wsl:     "/mnt/c/Users/jspier/Bbox/GS_AOI5.json"
    colab:   "/content/drive/MyDrive/Bbox/GS_AOI5.json"
  ```
- The existing `region:` block is **superseded**: callers get their crop bbox from
  `load_aoi(cfg).envelope_region()`. Keep the `region:` key only if any non-pipeline
  code reads it; otherwise mark it deprecated in a comment. (Resolved during plan
  writing by grepping remaining readers.)
- `config.py` loader resolves `aoi` path for the active platform the same way it
  resolves `data_root`/`out_root`.

### The 4 bbox→polygon swap sites

All currently do an inline rectangular `lon/lat in [min,max]` test. Each becomes
`aoi.contains(lon, lat)`:

1. `src/swot_eddy/review.py` `CatalogLabeler._in_box` (line ~513)
2. `scripts/run_cycle.py` `in_box` (line ~28)
3. `scripts/run_all.py` `in_box` (line ~102)
4. (preprocess `crop_to_region` stays — it crops the **envelope band**, correct by design)

Each site loads the AOI once per run (not per detection).

### Labeling restart (`CatalogLabeler`)

- Default `labels` → `("frontal", "coastal", "open ocean", "reject")`.
- Default `labels_path` → `out/aoi5/labels/eddy_labels_aoi5.csv` (new file; old
  `out/labels/eddy_labels.csv` stays as AOI3 reference).
- Manifest read path → `out/aoi5/catalog/passes_done.csv`.
- Two-phase keep/reject walk and resumability logic unchanged.
- `open ocean` is a label string with a space — confirm the hotkey/CSV/button code
  handles it (it numbers buttons 1..N and stores the raw string; spaces are fine).

### Output relocation (`out/aoi5/`)

- `scripts/run_all.py` writes its manifest + per-pass artifacts under `out/aoi5/`.
- `scripts/run_cycle.py` review PNGs under `out/aoi5/<cycle>_review/` (or a flag).
- Resolve exact subpaths during plan writing; principle: **nothing under the AOI5
  run touches the existing AOI3 `out/` files.**

## Data flow (unchanged except filter + paths)

```
.nc pass
  -> crop_to_region(ds, aoi.envelope_region())     # envelope band, full 69-px width
  -> compute_channels -> pad -> tile(128) -> inpaint
  -> detect_pass_split(tiles)  -> kept, rejected    # PET, polarity auto
  -> [d for d in kept if aoi.contains(d.center_lon, d.center_lat)]   # POLYGON clip
  -> CatalogLabeler: hand-label frontal/coastal/open ocean/reject
```

## Verification

1. **AOI load unit check:** `aoi.envelope_region()` lon ≈ (-85.48, -68.41),
   lat ≈ (23.32, 40.79); `contains` true for a GS-core point (e.g. -75, 36),
   false for an SE open-ocean point inside the envelope but outside the polygon
   (pick one in the clipped concave corner, e.g. ~-70, 25).
2. **Smoke test:** `.venv/bin/python scripts/run_cycle.py cycle_004` runs clean and
   writes AOI5 detections under `out/aoi5/`; detection count is **lower** than the
   AOI3 run on the same cycle (open ocean removed). Spot-check no kept detection
   center falls in the clipped SE corner.
3. **Labeler smoke:** `CatalogLabeler` instantiates with the 4 labels, points at
   `out/aoi5/...`, finds the AOI5 manifest, renders the first eddy with 4 buttons.
4. **Reference intact:** old `out/labels/eddy_labels.csv` and `out/catalog/` byte-identical
   after the AOI5 run; `git tag pipeline-aoi3-bbox` resolves to pre-change `master`.

## Out of scope

- Auto distance-to-coast classification (coastal is hand-labeled).
- Pixel-level polygon masking before detection (filter-only by decision).
- Any change to PET params, inpainting, tiling, or polarity assignment.
- Migrating old AOI3 labels into the new 4-class scheme (clean restart).

## Risks

- **Envelope vs polygon confusion:** crop uses envelope (rectangle), filter uses
  polygon. A reviewer might "fix" the crop to use the polygon — document why crop
  stays rectangular (inpaint edge artifacts) in the code comment.
- **lon convention:** the single most likely bug. `contains` must normalize
  0..360 → -180..180. Covered by verification #1's open-ocean false case.
- **shapely as implicit dep:** add `shapely` explicitly to `pyproject.toml`
  dependencies (currently transitive via cartopy) so it can't silently vanish.
```
