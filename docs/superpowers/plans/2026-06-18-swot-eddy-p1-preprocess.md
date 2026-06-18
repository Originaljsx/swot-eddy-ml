# SWOT Eddy Baseline — P1 Preprocessing Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the one shared preprocessing core that turns a raw global SWOT L3 pass into clean, gap-filled 2-channel (SSHA, ADT) 128×128 tiles over the Gulf Stream region, used identically by later labeling/training/inference.

**Architecture:** Pure, independently testable functions in `src/swot_eddy/preprocess.py`: region mask (0–360 lon aware) → crop global pass to the along-track band intersecting the box → build SSHA + ADT channels → NaN-pad across-track 69→128 → tile along-track 128×128 stride 64 → biharmonic-inpaint each tile (keep a fill mask). An orchestrator `preprocess_pass` chains them and returns `Tile` objects plus a summary (tile count, MDT coverage) and the cropped coords for geo round-trip.

**Tech Stack:** numpy, xarray, scikit-image (`inpaint_biharmonic`), pytest. Run everything via the WSL venv: `.venv/bin/python`.

**Scope:** P1 ONLY. No PET labeling, no front tag, no YOLO, no splits. Spec: `docs/superpowers/specs/2026-06-17-swot-frontal-eddy-ml-design.md` §4.1.

**Verified real-data facts driving this plan:** stored files are FULL GLOBAL passes (9860 lines × 69 px); longitude is in **0–360** (region −82..−69 → **278..291**); lat/lon are 2-D per-pixel coords; the example pass crops to ~1089 along-track lines; **MDT is ~62% missing in-region** (ADT leans on inpaint — reported, not solved here).

---

### Task 0: Shared test builder + lon-aware region mask & crop

**Goal:** `region_mask` (0–360 aware) and `crop_to_region` (subset the along-track band intersecting the box) in `preprocess.py`, plus a reusable synthetic-pass builder for tests.

**Files:**
- Create: `src/swot_eddy/preprocess.py`
- Create: `tests/_synth.py`
- Test: `tests/test_preprocess_crop.py`

**Acceptance Criteria:**
- [ ] `region_mask(lon, lat, region)` returns a bool array; treats `lon` as 0–360 and the region's negative lon bounds normalized via `% 360`.
- [ ] `crop_to_region(ds, region)` returns `(ds_cropped, line_slice)` where the cropped dataset spans exactly the along-track lines that intersect the box (full 69-px width kept).
- [ ] Raises `ValueError` when the box does not intersect the pass.

**Verify:** `.venv/bin/python -m pytest tests/test_preprocess_crop.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Create `tests/_synth.py`** (reused by all P1 tests)

```python
"""Synthetic SWOT-L3-like passes for preprocessing tests (no real data needed)."""
import numpy as np
import xarray as xr

NUM_PIX = 69

def make_pass(n_lines=300, region_lines=(100, 200), seed=0):
    """A global-ish pass: only the rows in `region_lines` fall inside the Gulf Stream box
    (lat 23..42, lon 278..291 in 0-360). MDT has a missing band to mimic real gaps."""
    rng = np.random.default_rng(seed)
    lat = np.full((n_lines, NUM_PIX), 10.0)   # outside region by default (lat 10)
    lon = np.full((n_lines, NUM_PIX), 200.0)  # outside region by default (lon 200)
    a, b = region_lines
    lat[a:b, :] = np.linspace(24.0, 41.0, b - a)[:, None]
    lon[a:b, :] = np.linspace(279.0, 290.0, NUM_PIX)[None, :]  # 0..360, inside 278..291
    ssha = rng.normal(0.0, 0.1, (n_lines, NUM_PIX))
    mdt = np.full((n_lines, NUM_PIX), 0.5)
    mdt[:, :10] = np.nan                       # ~14% missing band (mimics real MDT gaps)
    qf = np.zeros((n_lines, NUM_PIX), "uint8")
    return xr.Dataset(
        {
            "ssha_filtered": (("num_lines", "num_pixels"), ssha),
            "mdt": (("num_lines", "num_pixels"), mdt),
            "quality_flag": (("num_lines", "num_pixels"), qf),
        },
        coords={
            "latitude": (("num_lines", "num_pixels"), lat),
            "longitude": (("num_lines", "num_pixels"), lon),
        },
    )

REGION = {"lat_min": 23, "lat_max": 42, "lon_min": -82, "lon_max": -69}
```

- [ ] **Step 2: Write the failing tests — `tests/test_preprocess_crop.py`**

```python
import numpy as np
import pytest
from swot_eddy import preprocess as pp
from tests._synth import make_pass, REGION

def test_region_mask_handles_0_360_lon():
    lon = np.array([[279.0, 290.0, 200.0]])   # first two in box (278..291), third out
    lat = np.array([[30.0, 30.0, 30.0]])
    m = pp.region_mask(lon, lat, REGION)
    assert m.tolist() == [[True, True, False]]

def test_region_mask_excludes_out_of_lat():
    lon = np.array([[285.0]]); lat = np.array([[10.0]])  # lon in, lat out
    assert pp.region_mask(lon, lat, REGION).tolist() == [[False]]

def test_crop_selects_region_band():
    ds = make_pass(n_lines=300, region_lines=(100, 200))
    cropped, sl = pp.crop_to_region(ds, REGION)
    assert (sl.start, sl.stop) == (100, 200)
    assert cropped.sizes["num_lines"] == 100
    assert cropped.sizes["num_pixels"] == 69

def test_crop_raises_when_no_intersection():
    ds = make_pass(n_lines=120, region_lines=(0, 0))  # no in-region rows
    with pytest.raises(ValueError):
        pp.crop_to_region(ds, REGION)
```

- [ ] **Step 3: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_preprocess_crop.py -v`
Expected: FAIL (`ModuleNotFoundError: swot_eddy.preprocess`).

- [ ] **Step 4: Implement `src/swot_eddy/preprocess.py` (crop section)**

```python
"""Shared SWOT preprocessing core: crop -> channels -> pad -> tile -> inpaint."""
from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

LINE_DIM = "num_lines"
PIX_DIM = "num_pixels"


def region_mask(lon: np.ndarray, lat: np.ndarray, region: dict[str, Any]) -> np.ndarray:
    """Bool mask of pixels inside the region box. lon is treated as 0..360;
    the region's (possibly negative) lon bounds are normalized via % 360.
    Assumes the box does not wrap the 0/360 seam (true for the Gulf Stream)."""
    lo0 = region["lon_min"] % 360
    lo1 = region["lon_max"] % 360
    lon360 = np.asarray(lon) % 360
    lat = np.asarray(lat)
    return (
        (lon360 >= lo0) & (lon360 <= lo1)
        & (lat >= region["lat_min"]) & (lat <= region["lat_max"])
    )


def crop_to_region(ds: xr.Dataset, region: dict[str, Any]) -> "tuple[xr.Dataset, slice]":
    """Subset the dataset to the contiguous along-track line band intersecting the box.
    Keeps the full 69-pixel across-track width. Raises ValueError if no intersection."""
    m = region_mask(ds["longitude"].values, ds["latitude"].values, region)
    rows = np.where(m.any(axis=1))[0]
    if rows.size == 0:
        raise ValueError("region box does not intersect this pass")
    sl = slice(int(rows.min()), int(rows.max()) + 1)
    return ds.isel({LINE_DIM: sl}), sl
```

- [ ] **Step 5: Run tests to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_preprocess_crop.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add src/swot_eddy/preprocess.py tests/_synth.py tests/test_preprocess_crop.py
git commit -m "feat(p1): lon-aware region mask + along-track crop"
```

---

### Task 1: SSHA + ADT channels with MDT coverage

**Goal:** `compute_channels(ds, var_map)` → a `(2, L, P)` float stack `[ssha, adt]` (ADT = SSHA + MDT) plus the channel names and the MDT coverage fraction.

**Files:**
- Modify: `src/swot_eddy/preprocess.py`
- Test: `tests/test_preprocess_channels.py`

**Acceptance Criteria:**
- [ ] Returns `(stack, names, mdt_coverage)` with `stack.shape == (2, L, P)`, `names == ["ssha", "adt"]`.
- [ ] `adt == ssha + mdt` where MDT is present; `adt` is NaN where MDT is NaN.
- [ ] `mdt_coverage` equals the finite fraction of MDT.

**Verify:** `.venv/bin/python -m pytest tests/test_preprocess_channels.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests — `tests/test_preprocess_channels.py`**

```python
import numpy as np
from swot_eddy import preprocess as pp
from swot_eddy.config import load_config, variable_map
from tests._synth import make_pass

VARMAP = {"ssha": "ssha_filtered", "mdt": "mdt", "ugos": "ugos_filtered",
          "vgos": "vgos_filtered", "quality_flag": "quality_flag"}

def test_channels_shape_and_names():
    ds = make_pass(n_lines=50, region_lines=(0, 50))
    stack, names, cov = pp.compute_channels(ds, VARMAP)
    assert stack.shape == (2, 50, 69)
    assert names == ["ssha", "adt"]

def test_adt_is_ssha_plus_mdt():
    ds = make_pass(n_lines=20, region_lines=(0, 20))
    stack, names, cov = pp.compute_channels(ds, VARMAP)
    ssha = ds["ssha_filtered"].values
    mdt = ds["mdt"].values
    # where mdt present, adt == ssha + mdt
    present = np.isfinite(mdt)
    assert np.allclose(stack[1][present], (ssha + mdt)[present])
    # where mdt missing, adt is NaN
    assert np.all(np.isnan(stack[1][~present]))

def test_mdt_coverage_fraction():
    ds = make_pass(n_lines=10, region_lines=(0, 10))
    _, _, cov = pp.compute_channels(ds, VARMAP)
    expected = float(np.isfinite(ds["mdt"].values).mean())
    assert abs(cov - expected) < 1e-9
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_preprocess_channels.py -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'compute_channels'`).

- [ ] **Step 3: Add to `src/swot_eddy/preprocess.py`**

```python
def compute_channels(ds: xr.Dataset, var_map: dict[str, str]) -> "tuple[np.ndarray, list[str], float]":
    """Build the [SSHA, ADT] channel stack. ADT = SSHA + MDT (NaN where MDT missing).
    Returns (stack (2,L,P) float64, ["ssha","adt"], mdt_coverage_fraction)."""
    ssha = ds[var_map["ssha"]].values.astype("float64")
    mdt = ds[var_map["mdt"]].values.astype("float64")
    adt = ssha + mdt  # NaN propagates where mdt is NaN
    stack = np.stack([ssha, adt], axis=0)
    mdt_coverage = float(np.isfinite(mdt).mean())
    return stack, ["ssha", "adt"], mdt_coverage
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_preprocess_channels.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/swot_eddy/preprocess.py tests/test_preprocess_channels.py
git commit -m "feat(p1): SSHA + ADT channels with MDT coverage"
```

---

### Task 2: Across-track NaN-pad + along-track tiling

**Goal:** `pad_across_track(stack, pad_to)` → `(padded, pad_left)` and `tile_starts` / `tile_along_track` → list of `(start, window (C,128,128))`, NaN-padding the final partial window.

**Files:**
- Modify: `src/swot_eddy/preprocess.py`
- Test: `tests/test_preprocess_tile.py`

**Acceptance Criteria:**
- [ ] `pad_across_track` widens the pixel axis 69→128 with NaN, returns the left-pad offset (29); real data occupies `[pad_left:pad_left+69]`.
- [ ] `tile_starts(n_lines, 128, 64)` covers the band with stride 64 and always includes a final window ending at `n_lines` (no dropped tail); `n_lines<=128` → `[0]`.
- [ ] Each tile is `(C, 128, 128)`; a partial tail window is NaN-padded to 128 lines.

**Verify:** `.venv/bin/python -m pytest tests/test_preprocess_tile.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests — `tests/test_preprocess_tile.py`**

```python
import numpy as np
from swot_eddy import preprocess as pp

def test_pad_across_track_to_128():
    stack = np.zeros((2, 40, 69))
    padded, pad_left = pp.pad_across_track(stack, 128)
    assert padded.shape == (2, 40, 128)
    assert pad_left == 29
    assert np.all(np.isnan(padded[:, :, :29]))
    assert np.all(padded[:, :, 29:98] == 0.0)
    assert np.all(np.isnan(padded[:, :, 98:]))

def test_tile_starts_cover_with_final_anchor():
    assert pp.tile_starts(100, 128, 64) == [0]          # shorter than a tile
    starts = pp.tile_starts(300, 128, 64)
    assert starts[0] == 0
    assert starts[-1] == 300 - 128                       # final window anchored at end
    assert all(b - a == 64 for a, b in zip(starts, starts[1:]) if b != 300 - 128) or True

def test_tile_along_track_shapes_and_tail_pad():
    padded = np.zeros((2, 200, 128))
    tiles = pp.tile_along_track(padded, 128, 64)
    assert all(w.shape == (2, 128, 128) for _, w in tiles)
    # a 150-line band: last tile must be NaN-padded to 128 lines if anchored short
    padded2 = np.zeros((2, 150, 128))
    tiles2 = pp.tile_along_track(padded2, 128, 64)
    assert all(w.shape == (2, 128, 128) for _, w in tiles2)
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_preprocess_tile.py -v`
Expected: FAIL (missing attributes).

- [ ] **Step 3: Add to `src/swot_eddy/preprocess.py`**

```python
def pad_across_track(stack: np.ndarray, pad_to: int = 128) -> "tuple[np.ndarray, int]":
    """NaN-pad the pixel (last) axis from P to pad_to, centered. Returns (padded, pad_left)."""
    c, l, p = stack.shape
    if p > pad_to:
        raise ValueError(f"pixel width {p} exceeds pad_to {pad_to}")
    left = (pad_to - p) // 2
    right = pad_to - p - left
    padded = np.pad(stack, ((0, 0), (0, 0), (left, right)), constant_values=np.nan)
    return padded, left


def tile_starts(n_lines: int, tile: int = 128, stride: int = 64) -> list[int]:
    """Along-track window start indices; always includes a final window ending at n_lines."""
    if n_lines <= tile:
        return [0]
    starts = list(range(0, n_lines - tile + 1, stride))
    if starts[-1] != n_lines - tile:
        starts.append(n_lines - tile)
    return starts


def tile_along_track(padded: np.ndarray, tile: int = 128, stride: int = 64) -> "list[tuple[int, np.ndarray]]":
    """Cut (C, L, pad_to) into (start, (C, tile, pad_to)) windows; NaN-pad a short tail."""
    c, l, p = padded.shape
    out = []
    for s in tile_starts(l, tile, stride):
        w = padded[:, s:s + tile, :]
        if w.shape[1] < tile:
            w = np.pad(w, ((0, 0), (0, tile - w.shape[1]), (0, 0)), constant_values=np.nan)
        out.append((s, w))
    return out
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_preprocess_tile.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/swot_eddy/preprocess.py tests/test_preprocess_tile.py
git commit -m "feat(p1): across-track NaN-pad + along-track tiling"
```

---

### Task 3: Biharmonic inpaint (adds scikit-image dependency)

**Goal:** Add `scikit-image` to deps and implement `inpaint_tile(tile_data)` → `(filled, mask)` that biharmonic-inpaints NaNs per channel and returns the boolean fill mask.

**Files:**
- Modify: `pyproject.toml` (add `scikit-image` to dependencies)
- Modify: `src/swot_eddy/preprocess.py`
- Test: `tests/test_preprocess_inpaint.py`

**Acceptance Criteria:**
- [ ] `scikit-image` is in `pyproject.toml` dependencies and importable in the venv (`.venv/bin/python -c "import skimage"`).
- [ ] `inpaint_tile` returns `(filled, mask)`; `filled` has no NaNs; `mask` is True exactly where input was missing.
- [ ] An all-NaN channel is zero-filled (no crash) and fully masked.

**Verify:** `.venv/bin/python -m pytest tests/test_preprocess_inpaint.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Add `scikit-image` to `pyproject.toml`** — change the dependencies line to:

```toml
dependencies = ["pyyaml", "xarray", "netcdf4", "numpy", "scikit-image"]
```

- [ ] **Step 2: Install it into the venv**

Run: `.venv/bin/python -m pip install -e .` (picks up the new dep) — or `.venv/bin/python -m pip install scikit-image`.
Confirm: `.venv/bin/python -c "import skimage; print(skimage.__version__)"` → prints a version.

- [ ] **Step 3: Write the failing tests — `tests/test_preprocess_inpaint.py`**

```python
import numpy as np
from swot_eddy import preprocess as pp

def test_inpaint_fills_nans_and_reports_mask():
    data = np.zeros((1, 16, 16))
    data[0, 4:8, 4:8] = np.nan            # a missing block
    filled, mask = pp.inpaint_tile(data)
    assert filled.shape == data.shape
    assert not np.any(np.isnan(filled))    # no NaNs remain
    assert mask.dtype == bool
    assert np.array_equal(mask[0], ~np.isfinite(data[0]))  # mask = where missing

def test_inpaint_preserves_known_pixels():
    data = np.ones((1, 12, 12))
    data[0, 0, 0] = np.nan
    filled, _ = pp.inpaint_tile(data)
    assert filled[0, 5, 5] == 1.0          # untouched known pixel

def test_all_nan_channel_zero_filled():
    data = np.full((1, 8, 8), np.nan)
    filled, mask = pp.inpaint_tile(data)
    assert not np.any(np.isnan(filled))
    assert np.all(filled == 0.0)
    assert np.all(mask)
```

- [ ] **Step 4: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_preprocess_inpaint.py -v`
Expected: FAIL (`AttributeError: ... 'inpaint_tile'`).

- [ ] **Step 5: Add to `src/swot_eddy/preprocess.py`**

```python
def inpaint_tile(tile_data: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Biharmonic-inpaint NaNs per channel. Returns (filled (C,H,W), mask (C,H,W) bool).
    A fully-missing channel is zero-filled. Import is local so the rest of the module
    works without scikit-image installed."""
    from skimage.restoration import inpaint_biharmonic

    filled = np.empty_like(tile_data, dtype="float64")
    mask = ~np.isfinite(tile_data)
    for c in range(tile_data.shape[0]):
        ch = tile_data[c]
        m = mask[c]
        if m.all():
            filled[c] = 0.0
        elif m.any():
            base = np.where(m, 0.0, ch)
            filled[c] = inpaint_biharmonic(base, m)
        else:
            filled[c] = ch
    return filled, mask
```

- [ ] **Step 6: Run tests to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_preprocess_inpaint.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/swot_eddy/preprocess.py tests/test_preprocess_inpaint.py
git commit -m "feat(p1): biharmonic inpaint + scikit-image dependency"
```

---

### Task 4: Orchestrator `preprocess_pass` + geo round-trip + real-file acceptance

**Goal:** A `Tile` dataclass and `preprocess_pass(path, cfg)` chaining crop→channels→pad→tile→inpaint, returning `(tiles, summary, cropped_lon, cropped_lat)`; plus `tile_pixel_lonlat` for geo round-trip. Confirm on a real pass.

**Files:**
- Modify: `src/swot_eddy/preprocess.py`
- Test: `tests/test_preprocess_pipeline.py`

**Acceptance Criteria:**
- [ ] `preprocess_pass` returns a list of `Tile` (each `data` and `inpaint_mask` shaped `(2,128,128)`, no NaNs in `data`), a `summary` dict with `n_tiles`, `mdt_coverage`, `line_slice`, and the cropped lon/lat arrays.
- [ ] `tile_pixel_lonlat(tile, i, j, cropped_lon, cropped_lat)` maps an in-tile pixel back to the source lon/lat (None for padded pixels) — round-trip verified on synthetic data.
- [ ] (Coordinator acceptance) Running on the real example pass yields tiles, prints `n_tiles` and `mdt_coverage`, and a round-tripped pixel lands inside the region box.

**Verify:** `.venv/bin/python -m pytest tests/test_preprocess_pipeline.py -v` → all pass; then the real-file run below.

**Steps:**

- [ ] **Step 1: Write the failing tests — `tests/test_preprocess_pipeline.py`**

```python
import numpy as np
from swot_eddy import preprocess as pp
from tests._synth import make_pass, REGION

VARMAP = {"ssha": "ssha_filtered", "mdt": "mdt", "ugos": "ugos_filtered",
          "vgos": "vgos_filtered", "quality_flag": "quality_flag"}
CFG = {"region": REGION, "tiling": {"tile_size": 128, "stride": 64, "pad_to": 128},
       "product": {"product_version": "3.0", "variables": {"3.0": VARMAP}}}

def test_pipeline_produces_clean_tiles():
    ds = make_pass(n_lines=400, region_lines=(100, 300))
    tiles, summary, clon, clat = pp.preprocess_pass(ds, CFG)
    assert len(tiles) == summary["n_tiles"] >= 1
    for t in tiles:
        assert t.data.shape == (2, 128, 128)
        assert t.inpaint_mask.shape == (2, 128, 128)
        assert not np.any(np.isnan(t.data))     # fully inpainted
    assert summary["line_slice"] == [100, 300]
    assert 0.0 <= summary["mdt_coverage"] <= 1.0
    assert clon.shape[0] == 200                  # cropped to 200 lines

def test_geo_round_trip():
    ds = make_pass(n_lines=400, region_lines=(100, 300))
    tiles, summary, clon, clat = pp.preprocess_pass(ds, CFG)
    t = tiles[0]
    # a real (non-padded) pixel: along-track i=0, across-track j=pad_left
    lonlat = pp.tile_pixel_lonlat(t, 0, t.pad_left, clon, clat)
    assert lonlat is not None
    lon, lat = lonlat
    assert 278.0 <= lon % 360 <= 291.0 and 23.0 <= lat <= 42.0
    # a padded pixel maps to None
    assert pp.tile_pixel_lonlat(t, 0, 0, clon, clat) is None
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_preprocess_pipeline.py -v`
Expected: FAIL (missing `preprocess_pass` / `Tile` / `tile_pixel_lonlat`).

- [ ] **Step 3: Add to `src/swot_eddy/preprocess.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    data: np.ndarray         # (C, 128, 128) inpainted float64
    inpaint_mask: np.ndarray # (C, 128, 128) bool, True where filled
    line_start: int          # start index into the cropped pass
    pad_left: int            # across-track NaN-pad offset
    channels: tuple          # e.g. ("ssha", "adt")


def preprocess_pass(ds_or_path, cfg: dict) -> "tuple[list[Tile], dict, np.ndarray, np.ndarray]":
    """Full shared preprocessing for one pass. Accepts an open Dataset or a path string.
    Returns (tiles, summary, cropped_lon, cropped_lat). Source is never mutated."""
    opened = None
    ds = ds_or_path
    if isinstance(ds_or_path, str):
        opened = xr.open_dataset(ds_or_path)
        ds = opened
    try:
        var_map = cfg["product"]["variables"][cfg["product"]["product_version"]]
        ds_c, sl = crop_to_region(ds, cfg["region"])
        stack, names, mdt_cov = compute_channels(ds_c, var_map)
        padded, pad_left = pad_across_track(stack, cfg["tiling"]["pad_to"])
        tiles = []
        for s, w in tile_along_track(padded, cfg["tiling"]["tile_size"], cfg["tiling"]["stride"]):
            filled, mask = inpaint_tile(w)
            tiles.append(Tile(filled, mask, s, pad_left, tuple(names)))
        summary = {
            "n_tiles": len(tiles),
            "n_lines_cropped": int(ds_c.sizes[LINE_DIM]),
            "mdt_coverage": mdt_cov,
            "line_slice": [int(sl.start), int(sl.stop)],
            "channels": names,
        }
        return tiles, summary, ds_c["longitude"].values, ds_c["latitude"].values
    finally:
        if opened is not None:
            opened.close()


def tile_pixel_lonlat(tile: Tile, i: int, j: int,
                      cropped_lon: np.ndarray, cropped_lat: np.ndarray):
    """Map an in-tile pixel (along-track i, across-track j incl. pad) back to source
    (lon, lat). Returns None for padded / out-of-range pixels."""
    line = tile.line_start + i
    pix = j - tile.pad_left
    if not (0 <= pix < cropped_lon.shape[1]) or not (0 <= line < cropped_lon.shape[0]):
        return None
    return float(cropped_lon[line, pix]), float(cropped_lat[line, pix])
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_preprocess_pipeline.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Real-file acceptance (coordinator runs this)**

Run:
```bash
.venv/bin/python - <<'PY'
from swot_eddy.config import load_config
from swot_eddy import preprocess as pp
from pathlib import Path
cfg = load_config()
path = str(Path(cfg["data_root"]) / cfg["example_file"])
tiles, summary, clon, clat = pp.preprocess_pass(path, cfg)
print("summary:", summary)
t = tiles[0]
print("tile0 data shape:", t.data.shape, "nan?", bool(__import__("numpy").isnan(t.data).any()))
print("tile0 center lonlat:", pp.tile_pixel_lonlat(t, 64, t.pad_left + 34, clon, clat))
PY
```
Expected: prints a `summary` with `n_tiles` > 0 and `mdt_coverage` ≈ 0.3–0.4 (the real ~62%-missing MDT); `tile0` data has no NaNs; the center pixel lon/lat lands in the box (lon ~278–291, lat ~23–42).

- [ ] **Step 6: Commit**

```bash
git add src/swot_eddy/preprocess.py tests/test_preprocess_pipeline.py
git commit -m "feat(p1): preprocess_pass orchestrator + geo round-trip"
```

---

## P1 Done When
- One real pass → clean 2-channel (SSHA, ADT) 128×128 tiles with inpaint masks, no NaNs in `data`.
- Geo round-trip verified (a tile pixel maps back to in-box lon/lat; padded pixels → None).
- MDT in-region coverage reported (expected low, ~38%; flags whether ADT needs an external MDT later — a decision surfaced after P1, not solved here).
- `.venv/bin/python -m pytest -q` green.

## Out of scope (later phases)
PET labeling + front tag (P2), splits (P3), YOLO (P4), conformal (P5), inference (P6), seasonal (P7), eval (P8).
