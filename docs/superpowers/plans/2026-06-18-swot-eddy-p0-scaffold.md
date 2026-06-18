# SWOT Eddy Baseline — P0 Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the repo skeleton, a version-aware `config.yaml` with a Windows↔Colab path switch, a pinned-`pyEddyTracker` environment, and an inspection script that dumps a real SWOT `.nc` file's structure.

**Architecture:** A small `src/swot_eddy/` package holds a config loader (resolves data/out roots per platform), a `pet_version` recorder (pins the installed PET version into `config.yaml`), and an `inspect_nc` summarizer. Tests use synthetic data so they run anywhere; the real-file dump is a manual acceptance step.

**Tech Stack:** Python 3.11, conda-forge env, `pyEddyTracker` (pip), `xarray`/`netcdf4`, `pyyaml`, `pytest`.

**Scope:** P0 ONLY. Do not implement preprocessing, PET labeling, YOLO, splits, or any later phase. Spec: `docs/superpowers/specs/2026-06-17-swot-frontal-eddy-ml-design.md`.

---

### Task 0: Repo skeleton, `config.yaml`, config loader

**Goal:** A `src/swot_eddy` package, an editable install, and `config.yaml` + `config.py` that load config and resolve Windows↔Colab paths.

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `src/swot_eddy/__init__.py`
- Create: `src/swot_eddy/config.py`
- Test: `tests/test_config.py`

**Acceptance Criteria:**
- [ ] `pip install -e .` succeeds; `import swot_eddy` works.
- [ ] `load_config(platform="windows")` and `("colab")` return the matching `data_root`/`out_root`.
- [ ] Region box reads back as 23/42/-82/-69.
- [ ] `variable_map(cfg)` returns the v3.0 names (`ssha_filtered`, `ugos_filtered`, …).
- [ ] `SWOT_EDDY_ENV` env var overrides platform detection.

**Verify:** `pytest tests/test_config.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "swot-eddy"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["pyyaml", "xarray", "netcdf4", "numpy"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src", "."]
```

- [ ] **Step 2: Create `config.yaml`**

```yaml
# SWOT Gulf Stream eddy-detection baseline — project config.
# Paths are resolved per platform in src/swot_eddy/config.py (Windows <-> Colab).

paths:
  windows:
    data_root: "C:/Users/jspier/Data/SWOT/SWOT_L3_GulfStream"
    out_root:  "C:/Users/jspier/SWOT-eddy-ml/out"
  colab:
    data_root: "/content/drive/MyDrive/SWOT_L3_GulfStream"
    out_root:  "/content/drive/MyDrive/swot-eddy-ml/out"

region:           # study box; lon negative = West, lat positive = North
  lat_min: 23
  lat_max: 42
  lon_min: -82
  lon_max: -69

product:
  product_version: "3.0"
  variables:      # version-aware variable-name map, keyed on product_version
    "3.0":
      ssha: "ssha_filtered"
      mdt: "mdt"
      ugos: "ugos_filtered"
      vgos: "vgos_filtered"
      quality_flag: "quality_flag"

tiling:
  tile_size: 128
  stride: 64
  pad_to: 128     # across-track NaN-pad target width

pet:
  pet_version: null        # pinned by `python -m swot_eddy.pet_version` (Task 1)
  params:
    shape_error: 85
    min_radius_km: 5
    min_amplitude_m: 0.01
    gap_reject_frac: 0.40
    gap_reject_frac_nadir: 0.60
    high_pass_km: null     # PET detection high-pass; swept in P2

# Example file for the inspection script (Task 2). On WSL use /mnt/c/... instead of C:/...
example_file: "C:/Users/jspier/Data/SWOT/SWOT_L3_GulfStream/cycle_002/SWOT_L3_LR_SSH_Expert_002_091_20230814T072907_20230814T082033_v3.0.nc"
```

- [ ] **Step 3: Create `src/swot_eddy/__init__.py`**

```python
"""SWOT Gulf Stream eddy-detection baseline."""
__version__ = "0.0.1"
```

- [ ] **Step 4: Write the failing tests — `tests/test_config.py`**

```python
import textwrap
import pytest
from swot_eddy import config as cfg_mod

SAMPLE = textwrap.dedent('''
paths:
  windows:
    data_root: "C:/data"
    out_root: "C:/out"
  colab:
    data_root: "/content/drive/MyDrive/data"
    out_root: "/content/drive/MyDrive/out"
region: {lat_min: 23, lat_max: 42, lon_min: -82, lon_max: -69}
product:
  product_version: "3.0"
  variables:
    "3.0": {ssha: ssha_filtered, mdt: mdt, ugos: ugos_filtered, vgos: vgos_filtered, quality_flag: quality_flag}
tiling: {tile_size: 128, stride: 64, pad_to: 128}
pet: {pet_version: null, params: {shape_error: 85}}
''')

@pytest.fixture
def sample_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(SAMPLE, encoding="utf-8")
    return p

def test_windows_paths(sample_config):
    c = cfg_mod.load_config(sample_config, platform="windows")
    assert c["data_root"] == "C:/data"
    assert c["out_root"] == "C:/out"
    assert c["platform"] == "windows"

def test_colab_paths(sample_config):
    c = cfg_mod.load_config(sample_config, platform="colab")
    assert c["data_root"] == "/content/drive/MyDrive/data"

def test_region_box(sample_config):
    c = cfg_mod.load_config(sample_config, platform="windows")
    assert (c["region"]["lat_min"], c["region"]["lat_max"]) == (23, 42)
    assert (c["region"]["lon_min"], c["region"]["lon_max"]) == (-82, -69)

def test_variable_map(sample_config):
    c = cfg_mod.load_config(sample_config, platform="windows")
    vm = cfg_mod.variable_map(c)
    assert vm["ssha"] == "ssha_filtered"
    assert vm["ugos"] == "ugos_filtered"

def test_unknown_platform_raises(sample_config):
    with pytest.raises(KeyError):
        cfg_mod.load_config(sample_config, platform="mars")

def test_env_override(monkeypatch):
    monkeypatch.setenv("SWOT_EDDY_ENV", "colab")
    assert cfg_mod.detect_platform() == "colab"
```

- [ ] **Step 5: Run tests to confirm they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: swot_eddy.config` / attributes missing).

- [ ] **Step 6: Implement `src/swot_eddy/config.py`**

```python
"""Load project config and resolve Windows<->Colab paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# repo_root/config.yaml  (this file is src/swot_eddy/config.py -> parents[2] = repo root)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def detect_platform() -> str:
    """Return 'windows' or 'colab'. SWOT_EDDY_ENV overrides; else auto-detect."""
    env = os.environ.get("SWOT_EDDY_ENV", "").strip().lower()
    if env in {"windows", "colab"}:
        return env
    if Path("/content/drive").exists():  # Colab mounts Drive here
        return "colab"
    return "windows"


def load_config(path: "os.PathLike | str | None" = None,
                platform: "str | None" = None) -> dict[str, Any]:
    """Load config.yaml and inject resolved data_root/out_root for the active platform."""
    cfg_path = Path(path) if path is not None else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    plat = platform or detect_platform()
    if plat not in cfg["paths"]:
        raise KeyError(f"Unknown platform {plat!r}; config has {list(cfg['paths'])}")
    cfg["platform"] = plat
    cfg["data_root"] = cfg["paths"][plat]["data_root"]
    cfg["out_root"] = cfg["paths"][plat]["out_root"]
    return cfg


def variable_map(cfg: dict[str, Any]) -> dict[str, str]:
    """Return the variable-name map for the configured product_version."""
    pv = cfg["product"]["product_version"]
    return cfg["product"]["variables"][pv]
```

- [ ] **Step 7: Install package and run tests to confirm pass**

Run: `pip install -e . && pytest tests/test_config.py -v`
Expected: PASS (6 passed).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml config.yaml src/swot_eddy/__init__.py src/swot_eddy/config.py tests/test_config.py
git commit -m "feat(p0): repo skeleton, config.yaml, platform-aware config loader"
```

---

### Task 1: Conda environment + `pyEddyTracker` install + pin `pet_version`

**Goal:** A reproducible env that installs `pyEddyTracker` per its docs, plus `python -m swot_eddy.pet_version` that writes the installed version into `config.yaml`.

**Files:**
- Create: `environment.yml`
- Create: `src/swot_eddy/pet_version.py`
- Test: `tests/test_pet_version.py`

**Acceptance Criteria:**
- [ ] `conda env create -f environment.yml` builds; `conda activate swot-eddy` then `python -c "import py_eddy_tracker"` succeeds (imports cleanly).
- [ ] `set_pet_version_in_text` replaces a `pet_version:` line (null or existing value) and raises if absent.
- [ ] `python -m swot_eddy.pet_version` writes the real installed version into `config.yaml` (`pet_version` no longer `null`).

**Verify:** `pytest tests/test_pet_version.py -v` → all pass; then (in the conda env) `python -m swot_eddy.pet_version` → prints `Recorded pet_version: <X> -> .../config.yaml`.

**Steps:**

- [ ] **Step 1: Create `environment.yml`**

```yaml
name: swot-eddy
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy
  - scipy
  - xarray
  - netcdf4
  - matplotlib
  - pyyaml
  - pytest
  - pip:
      - pyEddyTracker
```

- [ ] **Step 2: Write the failing tests — `tests/test_pet_version.py`**

```python
import pytest
from swot_eddy.pet_version import set_pet_version_in_text

def test_replaces_null():
    text = "pet:\n  pet_version: null\n  params: {}\n"
    out = set_pet_version_in_text(text, "3.6.1")
    assert 'pet_version: "3.6.1"' in out
    assert "null" not in out.split("\n")[1]

def test_replaces_existing_value():
    text = 'pet:\n  pet_version: "0.0.0"\n'
    out = set_pet_version_in_text(text, "3.6.1")
    assert 'pet_version: "3.6.1"' in out
    assert "0.0.0" not in out

def test_preserves_indentation():
    text = "pet:\n  pet_version: null\n"
    out = set_pet_version_in_text(text, "3.6.1")
    assert out.startswith("pet:\n  pet_version:")

def test_missing_line_raises():
    with pytest.raises(ValueError):
        set_pet_version_in_text("pet: {}\n", "3.6.1")
```

- [ ] **Step 3: Run tests to confirm they fail**

Run: `pytest tests/test_pet_version.py -v`
Expected: FAIL (`ModuleNotFoundError: swot_eddy.pet_version`).

- [ ] **Step 4: Implement `src/swot_eddy/pet_version.py`**

```python
"""Record the installed pyEddyTracker version into config.yaml (pet_version)."""
from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

# parents[2] = repo root (this file is src/swot_eddy/pet_version.py)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def get_pet_version() -> str:
    """Installed pyEddyTracker version (pip distribution name is 'pyEddyTracker')."""
    return metadata.version("pyEddyTracker")


def set_pet_version_in_text(text: str, version: str) -> str:
    """Replace the value on the `pet_version:` line, preserving indentation."""
    pattern = re.compile(r"^(\s*pet_version:\s*).*$", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError("No 'pet_version:' line found in config.yaml")
    return pattern.sub(rf'\g<1>"{version}"', text, count=1)


def main() -> int:
    version = get_pet_version()
    text = CONFIG_PATH.read_text(encoding="utf-8")
    CONFIG_PATH.write_text(set_pet_version_in_text(text, version), encoding="utf-8")
    print(f"Recorded pet_version: {version} -> {CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to confirm pass**

Run: `pytest tests/test_pet_version.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Build the env and pin the version (manual, in conda)**

Run (Windows Anaconda Prompt or any conda shell):
```bash
conda env create -f environment.yml
conda activate swot-eddy
pip install -e .
python -c "import py_eddy_tracker; print('import OK')"
python -m swot_eddy.pet_version
```
Expected: `import OK`, then `Recorded pet_version: <version> -> .../config.yaml`, and `config.yaml` shows `pet_version: "<version>"`.

> Note: pip distribution name is `pyEddyTracker`; the import module is `py_eddy_tracker`. Install docs: https://py-eddy-tracker.readthedocs.io/en/latest/installation.html

- [ ] **Step 7: Commit**

```bash
git add environment.yml src/swot_eddy/pet_version.py tests/test_pet_version.py config.yaml
git commit -m "feat(p0): conda env + pyEddyTracker install, pin pet_version into config"
```

---

### Task 2: Inspection script for a real SWOT `.nc` file

**Goal:** `swot_eddy.inspect_nc` summarizes a SWOT L3 file — variables/dims, key global attrs, `quality_flag` flag_values/flag_meanings, `temporality`, lon/lat extent — printed and optionally saved.

**Files:**
- Create: `src/swot_eddy/inspect_nc.py`
- Test: `tests/test_inspect_nc.py`

**Acceptance Criteria:**
- [ ] `inspect_dataset(ds)` returns dims, variable list, the selected attrs, `quality_flag` values+meanings, and lon/lat extents.
- [ ] CLI `python -m swot_eddy.inspect_nc --file <path> [--out <json>]` prints the summary and writes the JSON if `--out` given.
- [ ] Running on the real example file prints `product_version=3.0`, `temporality`, quality_flag meanings, and lon/lat extents within the study region.

**Verify:** `pytest tests/test_inspect_nc.py -v` → all pass; then `python -m swot_eddy.inspect_nc --file "<example>" --out out/inspect_002_091.json` prints a populated summary.

**Steps:**

- [ ] **Step 1: Write the failing tests — `tests/test_inspect_nc.py`** (synthetic netCDF, runs anywhere)

```python
import numpy as np
import xarray as xr
from swot_eddy.inspect_nc import inspect_dataset

def make_synthetic():
    nl, npix = 10, 69
    lat = np.linspace(23, 42, nl)[:, None] * np.ones((1, npix))
    lon = np.linspace(-82, -69, npix)[None, :] * np.ones((nl, 1))
    qf = xr.DataArray(
        np.zeros((nl, npix), "uint8"),
        dims=("num_lines", "num_pixels"),
        attrs={"flag_values": np.array([0, 1, 2], "uint8"),
               "flag_meanings": "good suspect bad"},
    )
    return xr.Dataset(
        {"ssha_filtered": (("num_lines", "num_pixels"), np.zeros((nl, npix))),
         "quality_flag": qf},
        coords={"latitude": (("num_lines", "num_pixels"), lat),
                "longitude": (("num_lines", "num_pixels"), lon)},
        attrs={"product_version": "3.0", "temporality": "forward",
               "time_coverage_start": "2023-08-14T07:29:07Z"},
    )

def test_dims_and_vars():
    s = inspect_dataset(make_synthetic())
    assert s["dims"]["num_pixels"] == 69
    assert "ssha_filtered" in s["variables"]
    assert s["variables"]["ssha_filtered"]["dims"] == ["num_lines", "num_pixels"]

def test_quality_flag():
    s = inspect_dataset(make_synthetic())
    assert s["quality_flag"]["flag_meanings"] == "good suspect bad"
    assert s["quality_flag"]["flag_values"] == [0, 1, 2]

def test_attrs_and_extent():
    s = inspect_dataset(make_synthetic())
    assert s["attrs"]["product_version"] == "3.0"
    assert s["attrs"]["temporality"] == "forward"
    assert 22.9 <= s["lat_extent"][0] and s["lat_extent"][1] <= 42.1
    assert s["lon_extent"][0] >= -82.1 and s["lon_extent"][1] <= -68.9
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_inspect_nc.py -v`
Expected: FAIL (`ModuleNotFoundError: swot_eddy.inspect_nc`).

- [ ] **Step 3: Implement `src/swot_eddy/inspect_nc.py`**

```python
"""Open a SWOT L3 .nc file and summarize its structure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import xarray as xr

_ATTR_KEYS = (
    "product_version", "cycle_number", "pass_number", "temporality",
    "time_coverage_start", "time_coverage_end",
)


def _as_int_list(v):
    if v is None:
        return None
    try:
        return [int(x) for x in v]
    except TypeError:
        return [int(v)]


def inspect_dataset(ds: xr.Dataset) -> dict[str, Any]:
    """Extract a structured summary from an open SWOT dataset."""
    summary: dict[str, Any] = {
        "dims": dict(ds.sizes),
        "variables": {
            name: {"dims": list(da.dims), "dtype": str(da.dtype)}
            for name, da in ds.variables.items()
        },
        "attrs": {k: ds.attrs.get(k) for k in _ATTR_KEYS},
        "quality_flag": None,
        "lon_extent": None,
        "lat_extent": None,
    }
    if "quality_flag" in ds:
        qf = ds["quality_flag"]
        summary["quality_flag"] = {
            "flag_values": _as_int_list(qf.attrs.get("flag_values")),
            "flag_meanings": qf.attrs.get("flag_meanings"),
        }
    for coord, key in (("longitude", "lon_extent"), ("latitude", "lat_extent")):
        if coord in ds:
            arr = ds[coord]
            summary[key] = [float(arr.min()), float(arr.max())]
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, default=str)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a SWOT L3 .nc file")
    parser.add_argument("--file", help="path to .nc (default: config example_file)")
    parser.add_argument("--out", help="optional path to write the JSON dump")
    args = parser.parse_args(argv)

    path = args.file
    if path is None:
        from swot_eddy.config import load_config
        path = load_config()["example_file"]

    with xr.open_dataset(path) as ds:
        summary = inspect_dataset(ds)

    text = format_summary(summary)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `pytest tests/test_inspect_nc.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run on the real file (manual acceptance — the P0 "done when")**

Run (WSL path shown; on Windows use the `C:/...` form):
```bash
python -m swot_eddy.inspect_nc \
  --file "/mnt/c/Users/jspier/Data/SWOT/SWOT_L3_GulfStream/cycle_002/SWOT_L3_LR_SSH_Expert_002_091_20230814T072907_20230814T082033_v3.0.nc" \
  --out out/inspect_002_091.json
```
Expected: prints a JSON summary showing `product_version: "3.0"`, a `temporality` value, `quality_flag` with `flag_values`+`flag_meanings`, and lon/lat extents; writes `out/inspect_002_091.json`.

- [ ] **Step 6: Commit**

```bash
git add src/swot_eddy/inspect_nc.py tests/test_inspect_nc.py
git commit -m "feat(p0): SWOT .nc inspection script with synthetic-data tests"
```

---

## P0 Done When
- `python -c "import py_eddy_tracker"` succeeds in the `swot-eddy` env.
- `config.yaml` `pet_version` is pinned to the installed version.
- The inspection dump prints for the real example file.
- `pytest -q` is green.

## Out of scope (later phases — do NOT build now)
Preprocessing (crop/pad/tile/inpaint), PET labeling + front tag, master table, YOLO dataset, splits, training, conformal, inference, seasonal analysis. See spec §14.
