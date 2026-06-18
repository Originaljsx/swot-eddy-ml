# SWOT Gulf Stream Eddy Detection

Machine-learning detection of eddies (esp. frontal eddies) in SWOT L3 LR SSH Expert v3.0
over the Gulf Stream. Baseline pipeline: py-eddy-tracker labels → YOLO. See the design
spec in `docs/superpowers/specs/` and the phase plans in `docs/superpowers/plans/`.

## Setup (verified on WSL / Linux, Python 3.12, no conda)

```bash
cd /mnt/c/Users/jspier/SWOT-eddy-ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# py-eddy-tracker: install GitHub master, NOT PyPI.
# PyPI pyEddyTracker (3.6.1) pins numpy<1.23 / numba<0.56 / netCDF4<1.6 and cannot
# install on Python >=3.12 (forces a source build of an ancient netCDF4 -> needs HDF5).
python -m pip install "git+https://github.com/AntSimi/py-eddy-tracker.git" -e . pytest
```

This resolves to a working, all-wheel stack (numpy auto-downgraded to satisfy numba):
`pyEddyTracker (master) · numpy 1.26.4 · numba 0.65.1 · netCDF4 1.7.4 · matplotlib 3.7.5
· xarray 2026.4.0 · scipy · zarr<3 · opencv · pint`.

> Native **Windows** (cmd/PowerShell, Python 3.12): same commands but
> `py -3.12 -m venv .venv` and `.venv\Scripts\activate`.
> **Colab / conda** users: `environment.yml` (also installs PET from GitHub).

## Verify the install (P0 acceptance)

```bash
python -c "import py_eddy_tracker; print('PET OK')"   # imports cleanly
python -m swot_eddy.pet_version                        # pins pet_version into config.yaml
python -m swot_eddy.inspect_nc --out out/inspect_002_091.json   # dumps a real .nc
pytest -q                                              # all tests green
```

`config.yaml` auto-detects the platform (windows / wsl / colab) and resolves data paths;
override with `SWOT_EDDY_ENV=wsl|windows|colab`.
