"""Load project config and resolve Windows<->Colab paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# repo_root/config.yaml  (this file is src/swot_eddy/config.py -> parents[2] = repo root)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def detect_platform() -> str:
    """Return 'windows', 'colab', or 'wsl'. SWOT_EDDY_ENV overrides; else auto-detect."""
    env = os.environ.get("SWOT_EDDY_ENV", "").strip().lower()
    if env in {"windows", "colab", "wsl"}:
        return env
    if Path("/content/drive").exists():  # Colab mounts Drive here
        return "colab"
    if Path("/mnt/c").exists():          # WSL exposes the Windows C: drive here
        return "wsl"
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
    # Single source of truth = the GS_AOI5 polygon. Derive the crop envelope from it and
    # OVERWRITE cfg['region'] so every existing crop_to_region/region_mask caller uses the
    # AOI5 envelope unchanged. Detections are filtered to the true polygon separately via
    # aoi.load_aoi(cfg).contains(...). Falls back to the yaml region if cfg['aoi'] is absent.
    if "aoi" in cfg:
        from swot_eddy.aoi import load_aoi
        cfg["region"] = load_aoi(cfg).envelope_region()
    return cfg


def variable_map(cfg: dict[str, Any]) -> dict[str, str]:
    """Return the variable-name map for the configured product_version."""
    pv = cfg["product"]["product_version"]
    return cfg["product"]["variables"][pv]
