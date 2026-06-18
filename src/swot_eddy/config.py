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
    return cfg


def variable_map(cfg: dict[str, Any]) -> dict[str, str]:
    """Return the variable-name map for the configured product_version."""
    pv = cfg["product"]["product_version"]
    return cfg["product"]["variables"][pv]
