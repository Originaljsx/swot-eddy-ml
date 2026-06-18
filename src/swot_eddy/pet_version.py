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
