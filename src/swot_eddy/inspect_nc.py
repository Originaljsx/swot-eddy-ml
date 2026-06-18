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
