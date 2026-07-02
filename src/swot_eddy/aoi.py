"""Gulf Stream AOI polygon (GS_AOI5): gate detections by point-in-polygon.

Crop / tile / inpaint run on the polygon's ENVELOPE (a rectangle); detections are
then filtered to the true (concave) polygon by their center point. Filter-only by
design -- masking pixels outside the polygon would let the biharmonic inpaint
fabricate field along the cut edge and invent eddies there. The envelope also feeds
crop_to_region unchanged, so preprocess never sees the polygon.

SWOT source longitude is 0..360; the GS_AOI5 polygon coords are -180..180, so
contains() normalizes any longitude to -180..180 before the test.

shapely is available transitively via cartopy (a project dependency) and is also
pinned explicitly in pyproject.toml.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from shapely.geometry import Point, shape
from shapely.ops import unary_union


class AOI:
    """One immutable area-of-interest polygon plus its bounding-box envelope."""

    def __init__(self, geom):
        self.geom = geom
        minx, miny, maxx, maxy = geom.bounds
        self._env = {"lon_min": minx, "lat_min": miny, "lon_max": maxx, "lat_max": maxy}

    def contains(self, lon: "float | None", lat: "float | None") -> bool:
        """True if (lon, lat) lies inside the polygon. lon may be 0..360 or -180..180."""
        if lon is None or lat is None:
            return False
        lon = ((lon + 180) % 360) - 180                 # -> -180..180 to match polygon coords
        return bool(self.geom.contains(Point(lon, lat)))

    def envelope_region(self) -> dict[str, float]:
        """Bounding box as a region dict (lon_min/lat_min/lon_max/lat_max), -180..180 lon.
        Drop-in for the old config region: feeds crop_to_region / region_mask unchanged."""
        return dict(self._env)


def _load_geojson(path: Path):
    """Union every geometry in a GeoJSON (GeometryCollection / FeatureCollection /
    Feature / bare geometry) into one shapely polygon-ish geometry."""
    with open(path, "r", encoding="utf-8") as fh:
        gj = json.load(fh)
    t = gj.get("type")
    if t == "GeometryCollection":
        geoms = [shape(g) for g in gj["geometries"]]
    elif t == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in gj["features"]]
    elif t == "Feature":
        geoms = [shape(gj["geometry"])]
    else:
        geoms = [shape(gj)]
    return unary_union(geoms)


def aoi_path(cfg: dict[str, Any]) -> Path:
    """Resolve the AOI geojson path for the active platform (windows/wsl/colab)."""
    plat = cfg.get("platform") or "wsl"
    return Path(cfg["aoi"][plat])


@lru_cache(maxsize=4)
def _load_cached(path_str: str) -> AOI:
    return AOI(_load_geojson(Path(path_str)))


def load_aoi(cfg: dict[str, Any]) -> AOI:
    """Load (and cache, per path) the AOI polygon referenced by cfg['aoi']."""
    return _load_cached(str(aoi_path(cfg)))
