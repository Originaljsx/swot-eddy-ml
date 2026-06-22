import numpy as np
from swot_eddy import preprocess as pp
from tests._synth import make_pass, REGION, VARMAP

CFG = {"region": REGION,
       "tiling": {"tile_size": 128, "stride": 64, "pad_to": 128},
       "product": {"product_version": "3.0", "quality_max": 3,
                   "variables": {"3.0": VARMAP}}}


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
