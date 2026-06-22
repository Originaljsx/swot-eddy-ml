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
