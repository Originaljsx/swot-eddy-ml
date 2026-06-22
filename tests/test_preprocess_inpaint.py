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
