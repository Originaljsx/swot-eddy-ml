import numpy as np
from swot_eddy import preprocess as pp
from tests._synth import make_pass, VARMAP


def test_channels_shape_and_names():
    ds = make_pass(n_lines=50, region_lines=(0, 50))
    stack, names, cov = pp.compute_channels(ds, VARMAP)
    assert stack.shape == (2, 50, 69)
    assert names == ["ssha", "adt"]


def test_adt_is_ssha_unedited_plus_mdt():
    ds = make_pass(n_lines=20, region_lines=(0, 20))
    stack, names, cov = pp.compute_channels(ds, VARMAP)
    base = ds["ssha_unedited"].values
    mdt = ds["mdt"].values
    present = np.isfinite(mdt)
    assert np.allclose(stack[1][present], (base + mdt)[present])  # adt = ssha_unedited + mdt
    assert np.all(np.isnan(stack[1][~present]))                   # NaN where mdt missing
    # channel 0 is ssha_filtered (the detection field), NOT ssha_unedited
    assert np.allclose(stack[0], ds["ssha_filtered"].values)


def test_mdt_coverage_fraction():
    ds = make_pass(n_lines=10, region_lines=(0, 10))
    _, _, cov = pp.compute_channels(ds, VARMAP)
    expected = float(np.isfinite(ds["mdt"].values).mean())
    assert abs(cov - expected) < 1e-9


def test_quality_mask_sets_bad_pixels_nan():
    # quality_flag=5 block; quality_max=3 -> those pixels become NaN on both channels
    ds = make_pass(n_lines=20, region_lines=(0, 20), bad_block=(2, 5, 20, 30))
    stack, _, _ = pp.compute_channels(ds, VARMAP, quality_max=3)
    assert np.all(np.isnan(stack[0][2:5, 20:30]))   # ssha masked
    assert np.all(np.isnan(stack[1][2:5, 20:30]))   # adt masked
    # a good pixel (qf=0) outside the block is finite on ssha
    assert np.isfinite(stack[0][10, 40])


def test_no_quality_mask_when_quality_max_none():
    ds = make_pass(n_lines=20, region_lines=(0, 20), bad_block=(2, 5, 20, 30))
    stack, _, _ = pp.compute_channels(ds, VARMAP, quality_max=None)
    assert np.all(np.isfinite(stack[0][2:5, 20:30]))  # not masked
