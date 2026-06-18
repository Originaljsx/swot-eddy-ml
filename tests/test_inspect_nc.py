import numpy as np
import xarray as xr
from swot_eddy.inspect_nc import inspect_dataset

def make_synthetic():
    nl, npix = 10, 69
    lat = np.linspace(23, 42, nl)[:, None] * np.ones((1, npix))
    lon = np.linspace(-82, -69, npix)[None, :] * np.ones((nl, 1))
    qf = xr.DataArray(
        np.zeros((nl, npix), "uint8"),
        dims=("num_lines", "num_pixels"),
        attrs={"flag_values": np.array([0, 1, 2], "uint8"),
               "flag_meanings": "good suspect bad"},
    )
    return xr.Dataset(
        {"ssha_filtered": (("num_lines", "num_pixels"), np.zeros((nl, npix))),
         "quality_flag": qf},
        coords={"latitude": (("num_lines", "num_pixels"), lat),
                "longitude": (("num_lines", "num_pixels"), lon)},
        attrs={"product_version": "3.0", "temporality": "forward",
               "time_coverage_start": "2023-08-14T07:29:07Z"},
    )

def test_dims_and_vars():
    s = inspect_dataset(make_synthetic())
    assert s["dims"]["num_pixels"] == 69
    assert "ssha_filtered" in s["variables"]
    assert s["variables"]["ssha_filtered"]["dims"] == ["num_lines", "num_pixels"]

def test_quality_flag():
    s = inspect_dataset(make_synthetic())
    assert s["quality_flag"]["flag_meanings"] == "good suspect bad"
    assert s["quality_flag"]["flag_values"] == [0, 1, 2]

def test_attrs_and_extent():
    s = inspect_dataset(make_synthetic())
    assert s["attrs"]["product_version"] == "3.0"
    assert s["attrs"]["temporality"] == "forward"
    assert 22.9 <= s["lat_extent"][0] and s["lat_extent"][1] <= 42.1
    assert s["lon_extent"][0] >= -82.1 and s["lon_extent"][1] <= -68.9
