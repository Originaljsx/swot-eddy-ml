"""Unit tests for src/swot_eddy/features.py (synthetic fields, no SWOT data needed)."""
import numpy as np
import pytest

from swot_eddy import features as ft


@pytest.fixture()
def linear_front():
    """ADT-like field stepping from 0.0 (south) to 0.6 (north) across lat=30: the 0.30 m
    contour is the lat=30 line. Grid: lon -80..-79, lat 29..31, ~2 km spacing."""
    lat = np.linspace(29, 31, 100)
    lon = np.linspace(-80, -79, 50)
    lon2d, lat2d = np.meshgrid(lon, lat)
    field = 0.3 + 0.3 * np.tanh((lat2d - 30.0) * 10)
    return field, lon2d, lat2d


def test_front_vertices_recovers_line(linear_front):
    field, lon2d, lat2d = linear_front
    fv = ft.front_vertices(field, lon2d, lat2d, 0.30, min_vertices=5)
    assert len(fv) > 10
    assert np.allclose(fv[:, 1], 30.0, atol=0.05)          # front sits on lat=30


def test_signed_front_dist_sign_and_magnitude(linear_front):
    field, lon2d, lat2d = linear_front
    fv = ft.front_vertices(field, lon2d, lat2d, 0.30, min_vertices=5)
    # point 0.5 deg north (high side) and 0.5 deg south (low side) of the front
    d, miss = ft.signed_front_dist_km([-79.5, -79.5], [30.5, 29.5], [0.6, 0.0], 0.30, fv)
    assert not miss.any()
    assert d[0] > 0 and d[1] < 0                            # + high/offshore, - low/inshore
    assert abs(d[0]) == pytest.approx(0.5 * ft.KM_PER_DEG_LAT, rel=0.05)


def test_signed_front_dist_missing_front():
    d, miss = ft.signed_front_dist_km([-79.5], [30.0], [0.4], 0.30, np.empty((0, 2)))
    assert miss.all() and np.isnan(d).all()


def test_front_vertices_handles_nan_gap(linear_front):
    field, lon2d, lat2d = linear_front
    field = field.copy()
    field[:, 20:25] = np.nan                                # nadir-gap style hole
    fv = ft.front_vertices(field, lon2d, lat2d, 0.30, min_vertices=5)
    assert len(fv) > 10                                     # contour survives, just breaks


def test_sample_nearest_finite_fallback():
    f = np.full((9, 9), np.nan)
    f[2, 2] = 7.0
    val, dist = ft.sample_nearest_finite(f, 4, 4)
    assert val == 7.0 and dist == pytest.approx(np.hypot(2, 2))
    val, dist = ft.sample_nearest_finite(np.full((9, 9), np.nan), 4, 4, max_px=2)
    assert np.isnan(val) and np.isnan(dist)


def test_find_granule_disambiguates_cycle_pass(tmp_path):
    """pass number == cycle number must not match every file (the *_048_* glob bug)."""
    d = tmp_path / "cycle_048"
    d.mkdir()
    (d / "SWOT_L3_LR_SSH_Expert_048_007_20260328T020112_x_v3.0.nc").touch()
    (d / "SWOT_L3_LR_SSH_Expert_048_048_20260329T131031_x_v3.0.nc").touch()
    f = ft.find_granule(tmp_path, "cycle_048", "048")
    assert f is not None and "_048_048_" in f.name
    f = ft.find_granule(tmp_path, "cycle_048", "007")
    assert f is not None and "_048_007_" in f.name
    assert ft.find_granule(tmp_path, "cycle_048", "999") is None
