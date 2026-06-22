"""Tests for the PET detection step.

The integration test feeds two clean Gaussian eddies (one positive=anticyclone, one
negative=cyclone) at known, asymmetric pixel locations so a transposed (X,Y) axis
mapping would be caught.
"""
import datetime as dt

import numpy as np
import pytest

from swot_eddy import detect as dt_mod
from swot_eddy.preprocess import Tile

CFG = {"pet": {"params": {"shape_error": 85, "min_radius_km": 5, "min_amplitude_m": 0.01,
                          "gap_reject_frac": 0.40, "gap_reject_frac_nadir": 0.60,
                          "pixel_limit_min": 3, "pixel_limit_max": 2000,
                          "high_pass_km": None}}}
DATE = dt.datetime(2023, 8, 14)


def _gauss(h, w, i0, j0, amp, sig):
    ii, jj = np.mgrid[0:h, 0:w]
    return amp * np.exp(-(((ii - i0) ** 2 + (jj - j0) ** 2) / (2.0 * sig ** 2)))


def _synth_tile(pad_left=29, nadir_cols=None):
    """A 128x128 tile: anticyclone (+) at (40,80), cyclone (-) at (90,45). No NaNs.
    Optionally mark `nadir_cols` as fully inpainted to exercise the nadir flag."""
    h = w = 128
    ssha = _gauss(h, w, 40, 80, 0.30, 7.0) - _gauss(h, w, 90, 45, 0.30, 7.0)
    data = np.stack([ssha, ssha + 0.5])           # adt channel irrelevant to detection
    mask = np.zeros((2, h, w), dtype=bool)
    if nadir_cols is not None:
        mask[:, :, nadir_cols] = True
    return Tile(data=data, inpaint_mask=mask, line_start=0, pad_left=pad_left,
                channels=("ssha", "adt"))


def _coords(h=128, w=69):
    """cropped_lon/lat (0..360) for one tile worth of lines; realistic Gulf Stream values."""
    lat = np.broadcast_to(np.linspace(30.0, 32.3, h)[:, None], (h, w)).copy()
    lon = np.broadcast_to(np.linspace(281.0, 283.0, w)[None, :], (h, w)).copy()
    return lon, lat


def test_detect_recovers_polarity_and_axis():
    tile = _synth_tile()
    clon, clat = _coords()
    dets = dt_mod.detect_tile(tile, 0, CFG, clon, clat, DATE)
    assert len(dets) >= 2
    anti = [d for d in dets if d.polarity == "anticyclonic"]
    cyc = [d for d in dets if d.polarity == "cyclonic"]
    assert anti and cyc
    # anticyclone center near (40, 80); a transposed axis would land near (80, 40)
    a = min(anti, key=lambda d: abs(d.center_line - 40) + abs(d.center_col - 80))
    assert abs(a.center_line - 40) <= 8 and abs(a.center_col - 80) <= 8
    c = min(cyc, key=lambda d: abs(d.center_line - 90) + abs(d.center_col - 45))
    assert abs(c.center_line - 90) <= 8 and abs(c.center_col - 45) <= 8
    # geo mapping populated (centers are over real, non-padded columns)
    assert a.center_lon is not None and 281.0 <= a.center_lon <= 283.0
    assert a.radius_km >= 5 and a.amplitude_m > 0          # radius_km = R_max (radius_s)


def test_gap_reject_fires_when_contour_over_gaps():
    tile = _synth_tile()
    # inpaint a wide block covering the anticyclone (center 40,80) so its speed contour
    # sits mostly on gaps -> gap_frac > 0.40 -> gap_reject
    tile.inpaint_mask[0, 25:56, 65:96] = True
    clon, clat = _coords()
    dets = dt_mod.detect_tile(tile, 0, CFG, clon, clat, DATE)
    a = min((d for d in dets if d.polarity == "anticyclonic"),
            key=lambda d: abs(d.center_line - 40) + abs(d.center_col - 80))
    assert a.gap_frac > 0.40 and a.gap_reject is True


def test_nadir_center_uses_relaxed_threshold():
    nadir = list(range(60, 68))                    # central across-track band, fully filled
    tile = _synth_tile(nadir_cols=nadir)
    clon, clat = _coords()
    dets = dt_mod.detect_tile(tile, 0, CFG, clon, clat, DATE)
    # the anticyclone at col 80 is away from the band -> not flagged as nadir center
    a = min((d for d in dets if d.polarity == "anticyclonic"),
            key=lambda d: abs(d.center_line - 40) + abs(d.center_col - 80))
    assert a.center_in_nadir is False
    assert all(isinstance(d.gap_reject, bool) for d in dets)


def _det(polarity, line, col, r=10.0, amp=0.2, edge=50, tile=0):
    return dt_mod.Detection(polarity=polarity, center_line=line, center_col=col,
                            center_lon=282.0, center_lat=31.0, radius_km=r, radius_eff_km=r * 1.3,
                            amplitude_m=amp, contour_pix=[], contour_lonlat=[],
                            gap_frac=0.0, gap_reject=False, center_in_nadir=False,
                            source_tile=tile, edge_dist=edge)


def test_dedup_merges_overlapping_same_polarity():
    # two near-identical anticyclones (centers 3 px apart, R=10km -> 5px) -> one survives
    a1 = _det("anticyclonic", 100, 80, edge=60, amp=0.25)
    a2 = _det("anticyclonic", 102, 81, edge=20, amp=0.20)
    far = _det("anticyclonic", 300, 80, edge=50)
    cyc = _det("cyclonic", 101, 80, edge=50)           # opposite polarity, never merged
    kept = dt_mod.dedup_detections([a1, a2, far, cyc], CFG)
    anti = [d for d in kept if d.polarity == "anticyclonic"]
    assert len(anti) == 2                               # merged pair -> 1, plus the far one
    # the more interior (edge_dist 60) one wins the merge
    survivor = min(anti, key=lambda d: abs(d.center_line - 100))
    assert survivor.edge_dist == 60
    assert sum(d.polarity == "cyclonic" for d in kept) == 1


def test_detect_pass_summary_counts():
    tile = _synth_tile()
    clon, clat = _coords()
    dets, summary = dt_mod.detect_pass([tile], CFG, clon, clat, DATE)
    assert summary["n_dedup"] == len(dets)
    assert summary["n_anticyclonic"] + summary["n_cyclonic"] == len(dets)
    assert summary["n_raw"] >= summary["n_dedup"]
