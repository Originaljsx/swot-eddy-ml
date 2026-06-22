"""Synthetic SWOT-L3-like passes for preprocessing tests (no real data needed)."""
import numpy as np
import xarray as xr

NUM_PIX = 69

# Variable-name map matching config.yaml product.variables["3.0"].
VARMAP = {
    "ssha": "ssha_filtered",
    "ssha_adt_base": "ssha_unedited",
    "mdt": "mdt",
    "ugos": "ugos_filtered",
    "vgos": "vgos_filtered",
    "quality_flag": "quality_flag",
}
REGION = {"lat_min": 23, "lat_max": 42, "lon_min": -82, "lon_max": -69}


def make_pass(n_lines=300, region_lines=(100, 200), seed=0, bad_block=None):
    """A global-ish pass: only the rows in `region_lines` fall inside the Gulf Stream box
    (lat 23..42, lon 278..291 in 0-360). MDT has a missing band to mimic real gaps.
    `bad_block=(r0, r1, c0, c1)` sets quality_flag=5 (local_outliers) in that block."""
    rng = np.random.default_rng(seed)
    lat = np.full((n_lines, NUM_PIX), 10.0)   # outside region by default (lat 10)
    lon = np.full((n_lines, NUM_PIX), 200.0)  # outside region by default (lon 200)
    a, b = region_lines
    if b > a:
        lat[a:b, :] = np.linspace(24.0, 41.0, b - a)[:, None]
        lon[a:b, :] = np.linspace(279.0, 290.0, NUM_PIX)[None, :]  # 0..360, inside 278..291
    ssha = rng.normal(0.0, 0.1, (n_lines, NUM_PIX))
    ssha_unedited = ssha + 0.01            # distinct from ssha_filtered (ADT base)
    mdt = np.full((n_lines, NUM_PIX), 0.5)
    mdt[:, :10] = np.nan                    # ~14% missing band (mimics real MDT gaps)
    qf = np.zeros((n_lines, NUM_PIX), "uint8")
    if bad_block is not None:
        r0, r1, c0, c1 = bad_block
        qf[r0:r1, c0:c1] = 5                # local_outliers > quality_max(3)
    return xr.Dataset(
        {
            "ssha_filtered": (("num_lines", "num_pixels"), ssha),
            "ssha_unedited": (("num_lines", "num_pixels"), ssha_unedited),
            "mdt": (("num_lines", "num_pixels"), mdt),
            "quality_flag": (("num_lines", "num_pixels"), qf),
        },
        coords={
            "latitude": (("num_lines", "num_pixels"), lat),
            "longitude": (("num_lines", "num_pixels"), lon),
        },
    )
