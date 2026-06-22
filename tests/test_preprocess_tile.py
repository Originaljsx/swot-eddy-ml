import numpy as np
from swot_eddy import preprocess as pp


def test_pad_across_track_to_128():
    stack = np.zeros((2, 40, 69))
    padded, pad_left = pp.pad_across_track(stack, 128)
    assert padded.shape == (2, 40, 128)
    assert pad_left == 29
    assert np.all(np.isnan(padded[:, :, :29]))
    assert np.all(padded[:, :, 29:98] == 0.0)
    assert np.all(np.isnan(padded[:, :, 98:]))


def test_tile_starts_cover_with_final_anchor():
    assert pp.tile_starts(100, 128, 64) == [0]          # shorter than a tile
    starts = pp.tile_starts(300, 128, 64)
    assert starts[0] == 0
    assert starts[-1] == 300 - 128                       # final window anchored at end
    # consecutive starts never exceed the stride
    assert all(b - a <= 64 for a, b in zip(starts, starts[1:]))


def test_tile_along_track_shapes_and_tail_pad():
    padded = np.zeros((2, 200, 128))
    tiles = pp.tile_along_track(padded, 128, 64)
    assert all(w.shape == (2, 128, 128) for _, w in tiles)
    # a 150-line band: last tile must still be (2,128,128)
    padded2 = np.full((2, 150, 128), 1.0)
    tiles2 = pp.tile_along_track(padded2, 128, 64)
    assert all(w.shape == (2, 128, 128) for _, w in tiles2)
