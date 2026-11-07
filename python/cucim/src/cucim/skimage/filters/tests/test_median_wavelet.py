# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy import ndimage as cndi

from cucim.skimage.filters import _median_wavelet as mw, median

ALL_DTYPES = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
]


def _random_image(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        # include repeated values, zeros and infinities
        img = rng.standard_normal(shape).round(2).astype(dtype)
        img[rng.random(shape) < 0.01] = np.inf
        img[rng.random(shape) < 0.01] = -np.inf
        img[rng.random(shape) < 0.05] = 0
    else:
        info = np.iinfo(dtype)
        hi = min(info.max, 2**40)
        img = rng.integers(max(info.min, -hi), hi, shape, dtype=dtype)
    return cp.asarray(img)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("footprint_shape", [(3, 3), (5, 7), (9, 3)])
def test_wavelet_matches_sorting_dtypes(dtype, footprint_shape):
    img = _random_image((45, 61), dtype)
    footprint = np.ones(footprint_shape, dtype=bool)
    expected = median(img, footprint, algorithm="sorting")
    out = median(img, footprint, algorithm="wavelet_matrix")
    assert out.dtype == img.dtype
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "mode,cval",
    [
        ("nearest", 0.0),
        ("reflect", 0.0),
        ("mirror", 0.0),
        ("wrap", 0.0),
        ("constant", 0.0),
        ("constant", 100.0),
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
def test_wavelet_modes_and_cval(mode, cval, dtype):
    img = _random_image((40, 50), dtype)
    footprint = np.ones((7, 5), dtype=bool)
    kwargs = dict(mode=mode, cval=cval)
    expected = median(img, footprint, algorithm="sorting", **kwargs)
    out = median(img, footprint, algorithm="wavelet_matrix", **kwargs)
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "shape",
    [(3, 3), (5, 70), (70, 5), (33, 31), (130, 2100), (1100, 17)],
)
def test_wavelet_image_shapes(shape):
    # larger shapes use several construction blocks
    img = _random_image(shape, np.uint8)
    footprint = np.ones((3, 5), dtype=bool)
    expected = median(img, footprint, algorithm="sorting", mode="reflect")
    out = median(img, footprint, algorithm="wavelet_matrix", mode="reflect")
    cp.testing.assert_array_equal(out, expected)


def test_wavelet_constant_and_extreme_values():
    for value in (0, 255):
        img = cp.full((20, 30), value, dtype=np.uint8)
        out = median(img, (5, 5), algorithm="wavelet_matrix")
        cp.testing.assert_array_equal(out, img)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
def test_wavelet_rank(dtype):
    img = _random_image((36, 44), dtype)
    ry, rx = 2, 3
    window_size = (2 * ry + 1) * (2 * rx + 1)
    for rank in (0, 1, window_size // 4, window_size // 2, window_size - 1):
        out = mw._median_wavelet_filter(img, ry, rx, mode="nearest", rank=rank)
        expected = cndi.rank_filter(
            img, rank, size=(2 * ry + 1, 2 * rx + 1), mode="nearest"
        )
        cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("rank", [-1, 9])
def test_wavelet_rank_out_of_range(rank):
    img = _random_image((10, 10), np.uint8)
    with pytest.raises(ValueError, match="rank must be"):
        mw._median_wavelet_filter(img, 1, 1, rank=rank)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.float16])
def test_keys_are_monotonic_and_invertible(dtype):
    info = np.dtype(dtype)
    # every bit pattern of the 8/16-bit type
    bits = cp.arange(2 ** (8 * info.itemsize), dtype=f"u{info.itemsize}")
    values = bits.view(dtype)
    if info.kind == "f":
        values = values[~cp.isnan(values)]
    keys = mw._to_keys(values)
    # values are non-decreasing when ordered by their keys
    ordered = values[cp.argsort(keys)]
    assert bool(cp.all(ordered[1:] >= ordered[:-1]))
    back = mw._from_keys(keys, dtype)
    cp.testing.assert_array_equal(back == values, cp.ones(values.shape, bool))


def test_wavelet_footprint_checks():
    img = _random_image((30, 30), np.uint8)
    # footprints with zeros are not supported by the wavelet matrix algorithm
    footprint = np.ones((15, 15), dtype=bool)
    footprint[0, 0] = False
    with pytest.raises(ValueError, match="wavelet_matrix"):
        median(img, footprint, algorithm="wavelet_matrix")
    # ...but 'auto' falls back to another algorithm
    expected = median(img, footprint, algorithm="sorting")
    cp.testing.assert_array_equal(median(img, footprint), expected)

    # even footprint sizes and 3D images are not supported either
    with pytest.raises(ValueError, match="wavelet_matrix"):
        median(img, (4, 4), algorithm="wavelet_matrix")
    with pytest.raises(ValueError, match="wavelet_matrix"):
        median(
            cp.zeros((5, 5, 5), np.uint8), (3, 3, 3), algorithm="wavelet_matrix"
        )


def test_wavelet_unsupported_mode():
    img = _random_image((40, 40), np.uint8)
    footprint = np.ones((15, 15), dtype=bool)
    with pytest.raises(ValueError, match="Unsupported mode"):
        median(img, footprint, mode="grid-wrap", algorithm="wavelet_matrix")


def test_wavelet_out_argument():
    img = _random_image((50, 50), np.uint8)
    footprint = np.ones((7, 7), dtype=bool)
    expected = median(img, footprint, algorithm="sorting")
    out = cp.empty_like(img)
    ret = median(img, footprint, out=out, algorithm="wavelet_matrix")
    assert ret is out
    cp.testing.assert_array_equal(out, expected)


def test_wavelet_non_contiguous_input():
    img = _random_image((60, 40), np.uint16).T  # F-ordered view
    footprint = np.ones((5, 5), dtype=bool)
    expected = median(img, footprint, algorithm="sorting")
    out = median(img, footprint, algorithm="wavelet_matrix")
    cp.testing.assert_array_equal(out, expected)


def test_wavelet_width_limit():
    img = cp.zeros((3, mw.WM_MAX_PADDED_WIDTH - 1), dtype=np.uint8)
    ok, reason = mw._can_use_wavelet_matrix(img, (3, 3))
    assert not ok and "width" in reason
