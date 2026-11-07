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


@pytest.mark.parametrize(
    "dtype,shape", [(np.uint8, (4096, 4096)), (np.uint16, (300, 20000))]
)
def test_wavelet_large_images(dtype, shape):
    # exercises construction grids of hundreds of blocks
    img = _random_image(shape, dtype)
    footprint = np.ones((5, 3), dtype=bool)
    expected = median(img, footprint, algorithm="sorting")
    out = median(img, footprint, algorithm="wavelet_matrix")
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

    # even footprint sizes and 4D images are not supported either
    with pytest.raises(ValueError, match="wavelet_matrix"):
        median(img, (4, 4), algorithm="wavelet_matrix")
    with pytest.raises(ValueError, match="wavelet_matrix"):
        median(
            cp.zeros((5, 5, 5, 5), np.uint8),
            (3, 3, 3, 3),
            algorithm="wavelet_matrix",
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


def test_auto_skips_wavelet_when_memory_is_short(monkeypatch):
    img = _random_image((60, 60), np.uint8)
    footprint = np.ones((15, 15), dtype=bool)
    expected = median(img, footprint, algorithm="sorting")

    def fail(*args, **kwargs):
        raise AssertionError("wavelet matrix should not be used")

    monkeypatch.setattr(
        "cucim.skimage.filters._median._has_enough_memory",
        lambda image, footprint_shape: False,
    )
    monkeypatch.setattr(
        "cucim.skimage.filters._median._median_wavelet_filter", fail
    )
    cp.testing.assert_array_equal(median(img, footprint), expected)


def test_memory_estimate_is_close_to_actual():
    pool = cp.get_default_memory_pool()
    img = _random_image((512, 512), np.uint16)
    mw._median_wavelet_filter(img, 3, 3)  # compile kernels
    pool.free_all_blocks()
    out = mw._median_wavelet_filter(img, 3, 3)
    actual = pool.total_bytes()
    estimate = mw._estimate_memory_usage(512, 512, img.dtype, 3, 3)
    assert 0.8 * actual <= estimate <= 1.2 * actual
    del out


@pytest.mark.parametrize(
    "dtype,footprint_edge,expect_wavelet",
    [
        (np.uint8, 9, False),  # 81 elements
        (np.uint8, 11, True),  # 121 elements
        (np.uint16, 11, True),
        (np.float32, 11, False),  # types wider than 16 bits need > 150
        (np.float32, 13, True),  # 169 elements
    ],
)
def test_auto_algorithm_threshold(
    monkeypatch, dtype, footprint_edge, expect_wavelet
):
    calls = []
    original = mw._median_wavelet_filter

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "cucim.skimage.filters._median._median_wavelet_filter", spy
    )
    img = _random_image((40, 40), dtype)
    footprint = np.ones((footprint_edge,) * 2, dtype=bool)
    out = median(img, footprint)
    assert bool(calls) == expect_wavelet
    cp.testing.assert_array_equal(
        out, median(img, footprint, algorithm="sorting")
    )


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("mode", ["reflect", "constant", "wrap"])
def test_slabs_match_single_slab(dtype, mode):
    img = _random_image((300, 70), dtype)
    full = mw._median_wavelet_filter(
        img, 4, 3, mode=mode, cval=1, slab_rows=300
    )
    # slab heights that do and do not divide the height (299: two slabs)
    for rows in (64, 100, 299):
        slabbed = mw._median_wavelet_filter(
            img, 4, 3, mode=mode, cval=1, slab_rows=rows
        )
        cp.testing.assert_array_equal(slabbed, full)


def test_slabs_with_other_rank_and_tiny_slab():
    img = _random_image((50, 40), np.uint16)
    full = mw._median_wavelet_filter(img, 2, 2, rank=3, slab_rows=50)
    slabbed = mw._median_wavelet_filter(img, 2, 2, rank=3, slab_rows=1)
    cp.testing.assert_array_equal(slabbed, full)


def test_plan_slab_rows():
    args = (4096, 4096, np.uint8, 15, 15)
    whole = mw._slab_memory_estimate(4096, *args)
    # enough memory: one slab
    assert mw._plan_slab_rows(*args, budget=whole) == 4096
    # less memory: smaller slabs that fit the budget, never below the minimum
    prev = 4096
    for fraction in (0.5, 0.2, 0.08):
        rows = mw._plan_slab_rows(*args, budget=fraction * whole)
        assert mw.WM_MIN_SLAB_ROWS <= rows < prev
        assert mw._slab_memory_estimate(rows, *args) <= fraction * whole
        prev = rows
    # not even the smallest slab fits
    assert mw._plan_slab_rows(*args, budget=1024) is None


def test_default_slabbing_uses_memory_fraction(monkeypatch):
    img = _random_image((400, 64), np.uint8)
    expected = mw._median_wavelet_filter(img, 5, 5, slab_rows=400)
    whole = mw._estimate_memory_usage(400, 64, img.dtype, 5, 5)
    # available memory such that the budget is a quarter of the whole image
    monkeypatch.setattr(
        mw, "_available_memory", lambda: 0.25 * whole / mw.WM_MEMORY_FRACTION
    )
    rows = mw._plan_slab_rows(400, 64, img.dtype, 5, 5)
    assert 64 <= rows < 400
    cp.testing.assert_array_equal(
        mw._median_wavelet_filter(img, 5, 5), expected
    )


def test_slab_memory_matches_estimate():
    """The peak pool size, also with a shorter last slab, is as estimated."""
    pool = cp.get_default_memory_pool()
    img = _random_image((2000, 256), np.uint16)
    height, width, ry, rx = 2000, 256, 5, 5
    mw._median_wavelet_filter(img, ry, rx, slab_rows=700)  # compile kernels
    for rows in (700, 2000):
        pool.free_all_blocks()
        base = pool.total_bytes()  # the input image
        out = mw._median_wavelet_filter(img, ry, rx, slab_rows=rows)
        actual = pool.total_bytes() - base
        estimate = mw._slab_memory_estimate(
            rows, height, width, img.dtype, ry, rx
        )
        # the estimate counts the output, as `actual` does
        assert 0.8 * actual <= estimate <= 1.2 * actual
        del out


@pytest.mark.parametrize("n_channels", [3, 4], ids=["RGB", "RGBA"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("mode", ["reflect", "constant", "nearest"])
def test_wavelet_multichannel_matches_sorting(n_channels, dtype, mode):
    img = _random_image((90, 70, n_channels), dtype)
    footprint = np.ones((7, 9, 1), dtype=bool)
    expected = median(img, footprint, mode=mode, cval=2, algorithm="sorting")
    out = median(img, footprint, mode=mode, cval=2, algorithm="wavelet_matrix")
    assert out.shape == img.shape and out.dtype == img.dtype
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("n_channels", [3, 4], ids=["RGB", "RGBA"])
def test_wavelet_multichannel_is_per_channel_filtering(n_channels):
    """Equal to filtering every channel as a separate 2D image."""
    img = _random_image((80, 64, n_channels), np.uint8)
    footprint = np.ones((11, 11, 1), dtype=bool)
    out = median(img, footprint, algorithm="wavelet_matrix")
    for c in range(n_channels):
        expected = median(img[..., c], footprint[..., 0], algorithm="sorting")
        cp.testing.assert_array_equal(out[..., c], expected)


@pytest.mark.parametrize(
    "shape,footprint_shape,axis",
    [
        ((3, 80, 64), (1, 7, 9), 0),  # channels first
        ((80, 3, 64), (7, 1, 9), 1),  # channels in the middle
        ((80, 64, 3), (7, 9, 1), 2),
    ],
)
def test_wavelet_multichannel_channel_axis(shape, footprint_shape, axis):
    img = _random_image(shape, np.uint16)
    footprint = np.ones(footprint_shape, dtype=bool)
    expected = median(img, footprint, algorithm="sorting")
    out = median(img, footprint, algorithm="wavelet_matrix")
    cp.testing.assert_array_equal(out, expected)


def test_wavelet_multichannel_noncontiguous_and_out():
    base = _random_image((60, 50, 4), np.uint8)
    img = base[::-1, ::2, :]  # negative and non-unit strides
    footprint = np.ones((5, 5, 1), dtype=bool)
    expected = median(img, footprint, algorithm="sorting")
    out = cp.empty(img.shape, img.dtype)
    ret = median(img, footprint, out=out, algorithm="wavelet_matrix")
    assert ret is out
    cp.testing.assert_array_equal(out, expected)


def test_auto_uses_wavelet_per_channel_for_rgb(monkeypatch):
    img = _random_image((64, 64, 3), np.uint8)
    footprint = np.ones((13, 13, 1), dtype=bool)  # 169 elements
    expected = median(img, footprint, algorithm="sorting")
    calls = []
    original = mw._median_wavelet_filter

    def counting(image, *args, **kwargs):
        calls.append(image.shape)
        return original(image, *args, **kwargs)

    monkeypatch.setattr(
        "cucim.skimage.filters._median._median_wavelet_filter", counting
    )
    cp.testing.assert_array_equal(median(img, footprint), expected)
    assert calls == [(64, 64)] * 3


def test_wavelet_multichannel_unsupported_footprints():
    img = _random_image((40, 40, 3), np.uint8)
    # a footprint with zeros is not supported
    footprint = np.ones((5, 5, 1), dtype=bool)
    footprint[2, 2, 0] = False
    with pytest.raises(ValueError):
        median(img, footprint, algorithm="wavelet_matrix")
    cp.testing.assert_array_equal(
        median(img, footprint), median(img, footprint, algorithm="sorting")
    )
    # more than one axis of size 1
    with pytest.raises(ValueError):
        median(img, np.ones((5, 1, 1), dtype=bool), algorithm="wavelet_matrix")


# 3D volumes -----------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("mode", ["reflect", "constant", "wrap"])
def test_wavelet_3d_matches_sorting(dtype, mode):
    img = _random_image((9, 14, 12), dtype)
    footprint = np.ones((3, 5, 3), dtype=bool)
    expected = median(img, footprint, mode=mode, cval=2, algorithm="sorting")
    out = median(img, footprint, mode=mode, cval=2, algorithm="wavelet_matrix")
    assert out.shape == img.shape and out.dtype == img.dtype
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "footprint_shape", [(3, 3, 3), (5, 3, 3), (3, 3, 5), (5, 5, 3), (3, 7, 5)]
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_wavelet_3d_footprint_shapes(footprint_shape, dtype):
    img = _random_image((12, 16, 14), dtype)
    footprint = np.ones(footprint_shape, dtype=bool)
    expected = median(img, footprint, mode="mirror", algorithm="sorting")
    out = median(img, footprint, mode="mirror", algorithm="wavelet_matrix")
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("rank", [0, 5, 13, 26])
def test_wavelet_3d_rank(rank):
    img = _random_image((8, 11, 10), np.uint16)
    out = mw._median_wavelet_filter_3d(img, 1, 1, 1, mode="nearest", rank=rank)
    expected = cndi.rank_filter(img, rank, size=(3, 3, 3), mode="nearest")
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("mode", ["reflect", "constant"])
def test_wavelet_3d_slabs_match_single_slab(dtype, mode):
    img = _random_image((23, 12, 9), dtype)
    args = (2, 1, 1)
    full = mw._median_wavelet_filter_3d(
        img, *args, mode=mode, cval=1, slab_planes=23
    )
    # slab heights that do and do not divide the number of planes
    for planes in (1, 4, 7, 22):
        slabbed = mw._median_wavelet_filter_3d(
            img, *args, mode=mode, cval=1, slab_planes=planes
        )
        cp.testing.assert_array_equal(slabbed, full)


def test_plan_slab_planes():
    args = (128, 256, 256, np.uint8, 3, 3, 3)
    whole = mw._slab_memory_estimate_3d(128, *args)
    assert mw._plan_slab_planes(*args, budget=whole) == 128
    prev = 128
    for fraction in (0.7, 0.4, 0.2):
        planes = mw._plan_slab_planes(*args, budget=fraction * whole)
        assert mw.WM_MIN_SLAB_PLANES <= planes < prev
        assert mw._slab_memory_estimate_3d(planes, *args) <= fraction * whole
        prev = planes
    assert mw._plan_slab_planes(*args, budget=1024) is None


def test_wavelet_3d_memory_estimate():
    pool = cp.get_default_memory_pool()
    img = _random_image((40, 48, 48), np.uint16)
    radii = (2, 2, 2)
    for planes in (12, 40):
        mw._median_wavelet_filter_3d(img, *radii, slab_planes=planes)
        pool.free_all_blocks()
        base = pool.total_bytes()  # the input volume
        out = mw._median_wavelet_filter_3d(img, *radii, slab_planes=planes)
        actual = pool.total_bytes() - base
        estimate = mw._slab_memory_estimate_3d(
            planes, 40, 48, 48, img.dtype, *radii
        )
        # the estimate must not be low, and not much too high
        assert 0.9 * actual <= estimate <= 1.5 * actual
        del out


def test_wavelet_3d_footprint_limits():
    img = _random_image((8, 8, 8), np.uint8)
    # the depth radius is limited by the per-thread state of the query
    deep = np.ones((2 * mw.WM_MAX_RADIUS_Z + 3, 3, 3), dtype=bool)
    with pytest.raises(ValueError):
        median(cp.zeros((70, 8, 8), np.uint8), deep, algorithm="wavelet_matrix")
    # even sizes and footprints with zeros are not supported
    with pytest.raises(ValueError):
        median(img, np.ones((3, 3, 4), dtype=bool), algorithm="wavelet_matrix")
    holes = np.ones((3, 3, 3), dtype=bool)
    holes[1, 1, 1] = False
    with pytest.raises(ValueError):
        median(img, holes, algorithm="wavelet_matrix")


@pytest.mark.parametrize(
    "footprint_shape,expect_wavelet",
    [((3, 3, 3), False), ((5, 5, 5), False), ((5, 5, 7), True)],
)
def test_auto_algorithm_threshold_3d(
    monkeypatch, footprint_shape, expect_wavelet
):
    img = _random_image((12, 12, 12), np.uint8)
    footprint = np.ones(footprint_shape, dtype=bool)
    expected = median(img, footprint, algorithm="sorting")
    calls = []
    original = mw._median_wavelet_filter_3d

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "cucim.skimage.filters._median._median_wavelet_filter_3d", counting
    )
    cp.testing.assert_array_equal(median(img, footprint), expected)
    assert bool(calls) == expect_wavelet


def test_speed_slabs_planners():
    plan_3d = mw._plan_speed_slab_planes
    # uint8 and small volumes keep the memory plan
    assert plan_3d(256, 256, 256, np.uint8, 2, 2, 2, 256) == 256
    assert plan_3d(32, 64, 64, np.float32, 2, 2, 2, 32) == 32
    # large wide types use slabs of the target size
    planes = plan_3d(256, 256, 256, np.float32, 2, 2, 2, 256)
    assert planes == mw.WM_SPEED_SLAB_VOXELS // 256**2
    # never larger than the plan, and not below the minimum slab
    assert plan_3d(256, 256, 256, np.float32, 20, 2, 2, 256) == 80
    assert plan_3d(256, 256, 256, np.float32, 2, 2, 2, None) is None
    assert plan_3d(256, 256, 256, np.float32, 2, 2, 2, 3) == 3


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_speed_slabs_match_single_slab(dtype, monkeypatch):
    rng = np.random.default_rng(5)
    monkeypatch.setattr(mw, "WM_SPEED_SLAB_VOXELS", 2**9)
    shape = (24, 16, 16)
    if np.dtype(dtype).kind == "f":
        image = cp.asarray(rng.random(shape, dtype=dtype))
    else:
        image = cp.asarray(rng.integers(0, 60000, shape, dtype=dtype))
    # the default now uses slabs (6 of them) ...
    assert mw._plan_speed_slab_planes(*shape, dtype, 1, 1, 1, 24) == 4
    # ... and gives the same result as one slab
    cp.testing.assert_array_equal(
        mw._median_wavelet_filter_3d(image, 1, 1, 1),
        mw._median_wavelet_filter_3d(image, 1, 1, 1, slab_planes=24),
    )
