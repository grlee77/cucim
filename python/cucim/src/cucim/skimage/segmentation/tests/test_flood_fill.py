# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp
import numpy as np
import pytest
from skimage.morphology import flood as flood_cpu, flood_fill as flood_fill_cpu

from cucim.skimage.segmentation import flood, flood_fill

eps = 1e-12


def test_empty_input():
    # Test shortcut
    output = flood_fill(cp.empty(0), (), 2)
    assert output.size == 0

    # Boolean output type
    assert flood(cp.empty(0), ()).dtype == bool

    # Maintain shape, even with zero size present
    assert flood(cp.empty((20, 0, 4)), ()).shape == (20, 0, 4)


def test_float16():
    image = cp.array([9.0, 0.1, 42], dtype=cp.float16)
    with pytest.raises(TypeError, match="dtype of `image` is float16"):
        flood_fill(image, 0, 1)


@pytest.mark.parametrize("tolerance", [-150, 150, -379, 379])
def test_overrange_tolerance_int(tolerance):
    image = cp.arange(256, dtype=cp.uint8).reshape((8, 8, 4))
    seed = (3, 4, 2)
    expected = cp.zeros_like(image)
    output = flood_fill(image, seed, 0, tolerance=tolerance)
    cp.testing.assert_array_equal(output, expected)


def test_overrange_tolerance_float():
    max_value = cp.finfo(cp.float32).max

    image = cp.random.uniform(size=(64, 64), low=-1.0, high=1.0).astype(
        cp.float32
    )
    image *= max_value

    expected = cp.ones_like(image)
    output = flood_fill(image, (0, 1), 1.0, tolerance=max_value.item() * 10)

    cp.testing.assert_array_equal(output, expected)


def test_inplace_int():
    image = cp.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2, 2, 0],
            [0, 1, 1, 0, 2, 2, 0],
            [1, 0, 0, 0, 0, 0, 3],
            [0, 1, 1, 1, 3, 3, 4],
        ]
    )

    flood_fill(image, (0, 0), 5, in_place=True)

    expected = cp.array(
        [
            [5, 5, 5, 5, 5, 5, 5],
            [5, 1, 1, 5, 2, 2, 5],
            [5, 1, 1, 5, 2, 2, 5],
            [1, 5, 5, 5, 5, 5, 3],
            [5, 1, 1, 1, 3, 3, 4],
        ]
    )

    cp.testing.assert_array_equal(image, expected)


def test_inplace_float():
    image = cp.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2, 2, 0],
            [0, 1, 1, 0, 2, 2, 0],
            [1, 0, 0, 0, 0, 0, 3],
            [0, 1, 1, 1, 3, 3, 4],
        ],
        dtype=cp.float32,
    )

    flood_fill(image, (0, 0), 5, in_place=True)

    expected = cp.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 1.0, 1.0, 5.0, 2.0, 2.0, 5.0],
            [5.0, 1.0, 1.0, 5.0, 2.0, 2.0, 5.0],
            [1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0],
            [5.0, 1.0, 1.0, 1.0, 3.0, 3.0, 4.0],
        ],
        dtype=cp.float32,
    )

    cp.testing.assert_allclose(image, expected)


def test_inplace_noncontiguous():
    image = cp.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2, 2, 0],
            [0, 1, 1, 0, 2, 2, 0],
            [1, 0, 0, 0, 0, 0, 3],
            [0, 1, 1, 1, 3, 3, 4],
        ]
    )

    # Transpose is noncontiguous
    image2 = image[::2, ::2]

    flood_fill(image2, (0, 0), 5, in_place=True)

    # The inplace modified result
    expected2 = cp.array([[5, 5, 5, 5], [5, 1, 2, 5], [5, 1, 3, 4]])

    cp.testing.assert_allclose(image2, expected2)

    # Projected back through the view, `image` also modified
    expected = cp.array(
        [
            [5, 0, 5, 0, 5, 0, 5],
            [0, 1, 1, 0, 2, 2, 0],
            [5, 1, 1, 0, 2, 2, 5],
            [1, 0, 0, 0, 0, 0, 3],
            [5, 1, 1, 1, 3, 3, 4],
        ]
    )

    cp.testing.assert_allclose(image, expected)


def test_1d():
    image = cp.arange(11)
    expected = cp.array([0, 1, -20, -20, -20, -20, -20, -20, -20, 9, 10])

    output = flood_fill(image, 5, -20, tolerance=3)
    output2 = flood_fill(image, (5,), -20, tolerance=3)

    cp.testing.assert_array_equal(output, expected)
    cp.testing.assert_array_equal(output, output2)


def test_wraparound():
    # If the borders (or neighbors) aren't correctly accounted for, this fails,
    # because the algorithm uses an ravelled array.
    test = cp.zeros((5, 7), dtype=cp.float64)
    test[:, 3] = 100

    expected = cp.array(
        [
            [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )

    cp.testing.assert_array_equal(flood_fill(test, (0, 0), -1), expected)


def test_neighbors():
    # This test will only pass if the neighbors are exactly correct
    test = cp.zeros((5, 7), dtype=cp.float64)
    test[:, 3] = 100

    expected = cp.array(
        [
            [0, 0, 0, 255, 0, 0, 0],
            [0, 0, 0, 255, 0, 0, 0],
            [0, 0, 0, 255, 0, 0, 0],
            [0, 0, 0, 255, 0, 0, 0],
            [0, 0, 0, 255, 0, 0, 0],
        ]
    )
    output = flood_fill(test, (0, 3), 255)

    cp.testing.assert_array_equal(output, expected)

    test[2] = 100
    expected[2] = 255

    output2 = flood_fill(test, (2, 3), 255)

    cp.testing.assert_array_equal(output2, expected)


@pytest.mark.parametrize(
    "footprint",
    [
        [[0, 1, 1], [0, 1, 1], [0, 0, 0]],  # cannot grow left or down
        [[0, 0, 0], [1, 1, 0], [1, 1, 0]],  # cannot grow right or up
    ],
)
def test_footprint_not_symmetric(footprint):
    # The fill would follow directed neighbors, which is not supported
    with pytest.raises(NotImplementedError, match="symmetric"):
        flood_fill(
            cp.zeros((5, 6), dtype=cp.uint8),
            (3, 1),
            255,
            footprint=cp.array(footprint),
        )


def test_footprint():
    # Basic tests for nonstandard (symmetric) footprints
    footprint = cp.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
    image = cp.zeros((4, 5), dtype=cp.uint8)
    image[:, 2] = 1
    # Only the left and right neighbors (and the pixel itself) are used
    footprint = cp.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    output = flood_fill(image, (1, 3), 255, footprint=footprint)
    expected = cp.zeros_like(image)
    expected[:, 2] = 1
    expected[1, 3:] = 255
    cp.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize(
    "footprint, message",
    [
        (np.ones((3,), dtype=bool), "number of dimensions"),
        (np.ones((3, 4), dtype=bool), "odd"),
    ],
)
def test_footprint_invalid(footprint, message):
    with pytest.raises(ValueError, match=message):
        flood(cp.zeros((5, 6)), (0, 0), footprint=footprint)


def test_basic_nd():
    for dimension in (3, 4, 5):
        shape = (5,) * dimension
        hypercube = cp.zeros(shape)
        slice_mid = tuple(slice(1, -1, None) for dim in range(dimension))
        hypercube[slice_mid] = 1  # sum is 3**dimension
        filled = flood_fill(hypercube, (2,) * dimension, 2)

        # Test that the middle sum is correct
        assert filled.sum() == 3**dimension * 2

        # Test that the entire array is as expected
        cp.testing.assert_array_equal(
            filled, cp.pad(cp.ones((3,) * dimension) * 2, 1, "constant")
        )


@pytest.mark.parametrize("tolerance", [None, 0])
def test_f_order(tolerance):
    image = cp.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        order="F",
    )
    expected = cp.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=bool,
    )

    mask = flood(image, seed_point=(1, 0), tolerance=tolerance)
    cp.testing.assert_array_equal(expected, mask)

    mask = flood(image, seed_point=(2, 1), tolerance=tolerance)
    cp.testing.assert_array_equal(expected, mask)


def test_negative_indexing_seed_point():
    image = cp.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2, 2, 0],
            [0, 1, 1, 0, 2, 2, 0],
            [1, 0, 0, 0, 0, 0, 3],
            [0, 1, 1, 1, 3, 3, 4],
        ],
        dtype=cp.float32,
    )

    expected = cp.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 1.0, 1.0, 5.0, 2.0, 2.0, 5.0],
            [5.0, 1.0, 1.0, 5.0, 2.0, 2.0, 5.0],
            [1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0],
            [5.0, 1.0, 1.0, 1.0, 3.0, 3.0, 4.0],
        ],
        dtype=cp.float32,
    )

    image = flood_fill(image, (0, -1), 5)

    cp.testing.assert_allclose(image, expected)


def test_non_adjacent_footprint():
    # Basic tests for non-adjacent footprints
    footprint = cp.array(
        [
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
        ]
    )

    output = flood_fill(
        cp.zeros((5, 6), dtype=cp.uint8), (2, 3), 255, footprint=footprint
    )

    expected = cp.array(
        [
            [0, 255, 0, 0, 0, 255],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 255, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 255, 0, 0, 0, 255],
        ],
        dtype=cp.uint8,
    )

    cp.testing.assert_array_equal(output, expected)

    footprint = cp.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    image = cp.zeros((5, 10), dtype=cp.uint8)
    image[:, (3, 7, 8)] = 100

    output = flood_fill(image, (0, 0), 255, footprint=footprint)

    expected = cp.array(
        [
            [255, 255, 255, 100, 255, 255, 255, 100, 100, 0],
            [255, 255, 255, 100, 255, 255, 255, 100, 100, 0],
            [255, 255, 255, 100, 255, 255, 255, 100, 100, 0],
            [255, 255, 255, 100, 255, 255, 255, 100, 100, 0],
            [255, 255, 255, 100, 255, 255, 255, 100, 100, 0],
        ],
        dtype=cp.uint8,
    )

    cp.testing.assert_array_equal(output, expected)


def test_nan_seed():
    # Deviation from scikit-image: a NaN seed fills only the seed point,
    # also with a tolerance
    image = cp.array([[cp.nan, 1.0], [cp.nan, 1.0]], dtype=cp.float32)
    expected = cp.array([[True, False], [False, False]])
    cp.testing.assert_array_equal(flood(image, (0, 0)), expected)
    cp.testing.assert_array_equal(flood(image, (0, 0), tolerance=0.5), expected)
    # NaN pixels are never within the tolerance of a finite seed
    cp.testing.assert_array_equal(
        flood(image, (0, 1), tolerance=0.5),
        cp.array([[False, True], [False, True]]),
    )


@pytest.mark.parametrize(
    "seed_point, exception",
    [((5, 0), IndexError), ((0, -6), IndexError), ((0,), ValueError)],
)
def test_invalid_seed_point(seed_point, exception):
    with pytest.raises(exception):
        flood(cp.zeros((5, 5)), seed_point)


def test_flood_returns_mask_and_flood_fill_agrees():
    image = cp.zeros((6, 6), dtype=cp.uint8)
    image[2:4, 2:4] = 7
    mask = flood(image, (2, 2))
    assert mask.dtype == bool and mask.shape == image.shape
    expected = image.copy()
    expected[mask] = 9
    cp.testing.assert_array_equal(flood_fill(image, (2, 2), 9), expected)
    # the input is not modified unless in_place=True
    assert int(image.max()) == 7


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [cp.uint8, cp.int16, cp.float32, cp.float64])
@pytest.mark.parametrize("tolerance", [None, 0, 1, 2.5])
def test_matches_skimage(ndim, dtype, tolerance):
    rng = np.random.default_rng(ndim)
    shape = (40, 33, 17)[:ndim]
    image = rng.integers(0, 4, shape).astype(dtype)
    for connectivity in range(1, ndim + 1):
        for seed_point in [(0,) * ndim, tuple(s // 2 for s in shape)]:
            expected = flood_cpu(
                image,
                seed_point,
                connectivity=connectivity,
                tolerance=tolerance,
            )
            result = flood(
                cp.asarray(image),
                seed_point,
                connectivity=connectivity,
                tolerance=tolerance,
            )
            cp.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("ndim", [2, 3])
def test_matches_skimage_symmetric_footprints(ndim):
    rng = np.random.default_rng(5)
    shape = (30, 25, 12)[:ndim]
    image = (rng.random(shape) < 0.7).astype(np.uint8)
    for size in (3, 5):
        footprint = rng.random((size,) * ndim) < 0.4
        # make the footprint symmetric about its center
        footprint = footprint | footprint[(slice(None, None, -1),) * ndim]
        seed_point = tuple(s // 2 for s in shape)
        image[seed_point] = 1
        expected = flood_cpu(image, seed_point, footprint=footprint)
        result = flood(cp.asarray(image), seed_point, footprint=footprint)
        cp.testing.assert_array_equal(result, expected)


def test_winding_region():
    # A single corridor that winds through the image (long geodesic path)
    n = 65
    image = np.ones((n, n), dtype=np.uint8)
    for row in range(1, n, 2):
        image[row, 1 : n - 1] = 0
    for k, row in enumerate(range(2, n, 2)):
        image[row, (n - 2) if k % 2 == 0 else 1] = 0
    expected = flood_cpu(image, (1, 1), connectivity=1)
    assert expected.sum() > n * n // 3
    result = flood(cp.asarray(image), (1, 1), connectivity=1)
    cp.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [cp.uint8, cp.int32, cp.float32, cp.bool_])
@pytest.mark.parametrize("new_value", [5.7, 0, 1, cp.asarray(3.2), np.int64(4)])
def test_flood_fill_new_value_is_converted_like_an_assignment(dtype, new_value):
    image = cp.zeros((6, 6), dtype=dtype)
    image[:, 3] = 1
    result = flood_fill(image, (0, 0), new_value)
    mask = flood(image, (0, 0))
    expected = image.copy()
    expected[mask] = new_value
    cp.testing.assert_array_equal(result, expected)
    assert result.dtype == image.dtype


def test_flood_fill_new_value_array():
    # values that are not a scalar use the assignment (one value per pixel)
    image = cp.zeros((4, 4), dtype=cp.int32)
    mask = flood(image, (0, 0))
    values = cp.arange(int(mask.sum()), dtype=cp.int32)
    result = flood_fill(image, (0, 0), values)
    expected = image.copy()
    expected[mask] = values
    cp.testing.assert_array_equal(result, expected)


def test_flood_with_64_bit_labels(monkeypatch):
    # images with 2**31 or more elements use 64-bit indices; check the paths
    # (blocks and pixels) with a small image
    import cucim.skimage.segmentation._flood_fill as module

    rng = np.random.default_rng(0)
    image = (rng.random((40, 37, 11)) < 0.6).astype(np.uint8)
    seed = (20, 18, 5)
    image[seed] = 1
    monkeypatch.setattr(module, "_label_dtype_for_size", lambda size: cp.int64)
    for connectivity in (1, 3):
        expected = flood_cpu(image, seed, connectivity=connectivity)
        result = flood(cp.asarray(image), seed, connectivity=connectivity)
        cp.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "shape, footprint",
    [
        ((3,), np.array([False, True, False])),
        ((5, 6), np.array([[False] * 3, [False, True, False], [False] * 3])),
        ((4, 5), np.ones((1, 1), dtype=bool)),
    ],
)
def test_footprint_with_only_its_center(shape, footprint):
    # no neighbors: only the pixel of the seed (and equal pixels are not
    # connected to it)
    image = cp.ones(shape, dtype=cp.uint8)
    seed = (1,) * len(shape)
    expected = flood_cpu(cp.asnumpy(image), seed, footprint=footprint)
    result = flood(image, seed, footprint=footprint)
    np.testing.assert_array_equal(cp.asnumpy(result), expected)
    assert expected.sum() == 1
    filled = flood_fill(image, seed, 7, footprint=footprint)
    assert int((filled == 7).sum()) == 1


@pytest.mark.parametrize("seed_value", [0, 1])
@pytest.mark.parametrize("tolerance", [None, 0, 0.5, 1, 1.5])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_boolean_image_with_fractional_tolerance(ndim, tolerance, seed_value):
    # scikit-image works on the bytes of a boolean image, so a fractional
    # bound is truncated to an integer (it is not converted to True)
    rng = np.random.default_rng({1: 0, 2: 1, 3: 2}[ndim])
    shape = {1: (40,), 2: (24, 25), 3: (10, 11, 12)}[ndim]
    image = rng.random(shape) < 0.5
    seed = tuple(np.argwhere(image == bool(seed_value))[0])
    expected = flood_cpu(image, seed, tolerance=tolerance)
    result = flood(cp.asarray(image), seed, tolerance=tolerance)
    np.testing.assert_array_equal(cp.asnumpy(result), expected)
    expected_filled = flood_fill_cpu(image, seed, True, tolerance=tolerance)
    filled = flood_fill(cp.asarray(image), seed, True, tolerance=tolerance)
    np.testing.assert_array_equal(cp.asnumpy(filled), expected_filled)


def test_boolean_image_fractional_tolerance_example():
    image = cp.asarray([False, True, False, True])
    assert flood(image, 0, tolerance=0.5).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert flood(image, 3, tolerance=0.5).tolist() == [True, True, True, True]


def test_inplace_fill_snapshots_aliased_device_scalar(monkeypatch):
    import cucim.skimage.segmentation._flood_fill as module

    image = cp.arange(4096, dtype=cp.int32)
    replacement = image[77:78].reshape(())
    assert replacement.data.ptr == image.data.ptr + 77 * image.itemsize
    write = module._set_where

    def check_storage(mask, value, output):
        assert output is image
        assert value.data.ptr != replacement.data.ptr
        return write(mask, value, output)

    monkeypatch.setattr(module, "_set_where", check_storage)
    result = flood_fill(image, 0, replacement, tolerance=4096, in_place=True)
    assert result is image
    cp.testing.assert_array_equal(result, cp.full_like(image, 77))


@pytest.mark.parametrize("use_blocks", [False, True])
@pytest.mark.parametrize("tolerance", [np.uint8(10), np.int8(10)])
def test_flood_normalizes_numpy_integer_tolerance(tolerance, use_blocks):
    image = cp.array([240, 244, 250, 255], dtype=cp.uint8)
    seed = (2,)
    if use_blocks:
        image = cp.stack((image, image))
        seed = (0, 2)

    # All pixels are within 10 of 250 after clipping to the uint8 range.
    cp.testing.assert_array_equal(
        flood(image, seed, tolerance=tolerance),
        cp.ones(image.shape, dtype=bool),
    )


def test_flood_abs_of_minimum_signed_tolerance():
    image = cp.array([-2, 0, 1], dtype=cp.int64)
    cp.testing.assert_array_equal(
        flood(image, 2, tolerance=np.int64(-(2**63))),
        cp.ones(image.shape, dtype=bool),
    )


@pytest.mark.parametrize(
    "dtype, values, seed, tolerance, expected",
    [
        (cp.int64, [2**53, 2**53 + 1], 1, 0.0, [False, True]),
        (cp.int64, [2**63 - 2, 2**63 - 1], 1, 0.0, [False, True]),
        (cp.int64, [-(2**63), -(2**63) + 1], 0, 0.0, [True, False]),
        (cp.uint64, [2**64 - 2, 2**64 - 1], 1, 0.0, [False, True]),
        (
            cp.uint64,
            [2**63, 2**63 + 1, 2**63 + 2],
            1,
            0.5,
            [True, True, False],
        ),
        (
            cp.int64,
            [-(2**53) - 2, -(2**53) - 1, -(2**53)],
            1,
            0.5,
            [False, True, True],
        ),
    ],
)
@pytest.mark.parametrize("use_blocks", [False, True])
def test_flood_keeps_integer_seed_exact_with_float_tolerance(
    dtype, values, seed, tolerance, expected, use_blocks
):
    image = cp.array(values, dtype=dtype)
    expected = cp.array(expected)
    if use_blocks:
        image = cp.stack((image, image))
        expected = cp.stack((expected, expected))
        seed = (0, seed)
    cp.testing.assert_array_equal(
        flood(image, seed, tolerance=tolerance), expected
    )
