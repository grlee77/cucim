# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp
import pytest

from cucim.skimage.morphology import flood, flood_fill

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
    cp.testing.assert_equal(output, expected)


def test_overrange_tolerance_float():
    max_value = cp.finfo(cp.float32).max

    image = cp.random.uniform(size=(64, 64), low=-1.0, high=1.0).astype(
        cp.float32
    )
    image *= max_value

    expected = cp.ones_like(image)
    output = flood_fill(image, (0, 1), 1.0, tolerance=max_value.item() * 10)

    cp.testing.assert_equal(output, expected)


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

    cp.testing.assert_equal(output, expected)
    cp.testing.assert_equal(output, output2)


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

    cp.testing.assert_equal(flood_fill(test, (0, 0), -1), expected)


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

    cp.testing.assert_equal(output, expected)

    test[2] = 100
    expected[2] = 255

    output2 = flood_fill(test, (2, 3), 255)

    cp.testing.assert_equal(output2, expected)


def test_footprint():
    # Basic tests for nonstandard footprints
    footprint = cp.array(
        [[0, 1, 1], [0, 1, 1], [0, 0, 0]]
    )  # Cannot grow left or down

    output = flood_fill(
        cp.zeros((5, 6), dtype=cp.uint8), (3, 1), 255, footprint=footprint
    )

    expected = cp.array(
        [
            [0, 255, 255, 255, 255, 255],
            [0, 255, 255, 255, 255, 255],
            [0, 255, 255, 255, 255, 255],
            [0, 255, 255, 255, 255, 255],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=cp.uint8,
    )

    cp.testing.assert_equal(output, expected)

    footprint = cp.array(
        [[0, 0, 0], [1, 1, 0], [1, 1, 0]]
    )  # Cannot grow right or up

    output = flood_fill(
        cp.zeros((5, 6), dtype=cp.uint8), (1, 4), 255, footprint=footprint
    )

    expected = cp.array(
        [
            [0, 0, 0, 0, 0, 0],
            [255, 255, 255, 255, 255, 0],
            [255, 255, 255, 255, 255, 0],
            [255, 255, 255, 255, 255, 0],
            [255, 255, 255, 255, 255, 0],
        ],
        dtype=cp.uint8,
    )

    cp.testing.assert_equal(output, expected)


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
        cp.testing.assert_equal(
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

    cp.testing.assert_equal(output, expected)

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

    cp.testing.assert_equal(output, expected)
