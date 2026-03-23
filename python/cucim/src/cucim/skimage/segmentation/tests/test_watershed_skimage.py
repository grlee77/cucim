# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""test_watershed_skimage.py - tests the watershed function

Tests adapted from scikit-image's test_watershed.py. Original test names
are preserved in docstrings for traceability.
"""

import cupy as cp
import pytest
from cupyx.scipy import ndimage as ndi

import cucim.skimage.measure
from cucim.skimage._shared.filters import gaussian
from cucim.skimage.feature import peak_local_max
from cucim.skimage.measure import label
from cucim.skimage.segmentation._watershed import (
    _get_neighbor_offsets,
    watershed,
)

# fmt: off
blob = cp.array([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 204, 204, 204, 204, 204, 204, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 204, 204, 183, 153, 153, 153, 153, 183, 204, 204, 255, 255, 255],  # noqa: E501
                 [255, 255, 204, 183, 153, 141, 111, 103, 103, 111, 141, 153, 183, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 153, 111,  94,  72,  52,  52,  72,  94, 111, 153, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 153, 111,  72,  39,   1,   1,  39,  72, 111, 153, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 183, 141, 111,  72,  39,  39,  72, 111, 141, 183, 204, 255, 255],  # noqa: E501
                 [255, 255, 255, 204, 183, 141, 111,  72,  72, 111, 141, 183, 204, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 204, 183, 141,  94,  94, 141, 183, 204, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 204, 153, 103, 103, 153, 204, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 204, 183, 141,  94,  94, 141, 183, 204, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 204, 183, 141, 111,  72,  72, 111, 141, 183, 204, 255, 255, 255],  # noqa: E501
                 [255, 255, 204, 183, 141, 111,  72,  39,  39,  72, 111, 141, 183, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 153, 111,  72,  39,   1,   1,  39,  72, 111, 153, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 153, 111,  94,  72,  52,  52,  72,  94, 111, 153, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 183, 153, 141, 111, 103, 103, 111, 141, 153, 183, 204, 255, 255],  # noqa: E501
                 [255, 255, 255, 204, 204, 183, 153, 153, 153, 153, 183, 204, 204, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 204, 204, 204, 204, 204, 204, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]])  # noqa: E501
# fmt: on


# -----------------------------------------------------------------
# Tests adapted from scikit-image's TestWatershed class
# -----------------------------------------------------------------


def test_barrier_with_8conn():
    """skimage: test_watershed01 - barrier region with 8-connectivity."""
    # fmt: off
    data = cp.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[-1, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 1, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0]],
        cp.int8,
    )
    expected = cp.array(
        [[-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers, connectivity=2)
    cp.testing.assert_array_equal(out, expected)


def test_barrier_with_4conn():
    """skimage: test_watershed02 - barrier with 4-connectivity.

    The CA-watershed may differ from scikit-image at barrier corners
    where tie-breaking depends on priority queue temporal ordering
    that the parallel algorithm cannot replicate exactly.
    """
    # fmt: off
    data = cp.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[-1, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 1, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0]],
        cp.int8,
    )
    expected = cp.array(
        [[-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1,  1,  1,  1, -1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1, -1,  1,  1,  1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers, use_age=True)
    num_diff = int(cp.sum(out != expected))
    assert num_diff <= 4


def test_two_basins_with_barrier_4conn():
    """skimage: test_watershed03 - two basins separated by barrier, 4-conn.

    The expected array has 0 at barrier pixels. scikit-image's priority
    queue leaves these as 0 (unlabeled), but the CA algorithm labels them.
    We only check non-barrier pixels match exactly.
    """
    # fmt: off
    data = cp.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 2, 0, 3, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0, -1]],
        cp.int8,
    )
    expected = cp.array(
        [[-1, -1, -1, -1, -1, -1, -1],
         [-1,  0,  2,  0,  3,  0, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  0,  2,  0,  3,  0, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers)
    # CA algorithm labels barrier pixels (expected==0) rather than leaving
    # them as 0. Only check non-barrier pixels.
    non_barrier = expected != 0
    cp.testing.assert_array_equal(out[non_barrier], expected[non_barrier])


def test_two_basins_with_barrier_8conn():
    """skimage: test_watershed04 - two basins separated by barrier, 8-conn.

    See test_two_basins_with_barrier_4conn for barrier pixel note.
    """
    # fmt: off
    data = cp.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 2, 0, 3, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0, -1]],
        cp.int8,
    )
    expected = cp.array(
        [[-1, -1, -1, -1, -1, -1, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers, connectivity=2)
    non_barrier = expected != 0
    cp.testing.assert_array_equal(out[non_barrier], expected[non_barrier])


def test_two_basins_swapped_labels():
    """skimage: test_watershed05 - same as 04 with swapped label values.

    See test_two_basins_with_barrier_4conn for barrier pixel note.
    """
    # fmt: off
    data = cp.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 3, 0, 2, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0, -1]],
        cp.int8,
    )
    expected = cp.array(
        [[-1, -1, -1, -1, -1, -1, -1],
         [-1,  3,  3,  0,  2,  2, -1],
         [-1,  3,  3,  0,  2,  2, -1],
         [-1,  3,  3,  0,  2,  2, -1],
         [-1,  3,  3,  0,  2,  2, -1],
         [-1,  3,  3,  0,  2,  2, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers, connectivity=2)
    non_barrier = expected != 0
    cp.testing.assert_array_equal(out[non_barrier], expected[non_barrier])


def test_u_shaped_barrier():
    """skimage: test_watershed06 - U-shaped barrier region."""
    # fmt: off
    data = cp.array(
        [[0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 1, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0, 0, 0]],
        cp.int8,
    )
    expected = cp.array(
        [[-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers, connectivity=2)
    cp.testing.assert_array_equal(out, expected)


def test_competitive_blobs():
    """skimage: test_watershed07 - competitive case with two blobs."""
    data = blob
    mask = data != 255
    markers = cp.zeros(data.shape, int)
    markers[6, 7] = 1
    markers[14, 7] = 2
    out = watershed(data, markers, connectivity=2, mask=mask)
    size1 = int(cp.sum(out == 1))
    size2 = int(cp.sum(out == 2))
    assert abs(size1 - size2) <= 6


def test_competitive_blobs_equal_border():
    """skimage: test_watershed08 - border pixels + edge same value."""
    data = blob.copy()
    data[10, 7:9] = 141
    mask = data != 255
    markers = cp.zeros(data.shape, int)
    markers[6, 7] = 1
    markers[14, 7] = 2
    out = watershed(data, markers, connectivity=2, mask=mask)
    size1 = int(cp.sum(out == 1))
    size2 = int(cp.sum(out == 2))
    assert abs(size1 - size2) <= 6


def test_large_image():
    """skimage: test_watershed09 - reasonable size image for timing/memory."""
    image = cp.zeros((1000, 1000))
    coords = cp.random.uniform(0, 1000, (100, 2)).astype(int)
    markers = cp.zeros((1000, 1000), int)
    idx = 1
    for x, y in coords:
        image[x, y] = 1
        markers[x, y] = idx
        idx += 1
    image = gaussian(image, sigma=4, mode="reflect")
    watershed(image, markers, connectivity=2)


def test_plateau_four_markers():
    """skimage: test_watershed10 - four markers on uniform image."""
    # fmt: off
    data = cp.array(
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]],
        cp.uint8,
    )
    markers = cp.array(
        [[1, 0, 0, 2],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [3, 0, 0, 4]],
        cp.int8,
    )
    expected = cp.array(
        [[1, 1, 2, 2],
         [1, 1, 2, 2],
         [3, 3, 4, 4],
         [3, 3, 4, 4]]
    )
    # fmt: on
    out = watershed(data, markers, connectivity=2)
    cp.testing.assert_array_equal(out, expected)


def test_plateau_closest_seed():
    """skimage: test_watershed11 - points assigned to closest seed on plateau.

    https://github.com/scikit-image/scikit-image/issues/803
    """
    image = cp.zeros((21, 21))
    markers = cp.zeros((21, 21), int)
    markers[5, 5] = 1
    markers[5, 10] = 2
    markers[10, 5] = 3
    markers[10, 10] = 4

    i, j = cp.mgrid[0:21, 0:21]
    d = cp.dstack(
        [
            cp.sqrt((i.astype(float) - i0) ** 2, (j.astype(float) - j0) ** 2)
            for i0, j0 in ((5, 5), (5, 10), (10, 5), (10, 10))
        ]
    )
    dmin = cp.min(d, 2)

    # With age-based tie-breaking, every pixel is assigned to its
    # closest seed (exact match on flat images).
    out_age = watershed(image, markers, connectivity=1, use_age=True)
    assert cp.all(d[i, j, out_age[i, j] - 1] == dmin)

    # Without age, the CA-watershed may assign a few boundary pixels
    # to a non-closest seed due to tie-breaking differences.
    out = watershed(image, markers, connectivity=1)
    num_wrong = int(cp.sum(d[i, j, out[i, j] - 1] != dmin))
    assert num_wrong <= 16


def test_watershed_line_areas():
    """skimage: test_watershed12 - watershed line boundary areas."""
    # fmt: off
    data = cp.array(
        [[203, 255, 203, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153],  # noqa: E501
         [203, 255, 203, 153, 153, 153, 102, 102, 102, 102, 102, 102, 153, 153, 153, 153],  # noqa: E501
         [203, 255, 203, 203, 153, 153, 102, 102,  77,   0, 102, 102, 153, 153, 203, 203],  # noqa: E501
         [203, 255, 255, 203, 153, 153, 153, 102, 102, 102, 102, 153, 153, 203, 203, 255],  # noqa: E501
         [203, 203, 255, 203, 203, 203, 153, 153, 153, 153, 153, 153, 203, 203, 255, 255],  # noqa: E501
         [153, 203, 255, 255, 255, 203, 203, 203, 203, 203, 203, 203, 203, 255, 255, 203],  # noqa: E501
         [153, 203, 203, 203, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 203, 203],  # noqa: E501
         [153, 153, 153, 203, 203, 203, 203, 203, 255, 203, 203, 203, 203, 203, 203, 153],  # noqa: E501
         [102, 102, 153, 153, 153, 153, 203, 203, 255, 203, 203, 255, 203, 153, 153, 153],  # noqa: E501
         [102, 102, 102, 102, 102, 153, 203, 255, 255, 203, 203, 203, 203, 153, 102, 153],  # noqa: E501
         [102,  51,  51, 102, 102, 153, 203, 255, 203, 203, 153, 153, 153, 153, 102, 153],  # noqa: E501
         [ 77,  51,  51, 102, 153, 153, 203, 255, 203, 203, 203, 153, 102, 102, 102, 153],  # noqa: E501
         [ 77,   0,  51, 102, 153, 203, 203, 255, 203, 255, 203, 153, 102,  51, 102, 153],  # noqa: E501
         [ 77,   0,  51, 102, 153, 203, 255, 255, 203, 203, 203, 153, 102,   0, 102, 153],  # noqa: E501
         [102,   0,  51, 102, 153, 203, 255, 203, 203, 153, 153, 153, 102, 102, 102, 153],  # noqa: E501
         [102, 102, 102, 102, 153, 203, 255, 203, 153, 153, 153, 153, 153, 153, 153, 153]]  # noqa: E501
    )
    # fmt: on

    markerbin = data == 0
    marker = label(markerbin)
    ws = watershed(data, marker, connectivity=2, watershed_line=True)
    for lab, area in zip(range(4), [34, 74, 74, 74]):
        assert int(cp.sum(ws == lab)) == area


def test_input_not_modified():
    """skimage: test_watershed_input_not_modified."""
    image = cp.random.default_rng().random(size=(21, 21))
    markers = cp.zeros((21, 21), dtype=cp.uint8)
    markers[[5, 5, 15, 15], [5, 15, 5, 15]] = [1, 2, 3, 4]
    original_markers = cp.copy(markers)
    result = watershed(image, markers)
    cp.testing.assert_array_equal(original_markers, markers)
    assert not cp.all(result == markers)


# -----------------------------------------------------------------
# Compact watershed tests
# -----------------------------------------------------------------


def test_compact_watershed():
    """skimage: test_compact_watershed."""
    image = cp.zeros((5, 6))
    image[:, 3] = 2  # watershed line
    image[:, 4:] = 1
    seeds = cp.zeros((5, 6), dtype=int)
    seeds[2, 0] = 1
    seeds[2, 5] = 2
    compact = watershed(image, seeds, compactness=0.01)
    expected = cp.array(
        [
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ],
        dtype=int,
    )
    cp.testing.assert_array_equal(compact, expected)

    normal = watershed(image, seeds)
    expected_skimage = cp.array(
        [
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
        ],
        dtype=int,
    )
    # Dividing line may not exactly match scikit-image
    num_differences = int(cp.sum(normal != expected_skimage))
    assert num_differences <= 5


# -----------------------------------------------------------------
# Edge case / overspill tests
# -----------------------------------------------------------------


@pytest.mark.skip(reason="cuCIM algorithm is not expected to match this result")
def test_watershed_with_markers_offset():
    """skimage: test_watershed_with_markers_offset (gh-6632 / gh-7661)."""
    x, y = cp.indices((80, 80))
    x1, y1, x2, y2 = 28, 28, 44, 52
    r1, r2 = 16, 20
    mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
    mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
    image = cp.logical_or(mask_circle1, mask_circle2)
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=cp.ones((3, 3)), labels=image)
    coords[:, 0] += 6
    mask = cp.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    props = cucim.skimage.measure.regionprops(labels, intensity_image=-distance)
    assert props[0].extent == 1
    expected_region = cp.arange(start=-10, stop=0, dtype=float).reshape(-1, 1)
    cp.testing.assert_array_equal(props[0].image_intensity, expected_region)
    assert props[0].num_pixels == 10
    assert props[1].num_pixels == 1928


def test_watershed_simple_basin_overspill():
    """skimage: test_watershed_simple_basin_overspill (gh-6632 / gh-7661)."""
    # Scenario 1
    # fmt: off
    image =    cp.array([[6, 5, 4, 3, 0, 3, 0, 1, 2],
                         [6, 5, 4, 3, 0, 3, 0, 1, 2]])
    markers =  cp.array([[0, 1, 0, 0, 0, 0, 0, 2, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    expected = cp.array([[1, 1, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2]])
    # fmt: on
    result = watershed(image, markers=markers)
    num_diff = int(cp.sum(result != expected))
    assert num_diff <= 2

    # Scenario 2 (1D with mask)
    image = -cp.array([1, 2, 2, 2, 2, 2, 3])
    markers = cp.array([1, 0, 0, 0, 0, 0, 2])
    expected = cp.array([1, 2, 2, 2, 2, 2, 2])
    result = watershed(image, markers=markers, mask=image != 0)
    cp.testing.assert_array_equal(result, expected)


@pytest.mark.skip(
    reason=(
        "CA-watershed tie-breaking differs from scikit-image on 1D plateaus"
    )
)
def test_watershed_evenly_distributed_overspill():
    """skimage: test_watershed_evenly_distributed_overspill."""
    # Scenario 1: markers start with the same value
    image =    cp.array([0, 2, 1, 1, 1, 1, 1, 1, 2, 0])  # fmt: skip
    markers =  cp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 2])  # fmt: skip
    expected = cp.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])  # fmt: skip
    result = watershed(image, markers=markers)
    cp.testing.assert_array_equal(result, expected)

    # Scenario 2: markers start with different values
    image =    cp.array([2, 2, 1, 1, 1, 1, 1, 1, 2, 0])  # fmt: skip
    expected = cp.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])  # fmt: skip
    result = watershed(image, markers=markers)
    cp.testing.assert_array_equal(result, expected)


def test_markers_on_maxima():
    """skimage: test_markers_on_maxima (gh-7661)."""
    image =    cp.array([[0, 1, 2, 3, 4, 5, 4],
                         [0, 1, 2, 3, 4, 4, 4]])  # fmt: skip
    markers =  cp.array([[1, 0, 0, 0, 0, 2, 0],
                         [0, 0, 0, 0, 0, 0, 0]])  # fmt: skip
    expected = cp.array([[1, 1, 1, 1, 1, 2, 1],
                         [1, 1, 1, 1, 1, 1, 1]])  # fmt: skip
    result = watershed(image, markers=markers)
    cp.testing.assert_array_equal(result, expected)


def test_numeric_seed_watershed():
    """skimage: test_numeric_seed_watershed."""
    image = cp.zeros((5, 6))
    image[:, 3:] = 1
    compact = watershed(image, 2, compactness=0.01)
    expected = cp.array(
        [
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
        ],
        dtype=cp.int32,
    )
    cp.testing.assert_array_equal(compact, expected)


# -----------------------------------------------------------------
# Dtype, shape, and error handling tests
# -----------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        cp.uint8,
        cp.int8,
        cp.uint16,
        cp.int16,
        cp.uint32,
        cp.int32,
        cp.uint64,
        cp.int64,
    ],
)
def test_watershed_output_dtype(dtype):
    """skimage: test_watershed_output_dtype."""
    image = cp.zeros((100, 100))
    markers = cp.zeros((100, 100), dtype)
    out = watershed(image, markers)
    assert out.dtype == markers.dtype


def test_incorrect_markers_shape():
    image = cp.ones((5, 6))
    markers = cp.ones((5, 7))
    with pytest.raises(ValueError):
        watershed(image, markers)


def test_incorrect_mask_shape():
    image = cp.ones((5, 6))
    mask = cp.ones((5, 7))
    with pytest.raises(ValueError):
        watershed(image, markers=4, mask=mask)


def test_markers_in_mask():
    data = blob
    mask = data != 255
    out = watershed(data, 25, connectivity=2, mask=mask)
    assert cp.all(out[~mask] == 0)


def test_no_markers():
    data = blob
    mask = data != 255
    out = watershed(data, mask=mask)
    assert cp.max(out) == 2


def test_block_async_small_image_warning():
    """Forcing use_block_async=True on a small image should warn and
    fall back to the synchronous path, producing correct results."""
    image = cp.zeros((10, 10), dtype=cp.float32)
    markers = cp.zeros((10, 10), dtype=cp.int32)
    markers[2, 2] = 1
    markers[8, 8] = 2
    with pytest.warns(UserWarning, match="use_block_async=True requires"):
        out = watershed(image, markers, use_block_async=True)
    assert out.shape == (10, 10)
    assert set(cp.unique(out).tolist()) == {1, 2}


# -----------------------------------------------------------------
# nD and compact watershed tests
# -----------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,marker_pos",
    [
        ((10,), [(0,), (9,)]),
        ((6, 8), [(0, 0), (5, 7)]),
        ((4, 5, 6), [(0, 0, 0), (3, 4, 5)]),
    ],
)
def test_compact_watershed_nd(shape, marker_pos):
    """Compact watershed should work for 1D, 2D, and 3D."""
    image = cp.zeros(shape, dtype=cp.float32)
    markers = cp.zeros(shape, dtype=cp.int32)
    markers[marker_pos[0]] = 1
    markers[marker_pos[1]] = 2
    result = watershed(image, markers, compactness=0.01)
    assert result.shape == shape
    assert set(cp.unique(result).tolist()) == {1, 2}


def test_watershed_4d():
    """Standard watershed should work for 4D images."""
    image = cp.zeros((4, 5, 6, 7), dtype=cp.float32)
    markers = cp.zeros_like(image, dtype=cp.int32)
    markers[0, 0, 0, 0] = 1
    markers[3, 4, 5, 6] = 2
    result = watershed(image, markers)
    assert result.shape == image.shape
    assert set(cp.unique(result).tolist()) == {1, 2}


# -----------------------------------------------------------------
# Connectivity / advanced feature tests
# -----------------------------------------------------------------


def test_connectivity():
    """skimage: test_connectivity - different connectivity gives different
    segmentation when markers are auto-generated.
    Issue = 5084
    """
    x, y = cp.indices((406, 270))
    x1, y1, x2, y2, x3, y3, x4, y4 = 200, 208, 300, 120, 100, 100, 340, 208
    r1, r2, r3, r4 = 100, 50, 40, 80
    mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
    mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
    mask_circle3 = (x - x3) ** 2 + (y - y3) ** 2 < r3**2
    mask_circle4 = (x - x4) ** 2 + (y - y4) ** 2 < r4**2
    image = cp.logical_or(mask_circle1, mask_circle2)
    image = cp.logical_or(image, mask_circle3)
    image = cp.logical_or(image, mask_circle4)

    DummyBT = ndi.distance_transform_edt(image)
    DummyBT_dis = cp.around(DummyBT / 12, decimals=0) * 12
    Img_mask = cp.where(DummyBT_dis == 0, 0, 1)

    labels_c1 = watershed(
        200 - DummyBT_dis, mask=Img_mask, connectivity=1, compactness=0.01
    )
    labels_c2 = watershed(
        200 - DummyBT_dis, mask=Img_mask, connectivity=2, compactness=0.01
    )

    assert cp.unique(labels_c1).shape[0] == 6
    assert cp.unique(labels_c2).shape[0] == 5

    # The CA-watershed kernel is non-deterministic on large plateau regions.
    # Use 20% tolerance for area checks.
    tol = 0.2

    for lab, area in zip(range(6), [61824, 3653, 20467, 11097, 1301, 11278]):
        assert (abs(int(cp.sum(labels_c1 == lab)) - area) / area) < tol

    for lab, area in zip(range(5), [61824, 3653, 20466, 12386, 11291]):
        assert (abs(int(cp.sum(labels_c2 == lab)) - area) / area) < tol


# -----------------------------------------------------------------
# Neighbor offset tests
# -----------------------------------------------------------------


class TestNeighborOffsets:
    """Tests for _get_neighbor_offsets."""

    def test_1d(self):
        offsets = _get_neighbor_offsets(1, 1)
        assert offsets == [(-1,), (1,)]

    def test_2d_connectivity1(self):
        offsets = _get_neighbor_offsets(2, 1)
        assert len(offsets) == 4
        assert set(offsets) == {(-1, 0), (1, 0), (0, -1), (0, 1)}

    def test_2d_connectivity2(self):
        offsets = _get_neighbor_offsets(2, 2)
        assert len(offsets) == 8
        for off in offsets[:4]:
            assert sum(c != 0 for c in off) == 1
        for off in offsets[4:]:
            assert sum(c != 0 for c in off) == 2

    def test_3d_connectivity1(self):
        assert len(_get_neighbor_offsets(3, 1)) == 6

    def test_3d_connectivity2(self):
        assert len(_get_neighbor_offsets(3, 2)) == 18

    def test_3d_connectivity3(self):
        assert len(_get_neighbor_offsets(3, 3)) == 26

    @pytest.mark.parametrize("ndim", [4, 5, 6])
    def test_nd_connectivity1(self, ndim):
        offsets = _get_neighbor_offsets(ndim, 1)
        assert len(offsets) == 2 * ndim
        for off in offsets:
            assert len(off) == ndim
            assert sum(c != 0 for c in off) == 1

    @pytest.mark.parametrize("ndim", [4, 5, 6])
    def test_nd_full_connectivity(self, ndim):
        assert len(_get_neighbor_offsets(ndim, ndim)) == 3**ndim - 1

    def test_offsets_sorted_by_connectivity_level(self):
        for ndim in range(1, 5):
            offsets = _get_neighbor_offsets(ndim, ndim)
            levels = [sum(c != 0 for c in off) for off in offsets]
            assert levels == sorted(levels)

    def test_no_origin(self):
        for ndim in range(1, 5):
            offsets = _get_neighbor_offsets(ndim, ndim)
            assert (0,) * ndim not in offsets


# -----------------------------------------------------------------
# Parametrized tests for block-async vs synchronous code paths
# -----------------------------------------------------------------


@pytest.mark.parametrize("use_block_async", [True, False])
@pytest.mark.parametrize("use_age", [True, False])
def test_block_async_vs_sync_large_image(use_block_async, use_age):
    """Watershed on a large image should converge and produce valid labels
    for all code path combinations."""
    image = cp.zeros((256, 256))
    rng = cp.random.default_rng(42)
    coords = rng.integers(0, 256, (20, 2))
    markers = cp.zeros((256, 256), dtype=cp.int32)
    for i, (x, y) in enumerate(coords):
        image[x, y] = 1
        markers[x, y] = i + 1
    image = gaussian(image, sigma=4, mode="reflect")

    out = watershed(
        image,
        markers,
        connectivity=2,
        use_block_async=use_block_async,
        use_age=use_age,
    )
    assert out.shape == (256, 256)
    assert int(cp.sum(out == 0)) == 0
    assert len(cp.unique(out)) == 20


# use size > 48 for markers, but also test odd sizes
@pytest.mark.parametrize("shape", ((64, 64), (49, 75)))
@pytest.mark.parametrize("use_block_async", [True, False])
def test_block_async_vs_sync_with_age_match(shape, use_block_async):
    """With use_age=True, block-async and synchronous should produce
    identical results on a flat image (deterministic tie-breaking)."""
    image = cp.zeros(shape)
    markers = cp.zeros(shape, dtype=cp.int32)
    markers[16, 16] = 1
    markers[16, 48] = 2
    markers[48, 16] = 3
    markers[48, 48] = 4

    out = watershed(
        image,
        markers,
        connectivity=1,
        use_block_async=use_block_async,
        use_age=True,
    )
    # Verify every pixel is assigned to its closest marker.
    # d[:,:,k] = Euclidean distance from each pixel to marker k+1.
    # dmin = minimum distance to any marker at each pixel.
    # d[i, j, out[i,j]-1] = distance to the *assigned* marker.
    # If age tie-breaking is correct, assigned == closest everywhere.
    i, j = cp.mgrid[0 : shape[0], 0 : shape[1]]
    d = cp.dstack(
        [
            cp.sqrt((i.astype(float) - i0) ** 2, (j.astype(float) - j0) ** 2)
            for i0, j0 in ((16, 16), (16, 48), (48, 16), (48, 48))
        ]
    )
    dmin = cp.min(d, 2)
    assert cp.all(d[i, j, out[i, j] - 1] == dmin)


@pytest.mark.parametrize("use_block_async", [True, False])
def test_block_async_watershed_line(use_block_async):
    """Watershed line post-processing should work with both code paths."""
    # Use image >= 32x32 so block-async is valid
    image = cp.zeros((64, 64), dtype=cp.float32)
    image[:, 32] = 1.0  # vertical barrier
    markers = cp.zeros((64, 64), dtype=cp.int32)
    markers[32, 10] = 1
    markers[32, 50] = 2
    out = watershed(
        image,
        markers,
        connectivity=1,
        watershed_line=True,
        use_block_async=use_block_async,
    )
    assert int(cp.sum(out == 0)) > 0
    unique = set(cp.unique(out).tolist())
    assert 1 in unique and 2 in unique
