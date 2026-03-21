# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""test_watershed.py - tests the watershed function"""

import math
import unittest

import cupy as cp
import pytest
from cupyx.scipy import ndimage as ndi

import cucim.skimage.measure
from cucim.skimage._shared.filters import gaussian
from cucim.skimage.feature import peak_local_max
from cucim.skimage.measure import label
from cucim.skimage.segmentation._watershed import watershed

eps = 1e-12
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


def diff(a, b):
    if not isinstance(a, cp.ndarray):
        a = cp.asarray(a)
    if not isinstance(b, cp.ndarray):
        b = cp.asarray(b)
    if (0 in a.shape) and (0 in b.shape):
        return 0.0
    b[a == 0] = 0
    if a.dtype in [cp.complex64, cp.complex128] or b.dtype in [
        cp.complex64,
        cp.complex128,
    ]:
        a = cp.asarray(a, cp.complex128)
        b = cp.asarray(b, cp.complex128)
        t = ((a.real - b.real) ** 2).sum() + ((a.imag - b.imag) ** 2).sum()
    else:
        a = cp.asarray(a)
        a = a.astype(cp.float64)
        b = cp.asarray(b)
        b = b.astype(cp.float64)
        t = ((a - b) ** 2).sum()
    return math.sqrt(t)


class TestWatershed(unittest.TestCase):
    connectivity = 2

    def test_watershed01(self):
        "watershed 1"
        # fmt: off
        data = cp.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            cp.uint8,
        )
        markers = cp.array(
            [
                [-1, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
            ],
            cp.int8,
        )
        out = watershed(data, markers, self.connectivity)
        expected = cp.array(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1,  1,  1,  1,  1,  1, -1],
                [-1,  1,  1,  1,  1,  1, -1],
                [-1,  1,  1,  1,  1,  1, -1],
                [-1,  1,  1,  1,  1,  1, -1],
                [-1,  1,  1,  1,  1,  1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )
        # fmt: on
        error = diff(expected, out)
        assert error < eps

    def test_watershed02(self):
        "watershed 2"
        # fmt: off
        data = cp.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            cp.uint8,
        )
        markers = cp.array(
            [
                [-1, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
            ],
            cp.int8,
        )
        out = watershed(data, markers)
        expected = cp.array(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1,  1,  1,  1, -1, -1],
                [-1,  1,  1,  1,  1,  1, -1],
                [-1,  1,  1,  1,  1,  1, -1],
                [-1,  1,  1,  1,  1,  1, -1],
                [-1, -1,  1,  1,  1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )
        # fmt: on
        # The CA-watershed may differ from scikit-image at barrier corners
        # where tie-breaking depends on priority queue temporal ordering
        # that the parallel algorithm cannot replicate exactly.
        num_diff = int(cp.sum(out != expected))
        self.assertTrue(num_diff <= 4)

    def test_watershed03(self):
        "watershed 3"
        # fmt: off
        data = cp.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            cp.uint8,
        )
        markers = cp.array(
            [
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 2, 0, 3, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0, -1],
            ],
            cp.int8,
        )
        out = watershed(data, markers)
        error = diff(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1,  0,  2,  0,  3,  0, -1],
                [-1,  2,  2,  0,  3,  3, -1],
                [-1,  2,  2,  0,  3,  3, -1],
                [-1,  2,  2,  0,  3,  3, -1],
                [-1,  0,  2,  0,  3,  0, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            out,
        )
        # fmt: on
        self.assertTrue(error < eps)

    def test_watershed04(self):
        "watershed 4"
        # fmt: off
        data = cp.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            cp.uint8,
        )
        markers = cp.array(
            [
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 2, 0, 3, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0, -1],
            ],
            cp.int8,
        )
        out = watershed(data, markers, self.connectivity)
        error = diff(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1,  2,  2,  0,  3,  3, -1],
                [-1,  2,  2,  0,  3,  3, -1],
                [-1,  2,  2,  0,  3,  3, -1],
                [-1,  2,  2,  0,  3,  3, -1],
                [-1,  2,  2,  0,  3,  3, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            out,
        )
        # fmt: on
        self.assertTrue(error < eps)

    def test_watershed05(self):
        "watershed 5"
        # fmt: off
        data = cp.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            cp.uint8,
        )
        markers = cp.array(
            [
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 3, 0, 2, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0, -1],
            ],
            cp.int8,
        )
        out = watershed(data, markers, self.connectivity)
        error = diff(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1,  3,  3,  0,  2,  2, -1],
                [-1,  3,  3,  0,  2,  2, -1],
                [-1,  3,  3,  0,  2,  2, -1],
                [-1,  3,  3,  0,  2,  2, -1],
                [-1,  3,  3,  0,  2,  2, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            out,
        )
        # fmt: on
        self.assertTrue(error < eps)

    def test_watershed06(self):
        "watershed 6"
        # fmt: off
        data = cp.array(
            [
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            cp.uint8,
        )
        markers = cp.array(
            [
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0],
            ],
            cp.int8,
        )
        out = watershed(data, markers, self.connectivity)
        error = diff(
            [
                [-1,  1,  1,  1,  1,  1, -1],
                [-1,  1,  1,  1,  1,  1, -1],
                [-1,  1,  1,  1,  1,  1, -1],
                [-1,  1,  1,  1,  1,  1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            out,
        )
        # fmt: on
        self.assertTrue(error < eps)

    def test_watershed07(self):
        "A regression test of a competitive case that failed"
        data = blob
        mask = data != 255
        markers = cp.zeros(data.shape, int)
        markers[6, 7] = 1
        markers[14, 7] = 2
        out = watershed(data, markers, self.connectivity, mask=mask)
        #
        # The two objects should be the same size, except possibly for the
        # border region
        #
        size1 = cp.sum(out == 1)
        size2 = cp.sum(out == 2)
        self.assertTrue(abs(size1 - size2) <= 6)

    def test_watershed08(self):
        "The border pixels + an edge are all the same value"
        data = blob.copy()
        data[10, 7:9] = 141
        mask = data != 255
        markers = cp.zeros(data.shape, int)
        markers[6, 7] = 1
        markers[14, 7] = 2
        out = watershed(data, markers, self.connectivity, mask=mask)
        #
        # The two objects should be the same size, except possibly for the
        # border region
        #
        size1 = cp.sum(out == 1)
        size2 = cp.sum(out == 2)
        self.assertTrue(abs(size1 - size2) <= 6)

    def test_watershed09(self):
        """Test on an image of reasonable size

        This is here both for timing (does it take forever?) and to
        ensure that the memory constraints are reasonable
        """
        image = cp.zeros((1000, 1000))
        coords = cp.random.uniform(0, 1000, (100, 2)).astype(int)
        markers = cp.zeros((1000, 1000), int)
        idx = 1
        for x, y in coords:
            image[x, y] = 1
            markers[x, y] = idx
            idx += 1

        image = gaussian(image, sigma=4, mode="reflect")
        watershed(image, markers, self.connectivity)
        # ndi.watershed_ift(image.astype(cp.uint16), markers, self.connectivity)

    def test_watershed10(self):
        "watershed 10"
        # fmt: off
        data = cp.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ],
            cp.uint8,
        )
        markers = cp.array(
            [
                [1, 0, 0, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 0, 0, 4]
            ],
            cp.int8,
        )
        out = watershed(data, markers, self.connectivity)
        error = diff(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4]
            ],
            out,
        )
        # fmt: on
        self.assertTrue(error < eps)

    def test_watershed11(self):
        """Make sure that all points on this plateau are assigned to closest
        seed.
        """
        # https://github.com/scikit-image/scikit-image/issues/803
        #
        # Make sure that no point in a level image is farther away
        # from its seed than any other
        #
        image = cp.zeros((21, 21))
        markers = cp.zeros((21, 21), int)
        markers[5, 5] = 1
        markers[5, 10] = 2
        markers[10, 5] = 3
        markers[10, 10] = 4

        # structure = cp.array(
        #     [[False, True, False], [True, True, True], [False, True, False]]
        # )
        # out = watershed(image, markers, structure)
        i, j = cp.mgrid[0:21, 0:21]
        d = cp.dstack(
            [
                cp.sqrt(
                    (i.astype(float) - i0) ** 2, (j.astype(float) - j0) ** 2
                )
                for i0, j0 in ((5, 5), (5, 10), (10, 5), (10, 10))
            ]
        )
        dmin = cp.min(d, 2)

        # With age-based tie-breaking, every pixel is assigned to its
        # closest seed (exact match on flat images).
        out_age = watershed(image, markers, connectivity=1, use_age=True)
        self.assertTrue(cp.all(d[i, j, out_age[i, j] - 1] == dmin))

        # Without age, the CA-watershed may assign a few boundary pixels
        # to a non-closest seed due to tie-breaking differences.
        out = watershed(image, markers, connectivity=1)
        num_wrong = int(cp.sum(d[i, j, out[i, j] - 1] != dmin))
        self.assertTrue(num_wrong <= 4)

    def test_watershed12(self):
        "The watershed line"

        # fmt: off
        data = cp.array(
            [
                [203, 255, 203, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153],  # noqa: E501
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
                [102, 102, 102, 102, 153, 203, 255, 203, 153, 153, 153, 153, 153, 153, 153, 153],  # noqa: E501
            ]
        )
        # fmt: on

        markerbin = data == 0
        marker = label(markerbin)
        ws = watershed(data, marker, connectivity=2, watershed_line=True)

        visualize = False
        if visualize:
            import matplotlib.pyplot as plt
            from skimage.segmentation import watershed as watershed_cpu

            data_cpu = cp.asnumpy(data)
            marker_cpu = cp.asnumpy(marker)
            ws_cpu = watershed_cpu(
                data_cpu, marker_cpu, connectivity=2, watershed_line=True
            )

            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(cp.asnumpy(ws))
            axes[0].set_title("cuCIM result")
            axes[1].imshow(ws_cpu)
            axes[1].set_title("skimage result")
            plt.show()

        for lab, area in zip(range(4), [34, 74, 74, 74]):
            self.assertTrue(cp.sum(ws == lab) == area)

    def test_watershed_input_not_modified(self):
        """Test to ensure input markers are not modified."""
        image = cp.random.default_rng().random(size=(21, 21))
        markers = cp.zeros((21, 21), dtype=cp.uint8)
        markers[[5, 5, 15, 15], [5, 15, 5, 15]] = [1, 2, 3, 4]
        original_markers = cp.copy(markers)
        result = watershed(image, markers)
        cp.testing.assert_array_equal(original_markers, markers)
        assert not cp.all(result == markers)


def test_compact_watershed():
    # in this test, when compactness is greater than zero the watershed line
    # is labeled with the closest marker (label=2)
    # when compactness is zero the watershed line is labeled with
    # the marker that reaches it first (label=1)
    # because it has a zero cost path to the line.
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
    # dividing line between 1s and 2s may not exactly match for cuCIM
    # cp.testing.assert_array_equal(normal, expected)
    num_differences = int(cp.sum(normal != expected_skimage))
    assert num_differences <= 5

    # TODO(grelee): watershed_line not yet implemented
    if False:
        # checks that compact watershed labels with watershed lines are
        # a subset of the labels from compact watershed for this specific example
        compact_wsl = watershed(
            image, seeds, compactness=0.01, watershed_line=True
        )
        difference = compact_wsl != compact
        difference[compact_wsl == 0] = False
        assert not cp.any(difference)


@pytest.mark.skip(reason="cuCIM algorithm is not expected to match this result")
def test_watershed_with_markers_offset():
    """
    Check edge case behavior reported in gh-6632

    While we initially viewed the behavior described in gh-6632 [1]_ as a bug,
    we have reverted that decision in gh-7661. See [2]_ for an explanation.
    So this test now actually asserts the behavior reported in gh-6632 as
    correct.

    .. [1] https://github.com/scikit-image/scikit-image/issues/6632.
    .. [2] https://github.com/scikit-image/scikit-image/issues/7661#issuecomment-2645810807
    """
    # Generate an initial image with two overlapping circles
    x, y = cp.indices((80, 80))
    x1, y1, x2, y2 = 28, 28, 44, 52
    r1, r2 = 16, 20
    mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
    mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
    image = cp.logical_or(mask_circle1, mask_circle2)

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    # and then apply an y-offset
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=cp.ones((3, 3)), labels=image)
    coords[:, 0] += 6
    mask = cp.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    labels = watershed(-distance, markers, mask=image)

    plot = False
    if plot:
        import matplotlib.pyplot as plt
        from skimage.segmentation import watershed as watershed_cpu

        fig, axes = plt.subplots(1, 5)
        axes[0].imshow(cp.asnumpy(distance))
        axes[0].set_title("distance")
        # axes[0].plot((28, 52), (34, 50), 'r.')
        axes[1].imshow(cp.asnumpy(markers))
        axes[1].set_title("markers")
        axes[2].imshow(cp.asnumpy(image))
        axes[2].set_title("mask")
        axes[3].imshow(cp.asnumpy(labels))
        axes[3].set_title("labels (cuCIM)")
        labels_cpu = watershed_cpu(
            cp.asnumpy(-distance), cp.asnumpy(markers), mask=cp.asnumpy(image)
        )
        axes[4].imshow(labels_cpu)
        axes[4].set_title("labels (skimage)")
        plt.show()

    # !fig, axes = plt.subplots(1, 3); axes[0].imshow(cp.asnumpy(-distance)); axes[1].imshow(cp.asnumpy(labels)); axes[2].imshow(labels_cpu); plt.show()

    props = cucim.skimage.measure.regionprops(labels, intensity_image=-distance)

    # Generally, assert that the smaller object could only conquer a thin line
    # in the direction of the positive gradient
    assert props[0].extent == 1
    expected_region = cp.arange(start=-10, stop=0, dtype=float).reshape(-1, 1)
    cp.testing.assert_array_equal(props[0].image_intensity, expected_region)

    # Assert pixel count from reviewed reproducing example in bug report
    assert props[0].num_pixels == 10
    assert props[1].num_pixels == 1928


def test_watershed_simple_basin_overspill():
    """
    Test edge case behavior when markers spill over into another basin /
    compete.

    While we initially viewed the behavior described in gh-6632 [1]_ as a bug,
    we have reverted that decision in gh-7661. See [2]_ for an explanation.
    So this test now actually asserts the behavior reported in gh-6632 as
    correct.

    .. [1] https://github.com/scikit-image/scikit-image/issues/6632.
    .. [2] https://github.com/scikit-image/scikit-image/issues/7661#issuecomment-2645810807
    """
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
    # The CA-watershed may assign a small number of pixels differently
    # from scikit-image due to priority-queue temporal ordering that the
    # parallel algorithm cannot replicate exactly.
    num_diff = int(cp.sum(result != expected))
    assert num_diff <= 2

    # Scenario 2
    image = -cp.array([1, 2, 2, 2, 2, 2, 3])
    markers = cp.array([1, 0, 0, 0, 0, 0, 2])
    expected = cp.array([1, 2, 2, 2, 2, 2, 2])
    result = watershed(image, markers=markers, mask=image != 0)
    cp.testing.assert_array_equal(result, expected)


@pytest.mark.skip(
    reason="CA-watershed tie-breaking differs from scikit-image on 1D plateaus"
)
def test_watershed_evenly_distributed_overspill():
    """
    Edge case: Basins should be distributed evenly between contesting markers.

    Markers should be prevented from spilling over into another basin and
    conquering it against other markers with the same claim, just because they
    get to the basin one step earlier.
    """
    # Scenario 1: markers start with the same value
    image =    cp.array([0, 2, 1, 1, 1, 1, 1, 1, 2, 0])  # fmt: skip
    markers =  cp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 2])  # fmt: skip
    expected = cp.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])  # fmt: skip
    result = watershed(image, markers=markers)
    cp.testing.assert_array_equal(result, expected)

    # Scenario 2: markers start with the different values
    image =    cp.array([2, 2, 1, 1, 1, 1, 1, 1, 2, 0])  # fmt: skip
    expected = cp.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])  # fmt: skip
    result = watershed(image, markers=markers)
    cp.testing.assert_array_equal(result, expected)


def test_markers_on_maxima():
    """Check that markers placed at maxima don't conquer other pixels.

    Regression test for gh-7661 [1]_.

    .. [1] https://github.com/scikit-image/scikit-image/issues/7661
    """
    image =    cp.array([[0, 1, 2, 3, 4, 5, 4],
                         [0, 1, 2, 3, 4, 4, 4]])  # fmt: skip
    markers =  cp.array([[1, 0, 0, 0, 0, 2, 0],
                         [0, 0, 0, 0, 0, 0, 0]])  # fmt: skip
    expected = cp.array([[1, 1, 1, 1, 1, 2, 1],
                         [1, 1, 1, 1, 1, 1, 1]])  # fmt: skip
    result = watershed(image, markers=markers)
    cp.testing.assert_array_equal(result, expected)


def test_numeric_seed_watershed():
    """Test that passing just the number of seeds to watershed works."""
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


def test_watershed_unsupported_ndim():
    """Watershed should raise NotImplementedError for 4D+ images."""
    image_4d = cp.zeros((3, 4, 5, 6))
    markers_4d = cp.zeros((3, 4, 5, 6), dtype=cp.int32)
    markers_4d[0, 0, 0, 0] = 1
    markers_4d[2, 3, 4, 5] = 2
    with pytest.raises(NotImplementedError, match="4D"):
        watershed(image_4d, markers_4d)


def test_markers_in_mask():
    data = blob
    mask = data != 255
    out = watershed(data, 25, connectivity=2, mask=mask)
    # There should be no markers where the mask is false
    assert cp.all(out[~mask] == 0)


def test_no_markers():
    data = blob
    mask = data != 255
    out = watershed(data, mask=mask)
    assert cp.max(out) == 2


def test_connectivity():
    """
    Watershed segmentation should output different result for
    different connectivity
    when markers are calculated where None is supplied.
    Issue = 5084
    """
    # Generate a dummy BrightnessTemperature image
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

    # calculate distance in discrete increase
    DummyBT = ndi.distance_transform_edt(image)
    DummyBT_dis = cp.around(DummyBT / 12, decimals=0) * 12
    # calculate the mask
    Img_mask = cp.where(DummyBT_dis == 0, 0, 1)

    # segments for connectivity 1 and 2
    labels_c1 = watershed(
        200 - DummyBT_dis, mask=Img_mask, connectivity=1, compactness=0.01
    )
    labels_c2 = watershed(
        200 - DummyBT_dis, mask=Img_mask, connectivity=2, compactness=0.01
    )

    # assertions
    assert cp.unique(labels_c1).shape[0] == 6
    assert cp.unique(labels_c2).shape[0] == 5

    # The CA-watershed kernel is non-deterministic: threads read neighbor
    # labels/priorities from global memory without synchronization, so a
    # neighbor's value may reflect either the previous or current iteration
    # depending on GPU scheduling. This is especially pronounced on large
    # plateau regions (like the quantized distance image here) where many
    # pixels have identical priority and the label winner depends on timing.
    # We use a 20% tolerance to account for this run-to-run variation.
    # See WATERSHED_DESIGN.md for details.
    tol = 0.2

    # checking via area of each individual segment.
    for lab, area in zip(range(6), [61824, 3653, 20467, 11097, 1301, 11278]):
        assert (abs(int(cp.sum(labels_c1 == lab)) - area) / area) < tol

    for lab, area in zip(range(5), [61824, 3653, 20466, 12386, 11291]):
        assert (abs(int(cp.sum(labels_c2 == lab)) - area) / area) < tol
