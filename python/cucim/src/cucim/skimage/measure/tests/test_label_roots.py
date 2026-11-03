# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest

from cucim.skimage.measure import label
from cucim.skimage.measure._label import _get_structure
from cucim.skimage.measure._label_kernels import _label_roots


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("greyscale_mode", [False, True])
@pytest.mark.parametrize("dtype", [cp.int32, cp.int64])
def test_label_roots_same_regions_as_label(ndim, greyscale_mode, dtype):
    # _label_roots skips the renumbering: the entries of two foreground
    # pixels are equal if and only if label gives them the same label
    rng = np.random.default_rng(ndim)
    shape = (200, 61, 23)[:ndim]
    image = rng.integers(0, 3, shape).astype(np.uint8)
    for connectivity in range(1, ndim + 1):
        source = image if greyscale_mode else image > 0
        expected = label(cp.asarray(source), connectivity=connectivity).get()
        roots = cp.empty(shape, dtype=dtype)
        _label_roots(
            cp.asarray(image),
            _get_structure(ndim, connectivity),
            roots,
            greyscale_mode=greyscale_mode,
        )
        roots = roots.get()
        assert ((roots < 0) == (expected == 0)).all()
        # a one-to-one correspondence between roots and labels
        foreground = expected > 0
        pairs = np.unique(
            np.stack([roots[foreground], expected[foreground]]), axis=1
        )
        assert pairs.shape[1] == expected.max()
        assert len(np.unique(pairs[0])) == len(np.unique(pairs[1]))
        # the root of a region is the index of one of its pixels
        assert (roots.ravel()[roots[foreground]] == roots[foreground]).all()


def test_label_roots_larger_structure():
    # a structure that is larger than 3 along an axis (non-adjacent neighbors)
    image = np.zeros((5, 9), dtype=np.uint8)
    image[2, 1] = image[2, 4] = image[2, 7] = 1
    structure = np.zeros((1, 7), dtype=bool)
    structure[0, 3] = structure[0, 0] = structure[0, 6] = True
    roots = cp.empty(image.shape, dtype=cp.int32)
    _label_roots(cp.asarray(image), structure, roots)
    roots = roots.get()
    # only the pixels 3 apart along the last axis are connected
    assert roots[2, 1] == roots[2, 4] == roots[2, 7]
    assert roots[2, 1] != roots[0, 0]
