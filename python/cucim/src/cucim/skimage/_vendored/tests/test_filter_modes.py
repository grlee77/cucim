# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary modes of the vendored ndimage filters, compared with SciPy."""

import cupy as cp
import numpy as np
import pytest
import scipy.ndimage as ndi_cpu

from cucim.skimage._vendored import ndimage as ndi

_rng = np.random.default_rng(0)
_image = _rng.random((17, 19))
_w1 = _rng.random(5)
_w2 = _rng.random((3, 4))

# (name, positional arguments after the image)
_FILTERS = [
    ("correlate", (_w2,)),
    ("convolve", (_w2,)),
    ("correlate1d", (_w1, 0)),
    ("convolve1d", (_w1, 1)),
    ("uniform_filter", (3,)),
    ("uniform_filter1d", (4,)),
    ("gaussian_filter", (1.0,)),
    ("gaussian_filter1d", (1.0,)),
    ("sobel", (0,)),
    ("prewitt", (0,)),
    ("laplace", ()),
    ("gaussian_laplace", (1.0,)),
    ("gaussian_gradient_magnitude", (1.0,)),
    ("minimum_filter", (3,)),
    ("maximum_filter1d", (3,)),
    ("median_filter", (3,)),
    ("rank_filter", (2, 3)),
    ("percentile_filter", (30, 3)),
]


def _call(module, name, args, **kwargs):
    if module is ndi:
        image = cp.asarray(_image)
        args = tuple(
            cp.asarray(a) if isinstance(a, np.ndarray) else a for a in args
        )
    else:
        image = _image
    return getattr(module, name)(image, *args, **kwargs)


@pytest.mark.parametrize("name, args", _FILTERS, ids=[f[0] for f in _FILTERS])
def test_grid_constant_is_constant(name, args):
    """As in SciPy's filters, mode='grid-constant' is a synonym for
    'constant'."""
    expected = _call(ndi_cpu, name, args, mode="grid-constant", cval=0.3)
    result = _call(ndi, name, args, mode="grid-constant", cval=0.3)
    cp.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    cp.testing.assert_array_equal(
        result, _call(ndi, name, args, mode="constant", cval=0.3)
    )


@pytest.mark.parametrize("name, args", _FILTERS, ids=[f[0] for f in _FILTERS])
def test_grid_reflect_unsupported(name, args):
    """'grid-reflect' is not a filter mode (SciPy raises RuntimeError)."""
    with pytest.raises(RuntimeError, match="boundary mode not supported"):
        _call(ndi, name, args, mode="grid-reflect")


@pytest.mark.parametrize("name", ["sobel", "prewitt"])
@pytest.mark.parametrize("shape", [(9, 10), (9, 10, 11)])
@pytest.mark.parametrize(
    "mode", ["constant", "grid-constant", "reflect", "per-axis"]
)
def test_sobel_prewitt_match_scipy(name, shape, mode):
    """The derivative pass comes first, as in SciPy (this matters for
    mode='constant' with a nonzero cval), and `mode` may be one mode per
    axis."""
    image = np.random.default_rng(1).random(shape)
    if mode == "per-axis":
        mode = ("reflect", "constant", "wrap")[: len(shape)]
    for axis in range(-len(shape), len(shape)):
        expected = getattr(ndi_cpu, name)(image, axis, mode=mode, cval=0.7)
        result = getattr(ndi, name)(
            cp.asarray(image), axis, mode=mode, cval=0.7
        )
        cp.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32])
@pytest.mark.parametrize("cval", [0.5, 2.7, -1.25, -3, 70000])
@pytest.mark.parametrize("shape", [(9, 10), (6, 7, 8)])
@pytest.mark.parametrize(
    "name, args",
    [
        ("correlate1d", (np.array([0.5, 1.0, 0.25]), 1)),
        ("uniform_filter", (3,)),
        ("gaussian_filter", (1.0,)),
        ("sobel", (0,)),
    ],
)
def test_integer_input_cval(dtype, cval, shape, name, args):
    """Integer inputs in constant mode are padded with cval itself, as in
    SciPy, also when the input type cannot hold it (the shared-memory
    kernels converted cval to the input type)."""
    image = (_rng.random(shape) * 100).astype(dtype)
    kwargs = dict(mode="constant", cval=cval, output=np.float64)
    expected = getattr(ndi_cpu, name)(image, *args, **kwargs)
    cp_args = [cp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
    result = getattr(ndi, name)(cp.asarray(image), *cp_args, **kwargs)
    cp.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
