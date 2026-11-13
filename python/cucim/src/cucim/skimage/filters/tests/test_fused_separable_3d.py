# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single-launch (z-marching) 3D separable filters give the results of
the per-axis passes bit for bit."""

import cupy as cp
import numpy as np
import pytest

from cucim.skimage import filters
from cucim.skimage._vendored import ndimage as ndi
from cucim.skimage._vendored._ndimage_filters import _unsupported_shmem_cval
from cucim.skimage.filters import _fused_separable_3d as fused3d

# (2, 3, 4): smaller than the kernels; (70, 20, 130): several z chunks and
# tiles; (5, 40, 17): a last axis left to the elementwise 1D kernels by the
# per-axis path
_SHAPES = [(2, 3, 4), (9, 17, 70), (70, 20, 130), (5, 40, 17)]

_FILTERS = {
    "gaussian_0.5": lambda x, **kw: ndi.gaussian_filter(x, 0.5, **kw),
    "gaussian_1": lambda x, **kw: ndi.gaussian_filter(x, 1.0, **kw),
    "gaussian_aniso_order": lambda x, **kw: ndi.gaussian_filter(
        x, (0.7, 0.5, 0.6), order=(0, 1, 0), **kw
    ),
    "gaussian_axes021": lambda x, **kw: ndi.gaussian_filter(
        x, 0.5, axes=(0, 2, 1), **kw
    ),
    "uniform_origin": lambda x, **kw: ndi.uniform_filter(
        x, (3, 4, 2), origin=(0, 1, -1), **kw
    ),
    "sobel_axis0": lambda x, **kw: ndi.sobel(x, 0, **kw),
    "sobel_axis1": lambda x, **kw: ndi.sobel(x, 1, **kw),
    "prewitt_axis2": lambda x, **kw: ndi.prewitt(x, 2, **kw),
}

_MODES = [
    ("reflect", 0.0),
    ("mirror", 0.0),
    ("nearest", 0.0),
    ("wrap", 0.0),
    ("constant", 0.3),
    (("constant", "wrap", "mirror"), -2.0),
]


class _Counter:
    def __init__(self, monkeypatch, name="separable_3d"):
        self.calls = 0
        original = getattr(fused3d, name)

        def counting(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(fused3d, name, counting)


@pytest.fixture
def always_z_marching(monkeypatch):
    """Use the single-chain kernel wherever it applies, whatever the size,
    taps and dtype (the dispatch rule is tested separately)."""
    monkeypatch.setattr(fused3d, "prefer_separable_3d", lambda *args: True)


@pytest.mark.usefixtures("always_z_marching")
@pytest.mark.parametrize("name", list(_FILTERS))
@pytest.mark.parametrize("dtype", [cp.uint8, cp.int16, cp.float32, cp.float64])
@pytest.mark.parametrize("mode, cval", _MODES)
def test_z_marching_equals_per_axis(name, dtype, mode, cval):
    func = _FILTERS[name]
    rng = np.random.default_rng(0)
    for shape in _SHAPES:
        scale = 200 if cp.dtype(dtype).kind in "iu" else 1
        image = cp.asarray((rng.random(shape) * scale).astype(dtype))
        algorithm = (
            "elementwise"
            if "constant" in mode and _unsupported_shmem_cval(cval, image.dtype)
            else "shared_memory"
        )
        expected = func(image, mode=mode, cval=cval, algorithm=algorithm)
        result = func(image, mode=mode, cval=cval)
        assert result.dtype == expected.dtype
        cp.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "call, dtype, expected_calls",
    [
        (lambda x: ndi.uniform_filter(x, 3), cp.float32, 1),
        (lambda x: ndi.sobel(x, 1), cp.float32, 1),  # passes (1, 0, 2)
        (lambda x: ndi.uniform_filter(x, 3), cp.uint8, 1),  # float32 weights
        (lambda x: ndi.uniform_filter(x, 3), cp.float64, 0),
        (lambda x: ndi.gaussian_filter(x, 0.5), cp.float32, 0),  # 5 taps
        (lambda x: ndi.uniform_filter(x, 3, axes=(1, 2, 0)), cp.float32, 0),
        (lambda x: ndi.uniform_filter(x, (1, 3, 3)), cp.float32, 0),
        (
            lambda x: ndi.uniform_filter(x, 3, algorithm="shared_memory"),
            cp.float32,
            0,
        ),
    ],
)
def test_z_marching_dispatch(monkeypatch, call, dtype, expected_calls):
    """The rule, with the size threshold lowered for a small test volume."""
    monkeypatch.setattr(fused3d, "_PLAIN_MIN_SIZE_VS_PAIR", 0)
    counter = _Counter(monkeypatch)
    image = cp.asarray(
        (np.random.default_rng(0).random((20, 24, 70)) * 100).astype(dtype)
    )
    call(image)
    assert counter.calls == expected_calls


def test_z_marching_size_threshold(monkeypatch):
    counter = _Counter(monkeypatch)
    image = cp.random.default_rng(0).random((20, 24, 70), dtype=cp.float32)
    ndi.uniform_filter(image, 3)
    assert counter.calls == 0
    monkeypatch.setattr(fused3d, "_PLAIN_MIN_SIZE_VS_PAIR", image.size)
    ndi.uniform_filter(image, 3)
    assert counter.calls == 1


def test_z_marching_disabled_matches(monkeypatch):
    image = cp.random.default_rng(0).random((30, 33, 90), dtype=cp.float32)
    result = ndi.gaussian_filter(image, 0.8, mode="constant", cval=0.5)
    monkeypatch.setattr(fused3d, "_Z_MARCHING", False)
    expected = ndi.gaussian_filter(image, 0.8, mode="constant", cval=0.5)
    cp.testing.assert_array_equal(result, expected)


_EDGE_MODES = [
    ("reflect", 0.0),
    ("mirror", 0.0),
    ("nearest", 0.0),
    ("wrap", 0.0),
    ("constant", 0.4),
    ("grid-constant", -1.5),
]


@pytest.mark.usefixtures("always_z_marching")
@pytest.mark.parametrize("name", ["sobel", "scharr", "prewitt", "farid"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("mode, cval", _EDGE_MODES)
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_fused_3d_edge_filters_equal_per_axis(
    monkeypatch, name, dtype, mode, cval, axis
):
    func = getattr(filters, name)
    rng = np.random.default_rng(0)
    for shape in _SHAPES:
        image = cp.asarray(rng.random(shape).astype(dtype))
        result = func(image, axis=axis, mode=mode, cval=cval)
        with monkeypatch.context() as m:
            m.setattr(fused3d, "_Z_MARCHING", False)
            expected = func(image, axis=axis, mode=mode, cval=cval)
        assert result.dtype == expected.dtype
        cp.testing.assert_array_equal(result, expected)


def test_fused_3d_edge_kernels_are_used(monkeypatch):
    magnitude = _Counter(monkeypatch, "edge_magnitude_3d")
    single = _Counter(monkeypatch, "separable_3d")
    image = cp.random.default_rng(0).random((10, 20, 70), dtype=cp.float32)
    filters.sobel(image)
    filters.farid(image)
    filters.sobel(image.astype(cp.float64))
    assert (magnitude.calls, single.calls) == (3, 0)
    filters.farid(image.astype(cp.float64))  # queues too long
    filters.sobel(image, axis=(0, 1))  # magnitude over two axes: per axis
    filters.sobel(image, axis=2)  # small volume
    assert (magnitude.calls, single.calls) == (3, 0)
    monkeypatch.setattr(fused3d, "_PLAIN_MIN_SIZE_VS_PER_AXIS", 0)
    filters.sobel(image, axis=2)
    filters.farid(image, axis=2)  # 5 taps
    assert (magnitude.calls, single.calls) == (3, 1)
