# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single-launch 2D separable filters give the results of the per-axis
shared-memory passes (algorithm='shared_memory') bit for bit."""

import cupy as cp
import numpy as np
import pytest

from cucim.skimage._vendored import ndimage as ndi
from cucim.skimage._vendored._ndimage_filters import _unsupported_shmem_cval
from cucim.skimage.filters import _fused_separable_2d as fused2d

_SHAPES = [(1, 1), (2, 3), (70, 33), (130, 300)]

_FILTERS = {
    "gaussian_0.5": lambda x, **kw: ndi.gaussian_filter(x, 0.5, **kw),
    "gaussian_2": lambda x, **kw: ndi.gaussian_filter(x, 2.0, **kw),
    "gaussian_order": lambda x, **kw: ndi.gaussian_filter(
        x, (1.0, 1.5), order=(1, 0), **kw
    ),
    "gaussian_out_float32": lambda x, **kw: ndi.gaussian_filter(
        x, 1.0, output=cp.float32, **kw
    ),
    "uniform_origin": lambda x, **kw: ndi.uniform_filter(
        x, (4, 2), origin=(1, -1), **kw
    ),
    "sobel_axis1": lambda x, **kw: ndi.sobel(x, 1, **kw),
    "prewitt_axis0": lambda x, **kw: ndi.prewitt(x, 0, **kw),
}

_MODES = [
    ("reflect", 0.0),
    ("mirror", 0.0),
    ("nearest", 0.0),
    ("wrap", 0.0),
    ("constant", 0.3),
    (("constant", "wrap"), -2.0),
]


class _Counter:
    def __init__(self, monkeypatch):
        self.calls = 0
        original = fused2d.correlate_2d

        def counting(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(fused2d, "correlate_2d", counting)


@pytest.mark.parametrize("name", list(_FILTERS))
@pytest.mark.parametrize("dtype", [cp.uint8, cp.int16, cp.float32, cp.float64])
@pytest.mark.parametrize("mode, cval", _MODES)
def test_fused_equals_per_axis(name, dtype, mode, cval):
    func = _FILTERS[name]
    rng = np.random.default_rng(0)
    for shape in _SHAPES:
        scale = 200 if cp.dtype(dtype).kind in "iu" else 1
        image = cp.asarray((rng.random(shape) * scale).astype(dtype))
        # integer inputs with a cval they cannot hold use the elementwise
        # kernels (in constant mode), and the fused path declines
        algorithm = (
            "elementwise"
            if "constant" in mode and _unsupported_shmem_cval(cval, image.dtype)
            else "shared_memory"
        )
        expected = func(image, mode=mode, cval=cval, algorithm=algorithm)
        result = func(image, mode=mode, cval=cval)
        assert result.dtype == expected.dtype
        cp.testing.assert_array_equal(result, expected)


def test_fused_kernel_is_used(monkeypatch):
    counter = _Counter(monkeypatch)
    image = cp.random.default_rng(0).random((64, 80), dtype=cp.float32)
    ndi.gaussian_filter(image, 1.0)
    ndi.uniform_filter(image, 3)
    ndi.sobel(image, 1)
    assert counter.calls == 3
    # explicit algorithm, a single axis: per-axis passes
    ndi.gaussian_filter(image, 1.0, algorithm="shared_memory")
    ndi.gaussian_filter(image, 1.0, axes=(1,))
    assert counter.calls == 3


@pytest.mark.parametrize("dtype", [cp.float16, cp.complex64])
def test_unsupported_dtypes_fall_back(monkeypatch, dtype):
    counter = _Counter(monkeypatch)
    image = cp.random.default_rng(0).random((20, 30)).astype(dtype)
    ndi.uniform_filter(image, 3)
    assert counter.calls == 0


def test_integer_input_fractional_cval_falls_back(monkeypatch):
    counter = _Counter(monkeypatch)
    image = cp.random.default_rng(0).integers(0, 50, (20, 30), dtype=cp.uint8)
    ndi.uniform_filter(image, 3, mode="constant", cval=0.5, output=cp.float32)
    assert counter.calls == 0
    ndi.uniform_filter(image, 3, mode="constant", cval=2, output=cp.float32)
    assert counter.calls == 1


def test_overlapping_output_falls_back(monkeypatch):
    counter = _Counter(monkeypatch)
    image = cp.random.default_rng(0).random((40, 50), dtype=cp.float32)
    expected = ndi.gaussian_filter(image, 1.5, algorithm="shared_memory")
    ndi.gaussian_filter(image, 1.5, output=image)
    assert counter.calls == 0
    cp.testing.assert_array_equal(image, expected)


@pytest.mark.parametrize("ratio, expect_fused", [(64, False), (2, True)])
def test_float64_dispatch(monkeypatch, ratio, expect_fused):
    """For float64 the fused kernel is used for radii above 2 only on GPUs
    with high FP64 throughput."""
    monkeypatch.setattr(fused2d, "_fp64_ratio", lambda device_id: ratio)
    monkeypatch.setattr(fused2d, "_fast_fp64_devices", {})
    image = cp.random.default_rng(0).random((300, 400))
    counter = _Counter(monkeypatch)
    result = ndi.gaussian_filter(image, 1.0)  # radius 4
    assert (counter.calls == 1) == expect_fused
    cp.testing.assert_array_equal(
        result, ndi.gaussian_filter(image, 1.0, algorithm="shared_memory")
    )
    # radius 1 or 2: fused at any ratio
    ndi.gaussian_filter(image, 0.5)  # radius 2
    ndi.sobel(image, 0)
    assert counter.calls == 2 + int(expect_fused)


_FILTERS_3D = {
    "gaussian_0.5": lambda x, **kw: ndi.gaussian_filter(x, 0.5, **kw),
    "gaussian_aniso_order": lambda x, **kw: ndi.gaussian_filter(
        x, (0.7, 1.2, 0.5), order=(0, 1, 0), **kw
    ),
    "gaussian_0ss": lambda x, **kw: ndi.gaussian_filter(x, (0, 1.0, 0.5), **kw),
    "gaussian_ss0": lambda x, **kw: ndi.gaussian_filter(x, (1.0, 0.5, 0), **kw),
    "gaussian_s0s": lambda x, **kw: ndi.gaussian_filter(x, (1.0, 0, 0.5), **kw),
    "gaussian_axes21": lambda x, **kw: ndi.gaussian_filter(
        x, 0.8, axes=(2, 1), **kw
    ),
    "uniform_origin": lambda x, **kw: ndi.uniform_filter(
        x, (3, 4, 2), origin=(0, 1, -1), **kw
    ),
    "sobel_axis0": lambda x, **kw: ndi.sobel(x, 0, **kw),
    "sobel_axis1": lambda x, **kw: ndi.sobel(x, 1, **kw),
    "prewitt_axis-1": lambda x, **kw: ndi.prewitt(x, -1, **kw),
}

_MODES_3D = _MODES[:5] + [(("constant", "wrap", "mirror"), -2.0)]

# (with a pair (0, 1), e.g. sigma (s, s, 0), the per-axis passes are used)
_SHAPES_3D = [(1, 1, 1), (2, 3, 4), (5, 70, 33), (17, 20, 130), (40, 9, 3)]


@pytest.mark.parametrize("name", list(_FILTERS_3D))
@pytest.mark.parametrize("dtype", [cp.uint8, cp.int16, cp.float32, cp.float64])
@pytest.mark.parametrize("mode, cval", _MODES_3D)
def test_fused_3d_equals_per_axis(name, dtype, mode, cval):
    func = _FILTERS_3D[name]
    if isinstance(mode, tuple) and name == "gaussian_axes21":
        mode = mode[:2]
    rng = np.random.default_rng(0)
    for shape in _SHAPES_3D:
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
    "call, shape, expected_calls",
    [
        (lambda x: ndi.gaussian_filter(x, 1.0), (8, 20, 30), 1),
        (lambda x: ndi.gaussian_filter(x, (0, 1.0, 1.0)), (8, 20, 30), 1),
        (lambda x: ndi.sobel(x, 2), (8, 20, 30), 1),
        # pairs without the last axis (e.g. channels-last images)
        (lambda x: ndi.gaussian_filter(x, (1.0, 1.0, 0)), (8, 20, 30), 0),
        (lambda x: ndi.gaussian_filter(x, 1.0, axes=(0, 1)), (8, 20, 3), 0),
        (lambda x: ndi.gaussian_filter(x, 1.0, axes=(1,)), (8, 20, 30), 0),
        (lambda x: ndi.gaussian_filter(x, 1.0), (4, 5, 6, 7), 0),
    ],
)
def test_fused_3d_dispatch(monkeypatch, call, shape, expected_calls):
    counter = _Counter(monkeypatch)
    image = cp.random.default_rng(0).random(shape, dtype=cp.float32)
    call(image)
    assert counter.calls == expected_calls


def test_fused_3d_many_planes():
    """More planes than the grid's z dimension holds."""
    image = cp.random.default_rng(0).random((70000, 3, 5), dtype=cp.float32)
    expected = ndi.uniform_filter(image, (1, 3, 3), algorithm="shared_memory")
    cp.testing.assert_array_equal(
        ndi.uniform_filter(image, (1, 3, 3)), expected
    )


def test_fused_3d_noncontiguous_input():
    image = cp.random.default_rng(0).random((30, 20, 10), dtype=cp.float32)
    view = image.transpose(2, 0, 1)
    expected = ndi.gaussian_filter(view, 0.7, algorithm="shared_memory")
    cp.testing.assert_array_equal(ndi.gaussian_filter(view, 0.7), expected)


_EDGE_MODES = [
    ("reflect", 0.0),
    ("mirror", 0.0),
    ("wrap", 0.0),
    ("constant", 0.4),
    ("grid-constant", -1.5),
]


@pytest.mark.parametrize("name", ["sobel", "scharr", "prewitt", "farid"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("mode, cval", _EDGE_MODES)
@pytest.mark.parametrize("axis", [None, 0, -1, (1, 0)])
def test_fused_edge_filters_equal_per_axis(
    monkeypatch, name, dtype, mode, cval, axis
):
    from cucim.skimage import filters
    from cucim.skimage.filters import edges

    func = getattr(filters, name)
    rng = np.random.default_rng(0)
    for shape in _SHAPES:
        image = cp.asarray(rng.random(shape).astype(dtype))
        result = func(image, axis=axis, mode=mode, cval=cval)
        with monkeypatch.context() as m:
            m.setattr(edges, "_fused_edge_filter_2d", lambda *a: None)
            expected = func(image, axis=axis, mode=mode, cval=cval)
        assert result.dtype == expected.dtype
        cp.testing.assert_array_equal(result, expected)


def test_fused_edge_kernels_are_used(monkeypatch):
    from cucim.skimage import filters

    calls = []
    for name in ("correlate_2d", "edge_magnitude_2d"):
        original = getattr(fused2d, name)

        def counting(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(fused2d, name, counting)
    image = cp.random.default_rng(0).random((64, 80), dtype=cp.float32)
    filters.sobel(image)
    filters.farid(image, axis=1)
    filters.prewitt_h(image)
    assert calls == ["edge_magnitude_2d", "correlate_2d", "correlate_2d"]
    filters.sobel(cp.stack([image, image]))  # 3D: per-axis passes
    assert len(calls) == 3
