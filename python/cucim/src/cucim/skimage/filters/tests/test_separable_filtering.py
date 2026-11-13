# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import platform

import cupy as cp
import pytest

from cucim.skimage._vendored.ndimage import (
    convolve1d,
    correlate1d,
    gaussian_filter,
    gaussian_filter1d,
    gaussian_gradient_magnitude,
    gaussian_laplace,
    laplace,
    prewitt,
    sobel,
    uniform_filter,
    uniform_filter1d,
)


def _get_image(shape, dtype, seed=123):
    rng = cp.random.default_rng(seed)
    dtype = cp.dtype(dtype)
    if dtype.kind == "b":
        image = rng.integers(0, 1, shape, dtype=cp.uint8).astype(bool)
    elif dtype.kind in "iu":
        image = rng.integers(0, 128, shape, dtype=dtype)
    elif dtype.kind in "c":
        real_dtype = cp.asarray([], dtype=dtype).real.dtype
        image = rng.standard_normal(shape, dtype=real_dtype)
        image = image + 1j * rng.standard_normal(shape, dtype=real_dtype)
    else:
        if dtype == cp.float16:
            image = rng.standard_normal(shape).astype(dtype)
        else:
            image = rng.standard_normal(shape, dtype=dtype)
    return image


def _get_rtol_atol(dtype):
    real_dtype = cp.array([], dtype=dtype).real.dtype
    rtol = atol = 1e-5
    if real_dtype == cp.float64:
        rtol = atol = 1e-12
    elif real_dtype == cp.float16:
        rtol = atol = 1e-3
    return rtol, atol


def _compare_implementations(
    shape,
    kernel_size,
    axis,
    dtype,
    mode,
    cval=0.0,
    origin=0,
    output_dtype=None,
    kernel_dtype=None,
    output_preallocated=False,
    function=convolve1d,
):
    dtype = cp.dtype(dtype)
    if kernel_dtype is None:
        kernel_dtype = dtype
    image = _get_image(shape, dtype)
    kernel = _get_image((kernel_size,), kernel_dtype)
    rtol, atol = _get_rtol_atol(kernel.dtype)
    kwargs = dict(axis=axis, mode=mode, cval=cval, origin=origin)
    if output_dtype is not None:
        output_dtype = cp.dtype(output_dtype)
    if output_preallocated:
        if output_dtype is None:
            output_dtype = image.dtype
        output1 = cp.empty(image.shape, dtype=output_dtype)
        output2 = cp.empty(image.shape, dtype=output_dtype)
        function(
            image, kernel, output=output1, algorithm="elementwise", **kwargs
        )
        function(
            image, kernel, output=output2, algorithm="shared_memory", **kwargs
        )
        cp.testing.assert_allclose(output1, output2, rtol=rtol, atol=atol)
        return
    output1 = function(
        image, kernel, output=output_dtype, algorithm="elementwise", **kwargs
    )
    output2 = function(
        image, kernel, output=output_dtype, algorithm="shared_memory", **kwargs
    )
    cp.testing.assert_allclose(output1, output2, rtol=rtol, atol=atol)
    return


def _compare_implementations_other(
    shape,
    dtype,
    mode,
    cval=0.0,
    output_dtype=None,
    kernel_dtype=None,
    output_preallocated=False,
    function=convolve1d,
    func_kwargs={},
):
    dtype = cp.dtype(dtype)
    image = _get_image(shape, dtype)
    rtol, atol = _get_rtol_atol(image.dtype)
    kwargs = dict(mode=mode, cval=cval)
    if func_kwargs:
        kwargs.update(func_kwargs)
    if output_dtype is not None:
        output_dtype = cp.dtype(output_dtype)
    if output_preallocated:
        if output_dtype is None:
            output_dtype = image.dtype
        output1 = cp.empty(image.shape, dtype=output_dtype)
        output2 = cp.empty(image.shape, dtype=output_dtype)
        function(image, output=output1, algorithm="elementwise", **kwargs)
        function(image, output=output2, algorithm="shared_memory", **kwargs)
        cp.testing.assert_allclose(output1, output2, rtol=rtol, atol=atol)
        return
    output1 = function(
        image, output=output_dtype, algorithm="elementwise", **kwargs
    )
    output2 = function(
        image, output=output_dtype, algorithm="shared_memory", **kwargs
    )
    cp.testing.assert_allclose(output1, output2, rtol=rtol, atol=atol)
    return


@pytest.mark.parametrize("shape", ((64, 57), (1000, 500)))
@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("origin", ("min", 0, "max"))
@pytest.mark.parametrize("kernel_size", tuple(range(1, 17)))
@pytest.mark.parametrize("function", [convolve1d, correlate1d])
def test_separable_kernel_sizes_and_origins(
    shape, axis, origin, kernel_size, function
):
    if kernel_size == 1:
        origin = 0
    elif origin == "min":
        origin = -(kernel_size // 2)
    elif origin == "max":
        origin = kernel_size // 2
        if kernel_size % 2 == 0:
            origin -= 1
    _compare_implementations(
        shape,
        kernel_size=kernel_size,
        axis=axis,
        dtype=cp.float32,
        mode="nearest",
        origin=origin,
        function=function,
    )


@pytest.mark.parametrize("shape", ((64, 57), (1000, 500)))
@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize(
    "kernel_size", tuple(range(17, 129, 11)) + tuple(range(145, 275, 41))
)
def test_separable_kernel_larger_sizes(shape, axis, kernel_size):
    _compare_implementations(
        shape,
        kernel_size=kernel_size,
        axis=axis,
        dtype=cp.float32,
        mode="reflect",
        origin=0,
    )


@pytest.mark.parametrize("shape", ((1000, 500),))
@pytest.mark.parametrize("axis", (0, 1))
def test_separable_elementwise_very_large_size_fallback(shape, axis):
    """Very large kernel to make it likely shared memory will be exceeded."""
    _compare_implementations(
        shape,
        kernel_size=901,
        axis=axis,
        dtype=cp.float64,
        mode="nearest",
        origin=0,
    )


@pytest.mark.parametrize("shape", ((4000, 2000), (1, 1), (5, 500), (1500, 5)))
@pytest.mark.parametrize("axis", (-1, -2))
@pytest.mark.parametrize("kernel_size", (1, 38, 129))
@pytest.mark.parametrize(
    "mode",
    ("nearest", "reflect", "wrap", "mirror", "constant", ("constant", 1)),
)
def test_separable_image_shapes_and_modes(shape, axis, kernel_size, mode):
    if isinstance(mode, tuple):
        mode, cval = mode
    else:
        cval = 0

    _compare_implementations(
        shape,
        kernel_size=kernel_size,
        axis=axis,
        dtype=cp.float32,
        mode=mode,
        cval=cval,
        origin=0,
    )


image_dtypes_tested = (
    cp.float16,
    cp.float32,
    cp.float64,
    cp.complex64,
    cp.complex128,
    bool,
    cp.int8,
    cp.uint8,
    cp.int16,
    cp.uint16,
    cp.int32,
    cp.uint32,
    cp.int64,
    cp.uint64,
)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason=(
        "custom separable kernels disabled on Windows until dtype-related "
        "test failures can be resolved"
    ),
)
@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("image_dtype", image_dtypes_tested)
@pytest.mark.parametrize(
    "kernel_dtype", (None, cp.float32, cp.uint8, cp.complex64)
)
def test_separable_image_and_kernel_dtypes(axis, image_dtype, kernel_dtype):
    """Test many kernel and image dtype combinations"""

    _compare_implementations(
        (64, 32),
        kernel_size=3,
        axis=axis,
        dtype=image_dtype,
        mode="nearest",
        origin=0,
        kernel_dtype=kernel_dtype,
    )


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason=(
        "custom separable kernels disabled on Windows until dtype-related "
        "test failures can be resolved"
    ),
)
@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("image_dtype", image_dtypes_tested)
@pytest.mark.parametrize(
    "output_dtype", (None, cp.float32, cp.int32, cp.complex64)
)
@pytest.mark.parametrize("output_preallocated", (False, True))
def test_separable_input_and_output_dtypes(
    axis, image_dtype, output_dtype, output_preallocated
):
    """Test many kernel and image dtype combinations"""
    if cp.dtype(image_dtype).kind == "c" and output_dtype is not None:
        if not cp.dtype(output_dtype).kind == "c":
            pytest.skip("cannot cast complex values to real")
    _compare_implementations(
        (64, 32),
        kernel_size=3,
        axis=axis,
        dtype=image_dtype,
        mode="nearest",
        origin=0,
        kernel_dtype=None,
        output_dtype=output_dtype,
        output_preallocated=output_preallocated,
    )


@pytest.mark.parametrize("shape", ((64, 57),))
@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("origin", ("min", 0, "max"))
@pytest.mark.parametrize(
    "function, func_kwargs",
    [
        (gaussian_filter, dict(sigma=1.5)),
        (gaussian_filter1d, dict(sigma=1.5, axis=0)),
        (gaussian_filter1d, dict(sigma=1.5, axis=1)),
        (gaussian_gradient_magnitude, dict(sigma=3.5)),
        (gaussian_laplace, dict(sigma=2.5)),
        (laplace, {}),
        (prewitt, {}),
        (sobel, {}),
        (uniform_filter, dict(size=7)),
        (uniform_filter1d, dict(size=7, axis=0)),
        (uniform_filter1d, dict(size=7, axis=1)),
    ],
)
def test_separable_internal_kernel(shape, axis, origin, function, func_kwargs):
    """
    Test case to make sure the 'algorithm' kwarg works for all other separable
    ndimage filters as well.
    """
    _compare_implementations_other(
        shape,
        dtype=cp.float32,
        mode="nearest",
        function=function,
        func_kwargs=func_kwargs,
    )


@pytest.mark.parametrize("shape", ((16, 24, 32), (192, 128, 160)))
@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("kernel_size", tuple(range(1, 17, 3)))
@pytest.mark.parametrize("function", [convolve1d, correlate1d])
def test_separable_kernel_sizes_3d(shape, axis, kernel_size, function):
    _compare_implementations(
        shape,
        kernel_size=kernel_size,
        axis=axis,
        dtype=cp.float32,
        mode="nearest",
        origin=0,
        function=function,
    )


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("kernel_size", (65, 129, 198))
def test_separable_large_kernel_3d(axis, kernel_size):
    _compare_implementations(
        shape=(256, 128, 96),
        kernel_size=kernel_size,
        axis=axis,
        dtype=cp.float32,
        mode="reflect",
        origin=0,
    )


@pytest.mark.parametrize(
    "shape", ((64, 5, 64), (5, 64, 64), (64, 64, 5), (32, 32, 32))
)
@pytest.mark.parametrize("axis", (-1, -2, -3))
@pytest.mark.parametrize("kernel_size", (9,))
@pytest.mark.parametrize(
    "mode",
    ("nearest", "reflect", "wrap", "mirror", "constant", ("constant", 1)),
)
def test_separable_image_shapes_and_modes_3d(shape, axis, kernel_size, mode):
    if isinstance(mode, tuple):
        mode, cval = mode
    else:
        cval = 0
    _compare_implementations(
        shape,
        kernel_size=kernel_size,
        axis=axis,
        dtype=cp.float32,
        mode=mode,
        cval=cval,
        origin=0,
    )


@pytest.mark.parametrize(
    "shape, dtype, kernel_size, axis, expected",
    [
        ((20, 24, 28), cp.float32, 9, 0, True),
        ((20, 24, 28), cp.uint8, 33, 1, True),
        ((20, 24, 28), cp.float64, 17, 0, True),
        ((20, 24, 28), cp.float64, 33, 2, True),
        ((20, 24, 28), cp.float64, 33, -1, True),
        ((20, 24, 28), cp.float64, 33, 0, False),
        ((20, 24, 28), cp.float64, 33, -2, False),
        ((40, 30, 16), cp.float32, 9, 1, True),
        ((40, 30, 3), cp.float32, 9, 1, False),
        ((40, 30, 3), cp.float32, 9, 2, False),
    ],
)
def test_default_algorithm_3d(
    monkeypatch, shape, dtype, kernel_size, axis, expected
):
    """With algorithm=None, 3D inputs use the shared-memory kernel except
    when the last axis has fewer than 16 elements or for float64 with more
    than 17 taps along the first two axes, and the result equals that of the
    other kernel."""
    from cucim.skimage._vendored import _ndimage_filters

    calls = []
    original = _ndimage_filters._shmem_convolve1d

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_ndimage_filters, "_shmem_convolve1d", counting)
    image = _get_image(shape, dtype)
    weights = cp.random.default_rng(0).random(kernel_size)
    result = correlate1d(image, weights, axis=axis, mode="constant", cval=2)
    assert bool(calls) == expected
    other = "elementwise" if expected else "shared_memory"
    cp.testing.assert_array_equal(
        result,
        correlate1d(
            image, weights, axis=axis, mode="constant", cval=2, algorithm=other
        ),
    )


@pytest.mark.parametrize(
    "shape, dtype, kernel_size, mode, convolution",
    [
        ((37, 70), cp.float32, 17, "reflect", False),
        ((130, 40), cp.float64, 9, "constant", True),
        ((9, 12, 70), cp.float32, 3, "wrap", True),
        ((20, 33, 18), cp.uint8, 33, "constant", False),
        ((5, 3, 70), cp.int16, 6, "reflect", False),
        ((11, 40, 24), cp.float32, 64, "mirror", True),
    ],
)
def test_register_blocking_identical(
    monkeypatch, shape, dtype, kernel_size, mode, convolution
):
    """The register-blocked stage 2 of the shared-memory kernels gives the
    same results as the per-tap one, for every axis and for extreme
    origins."""
    from cucim.skimage.filters import _separable_filtering

    image = _get_image(shape, dtype)
    weights = cp.random.default_rng(1).random(kernel_size)
    func = convolve1d if convolution else correlate1d
    for axis in range(len(shape)):
        for origin in sorted({0, -(kernel_size // 2), (kernel_size - 1) // 2}):
            results = []
            for register_blocking in (False, True):
                monkeypatch.setattr(
                    _separable_filtering,
                    "_REGISTER_BLOCKING",
                    register_blocking,
                )
                results.append(
                    func(
                        image,
                        weights,
                        axis=axis,
                        mode=mode,
                        cval=2,
                        origin=origin,
                        algorithm="shared_memory",
                    )
                )
            cp.testing.assert_array_equal(results[1], results[0])
