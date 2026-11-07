# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import warnings

import cupy as cp
import pytest
from cupy.testing import assert_allclose
from cupyx.scipy import ndimage
from skimage import data

from cucim.skimage import morphology
from cucim.skimage.filters import _median as median_module, median, rank


@pytest.fixture
def image():
    return cp.array(
        [
            [1, 2, 3, 2, 1],
            [1, 1, 2, 2, 3],
            [3, 2, 1, 2, 1],
            [3, 2, 1, 1, 1],
            [1, 2, 1, 2, 3],
        ],
        dtype=cp.uint8,
    )


@pytest.fixture
def camera():
    return cp.array(data.camera())


@pytest.mark.parametrize(
    "mode, cval, behavior, warning_type",
    [
        ("nearest", 0.0, "ndimage", []),
        ("constant", 0.0, "rank", (UserWarning,)),
        ("nearest", 0.0, "rank", []),
    ],
)
def test_median_warning(image, mode, cval, behavior, warning_type):
    if warning_type:
        with pytest.warns(warning_type):
            median(image, mode=mode, behavior=behavior)
    else:
        median(image, mode=mode, behavior=behavior)


@pytest.mark.parametrize(
    "behavior, func",
    [("ndimage", ndimage.median_filter)],
    # ('rank', rank.median, {'footprint': cp.ones((3, 3), dtype=cp.uint8)})]
)
@pytest.mark.parametrize(
    "mode", ["reflect", "mirror", "nearest", "constant", "wrap"]
)
# include even shapes and singleton shape that force non-histogram code path.
# include some large shapes that always take the histogram-based code path.
@pytest.mark.parametrize(
    "footprint_shape",
    [
        (3, 3),
        (5, 5),
        (9, 15),
        (2, 2),
        (1, 1),
        (2, 7),
        (23, 23),
        (15, 35),
    ],
)
@pytest.mark.parametrize("footprint_tuple", (False, True))
@pytest.mark.parametrize("out", [None, cp.uint8, cp.float32, "array"])
def test_median_behavior(
    camera, behavior, func, mode, footprint_shape, footprint_tuple, out
):
    if footprint_tuple:
        footprint = footprint_shape
    else:
        footprint = cp.ones(footprint_shape, dtype=bool)
    cam2 = camera[:, :177]  # use anisotropic size
    assert cam2.dtype == cp.uint8
    if out == "array":
        out = cp.zeros_like(cam2)
    assert_allclose(
        median(cam2, footprint, mode=mode, behavior=behavior, out=out),
        func(cam2, size=footprint_shape, mode=mode, output=out),
    )


@pytest.mark.parametrize("footprint_tuple", (False, True))
@pytest.mark.parametrize("out", [None, "array"])
def test_median_rank_behavior_matches_rank_median(camera, footprint_tuple, out):
    footprint_shape = (3, 5)
    if footprint_tuple:
        footprint = footprint_shape
        rank_footprint = cp.ones(footprint_shape, dtype=bool)
    else:
        footprint = cp.ones(footprint_shape, dtype=bool)
        rank_footprint = footprint
    cam2 = camera[:64, :75]
    out_arg = cp.zeros_like(cam2) if out == "array" else None
    expected_out = cp.zeros_like(cam2) if out == "array" else None

    result = median(
        cam2,
        footprint,
        behavior="rank",
        out=out_arg,
    )
    expected = rank.median(cam2, footprint=rank_footprint, out=expected_out)

    if out_arg is not None:
        assert result is out_arg
        assert expected is expected_out
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "mode", ["reflect", "mirror", "nearest", "constant", "wrap"]
)
# the large anisotropic footprint triggers the wavelet matrix algorithm
@pytest.mark.parametrize("footprint_shape", [(3, 3), (3, 5), (15, 23)])
@pytest.mark.parametrize("int_dtype", [cp.uint8, cp.int8, cp.uint16, cp.int16])
@pytest.mark.parametrize("algorithm", ["auto", "wavelet_matrix", "sorting"])
def test_median_int_dtypes(mode, footprint_shape, int_dtype, algorithm):
    footprint = cp.ones(footprint_shape, dtype=bool)
    rng = cp.random.default_rng(123)
    shape = (350, 407)
    if int_dtype == cp.uint8:
        img = rng.integers(0, 256, shape, dtype=cp.uint8)
    elif int_dtype == cp.int8:
        img = rng.integers(-128, 128, shape, dtype=int).astype(cp.int8)
    elif int_dtype == cp.uint16:
        img = rng.integers(0, 1024, shape, dtype=cp.uint16)
    elif int_dtype == cp.int16:
        img = rng.integers(-128, 384, shape, dtype=int).astype(cp.int16)

    out = median(
        img, footprint, mode=mode, behavior="ndimage", algorithm=algorithm
    )
    expected = ndimage.median_filter(img, size=footprint.shape, mode=mode)
    assert_allclose(expected, out)


@pytest.mark.parametrize("mode", ["reflect"])
@pytest.mark.parametrize("footprint_shape", [(7, 11)])
@pytest.mark.parametrize(
    "int_dtype, irange",
    [
        (cp.uint16, (0, 256)),
        (cp.uint16, (0, 15)),
        (cp.uint16, (128, 384)),
        (cp.uint16, (0, 510)),
        (cp.uint16, (500, 550)),
        (cp.uint16, (0, 1024)),
        (cp.uint16, (0, 2048)),
        (cp.uint16, (1024, 3185)),
        (cp.uint16, (0, 65536)),
        (cp.int16, (0, 256)),
        (cp.int16, (-15, 15)),
        (cp.int16, (-128, 384)),
        (cp.int16, (-400, 400)),
        (cp.int16, (-1024, 2048)),
        (cp.int16, (150, 2048)),
        (cp.int16, (-32768, 32767)),
    ],
)
def test_median_16bit_value_ranges(mode, footprint_shape, int_dtype, irange):
    """Make sure 16-bit cases are robust to various value ranges"""
    footprint = cp.ones(footprint_shape, dtype=bool)
    rng = cp.random.default_rng(123)
    shape = (350, 407)
    img = rng.integers(irange[0], irange[1], shape, dtype=int).astype(int_dtype)
    out = median(
        img,
        footprint,
        mode=mode,
        behavior="ndimage",
        algorithm="wavelet_matrix",
    )
    expected = ndimage.median_filter(img, size=footprint.shape, mode=mode)
    assert_allclose(expected, out)


@pytest.mark.parametrize(
    "algorithm", ["auto", "wavelet_matrix", "sorting", "invalid"]
)
def test_median_algorithm_parameter(algorithm):
    """Call all algorithms for float32 input."""
    footprint = cp.ones((15, 23), dtype=bool)
    mode = "nearest"
    rng = cp.random.default_rng(123)
    shape = (350, 407)  # use anisotropic size
    img = rng.standard_normal(shape, dtype=cp.float32)
    if algorithm == "invalid":
        # 'invalid' is an uncrecognized algorithm
        with pytest.raises(ValueError):
            median(img, footprint, mode=mode, algorithm=algorithm)
    else:
        out = median(img, footprint, mode=mode, algorithm=algorithm)
        expected = ndimage.median_filter(img, size=footprint.shape, mode=mode)
        assert_allclose(expected, out)


def test_median_histogram_algorithm_is_deprecated(monkeypatch):
    """'histogram' warns once and behaves as 'auto'."""
    monkeypatch.setattr(median_module, "_histogram_warned", False)
    footprint = cp.ones((15, 23), dtype=bool)
    rng = cp.random.default_rng(123)
    img = rng.integers(0, 256, (120, 130), dtype=cp.uint8)
    expected = ndimage.median_filter(img, size=footprint.shape, mode="nearest")
    with pytest.warns(FutureWarning, match="histogram"):
        out = median(img, footprint, algorithm="histogram")
    assert_allclose(expected, out)
    # no warning the second time
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = median(img, footprint, algorithm="histogram")
    assert_allclose(expected, out)
    # float input used to be an error
    img = img.astype(cp.float32)
    out = median(img, footprint, algorithm="histogram")
    assert_allclose(
        ndimage.median_filter(img, size=footprint.shape, mode="nearest"), out
    )


def test_median_algorithm_kwargs_ignored():
    img = cp.zeros((20, 20), dtype=cp.uint8)
    with pytest.warns(UserWarning, match="algorithm_kwargs"):
        median(img, algorithm_kwargs={"partitions": 32})


@pytest.mark.parametrize("dtype", [cp.uint8, cp.uint16, cp.float32, cp.float64])
def test_median_preserve_dtype(image, dtype):
    median_image = median(image.astype(dtype), behavior="ndimage")
    assert median_image.dtype == dtype


@pytest.mark.parametrize(
    "img, behavior",
    [
        (cp.random.randint(0, 10, size=(3, 3), dtype=cp.uint8), "rank"),
        # note: upstream scikit-image is 2D-only in rank mode, but cuCIM is nD
        (cp.random.randint(0, 10, size=(3, 3, 3), dtype=cp.uint8), "rank"),
        (cp.random.randint(0, 10, size=(3, 3), dtype=cp.uint8), "ndimage"),
        (cp.random.randint(0, 10, size=(3, 3, 3), dtype=cp.uint8), "ndimage"),
    ],
)
def test_median(img, behavior):
    median(img, behavior=behavior)


def test_median_nonsquare():
    """Test non-uniform footprint.

    https://github.com/rapidsai/cucim/issues/520
    """
    rng = cp.random.default_rng()
    img = rng.integers(0, 256, (128, 128), dtype=cp.uint8)
    footprint = morphology.disk(5)
    mode = "nearest"
    out = median(img, footprint, mode=mode, behavior="ndimage")
    expected = ndimage.median_filter(img, footprint=footprint, mode=mode)
    cp.testing.assert_array_equal(out, expected)
