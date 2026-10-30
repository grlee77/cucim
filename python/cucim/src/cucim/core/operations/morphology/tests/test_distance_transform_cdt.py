# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
import scipy.ndimage as ndi_cpu

from cucim.core.operations.morphology import distance_transform_cdt

METRICS = ["taxicab", "chessboard"]


def random_image(shape, bg_fraction, seed=123, ensure_background=True):
    """uint8 image with about `bg_fraction` of the pixels equal to 0."""
    rng = np.random.default_rng(seed)
    img = (rng.random(shape) >= bg_fraction).astype(np.uint8)
    if ensure_background and img.all():
        img.flat[img.size // 2] = 0
    return img


def brute_force(img, metric):
    """Distance to the nearest background pixel by looking at every pair."""
    background = np.argwhere(img == 0)
    out = np.empty(img.shape, dtype=np.int64)
    for index in np.ndindex(*img.shape):
        diff = np.abs(background - np.asarray(index))
        out[index] = (
            diff.sum(axis=1) if metric == "taxicab" else diff.max(axis=1)
        ).min()
    return out


def distance_from_indices(img_shape, indices, metric):
    """Distance of every pixel to the pixel that `indices` refers to."""
    coords = np.indices(img_shape)
    diff = np.abs(np.asarray(indices) - coords)
    return diff.sum(axis=0) if metric == "taxicab" else diff.max(axis=0)


@pytest.mark.parametrize(
    "shape",
    [
        (17,),
        (1, 40),
        (40, 1),
        (23, 31),
        (64, 64),
        (1, 1, 1),
        (9, 10, 11),
        (3, 1, 20),
        (4, 5, 6, 7),
        (3, 3, 3, 3, 3),
    ],
)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("bg_fraction", [0.001, 0.05, 0.5, 0.97])
def test_distance_transform_cdt_matches_scipy(shape, metric, bg_fraction):
    img = random_image(shape, bg_fraction)
    out = distance_transform_cdt(cp.asarray(img), metric=metric)
    expected = ndi_cpu.distance_transform_cdt(img, metric=metric)
    assert out.dtype == cp.int32
    assert out.shape == img.shape
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("shape", [(12,), (9, 11), (5, 6, 7)])
def test_distance_transform_cdt_matches_brute_force(metric, shape):
    """Independent of SciPy: the definition of the distance."""
    img = random_image(shape, 0.1, seed=7)
    out = distance_transform_cdt(cp.asarray(img), metric=metric)
    np.testing.assert_array_equal(cp.asnumpy(out), brute_force(img, metric))


@pytest.mark.parametrize("alias", ["cityblock", "manhattan"])
def test_distance_transform_cdt_taxicab_aliases(alias):
    img = cp.asarray(random_image((20, 30), 0.1))
    cp.testing.assert_array_equal(
        distance_transform_cdt(img, metric=alias),
        distance_transform_cdt(img, metric="taxicab"),
    )


def test_distance_transform_cdt_default_metric_is_chessboard():
    img = cp.asarray(random_image((20, 30), 0.1))
    cp.testing.assert_array_equal(
        distance_transform_cdt(img),
        distance_transform_cdt(img, metric="chessboard"),
    )


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("shape", [(15,), (8, 9), (4, 5, 6)])
def test_distance_transform_cdt_single_background_pixel(metric, shape):
    img = np.ones(shape, dtype=np.uint8)
    img[tuple(s // 2 for s in shape)] = 0
    out = distance_transform_cdt(cp.asarray(img), metric=metric)
    expected = ndi_cpu.distance_transform_cdt(img, metric=metric)
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "shape", [(7,), (6, 8), (3, 4, 5), (50,), (40, 50), (3, 4, 70)]
)
def test_distance_transform_cdt_no_background(metric, shape):
    """SciPy returns -1 everywhere, and every pixel is its own feature."""
    img = np.ones(shape, dtype=np.uint8)
    out, ind = distance_transform_cdt(
        cp.asarray(img), metric=metric, return_indices=True
    )
    expected, expected_ind = ndi_cpu.distance_transform_cdt(
        img, metric=metric, return_indices=True
    )
    cp.testing.assert_array_equal(out, expected)
    assert (cp.asnumpy(out) == -1).all()
    cp.testing.assert_array_equal(ind, expected_ind)


@pytest.mark.parametrize("metric", METRICS)
def test_distance_transform_cdt_all_background(metric):
    img = np.zeros((6, 7), dtype=np.uint8)
    out, ind = distance_transform_cdt(
        cp.asarray(img), metric=metric, return_indices=True
    )
    assert (cp.asnumpy(out) == 0).all()
    np.testing.assert_array_equal(cp.asnumpy(ind), np.indices(img.shape))


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "shape",
    [(30,), (17, 23), (6, 7, 8), (3, 4, 3, 4), (200,), (33, 70), (5, 9, 65)],
)
@pytest.mark.parametrize("bg_fraction", [0.02, 0.3])
def test_distance_transform_cdt_indices(metric, shape, bg_fraction):
    """The indices are a nearest background pixel (not necessarily SciPy's)."""
    img = random_image(shape, bg_fraction)
    dist, ind = distance_transform_cdt(
        cp.asarray(img), metric=metric, return_indices=True
    )
    assert ind.dtype == cp.int32
    assert ind.shape == (img.ndim,) + img.shape
    dist = cp.asnumpy(dist)
    ind = cp.asnumpy(ind)
    # the feature of every pixel is a background pixel
    assert (img[tuple(ind)] == 0).all()
    # and it is at the distance that was found
    np.testing.assert_array_equal(
        distance_from_indices(img.shape, ind, metric), dist
    )


@pytest.mark.parametrize("metric", METRICS)
def test_distance_transform_cdt_return_flags(metric):
    img = cp.asarray(random_image((12, 14), 0.1))
    dist, ind = distance_transform_cdt(img, metric=metric, return_indices=True)
    only_ind = distance_transform_cdt(
        img, metric=metric, return_distances=False, return_indices=True
    )
    only_dist = distance_transform_cdt(img, metric=metric)
    cp.testing.assert_array_equal(dist, only_dist)
    cp.testing.assert_array_equal(ind, only_ind)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("ndim", [2, 3])
def test_distance_transform_cdt_inplace_outputs(metric, ndim):
    img = cp.asarray(random_image((10,) * ndim, 0.1))
    expected_dist, expected_ind = distance_transform_cdt(
        img, metric=metric, return_indices=True
    )
    distances = cp.empty(img.shape, dtype=cp.int32)
    indices = cp.empty((ndim,) + img.shape, dtype=cp.int32)
    assert (
        distance_transform_cdt(
            img,
            metric=metric,
            distances=distances,
            indices=indices,
            return_indices=True,
        )
        is None
    )
    cp.testing.assert_array_equal(distances, expected_dist)
    cp.testing.assert_array_equal(indices, expected_ind)

    # only one of them in place
    distances[...] = 0
    ind = distance_transform_cdt(
        img,
        metric=metric,
        distances=distances,
        return_indices=True,
    )
    cp.testing.assert_array_equal(distances, expected_dist)
    cp.testing.assert_array_equal(ind, expected_ind)


def test_distance_transform_cdt_invalid_arguments():
    img = cp.asarray(random_image((8, 8), 0.1))
    with pytest.raises(ValueError, match="invalid metric"):
        distance_transform_cdt(img, metric="garbage")
    with pytest.raises(NotImplementedError):
        distance_transform_cdt(img, metric=np.ones((3, 3), dtype=bool))
    with pytest.raises(RuntimeError):
        distance_transform_cdt(
            img, return_distances=False, return_indices=False
        )
    with pytest.raises(RuntimeError):
        distance_transform_cdt(
            img,
            return_distances=False,
            return_indices=True,
            distances=cp.empty(img.shape, dtype=cp.int32),
        )
    with pytest.raises(RuntimeError):
        distance_transform_cdt(
            img,
            return_indices=False,
            indices=cp.empty((2,) + img.shape, dtype=cp.int32),
        )
    with pytest.raises(ValueError, match="int32"):
        distance_transform_cdt(
            img, distances=cp.empty(img.shape, dtype=cp.float64)
        )
    with pytest.raises(ValueError, match="shape"):
        distance_transform_cdt(img, distances=cp.empty((8, 9), dtype=cp.int32))
    with pytest.raises(ValueError, match="int32"):
        distance_transform_cdt(
            img,
            return_indices=True,
            indices=cp.empty((2, 8, 8), dtype=cp.int64),
        )
    with pytest.raises(ValueError, match="shape"):
        distance_transform_cdt(
            img,
            return_indices=True,
            indices=cp.empty((8, 8), dtype=cp.int32),
        )
    with pytest.raises(ValueError):
        distance_transform_cdt(cp.asarray(3))
    with pytest.raises(NotImplementedError):
        distance_transform_cdt(cp.zeros((1,) * 9))


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "dtype",
    [
        bool,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
    ],
)
def test_distance_transform_cdt_input_dtypes(metric, dtype):
    """Any non-zero value is foreground."""
    rng = np.random.default_rng(5)
    values = rng.integers(0, 4, size=(13, 17))
    values[0, 0] = 0
    out = distance_transform_cdt(
        cp.asarray(values.astype(dtype)), metric=metric
    )
    expected = ndi_cpu.distance_transform_cdt(values != 0, metric=metric)
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("metric", METRICS)
def test_distance_transform_cdt_nan_is_foreground(metric):
    values = np.ones((9, 9), dtype=np.float32)
    values[2, 3] = 0
    values[5, 6] = np.nan
    out = distance_transform_cdt(cp.asarray(values), metric=metric)
    expected = ndi_cpu.distance_transform_cdt(values != 0, metric=metric)
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "view", ["fortran", "transposed", "strided", "reversed"]
)
def test_distance_transform_cdt_noncontiguous_input(metric, view):
    base = random_image((40, 50), 0.1)
    if view == "fortran":
        img = np.asfortranarray(base[:20, :30])
    elif view == "transposed":
        img = base[:20, :30].T
    elif view == "strided":
        img = base[::2, ::2]
    else:
        img = base[::-1, ::-1]
    out = distance_transform_cdt(cp.asarray(img), metric=metric)
    expected = ndi_cpu.distance_transform_cdt(
        np.ascontiguousarray(img), metric=metric
    )
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("metric", METRICS)
def test_distance_transform_cdt_empty(metric):
    out = distance_transform_cdt(
        cp.zeros((0, 5), dtype=cp.uint8), metric=metric
    )
    assert out.shape == (0, 5)
    assert out.dtype == cp.int32


@pytest.mark.parametrize("metric", METRICS)
def test_distance_transform_cdt_larger_image(metric):
    """Several blocks of threads, and lines longer than a block."""
    img = random_image((300, 700), 0.002, seed=9)
    out = distance_transform_cdt(cp.asarray(img), metric=metric)
    expected = ndi_cpu.distance_transform_cdt(img, metric=metric)
    cp.testing.assert_array_equal(out, expected)
