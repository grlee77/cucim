# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import cupy
import numpy as np

from ._pba_2d import _distance_tranform_arg_check

# Value that marks "no background pixel found yet" in the distances. The
# kernels add 1 to it, so it has to be well below 2**31.
_BIG = 1 << 30

# The kernels use local arrays with one entry per dimension, and the chessboard
# kernel has a loop over the 2**ndim corners of a box.
_MAX_NDIM = 8

_THREADS = 128

# The contiguous axis is processed with one warp per line when the lines are at
# least this long (shorter lines leave most lanes of the warp idle).
_WARP_MIN_LENGTH = 32

_TAXICAB_NAMES = ("taxicab", "cityblock", "manhattan")


def _get_chamfer_src(ndim):
    kernel_directory = os.path.join(os.path.dirname(__file__), "cuda")
    with open(os.path.join(kernel_directory, "chamfer_kernels.h")) as f:
        kernels = f.read()
    defines = f"#define NDIM {ndim}\n#define BIG {_BIG}\n"
    return defines + kernels


@cupy.memoize(for_each_device=True)
def _get_chamfer_kernels(ndim):
    """Return the kernels for images with `ndim` dimensions as a dictionary."""
    module = cupy.RawModule(code=_get_chamfer_src(ndim))
    names = (
        "l1_pass",
        "l1_pass_feature",
        "l1_pass_warp",
        "l1_pass_feature_warp",
        "sum_pass",
        "sum_pass_warp",
        "chessboard",
    )
    return {name: module.get_function(name) for name in names}


@cupy.memoize(for_each_device=True)
def _get_unreached_kernel():
    """Replace the distances of pixels that no background pixel reached by -1."""
    return cupy.ElementwiseKernel(
        in_params="int32 dist",
        out_params="int32 out",
        operation=f"out = (dist >= {_BIG // 2}) ? -1 : dist;",
        name="cucim_cdt_unreached",
    )


def _parse_metric(metric):
    if isinstance(metric, str):
        if metric in _TAXICAB_NAMES:
            return "taxicab"
        elif metric == "chessboard":
            return "chessboard"
        raise ValueError("invalid metric provided")
    raise NotImplementedError(
        "only the metrics 'chessboard' and 'taxicab' (also 'cityblock' and "
        "'manhattan') are supported, not an array of weights"
    )


def _c_strides(shape):
    """Strides in elements of a C-contiguous array."""
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return strides


def _taxicab_passes(dist, with_features):
    """Taxicab distance transform of `dist`, in place (and the flat index of a
    nearest background pixel if `with_features`).

    `dist` must be a C-contiguous int32 array with 0 at the background pixels
    and `_BIG` elsewhere.
    """
    shape = dist.shape
    kernels = _get_chamfer_kernels(dist.ndim)
    feature = None
    if with_features:
        feature = cupy.arange(dist.size, dtype=cupy.int32).reshape(shape)
    for axis, n in enumerate(shape):
        inner = math.prod(shape[axis + 1 :])
        lines = math.prod(shape[:axis]) * inner
        if inner == 1 and n >= _WARP_MIN_LENGTH:
            warps = _THREADS // 32
            grid = ((lines + warps - 1) // warps,)
            args = (np.int64(n), np.int64(lines))
            if with_features:
                kernels["l1_pass_feature_warp"](
                    grid, (_THREADS,), (dist, feature) + args
                )
            else:
                kernels["l1_pass_warp"](grid, (_THREADS,), (dist,) + args)
            continue
        grid = ((lines + _THREADS - 1) // _THREADS,)
        args = (np.int64(n), np.int64(inner), np.int64(lines))
        if with_features:
            kernels["l1_pass_feature"](
                grid, (_THREADS,), (dist, feature) + args
            )
        else:
            kernels["l1_pass"](grid, (_THREADS,), (dist,) + args)
    return dist, feature


def _flat_to_coordinates(feature):
    """Convert flat (C-order) indices to an array of coordinates."""
    shape = feature.shape
    indices = cupy.empty((feature.ndim,) + shape, dtype=cupy.int32)
    for d, stride in enumerate(_c_strides(shape)):
        coordinate = feature // np.int32(stride)
        if d > 0:
            coordinate %= np.int32(shape[d])
        indices[d] = coordinate
    return indices


def _chessboard(background, dist1, return_indices):
    """Chessboard distance from the taxicab distance and a summed-area table.

    `dist1` is the taxicab distance (not yet converted to -1 where there is no
    background).
    """
    shape = background.shape
    ndim = background.ndim
    size = background.size

    # Summed-area table with a leading zero along every axis: the running sums
    # along every axis of the background, in an array with one more element
    # along each axis (the zeros stay zero in the sums).
    sat_shape = tuple(n + 1 for n in shape)
    sat = cupy.zeros(sat_shape, dtype=cupy.int32)
    sat[(slice(1, None),) * ndim] = background
    kernels = _get_chamfer_kernels(ndim)
    for axis, n in enumerate(sat_shape):
        inner = math.prod(sat_shape[axis + 1 :])
        lines = math.prod(sat_shape[:axis]) * inner
        if inner == 1 and n >= _WARP_MIN_LENGTH:
            warps = _THREADS // 32
            kernels["sum_pass_warp"](
                ((lines + warps - 1) // warps,),
                (_THREADS,),
                (sat, np.int64(n), np.int64(lines)),
            )
        else:
            kernels["sum_pass"](
                ((lines + _THREADS - 1) // _THREADS,),
                (_THREADS,),
                (sat, np.int64(n), np.int64(inner), np.int64(lines)),
            )

    out = cupy.empty(shape, dtype=cupy.int32)
    if return_indices:
        indices = cupy.empty((ndim,) + shape, dtype=cupy.int32)
    else:
        indices = cupy.empty(1, dtype=cupy.int32)  # not used
    shape_arg = cupy.asarray(shape, dtype=cupy.int64)
    sat_strides = cupy.asarray(_c_strides(sat.shape), dtype=cupy.int64)

    kernel = kernels["chessboard"]
    kernel(
        ((size + _THREADS - 1) // _THREADS,),
        (_THREADS,),
        (
            sat,
            dist1,
            out,
            indices,
            np.int32(return_indices),
            shape_arg,
            sat_strides,
            np.int64(size),
        ),
    )
    return out, (indices if return_indices else None)


def distance_transform_cdt(
    image,
    metric="chessboard",
    return_distances=True,
    return_indices=False,
    distances=None,
    indices=None,
):
    """Distance transform for chamfer type of transforms.

    This function calculates the distance transform of the `image`, by
    replacing each foreground (non-zero) element, with its
    shortest distance to the background (any zero-valued element).

    In addition to the distance transform, the feature transform can
    be calculated. In this case the index of the closest background
    element to each foreground element is returned in a separate array.

    Parameters
    ----------
    image : cupy.ndarray
        Input. Values of 0 are treated as background.
    metric : {'chessboard', 'taxicab'}, optional
        The `metric` determines the type of distance that is computed:
        the chessboard distance (the largest difference of the coordinates,
        also called the Chebyshev distance) or the taxicab distance (the sum
        of the absolute differences of the coordinates, also called the
        cityblock or Manhattan distance). ``'cityblock'`` and ``'manhattan'``
        are also valid, and map to ``'taxicab'``. The default is
        ``'chessboard'``. Note: arrays of weights as `metric` (which
        :func:`scipy.ndimage.distance_transform_cdt` accepts) are not
        supported.
    return_distances : bool, optional
        Whether to calculate the distance transform.
        Default is True.
    return_indices : bool, optional
        Whether to calculate the feature transform.
        Default is False.
    distances : cupy.ndarray of dtype int32, optional
        An output array to store the calculated distance transform, instead of
        returning it. `return_distances` must be True. It must be the same
        shape as `image`.
    indices : cupy.ndarray of dtype int32, optional
        An output array to store the calculated feature transform, instead of
        returning it. `return_indices` must be True. Its shape must be
        ``(image.ndim,) + image.shape``.

    Returns
    -------
    distances : cupy.ndarray of dtype int32, optional
        The calculated distance transform. Returned only when
        `return_distances` is True, and `distances` is not supplied. It will
        have the same shape as the input array. If `image` does not contain
        any background element the distances are -1 (as in SciPy).
    indices : cupy.ndarray of dtype int32, optional
        The calculated feature transform. It has an input-shaped array for each
        dimension of the input. Returned only when `return_indices` is True,
        and `indices` is not supplied.

    Notes
    -----
    Both metrics are exact. The indices returned are the coordinates of a
    background element at the minimal distance, but where several background
    elements are at the same distance the one that is returned can differ from
    the one returned by :func:`scipy.ndimage.distance_transform_cdt`.

    The number of elements of `image` must be smaller than 2**31.

    See Also
    --------
    cucim.core.operations.morphology.distance_transform_edt
        Exact Euclidean distance transform.
    """
    distances_inplace = isinstance(distances, cupy.ndarray)
    indices_inplace = isinstance(indices, cupy.ndarray)
    _distance_tranform_arg_check(
        distances_inplace, indices_inplace, return_distances, return_indices
    )
    metric = _parse_metric(metric)

    image = cupy.asarray(image)
    shape = image.shape
    ndim = image.ndim
    if ndim == 0:
        raise ValueError("the input must have at least one dimension")
    if ndim > _MAX_NDIM:
        raise NotImplementedError(
            f"at most {_MAX_NDIM} dimensions are supported"
        )
    if image.size >= 2**31:
        raise ValueError("images with 2**31 or more elements are not supported")
    if sum(shape) >= _BIG // 2:
        raise ValueError("the sum of the axis sizes of the image is too large")

    if distances_inplace:
        if distances.dtype != cupy.int32:
            raise ValueError("distances must be of int32 type")
        if distances.shape != shape:
            raise ValueError("distances has wrong shape")
    if indices_inplace:
        if indices.dtype != cupy.int32:
            raise ValueError("indices array must be int32")
        if indices.shape != (ndim,) + shape:
            raise ValueError("indices array has wrong shape")

    if image.size == 0:
        dist = cupy.empty(shape, dtype=cupy.int32)
        idx = cupy.empty((ndim,) + shape, dtype=cupy.int32)
    else:
        background = image == 0
        dist = cupy.where(background, np.int32(0), np.int32(_BIG))
        dist = cupy.ascontiguousarray(dist, dtype=cupy.int32)

        if metric == "taxicab":
            dist, feature = _taxicab_passes(dist, return_indices)
            idx = _flat_to_coordinates(feature) if return_indices else None
        else:
            # The chessboard distance is bounded by the taxicab distance
            # (d_chessboard <= d_taxicab <= ndim * d_chessboard). The taxicab
            # distance of every pixel is needed anyway, so it is computed first.
            dist1, _ = _taxicab_passes(dist, False)
            dist, idx = _chessboard(background, dist1, return_indices)
        if metric == "taxicab":
            dist = _get_unreached_kernel()(dist)

    result = []
    if return_distances:
        if distances_inplace:
            distances[...] = dist
        else:
            result.append(dist)
    if return_indices:
        if indices_inplace:
            indices[...] = idx
        else:
            result.append(idx)
    if len(result) == 0:
        return None
    return result[0] if len(result) == 1 else tuple(result)
