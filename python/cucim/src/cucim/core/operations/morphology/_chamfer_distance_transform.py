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

# Limit of the number of dimensions (the kernels are compiled for every number
# of dimensions that is used).
_MAX_NDIM = 8

_THREADS = 128

# Below this number of lines a pass over the lines of an axis does not occupy
# the GPU with one thread per line.
_FEW_LINES = 1 << 17

# Shortest piece of a line that a thread handles when the lines are split.
_MIN_CHUNK = 64

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
        "flat_to_coordinates",
        "l1_chunk_summary",
        "l1_chunk_pass",
        "minmax_pass",
        "minmax_combine",
    )
    return {name: module.get_function(name) for name in names}


@cupy.memoize(for_each_device=True)
def _get_initial_kernel():
    """0 at the background pixels and _BIG elsewhere."""
    return cupy.ElementwiseKernel(
        in_params="bool background",
        out_params="int32 dist",
        operation=f"dist = background ? 0 : {_BIG};",
        name="cucim_cdt_initial",
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


def _taxicab_passes(dist, with_features, final=False, axes=None):
    """Taxicab distance transform of `dist`, in place (and the flat index of a
    nearest background pixel if `with_features`).

    `dist` must be a C-contiguous int32 array with 0 at the background pixels
    and `_BIG` elsewhere. With `final` the last pass stores -1 for the pixels
    that no background pixel reached. Only the axes in `axes` are processed (all
    by default, in the order given); the first of them is the one where the
    feature of every pixel is still its own flat index.
    """
    shape = dist.shape
    kernels = _get_chamfer_kernels(dist.ndim)
    feature = None
    if with_features:
        feature = cupy.empty(shape, dtype=cupy.int32)
    if axes is None:
        axes = range(len(shape))
    axes = tuple(axes)
    for axis in axes:
        n = shape[axis]
        is_last = final and axis == axes[-1]
        inner = math.prod(shape[axis + 1 :])
        lines = math.prod(shape[:axis]) * inner
        if inner == 1 and n >= _WARP_MIN_LENGTH:
            warps = _THREADS // 32
            grid = ((lines + warps - 1) // warps,)
            args = (np.int64(n), np.int64(lines), np.int32(is_last))
            if with_features:
                args = args + (np.int32(axis == axes[0]),)
                kernels["l1_pass_feature_warp"](
                    grid, (_THREADS,), (dist, feature) + args
                )
            else:
                kernels["l1_pass_warp"](grid, (_THREADS,), (dist,) + args)
            continue
        chunks = min(-(-_FEW_LINES // lines), n // _MIN_CHUNK)
        if chunks >= 2:
            # few lines: every line is split into chunks (see l1_chunk_pass)
            chunk_len = -(-n // chunks)
            chunks = -(-n // chunk_len)
            forward = cupy.empty(chunks * lines, dtype=cupy.uint64)
            backward = cupy.empty(chunks * lines, dtype=cupy.uint64)
            grid = ((lines * chunks + _THREADS - 1) // _THREADS,)
            args = (
                np.int64(n),
                np.int64(inner),
                np.int64(lines),
                np.int64(chunks),
                np.int64(chunk_len),
            )
            flags = (np.int32(axis == axes[0]), np.int32(with_features))
            fe = feature if with_features else dist
            kernels["l1_chunk_summary"](
                grid,
                (_THREADS,),
                (dist, fe, forward, backward) + args + flags,
            )
            kernels["l1_chunk_pass"](
                grid,
                (_THREADS,),
                (dist, fe, forward, backward)
                + args
                + (np.int32(is_last),)
                + flags,
            )
            continue
        grid = ((lines + _THREADS - 1) // _THREADS,)
        args = (
            np.int64(n),
            np.int64(inner),
            np.int64(lines),
            np.int32(is_last),
        )
        if with_features:
            args = args + (np.int32(axis == axes[0]),)
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
    _get_chamfer_kernels(feature.ndim)["flat_to_coordinates"](
        ((feature.size + _THREADS - 1) // _THREADS,),
        (_THREADS,),
        (
            feature,
            indices,
            cupy.asarray(shape, dtype=cupy.int64),
            np.int64(feature.size),
        ),
    )
    return indices


def _chessboard_passes(dist, with_features):
    """Chessboard distance transform of `dist` (see `_taxicab_passes`).

    The distance along the contiguous axis is the 1D taxicab distance. Every
    other axis then needs the transform min(max(|x - x'|, f(x'))) of the result,
    which takes one thread per line (the lines are not contiguous). Returns the
    distances (-1 where there is no background) and the flat feature indices.
    """
    shape = dist.shape
    ndim = dist.ndim
    dist, feature = _taxicab_passes(
        dist, with_features, final=(ndim == 1), axes=(ndim - 1,)
    )
    if ndim == 1:
        return dist, feature
    out = cupy.empty_like(dist)
    kernels = _get_chamfer_kernels(ndim)
    # Without features, the arrays for them are not used.
    fout = cupy.empty_like(dist) if with_features else out
    if not with_features:
        feature = out
    for axis in range(ndim - 2, -1, -1):
        n = shape[axis]
        inner = math.prod(shape[axis + 1 :])
        lines = math.prod(shape[:axis]) * inner
        # With few lines the forward and the backward sweeps of a line run in
        # different threads.
        split = lines < _FEW_LINES
        sweeps = 2 if split else 1
        stack = cupy.empty((sweeps,) + shape + (2,), dtype=cupy.int32)
        if split:
            out2 = cupy.empty_like(dist)
            fout2 = cupy.empty_like(dist) if with_features else out2
        else:
            out2, fout2 = out, fout
        final = int(axis == 0)
        kernels["minmax_pass"](
            ((lines + _THREADS - 1) // _THREADS, sweeps),
            (_THREADS,),
            (
                dist,
                out,
                feature,
                fout,
                out2,
                fout2,
                stack,
                np.int64(n),
                np.int64(inner),
                np.int64(lines),
                np.int32(final),
                np.int32(with_features),
                np.int32(split),
            ),
        )
        if split:
            size = dist.size
            kernels["minmax_combine"](
                ((size + _THREADS - 1) // _THREADS,),
                (_THREADS,),
                (
                    out,
                    out2,
                    fout,
                    fout2,
                    np.int64(size),
                    np.int32(final),
                    np.int32(with_features),
                ),
            )
        dist, out = out, dist
        if with_features:
            feature, fout = fout, feature
    return dist, (feature if with_features else None)


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
        background = cupy.ascontiguousarray(image == 0)
        dist = _get_initial_kernel()(background)

        if metric == "taxicab":
            dist, feature = _taxicab_passes(dist, return_indices, final=True)
            idx = _flat_to_coordinates(feature) if return_indices else None
        else:
            dist, feature = _chessboard_passes(dist, return_indices)
            idx = _flat_to_coordinates(feature) if return_indices else None

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
