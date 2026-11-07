# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import math
from warnings import warn

import cupy as cp
import numpy as np

import cucim.skimage._vendored.ndimage as ndi

from ._median_wavelet import (
    _can_use_wavelet_matrix,
    _has_enough_memory,
    _median_wavelet_filter,
    _median_wavelet_filter_3d,
)

_histogram_warned = False


def _get_wavelet_planes(image, footprint_shape):
    """Find the images that the wavelet matrix filters.

    Returns ``(channel_axis, plane, plane_footprint_shape)``. For a 2D image,
    or a 3D volume with a footprint without axes of size 1, ``channel_axis``
    is None and ``plane`` is the image. For a 3D image with a footprint of
    size 1 along exactly one axis, ``channel_axis`` is that axis, ``plane`` is
    the first plane (a view) and the footprint shape is that of the other two
    axes. Otherwise ``plane`` is None.
    """
    if image.ndim == 2:
        return None, image, footprint_shape
    if image.ndim == 3:
        ones = [i for i, s in enumerate(footprint_shape) if s == 1]
        if not ones:
            return None, image, footprint_shape
        if len(ones) == 1 and image.shape[ones[0]] > 0:
            axis = ones[0]
            plane = image[(slice(None),) * axis + (0,)]
            shape = tuple(s for i, s in enumerate(footprint_shape) if i != axis)
            return axis, plane, shape
    return None, None, footprint_shape


def median(
    image,
    footprint=None,
    out=None,
    mode="nearest",
    cval=0.0,
    behavior="ndimage",
    *,
    algorithm="auto",
    algorithm_kwargs={},
):
    """Return local median of an image.

    Parameters
    ----------
    image : array-like
        Input image.
    footprint : ndarray, tuple of int, or None
        If ``None``, ``footprint`` will be a N-D array with 3 elements for each
        dimension (e.g., vector, square, cube, etc.). If `footprint` is a
        tuple of integers, it will be an array of ones with the given shape.
        Otherwise, ``footprint`` is an N-D array of 1's and 0's with the same
        number of dimensions as ``image``.
        Note that upstream scikit-image currently does not support supplying
        a tuple for `footprint`. It is added here to avoid overhead of
        generating a small weights array in cases where it is not needed.
    out : ndarray, (same dtype as image), optional
        If None, a new array is allocated.
    mode : {'reflect', 'constant', 'nearest', 'mirror','‘wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        ``cval`` is the value when mode is equal to 'constant'. ``mode`` is
        only used when ``behavior='ndimage'``.
        Default is 'nearest'.
    cval : scalar, optional
        Value to fill past edges of input if mode is 'constant'. ``cval`` is
        only used when ``behavior='ndimage'``. Default is 0.0.
    behavior : {'ndimage', 'rank'}, optional
        Behavior 'ndimage' behaves like `cupyx.scipy.ndimage.median_filter`,
        while 'rank' uses :func:`cucim.skimage.filters.rank.median`. cuCIM rank
        filters use reflected boundary extension with a constant footprint
        size and support N-D images. This differs from scikit-image rank
        filters, which crop neighborhoods near image boundaries and support
        only 2-D and 3-D images. Default is 'ndimage'.

    Other Parameters
    ----------------
    algorithm : {'auto', 'wavelet_matrix', 'histogram', 'sorting'}
        Determines which algorithm is used to compute the median. The default
        of 'auto' will attempt to use a wavelet matrix-based algorithm for 2D
        images with square or rectangular footprints with odd length on each
        axis. Falls back to a sorting-based algorithm when the wavelet matrix
        is not suitable.

        - 'wavelet_matrix': Fast wavelet matrix algorithm [1]_. Best for larger
          footprints on 2D images.
        - 'sorting': Sorting-based algorithm via scipy.ndimage. Works for any
          dtype and dimensionality, but is slower than the wavelet_matrix
          approach for larger footprints.
        - 'histogram': Deprecated. The histogram-based algorithm that this
          option used to select was removed. It now behaves as 'auto' (with a
          one-time warning) and gives identical results.
        - 'auto': Automatically selects the best algorithm.

        Note: this parameter is cuCIM-specific and does not exist in upstream
        scikit-image.
    algorithm_kwargs : dict
        Deprecated and ignored (a warning is issued if it is not empty). It
        was only used by the removed 'histogram' algorithm.
        Note: this parameter is cuCIM-specific and does not exist in upstream
        scikit-image.

    Returns
    -------
    out : N-D array
        Output image with the same shape as the input image.

    See also
    --------
    skimage.filters.rank.median : Rank-based implementation of the median
        filtering offering more flexibility with additional parameters but
        dedicated for unsigned integer images.

    Notes
    -----
    For 2D images and full rectangular footprints, a wavelet matrix-based
    median filter [1]_ provides the fastest performance for larger kernel
    sizes (the run time is independent of the footprint size). All integer and
    floating point dtypes are supported. It builds a succinct data structure
    that answers range median queries in time proportional to the number of
    bits of the values (for dtypes wider than 16 bits, of the ranks of the
    values in the sorted image) times the number of bits of the image width.
    It uses roughly 70 (uint8) to 210 (float32) bytes of device memory per
    pixel. If that would take more than half of the available device memory,
    the image is filtered in slabs of rows, which takes a little longer but
    needs much less memory (about 350 MB for a 8192 x 8192 uint8 image instead
    of 4.6 GB, with a 31 x 31 footprint). 'auto' does not choose the wavelet
    matrix algorithm if not even a slab of a few dozen rows fits. Smaller
    footprints, other dimensionalities and footprints that are not full
    rectangles use the sorting-based algorithm.

    For 3D volumes and full box footprints the same data structure is used
    with the rows of all the planes stacked, and the query handles the window
    of every plane (its cost is proportional to the depth of the window). It is
    faster than sorting from about 150 footprint elements, e.g. 4.4 (uint8) to
    1.3 (float32) times for a 5 x 5 x 7 footprint, and 11 and 33 times for
    7 x 7 x 7 and 9 x 9 x 9 footprints of uint8 (measured on 96^3 to 128^3
    volumes).

    A 3D image with a footprint of size 1 along one axis (for example an RGB
    image and a footprint of shape ``(h, w, 1)``) is filtered one plane at a
    time with the wavelet matrix algorithm, which gives the same result as
    filtering each channel separately.

    When algorithm='auto', the best available algorithm is selected based on
    image dtype, dimensionality, and footprint size.

    References
    ----------
    .. [1] Y. Moroto and N. Umetani., "Constant Time Median Filter Using 2D
       Wavelet Matrix," in ACM Trans. Graph. Volume 41, Issue 6,
       Article No.: 267, December 2022.
       :DOI:`10.1145/3550454.3555512`

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage.morphology import disk
    >>> from cucim.skimage.filters import median
    >>> img = cp.array(data.camera())
    >>> med = median(img, disk(5))

    """
    if behavior == "rank":
        if mode != "nearest" or not np.isclose(cval, 0.0):
            warn(
                "Change 'behavior' to 'ndimage' if you want to use the "
                "parameters 'mode' or 'cval'. They will be discarded "
                "otherwise.",
                stacklevel=2,
            )
        from .rank import median as rank_median

        if isinstance(footprint, tuple):
            if len(footprint) != image.ndim:
                raise ValueError(
                    "tuple footprint must have ndim matching image"
                )
            footprint = cp.ones(footprint, dtype=bool)
        return rank_median(image, footprint=footprint, out=out)

    if footprint is None:
        footprint_shape = (3,) * image.ndim
    elif isinstance(footprint, tuple):
        if len(footprint) != image.ndim:
            raise ValueError("tuple footprint must have ndim matching image")
        footprint_shape = footprint
        footprint = None
    else:
        footprint_shape = footprint.shape

    # Validate algorithm parameter
    valid_algorithms = {"auto", "wavelet_matrix", "histogram", "sorting"}
    if algorithm not in valid_algorithms:
        raise ValueError(
            f"unknown algorithm: {algorithm}. "
            f"Valid options are: {sorted(valid_algorithms)}"
        )

    if algorithm == "histogram":
        global _histogram_warned
        if not _histogram_warned:
            _histogram_warned = True
            warn(
                "algorithm='histogram' is deprecated: the histogram-based "
                "algorithm was removed and the algorithm is now chosen as for "
                "algorithm='auto'. The output is identical.",
                FutureWarning,
                stacklevel=2,
            )
        algorithm = "auto"

    if algorithm_kwargs:
        warn(
            f"algorithm_kwargs={algorithm_kwargs} ignored: it was only used "
            "by the removed histogram-based algorithm",
            stacklevel=2,
        )

    # Check algorithm compatibility
    can_use_wavelet = False
    wm_reason = None
    # The wavelet matrix filters 2D images. A 3D image with a footprint of
    # size 1 along one axis (e.g. an RGB image with a footprint of shape
    # (h, w, 1)) is filtered plane by plane along that axis.
    channel_axis, plane, plane_footprint_shape = _get_wavelet_planes(
        image, footprint_shape
    )

    if algorithm in ["auto", "wavelet_matrix"]:
        # Wavelet matrix supports rectangular footprints with odd dimensions
        # and requires ALL elements in the footprint to be non-zero (full rectangle)
        if plane is None:
            can_use_wavelet = False
            wm_reason = (
                "Only 2D images, 3D volumes and 3D images with a footprint "
                "of size 1 along one axis (e.g. RGB), are supported"
            )
        elif footprint is not None and not bool(cp.all(cp.asarray(footprint))):
            # Footprint has zeros - wavelet matrix only supports full rectangles
            can_use_wavelet = False
            wm_reason = (
                "Footprint contains zeros. Wavelet matrix only supports "
                "full rectangular footprints (all elements non-zero)."
            )
        else:
            can_use_wavelet, wm_reason = _can_use_wavelet_matrix(
                plane, footprint_shape=plane_footprint_shape, mode=mode
            )

    # Explicit algorithm requests with validation
    if algorithm == "wavelet_matrix" and not can_use_wavelet:
        raise ValueError(
            "The wavelet_matrix algorithm was requested, but it cannot "
            f"be used for this image and footprint (reason: {wm_reason})."
        )

    # Algorithm selection for 'auto' mode
    # Priority: wavelet_matrix > sorting
    # Sorting is faster for small footprints. On an RTX A6000 with 2048 x 2048
    # images, wavelet matrix overtakes sorting at 11 x 11 (121 elements) for
    # uint16 (12.9 vs 30 ms) and uint8 (6.1 vs 7.1 ms), while sorting is
    # still faster at 9 x 9 (81 elements). Types wider than 16 bits are
    # slower with the wavelet matrix (they use ranks, see
    # `_median_wavelet._uses_rank_mode`): float32 takes 22 ms and sorting
    # still wins at 11 x 11 (10.8 ms), but not at 13 x 13 (96 ms).
    footprint_size = math.prod(footprint_shape)
    wavelet_min_size = 100 if image.dtype.itemsize <= 2 else 150
    if image.ndim == 3 and channel_axis is None:
        # volumes: on 96^3 to 128^3 volumes the wavelet matrix is faster than
        # sorting from about 150 elements (e.g. 5 x 5 x 7: 4.4 (uint8),
        # 2.4 (uint16) and 1.3 (float32) times faster) and slower below
        # (e.g. 5 x 5 x 5: 20 times slower for float32)
        wavelet_min_size = 150
    use_wavelet = False

    if algorithm == "auto":
        if (
            can_use_wavelet
            and footprint_size > wavelet_min_size
            and _has_enough_memory(plane, plane_footprint_shape)
        ):
            use_wavelet = True
    elif algorithm == "wavelet_matrix":
        use_wavelet = True

    # Try wavelet matrix first
    if use_wavelet:
        try:
            # as in SciPy, a user-provided `out` can be an array or a dtype
            output_array_provided = False
            out_dtype = None
            if out is not None:
                output_array_provided = isinstance(out, cp.ndarray)
                if not output_array_provided:
                    try:
                        out_dtype = cp.dtype(out)
                    except TypeError:
                        raise TypeError(
                            "out must be either a cupy.array or a valid input "
                            "to cupy.dtype"
                        )

            # Support rectangular footprints
            radius_y = plane_footprint_shape[-2] // 2
            radius_x = plane_footprint_shape[-1] // 2
            if channel_axis is None and image.ndim == 3:
                temp = _median_wavelet_filter_3d(
                    image,
                    radius_z=plane_footprint_shape[0] // 2,
                    radius_y=radius_y,
                    radius_x=radius_x,
                    mode=mode,
                    cval=cval,
                )
            elif channel_axis is None:
                temp = _median_wavelet_filter(
                    image,
                    radius_y=radius_y,
                    radius_x=radius_x,
                    mode=mode,
                    cval=cval,
                )
            else:
                temp = cp.empty_like(image)
                for c in range(image.shape[channel_axis]):
                    idx = (slice(None),) * channel_axis + (c,)
                    temp[idx] = _median_wavelet_filter(
                        image[idx],
                        radius_y=radius_y,
                        radius_x=radius_x,
                        mode=mode,
                        cval=cval,
                    )

            if output_array_provided:
                out[:] = temp
            else:
                if out_dtype is not None:
                    temp = temp.astype(out_dtype, copy=False)
                out = temp
            return out
        except cp.cuda.memory.OutOfMemoryError as e:
            # fall back to sorting if there is not enough memory even for
            # slabs of rows
            if algorithm == "wavelet_matrix":
                raise  # Re-raise if explicitly requested
            warn(
                f"Wavelet matrix median ran out of memory: {e}\n"
                "Falling back to sorting-based median."
            )

    size = footprint_shape if footprint is None else None
    return ndi.median_filter(
        image, size=size, footprint=footprint, output=out, mode=mode, cval=cval
    )
