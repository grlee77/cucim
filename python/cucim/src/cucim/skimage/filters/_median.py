# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import math
from warnings import warn

import cupy as cp
import numpy as np

import cucim.skimage._vendored.ndimage as ndi

from ._median_hist import KernelResourceError, _can_use_histogram, _median_hist
from ._median_wavelet import _can_use_wavelet_matrix, _median_wavelet_filter


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
        Otherwise, if ``behavior=='rank'``, ``footprint`` is a 2-D array of 1's
        and 0's. If ``behavior=='ndimage'``, ``footprint`` is a N-D array of
        1's and 0's with the same number of dimension as ``image``.
        Note that upstream scikit-image currently does not support supplying
        a tuple for `footprint`. It is added here to avoid overhead of
        generating a small weights array in cases where it is not needed.
    out : ndarray, (same dtype as image), optional
        If None, a new array is allocated.
    mode : {'reflect', 'constant', 'nearest', 'mirror','‘wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        ``cval`` is the value when mode is equal to 'constant'.
        Default is 'nearest'.

        .. versionadded:: 0.15
           ``mode`` is used when ``behavior='ndimage'``.
    cval : scalar, optional
        Value to fill past edges of input if mode is 'constant'. Default is 0.0

        .. versionadded:: 0.15
           ``cval`` was added in 0.15 is used when ``behavior='ndimage'``.
    behavior : {'ndimage', 'rank'}, optional
        Either to use the old behavior (i.e., < 0.15) or the new behavior.
        The old behavior will call the :func:`skimage.filters.rank.median`.
        The new behavior will call the :func:`scipy.ndimage.median_filter`.
        Default is 'ndimage'.

        .. versionadded:: 0.15
           ``behavior`` is introduced in 0.15
        .. versionchanged:: 0.16
           Default ``behavior`` has been changed from 'rank' to 'ndimage'

    Other Parameters
    ----------------
    algorithm : {'auto', 'wavelet_matrix', 'histogram', 'sorting'}
        Determines which algorithm is used to compute the median. The default
        of 'auto' will attempt to use a wavelet matrix-based algorithm for 2D
        images with 8 or 16-bit unsigned integer data types and sufficiently
        large footprints. Falls back to histogram-based or sorting-based
        algorithms when wavelet matrix is not suitable.

        - 'wavelet_matrix': Fast wavelet matrix algorithm [2]_. Best for larger
          footprints on 2D uint8/uint16 images.
        - 'histogram': Histogram-based algorithm [1]_. Works for 2D integer
          images.
        - 'sorting': Sorting-based algorithm via scipy.ndimage. Works for any
          dtype and dimensionality.
        - 'auto': Automatically selects the best algorithm.

        Note: this parameter is cuCIM-specific and does not exist in upstream
        scikit-image.
    algorithm_kwargs : dict
        Any additional algorithm-specific keywords. Currently can only be used
        to set the number of parallel partitions for the 'histogram' algorithm.
        (e.g. ``algorithm_kwargs={'partitions': 256}``). Note: this parameter is
        cuCIM-specific and does not exist in upstream scikit-image.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    See also
    --------
    skimage.filters.rank.median : Rank-based implementation of the median
        filtering offering more flexibility with additional parameters but
        dedicated for unsigned integer images.

    Notes
    -----
    For 2D images with uint8 or uint16 dtypes, a wavelet matrix-based median
    filter [2]_ provides the fastest performance for larger kernel sizes. This
    algorithm builds a succinct data structure enabling efficient range median
    queries with O(log(max_value)) complexity.

    For cases where the wavelet matrix approach is not applicable, a histogram-
    based median filter [1]_ may be used. It is faster than sorting for larger
    kernels (e.g., greater than 13x13) and has near-constant run time regardless
    of kernel size.

    When algorithm='auto', the best available algorithm is selected based on
    image dtype, dimensionality, and footprint size.

    References
    ----------
    .. [1] O. Green, "Efficient Scalable Median Filtering Using Histogram-Based
       Operations," in IEEE Transactions on Image Processing, vol. 27, no. 5,
       pp. 2217-2228, May 2018, https://doi.org/10.1109/TIP.2017.2781375.
    .. [2] Y. Sumida et al., "High-Performance 2D Median Filter Using Wavelet
       Matrix," in ACM SIGGRAPH Asia 2022 Technical Communications,
       https://doi.org/10.1145/3550454.3555512.

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
        raise NotImplementedError("rank behavior not currently implemented")
        # TODO: implement median rank filter
        # return generic.median(image, footprint=footprint, out=out)

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

    # Check algorithm compatibility
    can_use_wavelet = False
    can_use_histogram = False
    wm_reason = None
    hist_reason = None

    if algorithm in ["auto", "wavelet_matrix"]:
        # Wavelet matrix supports rectangular footprints with odd dimensions
        if image.ndim != 2:
            can_use_wavelet = False
            wm_reason = "Only 2D images are supported"
        else:
            can_use_wavelet, wm_reason = _can_use_wavelet_matrix(
                image, footprint_shape=footprint_shape
            )

    if algorithm in ["auto", "histogram"]:
        can_use_histogram, hist_reason = _can_use_histogram(
            image, footprint, footprint_shape
        )

    # Explicit algorithm requests with validation
    if algorithm == "wavelet_matrix" and not can_use_wavelet:
        raise ValueError(
            "The wavelet_matrix algorithm was requested, but it cannot "
            f"be used for this image and footprint (reason: {wm_reason})."
        )

    if algorithm == "histogram" and not can_use_histogram:
        raise ValueError(
            "The histogram-based algorithm was requested, but it cannot "
            f"be used for this image and footprint (reason: {hist_reason})."
        )

    # Algorithm selection for 'auto' mode
    # Priority: wavelet_matrix > histogram > sorting
    # Use sorting for small footprints (< ~150 elements)
    footprint_size = math.prod(footprint_shape)
    use_wavelet = False
    use_histogram = False

    if algorithm == "auto":
        # For small footprints, sorting is often fastest
        if (
            footprint_size <= 150
        ):  # TODO: check appropriate value for this threshold
            use_wavelet = False
            use_histogram = False
        elif can_use_wavelet:
            use_wavelet = True
        elif can_use_histogram:
            use_histogram = True
    elif algorithm == "wavelet_matrix":
        use_wavelet = True
    elif algorithm == "histogram":
        use_histogram = True

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
            radius_y = footprint_shape[0] // 2
            radius_x = footprint_shape[1] // 2
            temp = _median_wavelet_filter(
                image, radius_y=radius_y, radius_x=radius_x, mode=mode
            )

            if output_array_provided:
                out[:] = temp
            else:
                if out_dtype is not None:
                    temp = temp.astype(out_dtype, copy=False)
                out = temp
            return out
        except Exception as e:
            # Fall back to histogram or sorting if wavelet matrix fails
            if algorithm == "wavelet_matrix":
                raise  # Re-raise if explicitly requested
            warn(
                f"Wavelet matrix median failed: {e}\n"
                "Falling back to histogram or sorting-based median."
            )
            # Try histogram as fallback
            if can_use_histogram:
                use_histogram = True

    # Try histogram-based approach
    if use_histogram:
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

            # TODO: Can't currently pass an output array into _median_hist as a
            #       new array currently needs to be created during padding.

            # pass shape if explicit footprint isn't needed
            # (use new variable name in case KernelResourceError occurs)
            temp = _median_hist(
                image,
                footprint_shape if footprint is None else footprint,
                mode=mode,
                cval=cval,
                **algorithm_kwargs,
            )
            if output_array_provided:
                out[:] = temp
            else:
                if out_dtype is not None:
                    temp = temp.astype(out_dtype, copy=False)
                out = temp
            return out
        except KernelResourceError as e:
            # Fall back to sorting-based implementation if we encounter a
            # resource limit (e.g. insufficient shared memory per block).
            warn(
                "Kernel resource error encountered in histogram-based "
                f"median kernel: {e}\n"
                "Falling back to sorting-based median instead."
            )

    if algorithm_kwargs:
        warn(
            f"algorithm_kwargs={algorithm_kwargs} ignored for sorting-based "
            f"algorithm"
        )

    size = footprint_shape if footprint is None else None
    return ndi.median_filter(
        image, size=size, footprint=footprint, output=out, mode=mode, cval=cval
    )
