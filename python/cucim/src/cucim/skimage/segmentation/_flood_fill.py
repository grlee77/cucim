# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""Flood fill: find the pixels connected to a seed that are close to its value.

The region is the connected component, containing the seed, of the pixels
whose value is equal to (or within a tolerance of) the value at the seed. It
is found by thresholding and connected-component labeling (`_label_roots`)
instead of a flood from the seed.
"""

import operator

import cupy as cp
import numpy as np
from scipy import ndimage as cpu_ndi

from ..measure._label import _label_dtype_for_size
from ..measure._label_kernels import _label_roots


def _numeric_dtype_min_max(dtype):
    """Smallest and largest value of a dtype, as Python scalars."""
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return False, True
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        return int(info.min), int(info.max)
    info = np.finfo(dtype)
    return float(info.min), float(info.max)


def _resolve_seed_point(seed_point, shape):
    """The seed point as a tuple of non-negative Python ints."""
    try:
        iter(seed_point)
    except TypeError:
        seed_point = (seed_point,)
    seed_point = tuple(seed_point)
    if len(seed_point) != len(shape):
        raise ValueError(
            "`seed_point` must have one index for each dimension of `image` "
            f"({len(shape)}), got {len(seed_point)}"
        )
    resolved = []
    for axis, (index, size) in enumerate(zip(seed_point, shape)):
        index = operator.index(index)
        if not -size <= index < size:
            raise IndexError(
                f"index {index} is out of bounds for axis {axis} with size "
                f"{size}"
            )
        resolved.append(index % size)
    return tuple(resolved)


def _resolve_structure(footprint, connectivity, ndim):
    """The neighborhood as a boolean NumPy array with an odd shape."""
    if footprint is None:
        if connectivity is None:
            connectivity = ndim
        return cpu_ndi.generate_binary_structure(ndim, connectivity)
    if isinstance(footprint, cp.ndarray):
        footprint = footprint.get()
    footprint = np.asarray(footprint, dtype=bool)
    if footprint.ndim != ndim:
        raise ValueError(
            "number of dimensions in image and footprint do not match"
        )
    if any(s % 2 != 1 for s in footprint.shape):
        raise ValueError("footprint size must be odd along all dimensions")
    # The pixels are grouped into regions, so every neighbor relation must
    # hold in both directions (a footprint equal to its point reflection).
    if not np.array_equal(
        footprint, footprint[(slice(None, None, -1),) * ndim]
    ):
        raise NotImplementedError(
            "footprints that are not symmetric about their center (the fill "
            "would follow directed neighbors) are not supported"
        )
    return footprint


def flood(
    image, seed_point, *, footprint=None, connectivity=None, tolerance=None
):
    """Mask corresponding to a flood fill.

    Starting at a specific `seed_point`, connected points equal or within
    `tolerance` of the seed value are found.

    Parameters
    ----------
    image : cupy.ndarray
        An n-dimensional array.
    seed_point : tuple or int
        The point in `image` used as the starting point for the flood fill. If
        the image is 1D, this point may be given as an integer.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel. It must contain only 1's and 0's, have the
        same number of dimensions as `image`, an odd size along each axis, and
        be symmetric about its center. If not given, all adjacent pixels are
        considered as part of the neighborhood (fully connected).
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    tolerance : float or int, optional
        If None (default), adjacent values must be strictly equal to the
        initial value of `image` at `seed_point`. If a value is given, a
        comparison will be done at every point and if within tolerance of the
        initial value will also be filled (inclusive).

    Returns
    -------
    mask : cupy.ndarray
        A Boolean array with the same shape as `image` is returned, with True
        values for areas connected to and equal (or within tolerance of) the
        seed point. All other values are False.

    Raises
    ------
    NotImplementedError
        If `footprint` is not symmetric about its center.

    Notes
    -----
    The conceptual analogy of this operation is the 'paint bucket' tool in many
    raster graphics programs. This function returns just the mask
    representing the fill.

    If indices are desired rather than masks for memory reasons, the user can
    simply run `cupy.nonzero` on the result, save the indices, and discard
    this mask.

    The mask is found by labeling the connected components of the pixels that
    pass the test and keeping the one that contains the seed, so the time
    depends on the size of `image` and not on the size of the filled region.

    Differences from scikit-image:

    - `footprint` must be symmetric about its center. A footprint that is not
      (so that the fill follows directed neighbors) raises
      ``NotImplementedError``.
    - If the value at `seed_point` is NaN, only the seed point itself is
      filled, with or without `tolerance` (no value compares equal to, or
      lies within a tolerance of, NaN).

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.segmentation import flood
    >>> image = cp.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, with full connectivity (diagonals included):

    >>> mask = flood(image, (1, 1))
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [5, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, excluding diagonal points (connectivity 1):

    >>> mask = flood(image, (1, 1), connectivity=1)
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill with a tolerance:

    >>> mask = flood(image, (0, 0), tolerance=1)
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[5, 5, 5, 5, 5, 5, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 5, 5, 3]])
    """
    image = cp.asarray(image)

    # Shortcut for zero-size images
    if 0 in image.shape:
        return cp.zeros(image.shape, dtype=bool)

    if image.dtype == cp.float16:
        raise TypeError(
            "dtype of `image` is float16 which is not supported, "
            "try upcasting to float32"
        )

    seed_point = _resolve_seed_point(seed_point, image.shape)
    structure = _resolve_structure(footprint, connectivity, image.ndim)

    # The value at the seed is the only value copied to the host. The bounds
    # are computed on Python scalars so that they cannot overflow.
    seed_value = image[seed_point].item()
    if tolerance is None:
        mask = image == seed_value
    else:
        tolerance = abs(tolerance)
        min_value, max_value = _numeric_dtype_min_max(image.dtype)
        low = max(min_value, seed_value - tolerance)
        high = min(max_value, seed_value + tolerance)
        if seed_value != seed_value:
            # NaN seed: no pixel is within the tolerance of it
            mask = cp.zeros(image.shape, dtype=bool)
        else:
            # Like scikit-image, compare with the bounds converted to the
            # dtype of the image (a fractional bound of an integer image is
            # truncated toward zero).
            low, high = image.dtype.type(low), image.dtype.type(high)
            mask = (image >= low) & (image <= high)
    # the seed is always part of the region
    mask[seed_point] = True

    roots = cp.empty(image.shape, dtype=_label_dtype_for_size(image.size))
    _label_roots(cp.ascontiguousarray(mask), structure, roots)
    return roots == roots[seed_point]


def flood_fill(
    image,
    seed_point,
    new_value,
    *,
    footprint=None,
    connectivity=None,
    tolerance=None,
    in_place=False,
):
    """Perform flood filling on an image.

    Starting at a specific `seed_point`, connected points equal or within
    `tolerance` of the seed value are found, then set to `new_value`.

    Parameters
    ----------
    image : cupy.ndarray
        An n-dimensional array.
    seed_point : tuple or int
        The point in `image` used as the starting point for the flood fill. If
        the image is 1D, this point may be given as an integer.
    new_value : `image` type
        New value to set the entire fill. This must be chosen in agreement
        with the dtype of `image`.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel. It must contain only 1's and 0's, have the
        same number of dimensions as `image`, an odd size along each axis, and
        be symmetric about its center. If not given, all adjacent pixels are
        considered as part of the neighborhood (fully connected).
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    tolerance : float or int, optional
        If None (default), adjacent values must be strictly equal to the
        value of `image` at `seed_point` to be filled. If a tolerance is
        provided, adjacent points with values within plus or minus tolerance
        from the seed point are filled (inclusive).
    in_place : bool, optional
        If True, flood filling is applied to `image` in place. If False, the
        flood filled result is returned without modifying the input `image`
        (default).

    Returns
    -------
    filled : cupy.ndarray
        An array with the same shape as `image` is returned, with values in
        areas connected to and equal (or within tolerance of) the seed point
        replaced with `new_value`.

    Raises
    ------
    NotImplementedError
        If `footprint` is not symmetric about its center.

    See Also
    --------
    flood : The mask of the filled region.

    Notes
    -----
    The conceptual analogy of this operation is the 'paint bucket' tool in many
    raster graphics programs. See `flood` for how the region is found and for
    the differences from scikit-image.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.segmentation import flood_fill
    >>> image = cp.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, with full connectivity (diagonals included):

    >>> flood_fill(image, (1, 1), 5)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [5, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, excluding diagonal points (connectivity 1):

    >>> flood_fill(image, (1, 1), 5, connectivity=1)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill with a tolerance:

    >>> flood_fill(image, (0, 0), 5, tolerance=1)
    array([[5, 5, 5, 5, 5, 5, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 5, 5, 3]])
    """
    mask = flood(
        image,
        seed_point,
        footprint=footprint,
        connectivity=connectivity,
        tolerance=tolerance,
    )
    if not in_place:
        image = cp.array(image)
    image[mask] = new_value
    return image
