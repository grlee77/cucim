# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watershed segmentation using cellular-automaton-style relaxation.

This module provides a GPU-parallel watershed implementation inspired by
cellular automaton (CA) watershed work, including:

P. Quesada-Barriuso, D.B. Heras, F. Argüello, Efficient 2D and 3D watershed on
graphics processing unit: block-asynchronous approaches based on cellular
automata, Computers & Electrical Engineering, Volume 39, Issue 8, 2013,
pp. 2638-2655, ISSN 0045-7906,
:DOI:`10.1016/j.compeleceng.2013.04.020`

There is also an earlier publication for CA-based watershed using graphics
shaders:

Kauffmann, C., & Piché, N. (2008). Cellular automaton for ultra-fast watershed
transform on GPU, 2008 19th International Conference on Pattern Recognition
(ICPR), Tampa, FL, USA, 2008, pp. 1-4.
:DOI:`10.1109/ICPR.2008.4761628`

This code is not an exact reproduction of those papers' synchronous or
hill-climbing plateau automata. It adapts CA-style local relaxation to the
scikit-image watershed API, including arbitrary markers, masks,
n-dimensional connectivity, optional age-based tie-breaking, and a compact
watershed extension.

The marker-controlled watershed model follows the morphological flooding
formulation of Meyer and Beucher:

Meyer, F., & Beucher, S. (1990). Morphological segmentation. Journal of Visual
Communication and Image Representation, 1(1), pp. 21-46.
:DOI:`10.1016/1047-3203(90)90014-M`

The compact watershed extension follows the scikit-image approach, based on:

Neubert, P., & Protzel, P. (2014). Compact Watershed and Preemptive SLIC:
On Improving Trade-offs of Superpixel Segmentation Algorithms.
In Pattern Recognition (ICPR), 2014 22nd International Conference on.
"""

import operator
import warnings

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage as ndi

from ..morphology.extrema import local_minima
from ..util._regular_grid import regular_seeds
from ._watershed_ca import (
    _DTYPE_TO_CTYPE,
    _KERNEL_PREAMBLE,
    _generate_coord_code,
    _generate_neighbor_nidx,
    _get_neighbor_offsets,
    _watershed_synchronous,
)
from ._watershed_ca_block_async import (
    TILE_3D,
    TILE_H,
    TILE_W,
    _watershed_standard_block_async,
)

#########################################
# Watershed line post-processing kernel #
#########################################


@cp.memoize(for_each_device=True)
def _get_watershed_line_kernel(ndim, connectivity=1, label_ctype="int32_t"):
    """Get post-processing kernel to create 1-pixel watershed lines.

    After the CA iteration has converged and all pixels have final labels,
    this kernel identifies boundary pixels. To produce 1-pixel-wide lines
    (rather than 2-pixel), only the pixel that arrived "later" in the
    flooding (higher priority) is set to 0. For equal priorities, the
    pixel with the higher label value is zeroed as a consistent
    tie-breaker. Marker pixels are copied through unchanged, so—as in
    scikit-image—adjacent marker regions may not have a separating line.

    Uses a 1D grid for all dimensionalities.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    connectivity : int
        Neighborhood connectivity.
    label_ctype : str
        C type string for label arrays (e.g. "int32_t", "int8_t").
    """
    L = label_ctype
    dim_names, dim_params, size_expr, coord_code = _generate_coord_code(ndim)
    neighbors = _get_neighbor_offsets(ndim, connectivity)
    neighbor_info = _generate_neighbor_nidx(ndim, dim_names, neighbors)

    neighbor_check = ""
    for nc_decl_str, bounds_str, nidx_expr in neighbor_info:
        neighbor_check += f"""
        {{
            {nc_decl_str}
            if ({bounds_str}) {{
                int nidx = {nidx_expr};
                {L} nlabel = labels_in[nidx];
                if (nlabel != 0 && nlabel != my_label) {{
                    float npri = priority[nidx];
                    if (npri < my_pri ||
                        (npri == my_pri && nlabel < my_label)) {{
                        is_boundary = 1;
                    }}
                }}
            }}
        }}
"""

    kernel_code = (
        _KERNEL_PREAMBLE
        + f"""
extern "C" __global__
void watershed_line(
    const {L}* __restrict__ labels_in,
    const float* __restrict__ priority,
    const unsigned char* __restrict__ state,
    {L}* __restrict__ labels_out,
    {dim_params}
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = {size_expr};
    if (idx >= size) return;

    {coord_code}

    {L} my_label = labels_in[idx];
    if (state[idx] == 1 || my_label == 0) {{
        // Markers are fixed inputs and must never become watershed lines.
        labels_out[idx] = my_label;
        return;
    }}

    float my_pri = priority[idx];
    int is_boundary = 0;
    {neighbor_check}

    labels_out[idx] = is_boundary ? 0 : my_label;
}}
"""
    )
    return cp.RawKernel(kernel_code, "watershed_line")


def _validate_inputs(image, markers, mask, connectivity):
    """Validate and prepare inputs for watershed algorithm.

    See `watershed` for parameter descriptions.
    """
    # Check image first to get ndim
    if not isinstance(image, cp.ndarray):
        raise TypeError(
            f"image must be a cupy.ndarray, got {type(image).__name__}"
        )

    ndim = image.ndim

    # Check connectivity - must be an integer
    if not isinstance(connectivity, (int, np.integer, cp.integer)):
        raise TypeError(
            f"connectivity must be an integer, got {type(connectivity)}"
        )
    connectivity = int(connectivity)

    # Check connectivity value based on dimensionality
    if connectivity not in range(1, ndim + 1):
        raise ValueError(
            f"connectivity must be in [1, {ndim}] for {ndim}D images, "
            f"got {connectivity}"
        )

    # Convert image to float32 for processing
    if image.dtype != cp.float32:
        image = image.astype(cp.float32)

    # Validate mask before markers (needed for marker generation)
    if mask is not None:
        if not isinstance(mask, cp.ndarray):
            raise TypeError(
                f"mask must be a cupy.ndarray, got {type(mask).__name__}"
            )

        if mask.shape != image.shape:
            raise ValueError(
                f"mask shape {mask.shape} must match image shape {image.shape}"
            )

        mask = mask.astype(cp.uint8)

    # Handle markers
    if markers is None:
        # Auto-detect markers from local minima (matching scikit-image)
        markers_bool = local_minima(image, connectivity=connectivity)
        if mask is not None:
            markers_bool = markers_bool * mask.astype(bool)
        footprint = ndi.generate_binary_structure(ndim, connectivity)
        markers = ndi.label(markers_bool, structure=footprint)[0]
    elif not isinstance(markers, (cp.ndarray, list, tuple)):
        # Assume int: generate that many regularly-spaced markers.
        # Scale by the fraction of masked pixels (like scikit-image) so the
        # requested count refers to markers *within* the mask. The masked
        # pixel count is only needed here, so the reduction (and its
        # device->host sync) is deferred to this branch.
        n_markers = int(markers)
        if mask is not None:
            n_pixels = int(cp.sum(mask))
            n_markers = int(n_markers / (n_pixels / image.size))
        markers = regular_seeds(image.shape, n_markers)
        if mask is not None:
            markers *= mask.astype(markers.dtype)
    else:
        if not isinstance(markers, cp.ndarray):
            raise TypeError(
                f"markers must be a cupy.ndarray or int, "
                f"got {type(markers).__name__}"
            )

        if mask is not None:
            markers = markers * mask.astype(markers.dtype)

        if markers.shape != image.shape:
            raise ValueError(
                f"markers shape {markers.shape} must match "
                f"image shape {image.shape}"
            )

    return image, markers, mask, connectivity


def watershed(
    image,
    markers=None,
    connectivity=1,
    offset=None,
    mask=None,
    compactness=0,
    watershed_line=False,
    *,
    use_block_async=False,
    use_age=False,
    inner_iterations=None,
    convergence_check_interval=None,
):
    """Watershed segmentation using cellular-automaton-style relaxation.

    This function implements a GPU-accelerated watershed transform using
    CA-inspired local relaxation of marker labels and path priorities. It is
    designed to follow the scikit-image watershed API rather than reproduce one
    paper verbatim, supporting arbitrary markers, masks, n-dimensional
    connectivity, optional plateau tie-breaking, and compact watershed.

    Parameters
    ----------
    image : cupy.ndarray, shape (M, N[, ...])
        Input image (typically a gradient magnitude or distance transform).
        Lower values have higher priority for watershed expansion.
        n-dimensional images are supported (the opt-in block-asynchronous
        path is limited to 2D and 3D; see ``use_block_async``).
    markers : int, cupy.ndarray of int, or None, optional
        The desired number of basins, or an array marking the basins with
        the values to be assigned in the label matrix. Zero means not a
        marker. If None (default), markers are determined as the local
        minima of the image, using
        :func:`cucim.skimage.morphology.local_minima` followed by
        :func:`cupyx.scipy.ndimage.label` (with the same given
        `connectivity`). If an int, that many regularly-spaced markers
        are generated using :func:`cucim.skimage.util.regular_seeds`.
        If an array, non-zero values represent different regions to grow
        from. Negative markers are supported (e.g., -1 for background).
        Unlike scikit-image, Python ``list``/``tuple`` markers are not
        accepted; pass a :class:`cupy.ndarray` or an int.
    connectivity : int, optional
        Neighborhood connectivity as integer:
        - For 2D images:
          - 1: 4-connectivity (von Neumann neighborhood - orthogonal neighbors)
          - 2: 8-connectivity (Moore neighborhood - includes diagonals)
        - For 3D images:
          - 1: 6-connectivity (face neighbors)
          - 2: 18-connectivity (face + edge neighbors)
          - 3: 26-connectivity (face + edge + corner neighbors)
        Default is 1. Unlike scikit-image, only an integer is accepted; a
        connectivity given as an ndarray footprint raises ``TypeError``.
    offset : array_like of shape image.ndim, optional
        The coordinates of the center of the connectivity footprint. This
        parameter is accepted for compatibility with the scikit-image API,
        but only the default (``None``, i.e. a footprint centered on each
        pixel) is supported. Passing any other value raises
        :class:`NotImplementedError`.
    mask : cupy.ndarray of bool, shape (M, N[, ...]), optional
        If provided, only pixels/voxels where mask is True will be segmented.
        Useful for restricting watershed to regions of interest. Default is
        None (the whole image is segmented).
    compactness : float, optional
        Use compact watershed with given compactness parameter.
        Higher values give more regularly-shaped basins. When compactness
        is non-zero, the priority is computed as:
            priority = (
                image[pixel] + compactness * euclidean_distance(pixel, marker)
            )
        Typical values range from 0.001 to 1.0. Default is 0 (standard
        watershed based purely on image values).
    watershed_line : bool, optional
        If True, a one-pixel wide line separates the regions obtained by
        the watershed algorithm. The line has the label 0. This is
        implemented as a post-processing step after the CA iteration
        converges (see Notes). Default is False.
    use_block_async : bool, optional
        Opt-in faster algorithm variant for standard watershed. If False
        (default), uses the synchronous double-buffered relaxation. If True,
        uses a block-asynchronous algorithm with shared-memory tiling, which
        is faster for larger images but only approximately matches the
        synchronous result (tile-local updates are applied before global
        synchronization). Only implemented for 2D and 3D images and only
        affects the standard watershed (compactness=0); it is ignored (with
        a warning) otherwise.
    use_age : bool, optional
        If True, use an age (hop distance) counter as a tie-breaker when two
        labels arrive at a pixel with equal priority, producing fairer splits
        on plateau regions. Only affects the block-asynchronous path
        (``use_block_async=True``); it is a no-op for the default synchronous
        path, which already yields the closest-marker assignment on plateaus
        by construction. Default is False.
    inner_iterations : int or None, optional
        Number of iterations to perform within each block before writing back
        to global memory (block-async mode only).
        If None (default), uses 16 for 2D and 8 for 3D. Lower values
        improve agreement with the synchronous path at the cost of
        reduced performance. Has no effect when use_block_async=False.
    convergence_check_interval : int or None, optional
        Number of relaxation kernel launches to perform between successive
        convergence checks. Each check reads a device flag back to the host,
        which forces a synchronization and serializes the CPU and GPU.
        Running several launches back-to-back before checking reduces these
        host-device syncs by roughly this factor, at the cost of up to
        ``convergence_check_interval - 1`` extra launches after convergence.
        Because the relaxation is idempotent at its fixed point, those extra
        launches are no-ops and never change the result. If None (default),
        a value tuned to the algorithm is used: 8 for the synchronous path
        (many cheap launches, where syncs dominate) and 1 for the
        block-asynchronous path (few, expensive launches, where extra
        launches cost more than the syncs saved). Must be >= 1 if given.

    Returns
    -------
    labels : cupy.ndarray of int, shape (M, N[, ...])
        Labeled array of the same shape as ``image``. Each basin is labeled
        with the value of its marker (negative marker values are preserved),
        and the output dtype matches the marker dtype.

    Raises
    ------
    TypeError
        If an input has an unsupported type or `markers` does not have a
        signed or unsigned integer dtype.
    ValueError
        If input arrays have incompatible shapes or invalid parameters.
    NotImplementedError
        If a non-default ``offset`` is given (only ``offset=None`` is
        supported).

    Notes
    -----
    The marker-controlled watershed model follows the morphological flooding
    formulation of Meyer and Beucher [1]_, also used by tools such as
    MorphoLibJ and scikit-image. This implementation is related to the
    CA-watershed and Ford-Bellman-style GPU relaxations described in [2]_,
    [3]_, and [4]_, but it is not an exact implementation of the synchronous
    or hill-climbing plateau automata from those papers. It instead propagates
    marker labels by repeatedly relaxing a path priority, similar in spirit to
    seeded watershed but structured for GPU-wide parallel updates.

    The block-asynchronous path follows the plain tiled/shared-memory update
    pattern from [2]_. It does not implement the artifact-free distance
    correction proposed there; the optional ``use_age`` value is a
    scikit-image-oriented plateau tie-breaker, not that correction scheme.

    The compact watershed extension follows the scikit-image approach [5]_,
    adding a distance penalty to encourage more regularly-shaped regions. This
    is useful for superpixel generation.

    References
    ----------
    .. [1] Meyer, F., & Beucher, S. (1990). Morphological segmentation.
           Journal of Visual Communication and Image Representation, 1(1),
           pp. 21-46.
           :DOI:`10.1016/1047-3203(90)90014-M`
    .. [2] P. Quesada-Barriuso, D.B. Heras, F. Argüello, Efficient 2D and 3D
           watershed on graphics processing unit: block-asynchronous approaches
           based on cellular automata, Computers & Electrical Engineering,
           Volume 39, Issue 8, 2013, pp. 2638-2655, ISSN 0045-7906,
           :DOI:`10.1016/j.compeleceng.2013.04.020`
    .. [3] Kauffmann, C., & Piché, N. (2008). Cellular automaton for ultra-fast
           watershed transform on GPU, 2008 19th International Conference on
           Pattern Recognition (ICPR), Tampa, FL, USA, 2008, pp. 1-4.
           :DOI:`10.1109/ICPR.2008.4761628`
    .. [4] Kauffmann, C., & Piché, N. (2010). Seeded ND medical image
           segmentation by cellular automaton on GPU, International Journal of
           Computer Assisted Radiology and Surgery, Volume 5, Issue 3, 2010,
           pp. 251-262.
           :DOI:`10.1007/s11548-009-0392-0`
    .. [5] Neubert, P., & Protzel, P. (2014). Compact Watershed and Preemptive
           SLIC: On Improving Trade-offs of Superpixel Segmentation Algorithms,
           2014 22nd International Conference on Pattern Recognition (ICPR),
           Stockholm, Sweden, 2014, pp. 996-1001
           :DOI:`10.1109/ICPR.2014.181`

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import segmentation
    >>> from cupyx.scipy import ndimage as ndi

    Create a simple test image with two peaks:

    >>> image = cp.zeros((10, 10), dtype=cp.float32)
    >>> image[2, 2] = 1
    >>> image[7, 7] = 1
    >>> image = ndi.gaussian_filter(image, sigma=1.0)

    Create markers at the peaks:

    >>> markers = cp.zeros((10, 10), dtype=cp.int32)
    >>> markers[2, 2] = 1
    >>> markers[7, 7] = 2

    Apply watershed:

    >>> labels = segmentation.watershed(-image, markers)
    >>> labels.shape
    (10, 10)
    >>> cp.unique(labels)
    array([1, 2])

    Apply compact watershed for more regular regions:

    >>> labels_compact = segmentation.watershed(-image,
    >>>                                         markers, compactness=0.1)
    """
    if offset is not None:
        # `offset` is accepted only to keep the positional argument order
        # consistent with scikit-image. The CA kernels use symmetric
        # neighborhoods centered on each pixel, so a shifted footprint
        # center is not supported.
        raise NotImplementedError(
            "The `offset` parameter is accepted for scikit-image API "
            "compatibility but only the default (offset=None) is supported."
        )

    if inner_iterations is not None:
        try:
            inner_iterations = operator.index(inner_iterations)
        except TypeError as e:
            raise TypeError(
                "inner_iterations must be an integer or None, got "
                f"{type(inner_iterations).__name__}"
            ) from e
        if inner_iterations < 1:
            raise ValueError(
                "inner_iterations must be a positive integer, got "
                f"{inner_iterations}"
            )

    if convergence_check_interval is not None:
        convergence_check_interval = int(convergence_check_interval)
        if convergence_check_interval < 1:
            raise ValueError(
                "convergence_check_interval must be a positive integer, got "
                f"{convergence_check_interval}"
            )

    # Determine output label dtype from markers
    if isinstance(markers, cp.ndarray):
        label_dtype = cp.dtype(markers.dtype)
        if label_dtype not in _DTYPE_TO_CTYPE:
            raise TypeError(
                "markers must have a signed or unsigned integer dtype, got "
                f"{label_dtype}"
            )
    else:
        label_dtype = cp.dtype(cp.int32)

    # Validate and prepare inputs
    image, markers, mask, connectivity = _validate_inputs(
        image, markers, mask, connectivity
    )

    # User-provided marker dtypes were validated above. Keep internally
    # generated or mask-adjusted markers consistent with the kernel type.
    if markers.dtype != label_dtype:
        markers = markers.astype(label_dtype)

    ndim = image.ndim

    # Get dimensions
    size = image.size

    # Prepare mask
    if mask is not None:
        mask_flat = cp.ascontiguousarray(mask.ravel())
        has_mask = cp.int32(1)
    else:
        # Create a dummy array filled with 1s (all pixels valid)
        mask_flat = cp.ones(size, dtype=cp.uint8)
        has_mask = cp.int32(0)

    # Thread/block configuration
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block

    # Using image.size as the bound guarantees the result is never silently
    # truncated; the loop still exits early as soon as it converges, so this
    # ceiling has no effect on the common case.
    max_iterations = int(image.size)

    # Ensure image is contiguous and flat for kernel
    image_flat = cp.ascontiguousarray(image.ravel())

    # Block-async is an opt-in fast path, only available for 2D/3D
    # non-compact watershed. When requested but not applicable, warn and fall
    # back to the synchronous (double-buffered) path.
    if use_block_async:
        reason = None
        if compactness != 0:
            reason = "use_block_async is not supported with compactness > 0"
        elif ndim not in (2, 3):
            reason = f"use_block_async is only supported for 2D/3D, got {ndim}D"
        else:
            _min_ba = max(TILE_W, TILE_H) if ndim == 2 else TILE_3D
            if min(image.shape) < _min_ba:
                reason = (
                    f"use_block_async=True requires image dimensions >= "
                    f"{_min_ba} for {ndim}D; got image shape {image.shape}"
                )
        if reason is not None:
            warnings.warn(
                f"{reason}; falling back to synchronous.",
                stacklevel=2,
            )
            use_block_async = False

    label_ctype = _DTYPE_TO_CTYPE[label_dtype]

    # common kwargs shared across all watershed implementations
    common_kwargs = dict(
        image_flat=image_flat,
        markers=markers,
        mask_flat=mask_flat,
        has_mask=has_mask,
        connectivity=connectivity,
        size=size,
        blocks=blocks,
        threads_per_block=threads_per_block,
        max_iterations=max_iterations,
        label_dtype=label_dtype,
        convergence_check_interval=convergence_check_interval,
    )

    if use_block_async and ndim in (2, 3) and compactness == 0:
        labels, state, priority, changed = _watershed_standard_block_async(
            **common_kwargs,
            ndim=ndim,
            image_shape=image.shape,
            use_age=use_age,
            inner_iterations=inner_iterations,
        )
    else:
        labels, state, priority, changed = _watershed_synchronous(
            **common_kwargs,
            compactness=compactness,
            ndim=ndim,
            image_shape=image.shape,
        )

    # Post-processing: create watershed lines if requested
    if watershed_line:
        labels_out = cp.empty_like(labels)
        wl_kernel = _get_watershed_line_kernel(
            ndim, connectivity, label_ctype=label_ctype
        )
        dim_args = tuple(int(s) for s in image.shape)
        wl_kernel(
            (blocks,),
            (threads_per_block,),
            (labels, priority, state, labels_out, *dim_args),
        )
        labels = labels_out

    return labels.reshape(image.shape)
