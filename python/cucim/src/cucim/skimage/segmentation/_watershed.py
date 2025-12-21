# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watershed segmentation using cellular automaton algorithm.

This module implements the CA-watershed algorithm based on:
Kauffmann, C., & Piche, N. (2010). Cellular automaton for ultra-fast
watershed transform on GPU. In Pattern Recognition (ICPR), 2010 20th
International Conference on (pp. 447-450). IEEE.
"""

import cupy as cp


# CUDA kernel for initialization
@cp.memoize(for_each_device=True)
def _get_watershed_init_kernel():
    """Get initialization kernel for CA-watershed.

    This kernel initializes the label, state, and distance arrays
    from the marker image.
    """
    return cp.RawKernel(
        r"""
extern "C" __global__
void watershed_init(
    const int* markers,
    int* labels,
    unsigned char* state,
    int* distance,
    const unsigned char* mask,
    int has_mask,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    // Check if pixel is in the mask
    if (has_mask && !mask[idx]) {
        labels[idx] = 0;
        state[idx] = 0;  // WATERSHED (background)
        distance[idx] = 0;
        return;
    }

    // Initialize from markers
    int marker_label = markers[idx];

    if (marker_label != 0) {
        // Seed pixel - already labeled (can be positive or negative)
        labels[idx] = marker_label;
        state[idx] = 1;  // LABELED (will not change)
        distance[idx] = 0;
    } else {
        // Unlabeled pixel - to be processed
        labels[idx] = 0;
        state[idx] = 2;  // UNLABELED (needs processing)
        distance[idx] = 2147483647;  // INT_MAX
    }
}
""",
        "watershed_init",
    )


@cp.memoize(for_each_device=True)
def _get_watershed_step_kernel_2d(connectivity=1):
    """Get iteration kernel for 2D CA-watershed.

    Parameters
    ----------
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity

    Returns
    -------
    kernel : cupy.RawKernel
        Compiled CUDA kernel
    """
    # Define neighbor offsets based on connectivity
    if connectivity == 1:
        # 4-connectivity (cross pattern)
        neighbors = [
            (-1, 0),  # top
            (1, 0),  # bottom
            (0, -1),  # left
            (0, 1),  # right
        ]
    else:
        # 8-connectivity (square pattern)
        neighbors = [
            (-1, 0),  # top
            (1, 0),  # bottom
            (0, -1),  # left
            (0, 1),  # right
            (-1, -1),  # top-left
            (-1, 1),  # top-right
            (1, -1),  # bottom-left
            (1, 1),  # bottom-right
        ]

    # Generate neighbor checking code
    neighbor_code = ""
    for i, (dy, dx) in enumerate(neighbors):
        neighbor_code += f"""
    // Neighbor {i}: dy={dy}, dx={dx}
    {{
        int ny = y + ({dy});
        int nx = x + ({dx});
        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {{
            int nidx = ny * width + nx;
            int nlabel = labels[nidx];
            int ndist = distance[nidx];

            // If neighbor has a label (non-zero), it can propagate to current
            // pixel
            if (nlabel != 0) {{
                int new_dist = ndist + 1;

                // Update if this is a better path (shorter distance)
                if (new_dist < min_dist) {{
                    min_dist = new_dist;
                    best_label = nlabel;
                    found_label = 1;
                }}
            }}
        }}
    }}
"""

    kernel_code = f"""
extern "C" __global__
void watershed_step_2d(
    const float* __restrict__ image,
    int* __restrict__ labels,
    unsigned char* __restrict__ state,
    int* __restrict__ distance,
    int* __restrict__ changed,
    int width,
    int height
) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Only process unlabeled pixels
    if (state[idx] != 2) return;

    // Check all neighbors
    int min_dist = distance[idx];
    int best_label = 0;
    int found_label = 0;

    {neighbor_code}

    // Update pixel if we found a better label
    if (found_label && best_label != 0) {{
        if (min_dist < distance[idx]) {{
            labels[idx] = best_label;
            distance[idx] = min_dist;

            // // Mark as labeled if distance didn't change (converged)
            // // or if distance is small enough
            // if (min_dist <= 1) {{
            //     state[idx] = 1;  // LABELED
            // }}

            // Signal that a change occurred
            atomicAdd(changed, 1);
        }}
    }}
}}
"""

    return cp.RawKernel(kernel_code, "watershed_step_2d")


def _footprint_to_connectivity(footprint):
    """Convert a footprint array to connectivity integer.

    Parameters
    ----------
    footprint : ndarray
        Boolean array representing connectivity pattern

    Returns
    -------
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity

    Raises
    ------
    ValueError
        If footprint is not a recognized pattern
    """
    if not isinstance(footprint, cp.ndarray):
        footprint = cp.asarray(footprint)

    # Check for 2D
    if footprint.ndim != 2:
        raise ValueError(
            f"Only 2D footprints are supported, got {footprint.ndim}D"
        )

    # Check for standard 3x3 footprints
    if footprint.shape == (3, 3):
        # 8-connectivity: all True
        if cp.all(footprint):
            return 2

        # 4-connectivity: cross pattern
        four_conn = cp.array(
            [[False, True, False], [True, True, True], [False, True, False]],
            dtype=bool,
        )
        if cp.all(footprint == four_conn):
            return 1

    # If we get here, it's a non-standard footprint
    raise NotImplementedError(
        "Only standard 4-connectivity and 8-connectivity footprints "
        "are currently supported. Please use "
        "connectivity=1 or connectivity=2."
    )


def _validate_inputs(image, markers, mask, connectivity):
    """Validate and prepare inputs for watershed algorithm.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (gradient magnitude or similar)
    markers : cupy.ndarray
        Marker array with positive integers for seeds
    mask : cupy.ndarray or None
        Optional mask array
    connectivity : int or ndarray
        Neighborhood connectivity (integer or footprint array)

    Returns
    -------
    image : cupy.ndarray
        Validated image as float32
    markers : cupy.ndarray
        Validated markers as int32
    mask : cupy.ndarray or None
        Validated mask as uint8
    connectivity : int
        Connectivity as integer (1 or 2)
    """
    # Check and convert connectivity
    if isinstance(connectivity, cp.ndarray) or (
        hasattr(connectivity, "__array__")
        and not isinstance(connectivity, (int, float))
    ):
        # Connectivity is an array (footprint)
        connectivity = _footprint_to_connectivity(connectivity)
    elif not isinstance(connectivity, (int, cp.integer)):
        raise TypeError(
            "connectivity must be an integer or array, got "
            f"{type(connectivity)}"
        )

    # Check connectivity value
    if connectivity not in (1, 2):
        raise ValueError(
            f"connectivity must be 1 or 2 for 2D images, got {connectivity}"
        )

    # Check image
    if not isinstance(image, cp.ndarray):
        image = cp.asarray(image)

    # Convert image to float32 for processing
    if image.dtype != cp.float32:
        image = image.astype(cp.float32)

    # Check markers
    if markers is None:
        raise ValueError("markers must be provided (cannot be None)")

    if not isinstance(markers, cp.ndarray):
        markers = cp.asarray(markers)

    if markers.shape != image.shape:
        raise ValueError(
            f"markers shape {markers.shape} must match "
            f"image shape {image.shape}"
        )

    # Convert markers to int32
    if markers.dtype != cp.int32:
        markers = markers.astype(cp.int32)

    # Check mask
    if mask is not None:
        if not isinstance(mask, cp.ndarray):
            mask = cp.asarray(mask)

        if mask.shape != image.shape:
            raise ValueError(
                f"mask shape {mask.shape} must match image shape {image.shape}"
            )

        # Convert mask to uint8
        mask = mask.astype(cp.uint8)

    return image, markers, mask, connectivity


def watershed(
    image,
    markers=None,
    connectivity=1,
    mask=None,
    compactness=0,
    watershed_line=False,
):
    """Watershed segmentation using cellular automaton algorithm.

    This function implements a GPU-accelerated watershed transform using
    a cellular automaton approach. The algorithm is particularly efficient
    for seeded watershed segmentation on 2D images.

    Parameters
    ----------
    image : cupy.ndarray, shape (M, N)
        Input image (typically a gradient magnitude or distance transform).
        Lower values have higher priority for watershed expansion.
    markers : cupy.ndarray of int, shape (M, N)
        Array of markers (seeds) for the watershed. Non-zero values
        represent different regions. Zero values are the areas to be
        segmented. Each unique non-zero integer (positive or negative)
        represents a different basin to grow from. Negative markers are
        supported for compatibility with scikit-image (e.g., -1 for
        background regions).
    connectivity : int or ndarray, optional
        Neighborhood connectivity. Can be:
        - Integer 1 or 2:
          - 1: 4-connectivity (von Neumann neighborhood - orthogonal neighbors)
          - 2: 8-connectivity (Moore neighborhood - includes diagonals)
        - Array: Boolean footprint array (e.g., from
          `generate_binary_structure`).
          Only standard 3x3 footprints for 4- and 8-connectivity are supported.
        Default is 1.
    mask : cupy.ndarray of bool, shape (M, N), optional
        If provided, only pixels where mask is True will be segmented.
        Useful for restricting watershed to regions of interest.
    compactness : float, optional
        Not yet implemented. Use 0 (default). In future versions, this
        will allow compact watershed with given compactness parameter.
    watershed_line : bool, optional
        Not yet implemented. Use False (default). In future versions, if
        True, a one-pixel wide line will separate the basins.

    Returns
    -------
    labels : cupy.ndarray of int, shape (M, N)
        Labeled array, where each basin is assigned a unique positive
        integer label matching the input markers.

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or invalid parameters.
    NotImplementedError
        If image is not 2D, or if compactness or watershed_line are used.

    Notes
    -----
    This implementation uses a cellular automaton (CA) approach, which is
    well-suited for GPU parallelization. The algorithm iteratively propagates
    labels from seed points (markers) to neighboring pixels based on their
    values in the input image.

    The algorithm is based on the CA-watershed method described in [1]_.
    Unlike the classical priority queue-based watershed, this approach
    processes all pixels in parallel during each iteration, making it
    highly efficient on GPU architectures.

    Current limitations:
    - Only 2D images are supported
    - compactness parameter is not yet implemented
    - watershed_line parameter is not yet implemented

    References
    ----------
    .. [1] Kauffmann, C., & Piche, N. (2010). Cellular automaton for
           ultra-fast watershed transform on GPU. In Pattern Recognition
           (ICPR), 2010 20th International Conference on (pp. 447-450). IEEE.

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
    """
    # Check for unsupported features
    if compactness != 0:
        raise NotImplementedError(
            "compactness parameter is not yet implemented"
        )

    if watershed_line:
        raise NotImplementedError(
            "watershed_line parameter is not yet implemented"
        )

    # Check dimensionality
    if image.ndim != 2:
        raise NotImplementedError(
            f"Only 2D images are currently supported, got {image.ndim}D"
        )

    # Validate and prepare inputs
    image, markers, mask, connectivity = _validate_inputs(
        image, markers, mask, connectivity
    )

    height, width = image.shape
    size = height * width

    # Initialize state arrays (as contiguous flat arrays)
    # Note: cp.zeros creates C-contiguous arrays by default
    labels = cp.zeros(size, dtype=cp.int32)
    state = cp.zeros(size, dtype=cp.uint8)
    distance = cp.zeros(size, dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    # Initialize from markers
    init_kernel = _get_watershed_init_kernel()
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block

    if mask is not None:
        mask_flat = cp.ascontiguousarray(mask.ravel())
        has_mask = cp.int32(1)
    else:
        # Create a dummy array filled with 1s (all pixels valid)
        mask_flat = cp.ones(size, dtype=cp.uint8)
        has_mask = cp.int32(0)

    init_kernel(
        (blocks,),
        (threads_per_block,),
        (
            markers.ravel(),
            labels,
            state,
            distance,
            mask_flat,
            has_mask,
            int(size),
        ),
    )

    # Iteratively propagate labels
    step_kernel = _get_watershed_step_kernel_2d(connectivity)

    # 2D block and grid configuration
    block_size = (16, 16)
    grid_size = (
        (width + block_size[0] - 1) // block_size[0],
        (height + block_size[1] - 1) // block_size[1],
    )

    # Upper bound on iterations (shouldn't need this many)
    max_iterations = max(height, width) * 2

    # Ensure image is contiguous and flat for kernel
    image_flat = cp.ascontiguousarray(image.ravel())

    for iteration in range(max_iterations):
        # Reset change counter
        changed[0] = 0

        # Run one iteration
        step_kernel(
            grid_size,
            block_size,
            (
                image_flat,
                labels,
                state,
                distance,
                changed,
                int(width),
                int(height),
            ),
        )

        # Check for convergence
        if changed[0] == 0:
            break

    # Reshape back to 2D
    return labels.reshape(height, width)
