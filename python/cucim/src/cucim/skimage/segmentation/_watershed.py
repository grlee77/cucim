# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watershed segmentation using cellular automaton algorithm.

This module implements the CA-watershed algorithm based on:
Kauffmann, C., & Piche, N. (2010). Cellular automaton for ultra-fast
watershed transform on GPU. In Pattern Recognition (ICPR), 2010 20th
International Conference on (pp. 447-450). IEEE.

The compact watershed extension follows the approach from scikit-image,
which is based on:
Neubert, P., & Protzel, P. (2014). Compact Watershed and Preemptive SLIC:
On Improving Trade-offs of Superpixel Segmentation Algorithms.
In Pattern Recognition (ICPR), 2014 22nd International Conference on.
"""

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage as ndi

from ..morphology.extrema import local_minima
from ..util._regular_grid import regular_seeds


# CUDA kernel for initialization (standard watershed)
@cp.memoize(for_each_device=True)
def _get_watershed_init_kernel(use_age=False):
    """Get initialization kernel for CA-watershed.

    This kernel initializes the label, state, and priority arrays
    from the marker image. Priority is based on image intensity values
    to match scikit-image's watershed behavior. Optionally initializes
    an age array for tie-breaking (lower age = closer to marker).
    """
    age_params = "int* age," if use_age else ""
    age_init_mask = "age[idx] = 0;" if use_age else ""
    age_init_marker = "age[idx] = 0;" if use_age else ""
    age_init_unlabeled = "age[idx] = 2147483647;" if use_age else ""

    kernel_code = f"""
extern "C" __global__
void watershed_init(
    const float* image,
    const int* markers,
    int* labels,
    unsigned char* state,
    float* priority,
    {age_params}
    const unsigned char* mask,
    int has_mask,
    int size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    // Check if pixel is in the mask
    if (has_mask && !mask[idx]) {{
        labels[idx] = 0;
        state[idx] = 0;  // WATERSHED (background)
        priority[idx] = 0.0f;
        {age_init_mask}
        return;
    }}

    // Initialize from markers
    int marker_label = markers[idx];

    if (marker_label != 0) {{
        // Seed pixel - already labeled (can be positive or negative)
        // Priority is the image value at the marker (like scikit-image)
        labels[idx] = marker_label;
        state[idx] = 1;  // LABELED (will not change)
        priority[idx] = image[idx];
        {age_init_marker}
    }} else {{
        // Unlabeled pixel - to be processed
        labels[idx] = 0;
        state[idx] = 2;  // UNLABELED (needs processing)
        priority[idx] = 3.4028235e+38f;  // FLT_MAX
        {age_init_unlabeled}
    }}
}}
"""
    return cp.RawKernel(kernel_code, "watershed_init")


# CUDA kernel for initialization (compact watershed)
@cp.memoize(for_each_device=True)
def _get_watershed_compact_init_kernel():
    """Get initialization kernel for compact CA-watershed.

    This kernel initializes the label, state, priority, and source arrays
    from the marker image. The source arrays track the original marker
    location for each pixel, needed for Euclidean distance computation.
    """
    return cp.RawKernel(
        r"""
extern "C" __global__
void watershed_compact_init(
    const int* markers,
    int* labels,
    unsigned char* state,
    float* priority,
    int* source_x,
    int* source_y,
    const unsigned char* mask,
    int has_mask,
    int width,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    // Compute 2D coordinates from flat index
    int x = idx % width;
    int y = idx / width;

    // Check if pixel is in the mask
    if (has_mask && !mask[idx]) {
        labels[idx] = 0;
        state[idx] = 0;  // WATERSHED (background)
        priority[idx] = 0.0f;
        source_x[idx] = -1;
        source_y[idx] = -1;
        return;
    }

    // Initialize from markers
    int marker_label = markers[idx];

    if (marker_label != 0) {
        // Seed pixel - already labeled (can be positive or negative)
        labels[idx] = marker_label;
        state[idx] = 1;  // LABELED (will not change)
        priority[idx] = 0.0f;
        // Source is the marker's own location
        source_x[idx] = x;
        source_y[idx] = y;
    } else {
        // Unlabeled pixel - to be processed
        labels[idx] = 0;
        state[idx] = 2;  // UNLABELED (needs processing)
        priority[idx] = 3.4028235e+38f;  // FLT_MAX
        source_x[idx] = -1;
        source_y[idx] = -1;
    }
}
""",
        "watershed_compact_init",
    )


def _get_neighbor_offsets_1d():
    """Get 1D neighbor offsets (left and right)."""
    return [(-1,), (1,)]


def _get_neighbor_offsets_2d(connectivity):
    """Get 2D neighbor offsets based on connectivity.

    Parameters
    ----------
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity

    Returns
    -------
    neighbors : list of tuple
        List of (dy, dx) offsets for neighbors
    """
    if connectivity == 1:
        # 4-connectivity (cross pattern)
        return [
            (-1, 0),  # top
            (1, 0),  # bottom
            (0, -1),  # left
            (0, 1),  # right
        ]
    else:
        # 8-connectivity (square pattern)
        return [
            (-1, 0),  # top
            (1, 0),  # bottom
            (0, -1),  # left
            (0, 1),  # right
            (-1, -1),  # top-left
            (-1, 1),  # top-right
            (1, -1),  # bottom-left
            (1, 1),  # bottom-right
        ]


def _get_neighbor_offsets_3d(connectivity):
    """Get 3D neighbor offsets based on connectivity.

    Parameters
    ----------
    connectivity : int
        1 for 6-connectivity (face neighbors)
        2 for 18-connectivity (face + edge neighbors)
        3 for 26-connectivity (face + edge + corner neighbors)

    Returns
    -------
    neighbors : list of tuple
        List of (dz, dy, dx) offsets for neighbors
    """
    # Face neighbors (6-connectivity)
    face_neighbors = [
        (-1, 0, 0),  # front
        (1, 0, 0),  # back
        (0, -1, 0),  # top
        (0, 1, 0),  # bottom
        (0, 0, -1),  # left
        (0, 0, 1),  # right
    ]

    if connectivity == 1:
        return face_neighbors

    # Edge neighbors (12 additional)
    edge_neighbors = [
        (-1, -1, 0),
        (-1, 1, 0),
        (-1, 0, -1),
        (-1, 0, 1),
        (1, -1, 0),
        (1, 1, 0),
        (1, 0, -1),
        (1, 0, 1),
        (0, -1, -1),
        (0, -1, 1),
        (0, 1, -1),
        (0, 1, 1),
    ]

    if connectivity == 2:
        return face_neighbors + edge_neighbors

    # Corner neighbors (8 additional)
    corner_neighbors = [
        (-1, -1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, -1, -1),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 1),
    ]

    # connectivity == 3: 26-connectivity
    return face_neighbors + edge_neighbors + corner_neighbors


@cp.memoize(for_each_device=True)
def _get_watershed_step_kernel_2d(connectivity=1, use_age=False):
    """Get iteration kernel for 2D CA-watershed.

    This kernel uses image intensity as priority (like scikit-image).
    Priority = max(image[pixel], neighbor_priority) for monotonicity.
    Lower priority values are preferred (flooding from low to high).

    Parameters
    ----------
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity
    use_age : bool
        If True, use age (hop distance) as tie-breaker when priorities
        are equal. This more closely matches scikit-image's behavior.

    Returns
    -------
    kernel : cupy.RawKernel
        Compiled CUDA kernel
    """
    neighbors = _get_neighbor_offsets_2d(connectivity)

    # Generate neighbor checking code
    neighbor_code = ""
    for i, (dy, dx) in enumerate(neighbors):
        age_read = "int nage = age[nidx];" if use_age else ""
        age_new = "int new_age = nage + 1;" if use_age else ""
        if use_age:
            better_path_cmp = (
                "if (new_priority < best_priority ||\n"
                "                    (new_priority == best_priority "
                "&& new_age < best_age))"
            )
            age_update = "best_age = new_age;"
        else:
            better_path_cmp = "if (new_priority < best_priority)"
            age_update = ""

        neighbor_code += f"""
    // Neighbor {i}: dy={dy}, dx={dx}
    {{
        int ny = y + ({dy});
        int nx = x + ({dx});
        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {{
            int nidx = ny * width + nx;
            int nlabel = labels[nidx];
            float npriority = priority[nidx];
            {age_read}

            // If neighbor has a label (non-zero), it can propagate to current
            // pixel
            if (nlabel != 0) {{
                // Priority is the image value, with monotonicity constraint:
                // new priority = max(image[current], neighbor_priority)
                // This matches scikit-image's watershed behavior
                float new_priority = image[idx];
                if (new_priority < npriority) {{
                    new_priority = npriority;
                }}

                {age_new}

                // Update if this is a better path
                {better_path_cmp} {{
                    best_priority = new_priority;
                    {age_update}
                    best_label = nlabel;
                    found_label = 1;
                }}
            }}
        }}
    }}
"""

    age_param = "int* __restrict__ age," if use_age else ""
    age_best_init = "int best_age = age[idx];" if use_age else ""
    if use_age:
        better_path_final_cmp = (
            "if (best_priority < priority[idx] ||\n"
            "            (best_priority == priority[idx] "
            "&& best_age < age[idx]))"
        )
        age_final_update = "age[idx] = best_age;"
    else:
        better_path_final_cmp = "if (best_priority < priority[idx])"
        age_final_update = ""

    kernel_code = f"""
extern "C" __global__
void watershed_step_2d(
    const float* __restrict__ image,
    int* __restrict__ labels,
    unsigned char* __restrict__ state,
    float* __restrict__ priority,
    {age_param}
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
    float best_priority = priority[idx];
    {age_best_init}
    int best_label = 0;
    int found_label = 0;

    {neighbor_code}

    // Update pixel if we found a better label
    if (found_label && best_label != 0) {{
        {better_path_final_cmp} {{
            labels[idx] = best_label;
            priority[idx] = best_priority;
            {age_final_update}

            // Signal that a change occurred
            atomicAdd(changed, 1);
        }}
    }}
}}
"""

    return cp.RawKernel(kernel_code, "watershed_step_2d")


@cp.memoize(for_each_device=True)
def _get_watershed_step_kernel_1d(use_age=False):
    """Get iteration kernel for 1D CA-watershed.

    Uses 1D grid with left/right neighbors only (connectivity=1).
    """
    neighbors = _get_neighbor_offsets_1d()

    neighbor_code = ""
    for i, (dx,) in enumerate(neighbors):
        age_read = "int nage = age[nidx];" if use_age else ""
        age_new = "int new_age = nage + 1;" if use_age else ""
        if use_age:
            better_path_cmp = (
                "if (new_priority < best_priority ||\n"
                "                    (new_priority == best_priority "
                "&& new_age < best_age))"
            )
            age_update = "best_age = new_age;"
        else:
            better_path_cmp = "if (new_priority < best_priority)"
            age_update = ""

        neighbor_code += f"""
    {{
        int nx = x + ({dx});
        if (nx >= 0 && nx < size) {{
            int nidx = nx;
            int nlabel = labels[nidx];
            float npriority = priority[nidx];
            {age_read}

            if (nlabel != 0) {{
                float new_priority = image[idx];
                if (new_priority < npriority) {{
                    new_priority = npriority;
                }}

                {age_new}

                {better_path_cmp} {{
                    best_priority = new_priority;
                    {age_update}
                    best_label = nlabel;
                    found_label = 1;
                }}
            }}
        }}
    }}
"""

    age_param = "int* __restrict__ age," if use_age else ""
    age_best_init = "int best_age = age[idx];" if use_age else ""
    if use_age:
        better_path_final_cmp = (
            "if (best_priority < priority[idx] ||\n"
            "            (best_priority == priority[idx] "
            "&& best_age < age[idx]))"
        )
        age_final_update = "age[idx] = best_age;"
    else:
        better_path_final_cmp = "if (best_priority < priority[idx])"
        age_final_update = ""

    kernel_code = f"""
extern "C" __global__
void watershed_step_1d(
    const float* __restrict__ image,
    int* __restrict__ labels,
    unsigned char* __restrict__ state,
    float* __restrict__ priority,
    {age_param}
    int* __restrict__ changed,
    int size
) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size) return;

    int idx = x;
    if (state[idx] != 2) return;

    float best_priority = priority[idx];
    {age_best_init}
    int best_label = 0;
    int found_label = 0;

    {neighbor_code}

    if (found_label && best_label != 0) {{
        {better_path_final_cmp} {{
            labels[idx] = best_label;
            priority[idx] = best_priority;
            {age_final_update}

            atomicAdd(changed, 1);
        }}
    }}
}}
"""
    return cp.RawKernel(kernel_code, "watershed_step_1d")


@cp.memoize(for_each_device=True)
def _get_watershed_step_kernel_3d(connectivity=1, use_age=False):
    """Get iteration kernel for 3D CA-watershed.

    This kernel uses image intensity as priority (like scikit-image).
    Priority = max(image[pixel], neighbor_priority) for monotonicity.
    Lower priority values are preferred (flooding from low to high).

    Parameters
    ----------
    connectivity : int
        1 for 6-connectivity (face neighbors)
        2 for 18-connectivity (face + edge neighbors)
        3 for 26-connectivity (face + edge + corner neighbors)
    use_age : bool
        If True, use age (hop distance) as tie-breaker.

    Returns
    -------
    kernel : cupy.RawKernel
        Compiled CUDA kernel
    """
    neighbors = _get_neighbor_offsets_3d(connectivity)

    # Generate neighbor checking code
    neighbor_code = ""
    for i, (dz, dy, dx) in enumerate(neighbors):
        age_read = "int nage = age[nidx];" if use_age else ""
        age_new = "int new_age = nage + 1;" if use_age else ""
        if use_age:
            better_path_cmp = (
                "if (new_priority < best_priority ||\n"
                "                    (new_priority == best_priority "
                "&& new_age < best_age))"
            )
            age_update = "best_age = new_age;"
        else:
            better_path_cmp = "if (new_priority < best_priority)"
            age_update = ""

        neighbor_code += f"""
    // Neighbor {i}: dz={dz}, dy={dy}, dx={dx}
    {{
        int nz = z + ({dz});
        int ny = y + ({dy});
        int nx = x + ({dx});
        if (nz >= 0 && nz < depth && ny >= 0 && ny < height && nx >= 0 && nx < width) {{
            int nidx = (nz * height + ny) * width + nx;
            int nlabel = labels[nidx];
            float npriority = priority[nidx];
            {age_read}

            // If neighbor has a label (non-zero), it can propagate to current
            // voxel
            if (nlabel != 0) {{
                // Priority is the image value, with monotonicity constraint:
                // new priority = max(image[current], neighbor_priority)
                // This matches scikit-image's watershed behavior
                float new_priority = image[idx];
                if (new_priority < npriority) {{
                    new_priority = npriority;
                }}

                {age_new}

                // Update if this is a better path
                {better_path_cmp} {{
                    best_priority = new_priority;
                    {age_update}
                    best_label = nlabel;
                    found_label = 1;
                }}
            }}
        }}
    }}
"""  # noqa: E501

    age_param = "int* __restrict__ age," if use_age else ""
    age_best_init = "int best_age = age[idx];" if use_age else ""
    if use_age:
        better_path_final_cmp = (
            "if (best_priority < priority[idx] ||\n"
            "            (best_priority == priority[idx] "
            "&& best_age < age[idx]))"
        )
        age_final_update = "age[idx] = best_age;"
    else:
        better_path_final_cmp = "if (best_priority < priority[idx])"
        age_final_update = ""

    kernel_code = f"""
extern "C" __global__
void watershed_step_3d(
    const float* __restrict__ image,
    int* __restrict__ labels,
    unsigned char* __restrict__ state,
    float* __restrict__ priority,
    {age_param}
    int* __restrict__ changed,
    int width,
    int height,
    int depth
) {{
    // Use 1D grid for 3D data (simpler and handles all sizes)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height * depth;

    if (idx >= size) return;

    // Convert flat index to 3D coordinates
    int x = idx % width;
    int y = (idx / width) % height;
    int z = idx / (width * height);

    // Only process unlabeled voxels
    if (state[idx] != 2) return;

    // Check all neighbors
    float best_priority = priority[idx];
    {age_best_init}
    int best_label = 0;
    int found_label = 0;

    {neighbor_code}

    // Update voxel if we found a better label
    if (found_label && best_label != 0) {{
        {better_path_final_cmp} {{
            labels[idx] = best_label;
            priority[idx] = best_priority;
            {age_final_update}

            // Signal that a change occurred
            atomicAdd(changed, 1);
        }}
    }}
}}
"""

    return cp.RawKernel(kernel_code, "watershed_step_3d")


# Block-asynchronous watershed kernel constants
TILE_W = 32  # Tile width (must match block size)
TILE_H = 32  # Tile height (must match block size)
HALO = 1  # Halo size (1 pixel for immediate neighbors)


@cp.memoize(for_each_device=True)
def _get_watershed_block_async_kernel_2d(connectivity=1, inner_iterations=16):
    """Get block-asynchronous iteration kernel for 2D CA-watershed.

    This kernel uses shared memory tiling to reduce global memory traffic.
    Each block loads a tile + halo into shared memory and performs multiple
    iterations locally before writing back to global memory.

    Based on Section 4.3 of Kauffmann & Piche (2010).

    Parameters
    ----------
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity
    inner_iterations : int
        Number of iterations to perform within each block before
        synchronizing with global memory. Default is 8.

    Returns
    -------
    kernel : cupy.RawKernel
        Compiled CUDA kernel
    """
    neighbors = _get_neighbor_offsets_2d(connectivity)

    # Shared memory dimensions (tile + halo on each side)
    shared_w = TILE_W + 2 * HALO
    shared_h = TILE_H + 2 * HALO

    # Generate neighbor checking code for shared memory
    neighbor_code = ""
    for i, (dy, dx) in enumerate(neighbors):
        neighbor_code += f"""
            // Neighbor {i}: dy={dy}, dx={dx}
            {{
                int nsx = sx + ({dx});
                int nsy = sy + ({dy});
                int nsidx = nsy * {shared_w} + nsx;

                int nlabel = s_labels[nsidx];
                float npriority = s_priority[nsidx];

                // If neighbor has a label, it can propagate
                if (nlabel != 0) {{
                    // Priority with monotonicity constraint
                    float new_priority = my_image;
                    if (new_priority < npriority) {{
                        new_priority = npriority;
                    }}

                    // Update if better path
                    if (new_priority < best_priority) {{
                        best_priority = new_priority;
                        best_label = nlabel;
                        found_label = 1;
                    }}
                }}
            }}
"""

    kernel_code = f"""
extern "C" __global__
void watershed_block_async_2d(
    const float* __restrict__ image,
    int* __restrict__ labels,
    unsigned char* __restrict__ state,
    float* __restrict__ priority,
    int* __restrict__ global_changed,
    int width,
    int height
) {{
    // Shared memory for tile + halo
    // Layout: labels, state, priority, image
    __shared__ int s_labels[{shared_w} * {shared_h}];
    __shared__ unsigned char s_state[{shared_w} * {shared_h}];
    __shared__ float s_priority[{shared_w} * {shared_h}];
    __shared__ float s_image[{shared_w} * {shared_h}];
    __shared__ int s_changed;

    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global coordinates of this thread's main pixel
    int gx = blockIdx.x * {TILE_W} + tx;
    int gy = blockIdx.y * {TILE_H} + ty;

    // Shared memory coordinates (offset by halo)
    int sx = tx + {HALO};
    int sy = ty + {HALO};
    int sidx = sy * {shared_w} + sx;

    // ========================================
    // Phase 1: Cooperative load of tile + halo
    // ========================================

    // Load main tile (each thread loads one pixel)
    if (gx < width && gy < height) {{
        int gidx = gy * width + gx;
        s_labels[sidx] = labels[gidx];
        s_state[sidx] = state[gidx];
        s_priority[sidx] = priority[gidx];
        s_image[sidx] = image[gidx];
    }} else {{
        // Out of bounds - mark as background
        s_labels[sidx] = 0;
        s_state[sidx] = 0;
        s_priority[sidx] = 0.0f;
        s_image[sidx] = 0.0f;
    }}

    // Load halo regions (border threads load extra pixels)
    // Each thread may load up to 3 halo pixels (corner + 2 edges)

    // Left halo (threads with tx == 0)
    if (tx == 0) {{
        int hx = gx - {HALO};
        int hy = gy;
        int hsidx = sy * {shared_w} + (sx - {HALO});
        if (hx >= 0 && hx < width && hy >= 0 && hy < height) {{
            int hidx = hy * width + hx;
            s_labels[hsidx] = labels[hidx];
            s_state[hsidx] = state[hidx];
            s_priority[hsidx] = priority[hidx];
            s_image[hsidx] = image[hidx];
        }} else {{
            s_labels[hsidx] = 0;
            s_state[hsidx] = 0;
            s_priority[hsidx] = 0.0f;
            s_image[hsidx] = 0.0f;
        }}
    }}

    // Right halo (threads with tx == {TILE_W - 1})
    if (tx == {TILE_W - 1}) {{
        int hx = gx + {HALO};
        int hy = gy;
        int hsidx = sy * {shared_w} + (sx + {HALO});
        if (hx >= 0 && hx < width && hy >= 0 && hy < height) {{
            int hidx = hy * width + hx;
            s_labels[hsidx] = labels[hidx];
            s_state[hsidx] = state[hidx];
            s_priority[hsidx] = priority[hidx];
            s_image[hsidx] = image[hidx];
        }} else {{
            s_labels[hsidx] = 0;
            s_state[hsidx] = 0;
            s_priority[hsidx] = 0.0f;
            s_image[hsidx] = 0.0f;
        }}
    }}

    // Top halo (threads with ty == 0)
    if (ty == 0) {{
        int hx = gx;
        int hy = gy - {HALO};
        int hsidx = (sy - {HALO}) * {shared_w} + sx;
        if (hx >= 0 && hx < width && hy >= 0 && hy < height) {{
            int hidx = hy * width + hx;
            s_labels[hsidx] = labels[hidx];
            s_state[hsidx] = state[hidx];
            s_priority[hsidx] = priority[hidx];
            s_image[hsidx] = image[hidx];
        }} else {{
            s_labels[hsidx] = 0;
            s_state[hsidx] = 0;
            s_priority[hsidx] = 0.0f;
            s_image[hsidx] = 0.0f;
        }}
    }}

    // Bottom halo (threads with ty == {TILE_H - 1})
    if (ty == {TILE_H - 1}) {{
        int hx = gx;
        int hy = gy + {HALO};
        int hsidx = (sy + {HALO}) * {shared_w} + sx;
        if (hx >= 0 && hx < width && hy >= 0 && hy < height) {{
            int hidx = hy * width + hx;
            s_labels[hsidx] = labels[hidx];
            s_state[hsidx] = state[hidx];
            s_priority[hsidx] = priority[hidx];
            s_image[hsidx] = image[hidx];
        }} else {{
            s_labels[hsidx] = 0;
            s_state[hsidx] = 0;
            s_priority[hsidx] = 0.0f;
            s_image[hsidx] = 0.0f;
        }}
    }}

    // Corner halos (only corner threads)
    // Top-left corner
    if (tx == 0 && ty == 0) {{
        int hx = gx - {HALO};
        int hy = gy - {HALO};
        int hsidx = (sy - {HALO}) * {shared_w} + (sx - {HALO});
        if (hx >= 0 && hx < width && hy >= 0 && hy < height) {{
            int hidx = hy * width + hx;
            s_labels[hsidx] = labels[hidx];
            s_state[hsidx] = state[hidx];
            s_priority[hsidx] = priority[hidx];
            s_image[hsidx] = image[hidx];
        }} else {{
            s_labels[hsidx] = 0;
            s_state[hsidx] = 0;
            s_priority[hsidx] = 0.0f;
            s_image[hsidx] = 0.0f;
        }}
    }}

    // Top-right corner
    if (tx == {TILE_W - 1} && ty == 0) {{
        int hx = gx + {HALO};
        int hy = gy - {HALO};
        int hsidx = (sy - {HALO}) * {shared_w} + (sx + {HALO});
        if (hx >= 0 && hx < width && hy >= 0 && hy < height) {{
            int hidx = hy * width + hx;
            s_labels[hsidx] = labels[hidx];
            s_state[hsidx] = state[hidx];
            s_priority[hsidx] = priority[hidx];
            s_image[hsidx] = image[hidx];
        }} else {{
            s_labels[hsidx] = 0;
            s_state[hsidx] = 0;
            s_priority[hsidx] = 0.0f;
            s_image[hsidx] = 0.0f;
        }}
    }}

    // Bottom-left corner
    if (tx == 0 && ty == {TILE_H - 1}) {{
        int hx = gx - {HALO};
        int hy = gy + {HALO};
        int hsidx = (sy + {HALO}) * {shared_w} + (sx - {HALO});
        if (hx >= 0 && hx < width && hy >= 0 && hy < height) {{
            int hidx = hy * width + hx;
            s_labels[hsidx] = labels[hidx];
            s_state[hsidx] = state[hidx];
            s_priority[hsidx] = priority[hidx];
            s_image[hsidx] = image[hidx];
        }} else {{
            s_labels[hsidx] = 0;
            s_state[hsidx] = 0;
            s_priority[hsidx] = 0.0f;
            s_image[hsidx] = 0.0f;
        }}
    }}

    // Bottom-right corner
    if (tx == {TILE_W - 1} && ty == {TILE_H - 1}) {{
        int hx = gx + {HALO};
        int hy = gy + {HALO};
        int hsidx = (sy + {HALO}) * {shared_w} + (sx + {HALO});
        if (hx >= 0 && hx < width && hy >= 0 && hy < height) {{
            int hidx = hy * width + hx;
            s_labels[hsidx] = labels[hidx];
            s_state[hsidx] = state[hidx];
            s_priority[hsidx] = priority[hidx];
            s_image[hsidx] = image[hidx];
        }} else {{
            s_labels[hsidx] = 0;
            s_state[hsidx] = 0;
            s_priority[hsidx] = 0.0f;
            s_image[hsidx] = 0.0f;
        }}
    }}

    __syncthreads();

    // ========================================
    // Phase 2: Inner iterations in shared memory
    // ========================================

    // Cache image value for this pixel (doesn't change)
    float my_image = s_image[sidx];

    // Track if this thread made any changes
    int my_changed = 0;

    for (int iter = 0; iter < {inner_iterations}; iter++) {{
        // Only process unlabeled pixels in the main tile (not halo)
        if (s_state[sidx] == 2) {{
            float best_priority = s_priority[sidx];
            int best_label = 0;
            int found_label = 0;

            // Check all neighbors
            {neighbor_code}

            // Update if we found a better label
            if (found_label && best_label != 0 && best_priority < s_priority[sidx]) {{
                s_labels[sidx] = best_label;
                s_priority[sidx] = best_priority;
                my_changed = 1;
            }}
        }}

        __syncthreads();
    }}

    // ========================================
    // Phase 3: Write back to global memory
    // ========================================

    if (gx < width && gy < height) {{
        int gidx = gy * width + gx;
        labels[gidx] = s_labels[sidx];
        state[gidx] = s_state[sidx];
        priority[gidx] = s_priority[sidx];
    }}

    // Aggregate changes using shared memory reduction
    if (tx == 0 && ty == 0) {{
        s_changed = 0;
    }}
    __syncthreads();

    if (my_changed) {{
        atomicAdd(&s_changed, 1);
    }}
    __syncthreads();

    // One thread per block updates global counter
    if (tx == 0 && ty == 0 && s_changed > 0) {{
        atomicAdd(global_changed, s_changed);
    }}
}}
"""  # noqa: E501

    return cp.RawKernel(kernel_code, "watershed_block_async_2d")


@cp.memoize(for_each_device=True)
def _get_watershed_compact_step_kernel_2d(connectivity=1):
    """Get iteration kernel for 2D compact CA-watershed.

    This kernel implements the compact watershed variant, which adds a
    distance penalty to encourage more regularly-shaped regions.

    The priority for each pixel is:
        priority = image[pixel] + compactness * euclidean_distance(pixel, source)

    where source is the original marker location for that basin.

    Parameters
    ----------
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity

    Returns
    -------
    kernel : cupy.RawKernel
        Compiled CUDA kernel
    """  # noqa: E501
    neighbors = _get_neighbor_offsets_2d(connectivity)

    # Generate neighbor checking code for compact watershed
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

            // If neighbor has a label (non-zero), it can propagate to current
            // pixel
            if (nlabel != 0) {{
                // Get the source marker location for this neighbor's basin
                int src_x = source_x[nidx];
                int src_y = source_y[nidx];

                // Compute Euclidean distance from current pixel to source
                float dx_dist = (float)(x - src_x);
                float dy_dist = (float)(y - src_y);
                float euclidean_dist = sqrtf(dx_dist * dx_dist + dy_dist * dy_dist);

                // Compute priority: image value + compactness * distance
                float new_priority = image[idx] + compactness * euclidean_dist;

                // Enforce monotonicity: priority must not decrease
                // (matches scikit-image behavior for fair tie-breaking)
                float neighbor_priority = priority[nidx];
                if (new_priority < neighbor_priority) {{
                    new_priority = neighbor_priority;
                }}

                // Update if this is a better path (lower priority)
                if (new_priority < best_priority) {{
                    best_priority = new_priority;
                    best_label = nlabel;
                    best_src_x = src_x;
                    best_src_y = src_y;
                    found_label = 1;
                }}
            }}
        }}
    }}
"""  # noqa: E501

    kernel_code = f"""
extern "C" __global__
void watershed_compact_step_2d(
    const float* __restrict__ image,
    int* __restrict__ labels,
    unsigned char* __restrict__ state,
    float* __restrict__ priority,
    int* __restrict__ source_x,
    int* __restrict__ source_y,
    int* __restrict__ changed,
    float compactness,
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
    float best_priority = priority[idx];
    int best_label = 0;
    int best_src_x = -1;
    int best_src_y = -1;
    int found_label = 0;

    {neighbor_code}

    // Update pixel if we found a better label
    if (found_label && best_label != 0) {{
        if (best_priority < priority[idx]) {{
            labels[idx] = best_label;
            priority[idx] = best_priority;
            source_x[idx] = best_src_x;
            source_y[idx] = best_src_y;

            // Signal that a change occurred
            atomicAdd(changed, 1);
        }}
    }}
}}
"""

    return cp.RawKernel(kernel_code, "watershed_compact_step_2d")


# --- Watershed line post-processing kernel ---


@cp.memoize(for_each_device=True)
def _get_watershed_line_kernel(ndim, connectivity=1):
    """Get post-processing kernel to create 1-pixel watershed lines.

    After the CA iteration has converged and all pixels have final labels,
    this kernel identifies boundary pixels. To produce 1-pixel-wide lines
    (rather than 2-pixel), only the pixel that arrived "later" in the
    flooding (higher priority) is set to 0. For equal priorities, the
    pixel with the higher label value is zeroed as a consistent
    tie-breaker.

    Uses a 1D grid for all dimensionalities. The kernel takes flat
    arrays (labels_in, priority, labels_out) plus dimension sizes to
    recover coordinates for neighbor bounds checking.

    Parameters
    ----------
    ndim : int
        Number of dimensions (1, 2, or 3).
    connectivity : int
        Neighborhood connectivity.
    """
    # Generate coordinate extraction and neighbor check code
    if ndim == 1:
        coord_code = "int x = idx;"
        dim_params = "int size"
        size_expr = "size"
        neighbors = _get_neighbor_offsets_1d()
        neighbor_check = ""
        for i, (dx,) in enumerate(neighbors):
            neighbor_check += f"""
        {{
            int nx = x + ({dx});
            if (nx >= 0 && nx < size) {{
                int nidx = nx;
                int nlabel = labels_in[nidx];
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
    elif ndim == 2:
        coord_code = "int x = idx % width;\n    int y = idx / width;"
        dim_params = "int width, int height"
        size_expr = "width * height"
        neighbors = _get_neighbor_offsets_2d(connectivity)
        neighbor_check = ""
        for i, (dy, dx) in enumerate(neighbors):
            neighbor_check += f"""
        {{
            int ny = y + ({dy});
            int nx = x + ({dx});
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {{
                int nidx = ny * width + nx;
                int nlabel = labels_in[nidx];
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
    else:  # ndim == 3
        coord_code = (
            "int x = idx % width;\n"
            "    int y = (idx / width) % height;\n"
            "    int z = idx / (width * height);"
        )
        dim_params = "int width, int height, int depth"
        size_expr = "width * height * depth"
        neighbors = _get_neighbor_offsets_3d(connectivity)
        neighbor_check = ""
        for i, (dz, dy, dx) in enumerate(neighbors):
            neighbor_check += f"""
        {{
            int nz = z + ({dz});
            int ny = y + ({dy});
            int nx = x + ({dx});
            if (nz >= 0 && nz < depth && ny >= 0 && ny < height && nx >= 0 && nx < width) {{
                int nidx = (nz * height + ny) * width + nx;
                int nlabel = labels_in[nidx];
                if (nlabel != 0 && nlabel != my_label) {{
                    float npri = priority[nidx];
                    if (npri < my_pri ||
                        (npri == my_pri && nlabel < my_label)) {{
                        is_boundary = 1;
                    }}
                }}
            }}
        }}
"""  # noqa: E501

    kernel_code = f"""
extern "C" __global__
void watershed_line(
    const int* __restrict__ labels_in,
    const float* __restrict__ priority,
    int* __restrict__ labels_out,
    {dim_params}
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = {size_expr};
    if (idx >= size) return;

    {coord_code}

    int my_label = labels_in[idx];
    if (my_label == 0) {{
        labels_out[idx] = 0;
        return;
    }}

    float my_pri = priority[idx];
    int is_boundary = 0;
    {neighbor_check}

    labels_out[idx] = is_boundary ? 0 : my_label;
}}
"""
    return cp.RawKernel(kernel_code, "watershed_line")


def _validate_inputs(image, markers, mask, connectivity):
    """Validate and prepare inputs for watershed algorithm.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (gradient magnitude or similar), 2D or 3D
    markers : int, cupy.ndarray, or None
        The marker image. If None, markers are determined as the local
        minima of the image. If int, that many regularly-spaced markers
        are generated. If array, used as-is.
    mask : cupy.ndarray or None
        Optional mask array
    connectivity : int
        Neighborhood connectivity as integer.
        For 2D: 1 (4-connectivity) or 2 (8-connectivity)
        For 3D: 1 (6-connectivity), 2 (18-connectivity), or 3
        (26-connectivity)

    Returns
    -------
    image : cupy.ndarray
        Validated image as float32
    markers : cupy.ndarray
        Validated markers as int32
    mask : cupy.ndarray or None
        Validated mask as uint8
    connectivity : int
        Connectivity as integer
    """
    # Check image first to get ndim
    if not isinstance(image, cp.ndarray):
        image = cp.asarray(image)

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
    n_pixels = image.size
    if mask is not None:
        if not isinstance(mask, cp.ndarray):
            mask = cp.asarray(mask)

        if mask.shape != image.shape:
            raise ValueError(
                f"mask shape {mask.shape} must match image shape {image.shape}"
            )

        mask = mask.astype(cp.uint8)
        n_pixels = int(cp.sum(mask))

    # Handle markers
    if markers is None:
        # Auto-detect markers from local minima (matching scikit-image)
        markers_bool = local_minima(image, connectivity=connectivity)
        if mask is not None:
            markers_bool = markers_bool * mask.astype(bool)
        footprint = ndi.generate_binary_structure(ndim, connectivity)
        markers = ndi.label(markers_bool, structure=footprint)[0]
        markers = markers.astype(cp.int32)
    elif not isinstance(markers, (cp.ndarray, np.ndarray, list, tuple)):
        # Assume int: generate that many regularly-spaced markers
        n_markers = int(markers)
        # Scale n_markers by fraction of masked pixels (like scikit-image)
        markers = regular_seeds(
            image.shape, int(n_markers / (n_pixels / image.size))
        )
        if mask is not None:
            markers *= mask.astype(markers.dtype)
        markers = markers.astype(cp.int32)
    else:
        if not isinstance(markers, cp.ndarray):
            markers = cp.asarray(markers)

        if mask is not None:
            markers = markers * mask.astype(markers.dtype)

        if markers.shape != image.shape:
            raise ValueError(
                f"markers shape {markers.shape} must match "
                f"image shape {image.shape}"
            )

        # Convert markers to int32
        if markers.dtype != cp.int32:
            markers = markers.astype(cp.int32)

    return image, markers, mask, connectivity


def _watershed_standard(
    image_flat,
    markers,
    mask_flat,
    has_mask,
    connectivity,
    ndim,
    size,
    width,
    height,
    depth,
    blocks,
    threads_per_block,
    grid_size_2d,
    block_size_2d,
    max_iterations,
    use_age=True,
):
    """Run standard watershed algorithm (compactness=0).

    Uses image intensity as priority (like scikit-image) to produce
    regions that follow image gradients rather than compact shapes.
    Supports both 2D and 3D images.
    """
    # Initialize state arrays (as contiguous flat arrays)
    labels = cp.zeros(size, dtype=cp.int32)
    state = cp.zeros(size, dtype=cp.uint8)
    priority = cp.zeros(size, dtype=cp.float32)
    changed = cp.zeros(1, dtype=cp.int32)

    if use_age:
        age = cp.zeros(size, dtype=cp.int32)
    else:
        age = None

    # Initialize from markers (same kernel for 2D and 3D)
    init_kernel = _get_watershed_init_kernel(use_age=use_age)

    init_args = [
        image_flat,
        markers.ravel(),
        labels,
        state,
        priority,
    ]
    if use_age:
        init_args.append(age)
    init_args.extend([mask_flat, has_mask, int(size)])

    init_kernel(
        (blocks,),
        (threads_per_block,),
        tuple(init_args),
    )

    # Iteratively propagate labels
    if ndim == 1:
        step_kernel = _get_watershed_step_kernel_1d(use_age=use_age)

        step_args_base = [image_flat, labels, state, priority]
        if use_age:
            step_args_base.append(age)
        step_args_base.extend([changed, int(size)])
        step_args = tuple(step_args_base)

        for iteration in range(max_iterations):
            changed[0] = 0
            step_kernel((blocks,), (threads_per_block,), step_args)
            if changed[0] == 0:
                break
    elif ndim == 2:
        step_kernel = _get_watershed_step_kernel_2d(
            connectivity, use_age=use_age
        )

        step_args_base = [image_flat, labels, state, priority]
        if use_age:
            step_args_base.append(age)
        step_args_base.extend([changed, int(width), int(height)])
        step_args = tuple(step_args_base)

        for iteration in range(max_iterations):
            changed[0] = 0
            step_kernel(grid_size_2d, block_size_2d, step_args)
            if changed[0] == 0:
                break
    else:  # ndim == 3
        step_kernel = _get_watershed_step_kernel_3d(
            connectivity, use_age=use_age
        )

        step_args_base = [image_flat, labels, state, priority]
        if use_age:
            step_args_base.append(age)
        step_args_base.extend([changed, int(width), int(height), int(depth)])
        step_args = tuple(step_args_base)

        for iteration in range(max_iterations):
            changed[0] = 0
            step_kernel((blocks,), (threads_per_block,), step_args)
            if changed[0] == 0:
                break

    return labels, state, priority, changed


def _watershed_standard_block_async(
    image_flat,
    markers,
    mask_flat,
    has_mask,
    connectivity,
    size,
    width,
    height,
    blocks,
    threads_per_block,
    max_iterations,
    inner_iterations=16,
):
    """Run standard watershed with block-asynchronous algorithm.

    Uses shared memory tiling to reduce global memory traffic.
    Each block performs multiple iterations locally before synchronizing.

    Parameters
    ----------
    inner_iterations : int
        Number of iterations per block before global sync. Default is 8.
    """
    # Initialize state arrays (as contiguous flat arrays)
    labels = cp.zeros(size, dtype=cp.int32)
    state = cp.zeros(size, dtype=cp.uint8)
    priority = cp.zeros(size, dtype=cp.float32)
    changed = cp.zeros(1, dtype=cp.int32)

    # Initialize from markers (same as standard version)
    init_kernel = _get_watershed_init_kernel()

    init_kernel(
        (blocks,),
        (threads_per_block,),
        (
            image_flat,
            markers.ravel(),
            labels,
            state,
            priority,
            mask_flat,
            has_mask,
            int(size),
        ),
    )

    # Get block-async kernel
    step_kernel = _get_watershed_block_async_kernel_2d(
        connectivity, inner_iterations
    )

    # Block configuration must match TILE_W x TILE_H
    block_size = (TILE_W, TILE_H)
    grid_size = (
        (width + TILE_W - 1) // TILE_W,
        (height + TILE_H - 1) // TILE_H,
    )

    # Outer iteration loop (each outer iteration does inner_iterations locally)
    # Fewer outer iterations needed since each does multiple inner iterations
    max_outer_iterations = (
        max_iterations + inner_iterations - 1
    ) // inner_iterations

    for iteration in range(max_outer_iterations):
        # Reset change counter
        changed[0] = 0

        # Run block-async kernel (does inner_iterations per block)
        step_kernel(
            grid_size,
            block_size,
            (
                image_flat,
                labels,
                state,
                priority,
                changed,
                int(width),
                int(height),
            ),
        )

        # Check for convergence
        if changed[0] == 0:
            break

    return labels, state, priority, changed


def _watershed_compact(
    image_flat,
    markers,
    mask_flat,
    has_mask,
    connectivity,
    compactness,
    size,
    width,
    height,
    blocks,
    threads_per_block,
    grid_size,
    block_size,
    max_iterations,
):
    """Run compact watershed algorithm (compactness > 0).

    This code path uses floating-point priorities that combine
    image values with Euclidean distance from the source marker.
    """
    # Initialize state arrays (as contiguous flat arrays)
    labels = cp.zeros(size, dtype=cp.int32)
    state = cp.zeros(size, dtype=cp.uint8)
    priority = cp.zeros(size, dtype=cp.float32)
    source_x = cp.zeros(size, dtype=cp.int32)
    source_y = cp.zeros(size, dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    # Initialize from markers
    init_kernel = _get_watershed_compact_init_kernel()

    init_kernel(
        (blocks,),
        (threads_per_block,),
        (
            markers.ravel(),
            labels,
            state,
            priority,
            source_x,
            source_y,
            mask_flat,
            has_mask,
            int(width),
            int(size),
        ),
    )

    # Iteratively propagate labels
    step_kernel = _get_watershed_compact_step_kernel_2d(connectivity)

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
                priority,
                source_x,
                source_y,
                changed,
                cp.float32(compactness),
                int(width),
                int(height),
            ),
        )

        # Check for convergence
        if changed[0] == 0:
            break

    return labels, state, priority, source_x, source_y, changed


def watershed(
    image,
    markers=None,
    connectivity=1,
    mask=None,
    compactness=0,
    watershed_line=False,
    *,
    use_block_async=None,
    use_age=False,
):
    """Watershed segmentation using cellular automaton algorithm.

    This function implements a GPU-accelerated watershed transform using
    a cellular automaton approach. The algorithm is particularly efficient
    for seeded watershed segmentation on 2D and 3D images.

    Parameters
    ----------
    image : cupy.ndarray, shape (M, N) or (D, M, N)
        Input image (typically a gradient magnitude or distance transform).
        Lower values have higher priority for watershed expansion.
        Supports both 2D and 3D images.
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
    connectivity : int, optional
        Neighborhood connectivity as integer:
        - For 2D images:
          - 1: 4-connectivity (von Neumann neighborhood - orthogonal neighbors)
          - 2: 8-connectivity (Moore neighborhood - includes diagonals)
        - For 3D images:
          - 1: 6-connectivity (face neighbors)
          - 2: 18-connectivity (face + edge neighbors)
          - 3: 26-connectivity (face + edge + corner neighbors)
        Default is 1.
    mask : cupy.ndarray of bool, shape (M, N) or (D, M, N), optional
        If provided, only pixels/voxels where mask is True will be segmented.
        Useful for restricting watershed to regions of interest.
    compactness : float, optional
        Use compact watershed with given compactness parameter (2D only).
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
    use_block_async : bool or None, optional
        Parameter to control algorithm variant for standard watershed.
        If None (default), automatically chooses based on image size.
        If True, uses block-asynchronous algorithm with shared memory tiling.
        If False, uses synchronous algorithm.
        This variant is only implemented for 2D images and only affects the
        standard watershed (compactness=0). Not used when use_age=True.
    use_age : bool, optional
        If True (default), use an age (hop distance) counter as a
        tie-breaker when two labels arrive at a pixel with equal priority.
        This more closely matches scikit-image's age-based tie-breaking
        behavior, producing fairer splits on plateau regions. If False,
        ties are broken by neighbor iteration order (less deterministic).
        Only affects the standard watershed (compactness=0).

    Returns
    -------
    labels : cupy.ndarray of int, shape (M, N) or (D, M, N)
        Labeled array, where each basin is assigned a unique positive
        integer label matching the input markers.

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or invalid parameters.
    NotImplementedError
        If watershed_line is used, or if compactness is used with 3D images.

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

    The compact watershed extension follows the approach from scikit-image [2]_,
    which adds a distance penalty to encourage more regularly-shaped regions.
    This is useful for superpixel generation (2D only).

    Current limitations:
    - watershed_line parameter is not yet implemented
    - compactness parameter is only supported for 2D images
    - block-async optimization is only available for 2D images

    References
    ----------
    .. [1] Kauffmann, C., & Piche, N. (2010). Cellular automaton for
           ultra-fast watershed transform on GPU. In Pattern Recognition
           (ICPR), 2010 20th International Conference on (pp. 447-450). IEEE.

    .. [2] Neubert, P., & Protzel, P. (2014). Compact Watershed and Preemptive
           SLIC: On Improving Trade-offs of Superpixel Segmentation Algorithms.
           In Pattern Recognition (ICPR), 2014 22nd International Conference on.

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
    # Save original marker dtype for output casting
    if isinstance(markers, (cp.ndarray, np.ndarray)):
        out_dtype = markers.dtype
    else:
        out_dtype = cp.int32

    # Validate and prepare inputs
    image, markers, mask, connectivity = _validate_inputs(
        image, markers, mask, connectivity
    )

    ndim = image.ndim

    # Check dimensionality
    if ndim not in (1, 2, 3):
        raise NotImplementedError(
            f"Only 1D, 2D and 3D images are supported, got {ndim}D"
        )

    # Check compactness support
    if compactness != 0 and ndim != 2:
        raise NotImplementedError(
            "compactness parameter is only supported for 2D images"
        )

    # Get dimensions
    if ndim == 1:
        (size,) = image.shape
        height = width = depth = None
    elif ndim == 2:
        height, width = image.shape
        depth = None
        size = height * width
    else:  # ndim == 3
        depth, height, width = image.shape
        size = depth * height * width

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

    # 2D block and grid configuration for step kernel (only used for 2D)
    if ndim == 2:
        block_size_2d = (16, 16)
        grid_size_2d = (
            (width + block_size_2d[0] - 1) // block_size_2d[0],
            (height + block_size_2d[1] - 1) // block_size_2d[1],
        )
    else:
        block_size_2d = None
        grid_size_2d = None

    # Upper bound on iterations (shouldn't need this many)
    max_iterations = max(image.shape) * 2

    # Ensure image is contiguous and flat for kernel
    image_flat = cp.ascontiguousarray(image.ravel())

    # Use different code paths for standard vs compact watershed
    if compactness == 0:
        # Determine whether to use block-async algorithm (2D only)
        # Block-async does not support age tie-breaking yet
        if use_block_async is None:
            use_block_async = (
                ndim == 2 and min(height, width) >= 128 and not use_age
            )

    # common kwargs regardless of which watershed implementation is called
    common_kwargs = dict(
        image_flat=image_flat,
        markers=markers,
        mask_flat=mask_flat,
        has_mask=has_mask,
        connectivity=connectivity,
        size=size,
        width=width,
        height=height,
        blocks=blocks,
        threads_per_block=threads_per_block,
        max_iterations=max_iterations,
    )

    if compactness == 0:
        if use_block_async and ndim == 2 and not use_age:
            labels, state, priority, changed = _watershed_standard_block_async(
                **common_kwargs,
                inner_iterations=16,
            )
        else:
            labels, state, priority, changed = _watershed_standard(
                **common_kwargs,
                ndim=ndim,
                depth=depth,
                grid_size_2d=grid_size_2d,
                block_size_2d=block_size_2d,
                use_age=use_age,
            )
    else:
        (
            labels,
            state,
            priority,
            source_x,
            source_y,
            changed,
        ) = _watershed_compact(
            **common_kwargs,
            compactness=compactness,
            grid_size=grid_size_2d,
            block_size=block_size_2d,
        )

    # Post-processing: create watershed lines if requested
    if watershed_line:
        labels_out = cp.empty_like(labels)
        wl_kernel = _get_watershed_line_kernel(ndim, connectivity)
        # Kernel expects (width, height[, depth]) = reversed shape order
        dim_args = tuple(int(s) for s in reversed(image.shape))
        wl_kernel(
            (blocks,),
            (threads_per_block,),
            (labels, priority, labels_out, *dim_args),
        )
        labels = labels_out

    # Reshape back to original dimensions
    result = labels.reshape(image.shape)

    # Cast to original marker dtype
    if result.dtype != out_dtype:
        result = result.astype(out_dtype)
    return result
