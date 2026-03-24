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

import itertools

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage as ndi

from ..morphology.extrema import local_minima
from ..util._regular_grid import regular_seeds

# Preamble for CUDA kernels that use fixed-width integer types (libcudacxx)
_KERNEL_PREAMBLE = "#include <cuda/std/cstdint>\n"

# Mapping from CuPy/NumPy integer dtype to fixed-width C type for kernel codegen
_DTYPE_TO_CTYPE = {
    cp.dtype("int8"): "cuda::std::int8_t",
    cp.dtype("uint8"): "cuda::std::uint8_t",
    cp.dtype("int16"): "cuda::std::int16_t",
    cp.dtype("uint16"): "cuda::std::uint16_t",
    cp.dtype("int32"): "cuda::std::int32_t",
    cp.dtype("uint32"): "cuda::std::uint32_t",
    cp.dtype("int64"): "cuda::std::int64_t",
    cp.dtype("uint64"): "cuda::std::uint64_t",
}


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
    # fmt: off
    if connectivity == 1:
        # 4-connectivity (cross pattern)
        return [
            (-1,  0),  # top
            ( 1,  0),  # bottom
            ( 0, -1),  # left
            ( 0,  1),  # right
        ]
    else:
        # 8-connectivity (square pattern)
        return [
            (-1,  0),  # top
            ( 1,  0),  # bottom
            ( 0, -1),  # left
            ( 0,  1),  # right
            (-1, -1),  # top-left
            (-1,  1),  # top-right
            ( 1, -1),  # bottom-left
            ( 1,  1),  # bottom-right
        ]
    # fmt: on


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
    # fmt: off
    # Face neighbors (6-connectivity)
    face_neighbors = [
        (-1,  0,  0),
        ( 1,  0,  0),
        ( 0, -1,  0),
        ( 0,  1,  0),
        ( 0,  0, -1),
        ( 0,  0,  1),
    ]

    if connectivity == 1:
        return face_neighbors

    # Edge neighbors (12 additional)
    edge_neighbors = [
        (-1, -1,  0),
        (-1,  1,  0),
        (-1,  0, -1),
        (-1,  0,  1),
        ( 1, -1,  0),
        ( 1,  1,  0),
        ( 1,  0, -1),
        ( 1,  0,  1),
        ( 0, -1, -1),
        ( 0, -1,  1),
        ( 0,  1, -1),
        ( 0,  1,  1),
    ]

    if connectivity == 2:
        return face_neighbors + edge_neighbors

    # Corner neighbors (8 additional)
    corner_neighbors = [
        (-1, -1, -1),
        (-1, -1,  1),
        (-1,  1, -1),
        (-1,  1,  1),
        ( 1, -1, -1),
        ( 1, -1,  1),
        ( 1,  1, -1),
        ( 1,  1,  1),
    ]
    # fmt: on

    # connectivity == 3: 26-connectivity
    return face_neighbors + edge_neighbors + corner_neighbors


def _get_neighbor_offsets(ndim, connectivity):
    """Get neighbor offsets for the given dimensionality and connectivity.

    An offset is included if the number of non-zero components is at most
    ``connectivity``. This matches the convention used by
    ``scipy.ndimage.generate_binary_structure(ndim, connectivity)``.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    connectivity : int
        Maximum number of orthogonal steps to reach a neighbor.
        Must be in [1, ndim].

    Returns
    -------
    neighbors : list of tuple
        List of offset tuples, each of length ndim. Sorted by
        connectivity level (face neighbors first, then edge, then
        corner, etc.).
    """
    # Use hand-tuned lists for common cases (avoids numpy overhead)
    if ndim == 1:
        return _get_neighbor_offsets_1d()
    elif ndim == 2:
        return _get_neighbor_offsets_2d(connectivity)
    elif ndim == 3:
        return _get_neighbor_offsets_3d(connectivity)

    # General nD case

    # All offsets in {-1, 0, 1}^ndim, excluding the origin, filtered by
    # connectivity (number of non-zero components), sorted by level.
    offsets = []
    for off in itertools.product((-1, 0, 1), repeat=ndim):
        n_nonzero = sum(c != 0 for c in off)
        if 0 < n_nonzero <= connectivity:
            offsets.append((n_nonzero, off))
    offsets.sort(key=lambda x: x[0])
    return [off for _, off in offsets]


@cp.memoize(for_each_device=True)
def _generate_coord_code(ndim):
    """Generate CUDA code to extract nD coordinates from a flat index.

    Returns (dim_names, dim_params, size_expr, coord_code) where:
    - dim_names: list of dimension size variable names in C-order
    - dim_params: kernel parameter declaration string
    - size_expr: C expression for total size
    - coord_code: C code to compute c_0..c_{ndim-1} from idx
    """
    dim_names = [f"dim_{j}" for j in range(ndim)]
    dim_params = ", ".join(f"int {d}" for d in dim_names)
    size_expr = " * ".join(dim_names)

    coord_lines = []
    for j in range(ndim - 1, -1, -1):
        if j == ndim - 1:
            coord_lines.append(f"int c_{j} = idx % {dim_names[j]};")
            if ndim > 1:
                coord_lines.append(f"int _rem_{j} = idx / {dim_names[j]};")
        elif j == 0:
            coord_lines.append(f"int c_0 = _rem_{j + 1};")
        else:
            coord_lines.append(f"int c_{j} = _rem_{j + 1} % {dim_names[j]};")
            coord_lines.append(f"int _rem_{j} = _rem_{j + 1} / {dim_names[j]};")
    coord_code = "\n    ".join(coord_lines)

    return dim_names, dim_params, size_expr, coord_code


def _generate_neighbor_nidx(ndim, dim_names, offsets):
    """Generate CUDA code for neighbor coordinate + flat index computation.

    For each neighbor offset, generates:
    - nc_0..nc_{ndim-1} neighbor coordinate declarations
    - bounds check expression
    - nidx flat index expression

    Returns (nc_decl_str, bounds_str, nidx_expr) for each offset.
    """
    results = []
    for offset in offsets:
        nc_decls = []
        bounds = []
        for j, off in enumerate(offset):
            nc_decls.append(f"int nc_{j} = c_{j} + ({off});")
            bounds.append(f"nc_{j} >= 0 && nc_{j} < {dim_names[j]}")

        nc_decl_str = "\n        ".join(nc_decls)
        bounds_str = " && ".join(bounds)

        nidx_parts = [f"nc_{j}" for j in range(ndim)]
        nidx_expr = nidx_parts[0]
        for j in range(1, ndim):
            nidx_expr = f"({nidx_expr}) * {dim_names[j]} + {nidx_parts[j]}"
        results.append((nc_decl_str, bounds_str, nidx_expr))
    return results


##############################################
# Non-compact n-dimensional seeded watershed #
##############################################


# CUDA kernel for initialization (standard watershed)
@cp.memoize(for_each_device=True)
def _get_watershed_init_kernel(
    ndim=0, compact=False, use_age=False, label_ctype="cuda::std::int32_t"
):
    """Get initialization kernel for CA-watershed (standard or compact).

    Parameters
    ----------
    ndim : int
        Number of dimensions. Only needed when compact=True (for source
        coordinate extraction).
    compact : bool
        If True, initialize source coordinate arrays and set marker
        priority to 0. If False, set marker priority from image values.
    use_age : bool
        If True, initialize an age array for tie-breaking.
    label_ctype : str
        C type for label arrays.
    """
    L = label_ctype

    # Age arrays (non-compact only)
    age_params = "int* age," if use_age else ""
    age_init_mask = "age[idx] = 0;" if use_age else ""
    age_init_marker = "age[idx] = 0;" if use_age else ""
    age_init_unlabeled = "age[idx] = 2147483647;" if use_age else ""

    # Source coordinate arrays (compact only)
    if compact:
        dim_names, dim_params, size_expr, coord_code = _generate_coord_code(
            ndim
        )
        src_names = [f"source_{j}" for j in range(ndim)]
        src_params = ", ".join(f"int* {s}" for s in src_names) + ","
        src_init_mask = "\n        ".join(f"{s}[idx] = -1;" for s in src_names)
        src_init_marker = "\n        ".join(
            f"{s}[idx] = c_{j};" for j, s in enumerate(src_names)
        )
        src_init_unlabeled = "\n        ".join(
            f"{s}[idx] = -1;" for s in src_names
        )
        image_param = ""
        size_param = dim_params
        size_code = f"int size = {size_expr};"
        coord_extract = coord_code
        marker_priority = "0.0f"
    else:
        src_params = ""
        src_init_mask = ""
        src_init_marker = ""
        src_init_unlabeled = ""
        image_param = "const float* image,"
        size_param = "int size"
        size_code = ""
        coord_extract = ""
        marker_priority = "image[idx]"

    kernel_code = (
        _KERNEL_PREAMBLE
        + f"""
extern "C" __global__
void watershed_init(
    {image_param}
    const {L}* markers,
    {L}* labels,
    unsigned char* state,
    float* priority,
    {age_params}
    {src_params}
    const unsigned char* mask,
    int has_mask,
    {size_param}
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    {size_code}

    if (idx >= size) return;

    {coord_extract}

    if (has_mask && !mask[idx]) {{
        labels[idx] = 0;
        state[idx] = 0;
        priority[idx] = 0.0f;
        {age_init_mask}
        {src_init_mask}
        return;
    }}

    {L} marker_label = markers[idx];

    if (marker_label != 0) {{
        labels[idx] = marker_label;
        state[idx] = 1;  // LABELED
        priority[idx] = {marker_priority};
        {age_init_marker}
        {src_init_marker}
    }} else {{
        labels[idx] = 0;
        state[idx] = 2;  // UNLABELED
        priority[idx] = 3.4028235e+38f;  // FLT_MAX
        {age_init_unlabeled}
        {src_init_unlabeled}
    }}
}}
"""
    )
    return cp.RawKernel(kernel_code, "watershed_init")


@cp.memoize(for_each_device=True)
def _get_watershed_step_kernel(
    ndim,
    connectivity=1,
    compact=False,
    use_age=False,
    label_ctype="cuda::std::int32_t",
):
    """Get iteration kernel for nD CA-watershed (standard or compact).

    Uses a 1D grid for all dimensionalities, recovering nD coordinates
    from the flat index for neighbor bounds checking.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    connectivity : int
        Neighborhood connectivity (1 to ndim).
    compact : bool
        If True, generate compact watershed kernel with Euclidean
        distance penalty and source coordinate tracking.
    use_age : bool
        If True, use age (hop distance) as tie-breaker when priorities
        are equal (non-compact only).
    label_ctype : str
        C type for label arrays.

    Returns
    -------
    kernel : cupy.RawKernel
        Compiled CUDA kernel
    """
    L = label_ctype
    dim_names, dim_params, size_expr, coord_code = _generate_coord_code(ndim)
    neighbors = _get_neighbor_offsets(ndim, connectivity)
    neighbor_info = _generate_neighbor_nidx(ndim, dim_names, neighbors)

    # --- Age tie-breaking (non-compact only) ---
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

    # --- Source coordinate tracking (compact only) ---
    if compact:
        src_names = [f"source_{j}" for j in range(ndim)]
        src_params = (
            ", ".join(f"int* __restrict__ {s}" for s in src_names) + ","
        )
        best_src_decls = "\n    ".join(
            f"int best_src_{j} = -1;" for j in range(ndim)
        )
        final_src_updates = "\n            ".join(
            f"{s}[idx] = best_src_{j};" for j, s in enumerate(src_names)
        )
    else:
        src_params = ""
        best_src_decls = ""
        final_src_updates = ""

    # --- Per-neighbor code ---
    neighbor_code = ""
    for nc_decl_str, bounds_str, nidx_expr in neighbor_info:
        if compact:
            src_reads = "\n                ".join(
                f"int nsrc_{j} = {s}[nidx];" for j, s in enumerate(src_names)
            )
            dist_decls = "\n                ".join(
                f"float d_{j} = (float)(c_{j} - nsrc_{j});" for j in range(ndim)
            )
            dist_terms = " + ".join(f"d_{j} * d_{j}" for j in range(ndim))
            best_src_updates = "\n                    ".join(
                f"best_src_{j} = nsrc_{j};" for j in range(ndim)
            )
            priority_code = f"""
                {src_reads}

                {dist_decls}
                float euclidean_dist = sqrtf({dist_terms});

                float new_priority = image[idx] + compactness * euclidean_dist;

                // Enforce monotonicity
                float neighbor_priority = priority[nidx];
                if (new_priority < neighbor_priority) {{
                    new_priority = neighbor_priority;
                }}

                if (new_priority < best_priority) {{
                    best_priority = new_priority;
                    best_label = nlabel;
                    {best_src_updates}
                    found_label = 1;
                }}"""
        else:
            priority_code = f"""
                float new_priority = image[idx];
                // Enforce monotonicity
                if (new_priority < npriority) {{
                    new_priority = npriority;
                }}

                {age_new}

                {better_path_cmp} {{
                    best_priority = new_priority;
                    {age_update}
                    best_label = nlabel;
                    found_label = 1;
                }}"""

        npriority_read = "" if compact else "float npriority = priority[nidx];"

        neighbor_code += f"""
    {{
        {nc_decl_str}
        if ({bounds_str}) {{
            int nidx = {nidx_expr};
            {L} nlabel = labels[nidx];
            {npriority_read}
            {age_read}

            if (nlabel != 0) {{
                {priority_code}
            }}
        }}
    }}
"""

    # --- Kernel params and final update ---
    age_param = "int* __restrict__ age," if use_age else ""
    age_best_init = "int best_age = age[idx];" if use_age else ""
    compact_param = "float compactness," if compact else ""

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

    kernel_code = (
        _KERNEL_PREAMBLE
        + f"""
extern "C" __global__
void watershed_step(
    const float* __restrict__ image,
    {L}* __restrict__ labels,
    unsigned char* __restrict__ state,
    float* __restrict__ priority,
    {age_param}
    {src_params}
    int* __restrict__ changed,
    {compact_param}
    {dim_params}
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = {size_expr};
    if (idx >= size) return;

    {coord_code}

    if (state[idx] != 2) return;  // Only process UNLABELED pixels

    float best_priority = priority[idx];
    {age_best_init}
    {L} best_label = 0;
    {best_src_decls}
    int found_label = 0;

    {neighbor_code}

    if (found_label && best_label != 0) {{
        {better_path_final_cmp} {{
            labels[idx] = best_label;
            priority[idx] = best_priority;
            {age_final_update}
            {final_src_updates}

            atomicAdd(changed, 1);
        }}
    }}
}}
"""
    )

    return cp.RawKernel(kernel_code, "watershed_step")


def _watershed_synchronous(
    image_flat,
    markers,
    mask_flat,
    has_mask,
    connectivity,
    ndim,
    image_shape,
    size,
    blocks,
    threads_per_block,
    max_iterations,
    compactness=0,
    use_age=False,
    label_dtype=cp.int32,
    **kwargs,
):
    """Run synchronous watershed algorithm (standard or compact).

    When compactness=0, uses image intensity as priority. When
    compactness>0, adds a Euclidean distance penalty from the source
    marker to produce more regularly-shaped basins.
    """
    compact = compactness > 0
    label_ctype = _DTYPE_TO_CTYPE[cp.dtype(label_dtype)]

    labels = cp.zeros(size, dtype=label_dtype)
    state = cp.zeros(size, dtype=cp.uint8)
    priority = cp.zeros(size, dtype=cp.float32)
    changed = cp.zeros(1, dtype=cp.int32)

    # Age arrays (non-compact only)
    if use_age and not compact:
        age = cp.zeros(size, dtype=cp.int32)
    else:
        age = None

    # Source coordinate arrays (compact only)
    if compact:
        sources = [cp.zeros(size, dtype=cp.int32) for _ in range(ndim)]
    else:
        sources = []

    dim_args = tuple(int(s) for s in image_shape)

    # --- Initialization ---
    init_kernel = _get_watershed_init_kernel(
        ndim=ndim,
        compact=compact,
        use_age=age is not None,
        label_ctype=label_ctype,
    )

    init_args = []
    if not compact:
        init_args.append(image_flat)
    init_args.extend([markers.ravel(), labels, state, priority])
    if age is not None:
        init_args.append(age)
    init_args.extend(sources)
    if compact:
        init_args.extend([mask_flat, has_mask, *dim_args])
    else:
        init_args.extend([mask_flat, has_mask, int(size)])

    init_kernel(
        (blocks,),
        (threads_per_block,),
        tuple(init_args),
    )

    # --- Iteration ---
    step_kernel = _get_watershed_step_kernel(
        ndim,
        connectivity,
        compact=compact,
        use_age=age is not None,
        label_ctype=label_ctype,
    )

    step_args = [image_flat, labels, state, priority]
    if age is not None:
        step_args.append(age)
    step_args.extend(sources)
    step_args.append(changed)
    if compact:
        step_args.append(cp.float32(compactness))
    step_args.extend(dim_args)
    step_args = tuple(step_args)

    for iteration in range(max_iterations):
        changed[0] = 0
        step_kernel((blocks,), (threads_per_block,), step_args)
        if changed[0] == 0:
            break

    return labels, state, priority, changed


######################################################
# Block-asynchronous 2D Non-compact seeded watershed #
######################################################


# Block-asynchronous watershed kernel constants
TILE_W = 32  # Tile width (must match block size)
TILE_H = 32  # Tile height (must match block size)
HALO = 1  # Halo size (1 pixel for immediate neighbors)


@cp.memoize(for_each_device=True)
def _get_watershed_block_async_kernel_2d(
    connectivity=1,
    inner_iterations=16,
    use_age=False,
    label_ctype="cuda::std::int32_t",
):
    """Get block-asynchronous iteration kernel for 2D CA-watershed.

    This kernel uses shared memory tiling to reduce global memory traffic.
    Each block loads a tile + halo into shared memory and performs multiple
    local iterations before writing back to global memory, reducing the
    number of global synchronization rounds.

    Based on the "Block-asynchronous algorithm" described in Section 4.3
    of Kauffmann, C. & Piche, N. (2010), "Cellular automaton for
    ultra-fast watershed transform on GPU", ICPR 2010, pp. 447-450.

    Parameters
    ----------
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity
    inner_iterations : int
        Number of iterations to perform within each block before
        synchronizing with global memory.
    use_age : bool
        If True, include age array for tie-breaking.
    label_ctype : str
        C type for label arrays.

    Returns
    -------
    kernel : cupy.RawKernel
        Compiled CUDA kernel
    """
    L = label_ctype
    neighbors = _get_neighbor_offsets(2, connectivity)

    # Shared memory dimensions (tile + halo on each side)
    shared_w = TILE_W + 2 * HALO
    shared_h = TILE_H + 2 * HALO

    # Conditional age code fragments
    age_shared = (
        f"__shared__ int s_age[{shared_w} * {shared_h}];" if use_age else ""
    )
    age_param = "int* __restrict__ age," if use_age else ""

    def _load_block(idx_expr, sidx_expr, in_bounds):
        """Generate load code for one tile region (main, halo, or corner)."""
        if in_bounds:
            age_load = (
                f"s_age[{sidx_expr}] = age[{idx_expr}];" if use_age else ""
            )
            age_default = ""
        else:
            age_load = ""
            age_default = f"s_age[{sidx_expr}] = 2147483647;" if use_age else ""
        load = f"""
        s_labels[{sidx_expr}] = labels[{idx_expr}];
        s_state[{sidx_expr}] = state[{idx_expr}];
        s_priority[{sidx_expr}] = priority[{idx_expr}];
        s_image[{sidx_expr}] = image[{idx_expr}];
        {age_load}"""
        default = f"""
        s_labels[{sidx_expr}] = 0;
        s_state[{sidx_expr}] = 0;
        s_priority[{sidx_expr}] = 0.0f;
        s_image[{sidx_expr}] = 0.0f;
        {age_default}"""
        return load.rstrip(), default.rstrip()

    # Generate the 9 load sections (main tile + 4 edges + 4 corners)
    # Each section: condition, hx/hy expressions, sidx expression
    load_sections = []

    # Main tile
    main_load, main_default = _load_block("gidx", "sidx", True)
    load_sections.append(f"""
    // Load main tile
    if (gx < width && gy < height) {{
        int gidx = gy * width + gx;{main_load}
    }} else {{{main_default}
    }}""")

    # Helper for halo sections
    halo_defs = [
        (
            "Left",
            "tx == 0",
            f"gx - {HALO}",
            "gy",
            f"sy * {shared_w} + (sx - {HALO})",
        ),
        (
            "Right",
            f"tx == {TILE_W - 1}",
            f"gx + {HALO}",
            "gy",
            f"sy * {shared_w} + (sx + {HALO})",
        ),
        (
            "Top",
            "ty == 0",
            "gx",
            f"gy - {HALO}",
            f"(sy - {HALO}) * {shared_w} + sx",
        ),
        (
            "Bottom",
            f"ty == {TILE_H - 1}",
            "gx",
            f"gy + {HALO}",
            f"(sy + {HALO}) * {shared_w} + sx",
        ),
        (
            "Top-left",
            "tx == 0 && ty == 0",
            f"gx - {HALO}",
            f"gy - {HALO}",
            f"(sy - {HALO}) * {shared_w} + (sx - {HALO})",
        ),
        (
            "Top-right",
            f"tx == {TILE_W - 1} && ty == 0",
            f"gx + {HALO}",
            f"gy - {HALO}",
            f"(sy - {HALO}) * {shared_w} + (sx + {HALO})",
        ),
        (
            "Bottom-left",
            f"tx == 0 && ty == {TILE_H - 1}",
            f"gx - {HALO}",
            f"gy + {HALO}",
            f"(sy + {HALO}) * {shared_w} + (sx - {HALO})",
        ),
        (
            "Bottom-right",
            f"tx == {TILE_W - 1} && ty == {TILE_H - 1}",
            f"gx + {HALO}",
            f"gy + {HALO}",
            f"(sy + {HALO}) * {shared_w} + (sx + {HALO})",
        ),
    ]  # noqa: E501

    for name, cond, hx_expr, hy_expr, hsidx_expr in halo_defs:
        hload, hdefault = _load_block("hidx", "hsidx", True)
        load_sections.append(f"""
    // {name} halo
    if ({cond}) {{
        int hx = {hx_expr};
        int hy = {hy_expr};
        int hsidx = {hsidx_expr};
        if (hx >= 0 && hx < width && hy >= 0 && hy < height) {{
            int hidx = hy * width + hx;{hload}
        }} else {{{hdefault.replace("s_age[sidx]", "s_age[hsidx]")}
        }}
    }}""")

    all_loads = "\n".join(load_sections)

    # Generate neighbor checking code for shared memory
    if use_age:
        age_read_n = "int nage = s_age[nsidx];"
        age_new_n = "int new_age = nage + 1;"
        better_cmp = (
            "if (new_priority < best_priority ||\n"
            "                        (new_priority == best_priority "
            "&& new_age < best_age))"
        )
        age_update_n = "best_age = new_age;"
    else:
        age_read_n = ""
        age_new_n = ""
        better_cmp = "if (new_priority < best_priority)"
        age_update_n = ""

    neighbor_code = ""
    for i, (dy, dx) in enumerate(neighbors):
        neighbor_code += f"""
            // Neighbor {i}: dy={dy}, dx={dx}
            {{
                int nsx = sx + ({dx});
                int nsy = sy + ({dy});
                int nsidx = nsy * {shared_w} + nsx;

                {L} nlabel = s_labels[nsidx];
                float npriority = s_priority[nsidx];
                {age_read_n}

                if (nlabel != 0) {{
                    float new_priority = my_image;
                    if (new_priority < npriority) {{
                        new_priority = npriority;
                    }}

                    {age_new_n}

                    {better_cmp} {{
                        best_priority = new_priority;
                        {age_update_n}
                        best_label = nlabel;
                        found_label = 1;
                    }}
                }}
            }}
"""

    # Phase 2: inner loop age handling
    age_best_init = "int best_age = s_age[sidx];" if use_age else ""
    if use_age:
        inner_cmp = (
            "if (found_label && best_label != 0 && "
            "(best_priority < s_priority[sidx] || "
            "(best_priority == s_priority[sidx] && best_age < s_age[sidx])))"
        )
        age_inner_update = "s_age[sidx] = best_age;"
    else:
        inner_cmp = "if (found_label && best_label != 0 && best_priority < s_priority[sidx])"
        age_inner_update = ""

    # Phase 3: write back
    age_writeback = "age[gidx] = s_age[sidx];" if use_age else ""

    kernel_code = (
        _KERNEL_PREAMBLE
        + f"""
extern "C" __global__
void watershed_block_async_2d(
    const float* __restrict__ image,
    {L}* __restrict__ labels,
    unsigned char* __restrict__ state,
    float* __restrict__ priority,
    {age_param}
    int* __restrict__ global_changed,
    int width,
    int height
) {{
    __shared__ {L} s_labels[{shared_w} * {shared_h}];
    __shared__ unsigned char s_state[{shared_w} * {shared_h}];
    __shared__ float s_priority[{shared_w} * {shared_h}];
    __shared__ float s_image[{shared_w} * {shared_h}];
    {age_shared}
    __shared__ int s_changed;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * {TILE_W} + tx;
    int gy = blockIdx.y * {TILE_H} + ty;
    int sx = tx + {HALO};
    int sy = ty + {HALO};
    int sidx = sy * {shared_w} + sx;

    // ========================================
    // Phase 1: Cooperative load of tile + halo
    // ========================================
    {all_loads}

    __syncthreads();

    // ========================================
    // Phase 2: Inner iterations in shared memory
    // ========================================

    float my_image = s_image[sidx];
    int my_changed = 0;

    for (int iter = 0; iter < {inner_iterations}; iter++) {{
        if (s_state[sidx] == 2) {{  // UNLABELED
            float best_priority = s_priority[sidx];
            {age_best_init}
            {L} best_label = 0;
            int found_label = 0;

            {neighbor_code}

            {inner_cmp} {{
                s_labels[sidx] = best_label;
                s_priority[sidx] = best_priority;
                {age_inner_update}
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
        {age_writeback}
    }}

    if (tx == 0 && ty == 0) {{
        s_changed = 0;
    }}
    __syncthreads();

    if (my_changed) {{
        atomicAdd(&s_changed, 1);
    }}
    __syncthreads();

    if (tx == 0 && ty == 0 && s_changed > 0) {{
        atomicAdd(global_changed, s_changed);
    }}
}}
"""
    )  # noqa: E501

    return cp.RawKernel(kernel_code, "watershed_block_async_2d")


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
    use_age=False,
    label_dtype=cp.int32,
    **kwargs,
):
    """Run standard watershed with block-asynchronous algorithm.

    Uses shared memory tiling to reduce global memory traffic.
    Each block performs multiple iterations locally before synchronizing.
    """
    label_ctype = _DTYPE_TO_CTYPE[cp.dtype(label_dtype)]

    labels = cp.zeros(size, dtype=label_dtype)
    state = cp.zeros(size, dtype=cp.uint8)
    priority = cp.zeros(size, dtype=cp.float32)
    changed = cp.zeros(1, dtype=cp.int32)

    if use_age:
        age = cp.zeros(size, dtype=cp.int32)
    else:
        age = None

    init_kernel = _get_watershed_init_kernel(
        use_age=use_age, label_ctype=label_ctype
    )

    init_args = [image_flat, markers.ravel(), labels, state, priority]
    if use_age:
        init_args.append(age)
    init_args.extend([mask_flat, has_mask, int(size)])

    init_kernel(
        (blocks,),
        (threads_per_block,),
        tuple(init_args),
    )

    step_kernel = _get_watershed_block_async_kernel_2d(
        connectivity,
        inner_iterations,
        use_age=use_age,
        label_ctype=label_ctype,
    )

    block_size = (TILE_W, TILE_H)
    grid_size = (
        (width + TILE_W - 1) // TILE_W,
        (height + TILE_H - 1) // TILE_H,
    )

    step_args = [image_flat, labels, state, priority]
    if use_age:
        step_args.append(age)
    step_args.extend([changed, int(width), int(height)])
    step_args = tuple(step_args)

    max_outer_iterations = (
        max_iterations + inner_iterations - 1
    ) // inner_iterations

    for iteration in range(max_outer_iterations):
        changed[0] = 0
        step_kernel(grid_size, block_size, step_args)
        if changed[0] == 0:
            break

    return labels, state, priority, changed


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
    tie-breaker.

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
    {L}* __restrict__ labels_out,
    {dim_params}
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = {size_expr};
    if (idx >= size) return;

    {coord_code}

    {L} my_label = labels_in[idx];
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
    n_pixels = image.size
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
        n_pixels = int(cp.sum(mask))

    # Handle markers
    if markers is None:
        # Auto-detect markers from local minima (matching scikit-image)
        markers_bool = local_minima(image, connectivity=connectivity)
        if mask is not None:
            markers_bool = markers_bool * mask.astype(bool)
        footprint = ndi.generate_binary_structure(ndim, connectivity)
        markers = ndi.label(markers_bool, structure=footprint)[0]
    elif not isinstance(markers, (cp.ndarray, list, tuple)):
        # Assume int: generate that many regularly-spaced markers
        n_markers = int(markers)
        # Scale n_markers by fraction of masked pixels (like scikit-image)
        markers = regular_seeds(
            image.shape, int(n_markers / (n_pixels / image.size))
        )
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
        standard watershed (compactness=0).
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
    # Determine output label dtype from markers
    if isinstance(markers, cp.ndarray):
        label_dtype = cp.dtype(markers.dtype)
    else:
        label_dtype = cp.dtype(cp.int32)

    # Ensure it's a supported integer type for kernel codegen
    if label_dtype not in _DTYPE_TO_CTYPE:
        label_dtype = cp.dtype(cp.int32)

    # Validate and prepare inputs
    image, markers, mask, connectivity = _validate_inputs(
        image, markers, mask, connectivity
    )

    # Ensure markers match the label dtype
    if markers.dtype != label_dtype:
        markers = markers.astype(label_dtype)

    ndim = image.ndim

    # Get dimensions
    size = image.size
    # width/height needed for block-async and compact (2D only)
    if ndim >= 2:
        height, width = image.shape[-2], image.shape[-1]
    else:
        height = width = None

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

    # Upper bound on iterations (shouldn't need this many)
    max_iterations = max(image.shape) * 2

    # Ensure image is contiguous and flat for kernel
    image_flat = cp.ascontiguousarray(image.ravel())

    import warnings

    # Use different code paths for standard vs compact watershed
    if compactness == 0:
        # Determine whether to use block-async algorithm
        # (2D non-compact only; supports age)
        _min_block_async_size = max(TILE_W, TILE_H)
        if use_block_async is None:
            use_block_async = (
                ndim == 2 and min(height, width) >= _min_block_async_size
            )
        elif (
            use_block_async
            and ndim == 2
            and (min(height, width) < _min_block_async_size)
        ):
            warnings.warn(
                f"use_block_async=True requires image dimensions >= "
                f"{_min_block_async_size}; falling back to synchronous. "
                f"Got image shape {image.shape}.",
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
    )

    if use_block_async and ndim == 2 and compactness == 0:
        labels, state, priority, changed = _watershed_standard_block_async(
            **common_kwargs,
            width=width,
            height=height,
            inner_iterations=16,
            use_age=use_age,
        )
    else:
        labels, state, priority, changed = _watershed_synchronous(
            **common_kwargs,
            compactness=compactness,
            ndim=ndim,
            image_shape=image.shape,
            use_age=use_age,
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
            (labels, priority, labels_out, *dim_args),
        )
        labels = labels_out

    return labels.reshape(image.shape)
