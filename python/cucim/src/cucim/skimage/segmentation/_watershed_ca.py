# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watershed segmentation using a cellular automaton algorithm.

This module implements a synchronous CA-watershed algorithm based on:

Kauffmann, C., & Piche, N. (2010). Cellular automaton for ultra-fast
watershed transform on GPU. In Pattern Recognition (ICPR), 2010 20th
International Conference on (pp. 447-450). IEEE.

The compact watershed extension follows the approach from scikit-image,
which is based on:

Neubert, P., & Protzel, P. (2014). Compact Watershed and Preemptive SLIC:
On Improving Trade-offs of Superpixel Segmentation Algorithms.
In Pattern Recognition (ICPR), 2014 22nd International Conference on.

The implementation here is a general n-dimensional version of the algorithms.

See `_watershed_ca_block_async.py` for a faster block-asynchronous variant
supporting compactness=0 for 2D and 3D data only.
"""

import itertools

import cupy as cp

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
