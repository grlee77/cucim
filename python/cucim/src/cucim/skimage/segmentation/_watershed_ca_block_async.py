# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block-asynchronous watershed using cellular-automaton-style relaxation.

The `_watershed_ca.py` module implements the base GPU relaxation. The
block-asynchronous variant here is inspired by:

P. Quesada-Barriuso, D.B. Heras, F. Argüello, Efficient 2D and 3D watershed on
graphics processing unit: block-asynchronous approaches based on cellular
automata, Computers & Electrical Engineering, Volume 39, Issue 8, 2013,
pp. 2638-2655, ISSN 0045-7906,
:DOI:`10.1016/j.compeleceng.2013.04.020`

The structure follows the plain block-asynchronous approach from Section 4.2:
each block loads a tile plus halo into shared memory, performs several local
CA-like relaxation iterations, writes the tile back to global memory, and
converges through repeated outer kernel launches. This is not a literal
implementation of the paper's hill-climbing plateau automaton, and it does not
include the artifact-free distance-correction scheme from Section 4.3.

The implementation is adapted to cuCIM/scikit-image semantics: marker labels
are propagated by path priority, optional age values break equal-priority
plateau ties, and 2D/3D connectivities follow ndimage conventions. Compact
watershed is handled only by the synchronous/global-relaxation path.

"""  # noqa: E501

import cupy as cp

from ._watershed_ca import (
    _DTYPE_TO_CTYPE,
    _KERNEL_PREAMBLE,
    _get_neighbor_offsets,
    _get_watershed_init_kernel,
)

# Block-asynchronous watershed kernel constants
TILE_W = 32  # 2D tile width (must match block size)
TILE_H = 32  # 2D tile height (must match block size)
TILE_3D = 8  # 3D tile size per dimension (8x8x8 = 512 threads)
HALO = 1  # Halo size (1 pixel for immediate neighbors)


class _TiledArray:
    """Descriptor for an array that participates in shared-memory tiling.

    Parameters
    ----------
    gname : str
        Global memory variable name in the kernel.
    sname : str
        Shared memory variable name.
    ctype : str
        C type string (e.g. "float", "int").
    default : str
        C literal for out-of-bounds default value.
    readonly : bool
        If True, this array is not written back in Phase 3.
    """

    __slots__ = ("gname", "sname", "ctype", "default", "readonly")

    def __init__(self, gname, sname, ctype, default, readonly=False):
        self.gname = gname
        self.sname = sname
        self.ctype = ctype
        self.default = default
        self.readonly = readonly


def _gen_tiled_load(arrays, sidx, gidx):
    """Generate load (in-bounds) and default (out-of-bounds) CUDA code
    for all tiled arrays at a given shared/global index expression."""
    load = "\n        ".join(
        f"{a.sname}[{sidx}] = {a.gname}[{gidx}];" for a in arrays
    )
    default = "\n        ".join(
        f"{a.sname}[{sidx}] = {a.default};" for a in arrays
    )
    return load, default


def _get_tiled_arrays(label_ctype, use_age):
    """Return descriptors for arrays participating in shared-memory tiling."""
    arrays = [
        _TiledArray("labels", "s_labels", label_ctype, "0"),
        _TiledArray("state", "s_state", "unsigned char", "0"),
        _TiledArray("priority", "s_priority", "float", "0.0f"),
        _TiledArray("image", "s_image", "float", "0.0f", readonly=True),
    ]
    if use_age:
        arrays.append(_TiledArray("age", "s_age", "int", "2147483647"))
    return arrays


def _gen_tiled_kernel_fragments(arrays, shared_size):
    """Generate declaration, parameter, and writeback fragments."""
    shared_decls = "\n    ".join(
        f"__shared__ {a.ctype} {a.sname}[{shared_size}];" for a in arrays
    )
    kernel_params = "\n    ".join(
        f"{'const ' if a.readonly else ''}{a.ctype}* __restrict__ {a.gname},"
        for a in arrays
    )
    writeback_code = "\n        ".join(
        f"{a.gname}[gidx] = {a.sname}[sidx];" for a in arrays if not a.readonly
    )
    return shared_decls, kernel_params, writeback_code


def _gen_cooperative_load_fragments(arrays):
    """Generate in-bounds/default fragments for cooperative 1D halo loads."""
    coop_load = "\n            ".join(
        f"{a.sname}[s] = {a.gname}[gidx];" for a in arrays
    )
    coop_default = "\n            ".join(
        f"{a.sname}[s] = {a.default};" for a in arrays
    )
    return coop_load, coop_default


def _gen_best_state_init_code(label_ctype, use_age):
    """Generate per-pixel best-state initialization."""
    if use_age:
        return (
            "float best_priority = s_priority[sidx];\n"
            "            int best_age = s_age[sidx];\n"
            f"            {label_ctype} best_label = 0;\n"
            "            int found_label = 0;"
        )

    return (
        "float best_priority = s_priority[sidx];\n"
        "            \n"
        f"            {label_ctype} best_label = 0;\n"
        "            int found_label = 0;"
    )


def _gen_inner_update_code(use_age):
    """Generate the shared-memory update after neighbor inspection."""
    if use_age:
        inner_cmp = (
            "if (found_label && best_label != 0 && "
            "(best_priority < s_priority[sidx] || "
            "(best_priority == s_priority[sidx] "
            "&& best_age < s_age[sidx])))"
        )
        age_inner_update = "s_age[sidx] = best_age;"
    else:
        inner_cmp = (
            "if (found_label && best_label != 0 "
            "&& best_priority < s_priority[sidx])"
        )
        age_inner_update = ""

    return f"""{inner_cmp} {{
                s_labels[sidx] = best_label;
                s_priority[sidx] = best_priority;
                {age_inner_update}
                my_changed = 1;
            }}"""


def _gen_neighbor_check_code(label_ctype, neighbor_index_code, use_age):
    """Generate the common neighbor-inspection body."""
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
    for nsidx_code in neighbor_index_code:
        neighbor_code += f"""
            {{
                {nsidx_code}

                {label_ctype} nlabel = s_labels[nsidx];
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
    return neighbor_code


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

    Based on the plain block-asynchronous approach described in Section 4.2
    of Quesada-Barriuso, Heras & Argüello (2013), "Efficient 2D and 3D
    watershed on graphics processing unit: block-asynchronous approaches
    based on cellular automata". This is not the artifact-free
    distance-correction variant from Section 4.3 of that paper.

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
    neighbors = _get_neighbor_offsets(2, connectivity)

    shared_w = TILE_W + 2 * HALO
    shared_h = TILE_H + 2 * HALO
    shared_size = f"{shared_w} * {shared_h}"

    arrays = _get_tiled_arrays(label_ctype, use_age)
    shared_decls, kernel_params, writeback_code = _gen_tiled_kernel_fragments(
        arrays, shared_size
    )

    # --- Phase 1: Tile + halo loading ---
    load_sections = []

    # Main tile
    main_load, main_default = _gen_tiled_load(arrays, "sidx", "gidx")
    load_sections.append(f"""
    // Load main tile
    if (gx < width && gy < height) {{
        int gidx = gy * width + gx;
        {main_load}
    }} else {{
        {main_default}
    }}""")

    # 2D halo regions: 4 edges + 4 corners
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
        hload, hdefault = _gen_tiled_load(arrays, "hsidx", "hidx")
        load_sections.append(f"""
    // {name} halo
    if ({cond}) {{
        int hx = {hx_expr};
        int hy = {hy_expr};
        int hsidx = {hsidx_expr};
        if (hx >= 0 && hx < width && hy >= 0 && hy < height) {{
            int hidx = hy * width + hx;
            {hload}
        }} else {{
            {hdefault}
        }}
    }}""")

    all_loads = "\n".join(load_sections)

    best_state_init_code = _gen_best_state_init_code(label_ctype, use_age)
    inner_update_code = _gen_inner_update_code(use_age)
    neighbor_index_code = [
        (
            f"int nsx = sx + ({dx});\n"
            f"                int nsy = sy + ({dy});\n"
            f"                int nsidx = nsy * {shared_w} + nsx;"
        )
        for dy, dx in neighbors
    ]
    neighbor_code = _gen_neighbor_check_code(
        label_ctype, neighbor_index_code, use_age
    )

    # --- Assemble kernel ---
    kernel_code = (
        _KERNEL_PREAMBLE
        + f"""
extern "C" __global__
void watershed_block_async_2d(
    {kernel_params}
    int* __restrict__ global_changed,
    int width,
    int height
) {{
    {shared_decls}
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
            {best_state_init_code}

            {neighbor_code}

            {inner_update_code}
        }}

        __syncthreads();
    }}

    // ========================================
    // Phase 3: Write back to global memory
    // ========================================

    if (gx < width && gy < height) {{
        int gidx = gy * width + gx;
        {writeback_code}
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


@cp.memoize(for_each_device=True)
def _get_watershed_block_async_kernel_3d(
    connectivity=1,
    inner_iterations=8,
    use_age=False,
    label_ctype="cuda::std::int32_t",
):
    """Get block-asynchronous iteration kernel for 3D CA-watershed.

    Uses cooperative 1D halo loading: all threads in the block
    collaboratively load the entire shared memory tile (main + halo)
    via a linear sweep. This avoids enumerating 26 separate halo
    regions and generalizes cleanly to 3D.

    Based on the plain block-asynchronous approach (Section 4.2) of
    Quesada-Barriuso, Heras & Argüello (2013); not the artifact-free
    distance-correction variant from Section 4.3 of that paper.

    Parameters
    ----------
    connectivity : int
        1 (6-conn), 2 (18-conn), or 3 (26-conn).
    inner_iterations : int
        Number of local iterations before global sync.
    use_age : bool
        If True, include age array for tie-breaking.
    label_ctype : str
        C type for label arrays.
    """
    neighbors = _get_neighbor_offsets(3, connectivity)

    T = TILE_3D
    H = HALO
    shared_x = T + 2 * H
    shared_y = T + 2 * H
    shared_z = T + 2 * H
    shared_total = shared_x * shared_y * shared_z
    num_threads = T * T * T

    arrays = _get_tiled_arrays(label_ctype, use_age)
    shared_decls, kernel_params, writeback_code = _gen_tiled_kernel_fragments(
        arrays, shared_total
    )

    # --- Phase 1: Cooperative 1D halo loading ---
    coop_load, coop_default = _gen_cooperative_load_fragments(arrays)

    best_state_init_code = _gen_best_state_init_code(label_ctype, use_age)
    inner_update_code = _gen_inner_update_code(use_age)
    neighbor_index_code = [
        (
            f"int nsidx = (sz + ({dz})) * {shared_y * shared_x}\n"
            f"                          + (sy + ({dy})) * {shared_x}\n"
            f"                          + (sx + ({dx}));"
        )
        for dz, dy, dx in neighbors
    ]
    neighbor_code = _gen_neighbor_check_code(
        label_ctype, neighbor_index_code, use_age
    )

    kernel_code = (
        _KERNEL_PREAMBLE
        + f"""
extern "C" __global__
void watershed_block_async_3d(
    {kernel_params}
    int* __restrict__ global_changed,
    int dim_0,  // depth (slowest)
    int dim_1,  // height
    int dim_2   // width (fastest)
) {{
    {shared_decls}
    __shared__ int s_changed;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = (tz * {T} + ty) * {T} + tx;

    // Global coordinates of this thread's main-tile voxel
    int gx = blockIdx.x * {T} + tx;
    int gy = blockIdx.y * {T} + ty;
    int gz = blockIdx.z * {T} + tz;

    // Shared memory coordinates (offset by halo)
    int sx = tx + {H};
    int sy = ty + {H};
    int sz = tz + {H};
    int sidx = (sz * {shared_y} + sy) * {shared_x} + sx;

    // ========================================
    // Phase 1: Cooperative load of tile + halo
    // ========================================
    // All threads cooperatively load all {shared_total} shared elements
    // via a 1D linear sweep. This cleanly handles the 26 halo regions
    // without enumerating each one.

    for (int s = tid; s < {shared_total}; s += {num_threads}) {{
        // Convert linear shared index to 3D shared coords
        int s_sz = s / {shared_y * shared_x};
        int s_sy = (s / {shared_x}) % {shared_y};
        int s_sx = s % {shared_x};

        // Corresponding global coords (subtract halo, add block offset)
        int g_x = blockIdx.x * {T} + s_sx - {H};
        int g_y = blockIdx.y * {T} + s_sy - {H};
        int g_z = blockIdx.z * {T} + s_sz - {H};

        if (g_x >= 0 && g_x < dim_2 &&
            g_y >= 0 && g_y < dim_1 &&
            g_z >= 0 && g_z < dim_0) {{
            int gidx = (g_z * dim_1 + g_y) * dim_2 + g_x;
            {coop_load}
        }} else {{
            {coop_default}
        }}
    }}

    __syncthreads();

    // ========================================
    // Phase 2: Inner iterations in shared memory
    // ========================================

    float my_image = s_image[sidx];
    int my_changed = 0;

    for (int iter = 0; iter < {inner_iterations}; iter++) {{
        if (s_state[sidx] == 2) {{  // UNLABELED
            {best_state_init_code}

            {neighbor_code}

            {inner_update_code}
        }}

        __syncthreads();
    }}

    // ========================================
    // Phase 3: Write back to global memory
    // ========================================

    if (gx < dim_2 && gy < dim_1 && gz < dim_0) {{
        int gidx = (gz * dim_1 + gy) * dim_2 + gx;
        {writeback_code}
    }}

    if (tid == 0) {{
        s_changed = 0;
    }}
    __syncthreads();

    if (my_changed) {{
        atomicAdd(&s_changed, 1);
    }}
    __syncthreads();

    if (tid == 0 && s_changed > 0) {{
        atomicAdd(global_changed, s_changed);
    }}
}}
"""
    )

    return cp.RawKernel(kernel_code, "watershed_block_async_3d")


def _watershed_standard_block_async(
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
    inner_iterations=None,
    use_age=False,
    label_dtype=cp.int32,
    **kwargs,
):
    """Run standard watershed with block-asynchronous algorithm.

    Uses shared memory tiling to reduce global memory traffic.
    Each block performs multiple iterations locally before synchronizing.
    Supports 2D and 3D images.
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

    if ndim == 2:
        if inner_iterations is None:
            inner_iterations = 16
        step_kernel = _get_watershed_block_async_kernel_2d(
            connectivity,
            inner_iterations,
            use_age=use_age,
            label_ctype=label_ctype,
        )
        height, width = image_shape
        block_size = (TILE_W, TILE_H)
        grid_size = (
            (width + TILE_W - 1) // TILE_W,
            (height + TILE_H - 1) // TILE_H,
        )
        # Args match kernel param order from arrays table:
        # labels, state, priority, image, [age], global_changed, width, height
        step_args = [labels, state, priority, image_flat]
        if use_age:
            step_args.append(age)
        step_args.extend([changed, int(width), int(height)])

    elif ndim == 3:
        if inner_iterations is None:
            inner_iterations = 8
        step_kernel = _get_watershed_block_async_kernel_3d(
            connectivity,
            inner_iterations,
            use_age=use_age,
            label_ctype=label_ctype,
        )
        T = TILE_3D
        depth, height, width = image_shape
        block_size = (T, T, T)
        grid_size = (
            (width + T - 1) // T,
            (height + T - 1) // T,
            (depth + T - 1) // T,
        )
        # Args match 3D kernel param order:
        # labels, state, priority, image, [age],
        # global_changed, dim_0, dim_1, dim_2
        step_args = [labels, state, priority, image_flat]
        if use_age:
            step_args.append(age)
        step_args.extend([changed, int(depth), int(height), int(width)])
    else:
        raise NotImplementedError(
            f"Block-async only supports 2D and 3D, got {ndim}D"
        )

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
