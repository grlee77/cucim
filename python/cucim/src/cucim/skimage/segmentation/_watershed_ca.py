# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watershed segmentation using cellular-automaton-style relaxation.

This module implements a GPU-parallel watershed relaxation inspired by
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

The implementation is not a literal reproduction of either paper's
double-buffered synchronous CA or hill-climbing plateau automaton. Instead, it
uses an in-place global-memory relaxation of labels and path priorities,
adapted to the scikit-image watershed API: arbitrary marker labels, masks,
n-dimensional connectivity, optional age-based plateau tie-breaking, and a
compact watershed mode.

The compact watershed extension follows the scikit-image approach, based on:

Neubert, P., & Protzel, P. (2014). Compact Watershed and Preemptive SLIC: On
Improving Trade-offs of Superpixel Segmentation Algorithms, 2014 22nd
International Conference on Pattern Recognition (ICPR), Stockholm, Sweden,
2014, pp. 996-1001
:DOI:`10.1109/ICPR.2014.181`

See `_watershed_ca_block_async.py` for a faster block-asynchronous variant
for standard 2D and 3D watershed. That variant follows the plain
block-asynchronous structure of Quesada-Barriuso et al.; it is not the
artifact-free distance-correction algorithm from that paper.
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
        If True, initialize source coordinate arrays. Marker priorities are
        initialized from image values in both modes.
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
        size_param = dim_params
        size_code = f"int size = {size_expr};"
        coord_extract = coord_code
    else:
        src_params = ""
        src_init_mask = ""
        src_init_marker = ""
        src_init_unlabeled = ""
        size_param = "int size"
        size_code = ""
        coord_extract = ""

    kernel_code = (
        _KERNEL_PREAMBLE
        + f"""
extern "C" __global__
void watershed_init(
    const float* image,
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
        priority[idx] = image[idx];
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
    label_ctype="cuda::std::int32_t",
):
    """Get iteration kernel for nD CA-watershed (standard or compact).

    Uses a 1D grid for all dimensionalities, recovering nD coordinates
    from the flat index for neighbor bounds checking.

    The synchronous path is always double-buffered (a synchronous-CA /
    Jacobi update): the kernel reads from input arrays
    (``labels_in``/``priority_in``/...) and writes to separate output
    arrays (``labels_out``/...), copying unchanged cells through, and the
    caller swaps the buffers each iteration. This propagates exactly one
    neighborhood step per launch and is deterministic (no read/write race),
    and it yields the closest-marker assignment on plateaus without needing
    an age tie-breaker.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    connectivity : int
        Neighborhood connectivity (1 to ndim).
    compact : bool
        If True, generate compact watershed kernel with Euclidean
        distance penalty and source coordinate tracking.
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

    # The relaxation is double-buffered: it reads from ``*_in`` arrays and
    # writes to distinct ``*_out`` arrays (so __restrict__ is sound and reads
    # only ever see the previous sweep). The caller swaps them each iteration.

    # --- Source coordinate tracking (compact only) ---
    if compact:
        src_names = [f"source_{j}" for j in range(ndim)]
        src_params = (
            ", ".join(
                f"const int* __restrict__ {s}_in, int* __restrict__ {s}_out"
                for s in src_names
            )
            + ","
        )
        src_r = [f"{s}_in" for s in src_names]
        src_w = [f"{s}_out" for s in src_names]
        best_src_decls = "\n    ".join(
            f"int best_src_{j} = -1;" for j in range(ndim)
        )
        final_src_updates = "\n            ".join(
            f"{w}[idx] = best_src_{j};" for j, w in enumerate(src_w)
        )
    else:
        src_params = ""
        src_r = []
        src_w = []
        best_src_decls = ""
        final_src_updates = ""

    # --- Copy-through code: every output cell must be written each sweep, so
    #     cells that do not update copy their input forward. ---
    copy_lines = [
        "labels_out[idx] = labels_in[idx];",
        "priority_out[idx] = priority_in[idx];",
    ]
    copy_lines.extend(f"{w}[idx] = {r}[idx];" for r, w in zip(src_r, src_w))
    copy_through = "\n        ".join(copy_lines)

    # --- Per-neighbor code ---
    neighbor_code = ""
    for nc_decl_str, bounds_str, nidx_expr in neighbor_info:
        if compact:
            src_reads = "\n                ".join(
                f"int nsrc_{j} = {r}[nidx];" for j, r in enumerate(src_r)
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
                float neighbor_priority = priority_in[nidx];
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
            priority_code = """
                float new_priority = image[idx];
                // Enforce monotonicity
                if (new_priority < npriority) {
                    new_priority = npriority;
                }

                if (new_priority < best_priority) {
                    best_priority = new_priority;
                    best_label = nlabel;
                    found_label = 1;
                }"""

        npriority_read = (
            "" if compact else "float npriority = priority_in[nidx];"
        )

        neighbor_code += f"""
    {{
        {nc_decl_str}
        if ({bounds_str}) {{
            int nidx = {nidx_expr};
            {L} nlabel = labels_in[nidx];
            {npriority_read}

            if (nlabel != 0) {{
                {priority_code}
            }}
        }}
    }}
"""

    # --- Kernel params and final update ---
    compact_param = "float compactness," if compact else ""

    kernel_code = (
        _KERNEL_PREAMBLE
        + f"""
extern "C" __global__
void watershed_step(
    const float* __restrict__ image,
    unsigned char* __restrict__ state,
    const {L}* __restrict__ labels_in,
    {L}* __restrict__ labels_out,
    const float* __restrict__ priority_in,
    float* __restrict__ priority_out,
    {src_params}
    int* __restrict__ changed,
    {compact_param}
    {dim_params}
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = {size_expr};
    if (idx >= size) return;

    {coord_code}

    if (state[idx] != 2) {{
        // Fixed (marker/masked) cell: copy through unchanged.
        {copy_through}
        return;
    }}

    float best_priority = priority_in[idx];
    {L} best_label = 0;
    {best_src_decls}
    int found_label = 0;

    {neighbor_code}

    if (found_label && best_label != 0) {{
        if (best_priority < priority_in[idx]) {{
            labels_out[idx] = best_label;
            priority_out[idx] = best_priority;
            {final_src_updates}

            atomicAdd(changed, 1);
            return;
        }}
    }}
    // No improvement this sweep: copy through unchanged.
    {copy_through}
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
    label_dtype=cp.int32,
    convergence_check_interval=None,
    **kwargs,
):
    """Run synchronous watershed algorithm (standard or compact).

    When compactness=0, uses image intensity as priority. When
    compactness>0, adds a Euclidean distance penalty from the source
    marker to produce more regularly-shaped basins.

    The relaxation is double-buffered (a synchronous CA / Jacobi update):
    it reads from one set of buffers and writes to a separate set, swapping
    them each iteration. This propagates labels by exactly one neighborhood
    step per iteration, is deterministic (no read/write race), and yields
    the closest-marker assignment on plateaus without an age tie-breaker.
    """
    compact = compactness > 0
    label_ctype = _DTYPE_TO_CTYPE[cp.dtype(label_dtype)]

    labels = cp.zeros(size, dtype=label_dtype)
    state = cp.zeros(size, dtype=cp.uint8)
    priority = cp.zeros(size, dtype=cp.float32)
    changed = cp.zeros(1, dtype=cp.int32)

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
        use_age=False,
        label_ctype=label_ctype,
    )

    init_args = [image_flat]
    init_args.extend([markers.ravel(), labels, state, priority])
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
        label_ctype=label_ctype,
    )

    # Per-cell state that is read and written each iteration, in the order
    # the kernel expects: labels, priority, [sources...]. The image, state,
    # and changed flag are shared (read-only or accumulate-only).
    read_bufs = [labels, priority]
    read_bufs.extend(sources)

    # Separate output buffers; every cell is overwritten each sweep (updated
    # or copied through), so they need no initialization.
    write_bufs = [cp.empty_like(b) for b in read_bufs]

    trailing = [changed]
    if compact:
        trailing.append(cp.float32(compactness))
    trailing.extend(dim_args)
    trailing = tuple(trailing)

    def make_args(rd, wr):
        # Interleave (in, out) for each buffer to match the kernel's paired
        # parameter order.
        paired = []
        for r, w in zip(rd, wr):
            paired.append(r)
            paired.append(w)
        return (image_flat, state, *paired, *trailing)

    # Run kernels in batches of ``convergence_check_interval`` launches and
    # only read ``changed`` back to the host once per batch. Reading the flag
    # forces a device->host sync, so batching reduces the number of syncs by
    # roughly this factor. ``changed`` is reset once per batch and accumulates
    # across the batch's launches, so a zero after the batch means none of the
    # launches changed anything (i.e. convergence). The relaxation is a no-op
    # at its fixed point, so any launches past convergence within a batch are
    # harmless. These step kernels are cheap and numerous, so the per-launch
    # sync dominates; checking every 8th launch recovers most of that cost.
    if convergence_check_interval is None:
        convergence_check_interval = 8
    launched = 0
    while launched < max_iterations:
        changed[0] = 0
        batch = min(convergence_check_interval, max_iterations - launched)
        for _ in range(batch):
            step_kernel(
                (blocks,),
                (threads_per_block,),
                make_args(read_bufs, write_bufs),
            )
            read_bufs, write_bufs = write_bufs, read_bufs
        launched += batch
        if changed[0] == 0:
            break

    # After the final swap, read_bufs holds the most recently written values.
    labels = read_bufs[0]
    priority = read_bufs[1]
    return labels, state, priority, changed
