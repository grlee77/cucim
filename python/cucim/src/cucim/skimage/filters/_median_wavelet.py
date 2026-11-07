# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wavelet matrix-based rank (median) filter implementation for cuCIM.

This module provides a fast GPU-based median filter using the wavelet matrix
algorithm described in:
- Y. Moroto and N. Umetani., "Constant Time Median Filter Using 2D Wavelet
  Matrix," in ACM Trans. Graph. Volume 41, Issue 6, Article No.: 267,
  December 2022. https://doi.org/10.1145/3550454.3555512

The run time of the query does not depend on the size of the window. The same
data structure answers any rank query, so `_median_wavelet_filter` accepts a
``rank`` (the median is the default).

The implementation here was adapted from the MIT-licensed one provided by the
authors at:
https://github.com/TumoiYorozu/WMatrixMedian?tab=readme-ov-file

That same implementation was also contributed to Open-CV in the following PR:
https://github.com/opencv/opencv_contrib/pull/3627

Changes relative to the code of the authors and of OpenCV
---------------------------------------------------------
The kernels were rewritten for CuPy RawKernels and NVRTC (see
LICENSE-3rdparty.md for the attribution). Beyond that, this module adds:

- Support for every numeric dtype: signed integers, 64-bit integers and
  float16/32/64 (OpenCV handles unsigned 8, 16 and 32-bit integers and 32-bit
  floats). Types of up to 16 bits are mapped to order-preserving unsigned
  keys; wider types use the ranks of the values from an ``argsort``, so the
  number of value levels is the number of bits of the image size.
- The border modes of `scipy.ndimage` (``reflect``, ``mirror``, ``nearest``,
  ``constant`` and ``wrap``, with ``cval``) by padding the image explicitly,
  instead of cropping the border or shrinking the window.
- Any rank (the original code computes the median).
- The numbers of value and column bit levels are run-time arguments of the
  query kernel instead of template parameters.
- Filtering in slabs of rows, each with a halo of ``radius_y`` rows, so that
  large images need a bounded amount of device memory: the matrices need
  about 70 to 210 bytes per pixel, so a whole 8192 x 8192 `uint8` image would
  need 4.6 GB. The slab height is chosen from an estimate of the memory use
  (`_estimate_memory_usage`) and the available memory (`_plan_slab_rows`);
  small images are filtered in a single slab. Volumes of types wider than
  8 bits (more than 0.8M voxels) are also filtered in slabs of about 0.5M
  voxels, because the query is up to 7.6 times faster when the matrices of a
  slab fit in cache (`_plan_speed_slab_planes`).
- Query thread blocks ordered down the image (concurrently running blocks
  cover a tall strip), which matters for large images, a 16 x 64 thread block
  shape, and construction buffers that are not zero-filled when they are
  fully written.
- A check that the estimated memory use fits (`_has_enough_memory`), used by
  `median` to select an algorithm.
- A 3D version of the query for volumes with a box footprint
  (`_median_wavelet_filter_3d`): the padded planes are stacked as the rows of
  one 2D image, so the construction is unchanged, and the window of a voxel is
  one range of rows in each of the ``2 * radius_z + 1`` planes it covers. At
  each value level the query adds up the counts of the ranges, and every range
  then follows the same 0 or 1 branch, so the cost is that of the 2D query
  times the depth of the window (``wavelet_median3d``, with the state of the
  ranges in per-thread arrays, which limits ``radius_z`` to
  ``WM_MAX_RADIUS_Z``). Volumes are filtered in slabs of planes with a halo of
  ``radius_z`` planes.

Not included: the polygonal windows and the multi-channel launch of the
original, and the 64-bit bit-vector words.

Multi-channel images (`median` with a footprint of size 1 along the channel
axis) are filtered one channel at a time. Batching the channels in the kernels
(as the original does with ``blockIdx.z``) is not expected to give a large
benefit: the matrices of a channel depend only on that channel's values, so
the memory traffic does not shrink, and for images of about 1024 x 1024 and
larger the GPU is already full (three channels on concurrent streams took as
long as three sequential calls). Only the launch overhead, which dominates
below about 512 x 512, would be saved; CUDA graphs could address that.

Data flow
---------
1. Pad the image (``mode``/``cval``) by the filter radii.
2. Convert the values to unsigned integer keys that sort like the values
   (`_to_keys`).
3. Build the value wavelet matrix: one bit vector per key bit, with the
   elements stably partitioned by that bit (most significant bit first).
4. For every value level also build a *column* wavelet matrix over the column
   indices of the elements in that level's order (one bit vector per bit of
   the column index). Counting how many elements of a pixel window have a 0 at
   the current value bit is then a rank query on these.
5. One thread per output pixel selects the element of the requested rank by
   walking the value levels from the most significant bit down.
"""

import math
import os
from collections import namedtuple

import cupy as cp
import numpy as np

from cucim.skimage._shared.utils import _to_np_mode

# =============================================================================
# Constants
# =============================================================================

WM_WARP_SIZE = 32
WM_WORD_SIZE = 32  # bits per bit vector word (BlockT::nbit)
WM_BLOCK_T_SIZE = 8  # sizeof(BlockT): 4 bytes prefix count + 4 bytes word
WM_THREAD_PER_GRID = 512  # Threads per block for construction
WM_MAX_BLOCK_X = 1024  # Maximum grid X dimension
WM_SRC_CACHE_DIV = 2  # Shared memory caching factor of the up-sweep

# Median query kernel thread configuration
WM_THREADS_W = 16  # Threads per block in X for median query
WM_THREADS_H = 64  # Threads per block in Y for median query

# Column indices are stored as uint16
WM_MAX_PADDED_WIDTH = 65534

# The 3D query keeps the state of one row range per plane of the window in
# per-thread arrays
WM_MAX_RADIUS_Z = 31

# Large images are filtered in slabs of rows (each with a halo of ``radius_y``
# rows) when the matrices of the whole image would not fit in this fraction of
# the available device memory. The estimate is not exact and the memory pool
# fragments, so the fraction is well below 1.
WM_MEMORY_FRACTION = 0.5
# The slab height is at least this many rows and at least ``4 * radius_y``
# (so that recomputing the halo rows costs at most 50% more work).
WM_MIN_SLAB_ROWS = 64
# The same for the planes of a volume
WM_MIN_SLAB_PLANES = 4

# Volumes of types wider than 8 bits are also filtered in slabs of about this
# many voxels when they have at least ``WM_SPEED_SLAB_MIN_COUNT`` times as
# many, because the query is much faster (1.1x to 7.6x on an RTX A6000) when
# the matrices of a slab stay in cache. uint8 volumes do not benefit, and for
# images the gain is small and not consistent, so they are not slabbed for
# speed.
WM_SPEED_SLAB_VOXELS = 2**19
WM_SPEED_SLAB_MIN_COUNT = 1.5

# CuPy compilation options for CUB
CUB_COMPILE_OPTIONS = ("--std=c++17", "-DCUB_DISABLE_BF16_SUPPORT")

_KERNEL_DIR = os.path.join(os.path.dirname(__file__), "cuda")

_CUDA_TYPES = {
    np.dtype(np.uint8): "unsigned char",
    np.dtype(np.uint16): "unsigned short",
    np.dtype(np.uint32): "unsigned int",
}

_SUPPORTED_MODES = ("constant", "nearest", "reflect", "mirror", "wrap")

# =============================================================================
# Value handling
# =============================================================================


def _uses_rank_mode(dtype):
    """True if values must be replaced by their ranks in the sorted image.

    Types of up to 16 bits are mapped to order-preserving unsigned keys
    (`_to_keys`), which have at most 16 matrix levels. For wider types the
    ranks need fewer levels (about log2 of the number of pixels, e.g. 22 for
    a 2048 x 2048 image instead of 32 or 64), and ranking the image by sorting
    it is cheaper than the extra levels: measured on an RTX A6000, a 2048 x
    2048 uint32 or float32 image takes 22 ms with ranks and 30 ms with 32-bit
    keys.
    """
    return np.dtype(dtype).itemsize > 2


def _key_dtype(dtype):
    """Unsigned dtype of the keys stored in the matrix for ``dtype``."""
    dtype = np.dtype(dtype)
    if _uses_rank_mode(dtype):
        return np.dtype(np.uint32)
    return np.dtype(f"u{dtype.itemsize}")


def _to_keys(image):
    """Map an image of up to 16 bits per value to unsigned integer keys.

    The mapping is monotonic (``a < b`` implies ``key(a) < key(b)``), so
    ranks of keys equal ranks of values. Signed integers get their sign bit
    flipped. Floats are mapped with the usual IEEE trick (flip the sign bit of
    non-negative values, all bits of negative ones); NaNs sort last like
    ``numpy.sort``. ``-0.0`` sorts just before ``0.0`` (it compares equal).
    """
    dtype = image.dtype
    if dtype.kind == "u":
        return image
    udt = np.dtype(f"u{dtype.itemsize}")
    sign = udt.type(1 << (dtype.itemsize * 8 - 1))
    u = image.view(udt)
    if dtype.kind == "i":
        return u ^ sign
    keys = cp.where((u & sign) != 0, ~u, u | sign)
    return cp.where(cp.isnan(image), udt.type(np.iinfo(udt).max), keys)


def _from_keys(keys, dtype):
    """Inverse of `_to_keys` (NaNs come back as NaNs)."""
    dtype = np.dtype(dtype)
    if dtype.kind == "u":
        return keys
    sign = keys.dtype.type(1 << (dtype.itemsize * 8 - 1))
    if dtype.kind == "i":
        return (keys ^ sign).view(dtype)
    return cp.where((keys & sign) != 0, keys ^ sign, ~keys).view(dtype)


def _to_ranks(image):
    """Replace each value by its (unique) rank in the sorted image.

    Returns the uint32 ranks and the sorted values for the final lookup.
    """
    flat = image.ravel()
    order = cp.argsort(flat)
    ranks = cp.empty(flat.size, dtype=cp.uint32)
    ranks[order] = cp.arange(flat.size, dtype=cp.uint32)
    return ranks.reshape(image.shape), flat[order]


# =============================================================================
# Utility Functions
# =============================================================================


def _div_ceil(a, b):
    """Integer division rounding up."""
    return (a + b - 1) // b


def _get_bit_len(val):
    """Number of bits needed to represent ``val`` (at least 1)."""
    return max(int(val).bit_length(), 1)


def _next_power_of_2(v):
    """Round up to the next power of 2."""
    return 1 << (v - 1).bit_length() if v > 1 else 1


# =============================================================================
# Buffer Size Calculations
# =============================================================================


def _get_block_pair_num(
    size,
    thread_per_grid=WM_THREAD_PER_GRID,
    max_block_x=WM_MAX_BLOCK_X,
    warp_size=WM_WARP_SIZE,
):
    """
    Number of warp-rows (of ``warp_size`` elements) each construction block
    processes, chosen so that a block handles a multiple of 65536 elements
    once the image is large (or a power of two number of rows otherwise).
    """
    x_chunk = max(65536 // thread_per_grid // warp_size, 1)

    total_gridx = _div_ceil(size, thread_per_grid * warp_size)
    block_pair_num = _div_ceil(total_gridx, max_block_x)

    if block_pair_num <= x_chunk:
        block_pair_num = _next_power_of_2(block_pair_num)
    else:
        block_pair_num = _div_ceil(block_pair_num, x_chunk) * x_chunk

    return block_pair_num * warp_size


BufferSizes = namedtuple(
    "WaveletMatrixBufferSizes",
    [
        "size",  # Padded total pixels (multiple of the word size)
        "val_bit_len",  # Bits per value
        "w_bit_len",  # Bits for column index
        "bv_block_len",  # BlockT entries per bit level
        "bv_block_bytes_div32",  # Bytes of the value bit vectors / 32
        "wm_bv_block_bytes_div32",  # Bytes of the column bit vectors / 32
        "block_pair_num",  # Warp-rows per construction block
        "nsum_scan_buf_len",  # Scan buffer length
        "buf_bytes_div32",  # Working buffer bytes / 32
        "nsum_pos",  # Index of the BlockT holding the total count
        "grid_x",  # Grid X for construction
    ],
)


def _compute_buffer_sizes(height, width, val_bit_len, val_bytes):
    """Compute buffer sizes for a (padded) ``height`` x ``width`` image."""
    thread_per_grid = WM_THREAD_PER_GRID
    w_bit_len = _get_bit_len(width)

    size = _div_ceil(height * width, WM_WORD_SIZE) * WM_WORD_SIZE

    # BlockT entries per bit level: one per word, plus one for the total,
    # rounded up to a multiple of 16
    bv_block_len = (
        _div_ceil(size, thread_per_grid) * thread_per_grid // WM_WORD_SIZE + 1
    )
    bv_block_len = _div_ceil(bv_block_len, 16) * 16

    bv_block_bytes = WM_BLOCK_T_SIZE * val_bit_len * bv_block_len
    wm_bv_block_bytes = WM_BLOCK_T_SIZE * w_bit_len * bv_block_len * val_bit_len

    block_pair_num = _get_block_pair_num(size)
    nsum_scan_buf_len = _div_ceil(size, thread_per_grid * block_pair_num)
    nsum_scan_buf_len = _div_ceil(nsum_scan_buf_len, 4) * 4

    # Working buffer: 2 scan buffers, 2 index buffers and a value buffer
    buf_bytes = 4 * 2 * nsum_scan_buf_len + 2 * 2 * size + val_bytes * size

    return BufferSizes(
        size=size,
        val_bit_len=val_bit_len,
        w_bit_len=w_bit_len,
        bv_block_len=bv_block_len,
        bv_block_bytes_div32=_div_ceil(bv_block_bytes, 32),
        wm_bv_block_bytes_div32=_div_ceil(wm_bv_block_bytes, 32),
        block_pair_num=block_pair_num,
        nsum_scan_buf_len=nsum_scan_buf_len,
        buf_bytes_div32=_div_ceil(buf_bytes, 32),
        nsum_pos=size // WM_WORD_SIZE,
        grid_x=_div_ceil(size, thread_per_grid * block_pair_num),
    )


def _estimate_memory_usage(height, width, dtype, ry, rx):
    """Estimated peak device memory (bytes) of `_median_wavelet_filter`."""
    dtype = np.dtype(dtype)
    ph, pw = height + 2 * ry, width + 2 * rx
    val_bit_len = _get_val_bit_len(dtype, ph * pw)
    buf = _compute_buffer_sizes(ph, pw, val_bit_len, _key_dtype(dtype).itemsize)
    # the two matrices, the construction buffers and the two
    # (val_bit_len, size) uint16 column index arrays
    total = (buf.bv_block_bytes_div32 + buf.wm_bv_block_bytes_div32) * 32
    total += buf.buf_bytes_div32 * 32
    total += 2 * val_bit_len * buf.size * 2
    # padded input, keys and (rank mode) sorted values and ranks
    total += ph * pw * (dtype.itemsize + _key_dtype(dtype).itemsize)
    if _uses_rank_mode(dtype):
        total += ph * pw * (8 + dtype.itemsize + 4)
    return total + height * width * dtype.itemsize


def _slab_memory_estimate(rows, height, width, dtype, ry, rx):
    """Estimated peak memory (bytes) when filtering in slabs of ``rows``."""
    if rows >= height:
        return _estimate_memory_usage(height, width, dtype, ry, rx)
    # the matrices of one slab plus the padded image and the output, which are
    # kept for all slabs
    itemsize = np.dtype(dtype).itemsize
    whole = ((height + 2 * ry) * (width + 2 * rx) + height * width) * itemsize
    return _estimate_memory_usage(rows, width, dtype, ry, rx) + whole


def _available_memory():
    """Free device memory plus the free blocks of the CuPy memory pool."""
    free = cp.cuda.runtime.memGetInfo()[0]
    return free + cp.get_default_memory_pool().free_bytes()


def _largest_fitting(lo, hi, estimate, budget):
    """Largest n in [lo, hi] with ``estimate(n) <= budget`` (None if none).

    ``estimate`` must grow with n.
    """
    if estimate(lo) > budget:
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _plan_slab_rows(height, width, dtype, ry, rx, budget=None):
    """Number of output rows to filter at a time.

    Returns ``height`` (a single slab) if the whole image fits in ``budget``
    bytes (default: ``WM_MEMORY_FRACTION`` of the available memory), else the
    largest slab height that does. Returns None if not even the smallest slab
    (see ``WM_MIN_SLAB_ROWS``) fits.
    """
    if budget is None:
        budget = WM_MEMORY_FRACTION * _available_memory()
    # the matrix positions are 32-bit
    max_rows = min(height, (2**32 - 1) // (width + 2 * rx) - 2 * ry)
    if max_rows < 1:
        return None

    def estimate(rows):
        return _slab_memory_estimate(rows, height, width, dtype, ry, rx)

    if max_rows == height and estimate(height) <= budget:
        return height
    lo = min(max_rows, max(WM_MIN_SLAB_ROWS, 4 * ry))
    return _largest_fitting(lo, max_rows, estimate, budget)


def _slab_memory_estimate_3d(planes, nz, height, width, dtype, rz, ry, rx):
    """Estimated peak memory (bytes) of the 3D filter in slabs of ``planes``."""
    # the matrices are those of an image of (planes + 2 rz) * (H + 2 ry) rows
    rows = (planes + 2 * rz) * (height + 2 * ry)
    itemsize = np.dtype(dtype).itemsize
    # (the output has planes * H rows, not `rows`)
    slab = _estimate_memory_usage(rows, width, dtype, 0, rx)
    slab += (planes * height - rows) * width * itemsize
    if planes >= nz:
        return slab
    padded = (nz + 2 * rz) * (height + 2 * ry) * (width + 2 * rx)
    return slab + (padded + nz * height * width) * itemsize


def _plan_slab_planes(nz, height, width, dtype, rz, ry, rx, budget=None):
    """Number of output planes of a volume to filter at a time (see
    `_plan_slab_rows`); None if not even the smallest slab fits."""
    if budget is None:
        budget = WM_MEMORY_FRACTION * _available_memory()
    plane = (height + 2 * ry) * (width + 2 * rx)
    max_planes = min(nz, (2**32 - 1) // plane - 2 * rz)
    if max_planes < 1:
        return None

    def estimate(planes):
        return _slab_memory_estimate_3d(
            planes, nz, height, width, dtype, rz, ry, rx
        )

    if max_planes == nz and estimate(nz) <= budget:
        return nz
    lo = min(max_planes, max(WM_MIN_SLAB_PLANES, 4 * rz))
    return _largest_fitting(lo, max_planes, estimate, budget)


def _plan_speed_slab_planes(nz, height, width, dtype, rz, ry, rx, planned):
    """``planned`` (from `_plan_slab_planes`) reduced for speed, if worthwhile.

    The slabs are smaller than planned only if that needs no more memory.
    """
    voxels = height * width
    if (
        planned is None
        or np.dtype(dtype).itemsize == 1
        or nz * voxels < WM_SPEED_SLAB_MIN_COUNT * WM_SPEED_SLAB_VOXELS
    ):
        return planned
    planes = max(WM_MIN_SLAB_PLANES, 4 * rz, WM_SPEED_SLAB_VOXELS // voxels)
    if planes >= planned:
        return planned

    def estimate(n):
        return _slab_memory_estimate_3d(n, nz, height, width, dtype, rz, ry, rx)

    return planes if estimate(planes) <= estimate(planned) else planned


def _has_enough_memory(image, footprint_shape):
    """True if the image can be filtered within the memory budget."""
    radii = [s // 2 for s in footprint_shape]
    if image.ndim == 3:
        planned = _plan_slab_planes(*image.shape, image.dtype, *radii)
    else:
        planned = _plan_slab_rows(*image.shape, image.dtype, *radii)
    return planned is not None


def _get_val_bit_len(dtype, total_pixels):
    """Number of matrix levels (bits per key) for ``dtype``."""
    if _uses_rank_mode(dtype):
        return _get_bit_len(total_pixels)
    return np.dtype(dtype).itemsize * 8


# =============================================================================
# Kernel Compilation
# =============================================================================


def _gen_preamble(val_type):
    """CUDA macro definitions shared by the construction and query kernels."""
    return f"""
#define WM_VAL_T {val_type}
#define WM_THREAD_PER_GRID {WM_THREAD_PER_GRID}
#define WM_MAX_BLOCK_X {WM_MAX_BLOCK_X}
#define WM_SRC_CACHE_DIV {WM_SRC_CACHE_DIV}
#define WM_THREADS_W {WM_THREADS_W}
#define WM_THREADS_H {WM_THREADS_H}
#define WM_MAX_NZ {WM_MAX_RADIUS_Z * 2 + 1}
"""


def _read_kernel_source(*names):
    sources = []
    for name in names:
        with open(os.path.join(_KERNEL_DIR, name)) as f:
            sources.append(f.read())
    return "".join(sources)


@cp.memoize(for_each_device=True)
def _get_median_query_kernel(val_type, ndim=2):
    """Compile the query kernel (2D or 3D) for a matrix value type."""
    code = _gen_preamble(val_type) + _read_kernel_source(
        "wavelet_matrix_common.cuh", "wavelet_matrix_median.cu"
    )
    return cp.RawKernel(
        code, f"wavelet_median{ndim}d", options=CUB_COMPILE_OPTIONS
    )


_CONSTRUCTION_KERNELS = (
    "wavelet_first_pass",
    "wavelet_exclusive_sum",
    # Value up-sweep fused with the initialization of the column matrices
    "wavelet_upsweep_with_col_init",
    # Batched over the value levels (grid.y)
    "wavelet_wm_first_pass_batched",
    "wavelet_exclusive_sum_batched",
    "wavelet_wm_upsweep_batched",
)


@cp.memoize(for_each_device=True)
def _get_construction_kernels(val_type):
    """Compile the construction kernels (a dict of name -> RawKernel)."""
    code = _gen_preamble(val_type) + _read_kernel_source(
        "wavelet_matrix_common.cuh", "wavelet_matrix_construct.cu"
    )
    module = cp.RawModule(code=code, options=CUB_COMPILE_OPTIONS)
    return {name: module.get_function(name) for name in _CONSTRUCTION_KERNELS}


# =============================================================================
# Compatibility Check
# =============================================================================


def _can_use_wavelet_matrix(image, footprint_shape=None, mode=None):
    """
    Check if the wavelet matrix filter can be used.

    Parameters
    ----------
    image : cupy.ndarray
        The input image.
    footprint_shape : tuple of int, optional
        Footprint shape (size_axis0, size_axis1). Both must be odd. The
        footprint itself must be a full rectangle (checked by the caller).
    mode : str, optional
        Boundary mode of the filter.

    Returns
    -------
    compatible : bool
        Whether the wavelet matrix filter can be used
    reason : str or None
        Reason for incompatibility, or None if compatible
    """
    if image.ndim not in (2, 3):
        return False, "Only 2D and 3D images are supported"

    if image.dtype.kind not in "uif" or image.dtype.itemsize > 8:
        return False, f"Unsupported dtype: {image.dtype}"

    if mode is not None and mode not in _SUPPORTED_MODES:
        return False, f"Unsupported mode: {mode!r}"

    rx = 0
    if footprint_shape is not None:
        if len(footprint_shape) != image.ndim:
            return False, "Footprint must have as many axes as the image"
        if any(s % 2 == 0 for s in footprint_shape):
            return False, "Footprint dimensions must be odd"
        if any(s < 3 for s in footprint_shape):
            return False, "Footprint must be at least 3 along each axis"
        rx = footprint_shape[-1] // 2
        if image.ndim == 3 and footprint_shape[0] // 2 > WM_MAX_RADIUS_Z:
            return (
                False,
                f"Footprint radius along axis 0 exceeds {WM_MAX_RADIUS_Z}",
            )

    padded_w = image.shape[-1] + 2 * rx
    if padded_w > WM_MAX_PADDED_WIDTH:
        return False, f"Padded width exceeds maximum ({WM_MAX_PADDED_WIDTH})"

    return True, None


# =============================================================================
# Parameters and buffers
# =============================================================================


class WaveletMatrixMedianParams:
    """Sizes and strides for filtering a ``height`` x ``width`` image."""

    def __init__(self, height, width, dtype, radius_y, radius_x):
        self.height = height
        self.width = width
        self.dtype = np.dtype(dtype)
        self.radius_y = radius_y
        self.radius_x = radius_x

        # dimensions of the padded image the matrices are built on
        self.padded_h = height + 2 * radius_y
        self.padded_w = width + 2 * radius_x

        # dtype of the matrix values (keys or ranks, see `_key_dtype`)
        self.matrix_dtype = _key_dtype(self.dtype)
        self.val_type = _CUDA_TYPES[self.matrix_dtype]
        self.is_rank_mode = _uses_rank_mode(self.dtype)

        self.val_bit_len = _get_val_bit_len(
            self.dtype, self.padded_h * self.padded_w
        )
        self.buf_sizes = _compute_buffer_sizes(
            self.padded_h,
            self.padded_w,
            self.val_bit_len,
            self.matrix_dtype.itemsize,
        )
        self.w_bit_len = self.buf_sizes.w_bit_len
        self.nsum_pos = self.buf_sizes.nsum_pos
        self.bv_block_len = self.buf_sizes.bv_block_len
        # bytes per bit vector level / 32
        self.bv_block_h_byte_div32 = (self.bv_block_len * WM_BLOCK_T_SIZE) // 32

    def get_median_grid_block(self):
        """Get grid and block dimensions for median query kernel.

        ``blockIdx.x`` indexes tiles along the rows (axis 0) and
        ``blockIdx.y`` tiles along the columns, so that the blocks running at
        the same time cover a tall strip of the image. The query walks
        matrices whose layout follows the row-major order of the pixels; with
        the opposite order, float32 4096 x 4096 images took 300 ms instead of
        61 ms in the query kernel.
        """
        grid = (
            _div_ceil(self.height, WM_THREADS_H),
            _div_ceil(self.width, WM_THREADS_W),
            1,
        )
        return grid, (WM_THREADS_W, WM_THREADS_H, 1)

    def get_construct_grid_block(self):
        """Get grid and block dimensions for construction kernels."""
        grid = (self.buf_sizes.grid_x, 1, 1)
        block = (WM_WARP_SIZE, WM_THREAD_PER_GRID // WM_WARP_SIZE, 1)
        return grid, block


class WaveletMatrixBuffers:
    """
    GPU buffers used in wavelet matrix construction.

    - Value bit vectors (``bv_blocks``) and column bit vectors (``wm_blocks``)
      that make up the matrices (BlockT entries, stored as bytes)
    - Working buffers for the construction
    """

    def __init__(self, params):
        # Only the scan buffers need to start at zero: everything else is
        # fully written by the construction kernels before it is read.
        self.params = params
        buf = params.buf_sizes
        val_bit_len = params.val_bit_len

        self.bv_blocks = cp.empty(buf.bv_block_bytes_div32 * 32, cp.uint8)
        self.wm_blocks = cp.empty(buf.wm_bv_block_bytes_div32 * 32, cp.uint8)

        # Scan buffers (two, alternating)
        self.nsum_scan_buf = cp.zeros(buf.nsum_scan_buf_len, dtype=cp.uint32)
        self.nsum_scan_buf2 = cp.zeros(buf.nsum_scan_buf_len, dtype=cp.uint32)

        # Alternating value and column index buffers
        self.idx_buf1 = cp.empty(buf.size, dtype=cp.uint16)
        self.idx_buf2 = cp.empty(buf.size, dtype=cp.uint16)
        self.val_buf1 = cp.zeros(buf.size, dtype=params.matrix_dtype)
        self.val_buf2 = cp.empty(buf.size, dtype=params.matrix_dtype)

        # Column indices of the elements in the order of every value level
        # (populated at the input positions of each level during the value
        # up-sweep) and a scratch array of the same shape: the batched column
        # matrix construction processes all levels at once (grid.y)
        self.wm_idx_levels = cp.empty((val_bit_len, buf.size), cp.uint16)
        self.wm_idx_work = cp.empty((val_bit_len, buf.size), cp.uint16)

        # Scan buffers for the column matrices (val_bit_len sets, alternating)
        scan_len = buf.nsum_scan_buf_len * val_bit_len
        self.wm_scan_buf = cp.zeros(scan_len, dtype=cp.uint32)
        self.wm_scan_buf2 = cp.zeros(scan_len, dtype=cp.uint32)


# =============================================================================
# Construction Orchestration
# =============================================================================


def _run_wavelet_construction(src_padded, params, buffers, kernels):
    """
    Run the wavelet matrix construction pipeline (OpenCV-style batched approach).

    This function orchestrates the multi-pass construction of the wavelet matrix:

    Phase 1: Value WM Construction
    1. First pass: Initialize indices and count MSB zeros
    2. Exclusive sum: Compute prefix sums of block counts
    3. For each value bit (MSB to LSB):
       a. Up-sweep: Build bitvector and reorder values
       b. Store sorted column indices in wm_idx_levels[h]
       c. Exclusive sum: Update prefix sums

    Phase 2: Batched Column WM Construction
    For all value levels in parallel (using grid.y = val_bit_len):
    1. First pass: Count zeros for MSB of column indices
    2. Exclusive sum: Compute prefix sums
    3. For each column bit:
       a. Up-sweep: Build column WM bitvectors
       b. Exclusive sum: Update prefix sums

    Parameters
    ----------
    src_padded : cupy.ndarray
        Padded source image (contiguous, flattened or 2D)
    params : WaveletMatrixMedianParams
        Parameters object
    buffers : WaveletMatrixBuffers
        Allocated GPU buffers
    kernels : dict
        Dictionary of compiled kernels

    Returns
    -------
    None (results stored in buffers)
    """
    buf = params.buf_sizes
    val_bit_len = params.val_bit_len
    w_bit_len = params.w_bit_len

    # Flatten source if needed
    src_flat = src_padded.ravel()

    # Fill buffer with max value as sentinel (so padding has all bits = 1)
    # This ensures padding elements don't affect the wavelet matrix construction
    # For float mode, we use ranks which only need val_bit_len bits
    max_val = (1 << params.val_bit_len) - 1
    buffers.val_buf1.fill(max_val)

    # Copy source to value buffer 1
    buffers.val_buf1[: src_flat.size] = src_flat

    # Grid and block for construction kernels
    grid, block = params.get_construct_grid_block()

    # Grid and block for exclusive sum (single block per channel)
    exsum_grid = (1, 1, 1)
    exsum_block = (buf.grid_x, 1, 1)

    # -------------------------------------------------------------------------
    # Step 1: First pass - initialize indices and count MSB zeros
    # -------------------------------------------------------------------------
    size_div_warp = buf.size // WM_WARP_SIZE

    # Mask for MSB comparison
    max_val = (1 << val_bit_len) - 1
    msb_mask = max_val >> 1  # e.g., 0x7F for uint8

    # For float mode, use uint32 for masks; otherwise use the dtype
    first_pass_mask_dtype = params.matrix_dtype

    nsum_scan_buf, nsum_scan_buf2 = (
        buffers.nsum_scan_buf,
        buffers.nsum_scan_buf2,
    )
    wm_scan_buf, wm_scan_buf2 = buffers.wm_scan_buf, buffers.wm_scan_buf2

    kernels["wavelet_first_pass"](
        grid,
        block,
        (
            first_pass_mask_dtype.type(msb_mask),  # mask
            np.uint16(buf.block_pair_num),  # block_pair_num
            np.uint32(size_div_warp),  # size_div_warp
            buffers.val_buf1,  # src
            nsum_scan_buf,  # nsum_scan_buf
            np.uint32(buf.buf_bytes_div32),  # buf_byte_div32
            buffers.idx_buf1,  # buf_idx
            np.int32(params.padded_w),  # W
            np.uint32(params.padded_h * params.padded_w),  # WH
        ),
    )

    # -------------------------------------------------------------------------
    # Step 2: Exclusive sum for first level (MSB = level val_bit_len - 1)
    # -------------------------------------------------------------------------
    # Get pointer to MSB level (highest offset in OpenCV convention)
    bv_block_h_bytes = params.bv_block_len * WM_BLOCK_T_SIZE
    msb_bv_offset = (val_bit_len - 1) * bv_block_h_bytes
    bv_ptr = buffers.bv_blocks[msb_bv_offset:]

    kernels["wavelet_exclusive_sum"](
        exsum_grid,
        exsum_block,
        (
            nsum_scan_buf,  # nsum_scan_buf
            nsum_scan_buf2,  # nsum_buf_test2
            bv_ptr,  # nsum_p (stores total at nsum_pos)
            np.uint32(buf.buf_bytes_div32),  # buf_byte_div32
            np.uint32(buf.bv_block_bytes_div32),  # bv_block_byte_div32
            np.uint32(buf.nsum_pos),  # nsum_pos
        ),
    )

    # -------------------------------------------------------------------------
    # Step 3: Up-sweep for each value bit level (OpenCV-style fused column WM)
    # -------------------------------------------------------------------------
    # OpenCV insight: Column WM for level h+1 is populated during value level h.
    # The SOURCE positions during level h are sorted by value bits >= h+1,
    # which is exactly what column WM level h+1 needs.
    #
    # This kernel:
    #   1. Does value WM upsweep (sorting, bitvector creation)
    #   2. Writes column indices at SOURCE positions (coalesced writes!)
    #   3. Counts zeros for column WM MSB (fused, no separate first_pass)
    #
    # This eliminates:
    #   - Python memcpy operations
    #   - wavelet_wm_first_pass_batched kernel

    size_div_w = buf.size // WM_WORD_SIZE
    block_t_size = WM_BLOCK_T_SIZE
    bv_block_h_bytes = params.bv_block_len * block_t_size
    w_bit_len = params.w_bit_len

    # Alternating buffers
    src_val = buffers.val_buf1
    dst_val = buffers.val_buf2
    src_idx = buffers.idx_buf1
    dst_idx = buffers.idx_buf2

    # Compute bytes per column WM bit level (same as value WM)
    wm_block_h_bytes = params.bv_block_len * block_t_size

    # Pre-compute column WM parameters for fused kernel
    nsum_scan_stride = buf.nsum_scan_buf_len
    wm_msb_mask = (1 << (w_bit_len - 1)) - 1

    # Clear column WM scan buffers (fused kernel writes per-block counts)
    wm_scan_buf.fill(0)
    wm_scan_buf2.fill(0)

    for h in range(val_bit_len - 1, -1, -1):
        is_last = 1 if h == 0 else 0

        # Current mask (for bit h from MSB)
        bit_pos = val_bit_len - 1 - h
        mask = (1 << (val_bit_len - bit_pos - 1)) - 1

        # Get bitvector pointers for current and previous levels
        # OpenCV convention: level h is stored at offset h * bv_block_h_bytes
        # So MSB (level val_bit_len-1) is at the highest offset
        bv_curr_offset = h * bv_block_h_bytes
        bv_prev_offset = min(val_bit_len - 1, h + 1) * bv_block_h_bytes

        # For float mode, use uint32 for masks; otherwise use the dtype
        mask_dtype = params.matrix_dtype

        # OpenCV-style: Write column WM for level h+1 during value level h
        # First level (h = val_bit_len - 1): no column WM write
        # Other levels: write to column WM level h+1
        if h == val_bit_len - 1:
            # First value level: no column WM write
            wm_write_enabled = 0
            wm_idx_out = buffers.wm_idx_levels[0]  # Dummy, won't be used
            wm_scan_out = wm_scan_buf  # Dummy, won't be used
        else:
            # Write column indices for level h+1 at SOURCE positions
            wm_write_enabled = 1
            col_wm_level = h + 1  # Column WM level we're populating
            wm_idx_out = buffers.wm_idx_levels[col_wm_level]
            wm_scan_out = wm_scan_buf[col_wm_level * nsum_scan_stride :]

        # FUSED kernel: value upsweep + OpenCV-style column WM init + zero counting
        kernels["wavelet_upsweep_with_col_init"](
            grid,
            block,
            (
                mask_dtype.type(mask),  # mask
                np.uint16(buf.block_pair_num),  # block_pair_num
                np.uint32(size_div_w),  # size_div_w
                src_val,  # src
                dst_val,  # dst
                buffers.bv_blocks[bv_curr_offset:],  # nbit_bp
                nsum_scan_buf,  # nsum_buf_test
                nsum_scan_buf2,  # nsum_buf_test2
                np.uint32(buf.bv_block_bytes_div32),  # bv_block_byte_div32
                np.uint32(buf.buf_bytes_div32),  # buf_byte_div32
                src_idx,  # idx_p
                dst_idx,  # nxt_idx
                buffers.bv_blocks[bv_prev_offset:],  # nbit_bp_pre
                np.int32(is_last),  # is_last_val_bit
                # OpenCV-style column WM parameters
                wm_idx_out,  # wm_idx_out: col indices at SOURCE positions
                wm_scan_out,  # wm_scan_out: per-block zero counts
                np.uint16(wm_msb_mask),  # wm_mask: column WM MSB mask
                np.int32(wm_write_enabled),  # wm_write_enabled: flag
            ),
        )

        # Swap value buffers for next iteration
        src_val, dst_val = dst_val, src_val
        src_idx, dst_idx = dst_idx, src_idx

        # Exclusive sum for next value level (skip for last)
        if not is_last:
            # Swap scan buffers
            nsum_scan_buf, nsum_scan_buf2 = nsum_scan_buf2, nsum_scan_buf

            next_bv_offset = (h - 1) * bv_block_h_bytes
            kernels["wavelet_exclusive_sum"](
                exsum_grid,
                exsum_block,
                (
                    nsum_scan_buf,
                    nsum_scan_buf2,
                    buffers.bv_blocks[next_bv_offset:],
                    np.uint32(buf.buf_bytes_div32),
                    np.uint32(buf.bv_block_bytes_div32),
                    np.uint32(buf.nsum_pos),  # nsum_pos
                ),
            )

    # -------------------------------------------------------------------------
    # Step 3b: Handle column WM level 0 (final sorted indices)
    # -------------------------------------------------------------------------
    # Column WM level 0 corresponds to values fully sorted by ALL value bits.
    # This wasn't populated during the value upsweep loop, so we need to
    # copy the final sorted indices and count zeros.
    # After the loop with 8 swaps (for uint8), the final sorted indices
    # are in src_idx (the last dst before final swap).
    buffers.wm_idx_levels[0][:] = src_idx[:]

    # Count zeros for column WM level 0's MSB (single-level first pass)
    kernels["wavelet_wm_first_pass_batched"](
        (buf.grid_x, 1, 1),  # Single level, not batched
        block,
        (
            np.uint16(wm_msb_mask),  # mask
            np.uint16(buf.block_pair_num),  # block_pair_num
            np.uint32(size_div_w),  # size_div_warp
            buffers.wm_idx_levels[0],  # src_base (level 0 only)
            wm_scan_buf,  # nsum_scan_buf (level 0 is at offset 0)
            np.uint32(nsum_scan_stride),  # nsum_scan_stride
            np.uint32(buf.size),  # size
            np.uint32(buf.size),  # idx_buf_stride
        ),
    )

    # =========================================================================
    # Phase 2: Batched Column WM Construction
    # =========================================================================
    # Process all val_bit_len column WMs in parallel using grid.y = val_bit_len
    # NOTE: Zero counting for levels 1..val_bit_len-1 was fused into value upsweep.
    # Level 0 was handled above.

    # Grid for batched kernels: (grid_x, val_bit_len)
    batched_grid = (buf.grid_x, val_bit_len, 1)
    batched_exsum_grid = (1, val_bit_len, 1)

    # Stride between value level WM storage (in bytes)
    wm_block_stride = w_bit_len * wm_block_h_bytes

    # wm_idx_levels is already a contiguous 2D array (val_bit_len, buf.size)
    # Layout: [level_0_data, level_1_data, ..., level_{n-1}_data]
    idx_buf_stride = buf.size

    # Exclusive sum for first column WM level (MSB = w_bit_len - 1) - all levels
    # Zero counts are already in wm_scan_buf from fused counting
    wm_level_offset = (w_bit_len - 1) * wm_block_h_bytes
    kernels["wavelet_exclusive_sum_batched"](
        batched_exsum_grid,
        exsum_block,
        (
            wm_scan_buf,
            wm_scan_buf2,
            buffers.wm_blocks,  # nsum_p_base
            np.uint32(buf.nsum_pos),  # nsum_pos
            np.uint32(nsum_scan_stride),  # nsum_scan_stride
            np.uint32(wm_block_stride),  # wm_block_stride
            np.uint32(wm_level_offset),  # wm_level_offset
            np.int32(buf.grid_x),  # grid_x
        ),
    )

    # The two index buffers alternate as source and destination
    wm_src_base = buffers.wm_idx_levels.ravel()
    wm_dst_base = buffers.wm_idx_work.ravel()

    # Up-sweep for each column bit level (processing all value levels in parallel)
    for wm_h in range(w_bit_len - 1, -1, -1):
        is_last_wm = 1 if wm_h == 0 else 0

        # Current mask for column bit
        wm_bit_pos = w_bit_len - 1 - wm_h
        wm_mask = (1 << (w_bit_len - wm_bit_pos - 1)) - 1

        # Compute offsets for current and previous column WM levels
        wm_curr_offset = wm_h * wm_block_h_bytes
        prev_wm_level = min(w_bit_len - 1, wm_h + 1)
        wm_prev_offset = prev_wm_level * wm_block_h_bytes

        kernels["wavelet_wm_upsweep_batched"](
            batched_grid,
            block,
            (
                np.uint16(wm_mask),  # mask
                np.uint16(buf.block_pair_num),  # block_pair_num
                np.uint32(size_div_w),  # size_div_w
                wm_src_base,  # src_base
                wm_dst_base,  # dst_base
                buffers.wm_blocks,  # nbit_bp_base
                wm_scan_buf,  # nsum_buf_test
                wm_scan_buf2,  # nsum_buf_test2
                np.uint32(buf.wm_bv_block_bytes_div32),  # bv_block_div32
                buffers.wm_blocks,  # nbit_bp_pre_base
                np.int32(is_last_wm),  # is_last_bit
                np.uint32(nsum_scan_stride),  # nsum_scan_stride
                np.uint32(wm_block_stride),  # wm_block_stride
                np.uint32(wm_curr_offset),  # wm_curr_offset
                np.uint32(wm_prev_offset),  # wm_prev_offset
                np.uint32(buf.size),  # size
                np.uint32(idx_buf_stride),  # idx_buf_stride
                np.uint16((1 << w_bit_len) - 1),  # inf
            ),
        )

        # Swap buffers
        wm_src_base, wm_dst_base = wm_dst_base, wm_src_base

        # Exclusive sum for next level (skip for last)
        if not is_last_wm:
            wm_scan_buf, wm_scan_buf2 = wm_scan_buf2, wm_scan_buf

            next_level = wm_h - 1
            next_wm_offset = next_level * wm_block_h_bytes
            kernels["wavelet_exclusive_sum_batched"](
                batched_exsum_grid,
                exsum_block,
                (
                    wm_scan_buf,
                    wm_scan_buf2,
                    buffers.wm_blocks,  # nsum_p_base
                    np.uint32(buf.nsum_pos),  # nsum_pos
                    np.uint32(nsum_scan_stride),  # nsum_scan_stride
                    np.uint32(wm_block_stride),  # wm_block_stride
                    np.uint32(next_wm_offset),  # wm_level_offset
                    np.int32(buf.grid_x),  # grid_x
                ),
            )

    return buffers


# =============================================================================
# Query and full pipeline
# =============================================================================


def _run_median_query(params, buffers, output, rank):
    """Run the query kernel; ``output`` receives the keys/ranks selected."""
    kernel = _get_median_query_kernel(params.val_type)
    grid, block = params.get_median_grid_block()

    level_bytes = params.bv_block_len * WM_BLOCK_T_SIZE
    # The matrices are walked from the most significant level (highest offset)
    bv_start_offset = (params.val_bit_len - 1) * level_bytes
    wm_start_offset = (
        (params.val_bit_len - 1) * params.w_bit_len + (params.w_bit_len - 1)
    ) * level_bytes

    kernel(
        grid,
        block,
        (
            np.int32(params.height),
            np.int32(params.width),
            np.int32(params.padded_w),
            np.int32(params.radius_y),
            np.int32(params.radius_x),
            np.uint32(rank),
            output,
            buffers.wm_blocks[wm_start_offset:],
            np.uint32(params.nsum_pos),
            np.uint32(level_bytes // 32),
            buffers.bv_blocks[bv_start_offset:],
            np.uint8(params.w_bit_len),
            np.uint8(params.val_bit_len),
        ),
    )
    return output


def _median_wavelet_filter(
    image,
    radius_y,
    radius_x,
    mode="reflect",
    cval=0.0,
    rank=None,
    slab_rows=None,
):
    """
    Apply the wavelet matrix rank filter with a full rectangular footprint.

    The image is filtered in slabs of rows, each with a halo of ``radius_y``
    rows, when the matrices of the whole image would not fit in
    ``WM_MEMORY_FRACTION`` of the available device memory. The results do not
    depend on the slab height.

    Parameters
    ----------
    image : cupy.ndarray
        Input 2D image (see `_can_use_wavelet_matrix` for the supported
        dtypes).
    radius_y, radius_x : int
        Filter radii along axis 0 and 1 (footprint ``(2*radius_y + 1,
        2*radius_x + 1)``).
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        Boundary mode (scipy.ndimage names). The image is padded explicitly.
    cval : scalar
        Value used for ``mode='constant'``.
    rank : int, optional
        Index of the selected element when the window values are sorted in
        ascending order, in ``[0, (2*radius_y + 1) * (2*radius_x + 1))``.
        The default is the median, ``window_size // 2``.
    slab_rows : int, optional
        Number of output rows to filter at a time. The default is chosen from
        the available memory (see `_plan_slab_rows`).

    Returns
    -------
    result : cupy.ndarray
        Filtered image with the same shape and dtype as the input.
    """
    ok, reason = _can_use_wavelet_matrix(
        image,
        footprint_shape=(2 * radius_y + 1, 2 * radius_x + 1),
        mode=mode,
    )
    if not ok:
        raise ValueError(f"Cannot use wavelet matrix filter: {reason}")

    window_size = (2 * radius_y + 1) * (2 * radius_x + 1)
    if rank is None:
        rank = window_size // 2
    elif not 0 <= rank < window_size:
        raise ValueError(f"rank must be in [0, {window_size}), got {rank}")

    height, width = image.shape
    if slab_rows is None:
        slab_rows = _plan_slab_rows(
            height, width, image.dtype, radius_y, radius_x
        )
        if slab_rows is None:
            # nothing fits the budget: try the smallest slab anyway
            slab_rows = min(height, max(WM_MIN_SLAB_ROWS, 4 * radius_y))
    # use slabs of nearly equal height, so that the buffers can be reused
    n_slabs = _div_ceil(height, min(slab_rows, height))
    slab_rows = _div_ceil(height, n_slabs)

    # Pad (this also makes the array C-contiguous, as the kernels need)
    pad_kwargs = {"constant_values": cval} if mode == "constant" else {}
    padded = cp.pad(
        image,
        ((radius_y, radius_y), (radius_x, radius_x)),
        mode=_to_np_mode(mode),
        **pad_kwargs,
    )

    kernels = None
    params = buffers = None
    result = None if n_slabs == 1 else cp.empty_like(image)
    for r0 in range(0, height, slab_rows):
        r1 = min(height, r0 + slab_rows)
        # the rows of the padded image are contiguous
        slab = padded[r0 : r1 + 2 * radius_y]
        if params is None or params.height != r1 - r0:
            # (only the last slab can be shorter) release the old buffers
            # before allocating the new ones
            params = buffers = None
            params = WaveletMatrixMedianParams(
                r1 - r0, width, image.dtype, radius_y, radius_x
            )
            buffers = WaveletMatrixBuffers(params)
        if kernels is None:
            kernels = _get_construction_kernels(params.val_type)

        sorted_values = None
        if params.is_rank_mode:
            values, sorted_values = _to_ranks(slab)
        else:
            values = _to_keys(slab)
        _run_wavelet_construction(values, params, buffers, kernels)

        selected = cp.empty((r1 - r0, width), dtype=params.matrix_dtype)
        _run_median_query(params, buffers, selected, rank)
        if params.is_rank_mode:
            filtered = sorted_values[selected]
        else:
            filtered = _from_keys(selected, image.dtype)
        if result is None:
            return filtered
        result[r0:r1] = filtered
        # free the temporary arrays before the next slab allocates its own
        del values, sorted_values, selected, filtered
    return result


def _run_median_query_3d(params, buffers, output, rank, geometry):
    """Run the 3D query kernel (``output`` is ``(planes, H, W)``).

    ``params`` describe the matrices, which were built on the padded planes
    stacked as the rows of a 2D image; ``geometry`` is ``(planes, H, W,
    padded_H, rz, ry, rx)`` with ``planes`` the number of output planes.
    """
    planes, height, width, padded_h, rz, ry, rx = geometry
    kernel = _get_median_query_kernel(params.val_type, 3)
    level_bytes = params.bv_block_len * WM_BLOCK_T_SIZE
    bv_start_offset = (params.val_bit_len - 1) * level_bytes
    wm_start_offset = (
        (params.val_bit_len - 1) * params.w_bit_len + (params.w_bit_len - 1)
    ) * level_bytes
    grid = (
        _div_ceil(height, WM_THREADS_H),
        _div_ceil(width, WM_THREADS_W),
        planes,
    )
    kernel(
        grid,
        (WM_THREADS_W, WM_THREADS_H, 1),
        (
            np.int32(height),
            np.int32(width),
            np.int32(padded_h),
            np.int32(params.padded_w),
            np.int32(rz),
            np.int32(ry),
            np.int32(rx),
            np.uint32(rank),
            output,
            buffers.wm_blocks[wm_start_offset:],
            np.uint32(params.nsum_pos),
            np.uint32(level_bytes // 32),
            buffers.bv_blocks[bv_start_offset:],
            np.uint8(params.w_bit_len),
            np.uint8(params.val_bit_len),
        ),
    )
    return output


def _median_wavelet_filter_3d(
    image,
    radius_z,
    radius_y,
    radius_x,
    mode="reflect",
    cval=0.0,
    rank=None,
    slab_planes=None,
):
    """
    Apply the wavelet matrix rank filter with a full box footprint to a volume.

    The padded volume is the sequence of its planes, so the matrices are those
    of a 2D image with ``planes * (H + 2 * radius_y)`` rows, and the window of
    a voxel is one range of rows in each of ``2 * radius_z + 1`` planes. The
    query kernel handles these ranges together (cost proportional to the
    window depth). As in `_median_wavelet_filter`, the volume is filtered in
    slabs of planes, each with a halo of ``radius_z`` planes, when the matrices
    of the whole volume would not fit in ``WM_MEMORY_FRACTION`` of the
    available device memory; the results do not depend on the slab height.

    Parameters
    ----------
    image : cupy.ndarray
        Input 3D volume (see `_can_use_wavelet_matrix`).
    radius_z, radius_y, radius_x : int
        Filter radii along axes 0, 1 and 2.
    mode, cval :
        Boundary mode (scipy.ndimage names) and constant value.
    rank : int, optional
        Index of the selected element of the sorted window values, in
        ``[0, window_size)``. The default is the median, ``window_size // 2``.
    slab_planes : int, optional
        Number of output planes to filter at a time (default: chosen from the
        available memory).

    Returns
    -------
    result : cupy.ndarray
        Filtered volume with the same shape and dtype as the input.
    """
    footprint_shape = (2 * radius_z + 1, 2 * radius_y + 1, 2 * radius_x + 1)
    ok, reason = _can_use_wavelet_matrix(
        image, footprint_shape=footprint_shape, mode=mode
    )
    if not ok or image.ndim != 3:
        raise ValueError(f"Cannot use wavelet matrix filter: {reason}")

    window_size = math.prod(footprint_shape)
    if rank is None:
        rank = window_size // 2
    elif not 0 <= rank < window_size:
        raise ValueError(f"rank must be in [0, {window_size}), got {rank}")

    nz, height, width = image.shape
    if slab_planes is None:
        slab_planes = _plan_slab_planes(
            nz, height, width, image.dtype, radius_z, radius_y, radius_x
        )
        if slab_planes is None:
            # nothing fits the budget: try the smallest slab anyway
            slab_planes = min(nz, max(WM_MIN_SLAB_PLANES, 4 * radius_z))
        else:
            slab_planes = _plan_speed_slab_planes(
                nz,
                height,
                width,
                image.dtype,
                radius_z,
                radius_y,
                radius_x,
                slab_planes,
            )
    n_slabs = _div_ceil(nz, min(slab_planes, nz))
    slab_planes = _div_ceil(nz, n_slabs)

    pad_kwargs = {"constant_values": cval} if mode == "constant" else {}
    padded = cp.pad(
        image,
        (
            (radius_z, radius_z),
            (radius_y, radius_y),
            (radius_x, radius_x),
        ),
        mode=_to_np_mode(mode),
        **pad_kwargs,
    )
    padded_h = height + 2 * radius_y

    kernels = None
    params = buffers = None
    result = None if n_slabs == 1 else cp.empty_like(image)
    for z0 in range(0, nz, slab_planes):
        z1 = min(nz, z0 + slab_planes)
        # the planes of the padded volume are contiguous
        slab = padded[z0 : z1 + 2 * radius_z]
        if (
            params is None
            or params.height != (z1 - z0 + 2 * radius_z) * padded_h
        ):
            params = buffers = None
            params = WaveletMatrixMedianParams(
                (z1 - z0 + 2 * radius_z) * padded_h,
                width,
                image.dtype,
                0,
                radius_x,
            )
            buffers = WaveletMatrixBuffers(params)
        if kernels is None:
            kernels = _get_construction_kernels(params.val_type)

        sorted_values = None
        if params.is_rank_mode:
            values, sorted_values = _to_ranks(slab)
        else:
            values = _to_keys(slab)
        _run_wavelet_construction(values, params, buffers, kernels)

        selected = cp.empty((z1 - z0, height, width), dtype=params.matrix_dtype)
        _run_median_query_3d(
            params,
            buffers,
            selected,
            rank,
            (z1 - z0, height, width, padded_h, radius_z, radius_y, radius_x),
        )
        if params.is_rank_mode:
            filtered = sorted_values[selected]
        else:
            filtered = _from_keys(selected, image.dtype)
        if result is None:
            return filtered
        result[z0:z1] = filtered
        del values, sorted_values, selected, filtered
    return result
