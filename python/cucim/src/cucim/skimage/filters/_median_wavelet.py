# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wavelet matrix-based median filter implementation for cuCIM.

This module provides a fast GPU-based median filter using the wavelet matrix
algorithm described in:
- Sumida et al. (2022) "High-Performance 2D Median Filter using Wavelet Matrix"
  https://dl.acm.org/doi/10.1145/3550454.3555512

The wavelet matrix approach is faster than histogram-based methods, especially
for larger kernel sizes and higher bit-depth images.
"""

import os
from collections import namedtuple

import cupy as cp
import numpy as np

# =============================================================================
# Constants
# =============================================================================

# Default kernel configuration
WM_WARP_SIZE = 32
WM_WORD_SIZE = 32  # 32-bit bitvector words
WM_THREAD_PER_GRID = 512  # Threads per block for construction
WM_MAX_BLOCK_X = 1024  # Maximum grid X dimension

# Median query kernel thread configuration
WM_THREADS_W = 8  # Threads per block in X for median query
WM_THREADS_H = 64  # Threads per block in Y for median query

# CuPy compilation options for CUB
CUB_COMPILE_OPTIONS = ("--std=c++17", "-DCUB_DISABLE_BF16_SUPPORT")


# =============================================================================
# Utility Functions
# =============================================================================


def _div_ceil(a, b):
    """Integer division rounding up."""
    return (a + b - 1) // b


def _get_bit_len(val):
    """
    Compute the number of bits needed to represent a value.

    Returns ceil(log2(val + 1)).
    For val=0 returns 1, val=1 returns 1, val=255 returns 8, val=256 returns 9.
    """
    if val <= 0:
        return 1
    return val.bit_length()


def _next_power_of_2(v):
    """Round up to the next power of 2."""
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    return v


def _dtype_to_cuda_type(dtype):
    """Convert numpy/cupy dtype to CUDA type string."""
    dtype = np.dtype(dtype)
    type_map = {
        np.uint8: "unsigned char",
        np.uint16: "unsigned short",
        np.uint32: "unsigned int",
        np.uint64: "unsigned long long",
        np.int8: "signed char",
        np.int16: "short",
        np.int32: "int",
        np.int64: "long long",
        np.float16: "half",
        np.float32: "float",
        np.float64: "double",
    }
    if dtype.type not in type_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return type_map[dtype.type]


def _get_val_bit_len(dtype, total_pixels=None):
    """
    Get the number of bits for a value type.

    For small unsigned integer types (uint8, uint16, uint32), this is the bit width.
    For types using rank mode (floats, signed ints, uint64), we use ranks
    (0 to total_pixels-1), so we need ceil(log2(total_pixels)) bits.

    Parameters
    ----------
    dtype : numpy dtype
        The data type
    total_pixels : int, optional
        Total number of pixels (required for rank-mode types)

    Returns
    -------
    bit_len : int
        Number of bits needed
    """
    dtype = np.dtype(dtype)
    # For rank-based types, use ceil(log2(total_pixels))
    if _uses_rank_mode(dtype):
        if total_pixels is None:
            raise ValueError(f"total_pixels required for {dtype}")
        return _get_bit_len(total_pixels)
    return dtype.itemsize * 8


def _uses_rank_mode(dtype):
    """Check if dtype requires rank mode (sorting + rank lookup).

    This is used for dtypes where direct bit representation is impractical:
    - float16/32/64: Can't use bitwise operations for median
    - int8/16/32/64: Sign bit would cause incorrect ordering
    - uint64: 64 bit levels would be too slow, use ranks instead
    """
    dtype = np.dtype(dtype)
    # All floats need rank mode (bit patterns don't match value ordering)
    # All signed ints need rank mode (sign bit causes incorrect ordering)
    # uint64 needs rank mode (64 bit levels would be too slow)
    return dtype.kind in ("f", "i") or dtype == np.uint64


# Keep old name for backward compatibility
_is_float_mode = _uses_rank_mode


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
    Compute the number of block pairs for wavelet matrix construction.

    This determines how many elements each thread block processes during
    the wavelet matrix construction phase.

    Parameters
    ----------
    size : int
        Total number of elements (H * W, rounded up to word size multiple)
    thread_per_grid : int
        Threads per block
    max_block_x : int
        Maximum grid X dimension
    warp_size : int
        Warp size (32)

    Returns
    -------
    block_pair_num : int
        Number of block pairs
    """
    # Make pixels assigned per grid a multiple of 65536
    x_chunk = 65536 // thread_per_grid // warp_size
    if x_chunk <= 0:
        x_chunk = 1

    total_gridx = _div_ceil(size, thread_per_grid * warp_size)
    block_pair_num = _div_ceil(total_gridx, max_block_x)

    if block_pair_num <= x_chunk:
        # Round up to next power of 2
        block_pair_num = _next_power_of_2(block_pair_num)
    else:
        # Round up to multiple of x_chunk
        block_pair_num = _div_ceil(block_pair_num, x_chunk) * x_chunk

    block_pair_num *= warp_size
    return block_pair_num


def _compute_buffer_sizes(
    height,
    width,
    dtype,
    word_size=WM_WORD_SIZE,
    thread_per_grid=WM_THREAD_PER_GRID,
):
    """
    Compute buffer sizes needed for wavelet matrix construction and query.

    Parameters
    ----------
    height : int
        Image height
    width : int
        Image width
    dtype : numpy dtype
        Image data type (uint8, uint16, float32)
    word_size : int
        Bitvector word size (32 or 64)
    thread_per_grid : int
        Threads per block for construction kernels

    Returns
    -------
    BufferSizes : namedtuple
        Named tuple with all computed buffer sizes
    """
    dtype = np.dtype(dtype)
    total_pixels = height * width
    val_bit_len = _get_val_bit_len(dtype, total_pixels)
    w_bit_len = _get_bit_len(width)

    # Total size rounded up to word_size multiple
    total_pixels = height * width
    size = _div_ceil(total_pixels, word_size) * word_size

    # BlockT size: nsum (4 bytes) + nbit (4 or 8 bytes)
    block_t_size = 4 + (word_size // 8)

    # Bitvector block length
    # Formula: ceil(size / thread_per_grid) * thread_per_grid / word_size + 1
    # Then round up to multiple of 16
    bv_block_len = (
        _div_ceil(size, thread_per_grid) * thread_per_grid // word_size + 1
    )
    bv_block_len = _div_ceil(bv_block_len, 16) * 16

    # Total bitvector storage for value wavelet matrix
    # One bitvector array per bit level
    bv_block_bytes = block_t_size * val_bit_len * bv_block_len
    bv_block_bytes_div32 = _div_ceil(bv_block_bytes, 32)

    # Block pair number for construction
    block_pair_num = _get_block_pair_num(size, thread_per_grid)

    # Scan buffer length
    nsum_scan_buf_len = _div_ceil(size, thread_per_grid * block_pair_num)
    nsum_scan_buf_len = _div_ceil(nsum_scan_buf_len, 4) * 4

    # Working buffer for construction
    # - 2 * nsum_scan_buf_len uint32 for scan buffers
    # - 2 * size uint16 for index buffers
    # - 1 * size * val_bytes for value buffer (single channel)
    val_bytes = dtype.itemsize
    buf_bytes = (
        4 * 2 * nsum_scan_buf_len  # XYIdxT scan buffers
        + 2 * 2 * size  # XIdxT index buffers
        + val_bytes * size  # Value buffer
    )
    buf_bytes_div32 = _div_ceil(buf_bytes, 32)

    # Column wavelet matrix buffers (for X dimension)
    # Similar structure but with w_bit_len levels
    wm_bv_block_bytes = block_t_size * w_bit_len * bv_block_len * val_bit_len
    wm_bv_block_bytes_div32 = _div_ceil(wm_bv_block_bytes, 32)

    # Position of total count in wavelet matrix
    nsum_pos = size // word_size

    # Grid dimensions for construction
    grid_x = _div_ceil(size, thread_per_grid * block_pair_num)

    BufferSizes = namedtuple(
        "WaveletMatrixBufferSizes",
        [
            "size",  # Padded total pixels
            "val_bit_len",  # Bits per value
            "w_bit_len",  # Bits for column index
            "bv_block_len",  # Blocks per bit level
            "bv_block_bytes",  # Total bytes for value bitvectors
            "bv_block_bytes_div32",
            "wm_bv_block_bytes",  # Total bytes for column bitvectors
            "wm_bv_block_bytes_div32",
            "block_pair_num",  # Block pairs for construction
            "nsum_scan_buf_len",  # Scan buffer length
            "buf_bytes",  # Working buffer bytes
            "buf_bytes_div32",
            "nsum_pos",  # Position of total count
            "grid_x",  # Grid X for construction
            "block_t_size",  # Size of BlockT struct
        ],
    )

    return BufferSizes(
        size=size,
        val_bit_len=val_bit_len,
        w_bit_len=w_bit_len,
        bv_block_len=bv_block_len,
        bv_block_bytes=bv_block_bytes,
        bv_block_bytes_div32=bv_block_bytes_div32,
        wm_bv_block_bytes=wm_bv_block_bytes,
        wm_bv_block_bytes_div32=wm_bv_block_bytes_div32,
        block_pair_num=block_pair_num,
        nsum_scan_buf_len=nsum_scan_buf_len,
        buf_bytes=buf_bytes,
        buf_bytes_div32=buf_bytes_div32,
        nsum_pos=nsum_pos,
        grid_x=grid_x,
        block_t_size=block_t_size,
    )


def _estimate_memory_usage(height, width, dtype, radius):
    """
    Estimate total GPU memory usage for wavelet matrix median filter.

    Parameters
    ----------
    height : int
        Image height
    width : int
        Image width
    dtype : numpy dtype
        Image data type
    radius : int
        Filter radius

    Returns
    -------
    memory_bytes : int
        Estimated total GPU memory usage in bytes
    """
    # Account for padding
    padded_h = height + 2 * radius
    padded_w = width + 2 * radius

    buf_sizes = _compute_buffer_sizes(padded_h, padded_w, dtype)

    # Bitvector storage (value + column wavelet matrices)
    bv_memory = buf_sizes.bv_block_bytes_div32 * 32
    wm_memory = buf_sizes.wm_bv_block_bytes_div32 * 32

    # Working buffers
    work_memory = buf_sizes.buf_bytes_div32 * 32

    # Input and output images
    dtype = np.dtype(dtype)
    input_memory = padded_h * padded_w * dtype.itemsize
    output_memory = height * width * dtype.itemsize

    total = bv_memory + wm_memory + work_memory + input_memory + output_memory
    return total


# =============================================================================
# Kernel Preamble Generation
# =============================================================================


def _gen_wavelet_preamble(
    val_type,
    val_bit_len,
    w_bit_len,
    word_size=WM_WORD_SIZE,
    threads_w=WM_THREADS_W,
    threads_h=WM_THREADS_H,
):
    """
    Generate the CUDA preamble with #define statements for wavelet kernels.

    Parameters
    ----------
    val_type : str
        CUDA type string for values (e.g., "unsigned char")
    val_bit_len : int
        Number of bits per value
    w_bit_len : int
        Number of bits for column indices
    word_size : int
        Bitvector word size
    threads_w : int
        Threads per block in X for median query
    threads_h : int
        Threads per block in Y for median query

    Returns
    -------
    preamble : str
        CUDA code with all necessary definitions
    """
    preamble = f"""
// CUB includes
#include <cub/warp/warp_scan.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_scan.cuh>

// Wavelet matrix configuration
#define WM_WORD_SIZE {word_size}
#define WM_WARP_SIZE 32
#define WM_THREAD_PER_GRID {WM_THREAD_PER_GRID}
#define WM_THREADS_DIM_Y (WM_THREAD_PER_GRID / WM_WARP_SIZE)

// Value type configuration
#define WM_VAL_T {val_type}
#define WM_VAL_BIT_LEN {val_bit_len}
#define WM_W_BIT_LEN {w_bit_len}

// Median query thread configuration
#define WM_THREADS_W {threads_w}
#define WM_THREADS_H {threads_h}

// Index type definitions
typedef unsigned short XIdxT;
typedef unsigned short YIdxT;
typedef unsigned int XYIdxT;

// BlockT definition for {word_size}-bit words
struct __align__(8) BlockT {{
    unsigned int nsum;  // Prefix sum of set bits
    unsigned int nbit;  // Bitvector word
}};
"""
    return preamble


# =============================================================================
# Kernel Compilation
# =============================================================================


@cp.memoize(for_each_device=True)
def _get_median_query_kernel(val_type, val_bit_len, w_bit_len, padded=False):
    """
    Get the compiled median query kernel.

    Parameters
    ----------
    val_type : str
        CUDA type string for values
    val_bit_len : int
        Number of bits per value
    w_bit_len : int
        Number of bits for column indices
    padded : bool
        If True, return the padded variant

    Returns
    -------
    kernel : cp.RawKernel
        Compiled CUDA kernel
    """
    preamble = _gen_wavelet_preamble(val_type, val_bit_len, w_bit_len)

    # Read the kernel file
    kernel_dir = os.path.join(os.path.dirname(__file__), "cuda")
    kernel_path = os.path.join(kernel_dir, "wavelet_matrix_median.cu")

    with open(kernel_path) as f:
        kernel_code = f.read()

    full_code = preamble + kernel_code
    kernel_name = "wavelet_median2d_padded" if padded else "wavelet_median2d"

    return cp.RawKernel(
        code=full_code,
        name=kernel_name,
        options=CUB_COMPILE_OPTIONS,
    )


# =============================================================================
# Compatibility Check
# =============================================================================


def _can_use_wavelet_matrix(
    image, footprint_shape=None, radius=None, radius_y=None, radius_x=None
):
    """
    Check if the wavelet matrix median filter can be used.

    Parameters
    ----------
    image : cupy.ndarray
        The input image (2D, C-contiguous assumed)
    footprint_shape : tuple of int, optional
        The filter footprint shape (size_axis0, size_axis1). Both must be odd.
    radius : int, optional
        The filter radius for square footprints (alternative to footprint_shape)
    radius_y : int, optional
        The filter radius along axis 0 (rows). Alternative to footprint_shape.
    radius_x : int, optional
        The filter radius along axis 1 (columns). Alternative to footprint_shape.

    Returns
    -------
    compatible : bool
        Whether wavelet matrix can be used
    reason : str or None
        Reason for incompatibility, or None if compatible
    """
    # Check image dimensions
    if image.ndim != 2:
        return False, "Only 2D images are supported"

    # Check dtype - support unsigned ints, signed ints, and floats
    supported_dtypes = [
        cp.uint8,
        cp.uint16,
        cp.uint32,
        cp.uint64,
        cp.int8,
        cp.int16,
        cp.int32,
        cp.int64,
        cp.float16,
        cp.float32,
        cp.float64,
    ]
    if image.dtype not in supported_dtypes:
        return (
            False,
            "Unsupported dtype. Supported: uint8/16/32/64, int8/16/32/64, float16/32/64",
        )

    # Check image width (must fit in uint16 for column indices)
    if image.shape[1] >= 65535:
        return False, "Image width must be less than 65535"

    # Normalize radius parameters
    if radius is not None:
        ry, rx = radius, radius
    elif radius_y is not None and radius_x is not None:
        ry, rx = radius_y, radius_x
    elif footprint_shape is not None:
        if len(footprint_shape) != 2:
            return False, "Footprint must be 2D"
        if footprint_shape[0] % 2 == 0 or footprint_shape[1] % 2 == 0:
            return False, "Footprint dimensions must be odd"
        ry = footprint_shape[0] // 2
        rx = footprint_shape[1] // 2
    else:
        ry, rx = None, None

    # For rank-based dtypes, check total pixels (ranks stored as uint32)
    if _uses_rank_mode(image.dtype):
        # Account for padding (radius added on each side)
        if ry is not None and rx is not None:
            padded_h = image.shape[0] + 2 * ry
            padded_w = image.shape[1] + 2 * rx
        else:
            padded_h = image.shape[0]
            padded_w = image.shape[1]

        total_pixels = padded_h * padded_w
        if total_pixels > 2**32 - 1:
            return (
                False,
                f"Image too large for rank-based dtype ({image.dtype}). "
                f"Total pixels ({total_pixels:,}) exceeds uint32 max ({2**32 - 1:,})",
            )

    # Validate radius values
    if ry is not None and rx is not None:
        if ry < 1 or rx < 1:
            return False, "Radius must be at least 1"
        # Check padded dimensions
        padded_w = image.shape[1] + 2 * rx
        if padded_w >= 65535:
            return False, "Padded width exceeds maximum (65535)"

    return True, None


# =============================================================================
# Main Entry Point (placeholder - full implementation in Phase 5)
# =============================================================================


class WaveletMatrixMedianParams:
    """
    Container for wavelet matrix median filter parameters.

    This class computes and stores all the parameters needed for
    wavelet matrix construction and median queries.
    """

    def __init__(
        self, height, width, dtype, radius=None, radius_y=None, radius_x=None
    ):
        """
        Initialize parameters for wavelet matrix median filter.

        Parameters
        ----------
        height : int
            Original image height (before padding), i.e., image.shape[0]
        width : int
            Original image width (before padding), i.e., image.shape[1]
        dtype : numpy dtype
            Image data type
        radius : int, optional
            Filter radius for square footprints
        radius_y : int, optional
            Filter radius along axis 0 (rows/height)
        radius_x : int, optional
            Filter radius along axis 1 (columns/width)
        """
        self.height = height
        self.width = width
        self.dtype = np.dtype(dtype)

        # Handle radius parameters
        if radius is not None:
            self.radius_y = radius
            self.radius_x = radius
        elif radius_y is not None and radius_x is not None:
            self.radius_y = radius_y
            self.radius_x = radius_x
        else:
            raise ValueError(
                "Must provide either radius or both radius_y and radius_x"
            )

        # Keep single radius for backward compatibility (square case)
        self.radius = self.radius_y if self.radius_y == self.radius_x else None

        # Padded dimensions
        self.padded_h = height + 2 * self.radius_y
        self.padded_w = width + 2 * self.radius_x

        # Float mode: use sorted ranks instead of direct values
        self.is_float_mode = _is_float_mode(self.dtype)

        # Compute buffer sizes
        self.buf_sizes = _compute_buffer_sizes(
            self.padded_h, self.padded_w, self.dtype
        )

        # Store commonly used values
        # For float mode, we use uint32 ranks as the "value type"
        if self.is_float_mode:
            self.val_type = "unsigned int"  # ranks are uint32
            self.rank_dtype = cp.uint32
        else:
            self.val_type = _dtype_to_cuda_type(self.dtype)
            self.rank_dtype = None

        self.val_bit_len = self.buf_sizes.val_bit_len
        self.w_bit_len = self.buf_sizes.w_bit_len
        self.nsum_pos = self.buf_sizes.nsum_pos
        self.bv_block_len = self.buf_sizes.bv_block_len

        # Compute bv_block_h_byte_div32 (bytes per bit level / 32)
        block_t_size = self.buf_sizes.block_t_size
        self.bv_block_h_byte_div32 = (self.bv_block_len * block_t_size) // 32

    def get_median_grid_block(self):
        """Get grid and block dimensions for median query kernel."""
        grid = (
            _div_ceil(self.width, WM_THREADS_W),
            _div_ceil(self.height, WM_THREADS_H),
            1,
        )
        block = (WM_THREADS_W, WM_THREADS_H, 1)
        return grid, block

    def estimate_memory(self):
        """Estimate total GPU memory usage."""
        return _estimate_memory_usage(
            self.height, self.width, self.dtype, self.radius
        )

    def get_construct_grid_block(self):
        """Get grid and block dimensions for construction kernels."""
        grid = (self.buf_sizes.grid_x, 1, 1)
        block = (WM_WARP_SIZE, WM_THREAD_PER_GRID // WM_WARP_SIZE, 1)
        return grid, block


# =============================================================================
# Construction Kernel Compilation
# =============================================================================


def _gen_construct_preamble(
    val_type, val_bit_len, w_bit_len, word_size=WM_WORD_SIZE
):
    """
    Generate the CUDA preamble for construction kernels.

    Parameters
    ----------
    val_type : str
        CUDA type string for values
    val_bit_len : int
        Number of bits per value
    w_bit_len : int
        Number of bits for column indices
    word_size : int
        Bitvector word size

    Returns
    -------
    preamble : str
        CUDA code with definitions for construction kernels
    """
    return f"""
#define WM_VAL_T {val_type}
#define WM_VAL_BIT_LEN {val_bit_len}
#define WM_W_BIT_LEN {w_bit_len}
#define WM_WORD_SIZE {word_size}
#define WM_THREAD_PER_GRID {WM_THREAD_PER_GRID}
#define WM_MAX_BLOCK_X {WM_MAX_BLOCK_X}
#define WM_SRC_CACHE_DIV 2
"""


@cp.memoize(for_each_device=True)
def _get_construction_kernels(val_type, val_bit_len, w_bit_len):
    """
    Get all compiled construction kernels.

    Parameters
    ----------
    val_type : str
        CUDA type string for values
    val_bit_len : int
        Number of bits per value
    w_bit_len : int
        Number of bits for column indices

    Returns
    -------
    kernels : dict
        Dictionary of kernel name -> cp.RawKernel
    """
    preamble = _gen_construct_preamble(val_type, val_bit_len, w_bit_len)

    # Read the common header and construction kernel file
    kernel_dir = os.path.join(os.path.dirname(__file__), "cuda")
    common_path = os.path.join(kernel_dir, "wavelet_matrix_common.cuh")
    construct_path = os.path.join(kernel_dir, "wavelet_matrix_construct.cu")

    with open(common_path) as f:
        common_code = f.read()

    with open(construct_path) as f:
        construct_code = f.read()

    full_code = preamble + common_code + construct_code

    kernel_names = [
        "wavelet_first_pass",
        "wavelet_upsweep",
        "wavelet_exclusive_sum",
        "wavelet_last_pass",
        "wavelet_upsweep_wm",
        "wavelet_wm_first_pass",
        "wavelet_wm_upsweep",
    ]

    kernels = {}
    for name in kernel_names:
        kernels[name] = cp.RawKernel(
            code=full_code,
            name=name,
            options=CUB_COMPILE_OPTIONS,
        )

    return kernels


# =============================================================================
# Buffer Allocation
# =============================================================================


class WaveletMatrixBuffers:
    """
    Container for all GPU buffers used in wavelet matrix construction.

    This class allocates and manages the GPU memory needed for:
    - Value bitvector storage (bv_blocks)
    - Column wavelet matrix storage (wm_blocks)
    - Working buffers for construction
    """

    def __init__(self, params):
        """
        Allocate GPU buffers for wavelet matrix construction.

        Parameters
        ----------
        params : WaveletMatrixMedianParams
            Parameters object with computed buffer sizes
        """
        self.params = params
        buf = params.buf_sizes

        # Value bitvector storage
        # Shape: (val_bit_len, bv_block_len) of BlockT structs
        bv_bytes = buf.bv_block_bytes_div32 * 32
        self.bv_blocks = cp.zeros(bv_bytes, dtype=cp.uint8)

        # Column wavelet matrix bitvector storage
        wm_bytes = buf.wm_bv_block_bytes_div32 * 32
        self.wm_blocks = cp.zeros(wm_bytes, dtype=cp.uint8)

        # Working buffer for construction
        work_bytes = buf.buf_bytes_div32 * 32
        self.work_buffer = cp.zeros(work_bytes, dtype=cp.uint8)

        # Scan buffers (need two for alternating)
        self.nsum_scan_buf = cp.zeros(buf.nsum_scan_buf_len, dtype=cp.uint32)
        self.nsum_scan_buf2 = cp.zeros(buf.nsum_scan_buf_len, dtype=cp.uint32)

        # Index buffers (XIdxT = uint16)
        self.idx_buf1 = cp.zeros(buf.size, dtype=cp.uint16)
        self.idx_buf2 = cp.zeros(buf.size, dtype=cp.uint16)

        # Value buffers
        # For rank mode (float32, uint64), we use uint32 ranks instead of values
        if params.is_float_mode:
            self.val_buf1 = cp.zeros(buf.size, dtype=cp.uint32)
            self.val_buf2 = cp.zeros(buf.size, dtype=cp.uint32)
            # Store sorted values for final lookup (use original dtype)
            self.sorted_values = cp.zeros(buf.size, dtype=params.dtype)
        else:
            self.val_buf1 = cp.zeros(buf.size, dtype=params.dtype)
            self.val_buf2 = cp.zeros(buf.size, dtype=params.dtype)
            self.sorted_values = None

        # Working buffers for column WM construction
        # Memory optimization: Instead of storing all val_bit_len levels of
        # sorted column indices (wm_idx_levels), we build each column WM
        # immediately after the corresponding value level upsweep.
        # We only need 2 working buffers instead of val_bit_len + 2.
        self.wm_idx_buf1 = cp.zeros(buf.size, dtype=cp.uint16)
        self.wm_idx_buf2 = cp.zeros(buf.size, dtype=cp.uint16)
        self.wm_scan_buf = cp.zeros(buf.nsum_scan_buf_len, dtype=cp.uint32)
        self.wm_scan_buf2 = cp.zeros(buf.nsum_scan_buf_len, dtype=cp.uint32)

    def get_bv_level_ptr(self, level):
        """Get pointer to bitvector storage for a specific bit level."""
        block_t_size = self.params.buf_sizes.block_t_size
        offset = level * self.params.bv_block_len * block_t_size
        return self.bv_blocks[offset:]

    def get_wm_level_ptr(self, val_level, wm_level):
        """Get pointer to column WM storage for specific levels."""
        block_t_size = self.params.buf_sizes.block_t_size
        w_bit_len = self.params.w_bit_len
        bv_block_len = self.params.bv_block_len
        level_idx = val_level * w_bit_len + wm_level
        offset = level_idx * bv_block_len * block_t_size
        return self.wm_blocks[offset:]


# =============================================================================
# Construction Orchestration
# =============================================================================


def _run_wavelet_construction(src_padded, params, buffers, kernels):
    """
    Run the wavelet matrix construction pipeline.

    This function orchestrates the multi-pass construction of the wavelet matrix
    1. First pass: Initialize indices and count MSB zeros
    2. Exclusive sum: Compute prefix sums of block counts
    3. For each value bit (MSB to LSB):
       a. Up-sweep: Build bitvector and reorder values
       b. Exclusive sum: Update prefix sums
    4. For each column bit:
       a. Up-sweep WM: Build column wavelet matrix
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
    # w_bit_len used for column wavelet matrix (future expansion)
    # w_bit_len = params.w_bit_len

    # Create a secondary stream for overlapping buffer clears with kernel execution
    stream_aux = cp.cuda.Stream(non_blocking=True)

    # Flatten source if needed
    src_flat = src_padded.ravel()

    # Fill buffer with max value as sentinel (so padding has all bits = 1)
    # This ensures padding elements don't affect the wavelet matrix construction
    # For float mode, we use ranks which only need val_bit_len bits
    if params.is_float_mode:
        # For float mode, max_val should be >= max_rank but within val_bit_len bits
        # The max rank is total_pixels - 1, so we use 2^val_bit_len - 1
        max_val = (1 << params.val_bit_len) - 1
    else:
        max_val = np.iinfo(params.dtype).max
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
    first_pass_mask_dtype = (
        np.dtype(np.uint32) if params.is_float_mode else params.dtype
    )

    kernels["wavelet_first_pass"](
        grid,
        block,
        (
            first_pass_mask_dtype.type(msb_mask),  # mask
            np.uint16(buf.block_pair_num),  # block_pair_num
            np.uint32(size_div_warp),  # size_div_warp
            buffers.val_buf1,  # src
            buffers.nsum_scan_buf,  # nsum_scan_buf
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
    bv_block_h_bytes = params.bv_block_len * buf.block_t_size
    msb_bv_offset = (val_bit_len - 1) * bv_block_h_bytes
    bv_ptr = buffers.bv_blocks[msb_bv_offset:]

    kernels["wavelet_exclusive_sum"](
        exsum_grid,
        exsum_block,
        (
            buffers.nsum_scan_buf,  # nsum_scan_buf
            buffers.nsum_scan_buf2,  # nsum_buf_test2
            bv_ptr,  # nsum_p (stores total at nsum_pos)
            np.uint32(buf.buf_bytes_div32),  # buf_byte_div32
            np.uint32(buf.bv_block_bytes_div32),  # bv_block_byte_div32
            np.uint32(buf.nsum_pos),  # nsum_pos
        ),
    )

    # -------------------------------------------------------------------------
    # Step 3: Up-sweep for each value bit level
    # -------------------------------------------------------------------------
    size_div_w = buf.size // WM_WORD_SIZE
    block_t_size = buf.block_t_size
    bv_block_h_bytes = params.bv_block_len * block_t_size
    w_bit_len = params.w_bit_len

    # Alternating buffers
    src_val = buffers.val_buf1
    dst_val = buffers.val_buf2
    src_idx = buffers.idx_buf1
    dst_idx = buffers.idx_buf2

    # Compute bytes per column WM bit level (same as value WM)
    wm_block_h_bytes = params.bv_block_len * block_t_size

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
        mask_dtype = (
            np.dtype(np.uint32) if params.is_float_mode else params.dtype
        )

        kernels["wavelet_upsweep"](
            grid,
            block,
            (
                mask_dtype.type(mask),  # mask
                np.uint16(buf.block_pair_num),  # block_pair_num
                np.uint32(size_div_w),  # size_div_w
                src_val,  # src
                dst_val,  # dst
                buffers.bv_blocks[bv_curr_offset:],  # nbit_bp
                buffers.nsum_scan_buf,  # nsum_buf_test
                buffers.nsum_scan_buf2,  # nsum_buf_test2
                np.uint32(buf.bv_block_bytes_div32),  # bv_block_byte_div32
                np.uint32(buf.buf_bytes_div32),  # buf_byte_div32
                src_idx,  # idx_p
                dst_idx,  # nxt_idx
                buffers.bv_blocks[bv_prev_offset:],  # nbit_bp_pre
                np.int32(is_last),  # is_last_val_bit
            ),
        )

        # ---------------------------------------------------------------------
        # Build column WM for value level h immediately (memory optimization)
        # ---------------------------------------------------------------------
        # Copy sorted column indices to working buffer before swap
        # The column WM for value level h uses indices sorted by bits >= h
        buffers.wm_idx_buf1[:] = dst_idx[:]

        # Swap value buffers for next iteration
        src_val, dst_val = dst_val, src_val
        src_idx, dst_idx = dst_idx, src_idx

        # Exclusive sum for next value level (skip for last)
        if not is_last:
            # Swap scan buffers
            buffers.nsum_scan_buf, buffers.nsum_scan_buf2 = (
                buffers.nsum_scan_buf2,
                buffers.nsum_scan_buf,
            )

            next_bv_offset = (h - 1) * bv_block_h_bytes

            # Start clearing column WM scan buffers in auxiliary stream
            # This can overlap with the value exclusive_sum kernel
            with stream_aux:
                buffers.wm_scan_buf.fill(0)
                buffers.wm_scan_buf2.fill(0)

            kernels["wavelet_exclusive_sum"](
                exsum_grid,
                exsum_block,
                (
                    buffers.nsum_scan_buf,
                    buffers.nsum_scan_buf2,
                    buffers.bv_blocks[next_bv_offset:],
                    np.uint32(buf.buf_bytes_div32),
                    np.uint32(buf.bv_block_bytes_div32),
                    np.uint32(buf.nsum_pos),  # nsum_pos
                ),
            )

            # Sync auxiliary stream before column WM construction
            stream_aux.synchronize()
        else:
            # For last value level, just clear buffers synchronously
            buffers.wm_scan_buf.fill(0)
            buffers.wm_scan_buf2.fill(0)

        # ---------------------------------------------------------------------
        # Column WM construction for value level h
        # ---------------------------------------------------------------------

        # MSB mask for column indices
        wm_msb_mask = (1 << (w_bit_len - 1)) - 1

        # First pass: count zeros for MSB
        kernels["wavelet_wm_first_pass"](
            grid,
            block,
            (
                np.uint16(wm_msb_mask),  # mask
                np.uint16(buf.block_pair_num),  # block_pair_num
                np.uint32(size_div_w),  # size_div_warp (actually size_div_w)
                buffers.wm_idx_buf1,  # src
                buffers.wm_scan_buf,  # nsum_scan_buf
                np.uint32(buf.buf_bytes_div32),  # buf_byte_div32
            ),
        )

        # Compute offset for this value level's column WM
        wm_base_offset = h * w_bit_len * wm_block_h_bytes

        # Exclusive sum for first column WM level (MSB = w_bit_len - 1)
        kernels["wavelet_exclusive_sum"](
            exsum_grid,
            exsum_block,
            (
                buffers.wm_scan_buf,
                buffers.wm_scan_buf2,
                buffers.wm_blocks[
                    wm_base_offset + (w_bit_len - 1) * wm_block_h_bytes :
                ],
                np.uint32(buf.buf_bytes_div32),
                np.uint32(buf.wm_bv_block_bytes_div32),
                np.uint32(buf.nsum_pos),  # nsum_pos
            ),
        )

        # Alternating buffers for column WM construction
        wm_src = buffers.wm_idx_buf1
        wm_dst = buffers.wm_idx_buf2

        # Up-sweep for each column bit level
        for wm_h in range(w_bit_len - 1, -1, -1):
            is_last_wm = 1 if wm_h == 0 else 0

            # Current mask for column bit
            wm_bit_pos = w_bit_len - 1 - wm_h
            wm_mask = (1 << (w_bit_len - wm_bit_pos - 1)) - 1

            # Compute offsets for current and previous column WM levels
            # OpenCV convention: level wm_h at offset wm_h * wm_block_h_bytes
            wm_curr_offset = wm_base_offset + wm_h * wm_block_h_bytes
            prev_wm_level = min(w_bit_len - 1, wm_h + 1)
            wm_prev_offset = wm_base_offset + prev_wm_level * wm_block_h_bytes

            kernels["wavelet_wm_upsweep"](
                grid,
                block,
                (
                    np.uint16(wm_mask),  # mask
                    np.uint16(buf.block_pair_num),  # block_pair_num
                    np.uint32(size_div_w),  # size_div_w
                    wm_src,  # src
                    wm_dst,  # dst
                    buffers.wm_blocks[wm_curr_offset:],  # nbit_bp
                    buffers.wm_scan_buf,  # nsum_buf_test
                    buffers.wm_scan_buf2,  # nsum_buf_test2
                    np.uint32(buf.wm_bv_block_bytes_div32),  # bv_block_div32
                    np.uint32(buf.buf_bytes_div32),  # buf_byte_div32
                    buffers.wm_blocks[wm_prev_offset:],  # nbit_bp_pre
                    np.int32(is_last_wm),  # is_last_bit
                ),
            )

            # Swap buffers
            wm_src, wm_dst = wm_dst, wm_src

            # Exclusive sum for next level (skip for last)
            if not is_last_wm:
                buffers.wm_scan_buf, buffers.wm_scan_buf2 = (
                    buffers.wm_scan_buf2,
                    buffers.wm_scan_buf,
                )

                next_level = wm_h - 1
                next_wm_offset = wm_base_offset + next_level * wm_block_h_bytes
                kernels["wavelet_exclusive_sum"](
                    exsum_grid,
                    exsum_block,
                    (
                        buffers.wm_scan_buf,
                        buffers.wm_scan_buf2,
                        buffers.wm_blocks[next_wm_offset:],
                        np.uint32(buf.buf_bytes_div32),
                        np.uint32(buf.wm_bv_block_bytes_div32),
                        np.uint32(buf.nsum_pos),  # nsum_pos
                    ),
                )

    return buffers


# =============================================================================
# Median Query Execution
# =============================================================================


def _run_median_query(params, buffers, output, use_padded_kernel=True):
    """
    Run the wavelet matrix median query kernel.

    This function executes the median query kernel using the pre-constructed
    wavelet matrices stored in buffers.

    Parameters
    ----------
    params : WaveletMatrixMedianParams
        Parameters object with image dimensions and configuration
    buffers : WaveletMatrixBuffers
        Buffers containing constructed wavelet matrices
    output : cupy.ndarray
        Output array for median values (shape: original image size)
    use_padded_kernel : bool
        If True, use the padded variant (assumes input was padded)

    Returns
    -------
    output : cupy.ndarray
        The median-filtered result
    """
    # Get the compiled kernel
    kernel = _get_median_query_kernel(
        params.val_type,
        params.val_bit_len,
        params.w_bit_len,
        padded=use_padded_kernel,
    )

    # Compute grid and block dimensions
    grid, block = params.get_median_grid_block()

    # Get buffer sizes
    buf = params.buf_sizes
    block_t_size = buf.block_t_size
    bv_block_h_bytes = params.bv_block_len * block_t_size

    # Compute bv_block_h_byte_div32
    bv_block_h_byte_div32 = bv_block_h_bytes // 32

    # Get pointers to the MSB level of each wavelet matrix
    # Value WM: Start at level (val_bit_len - 1)
    bv_start_offset = (params.val_bit_len - 1) * bv_block_h_bytes

    # Column WM: Start at level (w_bit_len - 1) of value level (val_bit_len - 1)
    wm_start_offset = (
        (params.val_bit_len - 1) * params.w_bit_len + (params.w_bit_len - 1)
    ) * bv_block_h_bytes

    if use_padded_kernel:
        # Padded kernel: input is padded, output is original size
        kernel(
            grid,
            block,
            (
                np.int32(params.height),  # H (output height)
                np.int32(params.width),  # W (output width)
                np.int32(params.padded_w),  # padded_W
                np.int32(params.width),  # res_step_num
                np.int32(params.radius_y),  # ry
                np.int32(params.radius_x),  # rx
                output,  # res_cu
                buffers.wm_blocks[wm_start_offset:],  # wm_nbit_bp
                np.uint32(buf.nsum_pos),  # nsum_pos
                np.uint32(bv_block_h_byte_div32),  # bv_block_h_byte_div32
                np.uint32(params.bv_block_len),  # bv_block_len
                buffers.bv_blocks[bv_start_offset:],  # bv_nbit_bp
                np.uint8(params.w_bit_len),  # w_bit_len
                np.uint8(params.val_bit_len),  # val_bit_len
            ),
        )
    else:
        # Non-padded kernel: handles boundaries internally
        kernel(
            grid,
            block,
            (
                np.int32(params.height),  # H
                np.int32(params.width),  # W
                np.int32(params.width),  # res_step_num
                np.int32(params.radius_y),  # ry
                np.int32(params.radius_x),  # rx
                output,  # res_cu
                buffers.wm_blocks[wm_start_offset:],  # wm_nbit_bp
                np.uint32(buf.nsum_pos),  # nsum_pos
                np.uint32(bv_block_h_byte_div32),  # bv_block_h_byte_div32
                np.uint32(params.bv_block_len),  # bv_block_len
                buffers.bv_blocks[bv_start_offset:],  # bv_nbit_bp
                np.uint8(params.w_bit_len),  # w_bit_len
                np.uint8(params.val_bit_len),  # val_bit_len
            ),
        )

    return output


# =============================================================================
# Full Median Filter Pipeline
# =============================================================================


def _prepare_float_ranks(padded, params, buffers):
    """
    Prepare rank values for float32 images.

    For float32, we sort the values and use their ranks (0 to n-1) as the
    "values" for the wavelet matrix. The wavelet matrix then operates on
    integers, and the median rank is looked up in the sorted values array
    to get the actual float result.

    Parameters
    ----------
    padded : cupy.ndarray
        Padded float32 image
    params : WaveletMatrixMedianParams
        Filter parameters
    buffers : WaveletMatrixBuffers
        Pre-allocated buffers

    Returns
    -------
    ranks : cupy.ndarray
        uint32 array of ranks with same shape as padded
    """
    # Flatten and sort
    flat = padded.ravel()
    sorted_indices = cp.argsort(flat)

    # Store sorted values for later lookup
    buffers.sorted_values[: len(flat)] = flat[sorted_indices]

    # Create rank array: rank[original_idx] = sorted_position
    ranks = cp.empty_like(flat, dtype=cp.uint32)
    ranks[sorted_indices] = cp.arange(len(flat), dtype=cp.uint32)

    return ranks.reshape(padded.shape)


def _lookup_float_results(rank_output, buffers):
    """
    Convert rank output back to float values.

    Parameters
    ----------
    rank_output : cupy.ndarray
        Output array containing median ranks (uint32)
    buffers : WaveletMatrixBuffers
        Buffers containing sorted values

    Returns
    -------
    float_output : cupy.ndarray
        Float32 array with actual median values
    """
    flat_ranks = rank_output.ravel()
    float_output = buffers.sorted_values[flat_ranks]
    return float_output.reshape(rank_output.shape)


def _median_wavelet_filter(
    image, radius=None, radius_y=None, radius_x=None, mode="reflect"
):
    """
    Apply wavelet matrix median filter to a 2D image.

    This function performs the complete wavelet matrix median filtering:
    1. Pad the input image
    2. For float32: sort values and prepare ranks
    3. Construct wavelet matrices (value WM + column WMs)
    4. Run median query kernel
    5. For float32: look up actual values from sorted array
    6. Return the filtered result

    Parameters
    ----------
    image : cupy.ndarray
        Input 2D image (uint8, uint16, or float32). Must be C-contiguous or
        will be copied to a contiguous array.
    radius : int, optional
        Filter radius for square footprints (kernel size = 2*radius + 1)
    radius_y : int, optional
        Filter radius along axis 0 (rows/height). Kernel height = 2*radius_y + 1.
    radius_x : int, optional
        Filter radius along axis 1 (columns/width). Kernel width = 2*radius_x + 1.
    mode : str
        Padding mode for boundary handling. Options:
        - 'reflect': Symmetric reflection (d c b a | a b c d | d c b a)
        - 'constant': Pad with constant value (zeros)
        - 'nearest': Pad with nearest edge value
        - 'wrap': Circular wrap around

    Returns
    -------
    result : cupy.ndarray
        Median-filtered image with same shape and dtype as input

    Notes
    -----
    The CUDA kernels assume C-contiguous (row-major) memory layout where
    axis 0 corresponds to rows (Y direction) and axis 1 corresponds to
    columns (X direction).
    """
    # Normalize radius parameters
    if radius is not None:
        ry, rx = radius, radius
    elif radius_y is not None and radius_x is not None:
        ry, rx = radius_y, radius_x
    else:
        raise ValueError(
            "Must provide either radius or both radius_y and radius_x"
        )

    # Validate input
    ok, reason = _can_use_wavelet_matrix(image, radius_y=ry, radius_x=rx)
    if not ok:
        raise ValueError(f"Cannot use wavelet matrix filter: {reason}")

    # Ensure C-contiguous layout (kernels assume row-major order)
    image = cp.ascontiguousarray(image)

    height, width = image.shape
    dtype = image.dtype

    # Create parameters
    params = WaveletMatrixMedianParams(
        height, width, dtype, radius_y=ry, radius_x=rx
    )

    # Allocate buffers
    buffers = WaveletMatrixBuffers(params)

    # Get construction kernels
    construction_kernels = _get_construction_kernels(
        params.val_type, params.val_bit_len, params.w_bit_len
    )

    # Pad the input image with asymmetric padding for rectangular footprints
    # Note: scipy's 'reflect' = cupy's 'symmetric' (edge is included)
    pad_mode = "symmetric" if mode == "reflect" else mode
    pad_width = ((ry, ry), (rx, rx))
    padded = cp.pad(image, pad_width, mode=pad_mode)

    # For float mode, convert to ranks
    if params.is_float_mode:
        padded_values = _prepare_float_ranks(padded, params, buffers)
    else:
        padded_values = padded

    # Run construction
    _run_wavelet_construction(
        padded_values, params, buffers, construction_kernels
    )

    # Allocate output
    # For float mode, query returns ranks (uint32), then we look up values
    if params.is_float_mode:
        rank_output = cp.empty((height, width), dtype=cp.uint32)
        _run_median_query(params, buffers, rank_output, use_padded_kernel=True)
        output = _lookup_float_results(rank_output, buffers)
    else:
        output = cp.empty((height, width), dtype=dtype)
        _run_median_query(params, buffers, output, use_padded_kernel=True)

    return output
