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
        np.int8: "signed char",
        np.int16: "short",
        np.int32: "int",
    }
    if dtype.type not in type_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return type_map[dtype.type]


def _get_val_bit_len(dtype):
    """Get the number of bits for a value type."""
    dtype = np.dtype(dtype)
    return dtype.itemsize * 8


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
        Image data type (uint8, uint16, or uint32)
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
    val_bit_len = _get_val_bit_len(dtype)
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


def _can_use_wavelet_matrix(image, footprint_shape=None, radius=None):
    """
    Check if the wavelet matrix median filter can be used.

    Parameters
    ----------
    image : cupy.ndarray
        The input image
    footprint_shape : tuple of int, optional
        The filter footprint shape
    radius : int, optional
        The filter radius (alternative to footprint_shape)

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

    # Check dtype
    if image.dtype not in [cp.uint8, cp.uint16]:
        return False, "Only uint8 and uint16 dtypes are supported"

    # Check image width (must fit in uint16)
    if image.shape[1] >= 65535:
        return False, "Image width must be less than 65535"

    # Check footprint
    if footprint_shape is not None:
        if len(footprint_shape) != 2:
            return False, "Footprint must be 2D"
        if footprint_shape[0] != footprint_shape[1]:
            return False, "Only square footprints are supported"
        if footprint_shape[0] % 2 == 0:
            return False, "Footprint size must be odd"
        radius = footprint_shape[0] // 2

    if radius is not None:
        if radius < 1:
            return False, "Radius must be at least 1"
        # Check padded dimensions
        padded_w = image.shape[1] + 2 * radius
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

    def __init__(self, height, width, dtype, radius):
        """
        Initialize parameters for wavelet matrix median filter.

        Parameters
        ----------
        height : int
            Original image height (before padding)
        width : int
            Original image width (before padding)
        dtype : numpy dtype
            Image data type
        radius : int
            Filter radius
        """
        self.height = height
        self.width = width
        self.dtype = np.dtype(dtype)
        self.radius = radius

        # Padded dimensions
        self.padded_h = height + 2 * radius
        self.padded_w = width + 2 * radius

        # Compute buffer sizes
        self.buf_sizes = _compute_buffer_sizes(
            self.padded_h, self.padded_w, self.dtype
        )

        # Store commonly used values
        self.val_type = _dtype_to_cuda_type(self.dtype)
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
