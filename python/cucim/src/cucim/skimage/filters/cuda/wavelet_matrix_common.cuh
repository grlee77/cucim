/*
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 *
 * Definitions shared by the wavelet matrix construction and query kernels.
 *
 * The Python code in ../_median_wavelet.py defines these macros before this
 * file is included:
 *   WM_VAL_T            value type stored in the matrix (unsigned integer)
 *   WM_THREAD_PER_GRID  threads per block of the construction kernels
 *   WM_MAX_BLOCK_X      maximum number of blocks in the construction grid
 *   WM_SRC_CACHE_DIV    shared-memory caching factor of the up-sweep
 *   WM_THREADS_W/H      block shape of the query kernel
 *
 * See wavelet_matrix_median.cu for references.
 */

#ifndef CUCIM_WAVELET_MATRIX_COMMON_CUH
#define CUCIM_WAVELET_MATRIX_COMMON_CUH

#define WM_WARP_SIZE 32
// Bits per bit-vector word (BlockT::nbit)
#define WM_WORD_SIZE 32
#define WM_THREADS_DIM_Y (WM_THREAD_PER_GRID / WM_WARP_SIZE)
#define WM_WORD_DIV_WARP (WM_WORD_SIZE / WM_WARP_SIZE)
#define FULL_WARP_MASK 0xFFFFFFFF

// Column index within a row (images must be narrower than 65535 pixels)
typedef unsigned short XIdxT;
// Linear pixel index and bit-vector rank/count (at most 2^32 - 1 pixels)
typedef unsigned int XYIdxT;

/*
 * One 32-bit word of a bit vector plus the number of set bits before it, so
 * that rank queries need a single 8-byte load.
 *
 * Bit j of `nbit` is 1 when element j has the current bit CLEAR (so the set
 * bits count the elements that go to the 0 side of the wavelet matrix).
 */
struct __align__(8) BlockT {
    unsigned int nsum;
    unsigned int nbit;
};

/*
 * Number of set bits of the bit vector in positions [0, i), i.e. the number of
 * elements before position i whose current bit is clear.
 */
__device__ __forceinline__ XYIdxT wavelet_rank0(
    const XYIdxT i,
    const BlockT* __restrict__ nbit_bp
) {
    const BlockT block = nbit_bp[i / WM_WORD_SIZE];
    return block.nsum + __popc(block.nbit & ((1u << (i % WM_WORD_SIZE)) - 1));
}

#endif  // CUCIM_WAVELET_MATRIX_COMMON_CUH
