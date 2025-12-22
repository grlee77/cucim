/*
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 *
 * Common definitions and utilities for the wavelet matrix median filter.
 *
 * This header provides:
 * - CUB library includes
 * - Utility macros and functions
 * - Block type definitions for bitvector storage
 * - Common device functions used across kernels
 *
 * Based on the wavelet matrix 2D median algorithm described in:
 * - Sumida et al. (2022) "High-Performance 2D Median Filter using Wavelet Matrix"
 *   https://dl.acm.org/doi/10.1145/3550454.3555512
 * - Adams (2021) "Fast Median Filters Using Separable Sorting Networks"
 *   https://dl.acm.org/doi/10.1145/3450626.3459773
 */

#ifndef CUCIM_WAVELET_MATRIX_COMMON_CUH
#define CUCIM_WAVELET_MATRIX_COMMON_CUH

// CUB library includes for scan and reduce operations
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

// ============================================================================
// Compile-time constants (defined via Python preamble)
// ============================================================================

// These will be defined by the Python code when generating the kernel:
// - VAL_T: Value type (uint8_t, uint16_t, or uint32_t for float proxy)
// - IDX_T: Index type (uint16_t for X index, uint32_t for XY index)
// - WORD_SIZE: Bit width for bitvector words (32 or 64)
// - WARP_SIZE: GPU warp size (32)
// - THREAD_PER_GRID: Threads per block
// - VAL_BIT_LEN: Number of bits in value type
// - W_BIT_LEN: Number of bits needed to represent image width

// ============================================================================
// Utility macros
// ============================================================================

// Integer division rounding up
#ifndef DIV_CEIL
#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))
#endif

// Alignment macro for struct definitions
#ifndef ALIGN
#define ALIGN(n) __align__(n)
#endif

// ============================================================================
// Block type for bitvector storage
// ============================================================================

/*
 * BlockT stores a bitvector block with:
 * - nsum: Cumulative count of set bits (prefix sum) up to this block
 * - nbit: The actual bitvector data for this block
 *
 * For WORD_SIZE=32:
 *   sizeof(BlockT) = 8 bytes (4 + 4, with alignment)
 *
 * For WORD_SIZE=64:
 *   sizeof(BlockT) = 12 bytes (4 + 8, with padding)
 */

#if WORD_SIZE == 32

struct ALIGN(8) BlockT {
    unsigned int nsum;  // Prefix sum of set bits
    unsigned int nbit;  // Bitvector word (32 bits)
};

// Population count for 32-bit word
__device__ __forceinline__ int block_popc(unsigned int bits) {
    return __popc(bits);
}

// Mask for extracting lower bits
__device__ __forceinline__ unsigned int block_mask(int bit_pos) {
    return (1u << bit_pos) - 1;
}

#elif WORD_SIZE == 64

struct ALIGN(8) BlockT {
    unsigned int nsum;       // Prefix sum of set bits
    unsigned long long nbit; // Bitvector word (64 bits)
};

// Population count for 64-bit word
__device__ __forceinline__ int block_popc(unsigned long long bits) {
    return __popcll(bits);
}

// Mask for extracting lower bits
__device__ __forceinline__ unsigned long long block_mask(int bit_pos) {
    return (1ull << bit_pos) - 1ull;
}

#endif  // WORD_SIZE

// ============================================================================
// Rank query function
// ============================================================================

/*
 * Compute rank0(i) - the count of 0-bits in the bitvector up to position i.
 *
 * This is the core operation for wavelet matrix queries.
 * rank0(i) = i - rank1(i), where rank1(i) is the count of 1-bits.
 *
 * Parameters:
 *   i: Position in the bitvector (0-indexed)
 *   nbit_bp: Array of BlockT structures containing the bitvector
 *
 * Returns:
 *   Number of 0-bits in positions [0, i)
 */
template <typename IdxType>
__device__ __forceinline__ IdxType wavelet_rank0(
    const IdxType i,
    const BlockT* __restrict__ nbit_bp
) {
    const IdxType bi = i / WORD_SIZE;  // Block index
    const int ai = i % WORD_SIZE;       // Position within block

    const BlockT block = nbit_bp[bi];

    // rank1 = nsum (prefix sum) + popcount of bits below position ai
    IdxType rank1 = block.nsum + block_popc(block.nbit & block_mask(ai));

    // rank0 = i - rank1
    return i - rank1;
}

// Alternate version that returns rank1 (count of 1-bits)
template <typename IdxType>
__device__ __forceinline__ IdxType wavelet_rank1(
    const IdxType i,
    const BlockT* __restrict__ nbit_bp
) {
    const IdxType bi = i / WORD_SIZE;
    const int ai = i % WORD_SIZE;

    const BlockT block = nbit_bp[bi];
    return block.nsum + block_popc(block.nbit & block_mask(ai));
}

// ============================================================================
// Warp-level utilities
// ============================================================================

// Get lane ID within warp
__device__ __forceinline__ int lane_id() {
    return threadIdx.x % WARP_SIZE;
}

// Get warp ID within block
__device__ __forceinline__ int warp_id() {
    return threadIdx.x / WARP_SIZE;
}

// Full warp mask
#define FULL_WARP_MASK 0xFFFFFFFF

// Ballot to create bitvector from predicate
__device__ __forceinline__ unsigned int warp_ballot(bool predicate) {
    return __ballot_sync(FULL_WARP_MASK, predicate);
}

// ============================================================================
// Min/Max utilities
// ============================================================================

template <typename T>
__device__ __forceinline__ T device_min(T a, T b) {
    return (a < b) ? a : b;
}

template <typename T>
__device__ __forceinline__ T device_max(T a, T b) {
    return (a > b) ? a : b;
}

// Clamp value to range [lo, hi]
template <typename T>
__device__ __forceinline__ T device_clamp(T val, T lo, T hi) {
    return device_max(lo, device_min(val, hi));
}

// ============================================================================
// Shuffle operations (for warp-level communication)
// ============================================================================

// Shuffle down within warp
template <typename T>
__device__ __forceinline__ T warp_shuffle(T val, int src_lane) {
    return __shfl_sync(FULL_WARP_MASK, val, src_lane);
}

template <typename T>
__device__ __forceinline__ T warp_shuffle_down(T val, int delta) {
    return __shfl_down_sync(FULL_WARP_MASK, val, delta);
}

template <typename T>
__device__ __forceinline__ T warp_shuffle_up(T val, int delta) {
    return __shfl_up_sync(FULL_WARP_MASK, val, delta);
}

#endif  // CUCIM_WAVELET_MATRIX_COMMON_CUH
