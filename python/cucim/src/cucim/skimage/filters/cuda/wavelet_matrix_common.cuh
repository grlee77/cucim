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
// Compile-time constants (can be overridden via Python preamble #defines)
// ============================================================================
// NOTE: We use WM_ prefix to avoid conflicts with CUB internal definitions
// (e.g., CUB uses WORD_SIZE internally in util_type.cuh)

// GPU architecture constants
#ifndef WM_WARP_SIZE
#define WM_WARP_SIZE 32
#endif

// Bitvector word size (32 or 64 bits)
#ifndef WM_WORD_SIZE
#define WM_WORD_SIZE 32
#endif

// Threads per block (valid values: 32, 64, 128, 256, 512, 1024)
#ifndef WM_THREAD_PER_GRID
#define WM_THREAD_PER_GRID 512
#endif

// Maximum blocks in X dimension for grid
#ifndef WM_MAX_BLOCK_X
#define WM_MAX_BLOCK_X 1024
#endif

// Source cache divisor for shared memory optimization
#ifndef WM_SRC_CACHE_DIV
#define WM_SRC_CACHE_DIV 2
#endif

// Shared memory usage in KB (64KB is common max for modern GPUs)
#ifndef WM_SHMEM_USE_KB
#define WM_SHMEM_USE_KB 64
#endif

// Minimum destination buffer size in KB
#ifndef WM_MIN_DSTBUF_KB
#define WM_MIN_DSTBUF_KB 4
#endif

// Value bit length (8, 16, or 32 depending on dtype)
// Will be set by Python based on image dtype
#ifndef WM_VAL_BIT_LEN
#define WM_VAL_BIT_LEN 8
#endif

// Width bit length (number of bits needed to represent image width)
// Will be set by Python based on image dimensions
#ifndef WM_W_BIT_LEN
#define WM_W_BIT_LEN 16
#endif

// Number of channels (1 for grayscale, focus on single-channel for cuCIM)
#ifndef WM_CH_NUM
#define WM_CH_NUM 1
#endif

// Derived constants
#define WM_THREADS_DIM_Y (WM_THREAD_PER_GRID / WM_WARP_SIZE)
#define WM_WORD_DIV_WARP (WM_WORD_SIZE / WM_WARP_SIZE)

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
// Index type definitions
// ============================================================================

// X index type (for column indices within a row)
// uint16_t supports images up to 65535 pixels wide
typedef unsigned short XIdxT;

// Y index type (for row indices)
typedef unsigned short YIdxT;

// Combined XY index type (for linear pixel indices)
// uint32_t supports images up to ~4 billion pixels total
typedef unsigned int XYIdxT;

// ============================================================================
// Block type for bitvector storage
// ============================================================================

/*
 * BlockT stores a bitvector block with:
 * - nsum: Cumulative count of set bits (prefix sum) up to this block
 * - nbit: The actual bitvector data for this block
 *
 * For WM_WORD_SIZE=32:
 *   sizeof(BlockT) = 8 bytes (4 + 4, with alignment)
 *
 * For WM_WORD_SIZE=64:
 *   sizeof(BlockT) = 12 bytes (4 + 8, with padding)
 */

#if WM_WORD_SIZE == 32

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

#elif WM_WORD_SIZE == 64

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

#endif  // WM_WORD_SIZE

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
    const IdxType bi = i / WM_WORD_SIZE;  // Block index
    const int ai = i % WM_WORD_SIZE;       // Position within block

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
    const IdxType bi = i / WM_WORD_SIZE;
    const int ai = i % WM_WORD_SIZE;

    const BlockT block = nbit_bp[bi];
    return block.nsum + block_popc(block.nbit & block_mask(ai));
}

// ============================================================================
// Warp-level utilities
// ============================================================================

// Get lane ID within warp
__device__ __forceinline__ int lane_id() {
    return threadIdx.x % WM_WARP_SIZE;
}

// Get warp ID within block
__device__ __forceinline__ int warp_id() {
    return threadIdx.x / WM_WARP_SIZE;
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

// ============================================================================
// Host-side utility functions (for Python buffer calculations)
// These are also defined in the Python code but included here for reference
// ============================================================================

#ifdef __CUDACC__
// These are device-side only

/*
 * Compute the number of bits needed to represent a value.
 * Returns ceil(log2(val + 1)).
 * For val=0, returns 0. For val=1, returns 1. For val=255, returns 8.
 */
__host__ __device__ __forceinline__ int get_bit_len(unsigned long long val) {
    if (val == 0) return 0;
    int bits = 0;
    while (val > 0) {
        val >>= 1;
        bits++;
    }
    return bits;
}

/*
 * Round up to the next power of 2.
 * Used for buffer size alignment.
 */
__host__ __device__ __forceinline__ unsigned int next_power_of_2(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

#endif  // __CUDACC__

// ============================================================================
// Shared memory size computation
// ============================================================================

/*
 * Compute shared memory size needed for source cache.
 * Formula: sizeof(ValT) * WM_THREADS_DIM_Y * (WM_WARP_SIZE - 1) * WM_WARP_SIZE
 *
 * For uint8_t with WM_THREAD_PER_GRID=512:
 *   1 * 16 * 31 * 32 = 15,872 bytes
 *
 * For uint16_t with WM_THREAD_PER_GRID=512:
 *   2 * 16 * 31 * 32 = 31,744 bytes
 */
#define WM_SRC_CACHE_SIZE(val_size) \
    ((val_size) * WM_THREADS_DIM_Y * (WM_WARP_SIZE - 1) * WM_WARP_SIZE)

/*
 * Compute destination buffer size for shared memory.
 * This is used for coalesced writes during the up-sweep phase.
 */
#define WM_DST_BUF_SIZE_BYTES(src_cache_bytes) \
    ((WM_SHMEM_USE_KB * 1024 - (src_cache_bytes)) & ~(WM_MIN_DSTBUF_KB * 1024 - 1))

#endif  // CUCIM_WAVELET_MATRIX_COMMON_CUH
