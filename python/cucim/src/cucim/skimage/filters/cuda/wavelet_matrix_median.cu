/*
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 *
 * Wavelet matrix median filter kernels for cuCIM.
 *
 * This file contains the CUDA kernels for the wavelet matrix-based median filter:
 * - wavelet_median2d: The main median query kernel
 *
 * The wavelet matrix construction kernels are in a separate file
 * (wavelet_matrix_construct.cu) as they are more complex.
 *
 * Based on the wavelet matrix 2D median algorithm described in:
 * - Sumida et al. (2022) "High-Performance 2D Median Filter using Wavelet Matrix"
 *   https://dl.acm.org/doi/10.1145/3550454.3555512
 *
 * Note: This file cannot be compiled standalone. It requires a preamble with
 * #define statements for the configuration parameters. See _median_wavelet.py
 * for the Python code that generates the full kernel.
 */

// ============================================================================
// The following defines are expected to be provided by the Python preamble:
//
// #define WM_VAL_T         unsigned char  // or unsigned short, unsigned int
// #define WM_VAL_BIT_LEN   8              // 8, 16, or 32
// #define WM_W_BIT_LEN     10             // ceil(log2(width))
// #define WM_WORD_SIZE     32             // bitvector word size
// #define WM_THREADS_W     8              // threads per block in X
// #define WM_THREADS_H     64             // threads per block in Y
// ============================================================================

// Include the common header (CUB, BlockT, rank functions, etc.)
// Note: In actual use, this content will be inlined by Python
// #include "wavelet_matrix_common.cuh"

// ============================================================================
// Specialized rank0 function for median query
// ============================================================================

/*
 * Compute rank0(i) for the median query kernel.
 * Returns the count of 0-bits in positions [0, i).
 *
 * This is equivalent to wavelet_rank1 from the common header, but inlined
 * here for clarity. The median kernel uses rank1 (count of 1-bits) to
 * navigate the wavelet matrix.
 */
__device__ __forceinline__ XYIdxT median_rank0(
    const XYIdxT i,
    const BlockT* __restrict__ nbit_bp
) {
    const XYIdxT bi = i / WM_WORD_SIZE;
    const int ai = i % WM_WORD_SIZE;
    const BlockT block = nbit_bp[bi];

#if WM_WORD_SIZE == 32
    return block.nsum + __popc(block.nbit & ((1u << ai) - 1));
#else
    return block.nsum + __popcll(block.nbit & ((1ull << ai) - 1ull));
#endif
}

// ============================================================================
// 2D Median Query Kernel
// ============================================================================

/*
 * Wavelet matrix 2D median filter kernel.
 *
 * This kernel computes the median value for each pixel using pre-constructed
 * wavelet matrices. Each thread processes one output pixel.
 *
 * Algorithm overview:
 * For each output pixel (x, y) with filter radius r:
 * 1. Define the query rectangle [ya, yb) x [xa, xb) around the pixel
 * 2. For each bit level h (from MSB to LSB):
 *    a. Count 0-bits in the rectangle using rank queries on the value wavelet matrix
 *    b. For the X dimension, use the column wavelet matrix to compute counts
 *    c. Based on the median position k, decide if the current bit is 0 or 1
 *    d. Update the search bounds accordingly
 * 3. The accumulated bits form the median value
 *
 * Parameters:
 *   H, W: Image dimensions (after padding if CUT_BORDER=true)
 *   res_step_num: Output row stride in elements
 *   r: Filter radius (kernel size = 2*r + 1)
 *   res_cu: Output buffer
 *   wm_nbit_bp: Column wavelet matrix (for X dimension queries)
 *   nsum_pos: Position of the total count in the wavelet matrix
 *   bv_block_h_byte_div32: Bytes per bit level divided by 32
 *   bv_block_len: Number of blocks per bit level
 *   bv_nbit_bp: Value wavelet matrix (for Y dimension and value queries)
 *   w_bit_len: Number of bits for column indices
 *   val_bit_len: Number of bits for values
 */
extern "C" __global__ void wavelet_median2d(
    const int H,
    const int W,
    const int res_step_num,
    const int r,
    WM_VAL_T* __restrict__ res_cu,
    const BlockT* __restrict__ wm_nbit_bp,
    const unsigned int nsum_pos,
    const unsigned int bv_block_h_byte_div32,
    const unsigned int bv_block_len,
    const BlockT* __restrict__ bv_nbit_bp,
    const unsigned char w_bit_len,
    const unsigned char val_bit_len
) {
    // Compute output pixel coordinates
    const int y = blockIdx.y * WM_THREADS_H + threadIdx.y;
    if (y >= H) return;
    const int x = blockIdx.x * WM_THREADS_W + threadIdx.x;
    if (x >= W) return;

    // Define the query rectangle bounds
    // Using clamp-to-border mode (non-CUT_BORDER case)
    XYIdxT ya = (y < r) ? 0 : (y - r);
    XIdxT  xa = (x < r) ? 0 : (x - r);
    XYIdxT yb = y + r + 1;
    if (yb > (XYIdxT)H) yb = H;
    XIdxT  xb = x + r + 1;
    if (xb > (XIdxT)W) xb = W;

    // Median position: floor(count / 2)
    XYIdxT k = (XYIdxT)(yb - ya) * (xb - xa) / 2;

    // Accumulator for the result value
    WM_VAL_T res = 0;

    // Convert row bounds to linear indices
    ya *= W;
    yb *= W;

    // Pointers to current bit level in wavelet matrices
    // Start at the highest bit level (MSB)
    const BlockT* bv_ptr = bv_nbit_bp;
    const BlockT* wm_ptr = wm_nbit_bp;

    // Process each bit level from MSB to LSB
    for (int h = val_bit_len; h-- > 0; ) {
        // Rank queries on the value wavelet matrix for Y boundaries
        const XYIdxT top0 = median_rank0(ya, bv_ptr);
        const XYIdxT bot0 = median_rank0(yb, bv_ptr);

        // Initialize X boundary trackers
        XYIdxT l_ya_xa = top0;
        XYIdxT l_yb_xa = bot0;
        XYIdxT l_ya_xb = top0;
        XYIdxT l_yb_xb = bot0;

        // Count of elements less than current threshold
        XYIdxT d = 0;

        // Navigate the column wavelet matrix to count elements in rectangle
        const BlockT* wm_level_ptr = wm_ptr;
        for (int j = w_bit_len; j-- > 0; ) {
            // Total zeros at this level
            const XYIdxT zeros = wm_level_ptr[nsum_pos].nsum;

            // Rank queries for all four corners
            const XYIdxT l_ya_xa_rank0 = median_rank0(l_ya_xa, wm_level_ptr);
            const XYIdxT l_ya_xb_rank0 = median_rank0(l_ya_xb, wm_level_ptr);
            const XYIdxT l_yb_xb_rank0 = median_rank0(l_yb_xb, wm_level_ptr);
            const XYIdxT l_yb_xa_rank0 = median_rank0(l_yb_xa, wm_level_ptr);

            // Update based on xa bit
            if (((xa >> j) & 1) == 0) {
                // Bit is 0: stay in the 0-region
                l_ya_xa = l_ya_xa_rank0;
                l_yb_xa = l_yb_xa_rank0;
            } else {
                // Bit is 1: move to the 1-region, accumulate count
                d += l_ya_xa_rank0;
                l_ya_xa += zeros - l_ya_xa_rank0;
                d -= l_yb_xa_rank0;
                l_yb_xa += zeros - l_yb_xa_rank0;
            }

            // Update based on xb bit
            if (((xb >> j) & 1) == 0) {
                l_ya_xb = l_ya_xb_rank0;
                l_yb_xb = l_yb_xb_rank0;
            } else {
                d -= l_ya_xb_rank0;
                l_ya_xb += zeros - l_ya_xb_rank0;
                d += l_yb_xb_rank0;
                l_yb_xb += zeros - l_yb_xb_rank0;
            }

            // Move to next bit level in column wavelet matrix
            wm_level_ptr = (const BlockT*)((const char*)wm_level_ptr -
                                           bv_block_h_byte_div32 * 32ull);
        }

        // Get total zeros at this value bit level
        const XYIdxT bv_h_zeros = bv_ptr[nsum_pos].nsum;

        // Decide if the median has a 0 or 1 at this bit position
        if (k < d) {
            // Median has 0 at this bit: stay in the 0-region
            ya = top0;
            yb = bot0;
        } else {
            // Median has 1 at this bit: move to 1-region, set bit in result
            k -= d;
            res |= (WM_VAL_T)1 << h;
            ya += bv_h_zeros - top0;
            yb += bv_h_zeros - bot0;
        }

        // Move to next bit level in value wavelet matrix
        bv_ptr = (const BlockT*)((const char*)bv_ptr -
                                 bv_block_h_byte_div32 * 32ull);
    }

    // Write result
    res_cu[(XYIdxT)y * res_step_num + x] = res;
}

// ============================================================================
// Variant with pre-padded input (CUT_BORDER mode)
// ============================================================================

/*
 * Wavelet matrix 2D median filter kernel for pre-padded images.
 *
 * When the input image has already been padded by r pixels on each side,
 * we can skip boundary checking and use simpler index calculations.
 *
 * Input dimensions: (H + 2*r) x (W + 2*r)
 * Output dimensions: H x W
 */
extern "C" __global__ void wavelet_median2d_padded(
    const int H,
    const int W,
    const int padded_W,
    const int res_step_num,
    const int r,
    WM_VAL_T* __restrict__ res_cu,
    const BlockT* __restrict__ wm_nbit_bp,
    const unsigned int nsum_pos,
    const unsigned int bv_block_h_byte_div32,
    const unsigned int bv_block_len,
    const BlockT* __restrict__ bv_nbit_bp,
    const unsigned char w_bit_len,
    const unsigned char val_bit_len
) {
    // Compute output pixel coordinates
    const int y = blockIdx.y * WM_THREADS_H + threadIdx.y;
    if (y >= H) return;
    const int x = blockIdx.x * WM_THREADS_W + threadIdx.x;
    if (x >= W) return;

    // For padded input, the query rectangle is always full-sized
    // No boundary checking needed
    XYIdxT ya = y;
    XIdxT  xa = x;
    XYIdxT yb = y + 2 * r + 1;
    XIdxT  xb = x + 2 * r + 1;

    // Median position for full kernel
    const XYIdxT kernel_size = (2 * r + 1) * (2 * r + 1);
    XYIdxT k = kernel_size / 2;

    // Accumulator for the result value
    WM_VAL_T res = 0;

    // Convert row bounds to linear indices (using padded width)
    ya *= padded_W;
    yb *= padded_W;

    // Pointers to current bit level in wavelet matrices
    const BlockT* bv_ptr = bv_nbit_bp;
    const BlockT* wm_ptr = wm_nbit_bp;

    // Process each bit level from MSB to LSB
    for (int h = val_bit_len; h-- > 0; ) {
        const XYIdxT top0 = median_rank0(ya, bv_ptr);
        const XYIdxT bot0 = median_rank0(yb, bv_ptr);

        XYIdxT l_ya_xa = top0;
        XYIdxT l_yb_xa = bot0;
        XYIdxT l_ya_xb = top0;
        XYIdxT l_yb_xb = bot0;
        XYIdxT d = 0;

        const BlockT* wm_level_ptr = wm_ptr;
        for (int j = w_bit_len; j-- > 0; ) {
            const XYIdxT zeros = wm_level_ptr[nsum_pos].nsum;
            const XYIdxT l_ya_xa_rank0 = median_rank0(l_ya_xa, wm_level_ptr);
            const XYIdxT l_ya_xb_rank0 = median_rank0(l_ya_xb, wm_level_ptr);
            const XYIdxT l_yb_xb_rank0 = median_rank0(l_yb_xb, wm_level_ptr);
            const XYIdxT l_yb_xa_rank0 = median_rank0(l_yb_xa, wm_level_ptr);

            if (((xa >> j) & 1) == 0) {
                l_ya_xa = l_ya_xa_rank0;
                l_yb_xa = l_yb_xa_rank0;
            } else {
                d += l_ya_xa_rank0;
                l_ya_xa += zeros - l_ya_xa_rank0;
                d -= l_yb_xa_rank0;
                l_yb_xa += zeros - l_yb_xa_rank0;
            }

            if (((xb >> j) & 1) == 0) {
                l_ya_xb = l_ya_xb_rank0;
                l_yb_xb = l_yb_xb_rank0;
            } else {
                d -= l_ya_xb_rank0;
                l_ya_xb += zeros - l_ya_xb_rank0;
                d += l_yb_xb_rank0;
                l_yb_xb += zeros - l_yb_xb_rank0;
            }

            wm_level_ptr = (const BlockT*)((const char*)wm_level_ptr -
                                           bv_block_h_byte_div32 * 32ull);
        }

        const XYIdxT bv_h_zeros = bv_ptr[nsum_pos].nsum;

        if (k < d) {
            ya = top0;
            yb = bot0;
        } else {
            k -= d;
            res |= (WM_VAL_T)1 << h;
            ya += bv_h_zeros - top0;
            yb += bv_h_zeros - bot0;
        }

        bv_ptr = (const BlockT*)((const char*)bv_ptr -
                                 bv_block_h_byte_div32 * 32ull);
    }

    // Write result
    res_cu[(XYIdxT)y * res_step_num + x] = res;
}
