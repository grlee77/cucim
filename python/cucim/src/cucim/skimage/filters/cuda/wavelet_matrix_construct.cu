/*
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 *
 * Wavelet matrix construction kernels for 2D median filter.
 *
 * This file contains the kernels for building the wavelet matrix data structure:
 * - wavelet_first_pass: Initial setup and MSB counting
 * - wavelet_upsweep: Main construction kernel (one bit level per call)
 * - wavelet_exclusive_sum: Prefix sum for block counts
 * - wavelet_last_pass: Final column wavelet matrix construction
 *
 * Adapted from OpenCV's wavelet_matrix_2d.cuh for use with CuPy RawKernels.
 */

// Note: wavelet_matrix_common.cuh is prepended by Python
// Do not include it here to avoid NVRTC include path issues

// ============================================================================
// Value type definition (set by Python preamble)
// ============================================================================
#ifndef WM_VAL_T
#define WM_VAL_T unsigned char
#endif

// ============================================================================
// Kernel 1: First Pass - Initial Setup
// ============================================================================
/*
 * This kernel performs the first pass of wavelet matrix construction:
 * 1. Copies source data while counting zeros for the MSB
 * 2. Initializes column index array (x coordinate for each pixel)
 * 3. Accumulates zero counts per block using warp reductions
 *
 * Parameters:
 *   mask: Bit mask for comparing values (e.g., 0x80 for MSB of uint8)
 *   block_pair_num: Number of block pairs per grid block
 *   size_div_warp: Total size divided by warp size
 *   src: Source pixel values
 *   nsum_scan_buf: Buffer for block zero counts (output)
 *   buf_byte_div32: Buffer byte size / 32
 *   buf_idx: Column index buffer (output)
 *   W: Image width
 *   WH: Total pixels (W * H)
 */
extern "C" __global__ void wavelet_first_pass(
    const WM_VAL_T mask,
    const unsigned short block_pair_num,
    const XYIdxT size_div_warp,
    const WM_VAL_T* __restrict__ src,
    XYIdxT* __restrict__ nsum_scan_buf,
    const unsigned int buf_byte_div32,
    XIdxT* __restrict__ buf_idx,
    const int W,
    const XYIdxT WH
) {
    // Thread configuration constants
    constexpr int THREAD_PER_GRID = WM_THREAD_PER_GRID;
    constexpr int THREADS_DIM_Y = WM_THREADS_DIM_Y;

    // Per-thread zero counter
    XYIdxT cs = 0;

    // Calculate starting position for this thread block
    XYIdxT i = (XYIdxT)blockIdx.x * block_pair_num * THREADS_DIM_Y + threadIdx.y;

    // Calculate initial x index for this thread
    XIdxT x_idx = (i * WM_WARP_SIZE + threadIdx.x) % W;
    const XIdxT x_diff = THREAD_PER_GRID % W;

    // Process block_pair_num iterations
    for (XYIdxT k = 0; k < block_pair_num; ++k, i += THREADS_DIM_Y) {
        if (i >= size_div_warp) break;

        const XYIdxT idx = i * WM_WARP_SIZE + threadIdx.x;

        // Handle boundary - set index to 0 for out-of-bounds pixels
        if (idx >= WH) x_idx = 0;

        // Load source value
        const WM_VAL_T v = src[idx];

        // Count zeros (values <= mask means bit is 0)
        if (v <= mask) {
            ++cs;
        }

        // Store column index
        buf_idx[idx] = x_idx;

        // Update x index for next iteration
        x_idx += x_diff;
        if (x_idx >= W) x_idx -= W;
    }

    // Warp-level reduction to sum zero counts
    using WarpReduce = cub::WarpReduce<unsigned int>;
    __shared__ typename WarpReduce::TempStorage WarpReduce_temp_storage[WM_THREADS_DIM_Y];
    __shared__ XYIdxT cs_sum_sh[WM_THREADS_DIM_Y];

    cs = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(cs);

    // First lane of each warp stores to shared memory
    if (threadIdx.x == 0) {
        cs_sum_sh[threadIdx.y] = cs;
    }
    __syncthreads();

    // Only first warp performs final reduction
    if (threadIdx.y != 0) return;

    XYIdxT cs_bsum = (threadIdx.x < WM_THREADS_DIM_Y ? cs_sum_sh[threadIdx.x] : 0);
    cs_bsum = WarpReduce(WarpReduce_temp_storage[0]).Sum(cs_bsum);

    // First thread writes block total to scan buffer
    if (threadIdx.x == 0) {
        nsum_scan_buf[blockIdx.x] = cs_bsum;
    }
}


// ============================================================================
// Kernel 2: Up-Sweep - Main Construction Kernel
// ============================================================================
/*
 * This is the main construction kernel, called once per bit level.
 * It performs:
 * 1. Reads source values and creates bitvector based on current bit
 * 2. Computes prefix sums for ranking
 * 3. Reorders values and indices based on bit values (0s before 1s)
 * 4. Counts zeros for next level
 *
 * Parameters:
 *   mask: Bit mask for current level
 *   block_pair_num: Number of block pairs per grid block
 *   size_div_w: Size divided by word size
 *   src: Current level source values
 *   dst: Next level destination values (reordered)
 *   nbit_bp: Block pointer for current level bitvector (output)
 *   nsum_buf_test: Prefix sum buffer (input - from previous exclusive sum)
 *   nsum_buf_test2: Block counts for next level (output)
 *   bv_block_byte_div32: Bitvector block bytes / 32
 *   buf_byte_div32: Buffer bytes / 32
 *   idx_p: Current column index buffer
 *   nxt_idx: Next level column index buffer (output)
 *   nbit_bp_pre: Previous level bitvector (for total count)
 *   is_last_val_bit: True if this is the last value bit level
 */
extern "C" __global__ void wavelet_upsweep(
    const WM_VAL_T mask,
    const unsigned short block_pair_num,
    const XYIdxT size_div_w,
    const WM_VAL_T* __restrict__ src,
    WM_VAL_T* __restrict__ dst,
    BlockT* __restrict__ nbit_bp,
    const XYIdxT* __restrict__ nsum_buf_test,
    XYIdxT* __restrict__ nsum_buf_test2,
    const unsigned int bv_block_byte_div32,
    const unsigned int buf_byte_div32,
    const XIdxT* __restrict__ idx_p,
    XIdxT* __restrict__ nxt_idx,
    const BlockT* __restrict__ nbit_bp_pre,
    const int is_last_val_bit
) {
    // Thread configuration
    constexpr int THREAD_PER_GRID = WM_THREAD_PER_GRID;
    constexpr int THREADS_DIM_Y = WM_THREADS_DIM_Y;
    constexpr int SRC_CACHE_DIV = WM_SRC_CACHE_DIV;

    // CUB types for warp operations
    using WarpScanX = cub::WarpScan<XYIdxT, WM_WARP_SIZE / SRC_CACHE_DIV>;
    using WarpScanY = cub::WarpScan<XYIdxT, THREADS_DIM_Y>;
    using WarpReduce = cub::WarpReduce<unsigned int>;
    using WarpReduceY = cub::WarpReduce<unsigned int, THREADS_DIM_Y>;

    // Shared memory for value and index caching
    __shared__ WM_VAL_T src_val_cache[THREADS_DIM_Y][(WM_WARP_SIZE/SRC_CACHE_DIV)-1][WM_WARP_SIZE];
    __shared__ XIdxT vidx_val_cache[THREADS_DIM_Y][(WM_WARP_SIZE/SRC_CACHE_DIV)-1][WM_WARP_SIZE];

    // Shared memory for reductions and scans
    __shared__ uint4 nsum_count_sh[THREADS_DIM_Y];
    __shared__ XYIdxT pre_sum_share[2];
    __shared__ XYIdxT warp_scan_sums[THREADS_DIM_Y];
    __shared__ typename WarpScanX::TempStorage s_scanStorage;
    __shared__ typename WarpScanY::TempStorage s_scanStorage2;
    __shared__ typename WarpReduce::TempStorage WarpReduce_temp_storage[THREADS_DIM_Y];
    __shared__ typename WarpReduceY::TempStorage WarpReduceY_temp_storage;

    const XYIdxT size_div_warp = size_div_w * WM_WORD_DIV_WARP;
    const XYIdxT nsum = nbit_bp[size_div_w].nsum;  // Total zeros from previous level
    const XYIdxT nsum_offset = nsum_buf_test[blockIdx.x];
    const XYIdxT nsum_pre = nbit_bp_pre[size_div_w].nsum;

    // Compute bounds for counting
    XYIdxT nsum_idx0_org = nsum_offset;
    XYIdxT nsum_idx1_org = (XYIdxT)blockIdx.x * block_pair_num * THREAD_PER_GRID + nsum - nsum_idx0_org;
    nsum_idx0_org /= (XYIdxT)block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE;
    nsum_idx1_org /= (XYIdxT)block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE;
    const XYIdxT nsum_idx0_bound = (nsum_idx0_org + 1) * block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE;
    const XYIdxT nsum_idx1_bound = (nsum_idx1_org + 1) * block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE;
    uint4 nsum_count = make_uint4(0, 0, 0, 0);

    const unsigned short th_idx = threadIdx.y * WM_WARP_SIZE + threadIdx.x;
    if (th_idx == 0) {
        pre_sum_share[0] = nsum_offset;
    }

    // Main processing loop
    for (XYIdxT ka = 0; ka < block_pair_num; ka += WM_WARP_SIZE / SRC_CACHE_DIV) {
        const XYIdxT ibb = ((XYIdxT)blockIdx.x * block_pair_num + ka) * THREADS_DIM_Y;
        if (ibb >= size_div_warp) break;

        unsigned int my_bits = 0;
        WM_VAL_T first_val;
        XIdxT first_idxval;

        // Load values and create bitvector
        for (XYIdxT kb = 0, i = ibb + WM_WARP_SIZE / SRC_CACHE_DIV * threadIdx.y;
             kb < WM_WARP_SIZE / SRC_CACHE_DIV; ++kb, ++i) {
            if (i >= size_div_warp) break;

            unsigned int bits;
            const XYIdxT ij = i * WM_WARP_SIZE + threadIdx.x;
            const WM_VAL_T v = src[ij];
            const XIdxT idx_v = idx_p[ij];

            if (kb == 0) {
                first_val = v;
                first_idxval = idx_v;
            } else {
                src_val_cache[threadIdx.y][kb - 1][threadIdx.x] = v;
                vidx_val_cache[threadIdx.y][kb - 1][threadIdx.x] = idx_v;
            }

            // Create bitvector based on value comparison with mask
            if (v <= mask) {
                bits = __activemask();
            } else {
                bits = ~__activemask();
            }

            if (threadIdx.x == kb) {
                my_bits = bits;
            }
        }

        // Compute prefix sum of zero counts
        XYIdxT c, t = 0;
        if (threadIdx.y < THREADS_DIM_Y) {
            c = __popc(my_bits);

            WarpScanX(s_scanStorage).ExclusiveSum(c, t);
            if (threadIdx.x == WM_WARP_SIZE / SRC_CACHE_DIV - 1) {
                warp_scan_sums[threadIdx.y] = c + t;
            }
        }

        __syncthreads();

        XYIdxT pre_sum = pre_sum_share[(ka & (WM_WARP_SIZE / SRC_CACHE_DIV)) > 0 ? 1 : 0];
        XYIdxT s = threadIdx.x < THREADS_DIM_Y ? warp_scan_sums[threadIdx.x] : 0;
        WarpScanY(s_scanStorage2).ExclusiveSum(s, s);

        s = __shfl_sync(FULL_WARP_MASK, s, threadIdx.y);
        s += t + pre_sum;

        // Store bitvector block
        if (SRC_CACHE_DIV == 1 || threadIdx.x < WM_WARP_SIZE / SRC_CACHE_DIV) {
            if (th_idx == THREAD_PER_GRID - WM_WARP_SIZE + WM_WARP_SIZE / SRC_CACHE_DIV - 1) {
                pre_sum_share[(ka & (WM_WARP_SIZE / SRC_CACHE_DIV)) == 0 ? 1 : 0] = s + c;
            }
            const XYIdxT bi = ibb + threadIdx.y * WM_WARP_SIZE / SRC_CACHE_DIV + threadIdx.x;
            if (bi < size_div_warp) {
                nbit_bp[bi].nsum = s;
                nbit_bp[bi].nbit = my_bits;
            }
        }

        // Handle last value bit level (no value reordering needed)
        if (is_last_val_bit) {
            WM_VAL_T vo = first_val;
            XIdxT idx_v = first_idxval;
            for (XYIdxT j = 0, i = ibb + WM_WARP_SIZE / SRC_CACHE_DIV * threadIdx.y;
                 j < WM_WARP_SIZE / SRC_CACHE_DIV; ++j, ++i) {
                if (i >= size_div_warp) break;

                const unsigned int e_nbit = __shfl_sync(FULL_WARP_MASK, my_bits, j);
                const XYIdxT e_nsum = __shfl_sync(FULL_WARP_MASK, s, j);
                XYIdxT rank = __popc(e_nbit << (WM_WARP_SIZE - threadIdx.x));
                const XYIdxT idx0 = e_nsum + rank;
                XYIdxT idx = idx0;

                if (vo > mask) {  // bit is 1
                    const XYIdxT ij = i * WM_WARP_SIZE + threadIdx.x;
                    idx = ij + nsum - idx;
                }

                if (idx < size_div_warp * WM_WARP_SIZE) {
                    nxt_idx[idx] = idx_v;
                }

                if (j == WM_WARP_SIZE / SRC_CACHE_DIV - 1) break;
                vo = src_val_cache[threadIdx.y][j][threadIdx.x];
                idx_v = vidx_val_cache[threadIdx.y][j][threadIdx.x];
            }
            continue;
        }

        // Reorder values and indices based on bit values
        const WM_VAL_T mask_2 = mask >> 1;
        WM_VAL_T vo = first_val;
        XIdxT idx_v = first_idxval;

        for (XYIdxT j = 0, i = ibb + WM_WARP_SIZE / SRC_CACHE_DIV * threadIdx.y;
             j < WM_WARP_SIZE / SRC_CACHE_DIV; ++j, ++i) {
            if (i >= size_div_warp) break;

            const unsigned int e_nbit = __shfl_sync(FULL_WARP_MASK, my_bits, j);
            const XYIdxT e_nsum = __shfl_sync(FULL_WARP_MASK, s, j);
            XYIdxT rank = __popc(e_nbit << (WM_WARP_SIZE - threadIdx.x));
            const XYIdxT idx0 = e_nsum + rank;

            WM_VAL_T v = vo;
            XYIdxT idx = idx0;
            if (vo > mask) {  // bit is 1
                const XYIdxT ij = i * WM_WARP_SIZE + threadIdx.x;
                idx = ij + nsum - idx;
                v &= mask;  // Clear the processed bit
            }

            if (idx < size_div_warp * WM_WARP_SIZE) {
                dst[idx] = v;
                nxt_idx[idx] = idx_v;
            }

            // Count zeros for next level
            if (v <= mask_2) {
                if (vo <= mask) {
                    if (idx < nsum_idx0_bound) {
                        nsum_count.x++;
                    } else {
                        nsum_count.y++;
                    }
                } else {
                    if (idx < nsum_idx1_bound) {
                        nsum_count.z++;
                    } else {
                        nsum_count.w++;
                    }
                }
            }

            if (j == WM_WARP_SIZE / SRC_CACHE_DIV - 1) break;
            vo = src_val_cache[threadIdx.y][j][threadIdx.x];
            idx_v = vidx_val_cache[threadIdx.y][j][threadIdx.x];
        }
    }

    // Store final nsum for last block
    if (blockIdx.x == gridDim.x - 1 && th_idx == 0) {
        nbit_bp[size_div_warp / WM_WORD_DIV_WARP].nsum = nsum;
    }

    // Reduce and store next-level counts
    nsum_count.x = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.x);
    nsum_count.y = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.y);
    nsum_count.z = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.z);
    nsum_count.w = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.w);

    if (threadIdx.x == 0) {
        nsum_count_sh[threadIdx.y] = nsum_count;
    }
    __syncthreads();

    if (threadIdx.x < THREADS_DIM_Y) {
        nsum_count = nsum_count_sh[threadIdx.x];
        nsum_count.x = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.x);
        nsum_count.y = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.y);
        nsum_count.z = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.z);
        nsum_count.w = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.w);

        if (th_idx == 0 && !is_last_val_bit) {
            const XYIdxT idx0_org = nsum_idx0_bound / ((XYIdxT)block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE);
            const XYIdxT idx1_org = nsum_idx1_bound / ((XYIdxT)block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE);

            if (nsum_count.x > 0) atomicAdd(nsum_buf_test2 + idx0_org - 1, nsum_count.x);
            if (nsum_count.y > 0) atomicAdd(nsum_buf_test2 + idx0_org - 0, nsum_count.y);
            if (nsum_count.z > 0) atomicAdd(nsum_buf_test2 + idx1_org - 1, nsum_count.z);
            if (nsum_count.w > 0) atomicAdd(nsum_buf_test2 + idx1_org - 0, nsum_count.w);
        }
    }
}


// ============================================================================
// Kernel 3: Exclusive Sum - Prefix Sum of Block Counts
// ============================================================================
/*
 * Computes exclusive prefix sum of block counts using CUB BlockScan.
 * This prepares the offset array for the next up-sweep iteration.
 *
 * Parameters:
 *   nsum_scan_buf: Input block counts, output exclusive prefix sums
 *   nsum_buf_test2: Secondary buffer (cleared to 0)
 *   nsum_p: Block pointer to store total sum
 *   buf_byte_div32: Buffer bytes / 32
 *   bv_block_byte_div32: Bitvector block bytes / 32
 */
extern "C" __global__ void wavelet_exclusive_sum(
    XYIdxT* __restrict__ nsum_scan_buf,
    XYIdxT* __restrict__ nsum_buf_test2,
    BlockT* __restrict__ nsum_p,
    const unsigned int buf_byte_div32,
    const unsigned int bv_block_byte_div32
) {
    typedef cub::BlockScan<XYIdxT, WM_MAX_BLOCK_X> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    XYIdxT thread_data1;
    XYIdxT thread_data2;

    // Offset buffer pointers for multi-channel (though we focus on single channel)
    nsum_scan_buf = (XYIdxT*)((unsigned char*)nsum_scan_buf +
                              (size_t)blockIdx.x * (buf_byte_div32 * 32ull));
    nsum_buf_test2 = (XYIdxT*)((unsigned char*)nsum_buf_test2 +
                               (size_t)blockIdx.x * (buf_byte_div32 * 32ull));

    // Load, scan, store
    thread_data1 = nsum_scan_buf[threadIdx.x];
    BlockScan(temp_storage).ExclusiveSum(thread_data1, thread_data2);

    nsum_scan_buf[threadIdx.x] = thread_data2;
    nsum_buf_test2[threadIdx.x] = 0;

    // Last thread stores total sum
    if (threadIdx.x == blockDim.x - 1) {
        thread_data2 += thread_data1;
        nsum_p = (BlockT*)((unsigned char*)nsum_p +
                           (size_t)blockIdx.x * (bv_block_byte_div32 * 32ull));
        nsum_p->nsum = thread_data2;
    }
}


// ============================================================================
// Kernel 4: Last Pass - Column Wavelet Matrix Construction
// ============================================================================
/*
 * Finalizes construction of the column (width) wavelet matrix.
 * Copies sorted column indices to the final wavelet matrix buffer.
 *
 * Parameters:
 *   block_pair_num: Number of block pairs per grid block
 *   size_div_w: Size divided by word size
 *   buf_byte_div32: Buffer bytes / 32
 *   idx_p: Final sorted column index buffer
 *   inf: Infinity value for padding (W for column indices)
 *   wm: Output column wavelet matrix
 *   wm_nsum_scan_buf: Buffer for column zero counts
 *   cwm_buf_byte_div32: Column wavelet matrix buffer bytes / 32
 *   nbit_bp_pre: Previous bitvector for size reference
 *   bv_block_byte_div32: Bitvector block bytes / 32
 */
extern "C" __global__ void wavelet_last_pass(
    const unsigned short block_pair_num,
    const XYIdxT size_div_w,
    const unsigned int buf_byte_div32,
    const XIdxT* __restrict__ idx_p,
    const XIdxT inf,
    XIdxT* __restrict__ wm,
    XYIdxT* __restrict__ wm_nsum_scan_buf,
    const XYIdxT cwm_buf_byte_div32,
    const BlockT* __restrict__ nbit_bp_pre,
    const unsigned int bv_block_byte_div32
) {
    constexpr int THREADS_DIM_Y = WM_THREADS_DIM_Y;
    constexpr int THREAD_PER_GRID = WM_THREAD_PER_GRID;

    // Get total count from previous bitvector level
    const XYIdxT nsum_pre = nbit_bp_pre[size_div_w].nsum;

    using WarpReduce = cub::WarpReduce<unsigned int>;
    using WarpReduceY = cub::WarpReduce<unsigned int, THREADS_DIM_Y>;

    __shared__ XYIdxT wm_zero_count_sh[THREADS_DIM_Y];
    __shared__ typename WarpReduce::TempStorage WarpReduce_temp_storage[THREADS_DIM_Y];
    __shared__ typename WarpReduceY::TempStorage WarpReduceY_temp_storage;

    XYIdxT wm_zero_count = 0;
    const XYIdxT size_div_warp = size_div_w * WM_WORD_DIV_WARP;
    const unsigned short th_idx = threadIdx.y * WM_WARP_SIZE + threadIdx.x;

    const int block_num = block_pair_num / WM_WARP_SIZE;
    for (XYIdxT ka = 0; ka < block_num; ++ka) {
        const XYIdxT ibb = ((XYIdxT)blockIdx.x * block_num + ka) * THREAD_PER_GRID +
                           WM_WARP_SIZE * threadIdx.y;
        if (ibb >= size_div_warp) break;

        for (XYIdxT kb = 0; kb < WM_WARP_SIZE; ++kb) {
            XYIdxT i = ibb + kb;
            if (i >= size_div_warp) break;

            const XYIdxT ij = i * WM_WARP_SIZE + threadIdx.x;

            if (ij < nsum_pre) {
                const XIdxT wm_idxv = idx_p[ij];
                wm[ij] = wm_idxv;
                // Count zeros (x index * 2 <= inf means it's in first half)
                if (wm_idxv * 2 <= inf) {
                    ++wm_zero_count;
                }
            } else {
                wm[ij] = inf;
            }
        }
    }

    // Reduce zero counts
    wm_zero_count = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(wm_zero_count);
    if (threadIdx.x == 0) {
        wm_zero_count_sh[threadIdx.y] = wm_zero_count;
    }
    __syncthreads();

    if (threadIdx.x < THREADS_DIM_Y) {
        wm_zero_count = WarpReduceY(WarpReduceY_temp_storage).Sum(wm_zero_count_sh[threadIdx.x]);
        if (th_idx == 0) {
            wm_nsum_scan_buf[blockIdx.x] = wm_zero_count;
        }
    }
}


// ============================================================================
// Kernel 5: Up-Sweep for Column Wavelet Matrix
// ============================================================================
/*
 * Similar to wavelet_upsweep but for column indices (XIdxT instead of ValT).
 * Processes one bit level of the column wavelet matrix.
 *
 * This is a simplified version that only reorders indices (no value data).
 */
extern "C" __global__ void wavelet_upsweep_wm(
    const XIdxT mask,
    const unsigned short block_pair_num,
    const XYIdxT size_div_w,
    const XIdxT* __restrict__ src,
    BlockT* __restrict__ nbit_bp,
    const XYIdxT* __restrict__ nsum_buf_test,
    XYIdxT* __restrict__ nsum_buf_test2,
    const unsigned int bv_block_byte_div32,
    const unsigned int cwm_buf_byte_div32,
    const XIdxT inf,
    XIdxT* __restrict__ nxt_idx,
    const BlockT* __restrict__ nbit_bp_pre,
    const int is_last_wm_bit
) {
    constexpr int THREAD_PER_GRID = WM_THREAD_PER_GRID;
    constexpr int THREADS_DIM_Y = WM_THREADS_DIM_Y;
    constexpr int SRC_CACHE_DIV = WM_SRC_CACHE_DIV;

    using WarpScanX = cub::WarpScan<XYIdxT, WM_WARP_SIZE / SRC_CACHE_DIV>;
    using WarpScanY = cub::WarpScan<XYIdxT, THREADS_DIM_Y>;
    using WarpReduce = cub::WarpReduce<unsigned int>;
    using WarpReduceY = cub::WarpReduce<unsigned int, THREADS_DIM_Y>;

    __shared__ XIdxT src_val_cache[THREADS_DIM_Y][(WM_WARP_SIZE/SRC_CACHE_DIV)-1][WM_WARP_SIZE];

    __shared__ uint4 nsum_count_sh[THREADS_DIM_Y];
    __shared__ XYIdxT wm_zero_count_sh[THREADS_DIM_Y];
    __shared__ XYIdxT pre_sum_share[2];
    __shared__ XYIdxT warp_scan_sums[THREADS_DIM_Y];
    __shared__ typename WarpScanX::TempStorage s_scanStorage;
    __shared__ typename WarpScanY::TempStorage s_scanStorage2;
    __shared__ typename WarpReduce::TempStorage WarpReduce_temp_storage[THREADS_DIM_Y];
    __shared__ typename WarpReduceY::TempStorage WarpReduceY_temp_storage;

    XYIdxT wm_zero_count = 0;

    const XYIdxT size_div_warp = size_div_w * WM_WORD_DIV_WARP;
    const XYIdxT nsum = nbit_bp[size_div_w].nsum;
    const XYIdxT nsum_offset = nsum_buf_test[blockIdx.x];
    const XYIdxT nsum_pre = nbit_bp_pre[size_div_w].nsum;

    XYIdxT nsum_idx0_org = nsum_offset;
    XYIdxT nsum_idx1_org = (XYIdxT)blockIdx.x * block_pair_num * THREAD_PER_GRID + nsum - nsum_idx0_org;
    nsum_idx0_org /= (XYIdxT)block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE;
    nsum_idx1_org /= (XYIdxT)block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE;
    const XYIdxT nsum_idx0_bound = (nsum_idx0_org + 1) * block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE;
    const XYIdxT nsum_idx1_bound = (nsum_idx1_org + 1) * block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE;
    uint4 nsum_count = make_uint4(0, 0, 0, 0);

    const unsigned short th_idx = threadIdx.y * WM_WARP_SIZE + threadIdx.x;
    if (th_idx == 0) {
        pre_sum_share[0] = nsum_offset;
    }

    for (XYIdxT ka = 0; ka < block_pair_num; ka += WM_WARP_SIZE / SRC_CACHE_DIV) {
        const XYIdxT ibb = ((XYIdxT)blockIdx.x * block_pair_num + ka) * THREADS_DIM_Y;
        if (ibb >= size_div_warp) break;

        unsigned int my_bits = 0;
        XIdxT first_val;

        for (XYIdxT kb = 0, i = ibb + WM_WARP_SIZE / SRC_CACHE_DIV * threadIdx.y;
             kb < WM_WARP_SIZE / SRC_CACHE_DIV; ++kb, ++i) {
            if (i >= size_div_warp) break;

            unsigned int bits;
            const XYIdxT ij = i * WM_WARP_SIZE + threadIdx.x;
            const XIdxT v = src[ij];

            if (kb == 0) {
                first_val = v;
            } else {
                src_val_cache[threadIdx.y][kb - 1][threadIdx.x] = v;
            }

            if (v <= mask) {
                bits = __activemask();
            } else {
                bits = ~__activemask();
            }

            if (threadIdx.x == kb) {
                my_bits = bits;
            }

            // Store to output for column WM construction
            if (ij < nsum_pre) {
                nxt_idx[ij] = v;
                if (v * 2 <= inf) {
                    ++wm_zero_count;
                }
            } else {
                nxt_idx[ij] = inf;
            }
        }

        XYIdxT c, t = 0;
        if (threadIdx.y < THREADS_DIM_Y) {
            c = __popc(my_bits);

            WarpScanX(s_scanStorage).ExclusiveSum(c, t);
            if (threadIdx.x == WM_WARP_SIZE / SRC_CACHE_DIV - 1) {
                warp_scan_sums[threadIdx.y] = c + t;
            }
        }

        __syncthreads();

        XYIdxT pre_sum = pre_sum_share[(ka & (WM_WARP_SIZE / SRC_CACHE_DIV)) > 0 ? 1 : 0];
        XYIdxT s = threadIdx.x < THREADS_DIM_Y ? warp_scan_sums[threadIdx.x] : 0;
        WarpScanY(s_scanStorage2).ExclusiveSum(s, s);

        s = __shfl_sync(FULL_WARP_MASK, s, threadIdx.y);
        s += t + pre_sum;

        if (SRC_CACHE_DIV == 1 || threadIdx.x < WM_WARP_SIZE / SRC_CACHE_DIV) {
            if (th_idx == THREAD_PER_GRID - WM_WARP_SIZE + WM_WARP_SIZE / SRC_CACHE_DIV - 1) {
                pre_sum_share[(ka & (WM_WARP_SIZE / SRC_CACHE_DIV)) == 0 ? 1 : 0] = s + c;
            }
            const XYIdxT bi = ibb + threadIdx.y * WM_WARP_SIZE / SRC_CACHE_DIV + threadIdx.x;
            if (bi < size_div_warp) {
                nbit_bp[bi].nsum = s;
                nbit_bp[bi].nbit = my_bits;
            }
        }

        // Reorder indices
        const XIdxT mask_2 = mask >> 1;
        XIdxT vo = first_val;

        for (XYIdxT j = 0, i = ibb + WM_WARP_SIZE / SRC_CACHE_DIV * threadIdx.y;
             j < WM_WARP_SIZE / SRC_CACHE_DIV; ++j, ++i) {
            if (i >= size_div_warp) break;

            const unsigned int e_nbit = __shfl_sync(FULL_WARP_MASK, my_bits, j);
            const XYIdxT e_nsum = __shfl_sync(FULL_WARP_MASK, s, j);
            XYIdxT rank = __popc(e_nbit << (WM_WARP_SIZE - threadIdx.x));
            const XYIdxT idx0 = e_nsum + rank;

            XIdxT v = vo;
            XYIdxT idx = idx0;
            if (vo > mask) {
                const XYIdxT ij = i * WM_WARP_SIZE + threadIdx.x;
                idx = ij + nsum - idx;
                v &= mask;
            }

            if (idx < size_div_warp * WM_WARP_SIZE && !is_last_wm_bit) {
                nxt_idx[idx] = v;
            }

            if (!is_last_wm_bit && v <= mask_2) {
                if (vo <= mask) {
                    if (idx < nsum_idx0_bound) {
                        nsum_count.x++;
                    } else {
                        nsum_count.y++;
                    }
                } else {
                    if (idx < nsum_idx1_bound) {
                        nsum_count.z++;
                    } else {
                        nsum_count.w++;
                    }
                }
            }

            if (j == WM_WARP_SIZE / SRC_CACHE_DIV - 1) break;
            vo = src_val_cache[threadIdx.y][j][threadIdx.x];
        }
    }

    if (blockIdx.x == gridDim.x - 1 && th_idx == 0) {
        nbit_bp[size_div_warp / WM_WORD_DIV_WARP].nsum = nsum;
    }

    // Reduce counts
    nsum_count.x = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.x);
    nsum_count.y = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.y);
    nsum_count.z = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.z);
    nsum_count.w = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.w);
    wm_zero_count = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(wm_zero_count);

    if (threadIdx.x == 0) {
        nsum_count_sh[threadIdx.y] = nsum_count;
        wm_zero_count_sh[threadIdx.y] = wm_zero_count;
    }
    __syncthreads();

    if (threadIdx.x < THREADS_DIM_Y) {
        nsum_count = nsum_count_sh[threadIdx.x];
        nsum_count.x = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.x);
        nsum_count.y = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.y);
        nsum_count.z = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.z);
        nsum_count.w = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.w);
        wm_zero_count = WarpReduceY(WarpReduceY_temp_storage).Sum(wm_zero_count_sh[threadIdx.x]);

        if (th_idx == 0) {
            const XYIdxT idx0_org = nsum_idx0_bound / ((XYIdxT)block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE);
            const XYIdxT idx1_org = nsum_idx1_bound / ((XYIdxT)block_pair_num * THREADS_DIM_Y * WM_WARP_SIZE);

            if (!is_last_wm_bit) {
                if (nsum_count.x > 0) atomicAdd(nsum_buf_test2 + idx0_org - 1, nsum_count.x);
                if (nsum_count.y > 0) atomicAdd(nsum_buf_test2 + idx0_org - 0, nsum_count.y);
                if (nsum_count.z > 0) atomicAdd(nsum_buf_test2 + idx1_org - 1, nsum_count.z);
                if (nsum_count.w > 0) atomicAdd(nsum_buf_test2 + idx1_org - 0, nsum_count.w);
            }
            // Store column WM zero count
            nsum_buf_test2[blockIdx.x] = wm_zero_count;
        }
    }
}
