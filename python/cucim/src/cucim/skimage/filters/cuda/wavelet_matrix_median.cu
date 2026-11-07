/*
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 *
 * Wavelet matrix query kernel for the 2D rank (median) filter.
 *
 * The matrices are built by the kernels in wavelet_matrix_construct.cu. Based
 * on the 2D median algorithm described in:
 *
 * Y. Moroto and N. Umetani., "Constant Time Median Filter Using 2D Wavelet
 * Matrix," in ACM Trans. Graph. Volume 41, Issue 6, Article No.: 267,
 * December 2022. https://doi.org/10.1145/3550454.3555512
 *
 * The PR adapting that code to OpenCV was:
 * https://github.com/opencv/opencv_contrib/pull/3627
 *
 * The OpenCV contribution above was made by the authors of the publication.
 * Their original MIT-licensed source code is at:
 * https://github.com/TumoiYorozu/WMatrixMedian
 *
 * The code here is NOT a direct copy of the OpenCV code, but was implemented
 * using it as a reference.
 *
 * Layout: images are C-contiguous; axis 0 (rows) is Y and axis 1 (columns) is
 * X; ry and rx are the filter radii along axes 0 and 1.
 *
 * This file cannot be compiled standalone: the Python code prepends the
 * macro definitions and wavelet_matrix_common.cuh (see _median_wavelet.py).
 */

/*
 * Each thread computes the element of rank `k` (0-based, ascending order; the
 * median is k = window_size / 2) of the (2*ry+1) x (2*rx+1) window of one
 * output pixel. The input was padded by ry rows and rx columns on each side
 * before the matrices were built, so output pixel (y, x) corresponds to the
 * window [y, y + 2*ry + 1) x [x, x + 2*rx + 1) of the padded image, which has
 * `padded_W` columns.
 *
 * The value matrix has one level per value bit (most significant first). At
 * each level the number of window elements whose current bit is 0 is found
 * from rank queries on the rows y_a and y_b that bound the window, restricted
 * to columns [xa, xb) with the column wavelet matrix stored for that level
 * (inclusion-exclusion on the four window corners, `d` below). If k < d the
 * result bit is 0 and the search continues in the 0 half; otherwise the bit is
 * 1, k is reduced by d, and the search continues in the 1 half.
 *
 * Parameters:
 *   H, W:       output size
 *   padded_W:   width of the padded image the matrices were built on
 *   ry, rx:     filter radii along axes 0 and 1
 *   k:          rank to select, in [0, (2*ry+1)*(2*rx+1))
 *   res_cu:     output (H x W, C-contiguous), values in the matrix type
 *   wm_nbit_bp: column matrices, starting at the first level of the last
 *               value level (the pointers are walked towards lower addresses)
 *   bv_nbit_bp: value matrix, starting at its most significant level
 *   nsum_pos:   index of the extra BlockT holding the total of 0 bits
 *   bv_block_h_byte_div32: bytes per matrix level divided by 32
 *   w_bit_len, val_bit_len: number of column and value bits
 */
extern "C" __global__ void wavelet_median2d(
    const int H,
    const int W,
    const int padded_W,
    const int ry,
    const int rx,
    const unsigned int k_in,
    WM_VAL_T* __restrict__ res_cu,
    const BlockT* __restrict__ wm_nbit_bp,
    const unsigned int nsum_pos,
    const unsigned int bv_block_h_byte_div32,
    const BlockT* __restrict__ bv_nbit_bp,
    const unsigned char w_bit_len,
    const unsigned char val_bit_len
) {
    const int y = blockIdx.y * WM_THREADS_H + threadIdx.y;
    if (y >= H) return;
    const int x = blockIdx.x * WM_THREADS_W + threadIdx.x;
    if (x >= W) return;

    // Window bounds in the padded image (rows are converted to bit positions)
    const XIdxT xa = x;
    const XIdxT xb = x + 2 * rx + 1;
    XYIdxT ya = (XYIdxT)y * padded_W;
    XYIdxT yb = (XYIdxT)(y + 2 * ry + 1) * padded_W;
    XYIdxT k = k_in;

    WM_VAL_T res = 0;

    const size_t level_bytes = bv_block_h_byte_div32 * 32ull;
    const BlockT* bv_ptr = bv_nbit_bp;
    const BlockT* wm_ptr = wm_nbit_bp;

    for (int h = val_bit_len; h-- > 0; ) {
        const XYIdxT top0 = wavelet_rank0(ya, bv_ptr);
        const XYIdxT bot0 = wavelet_rank0(yb, bv_ptr);

        XYIdxT l_ya_xa = top0;
        XYIdxT l_yb_xa = bot0;
        XYIdxT l_ya_xb = top0;
        XYIdxT l_yb_xb = bot0;

        // Number of window elements with a 0 at the current value bit
        XYIdxT d = 0;

        const BlockT* wm_level_ptr = wm_ptr;
        for (int j = w_bit_len; j-- > 0; ) {
            const XYIdxT zeros = wm_level_ptr[nsum_pos].nsum;

            const XYIdxT l_ya_xa_rank0 = wavelet_rank0(l_ya_xa, wm_level_ptr);
            const XYIdxT l_ya_xb_rank0 = wavelet_rank0(l_ya_xb, wm_level_ptr);
            const XYIdxT l_yb_xb_rank0 = wavelet_rank0(l_yb_xb, wm_level_ptr);
            const XYIdxT l_yb_xa_rank0 = wavelet_rank0(l_yb_xa, wm_level_ptr);

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

            wm_level_ptr = (const BlockT*)((const char*)wm_level_ptr - level_bytes);
        }

        const XYIdxT bv_h_zeros = bv_ptr[nsum_pos].nsum;

        if (k < d) {
            // The result has a 0 at this bit: continue in the 0 half
            ya = top0;
            yb = bot0;
        } else {
            // The result has a 1 at this bit: continue in the 1 half
            k -= d;
            res |= (WM_VAL_T)1 << h;
            ya += bv_h_zeros - top0;
            yb += bv_h_zeros - bot0;
        }

        bv_ptr = (const BlockT*)((const char*)bv_ptr - level_bytes);
        // The column matrices of the next lower value level
        wm_ptr = (const BlockT*)((const char*)wm_ptr - level_bytes * w_bit_len);
    }

    res_cu[(XYIdxT)y * W + x] = res;
}
