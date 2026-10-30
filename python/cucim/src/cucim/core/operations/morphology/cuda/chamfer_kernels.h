/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Kernels of the taxicab and chessboard (chamfer) distance transforms.
//
// NDIM (the number of dimensions of the image) and BIG (the value that marks
// "no background found yet") are defined by the Python code before this file.
//
// Taxicab: one pass per axis over every line of the array, computing
//     g(x) = min over x' of ( |x - x'| + f(x') )
// as a forward and a backward sweep (g[i] = min(f[i], g[i-1] + 1)).
//
// Chessboard: the chessboard distance of a pixel is the smallest r for which
// the box of radius r around it contains a background pixel. The number of
// background pixels in a box is read from a summed-area table, and r is found
// by a binary search between bounds that follow from the taxicab distance d1
// (ceil(d1 / NDIM) <= r <= d1).
//
// This is an initial implementation: the line passes use one thread per line,
// which is not coalesced for the last (contiguous) axis.

// Number of background pixels in the box [lo, hi) of the image, from the
// summed-area table `sat` (an array with one more entry than the image along
// every axis, with a zero at index 0). Inclusion-exclusion over the 2**NDIM
// corners of the box.
__device__ long long box_count(const int *sat, const long long *sat_strides,
                               const long long *lo, const long long *hi)
{
    long long total = 0;
    for (int mask = 0; mask < (1 << NDIM); ++mask) {
        long long offset = 0;
        int sign = 1;
        for (int d = 0; d < NDIM; ++d) {
            if ((mask >> d) & 1) {
                offset += hi[d] * sat_strides[d];
            } else {
                offset += lo[d] * sat_strides[d];
                sign = -sign;
            }
        }
        total += sign * static_cast<long long>(sat[offset]);
    }
    return total;
}

// The distance that is stored by the last pass: with `final`, the pixels that no
// background pixel reached get the value -1 (as in SciPy).
__device__ __forceinline__ int finished(int v, int final)
{
    return (final && v >= BIG / 2) ? -1 : v;
}

extern "C" {

// One pass of the taxicab transform along one axis. `dist` has the shape
// (outer, n, inner) and is modified in place. There is one thread per line.
__global__ void l1_pass(int *dist, long long n, long long inner,
                        long long lines, int final)
{
    long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= lines)
        return;
    long long base = (t / inner) * n * inner + t % inner;

    int g = BIG;
    for (long long i = 0; i < n; ++i) {
        long long k = base + i * inner;
        g = min(dist[k], g + 1);
        dist[k] = g;
    }
    g = BIG;
    for (long long i = n - 1; i >= 0; --i) {
        long long k = base + i * inner;
        g = min(dist[k], g + 1);
        dist[k] = finished(g, final);
    }
}

// The same, and `feature` (the flat index of a nearest background pixel) is
// propagated with the distance.
__global__ void l1_pass_feature(int *__restrict__ dist,
                                int *__restrict__ feature, long long n,
                                long long inner, long long lines, int final)
{
    long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= lines)
        return;
    long long base = (t / inner) * n * inner + t % inner;

    // The loads do not depend on the running values and the choices are
    // selects (no branch that guards a load), which is much faster than loading
    // the feature only when it is needed.
    int g = BIG, gf = -1;
    for (long long i = 0; i < n; ++i) {
        long long k = base + i * inner;
        int v = dist[k];
        int f = feature[k];
        bool take = v <= g + 1;
        g = take ? v : g + 1;
        gf = take ? f : gf;
        dist[k] = g;
        feature[k] = gf;
    }
    g = BIG;
    for (long long i = n - 1; i >= 0; --i) {
        long long k = base + i * inner;
        int v = dist[k];
        int f = feature[k];
        bool take = g + 1 < v;
        v = take ? g + 1 : v;
        f = take ? gf : f;
        dist[k] = finished(v, final);
        feature[k] = f;
        g = v;
        gf = f;
    }
}

// The passes for the contiguous axis (inner == 1): one warp per line, so that
// the reads and writes are coalesced. The forward sweep g(i) = min(f(i),
// g(i-1) + 1) is, with u(i) = f(i) - i, g(i) = i + min over j <= i of u(j),
// a prefix minimum that is computed for 32 elements at a time with shuffles
// and a carry from the previous tiles. The backward sweep is the same for
// v(i) = f(i) + i with a suffix minimum: g(i) = min over j >= i of v(j) - i.
// A block has WARPS_PER_BLOCK warps (lines).
#define FULL_MASK 0xffffffffu

__global__ void l1_pass_warp(int *dist, long long n, long long lines,
                             int final)
{
    long long line = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) +
                     threadIdx.x / 32;
    if (line >= lines)
        return;
    int lane = threadIdx.x & 31;
    int *row = dist + line * n;

    int carry = BIG;
    for (long long base = 0; base < n; base += 32) {
        long long i = base + lane;
        int w = (i < n) ? row[i] - static_cast<int>(i) : 0x7fffffff;
        for (int k = 1; k < 32; k <<= 1) {
            int other = __shfl_up_sync(FULL_MASK, w, k);
            if (lane >= k)
                w = min(w, other);
        }
        w = min(w, carry);
        carry = __shfl_sync(FULL_MASK, w, 31);
        if (i < n)
            row[i] = w + static_cast<int>(i);
    }

    carry = 0x7fffffff;
    for (long long base = ((n - 1) / 32) * 32; base >= 0; base -= 32) {
        long long i = base + lane;
        int w = (i < n) ? row[i] + static_cast<int>(i) : 0x7fffffff;
        for (int k = 1; k < 32; k <<= 1) {
            int other = __shfl_down_sync(FULL_MASK, w, k);
            if (lane + k < 32)
                w = min(w, other);
        }
        w = min(w, carry);
        carry = __shfl_sync(FULL_MASK, w, 0);
        if (i < n)
            row[i] = finished(w - static_cast<int>(i), final);
    }
}

// The same with the feature: the distance term and the flat index of the
// background pixel are packed in one 64-bit key (distance term in the upper
// half, offset to be non-negative), so that a minimum of the keys is the
// minimum of the distance.
__global__ void l1_pass_feature_warp(int *dist, int *feature, long long n,
                                     long long lines, int final)
{
    typedef unsigned long long key_t_;
    const key_t_ NONE = ~0ULL;
    long long line = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) +
                     threadIdx.x / 32;
    if (line >= lines)
        return;
    int lane = threadIdx.x & 31;
    int *row = dist + line * n;
    int *frow = feature + line * n;

    key_t_ carry = NONE;
    for (long long base = 0; base < n; base += 32) {
        long long i = base + lane;
        key_t_ w = NONE;
        if (i < n) {
            w = (static_cast<key_t_>(row[i] - static_cast<int>(i) +
                                     static_cast<int>(n)) << 32) |
                static_cast<unsigned int>(frow[i]);
        }
        for (int k = 1; k < 32; k <<= 1) {
            key_t_ other = __shfl_up_sync(FULL_MASK, w, k);
            if (lane >= k)
                w = min(w, other);
        }
        w = min(w, carry);
        carry = __shfl_sync(FULL_MASK, w, 31);
        if (i < n) {
            row[i] = static_cast<int>(w >> 32) - static_cast<int>(n) +
                     static_cast<int>(i);
            frow[i] = static_cast<int>(w & 0xffffffffULL);
        }
    }

    carry = NONE;
    for (long long base = ((n - 1) / 32) * 32; base >= 0; base -= 32) {
        long long i = base + lane;
        key_t_ w = NONE;
        if (i < n) {
            w = (static_cast<key_t_>(row[i] + static_cast<int>(i)) << 32) |
                static_cast<unsigned int>(frow[i]);
        }
        for (int k = 1; k < 32; k <<= 1) {
            key_t_ other = __shfl_down_sync(FULL_MASK, w, k);
            if (lane + k < 32)
                w = min(w, other);
        }
        w = min(w, carry);
        carry = __shfl_sync(FULL_MASK, w, 0);
        if (i < n) {
            row[i] = finished(static_cast<int>(w >> 32) - static_cast<int>(i),
                              final);
            frow[i] = static_cast<int>(w & 0xffffffffULL);
        }
    }
}

// Running sums along one axis, in place (the passes that build the summed-area
// table). `data` has the shape (outer, n, inner); one thread per line.
__global__ void sum_pass(int *data, long long n, long long inner,
                         long long lines)
{
    long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= lines)
        return;
    long long base = (t / inner) * n * inner + t % inner;
    int total = 0;
    for (long long i = 0; i < n; ++i) {
        long long k = base + i * inner;
        total += data[k];
        data[k] = total;
    }
}

// The same for the contiguous axis (inner == 1): one warp per line.
__global__ void sum_pass_warp(int *data, long long n, long long lines)
{
    long long line = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) +
                     threadIdx.x / 32;
    if (line >= lines)
        return;
    int lane = threadIdx.x & 31;
    int *row = data + line * n;
    int carry = 0;
    for (long long base = 0; base < n; base += 32) {
        long long i = base + lane;
        int w = (i < n) ? row[i] : 0;
        for (int k = 1; k < 32; k <<= 1) {
            int other = __shfl_up_sync(FULL_MASK, w, k);
            if (lane >= k)
                w += other;
        }
        w += carry;
        carry = __shfl_sync(FULL_MASK, w, 31);
        if (i < n)
            row[i] = w;
    }
}

// Convert the flat (C-order) index of the feature of every pixel into
// coordinates, written to indices[d * size + i].
__global__ void flat_to_coordinates(const int *feature, int *indices,
                                    const long long *shape, long long size)
{
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    long long rest = feature[i];
    for (int d = NDIM - 1; d >= 0; --d) {
        indices[d * size + i] = static_cast<int>(rest % shape[d]);
        rest /= shape[d];
    }
}

// Chessboard distance of every pixel. `dist1` is the taxicab distance (BIG or
// more if the image has no background pixel), `sat` the summed-area table of
// the background. If `return_indices` is not 0, the coordinates of a nearest
// background pixel are written to indices[d * size + i].
__global__ void chessboard(const int *sat, const int *dist1, int *out,
                           int *indices, int return_indices,
                           const long long *shape,
                           const long long *sat_strides, long long size)
{
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size)
        return;

    long long p[NDIM], lo[NDIM], hi[NDIM], site[NDIM];
    long long rest = i;
    for (int d = NDIM - 1; d >= 0; --d) {
        p[d] = rest % shape[d];
        rest /= shape[d];
        site[d] = p[d];
    }

    int d1 = dist1[i];
    int result;
    if (d1 >= BIG / 2) {
        // no background pixel anywhere: the pixel is its own feature
        result = -1;
    } else if (d1 == 0) {
        result = 0;
    } else {
        int r_lo = (d1 + NDIM - 1) / NDIM;
        int r_hi = d1;
        while (r_lo < r_hi) {
            int r = (r_lo + r_hi) / 2;
            for (int d = 0; d < NDIM; ++d) {
                lo[d] = max(p[d] - r, 0LL);
                hi[d] = min(p[d] + r + 1, shape[d]);
            }
            if (box_count(sat, sat_strides, lo, hi) > 0)
                r_hi = r;
            else
                r_lo = r + 1;
        }
        result = r_lo;

        if (return_indices) {
            // Find a background pixel in the box of radius `result` with one
            // binary search per axis. The first background pixel along the
            // axis (in the box) is found with the counts of the part of the
            // box before a coordinate, then the axis is fixed to it.
            for (int d = 0; d < NDIM; ++d) {
                lo[d] = max(p[d] - result, 0LL);
                hi[d] = min(p[d] + result + 1, shape[d]);
            }
            for (int a = 0; a < NDIM; ++a) {
                long long first = lo[a], last = hi[a] - 1;
                long long axis_hi = hi[a];
                while (first < last) {
                    long long mid = (first + last) / 2;
                    hi[a] = mid + 1;
                    if (box_count(sat, sat_strides, lo, hi) > 0)
                        last = mid;
                    else
                        first = mid + 1;
                }
                hi[a] = axis_hi;
                site[a] = first;
                lo[a] = first;
                hi[a] = first + 1;
            }
        }
    }

    out[i] = result;
    if (return_indices) {
        for (int d = 0; d < NDIM; ++d)
            indices[d * size + i] = static_cast<int>(site[d]);
    }
}

}  // extern "C"
