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
// Chessboard: the distance along the contiguous axis, then for every other axis
// the transform min over x' of max(|x - x'|, f(x')) (see minmax_pass).
//
// The passes along the contiguous axis use one warp per line, the others one
// thread per line (neighboring threads handle neighboring lines).

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
// propagated with the distance. With `first` (the first pass) the feature of
// every pixel is its own flat index and `feature` is not read.
__global__ void l1_pass_feature(int *__restrict__ dist,
                                int *__restrict__ feature, long long n,
                                long long inner, long long lines, int final, int first)
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
        int f = first ? static_cast<int>(k) : feature[k];
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
                                     long long lines, int final, int first)
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
                static_cast<unsigned int>(first ? static_cast<int>(line * n + i)
                                                : frow[i]);
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

// One pass of the chessboard transform along one axis:
//     out(x) = min over x' of max(|x - x'|, f(x'))
// for every line of an array of the shape (outer, n, inner), one thread per
// line. The sites before x (x' <= x) and the sites after x are handled by a
// forward and a backward sweep. In the forward sweep a site x' is dominated, for
// this x and every later x, by a later site x'' with f(x'') <= f(x'), because
// it is closer. The sites that are not dominated are kept on a stack, with f
// increasing from the bottom to the top (the bottom sites are the farthest ones
// and have the smallest f). The value max(x - x', f(x')) over the stack falls
// and then rises (x - x' falls and f rises along the stack), and its minimum
// moves towards the top as x increases, so a pointer to the minimum only moves
// forward (and is clamped when sites are popped): O(n) per line.
//
// `stack` is scratch memory with two ints per element of the array. The sweeps read `f`
// and write the result to `out` (the forward sweep writes the minimum over the
// sites before x, the backward sweep takes the minimum with it), so `f` must not
// be the same array as `out`. If `with_feature` is not 0, the feature of the
// chosen site (`fin`, the flat index of a background pixel) is written to
// `fout`; at equal distances the sites before x win.
__global__ void minmax_pass(const int *__restrict__ f, int *__restrict__ out,
                            const int *__restrict__ fin,
                            int *__restrict__ fout, int2 *__restrict__ stack,
                            long long n, long long inner, long long lines,
                            int final, int with_feature)
{
    long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= lines)
        return;
    long long base = (t / inner) * n * inner + t % inner;

    // The stack holds (position, f at the position).
    int top = 0, p = 0;
    for (int i = 0; i < n; ++i) {
        int fx = f[base + i * inner];
        while (top > 0 && stack[base + (top - 1) * inner].y >= fx)
            --top;
        int limit = top > 0 ? top - 1 : 0;
        if (p > limit)
            p = limit;
        stack[base + top * inner] = make_int2(i, fx);
        ++top;
        int2 site = stack[base + p * inner];
        int cur = max(i - site.x, site.y);
        while (p + 1 < top) {
            int2 next = stack[base + (p + 1) * inner];
            int v2 = max(i - next.x, next.y);
            if (v2 > cur)
                break;
            cur = v2;
            site = next;
            ++p;
        }
        out[base + i * inner] = cur;
        if (with_feature)
            fout[base + i * inner] =
                fin[base + static_cast<long long>(site.x) * inner];
    }

    top = 0;
    p = 0;
    for (int i = n - 1; i >= 0; --i) {
        int fx = f[base + i * inner];
        while (top > 0 && stack[base + (top - 1) * inner].y >= fx)
            --top;
        int limit = top > 0 ? top - 1 : 0;
        if (p > limit)
            p = limit;
        stack[base + top * inner] = make_int2(i, fx);
        ++top;
        int2 site = stack[base + p * inner];
        int cur = max(site.x - i, site.y);
        while (p + 1 < top) {
            int2 next = stack[base + (p + 1) * inner];
            int v2 = max(next.x - i, next.y);
            if (v2 > cur)
                break;
            cur = v2;
            site = next;
            ++p;
        }
        long long k = base + i * inner;
        int before = out[k];
        if (cur < before) {
            if (with_feature)
                fout[k] = fin[base + static_cast<long long>(site.x) * inner];
        } else {
            cur = before;
        }
        out[k] = finished(cur, final);
    }
}

}  // extern "C"
