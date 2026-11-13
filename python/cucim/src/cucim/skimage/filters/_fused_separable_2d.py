# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-launch 2D separable filters.

A separable 2D filter is normally computed as two 1D passes, each a kernel
launch that reads and writes the whole image (``_separable_filtering``). The
kernels here compute both passes in one launch: each block loads its output
tile plus the halo of both passes into shared memory, applies the first pass
there, rounds the intermediate to the output dtype as the two-pass filter's
temporary array is, and applies the second pass.

The results equal those of the two shared-memory 1D passes bit for bit:

- every output sums its taps in the same order (ascending input position)
  with the same expression, ``sum = sum + (W)x * w[k]``, and the kernels are
  compiled with the same (default) FMA contraction;
- the intermediate is ``cast<D>`` of the first pass, D being the output
  dtype, and the second pass reads it as its input;
- boundaries: each axis has its own mode. For non-constant modes the halo is
  loaded through the same index mapping as the 1D kernels, which commutes
  with the other axis's pass. In constant mode the first pass pads its input
  with ``cval_in`` and the intermediate outside the image is ``cval_mid``
  (both cast to the respective input type), which is what two 1D passes
  with those cvals compute.

The plain kernel also filters a pair of axes of a 3D array, plane by plane:
the pair's columns are the contiguous last axis, its rows and the remaining
(batch) axis have arbitrary element strides, and the grid's z dimension runs
over the batch axis. With
one more 1D pass along the third axis this computes a 3D separable filter in
two launches instead of three.

A second kernel computes the gradient magnitude of the 2D edge filters
(``skimage.filters.sobel`` and others): both directional derivatives from one
shared tile, then ``sqrt(d0*d0 + d1*d1)`` with the square and sum rounded
separately, as the elementwise kernels of the per-axis implementation do.
"""

import math

import cupy as cp

from cucim.skimage._vendored._ndimage_filters_core import (
    _ndimage_CAST_FUNCTION,
    _ndimage_includes,
)

from ._separable_filtering import _dtype_char_to_c_types

TILE_X = 64
TILE_Y = 16
BLOCK = (64, 4)
MAX_SHARED_MEMORY = 48 * 1024  # static shared memory per block

# boundary modes (after _check_mode and the 'wrap' -> 'grid-wrap' mapping)
_MODE_CODES = {
    "reflect": 0,
    "grid-mirror": 0,
    "mirror": 1,
    "nearest": 2,
    "wrap": 3,
    "grid-wrap": 3,
    "constant": 4,
}

_preamble = r"""
#include "cupy/carray.cuh"  // for float16
"""

_index_functions = r"""
// mode codes: 0 reflect (grid-mirror), 1 mirror, 2 nearest, 3 wrap, 4 constant
// Maps an index outside [0, n) into it (any distance; not used for constant)
__device__ __forceinline__ int _fused_map(int i, int n, int mode)
{
    if (i >= 0 && i < n) return i;
    if (mode == 0) {
        if (n == 1) return 0;
        int period = 2 * n;
        i %= period;
        if (i < 0) i += period;
        return i < n ? i : period - 1 - i;
    } else if (mode == 1) {
        if (n == 1) return 0;
        int period = 2 * n - 2;
        i %= period;
        if (i < 0) i += period;
        return i < n ? i : period - i;
    } else if (mode == 2) {
        return i < 0 ? 0 : n - 1;
    }
    i %= n;
    if (i < 0) i += n;
    return i;
}
"""


def _plain_src(c):
    """Kernel source for a plain separable filter; ``c`` holds the
    compile-time configuration."""
    return f"""
typedef {c["T"]} T;
typedef {c["W0"]} W0;
typedef {c["W1"]} W1;
typedef {c["D"]} D;
#define TX {TILE_X}
#define TY {TILE_Y}
#define K0 {c["K0"]}
#define A0 {c["A0"]}
#define K1 {c["K1"]}
#define A1 {c["A1"]}
#define FLIP0 {int(c["flip0"])}
#define FLIP1 {int(c["flip1"])}
#define MODE0 {c["mode0"]}
#define MODE1 {c["mode1"]}
#define IN_H (TY + K0 - 1)
#define IN_W (TX + K1 - 1)
#define VB {c["VB"]}

// element offset of (row gy, column gx) in the current plane
#define AT(gy, gx) ((long long)(gy) * row_stride + (gx))

extern "C" __global__ void {c["name"]}(
    const T* __restrict__ src_base, D* __restrict__ dst_base, int H, int W,
    long long row_stride, int B,
    long long batch_stride,
    const W0* __restrict__ w0g, const W1* __restrict__ w1g,
    double cval_in, double cval_mid)
{{
    __shared__ T s_in[IN_H][IN_W];
#if {c["first"]} == 0
    __shared__ D s_mid[TY][IN_W];
#else
    __shared__ D s_mid[IN_H][TX];
#endif
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;
    const int y0 = blockIdx.y * TY - A0;
    const int x0 = blockIdx.x * TX - A1;
    const T pad_in = static_cast<T>(cval_in);
    const D pad_mid = static_cast<D>(cval_mid);

    W0 w0[K0];
    W1 w1[K1];
#pragma unroll
    for (int k = 0; k < K0; k++) w0[k] = w0g[FLIP0 ? K0 - 1 - k : k];
#pragma unroll
    for (int k = 0; k < K1; k++) w1[k] = w1g[FLIP1 ? K1 - 1 - k : k];

    // one plane (a 2D image, or one index along the batch axis) per step
    for (int b = blockIdx.z; b < B; b += gridDim.z) {{
    const T* __restrict__ src = src_base + (long long)b * batch_stride;
    D* __restrict__ dst = dst_base + (long long)b * batch_stride;
    if (b != blockIdx.z) __syncthreads();  // s_in and s_mid are reused

    // load the tile and the halo of both passes
    if (y0 >= 0 && x0 >= 0 && y0 + IN_H <= H && x0 + IN_W <= W) {{
        for (int ly = ty; ly < IN_H; ly += blockDim.y) {{
            const T* row = src + AT(y0 + ly, x0);
            for (int lx = tx; lx < IN_W; lx += blockDim.x) s_in[ly][lx] = row[lx];
        }}
    }} else {{
        for (int ly = ty; ly < IN_H; ly += blockDim.y) {{
            int gy = y0 + ly;
            bool out_y = gy < 0 || gy >= H;
            if (MODE0 != 4) gy = _fused_map(gy, H, MODE0);
            for (int lx = tx; lx < IN_W; lx += blockDim.x) {{
                int gx = x0 + lx;
                bool out_x = gx < 0 || gx >= W;
                if (MODE1 != 4) gx = _fused_map(gx, W, MODE1);
                if ((out_y && MODE0 == 4) || (out_x && MODE1 == 4)) {{
                    s_in[ly][lx] = pad_in;
                }} else {{
                    s_in[ly][lx] = src[AT(gy, gx)];
                }}
            }}
        }}
    }}
    __syncthreads();

#if {c["first"]} == 0
    // axis 0 first: VB consecutive outputs of one column per thread, the
    // inputs streamed through registers (each output still sums its taps
    // in order k = 0 .. K0 - 1)
    for (int item = tid; item < IN_W * (TY / VB); item += nthreads) {{
        int g = item / IN_W;
        int lx = item - g * IN_W;
        int ly0 = g * VB;
        int gx = x0 + lx;
        if (MODE1 == 4 && (gx < 0 || gx >= W)) {{
#pragma unroll
            for (int j = 0; j < VB; j++) s_mid[ly0 + j][lx] = pad_mid;
            continue;
        }}
        W0 acc[VB];
#pragma unroll
        for (int j = 0; j < VB; j++) acc[j] = static_cast<W0>(0);
#pragma unroll
        for (int i = 0; i < VB + K0 - 1; i++) {{
            W0 v = static_cast<W0>(s_in[ly0 + i][lx]);
#pragma unroll
            for (int j = 0; j < VB; j++) {{
                const int k = i - j;
                if (k >= 0 && k < K0) acc[j] = acc[j] + v * w0[k];
            }}
        }}
#pragma unroll
        for (int j = 0; j < VB; j++) s_mid[ly0 + j][lx] = cast<D>(acc[j]);
    }}
    __syncthreads();
    for (int ly = ty; ly < TY; ly += blockDim.y) {{
        int gy = blockIdx.y * TY + ly;
        if (gy >= H) break;
        for (int lx = tx; lx < TX; lx += blockDim.x) {{
            int gx = blockIdx.x * TX + lx;
            if (gx >= W) break;
            W1 sum = static_cast<W1>(0);
#pragma unroll
            for (int k = 0; k < K1; k++) {{
                sum = sum + static_cast<W1>(s_mid[ly][lx + k]) * w1[k];
            }}
            dst[AT(gy, gx)] = cast<D>(sum);
        }}
    }}
#else
    // axis 1 first, over all rows of the tile and halo
    for (int ly = ty; ly < IN_H; ly += blockDim.y) {{
        int gy = y0 + ly;
        bool pad = MODE0 == 4 && (gy < 0 || gy >= H);
        for (int lx = tx; lx < TX; lx += blockDim.x) {{
            if (pad) {{
                s_mid[ly][lx] = pad_mid;
                continue;
            }}
            W1 sum = static_cast<W1>(0);
#pragma unroll
            for (int k = 0; k < K1; k++) {{
                sum = sum + static_cast<W1>(s_in[ly][lx + k]) * w1[k];
            }}
            s_mid[ly][lx] = cast<D>(sum);
        }}
    }}
    __syncthreads();
    for (int ly = ty; ly < TY; ly += blockDim.y) {{
        int gy = blockIdx.y * TY + ly;
        if (gy >= H) break;
        for (int lx = tx; lx < TX; lx += blockDim.x) {{
            int gx = blockIdx.x * TX + lx;
            if (gx >= W) break;
            W0 sum = static_cast<W0>(0);
#pragma unroll
            for (int k = 0; k < K0; k++) {{
                sum = sum + static_cast<W0>(s_mid[ly + k][lx]) * w0[k];
            }}
            dst[AT(gy, gx)] = cast<D>(sum);
        }}
    }}
#endif
    }}  // planes
}}
"""


def _magnitude_src(c):
    """Kernel source for the 2D edge magnitude: d0 = edge along axis 0 then
    smoothing along axis 1, d1 = edge along axis 1 then smoothing along axis
    0 (K taps each, anchor K // 2), output sqrt(d0*d0 + d1*d1)."""
    mul, add = (
        ("__fmul_rn", "__fadd_rn")
        if c["T"] == "float"
        else (
            "__dmul_rn",
            "__dadd_rn",
        )
    )
    return f"""
typedef {c["T"]} T;
#define TX {TILE_X}
#define TY {TILE_Y}
#define K {c["K"]}
#define A (K / 2)
#define FLIP {int(c["flip"])}
#define MODE {c["mode"]}
#define IN_H (TY + K - 1)
#define IN_W (TX + K - 1)

extern "C" __global__ void {c["name"]}(
    const T* __restrict__ src, T* __restrict__ dst, int H, int W,
    const T* __restrict__ eg, const T* __restrict__ sg,
    double cval_in, double cval_mid)
{{
    __shared__ T s_in[IN_H][IN_W];
    __shared__ T s_a[TY][IN_W];
    __shared__ T s_b[IN_H][TX];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int y0 = blockIdx.y * TY - A;
    const int x0 = blockIdx.x * TX - A;
    const T pad_in = static_cast<T>(cval_in);
    const T pad_mid = static_cast<T>(cval_mid);

    if (y0 >= 0 && x0 >= 0 && y0 + IN_H <= H && x0 + IN_W <= W) {{
        for (int ly = ty; ly < IN_H; ly += blockDim.y) {{
            const T* row = src + (long long)(y0 + ly) * W + x0;
            for (int lx = tx; lx < IN_W; lx += blockDim.x) s_in[ly][lx] = row[lx];
        }}
    }} else {{
        for (int ly = ty; ly < IN_H; ly += blockDim.y) {{
            int gy = y0 + ly;
            bool out_y = gy < 0 || gy >= H;
            if (MODE != 4) gy = _fused_map(gy, H, MODE);
            for (int lx = tx; lx < IN_W; lx += blockDim.x) {{
                int gx = x0 + lx;
                bool out_x = gx < 0 || gx >= W;
                if (MODE != 4) gx = _fused_map(gx, W, MODE);
                if (MODE == 4 && (out_y || out_x)) {{
                    s_in[ly][lx] = pad_in;
                }} else {{
                    s_in[ly][lx] = src[(long long)gy * W + gx];
                }}
            }}
        }}
    }}
    __syncthreads();

    T e[K];
    T s[K];
#pragma unroll
    for (int k = 0; k < K; k++) {{
        e[k] = eg[FLIP ? K - 1 - k : k];
        s[k] = sg[FLIP ? K - 1 - k : k];
    }}

    // edge along axis 0 (all columns of tile and halo)
    for (int ly = ty; ly < TY; ly += blockDim.y) {{
        for (int lx = tx; lx < IN_W; lx += blockDim.x) {{
            int gx = x0 + lx;
            if (MODE == 4 && (gx < 0 || gx >= W)) {{
                s_a[ly][lx] = pad_mid;
                continue;
            }}
            T sum = static_cast<T>(0);
#pragma unroll
            for (int k = 0; k < K; k++) sum = sum + s_in[ly + k][lx] * e[k];
            s_a[ly][lx] = sum;
        }}
    }}
    // edge along axis 1 (all rows of tile and halo)
    for (int ly = ty; ly < IN_H; ly += blockDim.y) {{
        int gy = y0 + ly;
        bool pad = MODE == 4 && (gy < 0 || gy >= H);
        for (int lx = tx; lx < TX; lx += blockDim.x) {{
            if (pad) {{
                s_b[ly][lx] = pad_mid;
                continue;
            }}
            T sum = static_cast<T>(0);
#pragma unroll
            for (int k = 0; k < K; k++) sum = sum + s_in[ly][lx + k] * e[k];
            s_b[ly][lx] = sum;
        }}
    }}
    __syncthreads();
    for (int ly = ty; ly < TY; ly += blockDim.y) {{
        int gy = blockIdx.y * TY + ly;
        if (gy >= H) break;
        for (int lx = tx; lx < TX; lx += blockDim.x) {{
            int gx = blockIdx.x * TX + lx;
            if (gx >= W) break;
            T d0 = static_cast<T>(0);
            T d1 = static_cast<T>(0);
#pragma unroll
            for (int k = 0; k < K; k++) d0 = d0 + s_a[ly][lx + k] * s[k];
#pragma unroll
            for (int k = 0; k < K; k++) d1 = d1 + s_b[ly + k][lx] * s[k];
            // separate elementwise kernels in the per-axis implementation:
            // no contraction across them
            T m = {add}({mul}(d0, d0), {mul}(d1, d1));
            dst[(long long)gy * W + gx] = sqrt(m);
        }}
    }}
}}
"""


@cp.memoize(for_each_device=True)
def _get_kernel(kind, key):
    c = dict(key)
    c["name"] = f"cucim_fused2d_{kind}"
    body = _plain_src(c) if kind == "plain" else _magnitude_src(c)
    code = _preamble + _ndimage_includes + _ndimage_CAST_FUNCTION
    code += _index_functions + body
    module = cp.RawModule(code=code)
    return module.get_function(c["name"])


def _outputs_per_thread(w0_dtype):
    # registers per thread: VB accumulators; fewer for double
    return 4 if cp.dtype(w0_dtype).itemsize >= 8 else 8


def plain_shared_memory(k0, k1, first, image_dtype, output_dtype):
    """Static shared memory (bytes) of the plain kernel."""
    in_h, in_w = TILE_Y + k0 - 1, TILE_X + k1 - 1
    mid = TILE_Y * in_w if first == 0 else in_h * TILE_X
    return (
        in_h * in_w * cp.dtype(image_dtype).itemsize
        + mid * cp.dtype(output_dtype).itemsize
    )


def magnitude_shared_memory(k, dtype):
    """Static shared memory (bytes) of the magnitude kernel."""
    in_h, in_w = TILE_Y + k - 1, TILE_X + k - 1
    return (in_h * in_w + TILE_Y * in_w + in_h * TILE_X) * cp.dtype(
        dtype
    ).itemsize


# the grid's z dimension (planes beyond it are looped over in the kernel)
_MAX_GRID_Z = 65535


def correlate_2d(
    image,
    weights0,
    weights1,
    output,
    *,
    first_axis,
    modes,
    cval_in,
    cval_mid,
    anchors,
    flips=(False, False),
    axes=(0, 1),
):
    """Both 1D passes of a separable 2D filter in one launch.

    ``image`` is a C-contiguous 2D or 3D array. ``axes`` names the pair of
    axes filtered, called axis 0 and axis 1 of the filter below (in that
    order); axis 1 must be the last axis. For 3D input the remaining axis is
    iterated over, each plane filtered independently. ``weights0``/``weights1`` are 1D device arrays
    (of the accumulation types of the two passes) applied along filter axis
    0 and filter axis 1, in the order given by ``first_axis``; their taps at
    index k read input position ``i - anchor + k`` (or the reversed weight
    when the corresponding flip is True, i.e. a convolution). ``output``
    (C-contiguous, of the same shape, not overlapping ``image``) receives the
    result in its dtype, which is also the intermediate dtype. ``modes``
    holds the boundary mode of each filter axis.
    """
    ax0, ax1 = (a % image.ndim for a in axes)
    if ax1 != image.ndim - 1 or ax0 == ax1:
        raise ValueError("the second axis must be the last one")
    H, W = image.shape[ax0], image.shape[ax1]
    itemsize = image.dtype.itemsize
    strides = [s // itemsize for s in image.strides]
    if output.strides != tuple(s * output.dtype.itemsize for s in strides):
        raise ValueError("output must have the same layout as image")
    if image.ndim == 3:
        (bax,) = {0, 1, 2} - {ax0, ax1}
        B, batch_stride = image.shape[bax], strides[bax]
    else:
        B, batch_stride = 1, 0
    row_stride = strides[ax0]
    key = (
        ("T", _dtype_char_to_c_types[image.dtype.char]),
        ("W0", _dtype_char_to_c_types[weights0.dtype.char]),
        ("W1", _dtype_char_to_c_types[weights1.dtype.char]),
        ("D", _dtype_char_to_c_types[output.dtype.char]),
        ("K0", int(weights0.size)),
        ("A0", int(anchors[0])),
        ("K1", int(weights1.size)),
        ("A1", int(anchors[1])),
        ("flip0", bool(flips[0])),
        ("flip1", bool(flips[1])),
        ("mode0", _MODE_CODES[modes[0]]),
        ("mode1", _MODE_CODES[modes[1]]),
        ("first", int(first_axis)),
        ("VB", _outputs_per_thread(weights0.dtype)),
    )
    kernel = _get_kernel("plain", key)
    kernel(
        (math.ceil(W / TILE_X), math.ceil(H / TILE_Y), min(B, _MAX_GRID_Z)),
        BLOCK,
        (
            image,
            output,
            cp.int32(H),
            cp.int32(W),
            cp.int64(row_stride),
            cp.int32(B),
            cp.int64(batch_stride),
            weights0,
            weights1,
            float(cval_in),
            float(cval_mid),
        ),
    )
    return output


def edge_magnitude_2d(
    image, edge, smooth, output, *, mode, cval_in, cval_mid, flip=True
):
    """``sqrt(d0*d0 + d1*d1)`` of the 2D edge filter given by ``edge`` and
    ``smooth`` (same length, applied as convolutions when ``flip``), in one
    launch; image, weights and output share one float dtype."""
    H, W = image.shape
    key = (
        ("T", _dtype_char_to_c_types[image.dtype.char]),
        ("K", int(edge.size)),
        ("flip", bool(flip)),
        ("mode", _MODE_CODES[mode]),
    )
    kernel = _get_kernel("magnitude", key)
    kernel(
        (math.ceil(W / TILE_X), math.ceil(H / TILE_Y)),
        BLOCK,
        (
            image,
            output,
            cp.int32(H),
            cp.int32(W),
            edge,
            smooth,
            float(cval_in),
            float(cval_mid),
        ),
    )
    return output


@cp.memoize(for_each_device=True)
def _fp64_ratio(device_id):
    """FP32:FP64 throughput ratio reported by the device (e.g. 2 for
    data-center GPUs, 32 or 64 for most others)."""
    try:
        return int(
            cp.cuda.runtime.deviceGetAttribute(
                cp.cuda.runtime.cudaDevAttrSingleToDoublePrecisionPerfRatio,
                device_id,
            )
        )
    except Exception:  # unknown: assume low FP64 throughput
        return 64


# float64 dispatch on GPUs with low FP64 throughput: the fused kernel repeats
# the first pass for the halo of the second, which costs more than the
# launch and memory traffic it saves unless the radius is small. On an RTX
# A6000 (FP32:FP64 = 64) radius 1 and 2 were up to 1.8x faster at any size,
# radius 4 (Gaussian sigma = 1) was neutral and radius 8 or more slower.
_FLOAT64_MAX_RADIUS = 2
_FAST_FP64_RATIO = 4


_fast_fp64_devices = {}


def fast_fp64():
    """Whether the current device has high FP64 throughput (cached)."""
    device_id = cp.cuda.runtime.getDevice()
    fast = _fast_fp64_devices.get(device_id)
    if fast is None:
        fast = _fp64_ratio(device_id) <= _FAST_FP64_RATIO
        _fast_fp64_devices[device_id] = fast
    return fast


def float64_preferred(shape, max_radius):
    """Whether the fused kernel is expected to be faster for float64
    accumulation (``shape`` is kept for size-dependent rules)."""
    return max_radius <= _FLOAT64_MAX_RADIUS or fast_fp64()
