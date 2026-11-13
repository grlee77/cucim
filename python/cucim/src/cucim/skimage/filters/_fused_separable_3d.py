# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-launch 3D separable filters (z-marching kernels).

A separable 3D filter is three 1D passes. The kernels here compute all three
in one launch. Each block owns a TY x TX tile of the (y, x) plane and steps
through a range of z ("z-marching", the usual way of computing 3D stencils on
GPUs). Every thread owns a fixed set of (y, x) positions and keeps, for each,
the last K values along z in registers (a queue): one new plane enters the
queue per step and the pass along z is computed from it. The passes in the
plane run in shared memory, as in the fused 2D kernel.

A *chain* is the sequence of three passes of one separable filter. The pass
along z (axis 0) may come first or second:

- z first: the queue holds input values; the two passes in the plane are
  applied to the result of the z pass;
- z second: the first pass (along y or x) is applied to each new input plane
  and its results enter the queue; the third pass is applied to the result of
  the z pass.

Each pass computes ``sum = fma((W)x, w[k], sum)`` over its taps in order (the
per-axis kernels' ``sum = sum + (W)x * w[k]``, which the compiler contracts to
that; written out here because the compiler did not always contract it in
these larger kernels), and its result is rounded to the output dtype
(``cast<D>``), as in the per-axis passes, so the results are identical to
theirs. Boundaries follow the
per-axis semantics: each pass pads its own input along its axis (with its own
cval in constant mode, or by index mapping); positions outside the image along
other axes are computed through the same index mapping, which commutes with
the pass, or, in constant mode, are replaced by the cval of the pass that
reads them.

The kernel for the gradient magnitude of the 3D edge filters
(``skimage.filters.sobel`` and others) runs the three chains of the per-axis
implementation, (z, y, x), (y, z, x) and (x, z, y), from one shared input
plane, and writes ``sqrt((d0*d0 + d1*d1) + d2*d2)`` with the squares and sums
rounded separately, as the elementwise kernels do.
"""

import math

import cupy as cp

from cucim.skimage._vendored._ndimage_filters_core import (
    _ndimage_CAST_FUNCTION,
    _ndimage_includes,
)

from ._fused_separable_2d import _MODE_CODES, _index_functions, _preamble
from ._separable_filtering import _dtype_char_to_c_types

TILE_X = 64
TILE_Y = 16
BLOCK = (64, 4)
NTHREADS = BLOCK[0] * BLOCK[1]
MAX_SHARED_MEMORY = 48 * 1024
# registers per thread available for the queues (values of 4 bytes)
MAX_QUEUE_WORDS = 128

# When the single-chain kernel is used (RTX A6000, against the alternatives;
# see prefer_separable_3d). With 5 or 9 taps, or float64 accumulation, it
# was slower than the alternatives (e.g. 0.62-0.78x for a float32 Gaussian
# with sigma 1, 0.73-0.82x for float64 with sigma 0.5), and so it was for
# small volumes.
_PLAIN_MAX_TAPS = 3
# ... against one 1D pass plus a fused pair (two launches)
_PLAIN_MIN_SIZE_VS_PAIR = 2**24
# ... against three 1D passes
_PLAIN_MIN_SIZE_VS_PER_AXIS = 2**22


class _Pass:
    """One 1D pass: axis ("z", "y" or "x"), taps, anchor (taps read input
    position i - anchor + k), flip (convolution), mode code, accumulation
    type, and the names of its weight and cval kernel parameters."""

    def __init__(self, axis, k, anchor, flip, mode, wtype, wname, cname):
        self.axis = axis
        self.k = k
        self.a = anchor
        self.flip = flip
        self.mode = mode
        self.wtype = wtype
        self.wname = wname
        self.cname = cname

    def halo(self):
        """(low, high) extent of the input beyond the output range."""
        return self.a, self.k - 1 - self.a


class _Chain:
    def __init__(self, name, passes):
        assert [p.axis for p in passes].count("z") == 1
        self.name = name
        self.passes = passes
        self.q = [p.axis for p in passes].index("z")  # 0 or 1
        assert self.q in (0, 1)
        self.pz = passes[self.q]
        self.pre = passes[: self.q]
        self.post = passes[self.q + 1 :]

    def zregion(self):
        """Region (r0, nr, c0, nc) of the z pass output, relative to the
        tile origin: the tile plus the halo of the passes after it."""
        r0, nr, c0, nc = 0, TILE_Y, 0, TILE_X
        for p in self.post:
            lo, hi = p.halo()
            if p.axis == "y":
                r0, nr = r0 - lo, nr + lo + hi
            else:
                c0, nc = c0 - lo, nc + lo + hi
        return r0, nr, c0, nc

    def inregion(self):
        """Region of input values needed in the plane."""
        r0, nr, c0, nc = self.zregion()
        for p in self.pre:
            lo, hi = p.halo()
            if p.axis == "y":
                r0, nr = r0 - lo, nr + lo + hi
            else:
                c0, nc = c0 - lo, nc + lo + hi
        return r0, nr, c0, nc

    def queue_positions(self):
        _, nr, _, nc = self.zregion()
        return math.ceil(nr * nc / NTHREADS)


def _weights_code(p, tag):
    idx = f"{p.k} - 1 - k" if p.flip else "k"
    return f"""
    {p.wtype} w_{tag}[{p.k}];
#pragma unroll
    for (int k = 0; k < {p.k}; k++) w_{tag}[k] = {p.wname}[{idx}];
"""


def _outside(axis, var):
    n = {"z": "Z", "y": "H", "x": "W"}[axis]
    return f"({var} < 0 || {var} >= {n})"


def _pad_rule(passes_after, gy, gx, value, D):
    """Value written for a pass result at plane position (gy, gx), given the
    passes that read it (in order): in constant mode, a position outside the
    image along the axis of the next pass gets that pass's cval; positions
    outside only along the axis of a later pass are replaced later, so any
    value will do there."""
    if not passes_after:
        return value
    nxt = passes_after[0]
    if nxt.mode != _MODE_CODES["constant"] or nxt.axis == "z":
        return value
    g = gy if nxt.axis == "y" else gx
    return (
        f"({_outside(nxt.axis, g)} ? static_cast<{D}>({nxt.cname}) : {value})"
    )


def _chain_src(c, T, D, has_sin, sin_region):
    """Per-chain code fragments: declarations, the per-step queue update
    (z pass) and the passes after it."""
    n = c.name
    pz = c.pz
    zr0, znr, zc0, znc = c.zregion()
    NP = c.queue_positions()
    QT = T if c.q == 0 else D
    decl = f"""
    // chain {n}: passes {"".join(p.axis for p in c.passes)}
    __shared__ {D} s_z_{n}[{znr}][{znc}];
    {QT} q_{n}[{NP}][{pz.k}];
"""
    for i, p in enumerate(c.passes):
        decl += _weights_code(p, f"{n}_{i}")
    zi = c.q  # index of the z pass in c.passes

    # value of the queue's new plane at local z-region position (r, c)
    if c.q == 0:
        if has_sin:
            ir0, _, ic0, _ = sin_region
            newval = f"s_in[r + ({-ir0})][cc + ({-ic0})]"
        else:
            newval = None  # loaded from global memory
    else:
        p1 = c.pre[0]
        ir0, _, ic0, _ = sin_region
        if p1.axis == "y":
            tap = f"s_in[r + ({-ir0}) - {p1.a} + k][cc + ({-ic0})]"
        else:
            tap = f"s_in[r + ({-ir0})][cc + ({-ic0}) - {p1.a} + k]"
        newval = f"pre_{n}"  # computed below
        pre_code = f"""
                {p1.wtype} s1 = static_cast<{p1.wtype}>(0);
#pragma unroll
                for (int k = 0; k < {p1.k}; k++)
                    s1 = fma(static_cast<{p1.wtype}>({tap}), w_{n}_0[k], s1);
                const {D} pre_{n} = {_pad_rule(c.passes[1:], "gy", "gx", f"cast<{D}>(s1)", D)};
"""
    # queue update and z pass, for every owned position
    if c.q == 0:
        if newval is None:
            load = f"""
                {T} v;
                {{
                    int ly = gy, lx = gx;
                    bool pad = zpad;
                    if (MODE_Y_{n} == 4) pad = pad || {_outside("y", "ly")};
                    else ly = _fused_map(ly, H, MODE_Y_{n});
                    if (MODE_X_{n} == 4) pad = pad || {_outside("x", "lx")};
                    else lx = _fused_map(lx, W, MODE_X_{n});
                    v = pad ? static_cast<{T}>({c.passes[0].cname}) : plane[(long long)ly * W + lx];
                }}
"""
        else:
            load = f"""
                const {T} v = zpad ? static_cast<{T}>({pz.cname}) : {newval};
"""
    else:
        load = f"""
                {D} v;
                if (zpad) {{
                    v = static_cast<{D}>({pz.cname});
                }} else {{
{pre_code}
                    v = pre_{n};
                }}
"""
    step = f"""
        // chain {n}: new plane into the queue, then the pass along z
#pragma unroll
        for (int i = 0; i < {NP}; i++) {{
            const int pidx = tid + i * {NTHREADS};
            if (pidx < {znr * znc}) {{
                const int r = pidx / {znc} + ({zr0});
                const int cc = pidx % {znc} + ({zc0});
                const int gy = row0 + r;
                const int gx = col0 + cc;
{load}
#pragma unroll
                for (int k = 0; k < {pz.k} - 1; k++) q_{n}[i][k] = q_{n}[i][k + 1];
                q_{n}[i][{pz.k} - 1] = v;
                if (full) {{
                    {pz.wtype} s = static_cast<{pz.wtype}>(0);
#pragma unroll
                    for (int k = 0; k < {pz.k}; k++)
                        s = fma(static_cast<{pz.wtype}>(q_{n}[i][k]), w_{n}_{zi}[k], s);
                    s_z_{n}[r - ({zr0})][cc - ({zc0})] = {_pad_rule(c.post, "gy", "gx", f"cast<{D}>(s)", D)};
                }}
            }}
        }}
"""
    # passes after z
    post1 = post2 = ""
    if len(c.post) == 2:
        pa, pb = c.post
        # first post pass over its region: the tile along pa's axis, the
        # z region along pb's axis
        if pa.axis == "y":
            mr0, mnr, mc0, mnc = 0, TILE_Y, zc0, znc
            tap = f"s_z_{n}[r - ({zr0}) - {pa.a} + k][cc - ({zc0})]"
        else:
            mr0, mnr, mc0, mnc = zr0, znr, 0, TILE_X
            tap = f"s_z_{n}[r - ({zr0})][cc - ({zc0}) - {pa.a} + k]"
        decl += f"""
    __shared__ {D} s_m_{n}[{mnr}][{mnc}];
"""
        post1 = f"""
        // chain {n}: pass along {pa.axis}
        for (int pidx = tid; pidx < {mnr * mnc}; pidx += {NTHREADS}) {{
            const int r = pidx / {mnc} + ({mr0});
            const int cc = pidx % {mnc} + ({mc0});
            const int gy = row0 + r;
            const int gx = col0 + cc;
            {pa.wtype} s = static_cast<{pa.wtype}>(0);
#pragma unroll
            for (int k = 0; k < {pa.k}; k++)
                s = fma(static_cast<{pa.wtype}>({tap}), w_{n}_{zi + 1}[k], s);
            s_m_{n}[r - ({mr0})][cc - ({mc0})] = {_pad_rule([pb], "gy", "gx", f"cast<{D}>(s)", D)};
        }}
"""
        if pb.axis == "y":
            tap = f"s_m_{n}[r - ({mr0}) - {pb.a} + k][cc - ({mc0})]"
        else:
            tap = f"s_m_{n}[r - ({mr0})][cc - ({mc0}) - {pb.a} + k]"
        post2 = f"""
            {{
                {pb.wtype} s = static_cast<{pb.wtype}>(0);
#pragma unroll
                for (int k = 0; k < {pb.k}; k++)
                    s = fma(static_cast<{pb.wtype}>({tap}), w_{n}_{zi + 2}[k], s);
                res_{n}[j] = cast<{D}>(s);
            }}
"""
    else:
        (pb,) = c.post
        if pb.axis == "y":
            tap = f"s_z_{n}[r - ({zr0}) - {pb.a} + k][cc - ({zc0})]"
        else:
            tap = f"s_z_{n}[r - ({zr0})][cc - ({zc0}) - {pb.a} + k]"
        post1 = f"""
        // chain {n}: pass along {pb.axis}, at the outputs
        {{
            int j = 0;
            for (int r = threadIdx.y; r < {TILE_Y}; r += blockDim.y, j++) {{
                const int cc = threadIdx.x;
                {pb.wtype} s = static_cast<{pb.wtype}>(0);
#pragma unroll
                for (int k = 0; k < {pb.k}; k++)
                    s = fma(static_cast<{pb.wtype}>({tap}), w_{n}_{zi + 1}[k], s);
                res_{n}[j] = cast<{D}>(s);
            }}
        }}
"""
    modes = f"""
#define MODE_Y_{n} {next(p.mode for p in c.passes if p.axis == "y")}
#define MODE_X_{n} {next(p.mode for p in c.passes if p.axis == "x")}
"""
    return modes, decl, step, post1, post2


def _kernel_src(name, chains, T, D, epilogue):
    """Source of a z-marching kernel for the given chains; ``epilogue`` is
    "store" (one chain) or "magnitude"."""
    need_sin = any(c.q == 1 for c in chains) or len(chains) > 1
    sin_region = None
    if need_sin:
        regs = [c.inregion() for c in chains]
        r0 = min(r[0] for r in regs)
        c0 = min(r[2] for r in regs)
        r1 = max(r[0] + r[1] for r in regs)
        c1 = max(r[2] + r[3] for r in regs)
        sin_region = (r0, r1 - r0, c0, c1 - c0)
    pz0 = chains[0].pz
    # all chains share the z extent of the queue (same taps and anchor)
    assert all((c.pz.k, c.pz.a) == (pz0.k, pz0.a) for c in chains)
    zmode = pz0.mode
    assert all(c.pz.mode == zmode for c in chains)
    KZ, AZ = pz0.k, pz0.a
    parts = [_chain_src(c, T, D, need_sin, sin_region) for c in chains]
    pnames = sorted(
        {p.wname: p.wtype for c in chains for p in c.passes}.items()
    )
    cnames = sorted({p.cname for c in chains for p in c.passes})
    params = ", ".join(
        [
            "const T* __restrict__ src",
            f"{'D' if epilogue == 'store' else 'T'}* __restrict__ dst",
            "int Z",
            "int H",
            "int W",
            "int zchunk",
        ]
        + [f"const {t}* __restrict__ {nm}" for nm, t in pnames]
        + [f"double {nm}" for nm in cnames]
    )
    src = f"""
typedef {T} T;
typedef {D} D;
{"".join(m for m, *_ in parts)}
extern "C" __global__ void {name}({params})
{{
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int row0 = blockIdx.y * {TILE_Y};
    const int col0 = blockIdx.x * {TILE_X};
    const int z0 = blockIdx.z * zchunk;
    const int z1 = min(z0 + zchunk, Z);
"""
    if need_sin:
        sr0, snr, sc0, snc = sin_region
        src += f"""
    __shared__ T s_in[{snr}][{snc}];
"""
    for _, decl, *_ in parts:
        src += decl
    nres = TILE_Y // BLOCK[1]
    for c in chains:
        src += f"""
    {D} res_{c.name}[{nres}];
"""
    src += f"""
    // planes zi = z0 - AZ .. z1 - AZ + KZ - 2 enter the queues in turn; the
    // output plane of a step is zi + AZ - KZ + 1
    for (int zi = z0 - {AZ}; zi < z1 - {AZ} + {KZ} - 1; zi++) {{
        const bool full = zi >= z0 - {AZ} + {KZ} - 1;
        int zz = zi;
        bool zpad = false;
        if ({zmode} == 4) zpad = zz < 0 || zz >= Z;
        else zz = _fused_map(zz, Z, {zmode});
        const T* __restrict__ plane = src + (long long)zz * H * W;
"""
    if need_sin:
        sr0, snr, sc0, snc = sin_region
        mode_y = (
            chains[0].passes[[p.axis for p in chains[0].passes].index("y")].mode
        )
        mode_x = (
            chains[0].passes[[p.axis for p in chains[0].passes].index("x")].mode
        )
        pad_in = chains[0].passes[0].cname
        src += f"""
        if (!zpad) {{
            for (int pidx = tid; pidx < {snr * snc}; pidx += {NTHREADS}) {{
                int ly = row0 + pidx / {snc} + ({sr0});
                int lx = col0 + pidx % {snc} + ({sc0});
                bool pad = false;
                if ({mode_y} == 4) pad = {_outside("y", "ly")};
                else ly = _fused_map(ly, H, {mode_y});
                if ({mode_x} == 4) pad = pad || {_outside("x", "lx")};
                else lx = _fused_map(lx, W, {mode_x});
                s_in[pidx / {snc}][pidx % {snc}] = pad ? static_cast<T>({pad_in}) : plane[(long long)ly * W + lx];
            }}
        }}
        __syncthreads();
"""
    for _, _, step, _, _ in parts:
        src += step
    src += """
        if (full) {
            __syncthreads();
"""
    for *_, post1, _ in parts:
        src += post1
    src += f"""
            __syncthreads();
            const int z = zi + {AZ} - {KZ} + 1;
            const int cc = threadIdx.x;
            const int gx = col0 + cc;
            int j = 0;
            for (int r = threadIdx.y; r < {TILE_Y}; r += blockDim.y, j++) {{
                const int gy = row0 + r;
"""
    for *_, post2 in parts:
        src += post2
    if epilogue == "store":
        (c,) = chains
        src += f"""
                if (gy < H && gx < W)
                    dst[((long long)z * H + gy) * W + gx] = res_{c.name}[j];
"""
    else:
        mul, add = (
            ("__fmul_rn", "__fadd_rn")
            if T == "float"
            else ("__dmul_rn", "__dadd_rn")
        )
        d0, d1, d2 = (f"res_{c.name}[j]" for c in chains)
        src += f"""
                if (gy < H && gx < W) {{
                    T m = {add}({add}({mul}({d0}, {d0}), {mul}({d1}, {d1})), {mul}({d2}, {d2}));
                    dst[((long long)z * H + gy) * W + gx] = sqrt(m);
                }}
"""
    src += """
            }
        }
        __syncthreads();
    }
}
"""
    return src


@cp.memoize(for_each_device=True)
def _get_kernel(key):
    kind, T, D, chain_specs = key
    chains = [
        _Chain(name, [_Pass(*p) for p in passes])
        for name, passes in chain_specs
    ]
    name = "cucim_fused3d_" + kind
    code = _preamble + _ndimage_includes + _ndimage_CAST_FUNCTION
    code += _index_functions + _kernel_src(name, chains, T, D, kind)
    module = cp.RawModule(code=code)
    return module.get_function(name), chains


def _shared_memory(chains, t_size, d_size):
    total = 0
    for c in chains:
        _, nr, _, nc = c.zregion()
        total += nr * nc * d_size
        if len(c.post) == 2:
            pa = c.post[0]
            if pa.axis == "y":
                total += TILE_Y * nc * d_size
            else:
                total += nr * TILE_X * d_size
    if any(c.q == 1 for c in chains) or len(chains) > 1:
        regs = [c.inregion() for c in chains]
        nr = max(r[0] + r[1] for r in regs) - min(r[0] for r in regs)
        nc = max(r[2] + r[3] for r in regs) - min(r[2] for r in regs)
        total += nr * nc * t_size
    return total


def _queue_words(chains, t_size, d_size):
    words = 0
    for c in chains:
        size = t_size if c.q == 0 else d_size
        words += c.queue_positions() * c.pz.k * max(1, size // 4)
    return words


def _z_chunk(shape, ntiles):
    """Planes per block: enough blocks for the GPU, and long runs along z
    (each block also loads KZ - 1 planes before its first output)."""
    Z = shape[0]
    dev = cp.cuda.Device()
    n_sm = dev.attributes["MultiProcessorCount"]
    nz = max(1, math.ceil(4 * n_sm / ntiles))
    return max(min(Z, 16), math.ceil(Z / nz))


# Use the z-marching kernels where they apply (for A/B comparisons)
_Z_MARCHING = True


def _specs(passes, T, D):
    """Hashable chain description from (axis, weights, anchor, flip, mode,
    cval-name) tuples (axis 0, 1, 2 -> "z", "y", "x"); ``weights`` is an
    array or a (size, dtype) pair."""
    out = []
    for i, (axis, w, anchor, flip, mode, cname) in enumerate(passes):
        size, dtype = (w.size, w.dtype) if hasattr(w, "dtype") else w
        out.append(
            (
                "zyx"[axis],
                int(size),
                int(anchor),
                bool(flip),
                _MODE_CODES[mode],
                _dtype_char_to_c_types[cp.dtype(dtype).char],
                f"w{i}",
                cname,
            )
        )
    return tuple(out)


def prefer_separable_3d(size, passes, alternative_launches):
    """Whether the single-chain kernel is expected to be faster than the
    alternative (2: a 1D pass plus a fused pair; 3: three 1D passes), from
    the volume size and the passes' (axis, weights or (size, dtype), ...)
    tuples."""
    min_size = (
        _PLAIN_MIN_SIZE_VS_PAIR
        if alternative_launches == 2
        else _PLAIN_MIN_SIZE_VS_PER_AXIS
    )
    if size < min_size:
        return False
    for p in passes:
        w = p[1]
        k, dtype = (w.size, w.dtype) if hasattr(w, "dtype") else w
        if k > _PLAIN_MAX_TAPS or cp.dtype(dtype) != cp.float32:
            return False
    return True


def plan_separable_3d(shape, passes, image_dtype, output_dtype):
    """The compiled-in description of a single-chain kernel, or None when
    the case is not supported (z not first or second, too much shared
    memory or too many queue registers)."""
    axes = [p[0] for p in passes]
    if len(passes) != 3 or sorted(axes) != [0, 1, 2] or axes[2] == 0:
        return None
    T = _dtype_char_to_c_types[cp.dtype(image_dtype).char]
    D = _dtype_char_to_c_types[cp.dtype(output_dtype).char]
    specs = _specs([p[:5] + (f"cval{i}",) for i, p in enumerate(passes)], T, D)
    chain = _Chain("c", [_Pass(*s) for s in specs])
    t_size = cp.dtype(image_dtype).itemsize
    d_size = cp.dtype(output_dtype).itemsize
    if _shared_memory([chain], t_size, d_size) > MAX_SHARED_MEMORY:
        return None
    if _queue_words([chain], t_size, d_size) > MAX_QUEUE_WORDS:
        return None
    return ("store", T, D, (("c", specs),))


def separable_3d(image, passes, output, cvals, plan=None):
    """All three passes of a separable 3D filter in one launch.

    ``passes`` lists (axis, weights, anchor, flip, mode) in the order of
    the per-axis passes; the weights are device arrays of the accumulation
    type of their pass; ``cvals`` holds the cval of each pass. ``image`` and
    ``output`` are C-contiguous, of the same shape, and do not overlap.
    """
    if plan is None:
        plan = plan_separable_3d(image.shape, passes, image.dtype, output.dtype)
    kernel, chains = _get_kernel(plan)
    Z, H, W = image.shape
    ntiles = math.ceil(W / TILE_X) * math.ceil(H / TILE_Y)
    zchunk = _z_chunk(image.shape, ntiles)
    wnames = sorted(p.wname for p in chains[0].passes)
    by_name = {f"w{i}": p[1] for i, p in enumerate(passes)}
    args = [
        image,
        output,
        cp.int32(Z),
        cp.int32(H),
        cp.int32(W),
        cp.int32(zchunk),
    ]
    args += [by_name[nm] for nm in wnames]
    args += [float(c) for c in cvals]
    kernel(
        (math.ceil(W / TILE_X), math.ceil(H / TILE_Y), math.ceil(Z / zchunk)),
        BLOCK,
        tuple(args),
    )
    return output


def plan_edge_magnitude_3d(shape, k, dtype, mode):
    """Single-launch plan for the 3D edge magnitude, or None."""
    T = _dtype_char_to_c_types[cp.dtype(dtype).char]
    a = k // 2
    m = _MODE_CODES[mode]

    def p(axis, which, cname):
        w = "we" if which == "e" else "ws"
        return (axis, k, a, True, m, T, w, cname)

    specs = (
        ("d0", (p("z", "e", "c0"), p("y", "s", "c1"), p("x", "s", "c2"))),
        ("d1", (p("y", "e", "c0"), p("z", "s", "c1"), p("x", "s", "c2"))),
        ("d2", (p("x", "e", "c0"), p("z", "s", "c1"), p("y", "s", "c2"))),
    )
    chains = [_Chain(n, [_Pass(*s) for s in ps]) for n, ps in specs]
    size = cp.dtype(dtype).itemsize
    if _shared_memory(chains, size, size) > MAX_SHARED_MEMORY:
        return None
    if _queue_words(chains, size, size) > MAX_QUEUE_WORDS:
        return None
    return ("magnitude", T, T, specs)


def edge_magnitude_3d(image, edge, smooth, output, cvals, plan):
    """``sqrt((d0*d0 + d1*d1) + d2*d2)`` of the 3D edge filter (``edge`` and
    ``smooth`` applied as convolutions, anchor k // 2), in one launch;
    ``cvals`` are the cvals of the three passes of each chain."""
    kernel, _ = _get_kernel(plan)
    Z, H, W = image.shape
    ntiles = math.ceil(W / TILE_X) * math.ceil(H / TILE_Y)
    zchunk = _z_chunk(image.shape, ntiles)
    args = (
        image,
        output,
        cp.int32(Z),
        cp.int32(H),
        cp.int32(W),
        cp.int32(zchunk),
        edge,
        smooth,
    ) + tuple(float(c) for c in cvals)
    kernel(
        (math.ceil(W / TILE_X), math.ceil(H / TILE_Y), math.ceil(Z / zchunk)),
        BLOCK,
        args,
    )
    return output
