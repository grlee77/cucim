import math

import cupy as cp
from cupyx.scipy import ndimage as ndi

from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage._vendored._internal import _normalize_axis_index
from cucim.skimage._vendored import _ndimage_util as util


def get_shmem_limits(device_id=None):
    if device_id is None:
        device_id = cp.cuda.runtime.getDevice()
    device_props = cp.cuda.runtime.getDeviceProperties(device_id)
    shared_mp = device_props.get('sharedMemPerMultiprocessor', None)
    shared_block = device_props.get('sharedMemPerBlock', None)
    shared_block_optin = device_props.get('sharedMemPerBlockOptin', None)
    global_l1_cache_supported = device_props.get('globalL1CacheSupported', None)
    local_l1_cache_supported = device_props.get('localL1CacheSupported', None)
    l2_size = device_props.get('l2CacheSize', None)
    warp_size = device_props.get('warpSize', None)
    regs_per_block = device_props.get('regsPerBlock', None)
    return {
        'device_id': device_id,
        'shared_mp': shared_mp,
        'shared_block': shared_block,
        'shared_block_optin': shared_block_optin,
        'global_l1_cache_supported': global_l1_cache_supported,
        'local_l1_cache_supported': local_l1_cache_supported,
        'l2_size': l2_size,
        'warp_size': warp_size,
        'regs_per_block': regs_per_block,
    }


def get_constants(ndim, axis, kernel_size, anchor, patch_per_block=None):
    if ndim == 2:
        # note, in this file axis 0 = "y"
        #                    axis 1 = "x"
        # TODO: adjust as needed based on kernel size to stay
        #       within shared memory limits. Initial values
        #       below should be suitable for up to 4 channels with
        #       kernel size <= 32

        # for simplicity, keeping same halo size at both start and end
        if anchor is None:
            anchor = kernel_size // 2
        halo_pixels_needed = max(kernel_size - anchor, anchor)
        if patch_per_block is None:
            patch_per_block = 4
        if axis == 1:
            # as in OpenCV's column_filter.hpp
            block_x = 16
            block_y = 16
            halo_size = math.ceil(halo_pixels_needed / block_x)
        elif axis == 0:
            # as in OpenCV's row_filter.hpp
            block_x = 32  # 16 in CUDA example
            block_y = 8  # 4 in CUDA example
            halo_size = math.ceil(halo_pixels_needed / block_y)
        # TODO: check halo_size. may be larger than needed right now
    elif ndim == 3:
        raise NotImplementedError("TODO")
    else:
        raise NotImplementedError("TODO")
    block = (block_x, block_y, 1)
    return block, patch_per_block, halo_size


def get_smem_shape(ndim, axis, kernel_size, anchor=None, patch_per_block=None, dtype=cp.float32):
    block, patch_per_block, halo_size = get_constants(
        ndim, axis, kernel_size, anchor=anchor, patch_per_block=patch_per_block
    )
    bx, by, bz = block
    if ndim != 2:
        raise NotImplementedError("TODO")
    if axis == 0:
        shape = ((patch_per_block + 2 * halo_size) * by, bx)
    elif axis == 1:
        shape = (by, (patch_per_block + 2 * halo_size) * bx)

    nbytes = cp.dtype(dtype).itemsize * math.prod(shape)
    return shape, nbytes


_dtype_char_to_c_types = {
    'e': 'float16',
    'f': 'float',
    'd': 'double',
    'F': 'complex<float>',
    'D': 'complex<double>',
    'b': 'char',
    'h': 'short',
    'i': 'int',
    'l': 'long logn',
    'B': 'unsigned char',
    'H': 'unsigned short',
    'I': 'unsigned int',
    'L': 'unsigned long logn',
}


def _get_c_type(dtype):
    dtype = cp.dtype(dtype)
    return _dtype_char_to_c_types[dtype.char]


# Note: in OpenCV T is always float, float3 or float4    (so can replace saturate_cast with simply static_cast)
#                 D is can be floating or integer dtype and does need saturation
# Note: the version below is only single-channel
# Note: need to insert appropriate boundary condition for row/col

def _get_separable_conv_kernel_src(kernel_size, axis, ndim, anchor, image_c_type, kernel_c_type, output_c_type, mode, patch_per_block=None):

    blocks, patch_per_block, halo_size = get_constants(
        ndim, axis, kernel_size, anchor, patch_per_block
    )
    block_x, block_y, block_z = blocks

    func_name = f"convolve_size{kernel_size}_{ndim}d_axis{axis}"

    code = f"""
    #include "cupy/carray.cuh"  // for float16

    #define MAX_KERNEL_SIZE 32

    #define KSIZE {kernel_size}
    const int BLOCK_DIM_X = {block_x};
    const int BLOCK_DIM_Y = {block_y};
    const int PATCH_PER_BLOCK = {patch_per_block};
    const int HALO_SIZE = {halo_size};
    typedef {image_c_type}  T;
    typedef {output_c_type} D;
    typedef {kernel_c_type} W;

    extern "C"{{
    __global__ void {func_name}(const T *src, D *dst, const W* kernel, const int anchor, int n_rows, int n_cols)
    {{
    """

    if ndim == 2 and axis == 0:
        # as in OpenCV's column_filter.hpp
        code += """
        __shared__ T smem[(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_Y][BLOCK_DIM_X];
        const int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
        if (x >= n_cols){
            return;
        }
        const T* src_col = &src[x];
        const int yStart = blockIdx.y * (BLOCK_DIM_Y * PATCH_PER_BLOCK) + threadIdx.y;

        // memory is contiguous along last (columns) axis
        const int row_stride = n_cols; // stride (in elements) along axis 0
        int row;

        if (blockIdx.y > 0)
        {
            //Upper halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>(src[(yStart - (HALO_SIZE - j) * BLOCK_DIM_Y) * row_stride + x]);
        }
        else
        {
            // TODO, mode support: currently using replicate border condition
            // Upper halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j) {
                row = yStart - (HALO_SIZE - j) * BLOCK_DIM_Y;
        """
        code += _generate_boundary_condition_ops(mode, 'row', 'n_rows')
        code += """
                smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>(src_col[row * row_stride]);
            }
        }

        if (blockIdx.y + 2 < gridDim.y)
        {
            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>(src_col[(yStart + j * BLOCK_DIM_Y) * row_stride]);

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>(src_col[(yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y) * row_stride]);
        }
        else
        {
            // TODO, mode support: currently using replicate border condition

            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j) {
                row = yStart + j * BLOCK_DIM_Y;
        """
        code += _generate_boundary_condition_ops(mode, 'row', 'n_rows')
        code += """
                smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>(src_col[row * row_stride]);
            }

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
            {
                row = yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y;
        """
        code += _generate_boundary_condition_ops(mode, 'row', 'n_rows')
        code += """
                smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>(src_col[row * row_stride]);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < PATCH_PER_BLOCK; ++j)
        {
            const int y = yStart + j * BLOCK_DIM_Y;

            if (y < n_rows)
            {
                W sum = 0;

                #pragma unroll
                for (int k = 0; k < KSIZE; ++k)
                    sum += static_cast<W>(smem[threadIdx.y + (HALO_SIZE + j) * BLOCK_DIM_Y - anchor + k][threadIdx.x]) * kernel[k];

                // TODO: replace with appropriate saturating cast to D for dst
                dst[y * row_stride + x] = static_cast<D>(sum);
            }
        }
        """
    elif ndim == 2 and axis == 1:
        # as in OpenCV's row_filter.hpp
        code += """
        __shared__ T smem[BLOCK_DIM_Y][(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_X];
        const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
        if (y >= n_rows) {
            return;
        }
        const int row_stride = n_cols;  // stride (in elements) along axis 0
        int col;
        const T* src_row = &src[y * row_stride];
        const int xStart = blockIdx.x * (PATCH_PER_BLOCK * BLOCK_DIM_X) + threadIdx.x;

        if (blockIdx.x > 0)
        {
            //Load left halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = static_cast<T>(src_row[xStart - (HALO_SIZE - j) * BLOCK_DIM_X]);
        }
        else
        {
            // TODO, mode support: currently using replicate border condition

            //Load left halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j){
                col = xStart - (HALO_SIZE - j) * BLOCK_DIM_X;
"""
        code += _generate_boundary_condition_ops(mode, 'col', 'n_cols')
        code += """
                smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = static_cast<T>(src_row[col]);
            }
        }

        if (blockIdx.x + 2 < gridDim.x)
        {
            //Load main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] = static_cast<T>(src_row[xStart + j * BLOCK_DIM_X]);

            //Load right halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] = static_cast<T>(src_row[xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X]);
        }
        else
        {
            // TODO, mode support: currently using replicate border condition

            //Load main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j) {
                col = xStart + j * BLOCK_DIM_X;
        """
        code += _generate_boundary_condition_ops(mode, 'col', 'n_cols')
        code += """
                smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] = static_cast<T>(src_row[col]);
            }

            //Load right halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j){
                col = xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X;
        """
        code += _generate_boundary_condition_ops(mode, 'col', 'n_cols')
        code += """
                smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] = static_cast<T>(src_row[col]);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < PATCH_PER_BLOCK; ++j)
        {
            const int x = xStart + j * BLOCK_DIM_X;

            if (x < n_cols)
            {
                W sum = 0;

                #pragma unroll
                for (int k = 0; k < KSIZE; ++k)
                    sum += static_cast<W>(smem[threadIdx.y][threadIdx.x + (HALO_SIZE + j) * BLOCK_DIM_X - anchor + k]) * kernel[k];

                // TODO: replace with appropriate saturating cast to D for dst
                dst[y * row_stride + x] = static_cast<D>(sum);
            }
        }
        """
    code += f"""
    }}  // function
    }}  // extern "C"
    """
    return func_name, blocks, patch_per_block, code


def _get_separable_conv_kernel(kernel_size, axis, ndim, image_c_type,
                               kernel_c_type, output_c_type, anchor=None,
                               mode='nearest', patch_per_block=None, options=()):
    func_name, block, patch_per_block, code = _get_separable_conv_kernel_src(
        kernel_size=kernel_size,
        axis=axis,
        ndim=ndim,
        image_c_type=image_c_type,
        kernel_c_type=kernel_c_type,
        output_c_type=output_c_type,
        anchor=anchor,
        mode=mode,
        patch_per_block=patch_per_block
    )
    m = cp.RawModule(code=code)
    return m.get_function(func_name), block, patch_per_block


def _get_grid(shape, block, axis, patch_per_block):
    """Determine grid size from image shape and block parameters"""
    if len(shape) != 2:
        raise ValueError("grid calculation currently only implemented for 2D")
    if axis == 0:
        # column filter
        grid = (
            math.ceil(shape[1] / block[0]),
            math.ceil(shape[0] / (block[1] * patch_per_block)),
            1,
        )
    elif axis == 1:
        # row filter
        grid = (
            math.ceil(shape[1] / (block[0] * patch_per_block)),
            math.ceil(shape[0] / block[1]),
            1,
        )
    return grid


def _shmem_convolve1d(image, weights, axis=-1, output=None, mode="reflect",
                      cval=0.0, origin=0, convolution=False):

    ndim = image.ndim
    if weights.ndim != 1:
        raise ValueError("expected 1d weight array")
    axis = _normalize_axis_index(axis, ndim)
    origin = util._check_origin(origin, weights.size)
    if weights.size == 0:
        return cupy.zeros_like(input)
    # TODO: set weights upper size limit based on available shared memory
    if weights.nbytes >= (1 << 31):
        raise RuntimeError('weights must be 2 GiB or less, use FFTs instead')
    util._check_mode(mode)

    if convolution:
        weights = weights[::-1]
        origin = -origin
        if weights.size % 2 == 0:
            origin -= 1
    elif weights.dtype.kind == "c":
        # numpy.correlate conjugates weights rather than input.
        weights = weights.conj()

    anchor = weights.size // 2 + origin
    float_dtype = _supported_float_type(image.dtype, allow_complex=False)

    # CUDA kernels assume C-contiguous memory layout
    image = image.astype(float_dtype, copy=False)
    image = cp.ascontiguousarray(image)
    weights_dtype = util._get_weights_dtype(image, weights)  # TODO: currently casts all int types to float64
    weights = weights.astype(weights_dtype, copy=False)
    weights = cp.ascontiguousarray(weights)

    output = util._get_output(output, image)

    index_c_type = util._get_inttype(image)
    image_c_type = _get_c_type(float_dtype)
    weights_c_type = _get_c_type(weights_dtype)
    output_c_type = _get_c_type(output.dtype)

    #     block, patch_per_block, halo_size = get_constants(src.ndim, axis, kernel.size, anchor, patch_per_block=patch_per_block)
    # print(f"{halo_size=}")
    conv_axis_kernel, block, patch_per_block = _get_separable_conv_kernel(
        weights.size,
        axis=axis,
        ndim=ndim,
        anchor=anchor,
        image_c_type=image_c_type,
        kernel_c_type=weights_c_type,
        output_c_type=output_c_type,
        mode=mode,
        patch_per_block=None,
    )
    grid = _get_grid(
        image.shape, block=block, axis=axis, patch_per_block=patch_per_block
    )

    conv_axis_kernel(
        grid,
        block,
        (image, output, weights, anchor, image.shape[0], image.shape[1])
    )
    return output


def convolve1d(image, weights, axis=-1, output=None, mode="reflect", cval=0.0,
               origin=0, algorithm=None):
    # TODO: update conditions here once more edges and/or sizes are supported
    if algorithm is None:
       if image.ndim == 2 and weights.size <= 256:
           algorithm = 'shared_memory'
       else:
           algorithm = 'elementwise'
    algorithm = algorithm.lower()
    if algorithm == 'shared_memory':
        if image.ndim != 2:
            raise NotImplementedError(
                f"shared_memory not implemented for ndim={image.ndim}"
            )
        out = _shmem_convolve1d(image, weights, axis=axis, output=output,
                                mode=mode, cval=cval, origin=origin)
    elif algorithm == 'elementwise':
        out = ndi.convolve1d(image, weights, axis=axis, output=output,
                             mode=mode, cval=cval, origin=origin)
    else:
        raise ValueError(
            "algorithm must be 'shared_memory', 'elementwise' or None"
        )
    return out