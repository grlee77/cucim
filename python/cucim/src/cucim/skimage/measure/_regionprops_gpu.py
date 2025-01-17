import math
import warnings

import cupy as cp
import numpy as np
from packaging.version import parse

CUPY_GTE_13_3_0 = parse(cp.__version__) >= parse("13.3.0")

if CUPY_GTE_13_3_0:
    _includes = r"""
#include <cupy/cuda_workaround.h>  // provide std:: coverage
"""
else:
    _includes = r"""
#include <type_traits>  // let Jitify handle this
"""

__all__ = [
    "area_bbox_from_slices",
    "regionprops_area",
    "regionprops_area_bbox",
    "regionprops_bbox_coords",
    "regionprops_centroid",
    "regionprops_centroid_local",
    "regionprops_inertia_tensor",
    "regionprops_inertia_tensor_eigvals",
    "regionprops_intensity_max",
    "regionprops_intensity_mean",
    "regionprops_intensity_min",
    "regionprops_intensity_std",
    "regionprops_moments",
    "regionprops_moments_central",
    "regionprops_moments_hu",
    "regionprops_moments_normalized",
    "regionprops_num_pixels",
]


@cp.memoize(for_each_device=True)
def get_num_pixels_kernel(count_dtype):
    count_dtype = cp.dtype(count_dtype)

    # store only counts for label > 0  (label = 0 is the background)
    source = """
      if (label != 0) {
        atomicAdd(&counts[label - 1], 1);
      }\n"""

    name = f"cucim_num_pixels_{count_dtype.name}"
    return cp.ElementwiseKernel(
        "X label",  # inputs
        f"raw {count_dtype.name} counts",  # outputs
        source,
        name=name,
    )


@cp.memoize(for_each_device=True)
def get_sum_kernel(count_dtype, sum_dtype, num_channels):
    count_dtype = cp.dtype(count_dtype)
    sum_dtype = cp.dtype(sum_dtype)

    if sum_dtype.kind == "f":
        c_type = "double" if sum_dtype == cp.float64 else "float"
    elif sum_dtype.kind in "bu":
        itemsize = sum_dtype.itemsize
        c_type = "uint32_t" if itemsize <= 4 else "uint64_t"
    elif sum_dtype.kind in "i":
        itemsize = sum_dtype.itemsize
        c_type = "int32_t" if itemsize <= 4 else "int64_t"
    else:
        raise ValueError(
            "invalid sum_dtype. Must be unsigned, integer or floating point"
        )

    if num_channels == 1:
        channels_line = "uint32_t num_channels = 1;"
    else:
        channels_line = "uint32_t num_channels = sums.shape()[1];"

    # store only counts for label > 0  (label = 0 is the background)
    source = f"""
      if (label != 0) {{
        atomicAdd(&counts[label - 1], 1);
        {channels_line}
        uint32_t sum_offset = (label - 1) * num_channels;
        uint32_t img_offset = i * num_channels;
        atomicAdd(&sums[sum_offset],
                  static_cast<{c_type}>(img[img_offset]));\n"""
    if num_channels > 1:
        source += f"""
        for (int c = 1; c < num_channels; c++) {{
            atomicAdd(&sums[sum_offset + c],
                      static_cast<{c_type}>(img[img_offset + c]));
        }}\n"""
    source += """
      }\n"""
    inputs = "X label, raw Y img"
    outputs = f"raw {count_dtype.name} counts, raw {sum_dtype.name} sums"
    name = f"cucim_sum_and_count_{sum_dtype.name}_{count_dtype.name}"
    if num_channels > 1:
        name += "_multichannel"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


@cp.memoize(for_each_device=True)
def get_mean_kernel(count_dtype, sum_dtype, num_channels=1):
    count_dtype = cp.dtype(count_dtype)
    sum_dtype = cp.dtype(sum_dtype)

    if sum_dtype.kind == "f":
        c_type = "double" if sum_dtype == cp.float64 else "float"
    elif sum_dtype.kind in "bu":
        itemsize = sum_dtype.itemsize
        c_type = "uint32_t" if itemsize <= 4 else "uint64_t"
    elif sum_dtype.kind in "i":
        itemsize = sum_dtype.itemsize
        c_type = "int32_t" if itemsize <= 4 else "int64_t"
    else:
        raise ValueError(
            "invalid sum_dtype. Must be unsigned, integer or floating point"
        )

    if num_channels == 1:
        channels_line = "uint32_t num_channels = 1;"
    else:
        channels_line = "uint32_t num_channels = sums.shape()[1];"

    # store only counts for label > 0  (label = 0 is the background)
    source = f"""
      if (label != 0) {{
        atomicAdd(&counts[label - 1], 1);
        {channels_line}
        uint32_t sum_offset = (label - 1) * num_channels;
        uint32_t img_offset = i * num_channels;
        atomicAdd(&sums[sum_offset],
                  static_cast<{c_type}>(img[img_offset]));\n"""
    if num_channels > 1:
        source += f"""
        for (int c = 1; c < num_channels; c++) {{
            atomicAdd(&sums[sum_offset + c],
                      static_cast<{c_type}>(img[img_offset + c]));
        }}\n"""
    source += """
      }\n"""
    inputs = "X label, raw Y img"
    outputs = f"raw {count_dtype.name} counts, raw {sum_dtype.name} sums"
    name = f"cucim_sum_and_count_{sum_dtype.name}_{count_dtype.name}"
    if num_channels > 1:
        name += "_multichannel"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


@cp.memoize(for_each_device=True)
def get_sum_and_sumsq_kernel(count_dtype, sum_dtype, num_channels=1):
    count_dtype = cp.dtype(count_dtype)
    sum_dtype = cp.dtype(sum_dtype)

    if sum_dtype.kind == "f":
        c_type = "double" if sum_dtype == cp.float64 else "float"
    elif sum_dtype.kind in "bu":
        itemsize = sum_dtype.itemsize
        c_type = "uint32_t" if itemsize <= 4 else "uint64_t"
    elif sum_dtype.kind in "i":
        itemsize = sum_dtype.itemsize
        c_type = "int32_t" if itemsize <= 4 else "int64_t"
    else:
        raise ValueError(
            "invalid sum_dtype. Must be unsigned, integer or floating point"
        )

    if num_channels == 1:
        channels_line = "uint32_t num_channels = 1;"
    else:
        channels_line = "uint32_t num_channels = sums.shape()[1];"

    # store only counts for label > 0  (label = 0 is the background)
    source = f"""
      if (label != 0) {{
        atomicAdd(&counts[label - 1], 1);
        {channels_line}
        uint32_t sum_offset = (label - 1) * num_channels;
        uint32_t img_offset = i * num_channels;
        auto val = static_cast<{c_type}>(img[img_offset]);
        atomicAdd(&sums[sum_offset], val);
        atomicAdd(&sumsqs[sum_offset], val*val);\n"""
    if num_channels > 1:
        source += f"""
        for (int c = 1; c < num_channels; c++) {{
            val = static_cast<{c_type}>(img[img_offset + c]);
            atomicAdd(&sums[sum_offset + c], val);
            atomicAdd(&sumsqs[sum_offset + c], val*val);
        }}\n"""
    source += """
      }\n"""
    name = f"cucim_sum_and_count_{sum_dtype.name}_{count_dtype.name}"
    if num_channels > 1:
        name += "_multichannel"
    inputs = "X label, raw Y img"
    outputs = f"raw {count_dtype.name} counts, raw {sum_dtype.name} sums, "
    outputs += f"raw {sum_dtype.name} sumsqs"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def _check_count_dtype(dtype):
    """atomicAdd only supports int32, uint32, int64, uint64, float32, float64"""
    dtype = np.dtype(dtype)
    if dtype.kind not in "biuf":
        raise ValueError(
            "dtype must be a signed, unsigned or floating point dtype"
        )
    kernel_dtype = dtype
    if dtype.itemsize > 8:
        raise ValueError("dtype cannot be larger than 64-bit")
    elif dtype.itemsize < 4:
        if dtype.kind == "u":
            kernel_dtype = cp.uint32
        elif dtype.kind == "i":
            kernel_dtype = cp.int32
        elif dtype.kind == "f":
            kernel_dtype = cp.float32
        warnings.warn(
            f"For dtype={dtype.name}, the kernel will use {kernel_dtype} and "
            "then copy the output to the requested type.",
            stacklevel=2,
        )
    return dtype, kernel_dtype


def regionprops_num_pixels(label_image, max_label=None, count_dtype=np.uint32):
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label
    counts = cp.zeros(num_counts, dtype=count_dtype)

    count_dtype, kernel_dtype = _check_count_dtype(count_dtype)

    kernel = get_num_pixels_kernel(kernel_dtype)

    # make a copy if the labels array is not C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    kernel(label_image, counts)

    # allow converting output to requested dtype if it was smaller than 32-bit
    if kernel_dtype != count_dtype:
        counts = counts.astype(count_dtype)
    return counts


def regionprops_area(
    label_image, spacing=None, max_label=None, dtype=cp.float32
):
    # integer atomicAdd is faster than floating point so better to convert
    # after counting
    area = regionprops_num_pixels(
        label_image,
        max_label=max_label,
        count_dtype=np.uint32,
    )
    area = area.astype(dtype)
    if spacing is not None:
        if isinstance(spacing, cp.ndarray):
            pixel_area = cp.product(spacing)
        else:
            pixel_area = math.prod(spacing)
        area *= pixel_area
    return area


def _check_shapes(label_image, intensity_image):
    ndim = label_image.ndim
    if intensity_image.shape[:ndim] != label_image.shape:
        raise ValueError(
            "Initial dimensions of `intensity_image` must match the shape of "
            "`label_image`. (`intensity_image` may have additional trailing "
            "channels/batch dimensions)"
        )

    num_channels = (
        math.prod(intensity_image.shape[ndim:])
        if intensity_image.ndim > ndim
        else 1
    )
    return num_channels


def regionprops_intensity_mean(
    label_image,
    intensity_image,
    max_label=None,
    sum_dtype=None,
    mean_dtype=cp.float32,
):
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_shapes(label_image, intensity_image)

    count_dtype = cp.uint32
    if sum_dtype is None:
        if intensity_image.dtype.kind in "bui":
            sum_dtype = cp.uint64
        else:
            sum_dtype = cp.float64

    counts = cp.zeros(num_counts, dtype=count_dtype)
    sum_shape = (
        (num_counts,) if num_channels == 1 else (num_counts, num_channels)
    )
    sums = cp.zeros(sum_shape, dtype=sum_dtype)

    kernel = get_sum_kernel(count_dtype, sum_dtype, num_channels)

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    if not intensity_image.flags.c_contiguous:
        intensity_image = cp.ascontiguousarray(intensity_image)

    kernel(label_image, intensity_image, counts, sums)

    if num_channels > 1:
        means = sums / counts[:, cp.newaxis]
    else:
        means = sums / counts
    means = means.astype(mean_dtype, copy=False)
    return counts, means


@cp.memoize(for_each_device=True)
def get_mean_var_kernel(dtype, sample_std=False):
    dtype = cp.dtype(dtype)

    if dtype.kind != "f":
        raise ValueError("dtype must be a floating point type")
    if dtype == cp.float64:
        c_type = "double"
        nan_val = "CUDART_NAN"
    else:
        c_type = "float"
        nan_val = "CUDART_NAN_F"

    if sample_std:
        source = f"""
            if (count == 1) {{
              m = static_cast<{c_type}>(sum);
              var = {nan_val};
            }} else {{
              m = static_cast<double>(sum) / count;
              var = sqrt(
                  (static_cast<double>(sumsq) - m * m * count) / (count - 1));
            }}\n"""
    else:
        source = f"""
            if (count == 0) {{
              m = static_cast<{c_type}>(sum);
              var = {nan_val};
            }} else if (count == 1) {{
              m = static_cast<{c_type}>(sum);
              var = 0.0;
            }} else {{
              m = static_cast<double>(sum) / count;
              var = sqrt(
                  (static_cast<double>(sumsq) - m * m * count) / count);
            }}\n"""
    inputs = "X count, Y sum, Y sumsq"
    outputs = "Z m, Z var"
    name = f"cucim_sample_std_naive_{dtype.name}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_intensity_std(
    label_image,
    intensity_image,
    sample_std=False,
    max_label=None,
    sum_dtype=None,
    std_dtype=cp.float64,
):
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_shapes(label_image, intensity_image)

    count_dtype = cp.uint32
    if sum_dtype is None:
        if intensity_image.dtype.kind in "bui":
            sum_dtype = cp.uint64
        else:
            # float64 to help compensate for poor numeric stability of the
            # naive algorithm
            sum_dtype = cp.float64

    counts = cp.zeros(num_counts, dtype=count_dtype)
    sum_shape = (
        (num_counts,) if num_channels == 1 else (num_counts, num_channels)
    )
    sums = cp.zeros(sum_shape, dtype=sum_dtype)
    sumsqs = cp.zeros(sum_shape, dtype=sum_dtype)

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    if not intensity_image.flags.c_contiguous:
        intensity_image = cp.ascontiguousarray(intensity_image)

    approach = "naive"
    if approach == "naive":
        kernel = get_sum_and_sumsq_kernel(
            count_dtype, sum_dtype, num_channels=num_channels
        )
        kernel(label_image, intensity_image, counts, sums, sumsqs)

        if cp.dtype(std_dtype).kind != "f":
            raise ValueError("mean_dtype must be a floating point type")

        # compute means and standard deviations from the counts, sums and
        # squared sums (use float64 here since the numerical stability of this
        # approach is poor)
        means = cp.zeros(sum_shape, dtype=cp.float64)
        stds = cp.zeros(sum_shape, dtype=cp.float64)
        kernel2 = get_mean_var_kernel(std_dtype, sample_std=sample_std)
        if num_channels > 1:
            kernel2(counts[..., cp.newaxis], sums, sumsqs, means, stds)
        else:
            kernel2(counts, sums, sumsqs, means, stds)
    else:
        # TODO(grelee): May want to provide an approach with better stability
        # like the two-pass algorithm or Welford's online algorithm
        raise NotImplementedError("TODO")
    means = means.astype(std_dtype, copy=False)
    stds = stds.astype(std_dtype, copy=False)
    return counts, means, stds


@cp.memoize(for_each_device=True)
def get_min_or_max_kernel(min_dtype, do_min=True, num_channels=1):
    min_dtype = cp.dtype(min_dtype)

    if min_dtype.kind == "f":
        c_type = "double" if min_dtype == cp.float64 else "float"
    elif min_dtype.kind in "bu":
        itemsize = min_dtype.itemsize
        c_type = "uint32_t" if itemsize <= 4 else "uint64_t"
    elif min_dtype.kind in "i":
        itemsize = min_dtype.itemsize
        c_type = "int32_t" if itemsize <= 4 else "int64_t"
    else:
        raise ValueError(
            "invalid min_dtype. Must be unsigned, integer or floating point"
        )

    # Note: CuPy provides atomicMin and atomicMax for float and double in
    #       cupy/_core/include/atomics.cuh
    #       The integer variants are part of CUDA itself.
    if do_min:
        func_name = "atomicMin"
    else:
        func_name = "atomicMax"

    if num_channels == 1:
        channels_line = "uint32_t num_channels = 1;"
    else:
        channels_line = "uint32_t num_channels = out.shape()[1];"

    # store only counts for label > 0  (label = 0 is the background)
    source = f"""
      if (label != 0) {{
        {channels_line}
        uint32_t out_offset = (label - 1) * num_channels;
        uint32_t img_offset = i * num_channels;
        {func_name}(&out[out_offset],
                    static_cast<{c_type}>(img[img_offset]));\n"""
    if num_channels > 1:
        source += f"""
        for (int c = 1; c < num_channels; c++) {{
            {func_name}(&out[out_offset + c],
                        static_cast<{c_type}>(img[img_offset + c]));
        }}\n"""
    source += """
      }\n"""
    op_name = "min" if do_min else "max"

    inputs = "X label, raw Y img"
    outputs = f"raw {min_dtype.name} out"
    name = f"cucim_{op_name}_{min_dtype.name}"
    if num_channels > 1:
        name += "_multichannel"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def _regionprops_min_or_max_intensity(
    label_image, intensity_image, do_min=True, max_label=None
):
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_shapes(label_image, intensity_image)

    # use a data type supported by atomicMin and atomicMax
    if intensity_image.dtype.kind in "bu":
        if intensity_image.dtype.itemsize > 4:
            op_dtype = cp.uint64
        else:
            op_dtype = cp.uint32
        dtype_info = cp.iinfo(op_dtype)
    elif intensity_image.dtype.kind == "i":
        if intensity_image.dtype.itemsize > 4:
            op_dtype = cp.int64
        else:
            op_dtype = cp.int32
        dtype_info = cp.iinfo(op_dtype)
    elif intensity_image.dtype.kind == "f":
        if intensity_image.dtype.itemsize > 4:
            op_dtype = cp.float64
        else:
            op_dtype = cp.float32
        dtype_info = cp.finfo(op_dtype)

    initial_val = dtype_info.max if do_min else dtype_info.min

    out_shape = (
        (num_counts,) if num_channels == 1 else (num_counts, num_channels)
    )
    out = cp.full(out_shape, initial_val, dtype=op_dtype)

    kernel = get_min_or_max_kernel(
        op_dtype, do_min=do_min, num_channels=num_channels
    )

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    if not intensity_image.flags.c_contiguous:
        intensity_image = cp.ascontiguousarray(intensity_image)

    kernel(label_image, intensity_image, out)
    return out


def regionprops_intensity_min(label_image, intensity_image, max_label=None):
    return _regionprops_min_or_max_intensity(
        label_image, intensity_image, True, max_label
    )


def regionprops_intensity_max(label_image, intensity_image, max_label=None):
    return _regionprops_min_or_max_intensity(
        label_image, intensity_image, False, max_label
    )


def _unravel_loop_index(var_name, ndim, uint_t="unsigned int"):
    """
    declare a multi-index array in_coord and unravel the 1D index, i into it.
    This code assumes that the array is a C-ordered array.
    """
    code = f"""
        {uint_t} in_coord[{ndim}];
        {uint_t} s, t, idx = i;"""
    for j in range(ndim - 1, 0, -1):
        code += f"""
        s = {var_name}.shape()[{j}];
        t = idx / s;
        in_coord[{j}] = idx - t * s;
        idx = t;"""
    code += """
        in_coord[0] = idx;"""
    return code


@cp.memoize(for_each_device=True)
def get_bbox_coords_kernel(coord_dtype, ndim, compute_coordinate_sums=False):
    coord_dtype = cp.dtype(coord_dtype)
    if compute_coordinate_sums:
        coord_sum_dtype = cp.dtype(cp.uint64)
        count_dtype = cp.dtype(cp.uint32)

    uint_t = (
        "unsigned int" if coord_dtype.itemsize <= 4 else "unsigned long long"
    )

    source = """
          if (label[i] != 0) {"""
    source += _unravel_loop_index("label", ndim, uint_t=uint_t)
    for d in range(ndim):
        source += f"""
            atomicMin(&bbox[(label[i] - 1) * {2 * ndim} + {2*d}],
                      in_coord[{d}]);
            atomicMax(&bbox[(label[i] - 1) * {2 * ndim} + {2*d + 1}],
                      in_coord[{d}] + 1);"""
        if compute_coordinate_sums:
            source += f"""
            atomicAdd(&coord_sums[(label[i] - 1) * {ndim} + {d}],
                      in_coord[{d}]);"""
    if compute_coordinate_sums:
        source += """
            atomicAdd(&counts[label[i] - 1], 1);"""
    source += """
          }\n"""

    inputs = "raw X label"
    if compute_coordinate_sums:
        outputs = f"raw {coord_dtype.name} bbox, "
        outputs += f"raw {count_dtype.name} counts, "
        outputs += f"raw {coord_sum_dtype.name} coord_sums"
    else:
        outputs = f"raw {coord_dtype.name} bbox"
    if compute_coordinate_sums:
        name = f"cucim_centroid_{ndim}d_{coord_dtype.name}"
        name += f"_{count_dtype.name}_{coord_sum_dtype.name}"
    else:
        name = f"cucim_bbox_{ndim}d_{coord_dtype.name}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_bbox_coords(
    label_image, max_label=None, coord_dtype=cp.uint32, return_slices=False
):
    """
    Parameters
    ----------
    label_image : cp.ndarray
        Image containing labels where 0 is the background and sequential
        values > 0 are the labels.
    max_label : int or None
        The maximum label value present in label_image. Will be computed if not
        provided.
    coord_dtype : dtype, optional
        The data type to use for coordinate calculations. Should be
        ``cp.uint32`` or ``cp.uint64``.
    return_slices : bool, optional
        If True, convert the bounding box coordinates array to a list of slice
        tuples.

    Returns
    -------
    bbox_coords : cp.ndarray
        Raw bounding box coordinates array. The first axis is indexed by
        ``label - 1``. The second axis has the minimum coordinate for dimension
        ``d`` at index ``2*d`` and the maximum for coordinate at dimension
        ``d`` at index ``2*d + 1``. Unlike for `bbox_slices`, the maximum
        coordinate in `bbox_coords` is **inclusive** (the region's bounding box
        includes both the min and max coordinate).
    bbox_slices : list[tuple[slice]] or None
        Will be None if return_slices is False. To get a mask corresponding to
        the ith label, use
        ``mask = label_image[bbox_slices[label - 1]] == label`` to get the
        region corresponding to the ith bounding box.
    """
    if max_label is None:
        max_label = int(label_image.max())

    bbox_coords_kernel = get_bbox_coords_kernel(coord_dtype, label_image.ndim)

    ndim = label_image.ndim
    bbox_coords = cp.zeros((max_label, 2 * ndim), dtype=coord_dtype)

    # Initialize value for atomicMin on even coordinates
    # The value for atomicMax columns is already 0 as desired.
    bbox_coords[:, ::2] = cp.iinfo(coord_dtype).max

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    bbox_coords_kernel(label_image, bbox_coords, size=label_image.size)
    if return_slices:
        bbox_coords_cpu = cp.asnumpy(bbox_coords)
        if ndim == 2:
            # explicitly writing out the 2d case here for clarity
            bbox_slices = [
                (
                    slice(int(box[0]), int(box[1])),
                    slice(int(box[2]), int(box[3])),
                )
                for box in bbox_coords_cpu
            ]
        else:
            # general n-dimensional case
            bbox_slices = [
                tuple(
                    slice(int(box[2 * d]), int(box[2 * d + 1]))
                    for d in range(ndim)
                )
                for box in bbox_coords_cpu
            ]
    else:
        bbox_slices = None

    return bbox_coords, bbox_slices


def regionprops_centroid(label_image, max_label=None, coord_dtype=cp.uint32):
    """
    Parameters
    ----------
    label_image : cp.ndarray
        Image containing labels where 0 is the background and sequential
        values > 0 are the labels.
    max_label : int or None
        The maximum label value present in label_image. Will be computed if not
        provided.
    coord_dtype : dtype, optional
        The data type to use for coordinate calculations. Should be
        ``cp.uint32`` or ``cp.uint64``.

    Returns
    -------
    counts : cp.ndarray
        The number of samples in each region.
    centroid : np.ndarray
        The centroid of each region.
    """
    if max_label is None:
        max_label = int(label_image.max())

    bbox_coords_kernel = get_bbox_coords_kernel(
        coord_dtype, label_image.ndim, compute_coordinate_sums=True
    )

    ndim = label_image.ndim
    bbox_coords = cp.zeros((max_label, 2 * ndim), dtype=coord_dtype)
    centroid_counts = cp.zeros((max_label,), dtype=cp.uint32)
    centroid_sums = cp.zeros((max_label, ndim), dtype=cp.uint64)

    # Initialize value for atomicMin on even coordinates
    # The value for atomicMax columns is already 0 as desired.
    bbox_coords[:, ::2] = cp.iinfo(coord_dtype).max

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    bbox_coords_kernel(
        label_image,
        bbox_coords,
        centroid_counts,
        centroid_sums,
        size=label_image.size,
    )

    centroid = centroid_sums / centroid_counts[:, cp.newaxis]
    return centroid_counts, centroid


@cp.memoize(for_each_device=True)
def get_centroid_local_kernel(coord_dtype, ndim):
    coord_dtype = cp.dtype(coord_dtype)
    sum_dtype = cp.dtype(cp.uint64)
    count_dtype = cp.dtype(cp.uint32)
    uint_t = (
        "unsigned int" if coord_dtype.itemsize <= 4 else "unsigned long long"
    )

    source = """
          auto L = label[i];
          if (L != 0) {"""
    source += _unravel_loop_index("label", ndim, uint_t=uint_t)
    for d in range(ndim):
        source += f"""
            atomicAdd(&centroid_sums[(L - 1) * {ndim} + {d}],
                      in_coord[{d}] - bbox[(L - 1) * {2 * ndim} + {2*d}]);
        """
    source += """
        atomicAdd(&centroid_counts[L - 1], 1);
          }\n"""
    inputs = f"raw X label, raw {coord_dtype.name} bbox"
    outputs = f"raw {count_dtype.name} centroid_counts, "
    outputs += f"raw {sum_dtype.name} centroid_sums"
    name = f"cucim_centroid_local_{ndim}d_{coord_dtype.name}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_centroid_local(
    label_image, max_label=None, coord_dtype=cp.uint32
):
    """
    Parameters
    ----------
    label_image : cp.ndarray
        Image containing labels where 0 is the background and sequential
        values > 0 are the labels.
    max_label : int or None
        The maximum label value present in label_image. Will be computed if not
        provided.
    coord_dtype : dtype, optional
        The data type to use for coordinate calculations. Should be
        ``cp.uint32`` or ``cp.uint64``.

    Returns
    -------
    counts : cp.ndarray
        The number of samples in each region.
    centroid_local : cp.ndarray
        The local centroids
    """
    if max_label is None:
        max_label = int(label_image.max())

    bbox_coords_kernel = get_bbox_coords_kernel(
        coord_dtype,
        label_image.ndim,
        compute_coordinate_sums=False,
    )

    ndim = label_image.ndim
    bbox_coords = cp.zeros((max_label, 2 * ndim), dtype=coord_dtype)

    # Initialize value for atomicMin on even coordinates
    # The value for atomicMax columns is already 0 as desired.
    bbox_coords[:, ::2] = cp.iinfo(coord_dtype).max

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    bbox_coords_kernel(label_image, bbox_coords, size=label_image.size)

    counts = cp.zeros((max_label,), dtype=cp.uint32)
    centroids_sums = cp.zeros((max_label, ndim), dtype=cp.uint64)
    centroid_local_kernel = get_centroid_local_kernel(
        coord_dtype, label_image.ndim
    )
    centroid_local_kernel(
        label_image, bbox_coords, counts, centroids_sums, size=label_image.size
    )

    centroid_local = centroids_sums / counts[:, cp.newaxis]

    return counts, centroid_local


@cp.memoize(for_each_device=True)
def get_area_bbox_kernel(
    coord_dtype, area_dtype, ndim, compute_coordinate_sums=False
):
    coord_dtype = cp.dtype(coord_dtype)
    area_dtype = cp.dtype(area_dtype)
    uint_t = (
        "unsigned int" if coord_dtype.itemsize <= 4 else "unsigned long long"
    )

    source = f"""
       {uint_t} dim_max_offset;
       unsigned long long num_pixels_bbox = 1;
    """
    for d in range(ndim):
        source += f"""
           dim_max_offset = i * {2 * ndim} + {2*d + 1};
           num_pixels_bbox *= bbox[dim_max_offset] - bbox[dim_max_offset - 1];
        """
    source += """
        area_bbox = num_pixels_bbox * pixel_area;
    """
    inputs = f"raw {coord_dtype.name} bbox, float64 pixel_area"
    outputs = f"{area_dtype.name} area_bbox"
    name = f"cucim_area_bbox_{coord_dtype.name}_{area_dtype.name}_{ndim}d"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_area_bbox(bbox, area_dtype=cp.float32, spacing=None):
    num_label = bbox.shape[0]
    ndim = bbox.shape[1] // 2

    if spacing is None:
        pixel_area = 1.0
    else:
        if isinstance(spacing, cp.ndarray):
            pixel_area = cp.product(spacing)
        else:
            pixel_area = math.prod(spacing)

    # make a copy if the inputs are not already C-contiguous
    if not bbox.flags.c_contiguous:
        bbox = cp.ascontiguousarray(bbox)

    kernel = get_area_bbox_kernel(bbox.dtype, area_dtype, ndim)
    area_bbox = cp.empty((num_label,), dtype=area_dtype)
    kernel(bbox, pixel_area, area_bbox)
    return area_bbox


def area_bbox_from_slices(slices, area_dtype=cp.float32, spacing=None):
    num_label = len(slices)
    if spacing is None:
        pixel_area = 1.0
    else:
        if isinstance(spacing, cp.ndarray):
            pixel_area = cp.product(spacing)
        else:
            pixel_area = math.prod(spacing)

    area_bbox = np.empty((num_label,), dtype=area_dtype)
    for i, slice_tuple in enumerate(slices):
        num_pixels = 1
        for sl in slice_tuple:
            num_pixels *= sl.stop - sl.start
        area_bbox[i] = num_pixels * pixel_area

    return cp.asarray(area_bbox)


@cp.memoize(for_each_device=True)
def get_moments_kernel(
    coord_dtype,
    moments_dtype,
    ndim,
    order,
    spacing=None,
    weighted=False,
    num_channels=1,
):
    # note: ndim here is the number of spatial image dimensions

    coord_dtype = cp.dtype(coord_dtype)
    moments_dtype = cp.dtype(moments_dtype)

    use_floating_point = moments_dtype.kind == "f"
    if weighted and not use_floating_point:
        raise ValueError(
            "`moments_dtype` must be a floating point type for weighted "
            "moments calculations."
        )
    uint_t = (
        "unsigned int" if coord_dtype.itemsize <= 4 else "unsigned long long"
    )
    c_type = "double" if use_floating_point else f"{uint_t}"

    if spacing is not None:
        if len(spacing) != ndim:
            raise ValueError("len(spacing) must equal len(shape)")
        if moments_dtype.kind != "f":
            raise ValueError("moments must have a floating point data type")

    # number is for a densely populated moments matrix of size (order + 1) per
    # side (values at locations where order is greater than specified will be 0)
    num_moments = (order + 1) ** ndim

    source = """
        auto L = label[i];
        if (L != 0) {"""
    source += _unravel_loop_index("label", ndim, uint_t=uint_t)
    # using bounding box to transform the global coordinates to local ones
    # (c0 = local coordinate on axis 0, etc.)
    for d in range(ndim):
        source += f"""
            {c_type} c{d} = in_coord[{d}]
                            - bbox[(L - 1) * {2 * ndim} + {2*d}];"""
        if spacing:
            source += f"""
            c{d} *= spacing[{d}];"""
    if order > 3:
        raise ValueError("Only moments of orders 0-3 are supported")

    if num_channels > 1:
        # insert a loop over channels in multichannel case
        source += f"""
            {uint_t} num_channels = moments.shape()[1];
            for ({uint_t} c = 0; c < num_channels; c++) {{\n"""
    else:
        source += f"""
            {uint_t} num_channels = 1;
            {uint_t} c = 0;\n"""
    source += f"""
            {uint_t} offset = (L - 1) * {num_moments} * num_channels
                              + c * {num_moments};\n"""

    if weighted:
        source += f"""
            {uint_t} img_offset = i * num_channels + c;
            auto w = static_cast<{c_type}>(img[img_offset]);
            atomicAdd(&moments[offset], w);\n"""
    else:
        if use_floating_point:
            source += """
            atomicAdd(&moments[offset], 1.0);\n"""
        else:
            source += """
            atomicAdd(&moments[offset], 1);\n"""

    # need additional multiplication by the intensity value for weighted case
    w = "w * " if weighted else ""
    if ndim == 2:
        if order == 1:
            source += f"""
            atomicAdd(&moments[offset + 1], {w}c1);
            atomicAdd(&moments[offset + 2], {w}c0);\n"""
        elif order == 2:
            source += f"""
            atomicAdd(&moments[offset + 1], {w}c1);
            atomicAdd(&moments[offset + 2], {w}c1 * c1);
            atomicAdd(&moments[offset + 3], {w}c0);
            atomicAdd(&moments[offset + 4], {w}c0 * c1);
            atomicAdd(&moments[offset + 6], {w}c0 * c0);\n"""
        elif order == 3:
            source += f"""
            atomicAdd(&moments[offset + 1], {w}c1);
            atomicAdd(&moments[offset + 2], {w}c1 * c1);
            atomicAdd(&moments[offset + 3], {w}c1 * c1 * c1);
            atomicAdd(&moments[offset + 4], {w}c0);
            atomicAdd(&moments[offset + 5], {w}c0 * c1);
            atomicAdd(&moments[offset + 6], {w}c0 * c1 * c1);
            atomicAdd(&moments[offset + 8], {w}c0 * c0);
            atomicAdd(&moments[offset + 9], {w}c0 * c0 * c1);
            atomicAdd(&moments[offset + 12], {w}c0 * c0 * c0);\n"""
    elif ndim == 3:
        if order == 1:
            source += f"""
            atomicAdd(&moments[offset + 1], {w}c2);
            atomicAdd(&moments[offset + 2], {w}c1);
            atomicAdd(&moments[offset + 4], {w}c0);\n"""
        elif order == 2:
            source += f"""
            atomicAdd(&moments[offset + 1], {w}c2);
            atomicAdd(&moments[offset + 2], {w}c2 * c2);
            atomicAdd(&moments[offset + 3], {w}c1);
            atomicAdd(&moments[offset + 4], {w}c1 * c2);
            atomicAdd(&moments[offset + 6], {w}c1 * c1);
            atomicAdd(&moments[offset + 9], {w}c0);
            atomicAdd(&moments[offset + 10], {w}c0 * c2);
            atomicAdd(&moments[offset + 12], {w}c0 * c1);
            atomicAdd(&moments[offset + 18], {w}c0 * c0);\n"""
        elif order == 3:
            source += f"""
            atomicAdd(&moments[offset + 1], {w}c2);
            atomicAdd(&moments[offset + 2], {w}c2 * c2);
            atomicAdd(&moments[offset + 3], {w}c2 * c2 * c2);
            atomicAdd(&moments[offset + 4], {w}c1);
            atomicAdd(&moments[offset + 5], {w}c1 * c2);
            atomicAdd(&moments[offset + 6], {w}c1 * c2 * c2);
            atomicAdd(&moments[offset + 8], {w}c1 * c1);
            atomicAdd(&moments[offset + 9], {w}c1 * c1 * c2);
            atomicAdd(&moments[offset + 12], {w}c1 * c1 * c1);
            atomicAdd(&moments[offset + 16], {w}c0);
            atomicAdd(&moments[offset + 17], {w}c0 * c2);
            atomicAdd(&moments[offset + 18], {w}c0 * c2 * c2);
            atomicAdd(&moments[offset + 20], {w}c0 * c1);
            atomicAdd(&moments[offset + 21], {w}c0 * c1 * c2);
            atomicAdd(&moments[offset + 24], {w}c0 * c1 * c1);
            atomicAdd(&moments[offset + 32], {w}c0 * c0);
            atomicAdd(&moments[offset + 33], {w}c0 * c0 * c2);
            atomicAdd(&moments[offset + 36], {w}c0 * c0 * c1);
            atomicAdd(&moments[offset + 48], {w}c0 * c0 * c0);\n"""
    else:
        raise ValueError("only ndim = 2 or 3 is supported")
    if num_channels > 1:
        source += """
            }  // channels loop"
        """
    source += """
        }\n"""
    inputs = f"raw X label, raw {coord_dtype.name} bbox"
    if spacing:
        inputs += ", raw float64 spacing"
    if weighted:
        inputs += ", raw Y img"
    outputs = f"raw {moments_dtype.name} moments"
    weighted_str = "_weighted" if weighted else ""
    spacing_str = "_sp" if spacing else ""
    name = f"cucim_moments{weighted_str}{spacing_str}_order{order}_{ndim}d_"
    name += f"{coord_dtype.name}_{moments_dtype.name}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_moments(
    label_image,
    intensity_image=None,
    max_label=None,
    order=2,
    spacing=None,
    coord_dtype=cp.uint32,
):
    """
    Parameters
    ----------
    label_image : cp.ndarray
        Image containing labels where 0 is the background and sequential
        values > 0 are the labels.
    intensity_image : cp.ndarray, optional
        Image of intensities. If provided, weighted moments are computed. If
        this is a multi-channel image, moments are computed independently for
        each channel.
    max_label : int or None, optional
        The maximum label value present in label_image. Will be computed if not
        provided.
    coord_dtype : dtype, optional
        The data type to use for coordinate calculations. Should be
        ``cp.uint32`` or ``cp.uint64``.

    Returns
    -------
    moments : cp.ndarray
        The moments up to the specified order. Will be stored in an
        ``(order + 1, ) * ndim`` matrix where any elements corresponding to
        order greater than that specified will be set to 0.  For example, for
        the 2D case, the last two axes represent the 2D moments matrix, ``M``
        where each matrix would have the following sizes and non-zero entries:

            ```py
            # for a 2D image with order = 1
            M = [
               [m00, m01],
               [m10,   0],
            ]

            # for a 2D image with order = 2
            M = [
               [m00, m01, m02],
               [m10, m11,   0],
               [m20,   0,   0],
            ]

            # for a 2D image with order = 3
            M = [
               [m00, m01, m02, m03],
               [m10, m11, m12,   0],
               [m20, m21,   0,   0],
               [m30,   0,   0,   0],
            ]
            ```

        When there is no `intensity_image` or the `intensity_image` is single
        channel, the shape of the moments output is
        ``shape = (max_label, ) + (order + 1, ) * ndim``.
        When the ``intensity_image`` is multichannel a channel axis will be
        present in the `moments` output at position 1 to give
        ``shape = (max_label, ) + (num_channels, ) + (order + 1,) * ndim``.
    """
    if max_label is None:
        max_label = int(label_image.max())

    bbox_coords_kernel = get_bbox_coords_kernel(
        coord_dtype,
        label_image.ndim,
        compute_coordinate_sums=False,
    )

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    ndim = label_image.ndim
    moments_shape = (max_label,) + (order + 1,) * ndim
    if intensity_image is not None:
        if not intensity_image.flags.c_contiguous:
            intensity_image = cp.ascontiguousarray(intensity_image)

        num_channels = _check_shapes(label_image, intensity_image)
        if num_channels > 1:
            moments_shape = (max_label,) + (num_channels,) + (order + 1,) * ndim
        weighted = True
    else:
        num_channels = 1
        weighted = False

    bbox_coords = cp.zeros((max_label, 2 * ndim), dtype=coord_dtype)

    # Initialize value for atomicMin on even coordinates
    # The value for atomicMax columns is already 0 as desired.
    bbox_coords[:, ::2] = cp.iinfo(coord_dtype).max

    bbox_coords_kernel(label_image, bbox_coords, size=label_image.size)

    # total number of elements in the moments matrix
    moments = cp.zeros(moments_shape, dtype=cp.float64)
    moments_kernel = get_moments_kernel(
        coord_dtype,
        moments.dtype,
        label_image.ndim,
        order=order,
        spacing=spacing,
        weighted=weighted,
        num_channels=num_channels,
    )
    input_args = (
        label_image,
        bbox_coords,
    )
    if spacing:
        input_args = input_args + (cp.asarray(spacing, dtype=cp.float64),)
    if weighted:
        input_args = input_args + (intensity_image,)
    moments_kernel(*input_args, moments, size=label_image.size)
    return moments


@cp.memoize(for_each_device=True)
def get_moments_central_kernel(
    moments_dtype,
    ndim,
    order,
):
    """Applies analytical formulas to convert raw moments to central moments

    These are as in `_moments_raw_to_central_fast` from
    `_moments_analytical.py` but that kernel is scalar, while this one will be
    applied to all labeled regions (and any channels dimension) at once.
    """
    moments_dtype = cp.dtype(moments_dtype)

    uint_t = "unsigned int"

    # number is for a densely populated moments matrix of size (order + 1) per
    # side (values at locations where order is greater than specified will be 0)
    num_moments = (order + 1) ** ndim

    if moments_dtype.kind != "f":
        raise ValueError(
            "`moments_dtype` must be a floating point type for central moments "
            "calculations."
        )

    # floating point type used for the intermediate computations
    float_type = "double"

    source = f"""
            {uint_t} offset = i * {num_moments};\n"""
    if ndim == 2:
        if order <= 1:
            # only zeroth moment is non-zero for central moments
            source += """
            out[offset] = moments_raw[offset];\n"""
        elif order == 2:
            source += f"""
            // retrieve the 2nd order raw moments
            {float_type} m00 = moments_raw[offset];
            {float_type} m01 = moments_raw[offset + 1];
            {float_type} m02 = moments_raw[offset + 2];
            {float_type} m10 = moments_raw[offset + 3];
            {float_type} m11 = moments_raw[offset + 4];
            {float_type} m20 = moments_raw[offset + 6];

            // compute centroids
            // (TODO: add option to output the centroids as well?)
            {float_type} cx = m10 / m00;
            {float_type} cy = m01 / m00;

            // analytical expressions for the central moments
            out[offset] = m00;                  // out[0, 0]
            // 2nd order central moments
            out[offset + 2] = m02 - cy * m01;   // out[0, 2]
            out[offset + 4] = m11 - cx * m01;   // out[1, 1]
            out[offset + 6] = m20 - cx * m10;   // out[2, 0]\n"""
        elif order == 3:
            source += f"""
            // retrieve the 2nd order raw moments
            {float_type} m00 = moments_raw[offset];
            {float_type} m01 = moments_raw[offset + 1];
            {float_type} m02 = moments_raw[offset + 2];
            {float_type} m03 = moments_raw[offset + 3];
            {float_type} m10 = moments_raw[offset + 4];
            {float_type} m11 = moments_raw[offset + 5];
            {float_type} m12 = moments_raw[offset + 6];
            {float_type} m20 = moments_raw[offset + 8];
            {float_type} m21 = moments_raw[offset + 9];
            {float_type} m30 = moments_raw[offset + 12];

            // compute centroids
            {float_type} cx = m10 / m00;
            {float_type} cy = m01 / m00;

            // zeroth moment
            out[offset] = m00;                                                  // out[0, 0]
            // 2nd order central moments
            out[offset + 2] = m02 - cy * m01;                                   // out[0, 2]
            out[offset + 5] = m11 - cx * m01;                                   // out[1, 1]
            out[offset + 8] = m20 - cx * m10;                                   // out[2, 0]
            // 3rd order central moments
            out[offset + 3] = m03 - 3*cy*m02 + 2*cy*cy*m01;                     // out[0, 3]
            out[offset + 6] = m12 - 2*cy*m11 - cx*m02 + 2*cy*cx*m01;            // out[1, 2]
            out[offset + 9] = m21 - 2*cx*m11 - cy*m20 + cx*cx*m01 + cy*cx*m10;  // out[2, 1]
            out[offset + 12] = m30 - 3*cx*m20 + 2*cx*cx*m10;                    // out[3, 0]\n"""  # noqa: E501
        else:
            raise ValueError("only order <= 3 is supported")
    elif ndim == 3:
        if order <= 1:
            # only zeroth moment is non-zero for central moments
            source += """
            out[offset] = moments_raw[offset];\n"""
        elif order == 2:
            source += f"""
             // retrieve the 2nd order raw moments
            {float_type} m000 = moments_raw[offset];
            {float_type} m001 = moments_raw[offset + 1];
            {float_type} m002 = moments_raw[offset + 2];
            {float_type} m010 = moments_raw[offset + 3];
            {float_type} m011 = moments_raw[offset + 4];
            {float_type} m020 = moments_raw[offset + 6];
            {float_type} m100 = moments_raw[offset + 9];
            {float_type} m101 = moments_raw[offset + 10];
            {float_type} m110 = moments_raw[offset + 12];
            {float_type} m200 = moments_raw[offset + 18];

            // compute centroids
            {float_type} cx = m100 / m000;
            {float_type} cy = m010 / m000;
            {float_type} cz = m001 / m000;

            // zeroth moment
            out[offset] = m000;                  // out[0, 0, 0]
            // 2nd order central moments
            out[offset + 2] = -cz*m001 + m002;   // out[0, 0, 2]
            out[offset + 4] = -cy*m001 + m011;   // out[0, 1, 1]
            out[offset + 6] = -cy*m010 + m020;   // out[0, 2, 0]
            out[offset + 10] = -cx*m001 + m101;  // out[1, 0, 1]
            out[offset + 12] = -cx*m010 + m110;  // out[1, 1, 0]
            out[offset + 18] = -cx*m100 + m200;  // out[2, 0, 0]\n"""
        elif order == 3:
            source += f"""
             // retrieve the 3rd order raw moments
            {float_type} m000 = moments_raw[offset];
            {float_type} m001 = moments_raw[offset + 1];
            {float_type} m002 = moments_raw[offset + 2];
            {float_type} m003 = moments_raw[offset + 3];
            {float_type} m010 = moments_raw[offset + 4];
            {float_type} m011 = moments_raw[offset + 5];
            {float_type} m012 = moments_raw[offset + 6];
            {float_type} m020 = moments_raw[offset + 8];
            {float_type} m021 = moments_raw[offset + 9];
            {float_type} m030 = moments_raw[offset + 12];
            {float_type} m100 = moments_raw[offset + 16];
            {float_type} m101 = moments_raw[offset + 17];
            {float_type} m102 = moments_raw[offset + 18];
            {float_type} m110 = moments_raw[offset + 20];
            {float_type} m111 = moments_raw[offset + 21];
            {float_type} m120 = moments_raw[offset + 24];
            {float_type} m200 = moments_raw[offset + 32];
            {float_type} m201 = moments_raw[offset + 33];
            {float_type} m210 = moments_raw[offset + 36];
            {float_type} m300 = moments_raw[offset + 48];

            // compute centroids
            {float_type} cx = m100 / m000;
            {float_type} cy = m010 / m000;
            {float_type} cz = m001 / m000;

            // zeroth moment
            out[offset] = m000;
            // 2nd order central moments
            out[offset + 2] = -cz*m001 + m002;     // out[0, 0, 2]
            out[offset + 5] = -cy*m001 + m011;     // out[0, 1, 1]
            out[offset + 8] = -cy*m010 + m020;     // out[0, 2, 0]
            out[offset + 17] = -cx*m001 + m101;    // out[1, 0, 1]
            out[offset + 20] = -cx*m010 + m110;    // out[1, 1, 0]
            out[offset + 32] = -cx*m100 + m200;    // out[2, 0, 0]
            // 3rd order central moments
            out[offset + 3] = 2*cz*cz*m001 - 3*cz*m002 + m003;                               // out[0, 0, 3]
            out[offset + 6] = -cy*m002 + 2*cz*(cy*m001 - m011) + m012;                       // out[0, 1, 2]
            out[offset + 9] = cy*cy*m001 - 2*cy*m011 + cz*(cy*m010 - m020) + m021;           // out[0, 2, 1]
            out[offset + 12] = 2*cy*cy*m010 - 3*cy*m020 + m030;                              // out[0, 3, 0]
            out[offset + 18] = -cx*m002 + 2*cz*(cx*m001 - m101) + m102;                      // out[1, 0, 2]
            out[offset + 21] = -cx*m011 + cy*(cx*m001 - m101) + cz*(cx*m010 - m110) + m111;  // out[1, 1, 1]
            out[offset + 24] = -cx*m020 - 2*cy*(-cx*m010 + m110) + m120;                     // out[1, 2, 0]
            out[offset + 33] = cx*cx*m001 - 2*cx*m101 + cz*(cx*m100 - m200) + m201;          // out[2, 0, 1]
            out[offset + 36] = cx*cx*m010 - 2*cx*m110 + cy*(cx*m100 - m200) + m210;          // out[2, 1, 0]
            out[offset + 48] = 2*cx*cx*m100 - 3*cx*m200 + m300;                              // out[3, 0, 0]\n"""  # noqa: E501
        else:
            raise ValueError("only order <= 3 is supported")
    else:
        # note: ndim here is the number of spatial image dimensions
        raise ValueError("only ndim = 2 or 3 is supported")
    inputs = "raw X moments_raw"
    outputs = "raw X out"
    name = f"cucim_moments_central_order{order}_{ndim}d"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_moments_central(moments_raw, ndim):
    if moments_raw.ndim == 2 + ndim:
        num_channels = moments_raw.shape[1]
    elif moments_raw.ndim == 1 + ndim:
        num_channels = 1
    else:
        raise ValueError(
            f"{moments_raw.shape=} does not have expected length of `ndim + 1`"
            " (or `ndim + 2` for the multi-channel weighted moments case)."
        )
    order = moments_raw.shape[-1] - 1
    max_label = moments_raw.shape[0]

    if moments_raw.dtype.kind != "f":
        float_dtype = cp.promote_types(cp.float32, moments_raw.dtype)
        moments_raw = moments_raw.astype(float_dtype)

    # make a copy if the inputs are not already C-contiguous
    if not moments_raw.flags.c_contiguous:
        moments_raw = cp.ascontiguousarray(moments_raw)

    moments_kernel = get_moments_central_kernel(moments_raw.dtype, ndim, order)
    moments_central = cp.zeros_like(moments_raw)
    # kernel loops over moments so size is max_label * num_channels
    moments_kernel(moments_raw, moments_central, size=max_label * num_channels)
    return moments_central


@cp.memoize(for_each_device=True)
def get_moments_normalize_kernel(moments_dtype, ndim, order, unit_scale=False):
    """Normalizes central moments of order >=2"""
    moments_dtype = cp.dtype(moments_dtype)

    uint_t = "unsigned int"

    # number is for a densely populated moments matrix of size (order + 1) per
    # side (values at locations where order is greater than specified will be 0)
    num_moments = (order + 1) ** ndim

    if moments_dtype.kind != "f":
        raise ValueError(
            "`moments_dtype` must be a floating point type for central moments "
            "calculations."
        )

    # floating point type used for the intermediate computations
    float_type = "double"

    source = f"""
            {uint_t} offset = i * {num_moments};\n"""
    if ndim == 2:
        if order == 2:
            source += f"""
            // retrieve zeroth moment
            {float_type} m00 = moments_central[offset];\n"""

            # compute normalization factor
            source += f"""
            {float_type} norm_order2 = pow(m00, 2.0 / {ndim} + 1.0);"""
            if not unit_scale:
                source += """
                norm_order2 *= scale * scale;\n"""

            # normalize
            source += """
            // normalize the 2nd order central moments
            out[offset + 2] = moments_central[offset + 2] / norm_order2;  // out[0, 2]
            out[offset + 4] = moments_central[offset + 4] / norm_order2;  // out[1, 1]
            out[offset + 6] = moments_central[offset + 6] / norm_order2;  // out[2, 0]\n"""  # noqa: E501
        elif order == 3:
            source += f"""
            // retrieve zeroth moment
            {float_type} m00 = moments_central[offset];\n"""

            # compute normalization factor
            source += f"""
            {float_type} norm_order2 = pow(m00, 2.0 / {ndim} + 1.0);
            {float_type} norm_order3 = pow(m00, 3.0 / {ndim} + 1.0);"""
            if not unit_scale:
                source += """
                norm_order2 *= scale * scale;
                norm_order3 *= scale * scale * scale;\n"""

            # normalize
            source += """
            // normalize the 2nd order central moments
            out[offset + 2] = moments_central[offset + 2] / norm_order2;  // out[0, 2]
            out[offset + 5] = moments_central[offset + 5] / norm_order2;  // out[1, 1]
            out[offset + 8] = moments_central[offset + 8] / norm_order2;  // out[2, 0]
            // normalize the 3rd order central moments
            out[offset + 3] = moments_central[offset + 3] / norm_order3;    // out[0, 3]
            out[offset + 6] = moments_central[offset + 6] / norm_order3;    // out[1, 2]
            out[offset + 9] = moments_central[offset + 9] / norm_order3;    // out[2, 1]
            out[offset + 12] = moments_central[offset + 12] / norm_order3;  // out[3, 0]\n"""  # noqa: E501
        else:
            raise ValueError("only order = 2 or 3 is supported")
    elif ndim == 3:
        if order == 2:
            source += f"""
            // retrieve the zeroth moment
            {float_type} m000 = moments_central[offset];\n"""

            # compute normalization factor
            source += f"""
            {float_type} norm_order2 = pow(m000, 2.0 / {ndim} + 1.0);"""
            if not unit_scale:
                source += """
                norm_order2 *= scale * scale;\n"""

            # normalize
            source += """
            // normalize the 2nd order central moments
            out[offset + 2] = moments_central[offset + 2] / norm_order2;    // out[0, 0, 2]
            out[offset + 4] = moments_central[offset + 4] / norm_order2;    // out[0, 1, 1]
            out[offset + 6] = moments_central[offset + 6] / norm_order2;    // out[0, 2, 0]
            out[offset + 10] = moments_central[offset + 10] / norm_order2;  // out[1, 0, 1]
            out[offset + 12] = moments_central[offset + 12] / norm_order2;  // out[1, 1, 0]
            out[offset + 18] = moments_central[offset + 18] / norm_order2;  // out[2, 0, 0]\n"""  # noqa: E501
        elif order == 3:
            source += f"""
            // retrieve the zeroth moment
            {float_type} m000 = moments_central[offset];\n"""

            # compute normalization factor
            source += f"""
            {float_type} norm_order2 = pow(m000, 2.0 / {ndim} + 1.0);
            {float_type} norm_order3 = pow(m000, 3.0 / {ndim} + 1.0);"""
            if not unit_scale:
                source += """
                norm_order2 *= scale * scale;
                norm_order3 *= scale * scale * scale;\n"""

            # normalize
            source += """
            // normalize the 2nd order central moments
            out[offset + 2] = moments_central[offset + 2] / norm_order2;    // out[0, 0, 2]
            out[offset + 5] = moments_central[offset + 5] / norm_order2;    // out[0, 1, 1]
            out[offset + 8] = moments_central[offset + 8] / norm_order2;    // out[0, 2, 0]
            out[offset + 17] = moments_central[offset + 17] / norm_order2;  // out[1, 0, 1]
            out[offset + 20] = moments_central[offset + 20] / norm_order2;  // out[1, 1, 0]
            out[offset + 32] = moments_central[offset + 32] / norm_order2;  // out[2, 0, 0]
            // normalize the 3rd order central moments
            out[offset + 3] = moments_central[offset + 3] / norm_order3;    // out[0, 0, 3]
            out[offset + 6] = moments_central[offset + 6] / norm_order3;    // out[0, 1, 2]
            out[offset + 9] = moments_central[offset + 9] / norm_order3;    // out[0, 2, 1]
            out[offset + 12] = moments_central[offset + 12] / norm_order3;  // out[0, 3, 0]
            out[offset + 18] = moments_central[offset + 18] / norm_order3;  // out[1, 0, 2]
            out[offset + 21] = moments_central[offset + 21] / norm_order3;  // out[1, 1, 1]
            out[offset + 24] = moments_central[offset + 24] / norm_order3;  // out[1, 2, 0]
            out[offset + 33] = moments_central[offset + 33] / norm_order3;  // out[2, 0, 1]
            out[offset + 36] = moments_central[offset + 36] / norm_order3;  // out[2, 1, 0]
            out[offset + 48] = moments_central[offset + 48] / norm_order3;  // out[3, 0, 0]\n"""  # noqa: E501
        else:
            raise ValueError("only order = 2 or 3 is supported")
    else:
        # note: ndim here is the number of spatial image dimensions
        raise ValueError("only ndim = 2 or 3 is supported")
    inputs = "raw X moments_central"
    if not unit_scale:
        inputs += ", float64 scale"
    outputs = "raw X out"
    name = f"cucim_moments_normalized_order{order}_{ndim}d"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_moments_normalized(moments_central, ndim, spacing=None):
    if moments_central.ndim == 2 + ndim:
        num_channels = moments_central.shape[1]
    elif moments_central.ndim == 1 + ndim:
        num_channels = 1
    else:
        raise ValueError(
            f"{moments_central.shape=} does not have expected length of "
            " `ndim + 1` (or `ndim + 2` for the multi-channel weighted moments "
            "case)."
        )
    order = moments_central.shape[-1] - 1
    if order < 2 or order > 3:
        raise ValueError(
            "normalized moment calculations only implemented for order=2 "
            "and order=3"
        )
    if ndim < 2 or ndim > 3:
        raise ValueError(
            "moment normalization only implemented for 2D and 3D images"
        )
    max_label = moments_central.shape[0]

    if moments_central.dtype.kind != "f":
        raise ValueError("moments_central must have a floating point dtype")

    # make a copy if the inputs are not already C-contiguous
    if not moments_central.flags.c_contiguous:
        moments_central = cp.ascontiguousarray(moments_central)

    if spacing is None:
        unit_scale = True
        inputs = (moments_central,)
    else:
        if spacing:
            if isinstance(spacing, cp.ndarray):
                scale = spacing.min()
            else:
                scale = float(min(spacing))
        unit_scale = False
        inputs = (moments_central, scale)

    moments_norm_kernel = get_moments_normalize_kernel(
        moments_central.dtype, ndim, order, unit_scale=unit_scale
    )
    # output is NaN except for locations with orders in range [2, order]
    moments_norm = cp.full(
        moments_central.shape, cp.nan, dtype=moments_central.dtype
    )

    # kernel loops over moments so size is max_label * num_channels
    moments_norm_kernel(*inputs, moments_norm, size=max_label * num_channels)
    return moments_norm


@cp.memoize(for_each_device=True)
def get_moments_hu_kernel(moments_dtype):
    """Normalizes central moments of order >=2"""
    moments_dtype = cp.dtype(moments_dtype)

    uint_t = "unsigned int"

    # number is for a densely populated moments matrix of size (order + 1) per
    # side (values at locations where order is greater than specified will be 0)
    num_moments = 16

    if moments_dtype.kind != "f":
        raise ValueError(
            "`moments_dtype` must be a floating point type for central moments "
            "calculations."
        )

    # floating point type used for the intermediate computations
    float_type = "double"

    # compute offset to the current moment matrix and hu moment vector
    source = f"""
            {uint_t} offset_normalized = i * {num_moments};
            {uint_t} offset_hu = i * 7;\n"""

    source += f"""
    // retrieve 2nd and 3rd order normalized moments
    {float_type} m02 = moments_normalized[offset_normalized + 2];
    {float_type} m03 = moments_normalized[offset_normalized + 3];
    {float_type} m12 = moments_normalized[offset_normalized + 6];
    {float_type} m11 = moments_normalized[offset_normalized + 5];
    {float_type} m20 = moments_normalized[offset_normalized + 8];
    {float_type} m21 = moments_normalized[offset_normalized + 9];
    {float_type} m30 = moments_normalized[offset_normalized + 12];

    {float_type} t0 = m30 + m12;
    {float_type} t1 = m21 + m03;
    {float_type} q0 = t0 * t0;
    {float_type} q1 = t1 * t1;
    {float_type} n4 = 4 * m11;
    {float_type} s = m20 + m02;
    {float_type} d = m20 - m02;
    hu[offset_hu] = s;
    hu[offset_hu + 1] = d * d + n4 * m11;
    hu[offset_hu + 3] = q0 + q1;
    hu[offset_hu + 5] = d * (q0 - q1) + n4 * t0 * t1;
    t0 *= q0 - 3 * q1;
    t1 *= 3 * q0 - q1;
    q0 = m30- 3 * m12;
    q1 = 3 * m21 - m03;
    hu[offset_hu + 2] = q0 * q0 + q1 * q1;
    hu[offset_hu + 4] = q0 * t0 + q1 * t1;
    hu[offset_hu + 6] = q1 * t0 - q0 * t1;\n"""

    inputs = f"raw {moments_dtype.name} moments_normalized"
    outputs = f"raw {moments_dtype.name} hu"
    name = f"cucim_moments_hu_order_{moments_dtype.name}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_moments_hu(moments_normalized):
    if moments_normalized.ndim == 4:
        num_channels = moments_normalized.shape[1]
    elif moments_normalized.ndim == 3:
        num_channels = 1
    else:
        raise ValueError(
            "Hu's moments are only defined for 2D images. Expected "
            "`moments_normalized to have 3 dimensions (or 4 for the "
            "multi-channel `intensity_image` case)."
        )
    order = moments_normalized.shape[-1] - 1
    if order < 3:
        raise ValueError(
            "Calculating Hu's moments requires normalized moments of "
            "order >= 3 to be provided as input"
        )
    elif order > 3:
        # truncate any unused moments
        moments_normalized = cp.ascontiguousarray(
            moments_normalized[..., :4, :4]
        )
    max_label = moments_normalized.shape[0]

    if moments_normalized.dtype.kind != "f":
        raise ValueError("moments_normalized must have a floating point dtype")

    # make a copy if the inputs are not already C-contiguous
    if not moments_normalized.flags.c_contiguous:
        moments_normalized = cp.ascontiguousarray(moments_normalized)

    moments_hu_kernel = get_moments_hu_kernel(moments_normalized.dtype)
    # Hu's moments are a set of 7 moments stored instead of a moment matrix
    hu_shape = moments_normalized.shape[:-2] + (7,)
    moments_hu = cp.full(hu_shape, cp.nan, dtype=moments_normalized.dtype)

    # kernel loops over moments so size is max_label * num_channels
    moments_hu_kernel(
        moments_normalized, moments_hu, size=max_label * num_channels
    )
    return moments_hu


@cp.memoize(for_each_device=True)
def get_inertia_tensor_kernel(moments_dtype, ndim, compute_orientation):
    """Normalizes central moments of order >=2"""
    moments_dtype = cp.dtype(moments_dtype)

    # assume moments input was truncated to only hold order<=2 moments
    num_moments = 3**ndim

    # size of the inertia_tensor matrix
    num_out = ndim * ndim

    if moments_dtype.kind != "f":
        raise ValueError(
            "`moments_dtype` must be a floating point type for central moments "
            "calculations."
        )

    source = f"""
            unsigned int offset = i * {num_moments};
            unsigned int offset_out = i * {num_out};\n"""
    if ndim == 2:
        source += """
        F mu0 = moments_central[offset];
        F mxx = moments_central[offset + 6];
        F myy = moments_central[offset + 2];
        F mxy = moments_central[offset + 4];

        F a = myy / mu0;
        F b = -mxy / mu0;
        F c = mxx / mu0;
        out[offset_out + 0] = a;
        out[offset_out + 1] = b;
        out[offset_out + 2] = b;
        out[offset_out + 3] = c;
        """
        if compute_orientation:
            source += """
        if (a - c == 0) {
          // had to use <= 0 to get same result as Python's atan2 with < 0
          if (b < 0) {
            orientation[i] = -M_PI / 4.0;
          } else {
            orientation[i] = M_PI / 4.0;
          }
        } else {
          orientation[i] = 0.5 * atan2(-2 * b, c - a);
        }\n"""
    elif ndim == 3:
        if compute_orientation:
            raise ValueError("orientation can only be computed for 2d images")
        source += """
        F mu0 = moments_central[offset];       // [0, 0, 0]
        F mxx = moments_central[offset + 18];  // [2, 0, 0]
        F myy = moments_central[offset + 6];   // [0, 2, 0]
        F mzz = moments_central[offset + 2];   // [0, 0, 2]

        F mxy = moments_central[offset + 12];  // [1, 1, 0]
        F mxz = moments_central[offset + 10];  // [1, 0, 1]
        F myz = moments_central[offset + 4];   // [0, 1, 1]

        out[offset_out + 0] = (myy + mzz) / mu0;
        out[offset_out + 4] = (mxx + mzz) / mu0;
        out[offset_out + 8] = (mxx + myy) / mu0;
        out[offset_out + 1] = -mxy / mu0;
        out[offset_out + 3] = -mxy / mu0;
        out[offset_out + 2] = -mxz / mu0;
        out[offset_out + 6] = -mxz / mu0;
        out[offset_out + 5] = -myz / mu0;
        out[offset_out + 7] = -myz / mu0;\n"""
    else:
        # note: ndim here is the number of spatial image dimensions
        raise ValueError("only ndim = 2 or 3 is supported")
    inputs = "raw F moments_central"
    outputs = "raw F out"
    if compute_orientation:
        outputs += ", raw F orientation"
    name = f"cucim_inertia_tensor_{ndim}d"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_inertia_tensor(
    moments_central, ndim, compute_orientation=False
):
    if ndim < 2 or ndim > 3:
        raise ValueError("inertia tensor only implemented for 2D and 3D images")
    nbatch = math.prod(moments_central.shape[:-ndim])

    if moments_central.dtype.kind != "f":
        raise ValueError("moments_central must have a floating point dtype")

    # make a copy if the inputs are not already C-contiguous
    if not moments_central.flags.c_contiguous:
        moments_central = cp.ascontiguousarray(moments_central)

    order = moments_central.shape[-1] - 1
    if order < 2:
        raise ValueError(
            f"inertia tensor calculation requires order>=2, found {order}"
        )
    if order > 2:
        # truncate to only the 2nd order moments
        slice_kept = (Ellipsis,) + (slice(0, 3),) * ndim
        moments_central = cp.ascontiguousarray(moments_central[slice_kept])

    kernel = get_inertia_tensor_kernel(
        moments_central.dtype, ndim, compute_orientation=compute_orientation
    )
    itensor_shape = moments_central.shape[:-ndim] + (ndim, ndim)
    itensor = cp.zeros(itensor_shape, dtype=moments_central.dtype)
    if compute_orientation:
        if ndim != 2:
            raise ValueError("orientation can only be computed for ndim=2")
        orientation = cp.zeros(
            moments_central.shape[:-ndim], dtype=moments_central.dtype
        )
        kernel(moments_central, itensor, orientation, size=nbatch)
        return itensor, orientation

    kernel(moments_central, itensor, size=nbatch)
    return itensor


@cp.memoize(for_each_device=True)
def get_inertia_tensor_eigvals_kernel(ndim, compute_axis_lengths=False):
    """Compute inertia tensor eigenvalues

    C. Deledalle, L. Denis, S. Tabti, F. Tupin. Closed-form expressions
    of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices.

    [Research Report] Université de Lyon. 2017.
    https://hal.archives-ouvertes.fr/hal-01501221/file/matrix_exp_and_log_formula.pdf
    """  # noqa: E501

    # assume moments input was truncated to only hold order<=2 moments
    num_itensor = ndim * ndim

    # size of the inertia_tensor matrix
    source = f"""
            unsigned int offset = i * {num_itensor};
            unsigned int offset_out = i * {ndim};\n"""
    if ndim == 2:
        source += """
            F tmp1, tmp2;
            double m00 = static_cast<double>(inertia_tensor[offset]);
            double m01 = static_cast<double>(inertia_tensor[offset + 1]);
            double m11 = static_cast<double>(inertia_tensor[offset + 3]);
            tmp1 = m01 * m01;
            tmp1 *= 4;

            tmp2 = m00 - m11;
            tmp2 *= tmp2;
            tmp2 += tmp1;
            tmp2 = sqrt(tmp2);
            tmp2 /= 2;

            tmp1 = m00 + m11;
            tmp1 /= 2;

            // store in "descending" order and clip to positive values
            // (matrix is Hermitian, so negatives values can only be due to
            //  numerical errors)
            F lam1 = max(tmp1 + tmp2, 0.0);
            F lam2 = max(tmp1 - tmp2, 0.0);
            out[offset_out] = lam1;
            out[offset_out + 1] = lam2;\n"""
        if compute_axis_lengths:
            source += """
            axis_lengths[offset_out] = 4.0 * sqrt(lam1);
            axis_lengths[offset_out + 1] = 4.0 * sqrt(lam2);
            \n"""
    elif ndim == 3:
        source += """
            double x1, x2, phi;
            // extract triangle of (Hermitian) inertia tensor values
            // [a, d, f]
            // [-, b, e]
            // [-, -, c]
            double a = static_cast<double>(inertia_tensor[offset]);
            double b = static_cast<double>(inertia_tensor[offset + 4]);
            double c = static_cast<double>(inertia_tensor[offset + 8]);
            double d = static_cast<double>(inertia_tensor[offset + 1]);
            double e = static_cast<double>(inertia_tensor[offset + 5]);
            double f = static_cast<double>(inertia_tensor[offset + 2]);
            double d_sq = d * d;
            double e_sq = e * e;
            double f_sq = f * f;
            double tmpa = (2*a - b - c);
            double tmpb = (2*b - a - c);
            double tmpc = (2*c - a - b);
            x2 = - tmpa * tmpb * tmpc;
            x2 += 9 * (tmpc*d_sq + tmpb*f_sq + tmpa*e_sq);
            x2 -= 54 * (d * e * f);
            x1 = a*a + b*b + c*c - a*b - a*c - b*c + 3 * (d_sq + e_sq + f_sq);

            // grlee77: added max() here for numerical stability
            // (avoid NaN values in ridge filter test cases)
            x1 = max(x1, 0.0);

            if (x2 == 0.0) {
                phi = M_PI / 2.0;
            } else {
                // grlee77: added max() here for numerical stability
                // (avoid NaN values in test_hessian_matrix_eigvals_3d)
                double arg = max(4*x1*x1*x1 - x2*x2, 0.0);
                phi = atan(sqrt(arg)/x2);
                if (x2 < 0) {
                    phi += M_PI;
                }
            }
            double x1_term = (2.0 / 3.0) * sqrt(x1);
            double abc = (a + b + c) / 3.0;
            F lam1 = abc - x1_term * cos(phi / 3.0);
            F lam2 = abc + x1_term * cos((phi - M_PI) / 3.0);
            F lam3 = abc + x1_term * cos((phi + M_PI) / 3.0);

            // abc = 141.94321771
            // x1_term = 1279.25821493
            // M_PI = 3.14159265
            // phi = 1.91643394
            // cos(phi/3.0) = 0.80280507
            // cos((phi - M_PI) / 3.0) = 0.91776289

            F stmp;
            if (lam3 > lam2) {
                stmp = lam2;
                lam2 = lam3;
                lam3 = stmp;
            }
            if (lam3 > lam1) {
                stmp = lam1;
                lam1 = lam3;
                lam3 = stmp;
            }
            if (lam2 > lam1) {
                stmp = lam1;
                lam1 = lam2;
                lam2 = stmp;
            }
            // clip to positive values
            // (matrix is Hermitian, so negatives values can only be due to
            //  numerical errors)
            lam1 = max(lam1, 0.0);
            lam2 = max(lam2, 0.0);
            lam3 = max(lam3, 0.0);
            out[offset_out] = lam1;
            out[offset_out + 1] = lam2;
            out[offset_out + 2] = lam3;\n"""
        if compute_axis_lengths:
            source += """
            // formula reference:
            //   https://github.com/scikit-image/scikit-image/blob/v0.25.0/skimage/measure/_regionprops.py#L275-L295
            // note: added max to clip possible small (e.g. 1e-7) negative value due to numerical error
            axis_lengths[offset_out] = sqrt(10.0 * (lam1 + lam2 - lam3));
            axis_lengths[offset_out + 1] = sqrt(10.0 * (lam1 - lam2 + lam3));
            axis_lengths[offset_out + 2] = sqrt(10.0 * max(-lam1 + lam2 + lam3, 0.0));
            \n"""  # noqa: E501
    else:
        # note: ndim here is the number of spatial image dimensions
        raise ValueError("only ndim = 2 or 3 is supported")
    inputs = "raw F inertia_tensor"
    outputs = "raw F out"
    name = f"cucim_inertia_tensor_eigvals_{ndim}d"
    if compute_axis_lengths:
        outputs += ", raw F axis_lengths"
        name += "_with_axis"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_inertia_tensor_eigvals(
    inertia_tensor, compute_axis_lengths=False
):
    # inertia tensor should have shape (ndim, ndim) on last two axes
    ndim = inertia_tensor.shape[-1]
    if ndim < 2 or ndim > 3:
        raise ValueError("inertia tensor only implemented for 2D and 3D images")
    nbatch = math.prod(inertia_tensor.shape[:-2])

    if inertia_tensor.dtype.kind != "f":
        raise ValueError("moments_central must have a floating point dtype")

    if not inertia_tensor.flags.c_contiguous:
        inertia_tensor = cp.ascontiguousarray(inertia_tensor)

    kernel = get_inertia_tensor_eigvals_kernel(ndim, compute_axis_lengths)
    eigvals_shape = inertia_tensor.shape[:-2] + (ndim,)
    eigvals = cp.empty(eigvals_shape, dtype=inertia_tensor.dtype)
    if compute_axis_lengths:
        axis_lengths = cp.empty(eigvals_shape, dtype=inertia_tensor.dtype)
        kernel(inertia_tensor, eigvals, axis_lengths, size=nbatch)
        return eigvals, axis_lengths
    # kernel loops over moments so size is max_label * num_channels
    kernel(inertia_tensor, eigvals, size=nbatch)
    return eigvals


@cp.memoize(for_each_device=True)
def get_centroid_weighted_kernel(
    moments_dtype, ndim, local=True, unit_spacing=True, num_channels=1
):
    """Centroid (in global or local coordinates) from 1st order moment matrix"""
    moments_dtype = cp.dtype(moments_dtype)

    # assume moments input was truncated to only hold order<=2 moments
    num_moments = 2**ndim
    if moments_dtype.kind != "f":
        raise ValueError(
            "`moments_dtype` must be a floating point type for central moments "
            "calculations."
        )
    source = ""
    if not local:
        source += f"""
        unsigned int offset_coords = i * {2 * ndim};\n"""

    if num_channels > 1:
        source += f"""
        uint32_t num_channels = moments_raw.shape()[1];
        for (int c = 0; c < num_channels; c++) {{
            unsigned int offset = i * {num_moments} * num_channels + c * {num_moments};
            unsigned int offset_out = i * {ndim} * num_channels + c * {ndim};
            F m0 = moments_raw[offset];\n"""  # noqa: E501
    else:
        source += f"""
            unsigned int offset = i * {num_moments};
            unsigned int offset_out = i * {ndim};
            F m0 = moments_raw[offset];\n"""

    # general formula for the n-dimensional case
    #
    #   in 2D it gives:
    #     out[offset_out + 1] = moments_raw[offset + 1] / m0;  // m[0, 1]
    #     out[offset_out] = moments_raw[offset + 2] / m0;      // m[1, 0]
    #
    #   in 3D it gives:
    #     out[offset_out + 2] = moments_raw[offset + 1] / m0;  // m[0, 0, 1]
    #     out[offset_out + 1] = moments_raw[offset + 2] / m0;  // m[0, 1, 0]
    #     out[offset_out] = moments_raw[offset + 4] / m0;      // m[1, 0, 0]
    axis_offset = 1
    for d in range(ndim - 1, -1, -1):
        if local:
            source += f"""
            out[offset_out + {d}] = moments_raw[offset + {axis_offset}] / m0;"""  # noqa: E501
        else:
            spc = "" if unit_spacing else f" * spacing[{d}]"
            source += f"""
            out[offset_out + {d}] = moments_raw[offset + {axis_offset}] / m0 + bbox[offset_coords + {d * 2}]{spc};"""  # noqa: E501
        axis_offset *= 2
    if num_channels > 1:
        source += """
        }  // channels loop\n"""
    inputs = "raw F moments_raw"
    local_str = ""
    spacing_str = ""
    if not local:
        local_str = "_local"
        # bounding box coordinates
        inputs += ", raw Y bbox"
        if not unit_spacing:
            spacing_str = "_spacing"
            inputs += ", raw float64 spacing"
    outputs = "raw F out"
    name = f"cucim_centroid_weighted{local_str}{spacing_str}_{ndim}d"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_centroid_weighted(
    moments_raw, ndim, bbox=None, local=True, spacing=None
):
    max_label = moments_raw.shape[0]
    if moments_raw.ndim == ndim + 2:
        num_channels = moments_raw.shape[1]
    elif moments_raw.ndim == ndim + 1:
        num_channels = 1
    else:
        raise ValueError("moments_raw has unexpected shape")
    if not local and bbox is None:
        raise ValueError(
            "bbox coordinates must be provided to get the non-local centroid"
        )
    if moments_raw.dtype.kind != "f":
        raise ValueError("moments_raw must have a floating point dtype")
    order = moments_raw.shape[-1] - 1
    if order < 1:
        raise ValueError(
            f"inertia tensor calculation requires order>=1, found {order}"
        )
    if order >= 1:
        # truncate to only the 1st order moments
        slice_kept = (Ellipsis,) + (slice(0, 2),) * ndim
        moments_raw = cp.ascontiguousarray(moments_raw[slice_kept])

    # make a copy if the inputs are not already C-contiguous
    if not moments_raw.flags.c_contiguous:
        moments_raw = cp.ascontiguousarray(moments_raw)

    unit_spacing = spacing is None

    if local:
        inputs = (moments_raw,)
    else:
        if not bbox.flags.c_contiguous:
            bbox = cp.ascontiguousarray(bbox)
        inputs = (moments_raw, bbox)
        if not unit_spacing:
            inputs = inputs + (cp.asarray(spacing),)
    kernel = get_centroid_weighted_kernel(
        moments_raw.dtype,
        ndim,
        local=local,
        unit_spacing=unit_spacing,
        num_channels=num_channels,
    )
    centroid_shape = moments_raw.shape[:-ndim] + (ndim,)
    centroid = cp.zeros(centroid_shape, dtype=moments_raw.dtype)
    kernel(*inputs, centroid, size=max_label)
    return centroid
