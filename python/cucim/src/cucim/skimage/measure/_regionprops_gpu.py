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
    "regionprops_intensity_max",
    "regionprops_intensity_mean",
    "regionprops_intensity_min",
    "regionprops_intensity_std",
    "regionprops_moments",
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
def get_moments_kernel(coord_dtype, moments_dtype, ndim, order, spacing=None):
    coord_dtype = cp.dtype(coord_dtype)
    moments_dtype = cp.dtype(moments_dtype)
    uint_t = (
        "unsigned int" if coord_dtype.itemsize <= 4 else "unsigned long long"
    )

    if spacing is not None:
        if len(spacing) != ndim:
            raise ValueError("len(spacing) must equal len(shape)")
        if moments_dtype.kind != "f":
            raise ValueError("moments must have a floating point data type")

    source = """
          auto L = label[i];
          if (L != 0) {"""
    source += _unravel_loop_index("label", ndim, uint_t=uint_t)
    # using bounding box to transform the global coordinates to local ones
    # (c0 = local coordinate on axis 0, etc.)
    use_floating_point = moments_dtype.kind == "f"
    c_type = "double" if use_floating_point else f"{uint_t}"
    for d in range(ndim):
        source += f"""
                {c_type} c{d} = in_coord[{d}]
                                - bbox[(L - 1) * {2 * ndim} + {2*d}];"""
        if spacing:
            source += f"""
                c{d} *= {spacing[d]};"""
    if order > 3:
        raise ValueError("Only moments of orders 0-3 are supported")

    # number is for a densely populated moments matrix of size (order + 1) per
    # side (values at locations where order is greater than specified will be 0)
    num_moments = (order + 1) ** ndim
    source += f"""
                {uint_t} offset = (L - 1) * {num_moments};\n"""

    if use_floating_point:
        source += """
            atomicAdd(&moments[offset], 1.0);\n"""
    else:
        source += """
            atomicAdd(&moments[offset], 1);\n"""

    if ndim == 2:
        if order == 1:
            source += """
                atomicAdd(&moments[offset + 1], c1);
                atomicAdd(&moments[offset + 2], c0);\n"""
        elif order == 2:
            source += """
                atomicAdd(&moments[offset + 1], c1);
                atomicAdd(&moments[offset + 2], c1 * c1);
                atomicAdd(&moments[offset + 3], c0);
                atomicAdd(&moments[offset + 4], c0 * c1);
                atomicAdd(&moments[offset + 6], c0 * c0);\n"""
        elif order == 3:
            source += """
                atomicAdd(&moments[offset + 1], c1);
                atomicAdd(&moments[offset + 2], c1 * c1);
                atomicAdd(&moments[offset + 3], c1 * c1 * c1);
                atomicAdd(&moments[offset + 4], c0);
                atomicAdd(&moments[offset + 5], c0 * c1);
                atomicAdd(&moments[offset + 6], c0 * c1 * c1);
                atomicAdd(&moments[offset + 8], c0 * c0);
                atomicAdd(&moments[offset + 9], c0 * c0 * c1);
                atomicAdd(&moments[offset + 12], c0 * c0 * c0);\n"""
    elif ndim == 3:
        if order == 1:
            source += """
                atomicAdd(&moments[offset + 1], c2);
                atomicAdd(&moments[offset + 2], c1);
                atomicAdd(&moments[offset + 4], c0);\n"""
        elif order == 2:
            source += """
                atomicAdd(&moments[offset + 1], c2);
                atomicAdd(&moments[offset + 2], c2 * c2);
                atomicAdd(&moments[offset + 3], c1);
                atomicAdd(&moments[offset + 4], c1 * c2);
                atomicAdd(&moments[offset + 6], c1 * c1);
                atomicAdd(&moments[offset + 9], c0);
                atomicAdd(&moments[offset + 10], c0 * c2);
                atomicAdd(&moments[offset + 12], c0 * c1);
                atomicAdd(&moments[offset + 18], c0 * c0);\n"""
        elif order == 3:
            source += """
                atomicAdd(&moments[offset + 1], c2);
                atomicAdd(&moments[offset + 2], c2 * c2);
                atomicAdd(&moments[offset + 3], c2 * c2 * c2);
                atomicAdd(&moments[offset + 4], c1);
                atomicAdd(&moments[offset + 5], c1 * c2);
                atomicAdd(&moments[offset + 6], c1 * c2 * c2);
                atomicAdd(&moments[offset + 8], c1 * c1);
                atomicAdd(&moments[offset + 9], c1 * c1 * c2);
                atomicAdd(&moments[offset + 12], c1 * c1 * c1);
                atomicAdd(&moments[offset + 16], c0);
                atomicAdd(&moments[offset + 17], c0 * c2);
                atomicAdd(&moments[offset + 18], c0 * c2 * c2);
                atomicAdd(&moments[offset + 20], c0 * c1);
                atomicAdd(&moments[offset + 21], c0 * c1 * c2);
                atomicAdd(&moments[offset + 24], c0 * c1 * c1);
                atomicAdd(&moments[offset + 32], c0 * c0);
                atomicAdd(&moments[offset + 33], c0 * c0 * c2);
                atomicAdd(&moments[offset + 36], c0 * c0 * c1);
                atomicAdd(&moments[offset + 48], c0 * c0 * c0);\n"""
    else:
        raise ValueError("only 2d and 3d shapes are supported")
    source += """
          }\n"""
    inputs = f"raw X label, raw {coord_dtype.name} bbox"
    outputs = f"raw {moments_dtype.name} moments"
    name = f"cucim_moments_order{order}_{ndim}d_{coord_dtype.name}_"
    name += f"{moments_dtype.name}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_moments(
    label_image, max_label=None, order=2, spacing=None, coord_dtype=cp.uint32
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
    moments : cp.ndarray
        The moments up to the specified order. Will be stored in an
        ``(order + 1, ) * ndim`` matrix where any elements corresponding to
        order greater than that specified will be set to 0.
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

    bbox_coords_kernel(label_image, bbox_coords, size=label_image.size)

    # total number of elements in the moments matrix
    moments = cp.zeros((max_label,) + (order + 1,) * ndim, dtype=cp.float64)
    moments_kernel = get_moments_kernel(
        coord_dtype,
        moments.dtype,
        label_image.ndim,
        order=order,
        spacing=spacing,
    )
    moments_kernel(label_image, bbox_coords, moments, size=label_image.size)
    return moments
