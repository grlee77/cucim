import math
import warnings

import cupy as cp
import numpy as np
from packaging.version import parse

from cucim.skimage._vendored import ndimage as ndi, pad

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
    "regionprops_perimeter",
    "regionprops_perimeter_crofton",
]


def _get_bbox_code(uint_t, ndim, array_size):
    """
    Notes
    -----
    Local variables created:

        - bbox_min : shape (array_size, ndim)
            local minimum coordinates across the local set of labels encountered
        - bbox_max : shape (array_size, ndim)
            local maximum coordinates across the local set of labels encountered

    Output variables written to:

        - bbox : shape (max_label, 2 * ndim)
    """

    # declaration uses external variable:
    #    labels_size : total number of pixels in the label image
    source_pre = f"""
    // bounding box variables
    {uint_t} bbox_min[{ndim * array_size}];
    {uint_t} bbox_max[{ndim * array_size}] = {{0}};
    // initialize minimum coordinate to array size
    for (size_t ii = 0; ii < {ndim * array_size}; ii++) {{
      bbox_min[ii] = labels_size;
    }}\n"""

    # op uses external coordinate array variables:
    #    in_coord[0]...in_coord[ndim - 1] : coordinates
    #        coordinates in the labeled image at the current index
    source_operation = f"""
          bbox_min[{ndim}*offset] = min(in_coord[0], bbox_min[{ndim}*offset]);
          bbox_max[{ndim}*offset] = max(in_coord[0] + 1, bbox_max[{ndim}*offset]);"""  # noqa: E501
    for d in range(1, ndim):
        source_operation += f"""
          bbox_min[{ndim}*offset + {d}] = min(in_coord[{d}], bbox_min[{ndim}*offset + {d}]);
          bbox_max[{ndim}*offset + {d}] = max(in_coord[{d}] + 1, bbox_max[{ndim}*offset + {d}]);"""  # noqa: E501

    # post_operation uses external variables:
    #     ii : index into num_pixels array
    #     lab : label value that corresponds to location ii
    #     bbox : output with shape (max_label, 2 * ndim)
    source_post = f"""
          // bounding box outputs
          atomicMin(&bbox[(lab - 1)*{2 * ndim}], bbox_min[{ndim}*ii]);
          atomicMax(&bbox[(lab - 1)*{2 * ndim} + 1], bbox_max[{ndim}*ii]);"""
    for d in range(1, ndim):
        source_post += f"""
          atomicMin(&bbox[(lab - 1)*{2*ndim} + {2*d}], bbox_min[{ndim}*ii + {d}]);
          atomicMax(&bbox[(lab - 1)*{2*ndim} + {2*d + 1}], bbox_max[{ndim}*ii + {d}]);"""  # noqa: E501
    return source_pre, source_operation, source_post


def _get_num_pixels_code(pixels_per_thread, array_size):
    """
    Notes
    -----
    Local variables created:

        - num_pixels : shape (array_size, )
            The number of pixels encountered per label value

    Output variables written to:

        - counts : shape (max_label,)
    """
    pixel_count_dtype = "int8_t" if pixels_per_thread < 256 else "int16_t"

    source_pre = f"""
    // num_pixels variables
    {pixel_count_dtype} num_pixels[{array_size}] = {{0}};\n"""

    source_operation = """
        num_pixels[offset] += 1;\n"""

    # post_operation requires external variables:
    #     ii : index into num_pixels array
    #     lab : label value that corresponds to location ii
    #     counts : output with shape (max_label,)
    source_post = """
        atomicAdd(&counts[lab - 1], num_pixels[ii]);;"""
    return source_pre, source_operation, source_post


def _get_coord_sums_code(coord_sum_ctype, ndim, array_size):
    """
    Notes
    -----
    Local variables created:

        - bbox_min : shape (array_size, ndim)
            local minimum coordinates across the local set of labels encountered
        - bbox_max : shape (array_size, ndim)
            local maximum coordinates across the local set of labels encountered

    Output variables written to:

        - bbox : shape (max_label, 2 * ndim)
    """

    source_pre = f"""
    {coord_sum_ctype} coord_sum[{ndim * array_size}] = {{0}};\n"""

    # op uses external coordinate array variables:
    #    in_coord[0]...in_coord[ndim - 1] : coordinates
    #        coordinates in the labeled image at the current index
    source_operation = f"""
        coord_sum[{ndim}*offset] += in_coord[0];"""
    for d in range(1, ndim):
        source_operation += f"""
        coord_sum[{ndim}*offset + {d}] += in_coord[{d}];"""
    # post_operation uses external variables:
    #     ii : index into num_pixels array
    #     lab : label value that corresponds to location ii
    #     coord_sums : output with shape (max_label, ndim)
    source_post = f"""
        // bounding box outputs
        atomicAdd(&coord_sums[(lab - 1) * {ndim}], coord_sum[{ndim}*ii]);"""
    for d in range(1, ndim):
        source_post += f"""
        atomicAdd(&coord_sums[(lab - 1) * {ndim} + {d}],
                  coord_sum[{ndim}*ii + {d}]);"""
    return source_pre, source_operation, source_post


@cp.memoize(for_each_device=True)
def get_bbox_coords_kernel(
    ndim,
    int32_coords=True,
    int32_count=True,
    compute_bbox=True,
    compute_num_pixels=False,
    compute_coordinate_sums=False,
    pixels_per_thread=8,
):
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)
    if compute_num_pixels:
        count_dtype = cp.dtype(cp.uint32 if int32_count else cp.uint64)
    if compute_coordinate_sums:
        coord_sum_dtype = cp.dtype(cp.uint64)
        coord_sum_ctype = "uint64_t"

    if pixels_per_thread < 10:
        array_size = pixels_per_thread
    else:
        # highly unlikely for array to repeatedly swap labels at every pixel,
        # so use a smaller size
        array_size = pixels_per_thread // 2

    if coord_dtype.itemsize <= 4:
        uint_t = "unsigned int"
    else:
        uint_t = "unsigned long long"

    if not (compute_bbox or compute_num_pixels or compute_coordinate_sums):
        raise ValueError("no computation requested")

    if compute_bbox:
        bbox_pre, bbox_op, bbox_post = _get_bbox_code(
            uint_t=uint_t, ndim=ndim, array_size=array_size
        )
    if compute_num_pixels:
        count_pre, count_op, count_post = _get_num_pixels_code(
            pixels_per_thread=pixels_per_thread, array_size=array_size
        )
    if compute_coordinate_sums:
        coord_sums_pre, coord_sums_op, coord_sums_post = _get_coord_sums_code(
            coord_sum_ctype=coord_sum_ctype, ndim=ndim, array_size=array_size
        )
    # store only counts for label > 0  (label = 0 is the background)
    source = f"""
      uint64_t start_index = {pixels_per_thread}*i;
    """
    if compute_bbox:
        source += bbox_pre
    if compute_num_pixels:
        source += count_pre
    if compute_coordinate_sums:
        source += coord_sums_pre

    inner_op = ""
    if compute_bbox:
        source += _unravel_loop_index_declarations(
            "labels", ndim, uint_t=uint_t
        )

        inner_op += _unravel_loop_index(
            "labels",
            ndim=ndim,
            uint_t=uint_t,
            raveled_index="ii",
            omit_declarations=True,
        )
        inner_op += bbox_op
    if compute_num_pixels:
        inner_op += count_op
    if compute_coordinate_sums:
        inner_op += coord_sums_op

    source += f"""
      X encountered_labels[{array_size}] = {{0}};
      X current_label;
      X prev_label = labels[start_index];
      int offset = 0;
      encountered_labels[0] = prev_label;
      uint64_t ii_max = min(start_index + {pixels_per_thread}, labels_size);
      for (uint64_t ii = start_index; ii < ii_max; ii++) {{
        current_label = labels[ii];
        if (current_label == 0) {{ continue; }}
        if (current_label != prev_label) {{
            offset += 1;
            prev_label = current_label;
            encountered_labels[offset] = current_label;
        }}
        {inner_op}
      }}"""
    source += """
      for (size_t ii = 0; ii <= offset; ii++) {
        X lab = encountered_labels[ii];
        if (lab != 0) {"""

    if compute_bbox:
        source += bbox_post
    if compute_num_pixels:
        source += count_post
    if compute_coordinate_sums:
        source += coord_sums_post
    source += """
        }
      }\n"""

    # print(source)
    inputs = "raw X labels, raw uint64 labels_size"
    outputs = []
    name = "cucim_"
    if compute_bbox:
        outputs.append(f"raw {coord_dtype.name} bbox")
        name += f"_bbox{ndim}d"
    if compute_num_pixels:
        outputs.append(f"raw {count_dtype.name} counts")
        name += f"_numpix_dtype{count_dtype.char}"
    if compute_coordinate_sums:
        outputs.append(f"raw {coord_sum_dtype.name} coord_sums")
        name += f"_csums_dtype{coord_sum_dtype.char}"
    outputs = ", ".join(outputs)
    name += f"_batch{pixels_per_thread}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def _get_img_sums_code(
    c_sum_type,
    pixels_per_thread,
    array_size,
    num_channels=1,
    compute_num_pixels=True,
    compute_sum=True,
    compute_sum_sq=False,
):
    """
    Notes
    -----
    Local variables created:

        - num_pixels : shape (array_size, )
            The number of pixels encountered per label value

    Output variables written to:

        - counts : shape (max_label,)
    """
    pixel_count_dtype = "int8_t" if pixels_per_thread < 256 else "int16_t"

    source_pre = ""
    if compute_num_pixels:
        source_pre += f"""
    {pixel_count_dtype} num_pixels[{array_size}] = {{0}};"""
    if compute_sum:
        source_pre += f"""
    {c_sum_type} img_sums[{array_size * num_channels}] = {{0}};"""
    if compute_sum_sq:
        source_pre += f"""
    {c_sum_type} img_sum_sqs[{array_size * num_channels}] = {{0}};"""
    if compute_sum or compute_sum_sq:
        source_pre += f"""
    {c_sum_type} v = 0;\n"""

    # source_operation requires external variables:
    #     ii : index into labels array
    source_operation = ""
    if compute_num_pixels:
        source_operation += """
            num_pixels[offset] += 1;"""
    nc = f"{num_channels}*" if num_channels > 1 else ""
    if compute_sum or compute_sum_sq:
        for c in range(num_channels):
            source_operation += f"""
            v = static_cast<{c_sum_type}>(img[{nc}ii + {c}]);"""
            if compute_sum:
                source_operation += f"""
            img_sums[{nc}offset + {c}] += v;"""
            if compute_sum_sq:
                source_operation += f"""
            img_sum_sqs[{nc}offset + {c}] += v * v;\n"""

    # post_operation requires external variables:
    #     jj : index into num_pixels array
    #     lab : label value that corresponds to location ii
    #     counts : output with shape (max_label,)
    #     sums : output with shape (max_label, nc)
    #     sumsqs : output with shape (max_label, nc)
    source_post = ""
    if compute_num_pixels:
        source_post += """
            atomicAdd(&counts[lab - 1], num_pixels[jj]);"""
    if compute_sum:
        for c in range(num_channels):
            source_post += f"""
            atomicAdd(&sums[{nc}(lab - 1) + {c}], img_sums[{nc}jj + {c}]);"""
    if compute_sum_sq:
        for c in range(num_channels):
            source_post += f"""
            atomicAdd(&sumsqs[{nc}(lab - 1) + {c}], img_sum_sqs[{nc}jj + {c}]);"""  # noqa: E501
    return source_pre, source_operation, source_post


def _get_intensity_min_max_code(
    min_max_dtype,
    c_min_max_type,
    array_size,
    initial_min_val,
    initial_max_val,
    compute_min=True,
    compute_max=True,
    num_channels=1,
):
    min_max_dtype = cp.dtype(min_max_dtype)
    c_type = c_min_max_type

    # Note: CuPy provides atomicMin and atomicMax for float and double in
    #       cupy/_core/include/atomics.cuh
    #       The integer variants are part of CUDA itself.

    source_pre = ""
    if compute_min:
        source_pre += f"""
    {c_type} min_vals[{array_size * num_channels}];
    // initialize minimum coordinate to array size
    for (size_t ii = 0; ii < {array_size * num_channels}; ii++) {{
      min_vals[ii] = {initial_min_val};
    }}"""
    if compute_max:
        source_pre += f"""
    {c_type} max_vals[{array_size * num_channels}];
    // initialize minimum coordinate to array size
    for (size_t ii = 0; ii < {array_size * num_channels}; ii++) {{
      max_vals[ii] = {initial_max_val};
    }}"""
    source_pre += f"""
    {c_type} v = 0;\n"""

    # source_operation requires external variables:
    #     ii : index into labels array
    source_operation = ""
    nc = f"{num_channels}*" if num_channels > 1 else ""
    if compute_min or compute_max:
        for c in range(num_channels):
            source_operation += f"""
            v = static_cast<{c_type}>(img[{nc}ii + {c}]);"""
            if compute_min:
                source_operation += f"""
            min_vals[{nc}offset + {c}] = min(v, min_vals[{nc}offset + {c}]);"""
            if compute_max:
                source_operation += f"""
            max_vals[{nc}offset + {c}] = max(v, max_vals[{nc}offset + {c}]);\n"""  # noqa: E501

    # post_operation requires external variables:
    #     jj : index into num_pixels array
    #     lab : label value that corresponds to location ii
    #     counts : output with shape (max_label,)
    #     sums : output with shape (max_label, nc)
    #     sumsqs : output with shape (max_label, nc)
    source_post = ""
    if compute_min:
        for c in range(num_channels):
            source_post += f"""
            atomicMin(&minimums[{nc}(lab - 1) + {c}], min_vals[{nc}jj + {c}]);"""  # noqa: E501
    if compute_max:
        for c in range(num_channels):
            source_post += f"""
            atomicMax(&maximums[{nc}(lab - 1) + {c}], max_vals[{nc}jj + {c}]);"""  # noqa: E501
    return source_pre, source_operation, source_post


@cp.memoize()
def _get_intensity_img_kernel_dtypes(image_dtype):
    """Determine CuPy dtype and C++ type for image sum operations."""
    image_dtype = cp.dtype(image_dtype)
    if image_dtype.kind == "f":
        # use double for accuracy of mean/std computations
        c_sum_type = "double"
        dtype = cp.float64
        # atomicMin, atomicMax support 32 and 64-bit float
        if image_dtype.itemsize > 4:
            min_max_dtype = cp.float64
            c_min_max_type = "double"
        else:
            min_max_dtype = cp.float32
            c_min_max_type = "float"
    elif image_dtype.kind in "bu":
        c_sum_type = "uint64_t"
        dtype = cp.uint64
        if image_dtype.itemsize > 4:
            min_max_dtype = cp.uint64
            c_min_max_type = "uint64_t"
        else:
            min_max_dtype = cp.uint32
            c_min_max_type = "uint32_t"
    elif image_dtype.kind in "i":
        c_sum_type = "int64_t"
        dtype = cp.int64
        if image_dtype.itemsize > 4:
            min_max_dtype = cp.int64
            c_min_max_type = "int64_t"
        else:
            min_max_dtype = cp.int32
            c_min_max_type = "int32_t"
    else:
        raise ValueError(
            f"Invalid intensity image dtype {image_dtype.name}. "
            "Must be an unsigned, integer or floating point type."
        )
    return cp.dtype(dtype), c_sum_type, cp.dtype(min_max_dtype), c_min_max_type


@cp.memoize()
def _get_intensity_range(image_dtype):
    """Determine CuPy dtype and C++ type for image sum operations."""
    image_dtype = cp.dtype(image_dtype)
    if image_dtype.kind == "f":
        # use double for accuracy of mean/std computations
        info = cp.finfo(image_dtype)
    elif image_dtype.kind in "bui":
        info = cp.iinfo(image_dtype)
    else:
        raise ValueError(
            f"Invalid intensity image dtype {image_dtype.name}. "
            "Must be an unsigned, integer or floating point type."
        )
    return (info.min, info.max)


@cp.memoize(for_each_device=True)
def get_intensity_measure_kernel(
    image_dtype=None,
    int32_coords=True,
    int32_count=True,
    num_channels=1,
    compute_num_pixels=True,
    compute_sum=True,
    compute_sum_sq=False,
    compute_min=False,
    compute_max=False,
    pixels_per_thread=8,
):
    if compute_num_pixels:
        count_dtype = cp.dtype(cp.uint32 if int32_count else cp.uint64)

    (
        sum_dtype,
        c_sum_type,
        min_max_dtype,
        c_min_max_type,
    ) = _get_intensity_img_kernel_dtypes(image_dtype)

    if pixels_per_thread < 10:
        array_size = pixels_per_thread
    else:
        # highly unlikely for array to repeatedly swap labels at every pixel,
        # so use a smaller size
        array_size = pixels_per_thread // 2

    any_sums = compute_num_pixels or compute_sum or compute_sum_sq

    if any_sums:
        sums_pre, sums_op, sums_post = _get_img_sums_code(
            c_sum_type=c_sum_type,
            pixels_per_thread=pixels_per_thread,
            array_size=array_size,
            num_channels=num_channels,
            compute_num_pixels=compute_num_pixels,
            compute_sum=compute_sum,
            compute_sum_sq=compute_sum_sq,
        )

    any_min_max = compute_min or compute_max
    if any_min_max:
        if min_max_dtype is None:
            raise ValueError("min_max_dtype must be specified")
        range_min, range_max = _get_intensity_range(min_max_dtype)
        min_max_pre, min_max_op, min_max_post = _get_intensity_min_max_code(
            min_max_dtype=min_max_dtype,
            c_min_max_type=c_min_max_type,
            array_size=array_size,
            num_channels=num_channels,
            initial_max_val=range_min,
            initial_min_val=range_max,
            compute_min=compute_min,
            compute_max=compute_max,
        )

    if not (any_min_max or any_sums):
        raise ValueError("no output values requested")

    # store only counts for label > 0  (label = 0 is the background)
    source = f"""
      uint64_t start_index = {pixels_per_thread}*i;
    """

    if any_sums:
        source += sums_pre
    if any_min_max:
        source += min_max_pre

    inner_op = ""
    if any_sums:
        inner_op += sums_op
    if any_min_max:
        inner_op += min_max_op

    source += f"""
      X encountered_labels[{array_size}] = {{0}};
      X current_label;
      X prev_label = labels[start_index];
      int offset = 0;
      encountered_labels[0] = prev_label;
      uint64_t ii_max = min(start_index + {pixels_per_thread}, labels_size);
      for (uint64_t ii = start_index; ii < ii_max; ii++) {{
        current_label = labels[ii];
        if (current_label == 0) {{ continue; }}
        if (current_label != prev_label) {{
            offset += 1;
            prev_label = current_label;
            encountered_labels[offset] = current_label;
        }}
        {inner_op}
      }}"""
    source += """
      for (size_t jj = 0; jj <= offset; jj++) {
        X lab = encountered_labels[jj];
        if (lab != 0) {"""

    if any_sums:
        source += sums_post
    if any_min_max:
        source += min_max_post
    source += """
        }
      }\n"""

    # print(source)
    inputs = "raw X labels, raw uint64 labels_size, raw Y img"
    outputs = []
    name = "cucim_"
    if compute_num_pixels:
        outputs.append(f"raw {count_dtype.name} counts")
        name += f"_numpix_{count_dtype.char}"
    if compute_sum:
        outputs.append(f"raw {sum_dtype.name} sums")
        name += "_sum"
    if compute_sum_sq:
        outputs.append(f"raw {sum_dtype.name} sumsqs")
        name += "_sumsq"
    if compute_sum or compute_sum_sq:
        name += f"_{sum_dtype.char}"
    if compute_min:
        outputs.append(f"raw {min_max_dtype.name} minimums")
        name += "_min"
    if compute_max:
        outputs.append(f"raw {min_max_dtype.name} maximums")
        name += "_max"
    if compute_min or compute_max:
        name += f"{min_max_dtype.char}"
    outputs = ", ".join(outputs)
    name += f"_batch{pixels_per_thread}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def _get_count_dtype(label_image_size):
    """atomicAdd only supports int32, uint32, int64, uint64, float32, float64"""
    int32_count = label_image_size < 2**32
    count_dtype = cp.dtype(cp.uint32 if int32_count else cp.uint64)
    return count_dtype, int32_count


def regionprops_num_pixels(label_image, max_label=None, pixels_per_thread=16):
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    count_dtype, int32_count = _get_count_dtype(label_image.size)

    pixels_kernel = get_bbox_coords_kernel(
        int32_count=int32_count,
        ndim=label_image.ndim,
        compute_bbox=False,
        compute_num_pixels=True,
        compute_coordinate_sums=False,
        pixels_per_thread=pixels_per_thread,
    )
    counts = cp.zeros(num_counts, dtype=count_dtype)

    # make a copy if the labels array is not C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    pixels_kernel(
        label_image,
        label_image.size,
        counts,
        size=math.ceil(label_image.size / pixels_per_thread),
    )
    return counts


def regionprops_area(
    label_image,
    spacing=None,
    max_label=None,
    dtype=cp.float32,
    pixels_per_thread=16,
):
    # integer atomicAdd is faster than floating point so better to convert
    # after counting
    area = regionprops_num_pixels(
        label_image,
        max_label=max_label,
        pixels_per_thread=pixels_per_thread,
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
    mean_dtype=cp.float32,
    pixels_per_thread=16,
):
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_shapes(label_image, intensity_image)

    count_dtype, int32_count = _get_count_dtype(label_image.size)

    image_dtype = intensity_image.dtype
    sum_dtype, _, _, _ = _get_intensity_img_kernel_dtypes(image_dtype)

    counts = cp.zeros(num_counts, dtype=count_dtype)
    sum_shape = (
        (num_counts,) if num_channels == 1 else (num_counts, num_channels)
    )
    sums = cp.zeros(sum_shape, dtype=sum_dtype)

    kernel = get_intensity_measure_kernel(
        int32_count=int32_count,
        image_dtype=image_dtype,
        num_channels=num_channels,
        compute_num_pixels=True,
        compute_sum=True,
        compute_sum_sq=False,
        pixels_per_thread=pixels_per_thread,
    )

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    if not intensity_image.flags.c_contiguous:
        intensity_image = cp.ascontiguousarray(intensity_image)

    kernel(
        label_image,
        label_image.size,
        intensity_image,
        counts,
        sums,
        size=math.ceil(label_image.size / pixels_per_thread),
    )

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
    std_dtype=cp.float64,
    pixels_per_thread=4,
):
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_shapes(label_image, intensity_image)

    image_dtype = intensity_image.dtype
    sum_dtype, _, _, _ = _get_intensity_img_kernel_dtypes(image_dtype)

    count_dtype, int32_count = _get_count_dtype(label_image.size)
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
        kernel = get_intensity_measure_kernel(
            int32_count=int32_count,
            image_dtype=image_dtype,
            num_channels=num_channels,
            compute_num_pixels=True,
            compute_sum=True,
            compute_sum_sq=True,
            pixels_per_thread=pixels_per_thread,
        )
        kernel(
            label_image,
            label_image.size,
            intensity_image,
            counts,
            sums,
            sumsqs,
            size=math.ceil(label_image.size / pixels_per_thread),
        )

        if cp.dtype(std_dtype).kind != "f":
            raise ValueError("mean_dtype must be a floating point type")

        # compute means and standard deviations from the counts, sums and
        # squared sums (use float64 here since the numerical stability of this
        # approach is poor)
        means = cp.zeros(sum_shape, dtype=cp.float64)
        stds = cp.zeros(sum_shape, dtype=cp.float64)
        kernel2 = get_mean_var_kernel(stds.dtype, sample_std=sample_std)
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


def _regionprops_min_or_max_intensity(
    label_image,
    intensity_image,
    max_label=None,
    compute_min=True,
    compute_max=False,
    pixels_per_thread=8,
):
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_shapes(label_image, intensity_image)

    # use an appropriate data type supported by atomicMin and atomicMax
    image_dtype = intensity_image.dtype
    _, _, min_max_dtype, _ = _get_intensity_img_kernel_dtypes(image_dtype)
    range_min, range_max = _get_intensity_range(image_dtype)
    out_shape = (
        (num_counts,) if num_channels == 1 else (num_counts, num_channels)
    )
    if compute_min:
        minimums = cp.full(out_shape, range_max, dtype=min_max_dtype)
    if compute_max:
        maximums = cp.full(out_shape, range_min, dtype=min_max_dtype)

    kernel = get_intensity_measure_kernel(
        image_dtype=image_dtype,
        num_channels=num_channels,
        compute_num_pixels=False,
        compute_sum=False,
        compute_sum_sq=False,
        compute_min=compute_min,
        compute_max=compute_max,
        pixels_per_thread=pixels_per_thread,
    )

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    if not intensity_image.flags.c_contiguous:
        intensity_image = cp.ascontiguousarray(intensity_image)

    lab_size = label_image.size
    sz = math.ceil(label_image.size / pixels_per_thread)
    if compute_min and compute_max:
        kernel(
            label_image, lab_size, intensity_image, minimums, maximums, size=sz
        )  # noqa: E501
        return minimums, maximums
    elif compute_min:
        kernel(label_image, lab_size, intensity_image, minimums, size=sz)
        return minimums
    elif compute_max:
        kernel(label_image, lab_size, intensity_image, maximums, size=sz)
        return maximums


def regionprops_intensity_min(
    label_image, intensity_image, max_label=None, pixels_per_thread=8
):
    return _regionprops_min_or_max_intensity(
        label_image,
        intensity_image,
        max_label=max_label,
        compute_min=True,
        compute_max=False,
        pixels_per_thread=pixels_per_thread,
    )


def regionprops_intensity_max(
    label_image, intensity_image, max_label=None, pixels_per_thread=8
):
    return _regionprops_min_or_max_intensity(
        label_image,
        intensity_image,
        max_label=max_label,
        compute_min=False,
        compute_max=True,
        pixels_per_thread=pixels_per_thread,
    )


def _unravel_loop_index_declarations(var_name, ndim, uint_t="unsigned int"):
    code = f"""
        // variables for unraveling a linear index to a coordinate array
        {uint_t} in_coord[{ndim}];
        {uint_t} temp_floor;"""
    for d in range(ndim):
        code += f"""
        {uint_t} dim{d}_size = {var_name}.shape()[{d}];"""
    return code


def _unravel_loop_index(
    var_name,
    ndim,
    uint_t="unsigned int",
    raveled_index="i",
    omit_declarations=False,
):
    """
    declare a multi-index array in_coord and unravel the 1D index, i into it.
    This code assumes that the array is a C-ordered array.
    """
    code = (
        ""
        if omit_declarations
        else _unravel_loop_index_declarations(var_name, ndim, uint_t)
    )
    code += f"{uint_t} temp_idx = {raveled_index};"
    for d in range(ndim - 1, 0, -1):
        code += f"""
        temp_floor = temp_idx / dim{d}_size;
        in_coord[{d}] = temp_idx - temp_floor * dim{d}_size;
        temp_idx = temp_floor;"""
    code += """
        in_coord[0] = temp_idx;"""
    return code


def regionprops_bbox_coords(
    label_image,
    max_label=None,
    return_slices=False,
    pixels_per_thread=16,
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

    int32_coords = max(label_image.shape) < 2**32
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)

    bbox_kernel = get_bbox_coords_kernel(
        ndim=label_image.ndim,
        int32_coords=int32_coords,
        pixels_per_thread=pixels_per_thread,
    )

    ndim = label_image.ndim
    bbox_coords = cp.zeros((max_label, 2 * ndim), dtype=coord_dtype)

    # Initialize value for atomicMin on even coordinates
    # The value for atomicMax columns is already 0 as desired.
    bbox_coords[:, ::2] = cp.iinfo(coord_dtype).max

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    bbox_kernel(
        label_image,
        label_image.size,
        bbox_coords,
        size=math.ceil(label_image.size / pixels_per_thread),
    )
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


def regionprops_centroid(
    label_image, max_label=None, coord_dtype=cp.uint32, pixels_per_thread=16
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
    centroid : np.ndarray
        The centroid of each region.
    """
    if max_label is None:
        max_label = int(label_image.max())

    int32_coords = max(label_image.shape) < 2**32
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)

    bbox_coords_kernel = get_bbox_coords_kernel(
        ndim=label_image.ndim,
        int32_coords=int32_coords,
        compute_num_pixels=True,
        compute_coordinate_sums=True,
        pixels_per_thread=pixels_per_thread,
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
        label_image.size,
        bbox_coords,
        centroid_counts,
        centroid_sums,
        size=math.ceil(label_image.size / pixels_per_thread),
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
    label_image,
    max_label=None,
    coord_dtype=cp.uint32,
    pixels_per_thread=16,
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

    int32_coords = max(label_image.shape) < 2**32
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)

    bbox_coords_kernel = get_bbox_coords_kernel(
        ndim=label_image.ndim,
        int32_coords=int32_coords,
        compute_bbox=True,
        compute_num_pixels=False,
        compute_coordinate_sums=False,
        pixels_per_thread=pixels_per_thread,
    )

    ndim = label_image.ndim
    bbox_coords = cp.zeros((max_label, 2 * ndim), dtype=coord_dtype)

    # Initialize value for atomicMin on even coordinates
    # The value for atomicMax columns is already 0 as desired.
    bbox_coords[:, ::2] = cp.iinfo(coord_dtype).max

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    bbox_coords_kernel(
        label_image,
        label_image.size,
        bbox_coords,
        size=math.ceil(label_image.size / pixels_per_thread),
    )

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
    pixels_per_thread=16,
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

    int32_coords = max(label_image.shape) < 2**32
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)

    bbox_kernel = get_bbox_coords_kernel(
        ndim=label_image.ndim,
        int32_coords=int32_coords,
        compute_bbox=True,
        compute_num_pixels=False,
        compute_coordinate_sums=False,
        pixels_per_thread=pixels_per_thread,
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

    bbox_kernel(
        label_image,
        label_image.size,
        bbox_coords,
        size=math.ceil(label_image.size / pixels_per_thread),
    )

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
def get_moments_normalize_kernel(
    moments_dtype, ndim, order, unit_scale=False, pixel_correction=False
):
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

    # pixel correction for out[2, 0], out[0, 2] in 2D
    #                      out[2, 0, 0], out[0, 2, 0] and out[0, 0, 2] in 3D
    # See ITK paper:
    # Padfield2008 - A Label Geometry Image Filter for Multiple Object
    # Measurement
    if pixel_correction:
        correction_term = " + 0.08333333333333333"  # add 1/12
    else:
        correction_term = ""

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
            source += f"""
            // normalize the 2nd order central moments
            out[offset + 2] = moments_central[offset + 2] / norm_order2{correction_term};  // out[0, 2]
            out[offset + 4] = moments_central[offset + 4] / norm_order2;  // out[1, 1]
            out[offset + 6] = moments_central[offset + 6] / norm_order2{correction_term};  // out[2, 0]\n"""  # noqa: E501
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
            source += f"""
            // normalize the 2nd order central moments
            out[offset + 2] = moments_central[offset + 2] / norm_order2{correction_term};    // out[0, 0, 2]
            out[offset + 4] = moments_central[offset + 4] / norm_order2;    // out[0, 1, 1]
            out[offset + 6] = moments_central[offset + 6] / norm_order2{correction_term};    // out[0, 2, 0]
            out[offset + 10] = moments_central[offset + 10] / norm_order2;  // out[1, 0, 1]
            out[offset + 12] = moments_central[offset + 12] / norm_order2;  // out[1, 1, 0]
            out[offset + 18] = moments_central[offset + 18] / norm_order2{correction_term};  // out[2, 0, 0]\n"""  # noqa: E501
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


def regionprops_moments_normalized(
    moments_central, ndim, spacing=None, pixel_correction=False
):
    """

    Notes
    -----
    Default setting of `pixel_correction=False` matches the scikit-image
    behavior (as of v0.25).

    The `pixel_correction` is to account for pixel/voxel size and is only
    implemented for 2nd order moments currently based on the derivation in:

    The correction should need to be updated to take 'spacing' into account as
    it currently assumes unit size.

    Padfield D., Miller J. "A Label Geometry Image Filter for Multiple Object
    Measurement". The Insight Journal. 2013 Mar.
    https://doi.org/10.54294/saa3nn
    """
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
        moments_central.dtype,
        ndim,
        order,
        unit_scale=unit_scale,
        pixel_correction=pixel_correction,
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
def get_spd_matrix_eigvals_kernel(
    rank, compute_eigenvectors=False, compute_axis_lengths=False
):
    """Compute symmetric positive definite (SPD) matrix eigenvalues

    Implements closed-form analytical solutions for 2x2 and 3x3 matrices.

    C. Deledalle, L. Denis, S. Tabti, F. Tupin. Closed-form expressions
    of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices.

    [Research Report] Université de Lyon. 2017.
    https://hal.archives-ouvertes.fr/hal-01501221/file/matrix_exp_and_log_formula.pdf
    """  # noqa: E501

    # assume moments input was truncated to only hold order<=2 moments
    num_elements = rank * rank

    # size of the inertia_tensor matrix
    source = f"""
            unsigned int offset = i * {num_elements};
            unsigned int offset_out = i * {rank};\n"""
    if rank == 2:
        source += """
            F tmp1, tmp2;
            double m00 = static_cast<double>(spd_matrix[offset]);
            double m01 = static_cast<double>(spd_matrix[offset + 1]);
            double m11 = static_cast<double>(spd_matrix[offset + 3]);
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
            // (matrix is spd, so negatives values can only be due to
            //  numerical errors)
            F lam1 = max(tmp1 + tmp2, 0.0);
            F lam2 = max(tmp1 - tmp2, 0.0);
            out[offset_out] = lam1;
            out[offset_out + 1] = lam2;\n"""
        if compute_eigenvectors:
            source += """
            double ev_denom = max(m01, 1.0e-15);
            // first eigenvector
            evecs[offset] = (lam2 - m11) / ev_denom;
            evecs[offset + 2] = 1.0;
            // second eigenvector
            evecs[offset + 1] = (lam1 - m11) / ev_denom;
            evecs[offset + 3] = 1.0;
            \n"""
        if compute_axis_lengths:
            source += """
            axis_lengths[offset_out] = 4.0 * sqrt(lam1);
            axis_lengths[offset_out + 1] = 4.0 * sqrt(lam2);
            \n"""
    elif rank == 3:
        source += """
            double x1, x2, phi;
            // extract triangle of (spd) inertia tensor values
            // [a, d, f]
            // [-, b, e]
            // [-, -, c]
            double a = static_cast<double>(spd_matrix[offset]);
            double b = static_cast<double>(spd_matrix[offset + 4]);
            double c = static_cast<double>(spd_matrix[offset + 8]);
            double d = static_cast<double>(spd_matrix[offset + 1]);
            double e = static_cast<double>(spd_matrix[offset + 5]);
            double f = static_cast<double>(spd_matrix[offset + 2]);
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
            // (matrix is spd, so negatives values can only be due to
            //  numerical errors)
            lam1 = max(lam1, 0.0);
            lam2 = max(lam2, 0.0);
            lam3 = max(lam3, 0.0);
            out[offset_out] = lam1;
            out[offset_out + 1] = lam2;
            out[offset_out + 2] = lam3;\n"""
        if compute_eigenvectors:
            source += """
            double f_denom = f;
            if (f_denom == 0) {
                f = 1.0e-15;
            }
            double de = d * e;
            double ef = e * f;

            // first eigenvector
            double m_denom = f * (b - lam1) - de;
            if (m_denom == 0) {
                m_denom = 1.0e-15;
            }
            double m = (d * (c - lam1) - ef) / m_denom;
            evecs[offset] = (lam1 - c - e * m) / f_denom;
            evecs[offset + 3] = m;
            evecs[offset + 6] = 1.0;

            // second eigenvector
            m_denom = f * (b - lam2) - de;
            if (m_denom == 0) {
                m_denom = 1.0e-15;
            }
            m = (d * (c - lam2) - ef) / m_denom;
            evecs[offset + 1] = (lam2 - c - e * m) / f_denom;
            evecs[offset + 4] = m;
            evecs[offset + 7] = 1.0;

            // third eigenvector
            m_denom = f * (b - lam3) - de;
            if (m_denom == 0) {
                m_denom = 1.0e-15;
            }
            m = (d * (c - lam3) - ef) / m_denom;
            evecs[offset + 2] = (lam3 - c - e * m) / f_denom;
            evecs[offset + 5] = m;
            evecs[offset + 8] = 1.0;
            \n"""
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
        raise ValueError("only rank = 2 or 3 is supported")
    inputs = "raw F spd_matrix"
    outputs = "raw F out"
    if compute_eigenvectors:
        outputs += ", raw F eigvecs"
        ev_str = "eigvecs_"
    else:
        ev_str = ""
    name = f"cucim_spd_matrix_eigvals_{ev_str}{rank}d"
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

    kernel = get_spd_matrix_eigvals_kernel(
        rank=ndim,
        compute_axis_lengths=compute_axis_lengths,
        compute_eigenvectors=False,
    )
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


def regionprops_perimeter(labels, neighborhood=4, max_label=None, robust=False):
    """Calculate total perimeter of all objects in binary image.

    Parameters
    ----------
    labels : (M, N) ndarray
        Binary input image.
    neighborhood : 4 or 8, optional
        Neighborhood connectivity for border pixel determination. It is used to
        compute the contour. A higher neighborhood widens the border on which
        the perimeter is computed.
    max_label : int or None, optional
        The maximum label in labels can be provided to avoid recomputing it if
        it was already known.
    robust : bool, optional
        If True, extra computation will be done to detect if any labeled
        regions are <=2 pixel spacing from another. Any regions that meet that
        criteria will have their perimeter recomputed in isolation to avoid
        possible error that would otherwise occur in this case. Turning this
        on will make the run time substantially longer, so it should only be
        used when labeled regions may have a non-negligible portion of their
        boundary within a <2 pixel gap from another label.

    Returns
    -------
    perimeter : float
        Total perimeter of all objects in binary image.

    Notes
    -----
    The `perimeter` method does not consider the boundary along the image edge
    as image as part of the perimeter, while the `perimeter_crofton` method
    does. In any case, an object touching the image edge likely extends outside
    of the field of view, so an accurate perimeter cannot be measured for such
    objects.

    References
    ----------
    .. [1] K. Benkrid, D. Crookes. Design and FPGA Implementation of
           a Perimeter Estimator. The Queen's University of Belfast.
           http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc

    See Also
    --------
    perimeter_crofton

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage import util
    >>> from cucim.skimage.measure import label
    >>> # coins image (binary)
    >>> img_coins = cp.array(data.coins() > 110)
    >>> # total perimeter of all objects in the image
    >>> perimeter(img_coins, neighborhood=4)  # doctest: +ELLIPSIS
    array(7796.86799644)
    >>> perimeter(img_coins, neighborhood=8)  # doctest: +ELLIPSIS
    array(8806.26807333)
    """
    if max_label is None:
        max_label = int(labels.max())

    # maximum possible value for XF_labeled input to bincount
    # need to choose integer range large enough that this won't overflow
    max_val = (max_label + 1) * 16
    if max_val < 256:
        image_dtype = cp.uint8
    elif max_val < 65536:
        image_dtype = cp.uint16
    elif max_val < 2**32:
        image_dtype = cp.uint32
    else:
        image_dtype = cp.uint64

    if image_dtype == cp.uint8:
        # can directly view bool as uint8 without a copy
        image = (labels > 0).view(cp.uint8)
    else:
        image = (labels > 0).astype(image_dtype)

    if neighborhood == 4:
        footprint = cp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    else:
        footprint = 3

    eroded_image = ndi.binary_erosion(image, footprint, border_value=0)
    border_image = image - eroded_image

    perimeter_weights = np.zeros(50, dtype=cp.float64)
    perimeter_weights[[5, 7, 15, 17, 25, 27]] = 1
    perimeter_weights[[21, 33]] = math.sqrt(2)
    perimeter_weights[[13, 23]] = (1 + math.sqrt(2)) / 2
    perimeter_weights = cp.asarray(perimeter_weights)

    perimeter_image = ndi.convolve(
        border_image,
        cp.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]]),
        mode="constant",
        cval=0,
    )

    # dilate labels by 1 pixel so we can sum with values in XF to give
    # unique histogram bins for each labeled regions (as long as no labeled
    # regions are within < 2 pixels from another labeled region)
    labels_dilated = ndi.grey_dilation(labels, 3, mode="constant")

    if robust:
        # check possibly too-close regions for which we may need to manually
        # recompute the regions perimeter in isolation
        labels_dilated2 = ndi.grey_dilation(labels_dilated, 5, mode="constant")
        labels2 = labels_dilated2 * image
        diffs = labels != labels2

        labels_to_recompute = []
        # regions to recompute
        if cp.any(diffs):
            labels_to_recompute = cp.asnumpy(cp.unique(labels[diffs]))
            bbox, slices = regionprops_bbox_coords(labels, return_slices=True)
            warnings.warn(
                "some labeled regions are <= 2 pixels from another region. The "
                "perimeter may be underestimated for one of the labels when "
                "two labels are < 2 pixels from touching"
            )

    # values in XF are guaranteed to be in range [0, 15] so need to multiply
    # each label by 16 to make sure all labels have a unique set of values
    # during bincount
    perimeter_image = perimeter_image + 50 * labels_dilated

    minlength = 50 * (max_label + 1)
    h = cp.bincount(perimeter_image.ravel(), minlength=minlength)
    # values for label=1 start at index 50
    h = h[50:minlength].reshape((max_label, 50))

    perimeters = perimeter_weights @ h.T
    if robust:
        # recompute perimeter in isolation for each region that may be too
        # close to another one
        shape = image.shape
        for lab in labels_to_recompute:
            sl = slices[lab - 1]

            # keep boundary of 1 so object is not at 'edge' of cropped
            # region (unless it is at a true image edge)
            ld = labels_dilated[
                max(sl[0].start - 1, 0) : min(sl[0].stop + 1, shape[0]),
                max(sl[0].start - 1, 0) : min(sl[1].stop + 1, shape[1]),
            ]
            p = regionprops_perimeter(ld == lab, neighborhood=neighborhood)
            # print(f"label {lab}: old perimeter={perimeters[lab - 1]}, new_perimeter={p}")  # noqa: E501
            perimeters[lab - 1] = p[0]
    return perimeters


def regionprops_perimeter_crofton(
    labels, directions=4, max_label=None, robust=False, omit_image_edges=False
):
    """Calculate total Crofton perimeter of all objects in binary image.

    Parameters
    ----------
    labels : (M, N) ndarray
        Input image. If image is not binary, all values greater than zero
        are considered as the object.
    directions : 2 or 4, optional
        Number of directions used to approximate the Crofton perimeter. By
        default, 4 is used: it should be more accurate than 2.
        Computation time is the same in both cases.
    max_label : int or None, optional
        The maximum label in labels can be provided to avoid recomputing it if
        it was already known.
    robust : bool, optional
        If True, extra computation will be done to detect if any labeled
        regions are <=2 pixel spacing from another. Any regions that meet that
        criteria will have their perimeter recomputed in isolation to avoid
        possible error that would otherwise occur in this case. Turning this
        on will make the run time substantially longer, so it should only be
        used when labeled regions may have a non-negligible portion of their
        boundary within a <2 pixel gap from another label.
    omit_image_edges : bool, optional
        This can be set to avoid an additional padding step that includes the
        edges of objects that correspond to the image edge as part of the
        perimeter. We cannot accurately estimate the perimeter of objects
        falling partly outside of `image`, so it seems acceptable to just set
        this to True. The default remains False for consistency with upstream
        scikit-image.

    Returns
    -------
    perimeter : float
        Total perimeter of all objects in binary image.

    Notes
    -----
    This measure is based on Crofton formula [1], which is a measure from
    integral geometry. It is defined for general curve length evaluation via
    a double integral along all directions. In a discrete
    space, 2 or 4 directions give a quite good approximation, 4 being more
    accurate than 2 for more complex shapes.

    Similar to :func:`~.measure.perimeter`, this function returns an
    approximation of the perimeter in continuous space.

    The `perimeter` method does not consider the boundary along the image edge
    as image as part of the perimeter, while the `perimeter_crofton` method
    does. In any case, an object touching the image edge likely extends outside
    of the field of view, so an accurate perimeter cannot be measured for such
    objects.

    See Also
    --------
    perimeter

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Crofton_formula
    .. [2] S. Rivollier. Analyse d’image geometrique et morphometrique par
           diagrammes de forme et voisinages adaptatifs generaux. PhD thesis,
           2010.
           Ecole Nationale Superieure des Mines de Saint-Etienne.
           https://tel.archives-ouvertes.fr/tel-00560838
    """
    if max_label is None:
        max_label = int(labels.max())

    # maximum possible value for XF_labeled input to bincount
    # need to choose integer range large enough that this won't overflow
    max_val = (max_label + 1) * 16
    if max_val < 256:
        image_dtype = cp.uint8
    elif max_val < 65536:
        image_dtype = cp.uint16
    elif max_val < 2**32:
        image_dtype = cp.uint32
    else:
        image_dtype = cp.uint64

    if image_dtype == cp.uint8:
        # can directly view bool as uint8 without a copy
        image = (labels > 0).view(cp.uint8)
    else:
        image = (labels > 0).astype(image_dtype)

    if not omit_image_edges:
        image = pad(image, pad_width=1, mode="constant")
    image_filtered = ndi.convolve(
        image,
        cp.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]]),
        mode="constant",
        cval=0,
    )

    # dilate labels by 1 pixel so we can sum with values in XF to give
    # unique histogram bins for each labeled regions (as long as no labeled
    # regions are within < 2 pixels from another labeled region)
    if not omit_image_edges:
        labels_pad = cp.pad(labels, pad_width=1, mode="constant")
        labels_dilated = ndi.grey_dilation(labels_pad, 3, mode="constant")
    else:
        labels_dilated = ndi.grey_dilation(labels, 3, mode="constant")

    if robust:
        # check possibly too-close regions for which we may need to manually
        # recompute the regions perimeter in isolation
        if omit_image_edges:
            labels_dilated2 = ndi.grey_dilation(labels, 5, mode="constant")
            labels2 = labels_dilated2 * image
        else:
            labels_dilated2 = ndi.grey_dilation(
                labels_dilated, 5, mode="constant"
            )
            labels2 = labels_dilated2 * image
            labels2 = labels2[1:-1, 1:-1]
        diffs = labels != labels2

        labels_to_recompute = []
        # regions to recompute
        if cp.any(diffs):
            labels_to_recompute = cp.asnumpy(cp.unique(labels[diffs]))
            bbox, slices = regionprops_bbox_coords(labels, return_slices=True)
            warnings.warn(
                "some labeled regions are <= 2 pixels from another region. The "
                "perimeter may be underestimated for one of the labels when "
                "two labels are < 2 pixels from touching"
            )

    # values in XF are guaranteed to be in range [0, 15] so need to multiply
    # each label by 16 to make sure all labels have a unique set of values
    # during bincount
    image_filtered_labeled = image_filtered + 16 * labels_dilated

    minlength = 16 * (max_label + 1)
    h = cp.bincount(image_filtered_labeled.ravel(), minlength=minlength)
    # values for label=1 start at index 16
    h = h[16:minlength].reshape((max_label, 16))

    # definition of the LUT
    # fmt: off
    if directions == 2:
        coefs = [0, np.pi / 2, 0, 0, 0, np.pi / 2, 0, 0,
                 np.pi / 2, np.pi, 0, 0, np.pi / 2, np.pi, 0, 0]
    else:
        sq2 = math.sqrt(2)
        coefs = [0, np.pi / 4 * (1 + 1 / sq2),
                 np.pi / (4 * sq2),
                 np.pi / (2 * sq2), 0,
                 np.pi / 4 * (1 + 1 / sq2),
                 0, np.pi / (4 * sq2), np.pi / 4, np.pi / 2,
                 np.pi / (4 * sq2), np.pi / (4 * sq2),
                 np.pi / 4, np.pi / 2, 0, 0]

    coefs = cp.asarray(coefs, dtype=cp.float32)
    perimeters = coefs @ h.T
    if robust:
        # recompute perimeter in isolation for each region that may be too
        # close to another one
        shape = labels_dilated.shape
        for lab in labels_to_recompute:
            sl = slices[lab - 1]
            if omit_image_edges:
                # keep boundary of 1 so object is not at 'edge' of cropped
                # region (unless it is at a true image edge)
                ld = labels_dilated[
                    max(sl[0].start - 1, 0):min(sl[0].stop + 1, shape[0]),
                    max(sl[0].start - 1, 0):min(sl[1].stop + 1, shape[1])
                ]
            else:
                # keep boundary of 1 so object is not at 'edge' of cropped
                # region (unless it is at a true image edge)
                # + 2 is because labels_pad is padded, but labels was not
                ld = labels_pad[
                    sl[0].start:sl[0].stop + 2, sl[1].start:sl[1].stop + 2
                ]
            p = regionprops_perimeter_crofton(
                ld == lab,
                directions=directions,
                omit_image_edges=omit_image_edges)
            # print(f"label {lab}: old perimeter={perimeters[lab - 1]}, new_perimeter={p}")  # noqa: E501
            perimeters[lab - 1] = p[0]
    return perimeters
