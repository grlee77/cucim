import math

import cupy as cp
import numpy as np
from packaging.version import parse

from cucim.skimage._vendored import ndimage as ndi, pad
from cucim.skimage.util import map_array

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
    "regionprops_centroid_weighted",
    "regionprops_euler",
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

        - coord_sum : shape (array_size, ndim)
            local sum of coordinates across the local set of labels encountered

    Output variables written to:

        - coord_sums : shape (max_label, 2 * ndim)
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
    max_labels_per_thread=None,
):
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)
    if compute_num_pixels:
        count_dtype = cp.dtype(cp.uint32 if int32_count else cp.uint64)
    if compute_coordinate_sums:
        coord_sum_dtype = cp.dtype(cp.uint64)
        coord_sum_ctype = "uint64_t"

    array_size = pixels_per_thread
    if max_labels_per_thread is not None:
        array_size = min(pixels_per_thread, max_labels_per_thread)

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
    if compute_bbox or compute_coordinate_sums:
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
    if compute_bbox:
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
    max_labels_per_thread=None,
):
    if compute_num_pixels:
        count_dtype = cp.dtype(cp.uint32 if int32_count else cp.uint64)

    (
        sum_dtype,
        c_sum_type,
        min_max_dtype,
        c_min_max_type,
    ) = _get_intensity_img_kernel_dtypes(image_dtype)

    array_size = pixels_per_thread
    if max_labels_per_thread is not None:
        array_size = min(pixels_per_thread, max_labels_per_thread)

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


def _get_raw_moments_code(
    coord_c_type,
    moments_c_type,
    ndim,
    order,
    array_size,
    num_channels=1,
    has_spacing=False,
    has_weights=False,
):
    """
    Notes
    -----
    Local variables created:

        - local_moments : shape (array_size, num_channels, num_moments)
            local set of moments up to the specified order (1-3 supported)

    Output variables written to:

        - moments : shape (max_label, num_channels, num_moments)
    """

    # number is for a densely populated moments matrix of size (order + 1) per
    # side (values at locations where order is greater than specified will be 0)
    num_moments = (order + 1) ** ndim

    if order > 3:
        raise ValueError("Only moments of orders 0-3 are supported")

    use_floating_point = moments_c_type in ["float", "double"]

    source_pre = f"""
    {moments_c_type} local_moments[{array_size*num_channels*num_moments}] = {{0}};
    {coord_c_type} m_offset = 0;
    {coord_c_type} local_off = 0;\n"""  # noqa: E501
    if has_weights:
        source_pre += f"""
    {moments_c_type} w = 0.0;\n"""

    # op uses external coordinate array variables:
    #    bbox : bounding box coordinates, shape (max_label, 2*ndim)
    #    in_coord[0]...in_coord[ndim - 1] : coordinates
    #        coordinates in the labeled image at the current index
    #    ii : index into labels array
    #    current_label : value of the label image at location ii
    #    spacing (optional) : pixel spacings
    #    img (optional) : intensity image
    source_operation = ""
    # using bounding box to transform the global coordinates to local ones
    # (c0 = local coordinate on axis 0, etc.)
    for d in range(ndim):
        source_operation += f"""
                {moments_c_type} c{d} = in_coord[{d}]
                            - bbox[(current_label - 1) * {2 * ndim} + {2*d}];"""
        if has_spacing:
            source_operation += f"""
                c{d} *= spacing[{d}];"""

    # need additional multiplication by the intensity value for weighted case
    w = "w * " if has_weights else ""
    for c in range(num_channels):
        source_operation += f"""
                local_off = {num_moments*num_channels}*offset + {c * num_moments};\n"""  # noqa: E501

        # zeroth moment
        if has_weights:
            source_operation += f"""
                w = static_cast<{moments_c_type}>(img[{num_channels} * ii + {c}]);
                local_moments[local_off] += w;\n"""  # noqa: E501
        elif use_floating_point:
            source_operation += """
                local_moments[local_off] += 1.0;\n"""
        else:
            source_operation += """
                local_moments[local_off] += 1;\n"""

        # moments for order 1-3
        if ndim == 2:
            if order == 1:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c1;
                local_moments[local_off + 2] += {w}c0;\n"""
            elif order == 2:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c1;
                local_moments[local_off + 2] += {w}c1 * c1;
                local_moments[local_off + 3] += {w}c0;
                local_moments[local_off + 4] += {w}c0 * c1;
                local_moments[local_off + 6] += {w}c0 * c0;\n"""
            elif order == 3:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c1;
                local_moments[local_off + 2] += {w}c1 * c1;
                local_moments[local_off + 3] += {w}c1 * c1 * c1;
                local_moments[local_off + 4] += {w}c0;
                local_moments[local_off + 5] += {w}c0 * c1;
                local_moments[local_off + 6] += {w}c0 * c1 * c1;
                local_moments[local_off + 8] += {w}c0 * c0;
                local_moments[local_off + 9] += {w}c0 * c0 * c1;
                local_moments[local_off + 12] += {w}c0 * c0 * c0;\n"""
        elif ndim == 3:
            if order == 1:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c2;
                local_moments[local_off + 2] += {w}c1;
                local_moments[local_off + 4] += {w}c0;\n"""
            elif order == 2:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c2;
                local_moments[local_off + 2] += {w}c2 * c2;
                local_moments[local_off + 3] += {w}c1;
                local_moments[local_off + 4] += {w}c1 * c2;
                local_moments[local_off + 6] += {w}c1 * c1;
                local_moments[local_off + 9] += {w}c0;
                local_moments[local_off + 10] += {w}c0 * c2;
                local_moments[local_off + 12] += {w}c0 * c1;
                local_moments[local_off + 18] += {w}c0 * c0;\n"""
            elif order == 3:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c2;
                local_moments[local_off + 2] += {w}c2 * c2;
                local_moments[local_off + 3] += {w}c2 * c2 * c2;
                local_moments[local_off + 4] += {w}c1;
                local_moments[local_off + 5] += {w}c1 * c2;
                local_moments[local_off + 6] += {w}c1 * c2 * c2;
                local_moments[local_off + 8] += {w}c1 * c1;
                local_moments[local_off + 9] += {w}c1 * c1 * c2;
                local_moments[local_off + 12] += {w}c1 * c1 * c1;
                local_moments[local_off + 16] += {w}c0;
                local_moments[local_off + 17] += {w}c0 * c2;
                local_moments[local_off + 18] += {w}c0 * c2 * c2;
                local_moments[local_off + 20] += {w}c0 * c1;
                local_moments[local_off + 21] += {w}c0 * c1 * c2;
                local_moments[local_off + 24] += {w}c0 * c1 * c1;
                local_moments[local_off + 32] += {w}c0 * c0;
                local_moments[local_off + 33] += {w}c0 * c0 * c2;
                local_moments[local_off + 36] += {w}c0 * c0 * c1;
                local_moments[local_off + 48] += {w}c0 * c0 * c0;\n"""
        else:
            raise ValueError("only ndim = 2 or 3 is supported")

    # post_operation uses external variables:
    #     ii : index into num_pixels array
    #     lab : label value that corresponds to location ii
    #     coord_sums : output with shape (max_label, ndim)
    source_post = ""
    for c in range(0, num_channels):
        source_post += f"""
                // moments outputs
                m_offset = {num_moments*num_channels}*(lab - 1) + {c * num_moments};
                local_off = {num_moments*num_channels}*ii + {c * num_moments};
                atomicAdd(&moments[m_offset], local_moments[local_off]);\n"""  # noqa: E501

        if ndim == 2:
            if order == 1:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);\n"""  # noqa: E501
            elif order == 2:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);
                atomicAdd(&moments[m_offset + 3], local_moments[local_off + 3]);
                atomicAdd(&moments[m_offset + 4], local_moments[local_off + 4]);
                atomicAdd(&moments[m_offset + 6], local_moments[local_off + 6]);\n"""  # noqa: E501
            elif order == 3:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);
                atomicAdd(&moments[m_offset + 3], local_moments[local_off + 3]);
                atomicAdd(&moments[m_offset + 4], local_moments[local_off + 4]);
                atomicAdd(&moments[m_offset + 5], local_moments[local_off + 5]);
                atomicAdd(&moments[m_offset + 6], local_moments[local_off + 6]);
                atomicAdd(&moments[m_offset + 8], local_moments[local_off + 8]);
                atomicAdd(&moments[m_offset + 9], local_moments[local_off + 9]);
                atomicAdd(&moments[m_offset + 12], local_moments[local_off + 12]);\n"""  # noqa: E501
        elif ndim == 3:
            if order == 1:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);
                atomicAdd(&moments[m_offset + 4], local_moments[local_off + 4]);\n"""  # noqa: E501
            elif order == 2:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);
                atomicAdd(&moments[m_offset + 3], local_moments[local_off + 3]);
                atomicAdd(&moments[m_offset + 4], local_moments[local_off + 4]);
                atomicAdd(&moments[m_offset + 6], local_moments[local_off + 6]);
                atomicAdd(&moments[m_offset + 9], local_moments[local_off + 9]);
                atomicAdd(&moments[m_offset + 10], local_moments[local_off + 10]);
                atomicAdd(&moments[m_offset + 12], local_moments[local_off + 12]);
                atomicAdd(&moments[m_offset + 18], local_moments[local_off + 18]);\n"""  # noqa: E501
            elif order == 3:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);
                atomicAdd(&moments[m_offset + 3], local_moments[local_off + 3]);
                atomicAdd(&moments[m_offset + 4], local_moments[local_off + 4]);
                atomicAdd(&moments[m_offset + 5], local_moments[local_off + 5]);
                atomicAdd(&moments[m_offset + 6], local_moments[local_off + 6]);
                atomicAdd(&moments[m_offset + 8], local_moments[local_off + 8]);
                atomicAdd(&moments[m_offset + 9], local_moments[local_off + 9]);
                atomicAdd(&moments[m_offset + 12], local_moments[local_off + 12]);
                atomicAdd(&moments[m_offset + 16], local_moments[local_off + 16]);
                atomicAdd(&moments[m_offset + 17], local_moments[local_off + 17]);
                atomicAdd(&moments[m_offset + 18], local_moments[local_off + 18]);
                atomicAdd(&moments[m_offset + 20], local_moments[local_off + 20]);
                atomicAdd(&moments[m_offset + 21], local_moments[local_off + 21]);
                atomicAdd(&moments[m_offset + 24], local_moments[local_off + 24]);
                atomicAdd(&moments[m_offset + 32], local_moments[local_off + 32]);
                atomicAdd(&moments[m_offset + 33], local_moments[local_off + 33]);
                atomicAdd(&moments[m_offset + 36], local_moments[local_off + 36]);
                atomicAdd(&moments[m_offset + 48], local_moments[local_off + 48]);\n"""  # noqa: E501
    return source_pre, source_operation, source_post


@cp.memoize(for_each_device=True)
def get_raw_moments_kernel(
    ndim,
    order,
    moments_dtype=cp.float64,
    int32_coords=True,
    spacing=None,
    weighted=False,
    num_channels=1,
    pixels_per_thread=8,
    max_labels_per_thread=None,
):
    moments_dtype = cp.dtype(moments_dtype)

    array_size = pixels_per_thread
    if max_labels_per_thread is not None:
        array_size = min(pixels_per_thread, max_labels_per_thread)

    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)
    if coord_dtype.itemsize <= 4:
        coord_c_type = "unsigned int"
    else:
        coord_c_type = "unsigned long long"

    use_floating_point = moments_dtype.kind == "f"
    has_spacing = spacing is not None
    if (weighted or has_spacing) and not use_floating_point:
        raise ValueError(
            "`moments_dtype` must be a floating point type for weighted "
            "moments calculations or moment calculations using spacing."
        )
    moments_c_type = "double" if use_floating_point else "unsigned long long"
    if spacing is not None:
        if len(spacing) != ndim:
            raise ValueError("len(spacing) must equal len(shape)")
        if moments_dtype.kind != "f":
            raise ValueError("moments must have a floating point data type")

    moments_pre, moments_op, moments_post = _get_raw_moments_code(
        coord_c_type=coord_c_type,
        moments_c_type=moments_c_type,
        ndim=ndim,
        order=order,
        array_size=array_size,
        has_weights=weighted,
        has_spacing=spacing is not None,
        num_channels=num_channels,
    )

    # store only counts for label > 0  (label = 0 is the background)
    source = f"""
      uint64_t start_index = {pixels_per_thread}*i;
    """
    source += moments_pre

    inner_op = ""

    source += _unravel_loop_index_declarations(
        "labels", ndim, uint_t=coord_c_type
    )

    inner_op += _unravel_loop_index(
        "labels",
        ndim=ndim,
        uint_t=coord_c_type,
        raveled_index="ii",
        omit_declarations=True,
    )
    inner_op += moments_op

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

    source += moments_post
    source += """
        }
      }\n"""

    # print(source)
    inputs = (
        f"raw X labels, raw uint64 labels_size, raw {coord_dtype.name} bbox"
    )
    if spacing:
        inputs += ", raw float64 spacing"
    if weighted:
        inputs += ", raw Y img"
    outputs = f"raw {moments_dtype.name} moments"
    weighted_str = "_weighted" if weighted else ""
    spacing_str = "_sp" if spacing else ""
    name = f"cucim_moments{weighted_str}{spacing_str}_order{order}_{ndim}d"
    name += f"_{coord_dtype.char}_{moments_dtype.char}_batch{pixels_per_thread}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def _get_count_dtype(label_image_size):
    """atomicAdd only supports int32, uint32, int64, uint64, float32, float64"""
    int32_count = label_image_size < 2**32
    count_dtype = cp.dtype(cp.uint32 if int32_count else cp.uint64)
    return count_dtype, int32_count


def regionprops_num_pixels(
    label_image,
    max_label=None,
    pixels_per_thread=16,
    max_labels_per_thread=None,
    props_dict=None,
):
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
        max_labels_per_thread=max_labels_per_thread,
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
    if props_dict is not None:
        props_dict["num_pixels"] = counts
    return counts


def regionprops_area(
    label_image,
    spacing=None,
    max_label=None,
    dtype=cp.float32,
    pixels_per_thread=16,
    max_labels_per_thread=None,
    props_dict=None,
):
    # integer atomicAdd is faster than floating point so better to convert
    # after counting
    if props_dict is not None and "num_pixels" in props_dict:
        num_pixels = props_dict["num_pixels"]
    else:
        num_pixels = regionprops_num_pixels(
            label_image,
            max_label=max_label,
            pixels_per_thread=pixels_per_thread,
            max_labels_per_thread=max_labels_per_thread,
        )
        if props_dict is not None:
            props_dict["num_pixels"] = num_pixels

    area = num_pixels.astype(dtype)
    if spacing is not None:
        if isinstance(spacing, cp.ndarray):
            pixel_area = cp.product(spacing)
        else:
            pixel_area = math.prod(spacing)
        area *= pixel_area
    if props_dict is not None:
        props_dict["area"] = area
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
    max_labels_per_thread=None,
    props_dict=None,
):
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_shapes(label_image, intensity_image)

    count_dtype, int32_count = _get_count_dtype(label_image.size)

    image_dtype = intensity_image.dtype
    sum_dtype, _, _, _ = _get_intensity_img_kernel_dtypes(image_dtype)

    if props_dict is not None and "num_pixels" in props_dict:
        counts = props_dict["num_pixels"]
        if counts.dtype != count_dtype:
            counts = counts.astype(count_dtype, copy=False)
        compute_num_pixels = False
    else:
        counts = cp.zeros(num_counts, dtype=count_dtype)
        compute_num_pixels = True

    sum_shape = (
        (num_counts,) if num_channels == 1 else (num_counts, num_channels)
    )
    sums = cp.zeros(sum_shape, dtype=sum_dtype)

    kernel = get_intensity_measure_kernel(
        int32_count=int32_count,
        image_dtype=image_dtype,
        num_channels=num_channels,
        compute_num_pixels=compute_num_pixels,
        compute_sum=True,
        compute_sum_sq=False,
        pixels_per_thread=pixels_per_thread,
        max_labels_per_thread=max_labels_per_thread,
    )

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    if not intensity_image.flags.c_contiguous:
        intensity_image = cp.ascontiguousarray(intensity_image)

    if compute_num_pixels:
        outputs = (counts, sums)
    else:
        outputs = (sums,)

    kernel(
        label_image,
        label_image.size,
        intensity_image,
        *outputs,
        size=math.ceil(label_image.size / pixels_per_thread),
    )

    if num_channels > 1:
        means = sums / counts[:, cp.newaxis]
    else:
        means = sums / counts
    means = means.astype(mean_dtype, copy=False)
    if props_dict is not None:
        props_dict["intensity_mean"] = means
        if "num_pixels" not in props_dict:
            props_dict["num_pixels"] = counts
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
    max_labels_per_thread=None,
    props_dict=None,
):
    if max_label is None:
        max_label = int(label_image.max())
    num_counts = max_label

    num_channels = _check_shapes(label_image, intensity_image)

    image_dtype = intensity_image.dtype
    sum_dtype, _, _, _ = _get_intensity_img_kernel_dtypes(image_dtype)

    count_dtype, int32_count = _get_count_dtype(label_image.size)

    if props_dict is not None and "num_pixels" in props_dict:
        counts = props_dict["num_pixels"]
        if counts.dtype != count_dtype:
            counts = counts.astype(count_dtype, copy=False)
        compute_num_pixels = False
    else:
        counts = cp.zeros(num_counts, dtype=count_dtype)
        compute_num_pixels = True

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
            compute_num_pixels=compute_num_pixels,
            compute_sum=True,
            compute_sum_sq=True,
            pixels_per_thread=pixels_per_thread,
            max_labels_per_thread=max_labels_per_thread,
        )
        if compute_num_pixels:
            outputs = (counts, sums, sumsqs)
        else:
            outputs = (sums, sumsqs)
        kernel(
            label_image,
            label_image.size,
            intensity_image,
            *outputs,
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
    if props_dict is not None:
        props_dict["intensity_std"] = stds
        props_dict["intensity_mean"] = means
        if "num_pixels" not in props_dict:
            props_dict["num_pixels"] = counts
    return counts, means, stds


def _regionprops_min_or_max_intensity(
    label_image,
    intensity_image,
    max_label=None,
    compute_min=True,
    compute_max=False,
    pixels_per_thread=8,
    max_labels_per_thread=None,
    props_dict=None,
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
        max_labels_per_thread=max_labels_per_thread,
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
        if props_dict is not None:
            props_dict["intensity_min"] = minimums
        if props_dict is not None:
            props_dict["intensity_max"] = maximums
        return minimums, maximums
    elif compute_min:
        kernel(label_image, lab_size, intensity_image, minimums, size=sz)
        if props_dict is not None:
            props_dict["intensity_min"] = minimums
        return minimums
    elif compute_max:
        kernel(label_image, lab_size, intensity_image, maximums, size=sz)
        if props_dict is not None:
            props_dict["intensity_max"] = maximums
        return maximums


def regionprops_intensity_min(
    label_image,
    intensity_image,
    max_label=None,
    pixels_per_thread=8,
    max_labels_per_thread=None,
    props_dict=None,
):
    return _regionprops_min_or_max_intensity(
        label_image,
        intensity_image,
        max_label=max_label,
        compute_min=True,
        compute_max=False,
        pixels_per_thread=pixels_per_thread,
        max_labels_per_thread=max_labels_per_thread,
        props_dict=props_dict,
    )


def regionprops_intensity_max(
    label_image,
    intensity_image,
    max_label=None,
    pixels_per_thread=8,
    max_labels_per_thread=None,
    props_dict=None,
):
    return _regionprops_min_or_max_intensity(
        label_image,
        intensity_image,
        max_label=max_label,
        compute_min=False,
        compute_max=True,
        pixels_per_thread=pixels_per_thread,
        max_labels_per_thread=max_labels_per_thread,
        props_dict=props_dict,
    )


def regionprops_bbox_coords(
    label_image,
    max_label=None,
    return_slices=False,
    pixels_per_thread=16,
    max_labels_per_thread=None,
    props_dict=None,
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
        max_labels_per_thread=max_labels_per_thread,
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
    if props_dict is not None:
        props_dict["bbox"] = bbox_coords

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
        if props_dict is not None:
            props_dict["slice"] = bbox_slices
    else:
        bbox_slices = None

    return bbox_coords, bbox_slices


def regionprops_centroid(
    label_image,
    max_label=None,
    coord_dtype=cp.uint32,
    pixels_per_thread=16,
    max_labels_per_thread=None,
    props_dict=None,
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
    ndim = label_image.ndim

    int32_coords = max(label_image.shape) < 2**32
    if props_dict is not None and "num_pixels" in props_dict:
        centroid_counts = props_dict["num_pixels"]
        if centroid_counts.dtype != cp.uint32:
            centroid_counts = centroid_counts.astype(cp.uint32)
        compute_num_pixels = False
    else:
        centroid_counts = cp.zeros((max_label,), dtype=cp.uint32)
        compute_num_pixels = True

    bbox_coords_kernel = get_bbox_coords_kernel(
        ndim=label_image.ndim,
        int32_coords=int32_coords,
        compute_bbox=False,
        compute_num_pixels=compute_num_pixels,
        compute_coordinate_sums=True,
        pixels_per_thread=pixels_per_thread,
        max_labels_per_thread=max_labels_per_thread,
    )

    # bbox_coords = cp.zeros((max_label, 2 * ndim), dtype=coord_dtype)
    centroid_sums = cp.zeros((max_label, ndim), dtype=cp.uint64)

    # # Initialize value for atomicMin on even coordinates
    # # The value for atomicMax columns is already 0 as desired.
    # bbox_coords[:, ::2] = cp.iinfo(coord_dtype).max

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    if compute_num_pixels:
        outputs = (centroid_counts, centroid_sums)
    else:
        outputs = centroid_sums
    bbox_coords_kernel(
        label_image,
        label_image.size,
        *outputs,
        size=math.ceil(label_image.size / pixels_per_thread),
    )

    centroid = centroid_sums / centroid_counts[:, cp.newaxis]
    if props_dict is not None:
        props_dict["centroid"] = centroid
        if "num_pixels" not in props_dict:
            props_dict["num_pixels"] = centroid_counts
    return centroid


@cp.memoize(for_each_device=True)
def get_centroid_local_kernel(coord_dtype, ndim):
    """Keep this kernel for n-dimensional support as the raw_moments kernels
    currently only support 2D and 3D data.
    """
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
    max_labels_per_thread=None,
    props_dict=None,
):
    """Compute the central moments of the labeled regions.

    dimensions supported: nD

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

    ndim = label_image.ndim

    if props_dict is not None and "moments" in props_dict and ndim in [2, 3]:
        # already have the moments needed in previously computed properties
        moments = props_dict["moments"]
        # can't compute if only zeroth moment is present
        if moments.shape[-1] > 1:
            centroid_local = cp.empty((max_label, ndim), dtype=moments.dtype)
            if ndim == 2:
                m0 = moments[:, 0, 0]
                centroid_local[:, 0] = moments[:, 1, 0] / m0
                centroid_local[:, 1] = moments[:, 0, 1] / m0
            else:
                m0 = moments[:, 0, 0, 0]
                centroid_local[:, 0] = moments[:, 1, 0, 0] / m0
                centroid_local[:, 1] = moments[:, 0, 1, 0] / m0
                centroid_local[:, 2] = moments[:, 0, 0, 1] / m0
        props_dict["centroid_local"] = centroid_local
        return centroid_local

    if props_dict is not None and "bbox" in props_dict:
        # reuse previously computed bounding box coordinates
        bbox_coords = props_dict["bbox"]
        if bbox_coords.dtype != coord_dtype:
            bbox_coords = bbox_coords.astype(coord_dtype)

    else:
        bbox_coords_kernel = get_bbox_coords_kernel(
            ndim=label_image.ndim,
            int32_coords=int32_coords,
            compute_bbox=True,
            compute_num_pixels=False,
            compute_coordinate_sums=False,
            pixels_per_thread=pixels_per_thread,
            max_labels_per_thread=max_labels_per_thread,
        )

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
    if props_dict is not None:
        props_dict["centroid_local"] = centroid_local
        if "num_pixels" not in props_dict:
            props_dict["num_pixels"] = counts
    return centroid_local


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


def regionprops_area_bbox(
    bbox, area_dtype=cp.float32, spacing=None, props_dict=None
):
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
    if props_dict is not None:
        props_dict["area_bbox"] = area_bbox
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


def regionprops_moments(
    label_image,
    intensity_image=None,
    max_label=None,
    order=2,
    spacing=None,
    pixels_per_thread=10,
    max_labels_per_thread=None,
    props_dict=None,
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

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    ndim = label_image.ndim

    int32_coords = max(label_image.shape) < 2**32
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)
    if props_dict is not None and "bbox" in props_dict:
        bbox_coords = props_dict["bbox"]
        if bbox_coords.dtype != coord_dtype:
            bbox_coords = bbox_coords.astype(coord_dtype)
    else:
        bbox_kernel = get_bbox_coords_kernel(
            ndim=ndim,
            int32_coords=int32_coords,
            compute_bbox=True,
            compute_num_pixels=False,
            compute_coordinate_sums=False,
            pixels_per_thread=pixels_per_thread,
            max_labels_per_thread=max_labels_per_thread,
        )

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
        if props_dict is not None:
            props_dict["bbox"] = bbox_coords

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

    # total number of elements in the moments matrix
    moments = cp.zeros(moments_shape, dtype=cp.float64)
    moments_kernel = get_raw_moments_kernel(
        ndim=label_image.ndim,
        order=order,
        moments_dtype=moments.dtype,
        int32_coords=int32_coords,
        spacing=spacing,
        weighted=weighted,
        num_channels=num_channels,
        pixels_per_thread=pixels_per_thread,
        max_labels_per_thread=max_labels_per_thread,
    )
    input_args = (
        label_image,
        label_image.size,
        bbox_coords,
    )
    if spacing:
        input_args = input_args + (cp.asarray(spacing, dtype=cp.float64),)
    if weighted:
        input_args = input_args + (intensity_image,)
    size = math.ceil(label_image.size / pixels_per_thread)
    moments_kernel(*input_args, moments, size=size)
    if props_dict is not None:
        if weighted:
            props_dict["moments_weighted"] = moments
        else:
            props_dict["moments"] = moments
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


def regionprops_moments_central(
    moments_raw, ndim, weighted=False, props_dict=None
):
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
    if props_dict is not None:
        if weighted:
            props_dict["moments_weighted_central"] = moments_central
        else:
            props_dict["moments_central"] = moments_central
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
    moments_central,
    ndim,
    spacing=None,
    pixel_correction=False,
    weighted=False,
    props_dict=None,
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
    if props_dict is not None:
        if weighted:
            props_dict["moments_weighted_normalized"] = moments_norm
        else:
            props_dict["moments_normalized"] = moments_norm
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


def regionprops_moments_hu(moments_normalized, weighted=False, props_dict=None):
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
    if props_dict is not None:
        if weighted:
            props_dict["moments_weighted_hu"] = moments_hu
        else:
            props_dict["moments_hu"] = moments_hu
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
    moments_central, ndim, compute_orientation=False, props_dict=None
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
    if props_dict is not None:
        props_dict["inertia_tensor"] = itensor
    return itensor


@cp.memoize(for_each_device=True)
def get_spd_matrix_eigvals_kernel(
    rank,
    compute_eigenvectors=False,
    compute_axis_lengths=False,
    compute_eccentricity=False,
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
            evecs[offset + 3] = 1.0;\n"""
        if compute_axis_lengths:
            source += """
            axis_lengths[offset_out] = 4.0 * sqrt(lam1);
            axis_lengths[offset_out + 1] = 4.0 * sqrt(lam2);\n"""
        if compute_eccentricity:
            source += """
            eccentricity[i] =  sqrt(1.0 - lam2 / lam1);\n"""
    elif rank == 3:
        if compute_eccentricity:
            raise ValueError("eccentricity only supported for 2D images")

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
            evecs[offset + 8] = 1.0;\n"""
        if compute_axis_lengths:
            source += """
            // formula reference:
            //   https://github.com/scikit-image/scikit-image/blob/v0.25.0/skimage/measure/_regionprops.py#L275-L295
            // note: added max to clip possible small (e.g. 1e-7) negative value due to numerical error
            axis_lengths[offset_out] = sqrt(10.0 * (lam1 + lam2 - lam3));
            axis_lengths[offset_out + 1] = sqrt(10.0 * (lam1 - lam2 + lam3));
            axis_lengths[offset_out + 2] = sqrt(10.0 * max(-lam1 + lam2 + lam3, 0.0));\n"""  # noqa: E501
    else:
        # note: ndim here is the number of spatial image dimensions
        raise ValueError("only rank = 2 or 3 is supported")
    inputs = "raw F spd_matrix"
    outputs = ["raw F out"]
    if compute_eigenvectors:
        outputs.append("raw F eigvecs")
        ev_str = "eigvecs_"
    else:
        ev_str = ""
    name = f"cucim_spd_matrix_eigvals_{ev_str}{rank}d"
    if compute_axis_lengths:
        outputs.append("raw F axis_lengths")
        name += "_with_axis"
    if compute_eccentricity:
        outputs.append("raw F eccentricity")
        name += "_eccen"
    outputs = ", ".join(outputs)
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_inertia_tensor_eigvals(
    inertia_tensor,
    compute_axis_lengths=False,
    compute_eccentricity=False,
    props_dict=None,
):
    # inertia tensor should have shape (ndim, ndim) on last two axes
    ndim = inertia_tensor.shape[-1]
    if ndim < 2 or ndim > 3:
        raise ValueError("inertia tensor only implemented for 2D and 3D images")
    nbatch = math.prod(inertia_tensor.shape[:-2])

    if compute_eccentricity and ndim != 2:
        raise ValueError("eccentricity is only supported for 2D images")

    if inertia_tensor.dtype.kind != "f":
        raise ValueError("moments_central must have a floating point dtype")

    if not inertia_tensor.flags.c_contiguous:
        inertia_tensor = cp.ascontiguousarray(inertia_tensor)

    kernel = get_spd_matrix_eigvals_kernel(
        rank=ndim,
        compute_axis_lengths=compute_axis_lengths,
        compute_eigenvectors=False,
        compute_eccentricity=compute_eccentricity,
    )
    eigvals_shape = inertia_tensor.shape[:-2] + (ndim,)
    eigvals = cp.empty(eigvals_shape, dtype=inertia_tensor.dtype)
    outputs = [eigvals]
    if compute_axis_lengths:
        axis_lengths = cp.empty(eigvals_shape, dtype=inertia_tensor.dtype)
        outputs.append(axis_lengths)
    if compute_eccentricity:
        eccentricity = cp.empty(
            inertia_tensor.shape[:-2], dtype=inertia_tensor.dtype
        )
        outputs.append(eccentricity)
    kernel(inertia_tensor, *outputs, size=nbatch)
    if props_dict is not None:
        props_dict["inertia_tensor_eigvals"] = eigvals
        if compute_eccentricity:
            props_dict["eccentricity"] = eccentricity
        if compute_axis_lengths:
            props_dict["axis_lengths"] = axis_lengths
            props_dict["axis_length_major"] = axis_lengths[..., 0]
            props_dict["axis_length_minor"] = axis_lengths[..., -1]
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


@cp.memoize(for_each_device=True)
def get_centroid_weighted_kernel(
    moments_dtype,
    ndim,
    compute_local=True,
    compute_global=False,
    unit_spacing=True,
    num_channels=1,
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
    if compute_global:
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
        if compute_local:
            source += f"""
            out_local[offset_out + {d}] = moments_raw[offset + {axis_offset}] / m0;"""  # noqa: E501
        if compute_global:
            spc = "" if unit_spacing else f" * spacing[{d}]"
            source += f"""
            out_global[offset_out + {d}] = moments_raw[offset + {axis_offset}] / m0 + bbox[offset_coords + {d * 2}]{spc};"""  # noqa: E501
        axis_offset *= 2
    if num_channels > 1:
        source += """
        }  // channels loop\n"""
    name = f"cucim_centroid_weighted_{ndim}d"
    inputs = ["raw F moments_raw"]
    outputs = []
    if compute_global:
        name += "_global"
        outputs.append("raw F out_global")
        # bounding box coordinates
        inputs.append("raw Y bbox")
        if not unit_spacing:
            inputs.append("raw float64 spacing")
            name += "_spacing"
    if compute_local:
        name += "_local"
        outputs.append("raw F out_local")
    inputs = ", ".join(inputs)
    outputs = ", ".join(outputs)
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_centroid_weighted(
    moments_raw,
    ndim,
    bbox=None,
    compute_local=True,
    compute_global=False,
    weighted=True,
    spacing=None,
    props_dict=None,
):
    max_label = moments_raw.shape[0]
    if moments_raw.ndim == ndim + 2:
        num_channels = moments_raw.shape[1]
    elif moments_raw.ndim == ndim + 1:
        num_channels = 1
    else:
        raise ValueError("moments_raw has unexpected shape")

    if compute_global and bbox is None:
        raise ValueError(
            "bbox coordinates must be provided to get the non-local centroid"
        )

    if not (compute_local or compute_global):
        raise ValueError(
            "nothing to compute: either compute_global and/or compute_local "
            "must be true"
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

    if compute_local and not compute_global:
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
        compute_local=compute_local,
        compute_global=compute_global,
        unit_spacing=unit_spacing,
        num_channels=num_channels,
    )
    centroid_shape = moments_raw.shape[:-ndim] + (ndim,)
    outputs = []
    if compute_global:
        centroid_global = cp.zeros(centroid_shape, dtype=moments_raw.dtype)
        outputs.append(centroid_global)
    if compute_local:
        centroid_local = cp.zeros(centroid_shape, dtype=moments_raw.dtype)
        outputs.append(centroid_local)
    # Note: order of inputs and outputs here must match
    #       get_centroid_weighted_kernel
    kernel(*inputs, *outputs, size=max_label)
    if props_dict is not None:
        if compute_local:
            if weighted:
                props_dict["centroid_weighted_local"] = centroid_local
            else:
                props_dict["centroid_local"] = centroid_local
        if compute_global:
            if weighted:
                props_dict["centroid_weighted"] = centroid_global
            else:
                props_dict["centroid"] = centroid_global
    if compute_global and compute_local:
        return centroid_global, compute_local
    elif compute_global:
        return centroid_global
    return centroid_local


def _reverse_label_values(label_image, max_label):
    """reverses the value of all labels (keeping background value=0 the same)"""
    dtype = label_image.dtype
    labs = cp.asarray(tuple(range(max_label + 1)), dtype=dtype)
    rev_labs = cp.asarray((0,) + tuple(range(max_label, 0, -1)), dtype=dtype)
    return map_array(label_image, labs, rev_labs)


def _find_close_labels(labels, binary_image, max_label):
    # check possibly too-close regions for which we may need to
    # manually recompute the regions perimeter in isolation
    labels_dilated2 = ndi.grey_dilation(labels, 5, mode="constant")
    labels2 = labels_dilated2 * binary_image
    rev_labels = _reverse_label_values(labels, max_label=max_label)
    rev_labels = ndi.grey_dilation(rev_labels, 5, mode="constant")
    rev_labels = rev_labels * binary_image
    labels3 = _reverse_label_values(rev_labels, max_label=max_label)
    diffs = cp.logical_or(labels != labels2, labels != labels3)
    labels_close = cp.asnumpy(cp.unique(labels[diffs]))
    return labels_close


def regionprops_perimeter(
    labels,
    neighborhood=4,
    max_label=None,
    robust=True,
    labels_close=None,
    props_dict=None,
):
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
    labels_close : numpy.ndarray or sequence of int
        List of labeled regions that are less than 2 pixel gap from another
        label. Used when robust=True. If not provided and robust=True, it
        will be computed internally.

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

    TODO(grelee): should be able to make this faster with a customized
    filter/kernel instead of convolve + bincount, etc.

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

    binary_image = labels > 0
    if neighborhood == 4:
        footprint = cp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    else:
        footprint = 3

    eroded_image = ndi.binary_erosion(binary_image, footprint, border_value=0)
    border_image = binary_image.view(cp.uint8) - eroded_image

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
        if labels_close is None:
            labels_close = _find_close_labels(labels, binary_image, max_label)
        # regions to recompute
        if labels_close.size > 0:
            print(
                f"recomputing {labels_close.size} of {max_label} "
                "labels due to close proximity."
            )
            bbox, slices = regionprops_bbox_coords(labels, return_slices=True)

    max_val = 50  # 1 + sum of kernel used by ndi.convolve above
    # values in perimeter_image are guaranteed to be in range [0, max_val) so
    # need to multiply each label by max_val to make sure all labels have a
    # unique set of values during bincount
    perimeter_image = perimeter_image + max_val * labels_dilated

    minlength = max_val * (max_label + 1)

    # only need to bincount masked region near image boundary
    binary_image_mask = ndi.binary_dilation(border_image, 3)
    h = cp.bincount(perimeter_image[binary_image_mask], minlength=minlength)

    # values for label=1 start at index `max_val`
    h = h[max_val:minlength].reshape((max_label, max_val))

    perimeters = perimeter_weights @ h.T
    if robust:
        # recompute perimeter in isolation for each region that may be too
        # close to another one
        shape = binary_image.shape
        for lab in labels_close:
            sl = slices[lab - 1]

            # keep boundary of 1 so object is not at 'edge' of cropped
            # region (unless it is at a true image edge)
            ld = labels[
                max(sl[0].start - 1, 0) : min(sl[0].stop + 1, shape[0]),
                max(sl[1].start - 1, 0) : min(sl[1].stop + 1, shape[1]),
            ]

            # print(f"{lab=}, {sl=}")
            # import matplotlib.pyplot as plt
            # plt.figure(); plt.imshow(ld.get()); plt.show()

            p = regionprops_perimeter(
                ld == lab, max_label=1, neighborhood=neighborhood, robust=False
            )
            perimeters[lab - 1] = p[0]
    if props_dict is not None:
        props_dict["perimeter"] = perimeters
    return perimeters


def regionprops_perimeter_crofton(
    labels,
    directions=4,
    max_label=None,
    robust=True,
    omit_image_edges=False,
    labels_close=None,
    props_dict=None,
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
    labels_close : numpy.ndarray or sequence of int
        List of labeled regions that are less than 2 pixel gap from another
        label. Used when robust=True. If not provided and robust=True, it
        will be computed internally.

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

    TODO(grelee): should be able to make this faster with a customized
    filter/kernel instead of convolve + bincount, etc.

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

    binary_image = labels > 0
    if robust and labels_close is None:
        labels_close = _find_close_labels(labels, binary_image, max_label)
    if not omit_image_edges:
        # Dilate labels by 1 pixel so we can sum with values in image_filtered
        # to give unique histogram bins for each labeled regions (As long as no
        # labeled regions are within < 2 pixels from another labeled region)
        labels_pad = cp.pad(labels, pad_width=1, mode="constant")
        labels_dilated = ndi.grey_dilation(labels_pad, 3, mode="constant")
        binary_image = pad(binary_image, pad_width=1, mode="constant")
        # need dilated mask for later use for indexing into
        # `image_filtered_labeled` for bincount
        binary_image_mask = ndi.binary_dilation(binary_image, 3)
        binary_image_mask = cp.logical_xor(
            binary_image_mask, ndi.binary_erosion(binary_image, 3)
        )
    else:
        labels_dilated = ndi.grey_dilation(labels, 3, mode="constant")
        binary_image_mask = binary_image

    image_filtered = ndi.convolve(
        binary_image.view(cp.uint8),
        cp.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]]),
        mode="constant",
        cval=0,
    )

    if robust:
        if labels_close.size > 0:
            print(
                f"recomputing {labels_close.size} of {max_label} labels"
                " due to close proximity."
            )
            bbox, slices = regionprops_bbox_coords(labels, return_slices=True)

    # values in image_filtered are guaranteed to be in range [0, 15] so need to
    # multiply each label by 16 to make sure all labels have a unique set of
    # values during bincount
    image_filtered_labeled = image_filtered + 16 * labels_dilated

    minlength = 16 * (max_label + 1)
    h = cp.bincount(
        image_filtered_labeled[binary_image_mask], minlength=minlength
    )

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
        for lab in labels_close:
            sl = slices[lab - 1]
            ld = labels[
                max(sl[0].start, 0):min(sl[0].stop, shape[0]),
                max(sl[1].start, 0):min(sl[1].stop, shape[1])
            ]
            p = regionprops_perimeter_crofton(
                ld == lab,
                max_label=1,
                directions=directions,
                omit_image_edges=False,
                robust=False
            )
            perimeters[lab - 1] = p[0]
    if props_dict is not None:
        props_dict["perimeter_crofton"] = perimeters
    return perimeters


def regionprops_euler(
    labels,
    connectivity=None,
    max_label=None,
    robust=True,
    labels_close=None,
    props_dict=None,
):
    """Calculate the Euler characteristic in binary image.

    For 2D objects, the Euler number is the number of objects minus the number
    of holes. For 3D objects, the Euler number is obtained as the number of
    objects plus the number of holes, minus the number of tunnels, or loops.

    Parameters
    ----------
    labels: (M, N[, P]) cupy.ndarray
        Input image. If image is not binary, all values greater than zero
        are considered as the object.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.
        4 or 8 neighborhoods are defined for 2D images (connectivity 1 and 2,
        respectively).
        6 or 26 neighborhoods are defined for 3D images, (connectivity 1 and 3,
        respectively). Connectivity 2 is not defined.
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
    labels_close : numpy.ndarray or sequence of int
        List of labeled regions that are less than 2 pixel gap from another
        label. Used when robust=True. If not provided and robust=True, it
        will be computed internally.

    Returns
    -------
    euler_number : cp.ndarray of int
        Euler characteristic of the set of all objects in the image.

    Notes
    -----
    The Euler characteristic is an integer number that describes the
    topology of the set of all objects in the input image. If object is
    4-connected, then background is 8-connected, and conversely.

    The computation of the Euler characteristic is based on an integral
    geometry formula in discretized space. In practice, a neighborhood
    configuration is constructed, and a LUT is applied for each
    configuration. The coefficients used are the ones of Ohser et al.

    It can be useful to compute the Euler characteristic for several
    connectivities. A large relative difference between results
    for different connectivities suggests that the image resolution
    (with respect to the size of objects and holes) is too low.

    References
    ----------
    .. [1] S. Rivollier. Analyse d’image geometrique et morphometrique par
           diagrammes de forme et voisinages adaptatifs generaux. PhD thesis,
           2010. Ecole Nationale Superieure des Mines de Saint-Etienne.
           https://tel.archives-ouvertes.fr/tel-00560838
    .. [2] Ohser J., Nagel W., Schladitz K. (2002) The Euler Number of
           Discretized Sets - On the Choice of Adjacency in Homogeneous
           Lattices. In: Mecke K., Stoyan D. (eds) Morphology of Condensed
           Matter. Lecture Notes in Physics, vol 600. Springer, Berlin,
           Heidelberg.
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
    from cucim.skimage.measure._regionprops_utils import (
        EULER_COEFS2D_4,
        EULER_COEFS2D_8,
        EULER_COEFS3D_26,
    )

    if max_label is None:
        max_label = int(labels.max())

    # maximum possible value for XF_labeled input to bincount
    # need to choose integer range large enough that this won't overflow

    # check connectivity
    if connectivity is None:
        connectivity = labels.ndim

    # config variable is an adjacency configuration. A coefficient given by
    # variable coefs is attributed to each configuration in order to get
    # the Euler characteristic.
    if labels.ndim == 2:
        config = cp.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]])
        if connectivity == 1:
            coefs = EULER_COEFS2D_4
        else:
            coefs = EULER_COEFS2D_8
        filter_bins = 16
    else:  # 3D images
        if connectivity == 2:
            raise NotImplementedError(
                "For 3D images, Euler number is implemented "
                "for connectivities 1 and 3 only"
            )

        # fmt: off
        config = cp.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 1, 4], [0, 2, 8]],
                           [[0, 0, 0], [0, 16, 64], [0, 32, 128]]])
        # fmt: on
        if connectivity == 1:
            coefs = EULER_COEFS3D_26[::-1]
        else:
            coefs = EULER_COEFS3D_26
        filter_bins = 256

    binary_image = labels > 0

    if robust and labels_close is None:
        labels_close = _find_close_labels(labels, binary_image, max_label)

    binary_image = pad(binary_image, pad_width=1, mode="constant")
    image_filtered = ndi.convolve(
        binary_image.view(cp.uint8),
        config,
        mode="constant",
        cval=0,
    )

    # dilate labels by 1 pixel so we can sum with values in XF to give
    # unique histogram bins for each labeled regions (as long as no labeled
    # regions are within < 2 pixels from another labeled region)
    labels_pad = pad(labels, pad_width=1, mode="constant")
    labels_dilated = ndi.grey_dilation(labels_pad, 3, mode="constant")

    if robust and labels_close.size > 0:
        print(
            f"recomputing {labels_close.size} of {max_label} labels"
            " due to close proximity."
        )
        bbox, slices = regionprops_bbox_coords(labels, return_slices=True)

    # values in image_filtered are guaranteed to be in range [0, filter_bins)
    # so need to multiply each label by filter_bins to make sure all labels
    # have a unique set of values during bincount
    image_filtered_labeled = image_filtered + filter_bins * labels_dilated

    minlength = filter_bins * (max_label + 1)

    bincount_mask = cp.logical_xor(
        ndi.binary_dilation(binary_image, 3),
        ndi.binary_erosion(binary_image, 3),
    )
    h = cp.bincount(image_filtered_labeled[bincount_mask], minlength=minlength)
    # values for label=1 start at index filter_bins
    h = h[filter_bins:minlength].reshape((max_label, filter_bins))

    coefs = cp.asarray(coefs, dtype=cp.int32)
    if labels.ndim == 2:
        euler_number = coefs @ h.T
    else:
        euler_number = 0.125 * coefs @ h.T
        euler_number = euler_number.astype(cp.int64)

    if robust:
        # recompute perimeter in isolation for each region that may be too
        # close to another one
        shape = labels_dilated.shape
        for lab in labels_close:
            sl = slices[lab - 1]
            # keep boundary of 1 so object is not at 'edge' of cropped
            # region (unless it is at a true image edge)
            # + 2 is because labels_pad is padded, but labels was not
            ld = labels_pad[
                max(sl[0].start, 0) : min(sl[0].stop + 2, shape[0]),
                max(sl[1].start, 0) : min(sl[1].stop + 2, shape[1]),
            ]
            euler_num = regionprops_euler(
                ld == lab, connectivity=connectivity, max_label=1, robust=False
            )
            euler_number[lab - 1] = euler_num[0]
    if props_dict is not None:
        props_dict["euler_number"] = euler_number
    return euler_number


# Currently unused utilities
# Some properties can be computed faster using raveled labels and/or
# intensity_image.


def _get_min_integer_dtype(max_size, signed=False):
    # negate to get a signed integer type, but need to also subtract 1, due
    # to asymmetric range on positive side, e.g. we want
    #    max_sz = 127 -> int8  (signed)   uint8 (unsigned)
    #    max_sz = 128 -> int16 (signed)   uint8 (unsigned)
    func = cp.min_scalar_type
    return func(-max_size - 1) if signed else func(max_size)


def get_compressed_labels(
    labels, max_label, intensity_image=None, sort_labels=True
):
    label_dtype = _get_min_integer_dtype(max_label, signed=False)
    if labels.dtype != label_dtype:
        labels = labels.astype(dtype=label_dtype)
    coords_dtype = _get_min_integer_dtype(max(labels.shape), signed=False)
    label_coords = cp.where(labels)
    if label_coords[0].dtype != coords_dtype:
        label_coords = tuple(c.astype(coords_dtype) for c in label_coords)
    labels1d = labels[label_coords]
    if sort_labels:
        sort_indices = cp.argsort(labels1d)
        label_coords = tuple(c[sort_indices] for c in label_coords)
        labels1d = labels1d[sort_indices]
    if intensity_image:
        img1d = intensity_image[label_coords]
        return label_coords, labels1d, img1d
    # max_label = int(labels1d[-1])
    return label_coords, labels1d


need_moments_order1 = {
    "centroid",
    "centroid_local",  # unless ndim > 3
    "centroid_weighted",
    "centroid_weighted_local",
}
need_moments_order2 = {
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "moments",
    "moments_central",
    "moments_normalized",
    "moments_weighted",
    "moments_weighted_central",
    "moments_weighted_normalized",
    "orientation",
}
need_moments_order3 = {"moments_hu", "moments_weighted_hu"}

requires = {}

requires["inertia_tensor_eigvals"] = {
    "eccentricity",
    "inertia_tensor_eigvals",
}
requires["inertia_tensor"] = {
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "axis_major_length",
    "axis_minor_length",
    "inertia_tensor",
    "orientation",
} | requires["inertia_tensor_eigvals"]

requires["moments_normalized"] = {
    "moments_normalized",
    "moments_hu",
}

requires["moments_central"] = (
    {
        "moments_central",
    }
    | requires["moments_normalized"]
    | requires["inertia_tensor"]
)

requires["moments"] = {
    "centroid",
    "centroid_local",  # unless ndim > 3
    "moments",
} | requires["moments_central"]

requires["moments_weighted_normalized"] = {
    "moments_weighted_normalized",
    "moments_weighted_hu",
}

requires["moments_weighted_central"] = {
    "moments_weighted_central",
} | requires["moments_weighted_normalized"]

requires["moments_weighted"] = {
    "centroid_weighted",
    "centroid_weighted_local",
    "moments_weighted",
} | requires["moments_weighted_central"]

# Technically don't need bbox for centroid and centroid_weighted if no other
# moments are going to be computed, but probably best to just always compute
# them via the moments
requires["bbox"] = (
    {
        "bbox",
        "area_bbox",
        "slice",
    }
    | requires["moments"]
    | requires["moments_weighted"]
)

requires["num_pixels"] = {
    "area",
    "intensity_mean",
    "intensity_std",
    "num_pixels",
}

ndim_2_only = {
    "eccentricity",
    "moments_hu",
    "moments_weighted_hu",
    "orientation",
    "perimeter",
    "perimeter_crofton",
}

need_intensity_image = {
    "intensity_mean",
    "intensity_std",
    "intensity_max",
    "intensity_min",
}
need_intensity_image = need_intensity_image | requires["moments_weighted"]


def _check_moment_order(moment_order, requested_moment_props):
    if moment_order is None:
        if any(requested_moment_props | need_moments_order3):
            order = 3
        elif any(requested_moment_props | need_moments_order2):
            order = 2
        elif any(requested_moment_props | need_moments_order1):
            order = 1
        else:
            raise ValueError(
                "could not determine moment order from "
                "{requested_moment_props}"
            )
    else:
        # use user-provided moment_order
        order = moment_order
        if order < 3 and any(requested_moment_props | need_moments_order3):
            raise ValueError(
                f"can't compute {requested_moment_props} with " "moment_order<3"
            )
        if order < 2 and any(requested_moment_props | need_moments_order2):
            raise ValueError(
                f"can't compute {requested_moment_props} with " "moment_order<2"
            )
        if order < 1 and any(requested_moment_props | need_moments_order1):
            raise ValueError(
                f"can't compute {requested_moment_props} with " "moment_order<1"
            )
    return order


def regionprops_dict(
    label_image,
    intensity_image=None,
    spacing=None,
    moment_order=None,
    pixels_per_thread=32,
    max_labels_per_thread=4,
    properties=[],
):
    from cucim.skimage.measure._regionprops import PROPS

    supported_properties = set(PROPS.values())
    properties = set(properties)

    valid_names = properties & supported_properties
    invalid_names = set(properties) - valid_names
    valid_names = list(valid_names)
    for name in invalid_names:
        if name in PROPS:
            vname = PROPS[name]
            if vname in valid_names:
                raise ValueError(
                    f"Property name: {name} is a duplicate of {vname}"
                )
            else:
                valid_names.append(vname)
        else:
            raise ValueError(f"Unrecognized property name: {name}")

    requested_props = set(sorted(valid_names))

    ndim = label_image.ndim
    if ndim != 2:
        invalid_names = requested_props & ndim_2_only
        if any(invalid_names):
            raise ValueError(
                f"{label_image.ndim=}, but the following properties are for "
                "2D label images only: {invalid_names}"
            )
    if intensity_image is None:
        has_intensity = False
        invalid_names = requested_props & need_intensity_image
        if any(invalid_names):
            raise ValueError(
                "No intensity_image provided, but the following requested "
                "properties require one: {invalid_names}"
            )
    else:
        has_intensity = True

    out = {}
    max_label = int(label_image.max())
    label_dtype = _get_min_integer_dtype(max_label, signed=False)
    # For performance, shrink label's data type to the minimum possible
    # unsigned integer type.
    if label_image.dtype != label_dtype:
        label_image = label_image.astype(label_dtype)

    # create vector of label values
    if "label" in requested_props:
        out["label"] = cp.arange(1, max_label + 1, dtype=label_dtype)
        requested_props.discard("label")

    perf_kwargs = {}
    if pixels_per_thread is not None:
        perf_kwargs["pixels_per_thread"] = pixels_per_thread
    if max_labels_per_thread is not None:
        perf_kwargs["max_labels_per_thread"] = max_labels_per_thread

    requested_num_pixels_props = requested_props & requires["num_pixels"]
    compute_num_pixels = any(requested_num_pixels_props)
    if compute_num_pixels:
        regionprops_num_pixels(
            label_image,
            max_label=max_label,
            **perf_kwargs,
            props_dict=out,
        )
        requested_props.discard("num_pixels")

    if "area" in requested_props:
        regionprops_area(
            label_image,
            spacing=spacing,
            max_label=max_label,
            dtype=cp.float32,
            **perf_kwargs,
            props_dict=out,
        )
        requested_props.discard("area")

    if has_intensity:
        if "intensity_std" in requested_props:
            # std also computes mean
            regionprops_intensity_std(
                label_image,
                intensity_image,
                max_label=max_label,
                std_dtype=cp.float64,
                sample_std=False,
                **perf_kwargs,
                props_dict=out,
            )

        elif "intensity_mean" in requested_props:
            regionprops_intensity_mean(
                label_image,
                intensity_image,
                max_label=max_label,
                mean_dtype=cp.float32,
                **perf_kwargs,
                props_dict=out,
            )

        compute_min = "intensity_min" in requested_props
        compute_max = "intensity_max" in requested_props
        if compute_min or compute_max:
            _regionprops_min_or_max_intensity(
                label_image,
                intensity_image,
                max_label=max_label,
                compute_min=compute_min,
                compute_max=compute_max,
                **perf_kwargs,
                props_dict=out,
            )

    requested_bbox_props = requested_props & requires["bbox"]
    compute_bbox = any(requested_bbox_props)
    if compute_bbox:
        # compute bbox (and slice)
        regionprops_bbox_coords(
            label_image,
            max_label=max_label,
            return_slices="slice" in requested_bbox_props,
            **perf_kwargs,
            props_dict=out,
        )

        if "area_bbox" in requested_bbox_props:
            regionprops_area_bbox(
                out["bbox"],
                area_dtype=cp.float32,
                spacing=None,
                props_dict=out,
            )

    requested_unweighted_moment_props = requested_props & requires["moments"]
    compute_unweighted_moments = any(requested_unweighted_moment_props)
    requested_weighted_moment_props = (
        requested_props & requires["moments_weighted"]
    )
    compute_weighted_moments = any(requested_weighted_moment_props)

    requested_moment_props = (
        requested_unweighted_moment_props | requested_weighted_moment_props
    )  # noqa: E501
    compute_moments = any(requested_moment_props)

    requested_inertia_tensor_props = (
        requested_props & requires["inertia_tensor"]
    )
    compute_inertia_tensor = any(requested_inertia_tensor_props)

    if compute_moments:
        # determine minimum necessary order (or validate the user-provided one)
        order = _check_moment_order(moment_order, requested_moment_props)

        imgs = []
        if compute_unweighted_moments:
            imgs.append(None)
        if compute_weighted_moments:
            imgs.append(intensity_image)

        # compute raw moments (weighted and/or unweighted)
        for img in imgs:
            regionprops_moments(
                label_image,
                intensity_image=img,
                max_label=max_label,
                order=order,
                spacing=spacing,
                **perf_kwargs,
                props_dict=out,
            )

        compute_centroid_local = (
            "centroid_local" in requested_unweighted_moment_props
        )  # noqa:E501
        compute_centroid = "centroid" in requested_unweighted_moment_props
        if compute_centroid or compute_centroid_local:
            regionprops_centroid_weighted(
                moments_raw=out["moments"],
                ndim=label_image.ndim,
                bbox=out["bbox"],
                compute_local=compute_centroid_local,
                compute_global=compute_centroid,
                weighted=False,
                props_dict=out,
            )

        compute_centroid_weighted_local = (
            "centroid_weighted_local" in requested_weighted_moment_props
        )  # noqa: E501
        compute_centroid_weighted = (
            "centroid_weighted" in requested_weighted_moment_props
        )  # noqa: E501
        if compute_centroid_weighted or compute_centroid_weighted_local:
            regionprops_centroid_weighted(
                moments_raw=out["moments_weighted"],
                ndim=label_image.ndim,
                bbox=out["bbox"],
                compute_local=compute_centroid_weighted_local,
                compute_global=compute_centroid_weighted,
                weighted=True,
                props_dict=out,
            )

        if any(requested_unweighted_moment_props & requires["moments_central"]):
            regionprops_moments_central(
                out["moments"], ndim=ndim, weighted=False, props_dict=out
            )

            if any(
                requested_unweighted_moment_props
                & requires["moments_normalized"]
            ):
                regionprops_moments_normalized(
                    out["moments_central"],
                    ndim=ndim,
                    spacing=None,
                    pixel_correction=False,
                    weighted=False,
                    props_dict=out,
                )
                if "moments_hu" in requested_unweighted_moment_props:
                    regionprops_moments_hu(
                        out["moments_normalized"],
                        weighted=False,
                        props_dict=out,
                    )

        if any(
            requested_weighted_moment_props
            & requires["moments_weighted_central"]
        ):
            regionprops_moments_central(
                out["moments"], ndim, weighted=True, props_dict=out
            )

            if any(
                requested_weighted_moment_props
                & requires["moments_weighted_normalized"]
            ):
                regionprops_moments_normalized(
                    out["moments_weighted_central"],
                    ndim=ndim,
                    spacing=None,
                    pixel_correction=False,
                    weighted=True,
                    props_dict=out,
                )

                if "moments_weighted_hu" in requested_weighted_moment_props:
                    regionprops_moments_hu(
                        out["moments_weighted_normalized"],
                        weighted=True,
                        props_dict=out,
                    )

        # inertia tensor computations come after moment computations
        if compute_inertia_tensor:
            regionprops_inertia_tensor(
                out["moments_central"],
                ndim=ndim,
                compute_orientation=(
                    "orientation" in requested_inertia_tensor_props
                ),  # noqa: E501
                props_dict=out,
            )

            if "inertia_tensor_eigvals" in requested_inertia_tensor_props:
                compute_axis_lengths = (
                    "axis_minor_length" in requested_inertia_tensor_props
                    or "axis_major_length" in requested_inertia_tensor_props
                )
                regionprops_inertia_tensor_eigvals(
                    out["inertia_tensor"],
                    compute_axis_lengths=compute_axis_lengths,
                    compute_eccentricity=(
                        "eccentricity" in requested_inertia_tensor_props
                    ),  # noqa: E501
                    props_dict=out,
                )

        compute_perimeter = "perimeter" in requested_props
        compute_perimeter_crofton = "perimeter_crofton" in requested_props
        compute_euler = "euler_number" in requested_props

        if compute_euler or compute_perimeter or compute_perimeter_crofton:
            # precompute list of labels with <2 pixels space between them
            if label_image.dtype == cp.uint8:
                labels_mask = label_image.view("bool")
            else:
                labels_mask = label_image > 0
            labels_close = _find_close_labels(
                label_image, binary_image=labels_mask, max_label=max_label
            )

            if compute_perimeter:
                regionprops_perimeter(
                    label_image,
                    neighborhood=4,
                    max_label=max_label,
                    robust=True,
                    labels_close=labels_close,
                    props_dict=out,
                )
            if compute_perimeter_crofton:
                regionprops_perimeter_crofton(
                    label_image,
                    directions=4,
                    max_label=max_label,
                    robust=True,
                    omit_image_edges=False,
                    labels_close=labels_close,
                    props_dict=out,
                )
            if compute_euler:
                regionprops_euler(
                    label_image,
                    connectivity=None,
                    max_label=max_label,
                    robust=True,
                    labels_close=labels_close,
                    props_dict=out,
                )
    return out
