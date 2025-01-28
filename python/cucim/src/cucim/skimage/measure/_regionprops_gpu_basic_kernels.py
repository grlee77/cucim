import math

import cupy as cp
import numpy as np
from packaging.version import parse

__all__ = [
    "area_bbox_from_slices",
    "equivalent_diameter_area",
    "equivalent_diameter_area_2d",
    "equivalent_diameter_area_3d",
    "regionprops_area",
    "regionprops_area_bbox",
    "regionprops_bbox_coords",
    "regionprops_extent",
    "regionprops_num_pixels",
]


CUPY_GTE_13_3_0 = parse(cp.__version__) >= parse("13.3.0")

if CUPY_GTE_13_3_0:
    _includes = r"""
#include <cupy/cuda_workaround.h>  // provide std:: coverage
"""
else:
    _includes = r"""
#include <type_traits>  // let Jitify handle this
"""


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


@cp.fuse()
def equivalent_diameter_area_2d(area):
    return cp.sqrt(4.0 * area / cp.pi)


@cp.fuse()
def equivalent_diameter_area_3d(area):
    return cp.cbrt(6.0 * area / cp.pi)


@cp.fuse()
def equivalent_diameter_area(area, ndim):
    return cp.pow(2.0 * ndim * area / cp.pi, 1.0 / ndim)


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


def regionprops_extent(area, area_bbox, props_dict=None):
    extent = area / area_bbox
    if props_dict is not None:
        props_dict["extent"] = extent
    return extent
