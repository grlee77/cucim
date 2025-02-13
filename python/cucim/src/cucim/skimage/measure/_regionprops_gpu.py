import warnings
from copy import copy

import cupy as cp
import numpy as np

from cucim.skimage.measure._regionprops import COL_DTYPES, PROPS

from ._regionprops_gpu_basic_kernels import (
    area_bbox_from_slices,
    equivalent_diameter_area,
    equivalent_diameter_area_2d,
    equivalent_diameter_area_3d,
    regionprops_area,
    regionprops_area_bbox,
    regionprops_bbox_coords,
    regionprops_boundary_mask,
    regionprops_coords,
    regionprops_extent,
    regionprops_image,
    regionprops_label_filled,
    regionprops_num_boundary_pixels,
    regionprops_num_pixels,
    regionprops_num_pixels_perimeter,
)
from ._regionprops_gpu_convex import (
    regionprops_area_convex,
    regionprops_feret_diameter_max,
)
from ._regionprops_gpu_intensity_kernels import (
    regionprops_intensity_mean,
    regionprops_intensity_min_max,
    regionprops_intensity_std,
)
from ._regionprops_gpu_misc_kernels import (
    regionprops_euler,
    regionprops_perimeter,
    regionprops_perimeter_crofton,
)
from ._regionprops_gpu_moments_kernels import (
    _check_moment_order,
    regionprops_centroid,
    regionprops_centroid_local,
    regionprops_centroid_weighted,
    regionprops_inertia_tensor,
    regionprops_inertia_tensor_eigvals,
    regionprops_moments,
    regionprops_moments_central,
    regionprops_moments_hu,
    regionprops_moments_normalized,
    requires as moment_requirements,
)
from ._regionprops_gpu_utils import _find_close_labels, _get_min_integer_dtype

__all__ = [
    "area_bbox_from_slices",
    "equivalent_diameter_area_2d",
    "equivalent_diameter_area_3d",
    "equivalent_diameter_area",
    "regionprops_area",
    "regionprops_area_bbox",
    "regionprops_area_convex",
    "regionprops_bbox_coords",
    "regionprops_centroid",
    "regionprops_centroid_local",
    "regionprops_centroid_weighted",
    "regionprops_coords",
    "regionprops_dict",
    "regionprops_euler",
    "regionprops_extent",
    "regionprops_image",
    "regionprops_inertia_tensor",
    "regionprops_inertia_tensor_eigvals",
    "regionprops_intensity_mean",
    "regionprops_intensity_min_max",
    "regionprops_intensity_std",
    "regionprops_moments",
    "regionprops_moments_central",
    "regionprops_moments_hu",
    "regionprops_moments_normalized",
    "regionprops_num_pixels",
    "regionprops_perimeter",
    "regionprops_perimeter_crofton",
    # extra functions for cuCIM not currently in scikit-image
    "regionprops_boundary_mask",
    "regionprops_num_boundary_pixels",
    "regionprops_num_pixels_perimeter",
    "regionprops_label_filled",
]


# Master list of properties currently supported by regionprops_dict for faster
# computation on the GPU.
#
# One caveat is that centroid/moment/inertia_tensor properties currently only
# support 2D and 3D data with moments up to 3rd order.

# all properties from PROPS have been implemented
PROPS_GPU = copy(PROPS)
# extra properties not currently in scikit-image
PROPS_GPU_EXTRA = {
    "axis_lengths": "axis_lengths",
    "inertia_tensor_eigenvectors": "inertia_tensor_eigenvectors",
    "num_pixels_perimeter": "num_pixels_perimeter",
    "num_boundary_pixels": "num_boundary_pixels",
}
PROPS_GPU.update(PROPS_GPU_EXTRA)

CURRENT_PROPS_GPU = set(PROPS_GPU.values())

COL_DTYPES_EXTRA = {
    "axis_lengths": float,
    "inertia_tensor_eigenvectors": float,
    "num_pixels_perimeter": int,
    "num_boundary_pixels": int,
}
COL_DTYPES_GPU = copy(COL_DTYPES)
COL_DTYPES_GPU.update(COL_DTYPES_EXTRA)

# There is also "label_filled" but this is a global filled labels image

OBJECT_COLUMNS_GPU = [
    col for col, dtype in COL_DTYPES_GPU.items() if dtype == object
]

# requires dictionary has key value pairs where the values for a given key
# list the properties that require that key in order to compute.

requires = copy(moment_requirements)

# set of properties that require an intensity image
need_intensity_image = {
    "intensity_mean",
    "intensity_std",
    "intensity_max",
    "intensity_min",
}
need_intensity_image = need_intensity_image | requires["moments_weighted"]

requires["image_convex"] = {
    "area_convex",
    "feret_diameter_max" "image_convex",
    "solidity",
}

requires["area_convex"] = {
    "area_convex",
    "solidity",
}

requires["area"] = {
    "area",
    "solidity",
    "extent",
    "equivalent_diameter_area",
}

requires["area_bbox"] = {
    "area_bbox",
    "extent",
}

requires["bbox"] = (
    {
        "bbox",
        "slice",
    }
    | requires["area_bbox"]
    | requires["moments"]
    | requires["moments_weighted"]
)

requires["num_pixels"] = {
    "area",
    "coords",
    "coords_scaled",
    "intensity_mean",
    "intensity_std",
    "num_pixels",
}

requires["label_filled"] = {
    "area_filled",
    "image_filled",
    "label_filled",
}

# set of properties that can only be computed for 2D regions
ndim_2_only = {
    "eccentricity",
    "moments_hu",
    "moments_weighted_hu",
    "orientation",
    "perimeter",
    "perimeter_crofton",
}


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

    supported_properties = CURRENT_PROPS_GPU
    properties = set(properties)

    valid_names = properties & supported_properties
    invalid_names = set(properties) - valid_names
    valid_names = list(valid_names)

    # Use only the modern names internally, but keep list of mappings back to
    # any deprecated names in restore_legacy_names and use that at the end to
    # restore the requested deprecated property names.
    restore_legacy_names = dict()
    for name in invalid_names:
        if name in PROPS:
            vname = PROPS[name]
            if vname in valid_names:
                raise ValueError(
                    f"Property name: {name} is a duplicate of {vname}"
                )
            else:
                restore_legacy_names[vname] = name
                valid_names.append(vname)
        else:
            raise ValueError(f"Unrecognized property name: {name}")
    for v in restore_legacy_names.values():
        invalid_names.discard(v)
    # warn if there are any names that did not match a deprecated name
    if invalid_names:
        warnings.warn(
            "The following property names were unrecognized and will not be "
            "computed: {invalid_names}"
        )

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

    if any(requested_props & requires["area"]):
        regionprops_area(
            label_image,
            spacing=spacing,
            max_label=max_label,
            dtype=cp.float32,
            filled=False,
            **perf_kwargs,
            props_dict=out,
        )

        if "equivalent_diameter_area" in requested_props:
            if ndim == 2:
                ed = equivalent_diameter_area_2d(out["area"])
            elif ndim == 3:
                ed = equivalent_diameter_area_3d(out["area"])
            else:
                ed = equivalent_diameter_area(out["area"], float(ndim))
            out["equivalent_diameter_area"] = ed

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
            regionprops_intensity_min_max(
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

        if any(requested_props & requires["area_bbox"]):
            regionprops_area_bbox(
                out["bbox"],
                area_dtype=cp.float32,
                spacing=None,
                props_dict=out,
            )

        if "extent" in requested_props:
            out["extent"] = out["area"] / out["area_bbox"]

    if "num_boundary_pixels" in requested_props:
        regionprops_num_boundary_pixels(
            label_image,
            max_label=max_label,
            props_dict=out,
        )

    if "num_pixels_perimeter" in requested_props:
        regionprops_num_pixels_perimeter(
            label_image,
            max_label=max_label,
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

            if any(
                requires["inertia_tensor_eigvals"]
                & requested_inertia_tensor_props
            ):
                compute_axis_lengths = (
                    "axis_minor_length" in requested_inertia_tensor_props
                    or "axis_major_length" in requested_inertia_tensor_props
                )
                regionprops_inertia_tensor_eigvals(
                    out["inertia_tensor"],
                    compute_axis_lengths=compute_axis_lengths,
                    compute_eccentricity=(
                        "eccentricity" in requested_inertia_tensor_props
                    ),
                    compute_eigenvectors=(
                        "inertia_tensor_eigenvectors"
                        in requested_inertia_tensor_props
                    ),
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

    compute_images = "image" in requested_props
    compute_intensity_images = "image_intensity" in requested_props
    compute_convex = any(requires["image_convex"] & requested_props)
    if compute_intensity_images or compute_images or compute_convex:
        regionprops_image(
            label_image,
            intensity_image=intensity_image
            if compute_intensity_images
            else None,  # noqa: E501
            max_label=max_label,
            props_dict=out,
            compute_image=compute_images,
            compute_convex=compute_convex,
        )

    compute_area_convex = any(requires["area_convex"] & requested_props)
    if compute_area_convex:
        regionprops_area_convex(
            out["image_convex"], max_label=max_label, props_dict=out
        )

    compute_solidity = "solidity" in requested_props
    if compute_solidity:
        out["solidity"] = out["area"] / out["area_convex"]

    compute_feret_diameter_max = "feret_diameter_max" in requested_props
    if compute_feret_diameter_max:
        regionprops_feret_diameter_max(
            out["image_convex"],
            spacing=spacing,
            props_dict=out,
        )

    compute_coords = "coords" in requested_props
    compute_coords_scaled = "coords_scaled" in requested_props
    if compute_coords or compute_coords_scaled:
        regionprops_coords(
            label_image,
            max_label=max_label,
            spacing=spacing,
            compute_coords=compute_coords,
            compute_coords_scaled=compute_coords_scaled,
            props_dict=out,
        )

    compute_label_filled = any(requires["label_filled"] & requested_props)
    if compute_label_filled:
        regionprops_label_filled(
            label_image,
            max_label=max_label,
            props_dict=out,
        )
        if "area_filled" in requested_props:
            out["area_filled"] = regionprops_area(
                out["label_filled"],
                max_label=max_label,
                filled=True,
                props_dict=out,
            )
        if "image_filled" in requested_props:
            out["image_filled"], _, _ = regionprops_image(
                out["label_filled"],
                max_label=max_label,
                compute_image=True,
                compute_convex=False,
                props_dict=None,  # omit: using custom "image_filled" key
            )

    # If user had requested properties via their deprecated names, set the
    # canonical names for the computed properties to the corresponding
    # deprecated one.
    for k, v in restore_legacy_names.items():
        out[v] = out.pop(k)
    return out


def _props_dict_to_table(
    props_dict, properties, separator="-", copy_to_host=False
):
    out = {}
    for prop in properties:
        # Copy the original property name so the output will have the
        # user-provided property name in the case of deprecated names.
        orig_prop = prop
        # determine the current property name for any deprecated property.
        prop = PROPS_GPU.get(prop, prop)
        dtype = COL_DTYPES_GPU[
            prop
        ]  # TODO: also update for GPU-only properties?

        # is_0dim_array = isinstance(rp, cp.ndarray) and rp.ndim == 0
        rp = props_dict[orig_prop]

        is_scalar_prop = False
        is_multicolumn = False
        if isinstance(rp, cp.ndarray):
            is_scalar_prop = rp.ndim == 1
            is_multicolumn = not is_scalar_prop
        if is_scalar_prop:
            if copy_to_host:
                rp = cp.asnumpy(rp)
            out[orig_prop] = rp
            print(f"type({prop}) = 'scalar'")
        elif is_multicolumn:
            if copy_to_host:
                rp = cp.asnumpy(rp)
            shape = rp.shape[1:]
            # precompute property column names and locations
            modified_props = []
            locs = []
            for ind in np.ndindex(shape):
                modified_props.append(
                    separator.join(map(str, (orig_prop,) + ind))
                )
                locs.append((slice(None),) + ind)
            for i, modified_prop in enumerate(modified_props):
                out[modified_prop] = rp[locs[i]]
            print(f"type({prop}) = 'multi-column'")
        elif prop in OBJECT_COLUMNS_GPU:
            print(f"type({prop}) = 'object'")
            n = len(rp)
            # keep objects in a NumPy array
            column_buffer = np.empty(n, dtype=dtype)
            if copy_to_host:
                for i in range(n):
                    column_buffer[i] = cp.asnumpy(rp[i])
                out[orig_prop] = column_buffer
            else:
                for i in range(n):
                    column_buffer[i] = rp[i]
                out[orig_prop] = np.copy(column_buffer)
        else:
            warnings.warn(
                f"Type unknown for property: {prop}, storing it as-is."
            )
            out[orig_prop] = rp
    return out
