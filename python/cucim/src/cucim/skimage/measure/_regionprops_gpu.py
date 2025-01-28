import cupy as cp

from ._regionprops_gpu_basic_kernels import (
    area_bbox_from_slices,
    equivalent_diameter_area,
    equivalent_diameter_area_2d,
    equivalent_diameter_area_3d,
    regionprops_area,
    regionprops_area_bbox,
    regionprops_bbox_coords,
    regionprops_extent,
    regionprops_num_pixels,
)
from ._regionprops_gpu_intensity_kernels import (
    regionprops_intensity_mean,
    regionprops_intensity_min_max,
    regionprops_intensity_std,
)
from ._regionprops_gpu_misc_kernels import (
    _find_close_labels,
    regionprops_euler,
    regionprops_perimeter,
    regionprops_perimeter_crofton,
)
from ._regionprops_gpu_moments_kernels import (
    regionprops_centroid,
    regionprops_centroid_local,
    regionprops_centroid_weighted,
    regionprops_inertia_tensor,
    regionprops_inertia_tensor_eigvals,
    regionprops_moments,
    regionprops_moments_central,
    regionprops_moments_hu,
    regionprops_moments_normalized,
)

__all__ = [
    "area_bbox_from_slices",
    "equivalent_diameter_area_2d",
    "equivalent_diameter_area_3d",
    "equivalent_diameter_area",
    "regionprops_area",
    "regionprops_area_bbox",
    "regionprops_bbox_coords",
    "regionprops_centroid",
    "regionprops_centroid_local",
    "regionprops_centroid_weighted",
    "regionprops_dict",
    "regionprops_euler",
    "regionprops_extent",
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
]


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
    """Produce raveled list of coordinates and label values, excluding any
    background pixels.

    Some region properties can be applied to this data format more efficiently,
    than for the original labels image. I have not yet benchmarked when it may
    be worth doing this initial step, though.
    """
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

requires["area"] = {
    "area",
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


def _check_moment_order(moment_order: set, requested_moment_props: set):
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

    if any(requested_props & requires["area"]):
        regionprops_area(
            label_image,
            spacing=spacing,
            max_label=max_label,
            dtype=cp.float32,
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

    compute_images = "image" in requested_props
    compute_intensity_images = "intensity_image" in requested_props
    if compute_intensity_images or compute_images:
        masks = tuple(
            label_image[sl] == lab
            for lab, sl in enumerate(out["slice"], start=1)
        )

        if compute_intensity_images:
            out["intensity_image"] = tuple(
                img[sl] * mask
                for lab, (sl, mask) in enumerate(
                    zip(out["slice"], masks), start=1
                )
            )

        if compute_images:
            out["image"] = tuple(
                label_image[sl] * mask
                for lab, (sl, mask) in enumerate(
                    zip(out["slice"], masks), start=1
                )
            )
    return out
