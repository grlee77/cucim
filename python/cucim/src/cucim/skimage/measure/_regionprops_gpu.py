import cupy as cp
import numpy as np

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
from ._regionprops_gpu_utils import _find_close_labels, _get_min_integer_dtype

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
]


# Master list of properties currently supported by regionprops_dict for faster
# computation on the GPU.
#
# One caveat is that centroid/moment/inertia_tensor properties currently only
# support 2D and 3D data with moments up to 3rd order.
#
# Comment lines indicate currently missing items that would be nice to support
# in the future.

PROPS_GPU = {
    "area",
    "area_bbox",
    # area_convex
    # area_filled
    "bbox",
    "coords",
    "coords_scaled",
    "axis_major_length",
    "axis_minor_length",
    "centroid",
    "centroid_local",
    "centroid_weighted",
    "centroid_weighted_local",
    "eccentricity",
    "equivalent_diameter_area",
    "euler",
    "extent",
    # feret_diameter_mx
    "image",
    # image_convex
    # image_filled
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "intensity_image",
    "intensity_mean",
    "intensity_std",
    "intensity_max",
    "intensity_min",
    "label",
    "moments",
    "moments_central",
    "moments_hu",
    "moments_normalized",
    "moments_weighted",
    "moments_weighted_central",
    "moments_weighted_hu",
    "moments_weighted_normalized",
    "num_pixels",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "slice",
    # solidity
}


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
    label_coords = cp.nonzero(labels)
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


def regionprops_image(
    label_image,
    intensity_image=None,
    max_label=None,
    compute_image=True,
    props_dict=None,
):
    """Return tuples of images of isolated label and/or intensities.

    Each image incorporates only the bounding box region for a given label.

    Length of the tuple(s) is equal to `max_label`.

    Notes
    -----
    This is provided only for completeness, but unlike for the RegionProps
    class, these are not used to compute any of the other properties.
    """
    if max_label is None:
        max_label = int(label_image.max())
    if props_dict is None:
        props_dict = dict()

    if "slice" not in props_dict:
        regionprops_bbox_coords(
            label_image,
            max_label=max_label,
            return_slices=True,
            props_dict=props_dict,
        )
    slices = props_dict["slice"]

    # mask so there will only be a single label value in each returned slice
    masks = tuple(
        label_image[sl] == lab for lab, sl in enumerate(slices, start=1)
    )

    if compute_image:
        props_dict["image"] = masks
        if intensity_image is None:
            return props_dict["image"]

    if intensity_image is not None:
        if intensity_image.ndim > label_image.ndim:
            if intensity_image.ndim != label_image.ndim + 1:
                raise ValueError(
                    "Unexpected intensity_image.ndim. Should be "
                    "label_image.ndim or label_image.ndim + 1"
                )
            imslices = tuple(sl + (slice(None),) for sl in slices)
            props_dict["intensity_image"] = tuple(
                intensity_image[sl] * mask[..., cp.newaxis]
                for img, (sl, mask) in enumerate(zip(imslices, masks), start=1)
            )
        else:
            props_dict["intensity_image"] = tuple(
                intensity_image[sl] * mask
                for img, (sl, mask) in enumerate(zip(slices, masks), start=1)
            )
        if not compute_image:
            return props_dict["intensity_image"]
    return props_dict["image"], props_dict["intensity_image"]


def regionprops_coords(
    label_image,
    max_label=None,
    spacing=None,
    compute_coords=True,
    compute_coords_scaled=False,
    props_dict=None,
):
    """Return tuple(s) of arrays of coordinates for each labeled region.

    Length of the tuple(s) is equal to `max_label`.

    Notes
    -----
    This is provided only for completeness, but unlike for the RegionProps
    class, these are not used to compute any of the other properties.
    """
    if max_label is None:
        max_label = int(label_image.max())
    if props_dict is None:
        props_dict = dict()

    coords_concat, _ = get_compressed_labels(
        label_image, max_label=max_label, sort_labels=True
    )

    if "num_pixels" not in props_dict:
        num_pixels = regionprops_num_pixels(
            label_image, max_label=max_label, props_dict=props_dict
        )
    else:
        num_pixels = props_dict["num_pixels"]

    # stack ndim arrays into a single (pixels, ndim) array
    coords_concat = cp.stack(coords_concat, axis=-1)

    # scale based on spacing
    if compute_coords_scaled:
        max_exact_float32_int = 16777216  # 2 ** 24
        max_sz = max(label_image.shape)
        float_type = (
            cp.float32 if max_sz < max_exact_float32_int else cp.float64
        )
        coords_concat_scaled = coords_concat.astype(float_type)
        if spacing is not None:
            scale_factor = cp.asarray(spacing, dtype=float_type)[cp.newaxis, :]
            coords_concat_scaled *= scale_factor
        coords_scaled = []

    if compute_coords:
        coords = []

    # split separate labels out of the concatenated array above
    num_pixels_cpu = cp.asnumpy(num_pixels)
    slice_start = 0
    slice_stops = np.cumsum(num_pixels_cpu)
    for slice_stop in slice_stops:
        sl = slice(slice_start, slice_stop)
        if compute_coords:
            coords.append(coords_concat[sl, :])
        if compute_coords_scaled:
            coords_scaled.append(coords_concat_scaled[sl, :])
        slice_start = slice_stop
    coords = tuple(coords)
    if compute_coords:
        props_dict["coords"] = coords
        if not compute_coords_scaled:
            return coords
    if compute_coords_scaled:
        coords_scaled = tuple(coords_scaled)
        props_dict["coords_scaled"] = coords_scaled
        if not compute_coords:
            return coords_scaled
    return coords, coords_scaled


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
    "coords",
    "coords_scaled",
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
        regionprops_image(
            label_image,
            intensity_image=intensity_image
            if compute_intensity_images
            else None,  # noqa: E501
            max_label=max_label,
            props_dict=out,
            compute_image=compute_images,
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
    return out
