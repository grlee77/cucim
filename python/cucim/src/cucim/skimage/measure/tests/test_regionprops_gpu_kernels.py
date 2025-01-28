import functools
import math

import cupy as cp
import pytest
from cupy.testing import (
    assert_allclose,
    assert_array_equal,
)
from scipy.ndimage import find_objects as cpu_find_objects
from skimage import measure as measure_cpu

from cucim.skimage import data, measure
from cucim.skimage.measure._regionprops_gpu import (
    _find_close_labels,
    area_bbox_from_slices,
    regionprops_area,
    regionprops_area_bbox,
    regionprops_bbox_coords,
    regionprops_centroid,
    regionprops_centroid_local,
    regionprops_centroid_weighted,
    regionprops_euler,
    regionprops_inertia_tensor,
    regionprops_inertia_tensor_eigvals,
    regionprops_intensity_max,
    regionprops_intensity_mean,
    regionprops_intensity_min,
    regionprops_intensity_std,
    regionprops_moments,
    regionprops_moments_central,
    regionprops_moments_hu,
    regionprops_moments_normalized,
    regionprops_num_pixels,
    regionprops_perimeter,
    regionprops_perimeter_crofton,
)


def get_labels_nd(
    shape,
    blob_size_fraction=0.05,
    volume_fraction=0.25,
    rng=5,
    insert_holes=False,
):
    ndim = len(shape)
    blobs_kwargs = dict(
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
        rng=rng,
    )
    blobs = data.binary_blobs(max(shape), n_dim=ndim, **blobs_kwargs)
    # crop to rectangular
    blobs = blobs[tuple(slice(s) for s in shape)]

    if insert_holes:
        blobs2_kwargs = dict(
            blob_size_fraction=blob_size_fraction / 5,
            volume_fraction=0.1,
            rng=rng,
        )
        # create smaller blobs and invert them to create a holes mask to apply
        # to the original blobs
        temp = data.binary_blobs(max(shape), n_dim=ndim, **blobs2_kwargs)
        temp = temp[tuple(slice(s) for s in shape)]
        mask = cp.logical_and(blobs > 0, temp == 0)
        blobs = blobs * mask

    # binary blobs only creates square outputs
    labels = measure.label(blobs)
    # print(f"# labels generated = {labels.max()}")
    return labels


def get_intensity_image(shape, dtype=cp.float32, seed=5, num_channels=None):
    npixels = math.prod(shape)
    rng = cp.random.default_rng(seed)
    dtype = cp.dtype(dtype)
    if dtype.kind == "f":
        img = cp.arange(npixels, dtype=cp.float32) - npixels // 2
        img = img.reshape(shape)
        if dtype == cp.float16:
            temp = 100 * rng.standard_normal(img.shape, dtype=cp.float32)
            img += temp.astype(cp.float16)
        else:
            img += 100 * rng.standard_normal(img.shape, dtype=dtype)
    else:
        iinfo = cp.iinfo(dtype)
        imax = min(16384, iinfo.max)
        imin = max(0, iinfo.min)
        img = rng.integers(imin, imax, shape)

    if num_channels and num_channels > 1:
        # generate slightly shifted versions for the additional channels
        img = cp.stack((img,) * num_channels, axis=-1)
        for c in range(1, num_channels):
            img[..., c] = cp.roll(img[..., c], shift=c, axis=0)
    return img


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
def test_num_pixels(precompute_max, ndim):
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    labels = get_labels_nd(shape)

    max_label = int(cp.max(labels)) if precompute_max else None
    num_pixels = regionprops_num_pixels(labels, max_label=max_label)
    expected = measure.regionprops_table(labels, properties=["num_pixels"])
    assert_allclose(num_pixels, expected["num_pixels"])


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("area_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("spacing", [None, (0.5, 0.35, 0.75)])
def test_area(precompute_max, ndim, area_dtype, spacing):
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    labels = get_labels_nd(shape)
    if spacing is not None:
        spacing = spacing[:ndim]

    max_label = int(cp.max(labels)) if precompute_max else None
    area = regionprops_area(
        labels, spacing=spacing, max_label=max_label, dtype=area_dtype
    )
    expected = measure.regionprops_table(
        labels, spacing=spacing, properties=["area"]
    )
    assert_allclose(area, expected["area"])


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("image_dtype", [cp.uint16, cp.uint8, cp.float32])
@pytest.mark.parametrize("mean_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("num_channels", [1, 4])
def test_mean_intensity(
    precompute_max, ndim, image_dtype, mean_dtype, num_channels
):
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    labels = get_labels_nd(shape)
    intensity_image = get_intensity_image(
        shape, dtype=image_dtype, num_channels=num_channels
    )

    max_label = int(cp.max(labels)) if precompute_max else None
    counts, means = regionprops_intensity_mean(
        labels, intensity_image, max_label=max_label, mean_dtype=mean_dtype
    )
    expected = measure.regionprops_table(
        labels,
        intensity_image=intensity_image,
        properties=["num_pixels", "intensity_mean"],
    )
    assert_array_equal(counts, expected["num_pixels"])
    if num_channels == 1:
        assert_allclose(means, expected["intensity_mean"], rtol=1e-6)
    else:
        for c in range(num_channels):
            assert_allclose(
                means[..., c], expected[f"intensity_mean-{c}"], rtol=1e-6
            )


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "image_dtype", [cp.uint16, cp.uint8, cp.float16, cp.float32, cp.float64]
)
@pytest.mark.parametrize("op_name", ["intensity_min", "intensity_max"])
@pytest.mark.parametrize("num_channels", [1, 3])
def test_intensity_min_and_max(
    precompute_max, ndim, image_dtype, op_name, num_channels
):
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    labels = get_labels_nd(shape)
    intensity_image = get_intensity_image(
        shape, dtype=image_dtype, num_channels=num_channels
    )

    max_label = int(cp.max(labels)) if precompute_max else None

    func = (
        regionprops_intensity_min
        if op_name == "intensity_min"
        else regionprops_intensity_max
    )
    values = func(labels, intensity_image, max_label=max_label)
    expected = measure.regionprops_table(
        labels, intensity_image=intensity_image, properties=[op_name]
    )
    if num_channels == 1:
        assert_array_equal(values, expected[op_name])
    else:
        for c in range(num_channels):
            assert_array_equal(values[..., c], expected[f"{op_name}-{c}"])


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("image_dtype", [cp.uint16, cp.uint8, cp.float32])
@pytest.mark.parametrize("std_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("num_channels", [1, 5])
def test_intensity_std(
    precompute_max, ndim, image_dtype, std_dtype, num_channels
):
    shape = (1024, 2048) if ndim == 2 else (40, 64, 80)
    labels = get_labels_nd(shape)
    intensity_image = get_intensity_image(
        shape, dtype=image_dtype, num_channels=num_channels
    )

    max_label = int(cp.max(labels)) if precompute_max else None

    # add some specifically sized regions
    if ndim == 2 and precompute_max:
        # clear small region
        labels[50:54, 50:56] = 0
        # add a single pixel labeled region
        labels[51, 51] = max_label + 1
        # add a two pixel labeled region
        labels[53, 53:55] = max_label + 2
        max_label += 2

    counts, means, stds = regionprops_intensity_std(
        labels, intensity_image, max_label=max_label, std_dtype=std_dtype
    )
    expected = measure.regionprops_table(
        labels,
        intensity_image=intensity_image,
        properties=["num_pixels", "intensity_mean", "intensity_std"],
    )
    assert_array_equal(counts, expected["num_pixels"])
    if num_channels == 1:
        assert_allclose(means, expected["intensity_mean"], rtol=1e-5)
        assert_allclose(stds, expected["intensity_std"], rtol=1e-4)
    else:
        for c in range(num_channels):
            assert_allclose(
                means[..., c], expected[f"intensity_mean-{c}"], rtol=1e-5
            )
            assert_allclose(
                stds[..., c], expected[f"intensity_std-{c}"], rtol=1e-4
            )


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("dtype", [cp.uint32, cp.int64])
@pytest.mark.parametrize("return_slices", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
def test_bbox_coords_and_area(precompute_max, ndim, dtype, return_slices):
    shape = (1024, 512) if ndim == 2 else (80, 64, 48)
    labels = get_labels_nd(shape)

    max_label = int(cp.max(labels)) if precompute_max else None
    bbox, slices = regionprops_bbox_coords(
        labels,
        max_label=max_label,
        return_slices=return_slices,
    )
    assert bbox.dtype == cp.uint32
    if not return_slices:
        slices is None
    else:
        expected_slices = cpu_find_objects(cp.asnumpy(labels))
        assert slices == expected_slices

    spacing = (0.35, 0.75, 0.5)[:ndim]
    expected_bbox = measure.regionprops_table(
        labels, spacing=spacing, properties=["bbox", "area_bbox"]
    )
    if ndim == 2:
        # TODO make ordering of bbox consistent with regionprops bbox?
        assert_allclose(bbox[:, 0], expected_bbox["bbox-0"])
        assert_allclose(bbox[:, 2], expected_bbox["bbox-1"])
        assert_allclose(bbox[:, 1], expected_bbox["bbox-2"])
        assert_allclose(bbox[:, 3], expected_bbox["bbox-3"])
    elif ndim == 3:
        assert_allclose(bbox[:, 0], expected_bbox["bbox-0"])
        assert_allclose(bbox[:, 2], expected_bbox["bbox-1"])
        assert_allclose(bbox[:, 4], expected_bbox["bbox-2"])
        assert_allclose(bbox[:, 1], expected_bbox["bbox-3"])
        assert_allclose(bbox[:, 3], expected_bbox["bbox-4"])
        assert_allclose(bbox[:, 5], expected_bbox["bbox-5"])

    # compute area_bbox from bbox array
    area_bbox = regionprops_area_bbox(
        bbox, area_dtype=cp.float32, spacing=spacing
    )
    assert_allclose(area_bbox, expected_bbox["area_bbox"], rtol=1e-5)

    if return_slices:
        # compute area_bbox from slices
        area_bbox = area_bbox_from_slices(
            slices, area_dtype=cp.float32, spacing=spacing
        )
        assert_allclose(area_bbox, expected_bbox["area_bbox"], rtol=1e-5)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("via_moments", [False, True])
def test_centroid(via_moments, local, ndim):
    shape = (1024, 512) if ndim == 2 else (80, 64, 48)
    labels = get_labels_nd(shape)
    max_label = int(cp.max(labels))
    if via_moments:
        props = {}
        moments = regionprops_moments(
            labels, max_label=max_label, order=1, props_dict=props
        )
        assert "bbox" in props
        assert "moments" in props
    if local:
        name = "centroid_local"
        if via_moments:
            centroid = regionprops_centroid_weighted(
                moments_raw=moments,
                ndim=labels.ndim,
                bbox=props["bbox"],
                compute_local=True,
                compute_global=False,
                weighted=False,
                props_dict=props,
            )
            assert "centroid_local" in props
        else:
            centroid = regionprops_centroid_local(labels, max_label=max_label)
    else:
        name = "centroid"
        if via_moments:
            centroid = regionprops_centroid_weighted(
                moments_raw=moments,
                ndim=labels.ndim,
                bbox=props["bbox"],
                compute_local=False,
                compute_global=True,
                weighted=False,
                props_dict=props,
            )
            assert "centroid" in props
        else:
            centroid = regionprops_centroid(labels, max_label=max_label)
    expected = measure.regionprops_table(labels, properties=[name])
    assert_allclose(centroid[:, 0], expected[name + "-0"])
    if ndim > 1:
        assert_allclose(centroid[:, 1], expected[name + "-1"])
    if ndim > 2:
        assert_allclose(centroid[:, 2], expected[name + "-2"])


@pytest.mark.parametrize("spacing", [None, (0.8, 0.5), (0.2, 1.3)])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "weighted, intensity_dtype, num_channels",
    [
        (False, None, 1),
        (True, cp.float32, 1),
        (True, cp.uint8, 3),
    ],
)
@pytest.mark.parametrize("norm_type", ["raw", "central", "normalized", "hu"])
@pytest.mark.parametrize("blob_size_fraction", [0.03, 0.1, 0.3])
def test_moments_2d(
    spacing,
    order,
    weighted,
    intensity_dtype,
    num_channels,
    norm_type,
    blob_size_fraction,
):
    shape = (800, 600)
    labels = get_labels_nd(shape, blob_size_fraction=blob_size_fraction)
    max_label = int(cp.max(labels))
    kwargs = {"spacing": spacing}
    prop = "moments"
    if norm_type == "hu":
        if order != 3:
            pytest.skip("Hu moments require order = 3")
        elif spacing and spacing != (1.0, 1.0):
            pytest.skip("Hu moments only support spacing = (1.0, 1.0)")
    if norm_type == "normalized" and order < 2:
        pytest.skip("normalized case only supports order >=2")
    if weighted:
        intensity_image = get_intensity_image(
            shape, dtype=intensity_dtype, num_channels=num_channels
        )
        kwargs["intensity_image"] = intensity_image
        prop += "_weighted"
    if norm_type == "central":
        prop += "_central"
    elif norm_type == "normalized":
        prop += "_normalized"
    elif norm_type == "hu":
        prop += "_hu"
    expected = measure.regionprops_table(labels, properties=[prop], **kwargs)
    moments = regionprops_moments(
        labels, max_label=max_label, order=order, **kwargs
    )
    if norm_type in ["central", "normalized", "hu"]:
        ndim = len(shape)
        moments = regionprops_moments_central(moments, ndim=ndim)
        if norm_type in ["normalized", "hu"]:
            moments = regionprops_moments_normalized(
                moments, ndim=ndim, spacing=spacing
            )
            if norm_type == "normalized":
                # assert that np.nan values were set for non-computed orders
                orders = cp.arange(order + 1)[:, cp.newaxis]
                orders = orders + orders.T
                mask = cp.logical_and(orders < 1, orders > order)
                # prepend labels (and channels) axes
                if num_channels > 1:
                    mask = mask[cp.newaxis, cp.newaxis, ...]
                    mask = cp.tile(mask, moments.shape[:2] + (1, 1))
                else:
                    mask = mask[cp.newaxis, ...]
                    mask = cp.tile(mask, moments.shape[:1] + (1, 1))
                assert cp.all(cp.isnan(moments[mask]))

            if norm_type == "hu":
                moments = regionprops_moments_hu(moments)
                assert moments.shape[-1] == 7

    # regionprops does not use the more accurate analytical expressions for the
    # central moments, so need to relax tolerance in the "central" moments case
    rtol = 1e-4 if norm_type != "raw" else 1e-5
    atol = 1e-4 if norm_type != "raw" else 1e-7
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    if norm_type == "hu":
        # hu moments are stored as a 7-element vector
        if num_channels == 1:
            for d in range(7):
                allclose(moments[:, d], expected[prop + f"-{d}"])
        else:
            for c in range(num_channels):
                for d in range(7):
                    allclose(moments[:, c, d], expected[prop + f"-{d}-{c}"])
    else:
        # All other moment types produce a (order + 1, order + 1) matrix
        if num_channels == 1:
            # zeroth moment
            allclose(moments[:, 0, 0], expected[prop + "-0-0"])

            if order > 0 and norm_type != "normalized":
                # first-order moments
                if norm_type == "central":
                    assert_array_equal(moments[:, 0, 1], 0.0)
                    assert_array_equal(moments[:, 1, 0], 0.0)
                else:
                    allclose(moments[:, 0, 1], expected[prop + "-0-1"])
                    allclose(moments[:, 1, 0], expected[prop + "-1-0"])
            if order > 1:
                # second-order moments
                allclose(moments[:, 0, 2], expected[prop + "-0-2"])
                allclose(moments[:, 1, 1], expected[prop + "-1-1"])
                allclose(moments[:, 2, 0], expected[prop + "-2-0"])
            if order > 3:
                # third-order moments
                allclose(moments[:, 0, 3], expected[prop + "-0-3"])
                allclose(moments[:, 1, 2], expected[prop + "-1-2"])
                allclose(moments[:, 2, 1], expected[prop + "-2-1"])
                allclose(moments[:, 3, 0], expected[prop + "-3-0"])
        else:
            for c in range(num_channels):
                # zeroth moment
                allclose(moments[:, c, 0, 0], expected[prop + f"-0-0-{c}"])

            if order > 0 and norm_type != "normalized":
                # first-order moments
                if norm_type == "central":
                    assert_array_equal(moments[:, c, 0, 1], 0.0)
                    assert_array_equal(moments[:, c, 1, 0], 0.0)
                else:
                    allclose(moments[:, c, 0, 1], expected[prop + f"-0-1-{c}"])
                    allclose(moments[:, c, 1, 0], expected[prop + f"-1-0-{c}"])
            if order > 1:
                # second-order moments
                allclose(moments[:, c, 0, 2], expected[prop + f"-0-2-{c}"])
                allclose(moments[:, c, 1, 1], expected[prop + f"-1-1-{c}"])
                allclose(moments[:, c, 2, 0], expected[prop + f"-2-0-{c}"])
            if order > 3:
                # third-order moments
                allclose(moments[:, c, 0, 3], expected[prop + f"-0-3-{c}"])
                allclose(moments[:, c, 1, 2], expected[prop + f"-1-2-{c}"])
                allclose(moments[:, c, 2, 1], expected[prop + f"-2-1-{c}"])
                allclose(moments[:, c, 3, 0], expected[prop + f"-3-0-{c}"])


@pytest.mark.parametrize("spacing", [None, (0.8, 0.5, 0.75)])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "weighted, intensity_dtype, num_channels",
    [
        (False, None, 1),
        (True, cp.float32, 1),
        (True, cp.uint8, 3),
    ],
)
@pytest.mark.parametrize("norm_type", ["raw", "central", "normalized"])
def test_moments_3d(
    spacing, order, weighted, intensity_dtype, num_channels, norm_type
):
    shape = (80, 64, 48)
    labels = get_labels_nd(shape)
    max_label = int(cp.max(labels))
    kwargs = {"spacing": spacing}
    prop = "moments"
    if norm_type == "normalized" and order < 2:
        pytest.skip("normalized case only supports order >=2")
    if weighted:
        intensity_image = get_intensity_image(
            shape, dtype=intensity_dtype, num_channels=num_channels
        )
        kwargs["intensity_image"] = intensity_image
        prop += "_weighted"
    if norm_type == "central":
        prop += "_central"
    elif norm_type == "normalized":
        prop += "_normalized"
    expected = measure.regionprops_table(labels, properties=[prop], **kwargs)
    moments = regionprops_moments(
        labels, max_label=max_label, order=order, **kwargs
    )
    if norm_type in ["central", "normalized"]:
        ndim = len(shape)
        moments = regionprops_moments_central(moments, ndim=ndim)
        if norm_type == "normalized":
            moments = regionprops_moments_normalized(
                moments, ndim=ndim, spacing=spacing
            )

            # assert that np.nan values were set for non-computed orders
            orders = cp.arange(order + 1)
            orders = (
                orders[:, cp.newaxis, cp.newaxis]
                + orders[cp.newaxis, :, cp.newaxis]
                + orders[cp.newaxis, cp.newaxis, :]
            )
            mask = cp.logical_and(orders < 1, orders > order)
            # prepend labels (and channels) axes and replicate mask to match
            # the moments shape
            if num_channels > 1:
                mask = mask[cp.newaxis, cp.newaxis, ...]
                mask = cp.tile(mask, moments.shape[:2] + (1, 1, 1))
            else:
                mask = mask[cp.newaxis, ...]
                mask = cp.tile(mask, moments.shape[:1] + (1, 1, 1))
            assert cp.all(cp.isnan(moments[mask]))

    # regionprops does not use the more accurate analytical expressions for the
    # central moments, so need to relax tolerance in the "central" moments case
    rtol = 1e-3 if norm_type != "raw" else 1e-5
    atol = 1e-4 if norm_type != "raw" else 1e-7
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    if num_channels == 1:
        # zeroth moment
        allclose(moments[:, 0, 0, 0], expected[prop + "-0-0-0"])
        if order > 0 and norm_type != "normalized":
            # first-order moments
            if norm_type == "central":
                assert_array_equal(moments[:, 0, 0, 1], 0.0)
                assert_array_equal(moments[:, 0, 1, 0], 0.0)
                assert_array_equal(moments[:, 1, 0, 0], 0.0)
            else:
                allclose(moments[:, 0, 0, 1], expected[prop + "-0-0-1"])
                allclose(moments[:, 0, 1, 0], expected[prop + "-0-1-0"])
                allclose(moments[:, 1, 0, 0], expected[prop + "-1-0-0"])
        if order > 1:
            # second-order moments
            allclose(moments[:, 0, 0, 2], expected[prop + "-0-0-2"])
            allclose(moments[:, 0, 2, 0], expected[prop + "-0-2-0"])
            allclose(moments[:, 2, 0, 0], expected[prop + "-2-0-0"])
            allclose(moments[:, 1, 1, 0], expected[prop + "-1-1-0"])
            allclose(moments[:, 1, 0, 1], expected[prop + "-1-0-1"])
            allclose(moments[:, 0, 1, 1], expected[prop + "-0-1-1"])
        if order > 2:
            # third-order moments
            allclose(moments[:, 0, 0, 3], expected[prop + "-0-0-3"])
            allclose(moments[:, 0, 3, 0], expected[prop + "-0-3-0"])
            allclose(moments[:, 3, 0, 0], expected[prop + "-3-0-0"])
            allclose(moments[:, 1, 2, 0], expected[prop + "-1-2-0"])
            allclose(moments[:, 2, 1, 0], expected[prop + "-2-1-0"])
            allclose(moments[:, 1, 0, 2], expected[prop + "-1-0-2"])
            allclose(moments[:, 2, 0, 1], expected[prop + "-2-0-1"])
            allclose(moments[:, 0, 1, 2], expected[prop + "-0-1-2"])
            allclose(moments[:, 0, 2, 1], expected[prop + "-0-2-1"])
            allclose(moments[:, 1, 1, 1], expected[prop + "-1-1-1"])
    else:
        for c in range(num_channels):
            # zeroth moment
            allclose(moments[:, c, 0, 0, 0], expected[prop + f"-0-0-0-{c}"])
            if order > 0 and norm_type != "normalized":
                # first-order moments
                if norm_type == "central":
                    assert_array_equal(moments[:, c, 0, 0, 1], 0.0)
                    assert_array_equal(moments[:, c, 0, 1, 0], 0.0)
                    assert_array_equal(moments[:, c, 1, 0, 0], 0.0)
                else:
                    allclose(
                        moments[:, c, 0, 0, 1], expected[prop + f"-0-0-1-{c}"]
                    )
                    allclose(
                        moments[:, c, 0, 1, 0], expected[prop + f"-0-1-0-{c}"]
                    )
                    allclose(
                        moments[:, c, 1, 0, 0], expected[prop + f"-1-0-0-{c}"]
                    )
            if order > 1:
                # second-order moments
                allclose(moments[:, c, 0, 0, 2], expected[prop + f"-0-0-2-{c}"])
                allclose(moments[:, c, 0, 2, 0], expected[prop + f"-0-2-0-{c}"])
                allclose(moments[:, c, 2, 0, 0], expected[prop + f"-2-0-0-{c}"])
                allclose(moments[:, c, 1, 1, 0], expected[prop + f"-1-1-0-{c}"])
                allclose(moments[:, c, 1, 0, 1], expected[prop + f"-1-0-1-{c}"])
                allclose(moments[:, c, 0, 1, 1], expected[prop + f"-0-1-1-{c}"])
            if order > 2:
                # third-order moments
                allclose(moments[:, c, 0, 0, 3], expected[prop + f"-0-0-3-{c}"])
                allclose(moments[:, c, 0, 3, 0], expected[prop + f"-0-3-0-{c}"])
                allclose(moments[:, c, 3, 0, 0], expected[prop + f"-3-0-0-{c}"])
                allclose(moments[:, c, 1, 2, 0], expected[prop + f"-1-2-0-{c}"])
                allclose(moments[:, c, 2, 1, 0], expected[prop + f"-2-1-0-{c}"])
                allclose(moments[:, c, 1, 0, 2], expected[prop + f"-1-0-2-{c}"])
                allclose(moments[:, c, 2, 0, 1], expected[prop + f"-2-0-1-{c}"])
                allclose(moments[:, c, 0, 1, 2], expected[prop + f"-0-1-2-{c}"])
                allclose(moments[:, c, 0, 2, 1], expected[prop + f"-0-2-1-{c}"])
                allclose(moments[:, c, 1, 1, 1], expected[prop + f"-1-1-1-{c}"])


@pytest.mark.parametrize("spacing", [None, (0.8, 0.5, 1.2)])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("shape", [(500, 400), (64, 96, 32)])
@pytest.mark.parametrize("compute_orientation", [False, True])
@pytest.mark.parametrize("compute_axis_lengths", [False, True])
@pytest.mark.parametrize("blob_size_fraction", [0.03, 0.1, 0.3])
def test_inertia_tensor(
    shape,
    spacing,
    order,
    compute_orientation,
    compute_axis_lengths,
    blob_size_fraction,
):
    ndim = len(shape)
    labels = get_labels_nd(shape, blob_size_fraction=blob_size_fraction)
    max_label = int(cp.max(labels))
    if spacing is not None:
        # omit 3rd element for 2d images
        spacing = spacing[:ndim]
    props = ["inertia_tensor", "inertia_tensor_eigvals"]
    compute_eccentricity = True if ndim == 2 else False
    if compute_eccentricity:
        props += ["eccentricity"]
    if compute_orientation:
        props += ["orientation"]
    if compute_axis_lengths:
        props += ["axis_major_length", "axis_minor_length"]
    moments_raw = regionprops_moments(
        labels, max_label=max_label, order=order, spacing=spacing
    )
    moments_central = regionprops_moments_central(moments_raw, ndim=ndim)

    itensor_kwargs = dict(ndim=ndim, compute_orientation=compute_orientation)
    if order < 2:
        # can't compute inertia tensor without 2nd order moments
        with pytest.raises(ValueError):
            regionprops_inertia_tensor(moments_central, **itensor_kwargs)
        return

    if compute_orientation:
        if ndim != 2:
            with pytest.raises(ValueError):
                regionprops_inertia_tensor(moments_central, **itensor_kwargs)
            return
        itensor, orientation = regionprops_inertia_tensor(
            moments_central, **itensor_kwargs
        )
        assert orientation.shape == itensor.shape[:-2]
    else:
        itensor = regionprops_inertia_tensor(moments_central, **itensor_kwargs)

    assert itensor.shape[-2:] == (ndim, ndim)

    outputs = regionprops_inertia_tensor_eigvals(
        itensor,
        compute_axis_lengths=compute_axis_lengths,
        compute_eccentricity=compute_eccentricity,
    )
    if compute_eccentricity and compute_axis_lengths:
        eigvals, axis_lengths, eccentricity = outputs
        assert axis_lengths.shape == (max_label, ndim)
        assert eccentricity.shape == (max_label,)
    elif compute_axis_lengths:
        eigvals, axis_lengths = outputs
        assert axis_lengths.shape == (max_label, ndim)
    elif compute_eccentricity:
        eigvals, eccentricity = outputs
        assert eccentricity.shape == (max_label,)
    else:
        eigvals = outputs
        assert eigvals.shape == (max_label, ndim)

    expected = measure.regionprops_table(
        labels,
        properties=props,
        spacing=spacing,
    )

    # regionprops does not use the more accurate analytical expressions for the
    # central moments, so need to relax tolerance in the "central" moments case
    rtol = 1e-4
    atol = 1e-5
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    if ndim == 2:
        # valida inertia tensor
        allclose(itensor[:, 0, 0], expected["inertia_tensor-0-0"])
        allclose(itensor[:, 0, 1], expected["inertia_tensor-0-1"])
        allclose(itensor[:, 1, 0], expected["inertia_tensor-1-0"])
        allclose(itensor[:, 1, 1], expected["inertia_tensor-1-1"])

        # validate eigenvalues
        allclose(eigvals[:, 0], expected["inertia_tensor_eigvals-0"])
        allclose(eigvals[:, 1], expected["inertia_tensor_eigvals-1"])

        if compute_orientation:
            pass
            # Disabled orientation comparison as it is currently not robust
            # (fails for the spacing != None case)
            #
            # # validate orientation
            # # use sin/cos to avoid PI and -PI from being considered different
            # tol_kw = dict(rtol=1e-3, atol=1e-3)
            # assert_allclose(
            #     cp.cos(orientation), cp.cos(expected["orientation"]), **tol_kw
            # )
            # assert_allclose(
            #     cp.sin(orientation), cp.sin(expected["orientation"]), **tol_kw
            # )
        if compute_eccentricity:
            allclose(eccentricity, expected["eccentricity"])

    elif ndim == 3:
        # valida inertia tensor
        allclose(itensor[:, 0, 0], expected["inertia_tensor-0-0"])
        allclose(itensor[:, 0, 1], expected["inertia_tensor-0-1"])
        allclose(itensor[:, 0, 2], expected["inertia_tensor-0-2"])
        allclose(itensor[:, 1, 0], expected["inertia_tensor-1-0"])
        allclose(itensor[:, 1, 1], expected["inertia_tensor-1-1"])
        allclose(itensor[:, 1, 2], expected["inertia_tensor-1-2"])
        allclose(itensor[:, 2, 0], expected["inertia_tensor-2-0"])
        allclose(itensor[:, 2, 1], expected["inertia_tensor-2-1"])
        allclose(itensor[:, 2, 2], expected["inertia_tensor-2-2"])

        # validate eigenvalues
        allclose(eigvals[:, 0], expected["inertia_tensor_eigvals-0"])
        allclose(eigvals[:, 1], expected["inertia_tensor_eigvals-1"])
        allclose(eigvals[:, 2], expected["inertia_tensor_eigvals-2"])
    rtol = 1e-5
    # seems to be a larger fractional pixel error in length in 3D case
    atol = 5e-3 if ndim == 3 else 1e-5
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    if compute_axis_lengths:
        allclose(axis_lengths[..., 0], expected["axis_major_length"])
        allclose(axis_lengths[..., -1], expected["axis_minor_length"])


@pytest.mark.parametrize("spacing", [None, (0.8, 0.5, 1.2)])
# TODO (grlee77): enable num_channels > 1 case once bug is fixed
@pytest.mark.parametrize(
    "intensity_dtype, num_channels",
    [(cp.float32, 1), (cp.uint8, 3)],
)
@pytest.mark.parametrize("shape", [(800, 600), (80, 60, 40)])
@pytest.mark.parametrize("local", [False, True])
def test_centroid_weighted(
    shape, spacing, intensity_dtype, num_channels, local
):
    ndim = len(shape)
    labels = get_labels_nd(shape)

    max_label = int(cp.max(labels))
    if spacing is not None:
        # omit 3rd element for 2d images
        spacing = spacing[:ndim]
    intensity_image = get_intensity_image(
        shape, dtype=intensity_dtype, num_channels=num_channels
    )
    kwargs = {"spacing": spacing, "intensity_image": intensity_image}
    prop = "centroid_weighted"
    if local:
        prop += "_local"

    validate_via_cpu_skimage = False
    if not validate_via_cpu_skimage:
        expected = measure.regionprops_table(
            labels, properties=[prop], **kwargs
        )
    else:
        expected = measure_cpu.regionprops_table(
            cp.asnumpy(labels),
            properties=[prop],
            spacing=spacing,
            intensity_image=cp.asnumpy(intensity_image),
        )
    moments_raw = regionprops_moments(
        labels, max_label=max_label, order=1, **kwargs
    )

    if local:
        bbox = None
    else:
        bbox, _ = regionprops_bbox_coords(
            labels, max_label=max_label, return_slices=False
        )

    centroids = regionprops_centroid_weighted(
        moments_raw,
        ndim=ndim,
        bbox=bbox,
        compute_local=local,
        compute_global=not local,
        spacing=spacing,
    )
    assert centroids.shape[-1] == ndim

    rtol = 1e-7
    atol = 0
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    if num_channels == 1:
        for d in range(ndim):
            allclose(centroids[:, d], expected[prop + f"-{d}"])
    else:
        for c in range(num_channels):
            for d in range(ndim):
                allclose(centroids[:, c, d], expected[prop + f"-{d}-{c}"])


@pytest.mark.parametrize("shape", [(256, 512), (4096, 1024)])
@pytest.mark.parametrize("volume_fraction", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("blob_size_fraction", [0.025, 0.05, 0.1])
@pytest.mark.parametrize("robust", [False, True])
def test_perimeter(shape, robust, volume_fraction, blob_size_fraction):
    labels = get_labels_nd(
        shape,
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
    )

    max_label = int(cp.max(labels))
    if not robust:
        # remove any regions that are to close for non-robust algorithm
        labels_close = _find_close_labels(labels, labels > 0, max_label)
        for value in labels_close:
            labels[labels == value] = 0

        # relabel to ensure sequential
        labels = measure.label(labels > 0)
        max_label = int(cp.max(labels))

    max_label = int(cp.max(labels))
    values = regionprops_perimeter(labels, max_label=max_label, robust=robust)
    expected = measure.regionprops_table(labels, properties=["perimeter"])
    rtol = 1e-5
    atol = 1e-5
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    allclose(values, expected["perimeter"])


@pytest.mark.parametrize("shape", [(256, 512), (4096, 1024)])
@pytest.mark.parametrize("volume_fraction", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("blob_size_fraction", [0.025, 0.05, 0.1])
@pytest.mark.parametrize("robust", [False, True])
def test_perimeter_crofton(shape, robust, volume_fraction, blob_size_fraction):
    labels = get_labels_nd(
        shape,
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
    )

    max_label = int(cp.max(labels))
    if not robust:
        # remove any regions that are to close for non-robust algorithm
        labels_close = _find_close_labels(labels, labels > 0, max_label)
        for value in labels_close:
            labels[labels == value] = 0

        # relabel to ensure sequential
        labels = measure.label(labels > 0)
        max_label = int(cp.max(labels))

    values = regionprops_perimeter_crofton(
        labels, max_label=max_label, robust=robust
    )
    expected = measure.regionprops_table(
        labels, properties=["perimeter_crofton"]
    )
    rtol = 1e-5
    atol = 1e-5
    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
    allclose(values, expected["perimeter_crofton"])


@pytest.mark.parametrize(
    "shape, connectivity",
    [
        ((256, 512), None),
        ((256, 512), 1),
        ((256, 512), 2),
        ((96, 64, 48), None),
        ((96, 64, 48), 1),
        ((96, 64, 48), 3),
    ],
)
@pytest.mark.parametrize("volume_fraction", [0.25])
@pytest.mark.parametrize("blob_size_fraction", [0.04])
@pytest.mark.parametrize("insert_holes", [False, True])
@pytest.mark.parametrize("robust", [False, True])
def test_euler(
    shape,
    connectivity,
    robust,
    volume_fraction,
    blob_size_fraction,
    insert_holes,
):
    labels = get_labels_nd(
        shape,
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
        insert_holes=insert_holes,
    )

    max_label = int(cp.max(labels))
    if not robust:
        # remove any regions that are to close for non-robust algorithm
        labels_close = _find_close_labels(labels, labels > 0, max_label)
        for value in labels_close:
            labels[labels == value] = 0

        # relabel to ensure sequential
        labels = measure.label(labels > 0)
        max_label = int(cp.max(labels))

    values = regionprops_euler(
        labels, connectivity, max_label=max_label, robust=robust
    )
    if connectivity is None or connectivity == labels.ndim:
        # regionprops_table has hard-coded connectivity, can't use it to verify
        # other values
        expected = measure.regionprops_table(
            labels, properties=["euler_number"]
        )
        assert_array_equal(values, expected["euler_number"])
