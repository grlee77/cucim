import functools
import math

import cupy as cp
import pytest
from cupy.testing import (
    assert_allclose,
    assert_array_equal,
)
from scipy.ndimage import find_objects as cpu_find_objects

from cucim.skimage import data, measure
from cucim.skimage.measure._regionprops_gpu import (
    area_bbox_from_slices,
    moments_to_moments_central,
    normalize_central_moments,
    regionprops_area,
    regionprops_area_bbox,
    regionprops_bbox_coords,
    regionprops_centroid,
    regionprops_centroid_local,
    regionprops_intensity_max,
    regionprops_intensity_mean,
    regionprops_intensity_min,
    regionprops_intensity_std,
    regionprops_moments,
    regionprops_num_pixels,
)


def get_labels_nd(shape, blob_size_fraction=0.05, volume_fraction=0.35, rng=5):
    ndim = len(shape)
    blobs_kwargs = dict(
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
        rng=rng,
    )
    blobs = data.binary_blobs(max(shape), n_dim=ndim, **blobs_kwargs)
    # crop to rectangular
    blobs = blobs[tuple(slice(s) for s in shape)]

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
@pytest.mark.parametrize("count_dtype", [cp.uint32, cp.int32, cp.uint64])
def test_num_pixels(precompute_max, ndim, count_dtype):
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    labels = get_labels_nd(shape)

    max_label = int(cp.max(labels)) if precompute_max else None
    num_pixels = regionprops_num_pixels(
        labels, max_label=max_label, count_dtype=count_dtype
    )
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
        assert_allclose(means, expected["intensity_mean"], rtol=1e-6)
        assert_allclose(stds, expected["intensity_std"], rtol=1e-5)
    else:
        for c in range(num_channels):
            assert_allclose(
                means[..., c], expected[f"intensity_mean-{c}"], rtol=1e-6
            )
            assert_allclose(
                stds[..., c], expected[f"intensity_std-{c}"], rtol=1e-5
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
        coord_dtype=dtype,
    )
    assert bbox.dtype == dtype
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


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
def test_centroid(precompute_max, local, ndim):
    shape = (1024, 512) if ndim == 2 else (80, 64, 48)
    labels = get_labels_nd(shape)
    max_label = int(cp.max(labels)) if precompute_max else None
    if local:
        name = "centroid_local"
        counts, centroid = regionprops_centroid_local(
            labels, max_label=max_label
        )
    else:
        name = "centroid"
        counts, centroid = regionprops_centroid(labels, max_label=max_label)
    expected = measure.regionprops_table(
        labels, properties=["num_pixels", name]
    )
    assert_array_equal(counts, expected["num_pixels"])
    assert_allclose(centroid[:, 0], expected[name + "-0"])
    if ndim > 1:
        assert_allclose(centroid[:, 1], expected[name + "-1"])
    if ndim > 2:
        assert_allclose(centroid[:, 2], expected[name + "-2"])


@pytest.mark.parametrize("spacing", [None, (0.8, 0.5)])
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
def test_moments_2d(
    spacing, order, weighted, intensity_dtype, num_channels, norm_type
):
    shape = (800, 600)
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
        moments = moments_to_moments_central(moments, ndim=ndim)
        if norm_type == "normalized":
            moments = normalize_central_moments(
                moments, ndim=ndim, spacing=spacing
            )

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

    # regionprops does not use the more accurate analytical expressions for the
    # central moments, so need to relax tolerance in the "central" moments case
    rtol = 1e-4 if norm_type != "raw" else 1e-6
    atol = 1e-5 if norm_type != "raw" else 0

    allclose = functools.partial(assert_allclose, rtol=rtol, atol=atol)
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
    shape = (96, 64, 48)
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
        moments = moments_to_moments_central(moments, ndim=ndim)
        if norm_type == "normalized":
            moments = normalize_central_moments(
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
    rtol = 1e-4 if norm_type != "raw" else 1e-6
    atol = 1e-3 if norm_type != "raw" else 0

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
