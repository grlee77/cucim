import itertools

import cupy as cp
import numpy as np

from .._shared.utils import _supported_float_type, warn
from ..util import img_as_float
from . import rgb_colors
from .colorconv import gray2rgb, hsv2rgb, rgb2hsv

__all__ = ["color_dict", "label2rgb", "DEFAULT_COLORS"]


DEFAULT_COLORS = (
    "red",
    "blue",
    "yellow",
    "magenta",
    "green",
    "indigo",
    "darkorange",
    "cyan",
    "pink",
    "yellowgreen",
)


color_dict = {
    k: v for k, v in rgb_colors.__dict__.items() if isinstance(v, tuple)
}


def _rgb_vector(color):
    """Return RGB color as (1, 3) array.

    This RGB array gets multiplied by masked regions of an RGB image, which are
    partially flattened by masking (i.e. dimensions 2D + RGB -> 1D + RGB).

    Parameters
    ----------
    color : str or array
        Color name in `cucim.skimage.color.color_dict` or RGB float values
        between [0, 1].
    """
    if isinstance(color, str):
        color = color_dict[color]
    # Slice to handle RGBA colors.
    return np.asarray(color[:3])  # CuPy Backend: leave this array on the host


def _match_label_with_color(label, colors, bg_label, bg_color):
    """Return `unique_labels` and `color_cycle` for label array and color list.

    Colors are cycled for normal labels, but the background color should only
    be used for the background.
    """
    # Temporarily set background color; it will be removed later.
    if bg_color is None:
        bg_color = (0, 0, 0)
    bg_color = _rgb_vector(bg_color)

    # map labels to their ranks among all labels from small to large
    unique_labels, mapped_labels = cp.unique(label, return_inverse=True)
    # output of unique with return_inverse=True is no longer flat in NumPy 2.0
    # guarding against a potential corresponding future change for CuPy here
    mapped_labels = mapped_labels.reshape(-1)

    # The rank of each label is the index of the color it is matched to in
    # color cycle. bg_label should always be mapped to the first color, so
    # its rank must be 0. Other labels should be ranked from small to large
    # from 1.
    bg_coords = (label.ravel() == bg_label).nonzero()
    if bg_coords[0].size > 0:
        # get rank of bg_label
        bg_label_rank = mapped_labels[bg_coords[0][0]]
        if bg_label_rank > 0:
            mapped_labels[mapped_labels < bg_label_rank] += 1
            mapped_labels[bg_coords] = 0
    else:
        mapped_labels += 1

    # Modify labels and color cycle so background color is used only once.
    color_cycle = itertools.cycle(colors)
    color_cycle = itertools.chain([bg_color], color_cycle)
    return mapped_labels, color_cycle


def label2rgb(
    label,
    image=None,
    colors=None,
    alpha=0.3,
    bg_label=0,
    bg_color=(0, 0, 0),
    image_alpha=1,
    kind="overlay",
    *,
    saturation=0,
    channel_axis=-1,
):
    """Return an RGB image where color-coded labels are painted over the image.

    Parameters
    ----------
    label : ndarray
        Integer array of labels with the same shape as `image`.
    image : ndarray, optional
        Image used as underlay for labels. It should have the same shape as
        `labels`, optionally with an additional RGB (channels) axis. If `image`
        is an RGB image, it is converted to grayscale before coloring.
    colors : list, optional
        List of colors. If the number of labels exceeds the number of colors,
        then the colors are cycled.
    alpha : float [0, 1], optional
        Opacity of colorized labels. Ignored if image is `None`.
    bg_label : int, optional
        Label that's treated as the background. If `bg_label` is specified,
        `bg_color` is `None`, and `kind` is `overlay`,
        background is not painted by any colors.
    bg_color : str or array, optional
        Background color. Must be a name in `cucim.skimage.color.color_dict` or
        RGB float values between [0, 1].
    image_alpha : float [0, 1], optional
        Opacity of the image.
    kind : string, one of {'overlay', 'avg'}
        The kind of color image desired. 'overlay' cycles over defined colors
        and overlays the colored labels over the original image. 'avg' replaces
        each labeled segment with its average color, for a stained-class or
        pastel painting appearance.
    saturation : float [0, 1], optional
        Parameter to control the saturation applied to the original image
        between fully saturated (original RGB, `saturation=1`) and fully
        unsaturated (grayscale, `saturation=0`). Only applies when
        `kind='overlay'`.
    channel_axis : int, optional
        This parameter indicates which axis of the output array will correspond
        to channels. If `image` is provided, this must also match the axis of
        `image` that corresponds to channels.

    Returns
    -------
    result : array of float, shape (M, N, 3)
        The result of blending a cycling colormap (`colors`) for each distinct
        value in `label` with the image, at a certain alpha value.
    """
    if image is not None:
        image = cp.moveaxis(image, source=channel_axis, destination=-1)
    if kind == "overlay":
        rgb = _label2rgb_overlay(
            label,
            image,
            colors,
            alpha,
            bg_label,
            bg_color,
            image_alpha,
            saturation,
        )
    elif kind == "avg":
        rgb = _label2rgb_avg(label, image, bg_label, bg_color)
    else:
        raise ValueError("`kind` must be either 'overlay' or 'avg'.")
    return np.moveaxis(rgb, source=-1, destination=channel_axis)


alpha_scale_and_offset_ = cp.ElementwiseKernel(
    "float64 alpha",
    "X img1",
    """
    img1 = img1 * alpha + (1 - alpha);
""",
    name="alpha_scale_and_offset_",
)


alpha_blend_ = cp.ElementwiseKernel(
    "X img1, Y img2, float64 alpha",
    "Y result",
    """
    result = img1 * alpha + (1 - alpha) * img2;
""",
    name="alpha_scale_and_offset_",
)


def _label2rgb_overlay(
    label,
    image=None,
    colors=None,
    alpha=0.3,
    bg_label=-1,
    bg_color=None,
    image_alpha=1,
    saturation=0,
):
    """Return an RGB image where color-coded labels are painted over the image.

    Parameters
    ----------
    label : ndarray
        Integer array of labels with the same shape as `image`.
    image : ndarray, optional
        Image used as underlay for labels. It should have the same shape as
        `labels`, optionally with an additional RGB (channels) axis. If `image`
        is an RGB image, it is converted to grayscale before coloring.
    colors : list, optional
        List of colors. If the number of labels exceeds the number of colors,
        then the colors are cycled.
    alpha : float [0, 1], optional
        Opacity of colorized labels. Ignored if image is `None`.
    bg_label : int, optional
        Label that's treated as the background. If `bg_label` is specified and
        `bg_color` is `None`, background is not painted by any colors.
    bg_color : str or array, optional
        Background color. Must be a name in `cucim.skimage.color.color_dict` or
        RGB float values between [0, 1].
    image_alpha : float [0, 1], optional
        Opacity of the image.
    saturation : float [0, 1], optional
        Parameter to control the saturation applied to the original image
        between fully saturated (original RGB, `saturation=1`) and fully
        unsaturated (grayscale, `saturation=0`).

    Returns
    -------
    result : array of float, shape (M, N, 3)
        The result of blending a cycling colormap (`colors`) for each distinct
        value in `label` with the image, at a certain alpha value.
    """
    if not 0 <= saturation <= 1:
        warn(f"saturation must be in range [0, 1], got {saturation}")

    if colors is None:
        colors = DEFAULT_COLORS
    colors = [_rgb_vector(c) for c in colors]

    if image is not None:
        if (
            image.shape[: label.ndim] != label.shape
            or image.ndim > label.ndim + 1
        ):
            raise ValueError("`image` and `label` must be the same shape")

        if image.ndim == label.ndim + 1 and image.shape[-1] != 3:
            raise ValueError("`image` must be RGB (image.shape[-1] must be 3).")

        if image.min() < 0:
            warn("Negative intensities in `image` are not supported")

        float_dtype = _supported_float_type(image.dtype)
        image = img_as_float(image).astype(float_dtype, copy=False)
        if image.ndim > label.ndim and saturation != 1.0:
            hsv = rgb2hsv(image)
            hsv[..., 1] *= saturation
            image = hsv2rgb(hsv)
        elif image.ndim == label.ndim:
            image = gray2rgb(image)

        alpha_scale_and_offset_(image_alpha, image)

    # Ensure that all labels are non-negative so we can index into
    # `label_to_color` correctly.
    offset = min(int(label.min()), bg_label)
    if offset != 0:
        label = label - offset  # Make sure you don't modify the input array.
        bg_label -= offset

    new_type = np.min_scalar_type(int(label.max()))
    if new_type == bool:
        new_type = np.uint8
    label = label.astype(new_type, copy=False)

    mapped_labels_flat, color_cycle = _match_label_with_color(
        label, colors, bg_label, bg_color
    )

    if len(mapped_labels_flat) == 0:
        return image

    dense_labels = range(int(mapped_labels_flat.max()) + 1)

    # CuPy Backend: small color_cycle arrays are left on the CPU
    label_to_color = np.stack([c for i, c in zip(dense_labels, color_cycle)])
    # CuPy Backend: transfer to GPU after concatenation of small host arrays
    label_to_color = cp.asarray(label_to_color)

    mapped_labels = mapped_labels_flat.reshape(label.shape)
    label = mapped_labels

    colors = label_to_color[mapped_labels]

    if image is None:
        result = label_to_color[mapped_labels]

        # Remove background label if its color was not specified.
        remove_background = 0 in mapped_labels_flat and bg_color is None
        if remove_background:
            result[label == bg_label] = 0
    else:
        result = alpha_blend_(label_to_color[mapped_labels], image, alpha)

        # Remove background label if its color was not specified.
        remove_background = 0 in mapped_labels_flat and bg_color is None
        if remove_background:
            result[label == bg_label] = image[label == bg_label]

    return result


roi_sums_and_counts_ = cp.ElementwiseKernel(
    "raw Y img, X label, int32 bg_label",
    "raw float64 avg, raw int32 count",
    """
    if (label != bg_label) {
        atomicAdd(&avg[3*label], static_cast<double>(img[3*i]));
        atomicAdd(&avg[3*label + 1], static_cast<double>(img[3*i + 1]));
        atomicAdd(&avg[3*label + 2], static_cast<double>(img[3*i + 2]));
        atomicAdd(&count[label], 1);
    }
""",
    name="roi_sums_and_counts",
)

roi_assign_averages_ = cp.ElementwiseKernel(
    "raw float64 avg, Y label, int32 bg_label, raw X bg_color",
    "raw X out",
    """
    if (label != bg_label) {
        out[3*i] = avg[3*label];
        out[3*i + 1] = avg[3*label + 1];
        out[3*i + 2] = avg[3*label + 2];
    } else {
        out[3*i] = bg_color[0];
        out[3*i + 1] = bg_color[1];
        out[3*i + 2] = bg_color[2];
    }
""",
    name="roi_assign_averages",
)


def _label2rgb_avg(label_field, image, bg_label=0, bg_color=(0, 0, 0)):
    """Visualise each segment in `label_field` with its mean color in `image`.

    Parameters
    ----------
    label_field : ndarray of int
        A segmentation of an image.
    image : array, shape ``label_field.shape + (3,)``
        A color image of the same spatial shape as `label_field`.
    bg_label : int, optional
        A value in `label_field` to be treated as background.
    bg_color : 3-tuple of int, optional
        The color for the background label

    Returns
    -------
    out : ndarray, same shape and type as `image`
        The output visualization.
    """
    out = cp.empty(label_field.shape + (3,), dtype=image.dtype)
    if image.ndim == label_field.ndim and image.shape == label_field.shape:
        # convert to RGB as expected by roi_sums_and_counts_
        float_dtype = _supported_float_type(image.dtype)
        image = img_as_float(image).astype(float_dtype, copy=False)
        image = gray2rgb(image)
    if out.shape != image.shape:
        raise ValueError(
            f"{image.shape = }, but expected {out.shape} or {label_field.shape}"
        )

    # the kernels used assume C-order memory layout
    image = cp.ascontiguousarray(image)
    label_field = cp.ascontiguousarray(label_field)

    # the following should be reasonable if the labels are sequential
    # If the labels span a large range with many gaps, it may instead have been
    # best to get a unique list of the labels first.
    lmax = int(label_field.max())

    # kernels assume label_field starts from zero so adjust it if necessary
    lmin = int(label_field.min())
    if lmin < 0:
        label_field -= lmin
        lmax -= lmin
        bg_label -= lmin
        lmin = 0

    avg = cp.zeros((lmax + 1, 3), dtype=cp.float64)
    count = cp.zeros((lmax + 1, 1), dtype=cp.int32)
    roi_sums_and_counts_(image, label_field, bg_label, avg, count)

    # avoid divide by zero for background (average at bg_label not used)
    if bg_label >= lmin and bg_label <= lmax:
        count[bg_label] = 1
    avg /= count

    # assign the average or bg_color at each pixel in out
    bg_color = cp.asarray(bg_color, dtype=out.dtype)
    roi_assign_averages_(avg, label_field, bg_label, bg_color, out)
    return out
