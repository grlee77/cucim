import math

import cupy as cp
import numpy as np

from cucim.skimage._vendored import ndimage as ndi, pad

from ._regionprops_gpu_basic_kernels import regionprops_bbox_coords
from ._regionprops_gpu_utils import _find_close_labels, _get_min_integer_dtype

__all__ = [
    "regionprops_euler",
    "regionprops_perimeter",
    "regionprops_perimeter_crofton",
]


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

    If the labeled regions have holes, the hole edges will be included in this
    measurement. If this is not desired, use regionprops_label_filled to fill
    the holes and then pass the filled labels image to this function.

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
        footprint = cp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=cp.uint8)
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

    max_val = 50  # 1 + sum of kernel weights used for convolve above
    min_integer_type = _get_min_integer_dtype(
        (max_label + 1) * max_val, signed=False
    )
    if perimeter_image.dtype != min_integer_type:
        perimeter_image = perimeter_image.astype(min_integer_type)
    if labels_dilated.dtype != min_integer_type:
        # print(f"{min_integer_type=}, {max_label=}, {max_val=}")
        labels_dilated = labels_dilated.astype(min_integer_type)
    labels_dilated *= max_val

    # values in perimeter_image are guaranteed to be in range [0, max_val) so
    # need to multiply each label by max_val to make sure all labels have a
    # unique set of values during bincount
    perimeter_image = perimeter_image + labels_dilated

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

    If the labeled regions have holes, the hole edges will be included in this
    measurement. If this is not desired, use regionprops_label_filled to fill
    the holes and then pass the filled labels image to this function.

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

    max_val = 16  # 1 + (sum of the kernel weights) used for convolve above
    min_integer_type = _get_min_integer_dtype(
        (max_label + 1) * max_val, signed=False
    )
    if image_filtered.dtype != min_integer_type:
        image_filtered = image_filtered.astype(min_integer_type)
    if labels_dilated.dtype != min_integer_type:
        labels_dilated = labels_dilated.astype(min_integer_type)
    labels_dilated *= max_val

    # values in image_filtered are guaranteed to be in range [0, 15] so need to
    # multiply each label by 16 to make sure all labels have a unique set of
    # values during bincount
    image_filtered_labeled = image_filtered + labels_dilated

    minlength = max_val * (max_label + 1)
    h = cp.bincount(
        image_filtered_labeled[binary_image_mask], minlength=minlength
    )

    # values for label=1 start at index max_val
    h = h[max_val:minlength].reshape((max_label, max_val))

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

    min_integer_type = _get_min_integer_dtype(
        (max_label + 1) * filter_bins, signed=False
    )
    if image_filtered.dtype != min_integer_type:
        image_filtered = image_filtered.astype(min_integer_type)
    if labels_dilated.dtype != min_integer_type:
        labels_dilated = labels_dilated.astype(min_integer_type)
    labels_dilated *= filter_bins

    # values in image_filtered are guaranteed to be in range [0, filter_bins)
    # so need to multiply each label by filter_bins to make sure all labels
    # have a unique set of values during bincount
    image_filtered_labeled = image_filtered + labels_dilated

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
