import cupy as cp

from .._shared.utils import _supported_float_type, check_nD
from ..util import img_as_float


class DescriptorExtractor(object):

    def __init__(self):
        self.descriptors_ = cp.array([])

    def extract(self, image, keypoints):
        """Extract feature descriptors in image for given keypoints.

        Parameters
        ----------
        image : 2D array
            Input image.
        keypoints : (N, 2) array
            Keypoint locations as ``(row, col)``.

        """
        raise NotImplementedError()


def _prepare_grayscale_input_2D(image):
    image = cp.squeeze(image)
    check_nD(image, 2)
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    return image.astype(float_dtype, copy=False)


def _prepare_grayscale_input_nD(image):
    image = cp.squeeze(image)
    check_nD(image, range(2, 6))
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    return image.astype(float_dtype, copy=False)


@cp.fuse()
def _fused_mask_border(distance, rows, cols, k0, k1):
    mask = (distance - 1) < k0
    mask &= k0 < (rows - distance + 1)
    mask &= (distance - 1) < k1
    mask &= k1z < (cols - distance + 1)
    return mask


def _mask_border_keypoints(image_shape, keypoints, distance):
    """Mask coordinates that are within certain distance from the image border.

    Parameters
    ----------
    image_shape : (2, ) array_like
        Shape of the image as ``(rows, cols)``.
    keypoints : (N, 2) array
        Keypoint coordinates as ``(rows, cols)``.
    distance : int
        Image border distance.

    Returns
    -------
    mask : (N, ) bool array
        Mask indicating if pixels are within the image (``True``) or in the
        border region of the image (``False``).

    """

    rows = image_shape[0]
    cols = image_shape[1]

    return _fused_mask_border(
        distance, rows, cols, keypoints[:, 0], keypoints[:, 1]
    )
