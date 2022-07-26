import cupy as cp
import numpy as np

from .._shared.filters import gaussian
from .._shared.utils import check_nD
from .util import (DescriptorExtractor, _mask_border_keypoints,
                   _prepare_grayscale_input_2D)

"""brief_pythran.py

import numpy as np

#pythran export _brief_loop(float32[:,:] or float64[:,:], uint8[:,:], int64[:,2], int32[:,2], int32[:,2])
def _brief_loop(image, descriptors, keypoints, pos0, pos1):
    for p in range(pos0.shape[0]):
        pr0, pc0 = pos0[p]
        pr1, pc1 = pos1[p]
        for k in range(keypoints.shape[0]):
            kr, kc = keypoints[k]
            if image[kr + pr0, kc + pc0] < image[kr + pr1, kc + pc1]:
                descriptors[k, p] = True
"""


class BRIEF(DescriptorExtractor):
    """BRIEF binary descriptor extractor.

    BRIEF (Binary Robust Independent Elementary Features) is an efficient
    feature point descriptor. It is highly discriminative even when using
    relatively few bits and is computed using simple intensity difference
    tests.

    For each keypoint, intensity comparisons are carried out for a specifically
    distributed number N of pixel-pairs resulting in a binary descriptor of
    length N. For binary descriptors the Hamming distance can be used for
    feature matching, which leads to lower computational cost in comparison to
    the L2 norm.

    Parameters
    ----------
    descriptor_size : int, optional
        Size of BRIEF descriptor for each keypoint. Sizes 128, 256 and 512
        recommended by the authors. Default is 256.
    patch_size : int, optional
        Length of the two dimensional square patch sampling region around
        the keypoints. Default is 49.
    mode : {'normal', 'uniform'}, optional
        Probability distribution for sampling location of decision pixel-pairs
        around keypoints.
    sample_seed : {None, int, `numpy.random.Generator`}, optional
        If `sample_seed` is None the `numpy.random.Generator` singleton is
        used.
        If `sample_seed` is an int, a new ``Generator`` instance is used,
        seeded with `sample_seed`.
        If `sample_seed` is already a ``Generator`` instance then that instance
        is used.

        Seed for the random sampling of the decision pixel-pairs. From a square
        window with length `patch_size`, pixel pairs are sampled using the
        `mode` parameter to build the descriptors using intensity comparison.
        The value of `sample_seed` must be the same for the images to be
        matched while building the descriptors.
    sigma : float, optional
        Standard deviation of the Gaussian low-pass filter applied to the image
        to alleviate noise sensitivity, which is strongly recommended to obtain
        discriminative and good descriptors.

    Attributes
    ----------
    descriptors : (Q, `descriptor_size`) array of dtype bool
        2D ndarray of binary descriptors of size `descriptor_size` for Q
        keypoints after filtering out border keypoints with value at an
        index ``(i, j)`` either being ``True`` or ``False`` representing
        the outcome of the intensity comparison for i-th keypoint on j-th
        decision pixel-pair. It is ``Q == cp.sum(mask)``.
    mask : (N, ) array of dtype bool
        Mask indicating whether a keypoint has been filtered out
        (``False``) or is described in the `descriptors` array (``True``).

    Examples
    --------
    >>> from cucim.skimage.feature import (corner_harris, corner_peaks, BRIEF,
    ...                                    match_descriptors)
    >>> import cupy as cp
    >>> square1 = cp.zeros((8, 8), dtype=cp.int32)
    >>> square1[2:6, 2:6] = 1
    >>> square1
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
    >>> square2 = cp.zeros((9, 9), dtype=cp.int32)
    >>> square2[2:7, 2:7] = 1
    >>> square2
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
    >>> keypoints1 = corner_peaks(corner_harris(square1), min_distance=1)
    >>> keypoints2 = corner_peaks(corner_harris(square2), min_distance=1)
    >>> extractor = BRIEF(patch_size=5)
    >>> extractor.extract(square1, keypoints1)
    >>> descriptors1 = extractor.descriptors
    >>> extractor.extract(square2, keypoints2)
    >>> descriptors2 = extractor.descriptors
    >>> matches = match_descriptors(descriptors1, descriptors2)
    >>> matches
    array([[0, 0],
           [1, 1],
           [2, 2],
           [3, 3]])
    >>> keypoints1[matches[:, 0]]
    array([[2, 2],
           [2, 5],
           [5, 2],
           [5, 5]])
    >>> keypoints2[matches[:, 1]]
    array([[2, 2],
           [2, 6],
           [6, 2],
           [6, 6]])

    """

    def __init__(self, descriptor_size=256, patch_size=49,
                 mode='normal', sigma=1, sample_seed=1):

        mode = mode.lower()
        if mode not in ('normal', 'uniform'):
            raise ValueError("`mode` must be 'normal' or 'uniform'.")

        self.descriptor_size = descriptor_size
        self.patch_size = patch_size
        self.mode = mode
        self.sigma = sigma

        if sample_seed is None:
            self.sample_seed = np.random.SeedSequence().entropy
        else:
            self.sample_seed = sample_seed

        self.descriptors = None
        self.mask = None

    def extract(self, image, keypoints):
        """Extract BRIEF binary descriptors for given keypoints in image.

        Parameters
        ----------
        image : 2D array
            Input image.
        keypoints : (N, 2) array
            Keypoint coordinates as ``(row, col)``.

        """
        check_nD(image, 2)

        random = cp.random.default_rng(self.sample_seed)

        image = _prepare_grayscale_input_2D(image)

        # Gaussian low-pass filtering to alleviate noise sensitivity
        image = cp.ascontiguousarray(
            gaussian(image, self.sigma, mode='reflect')
        )

        # Sampling pairs of decision pixels in patch_size x patch_size window
        desc_size = self.descriptor_size
        patch_size = self.patch_size
        if self.mode == 'normal':
            samples = (patch_size / 5.0) * random.standard_normal(desc_size
                                                                  * 8)
            samples = cp.array(samples, dtype=cp.int32)
            samples = samples[(samples < (patch_size // 2))
                              & (samples > - (patch_size - 2) // 2)]

            pos1 = samples[:desc_size * 2].reshape(desc_size, 2)
            pos2 = samples[desc_size * 2:desc_size * 4].reshape(desc_size, 2)
        elif self.mode == 'uniform':
            samples = random.integers(-(patch_size - 2) // 2,
                                      (patch_size // 2) + 1,
                                      (desc_size * 2, 2))
            samples = cp.array(samples, dtype=cp.int32)
            pos1, pos2 = cp.split(samples, 2)

        pos1 = cp.ascontiguousarray(pos1)
        pos2 = cp.ascontiguousarray(pos2)

        # Removing keypoints that are within (patch_size / 2) distance from the
        # image border
        self.mask = _mask_border_keypoints(image.shape, keypoints,
                                           patch_size // 2)

        keypoints = cp.array(keypoints[self.mask, :], dtype=cp.int64,
                             order='C', copy=False)

        # TODO: can probably change cp.zeros -> cp.empty here
        self.descriptors = cp.zeros((keypoints.shape[0], desc_size),
                                    dtype=bool, order='C')


        # TODO: convert _brief_loop to a CuPy ElementwiseKernel using raw
        #       arrays with size set by pos1.shape[0]
        #       each CUDA thread could loop over all keypoints for a given position
        #       by default there are 49*49 = 2401 positions, so this should be
        #       enough to get decent parallelism,
        _brief_loop(image, self.descriptors.view(cp.uint8), keypoints,
                    pos1, pos2)
