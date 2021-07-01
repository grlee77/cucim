import itertools
from collections.abc import Iterable

import cupy as cp
import numpy as np

from ..transform import integral_image
from cucim import _misc


def _validate_window_size(axis_sizes):
    """Ensure all sizes in ``axis_sizes`` are odd.

    Parameters
    ----------
    axis_sizes : iterable of int

    Raises
    ------
    ValueError
        If any given axis size is even.
    """
    for axis_size in axis_sizes:
        if axis_size % 2 == 0:
            msg = ('Window size must not be even on any dimension. '
                   'Got {}'.format(axis_sizes))
            raise ValueError(msg)


def _to_np_mode(mode):
    """Convert padding modes from `ndi.correlate` to `np.pad`."""
    mode_translation_dict = dict(nearest='edge', reflect='symmetric',
                                 mirror='reflect')
    if mode in mode_translation_dict:
        mode = mode_translation_dict[mode]
    return mode


def _get_view(padded, kernel_shape, idx, val):
    """Get a view into `padded` that is offset by `idx` and scaled by `val`.

    If `padded` was created by padding the original image by `kernel_shape` as
    in _mean_std, then the view created here will match the size of the
    original image.
    """
    sl_shift = tuple([slice(c, s - (w_ - 1 - c))
                      for c, w_, s in zip(idx, kernel_shape, padded.shape)])
    v = padded[sl_shift]
    if val == 1:
        return v
    elif val == -1:
        return -v
    return val * v


def _correlate_sparse(image, kernel_shape, kernel_indices_and_values):
    """Perform correlation with a sparse kernel.

    Parameters
    ----------
    image : ndarray
        The (prepadded) image to be correlated.
    kernel_shape : tuple of int
        The shape of the sparse filter kernel.
    kernel_indices_and_values : list of 2-tuples
        This is a list of 2-tuples with length equal to the number of nonzero
        kernel entries. The first element of the tuple is the coordinate within
        `kernel_shape` and the second element is the kernel value at that
        coordinate.

    Returns
    -------
    out : ndarray
        The filtered image.

    Notes
    -----
    This function only returns results for the 'valid' region of the
    convolution, and thus `out` will be smaller than `image` by an amount
    equal to the kernel size along each axis.
    """
    idx, val = kernel_indices_and_values[0]
    # implementation assumes this corner is first in kernel_indices_in_values
    assert tuple(idx) == (0, ) * image.ndim
    out = _get_view(image, kernel_shape, idx, val)
    if not out.flags.owndata:
        # make out contiguous and avoid modifying image
        out = out.copy()
    for idx, val in kernel_indices_and_values[1:]:
        out += _get_view(image, kernel_shape, idx, val)
    return out


def correlate_sparse(image, kernel, mode='reflect'):
    """Compute valid cross-correlation of `padded_array` and `kernel`.

    This function is *fast* when `kernel` is large with many zeros.

    See ``scipy.ndimage.correlate`` for a description of cross-correlation.

    Parameters
    ----------
    image : ndarray, dtype float, shape (M, N,[ ...,] P)
        The input array. If mode is 'valid', this array should already be
        padded, as a margin of the same shape as kernel will be stripped
        off.
    kernel : ndarray, dtype float shape (Q, R,[ ...,] S)
        The kernel to be correlated. Must have the same number of
        dimensions as `padded_array`. For high performance, it should
        be sparse (few nonzero entries).
    mode : string, optional
        See `scipy.ndimage.correlate` for valid modes.
        Additionally, mode 'valid' is accepted, in which case no padding is
        applied and the result is the result for the smaller image for which
        the kernel is entirely inside the original data.

    Returns
    -------
    result : array of float, shape (M, N,[ ...,] P)
        The result of cross-correlating `image` with `kernel`. If mode
        'valid' is used, the resulting shape is (M-Q+1, N-R+1,[ ...,] P-S+1).
    """
    if mode == 'valid':
        padded_image = image
    else:
        np_mode = _to_np_mode(mode)
        _validate_window_size(kernel.shape)
        padded_image = cp.pad(
            image,
            [(w // 2, w // 2) for w in kernel.shape],
            mode=np_mode,
        )

    kernel = cp.asnumpy(kernel)
    indices = np.nonzero(kernel)
    values = kernel[indices].astype(padded_image.dtype, copy=False)
    indices = list(zip(*indices))
    kernel_indices_and_values = [(idx, v) for idx, v in zip(indices, values)]
    if (0, ) * kernel.ndim not in indices:
        kernel_indices_and_values = \
            [((0,) * kernel.ndim, 0.0)] + kernel_indices_and_values
    out = _correlate_sparse(
        padded_image, kernel.shape, kernel_indices_and_values
    )
    return out


def _uniform_filter(image, w):
    """Return local mean of each pixel using a neighborhood defined by a
    rectangular window size ``w``. The algorithm uses integral images to
    speedup computation.

    Parameters
    ----------
    image : ndarray
        Input image.
    w : int, or iterable of int
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).

    Returns
    -------
    m : ndarray of float, same shape as ``image``
        Local mean of the image.

    """
    from cucim.skimage.thresholding import _validate_window_size
    if not isinstance(w, Iterable):
        w = (w,) * image.ndim
    _validate_window_size(w)

    # TODO: should be (k + 1) // 2 instead of k // 2 + 1?
    pad_width = tuple((k // 2 + 1, k // 2) for k in w)

    float_dtype = np.result_type(image.dtype, np.float32)
    padded = cp.pad(image.astype(float_dtype), pad_width,
                    mode='reflect')

    integral = integral_image(padded)

    # Store the kernel as a list 2-tuples where:
    #     - The first element is an index into the kernel (of shape w).
    #     - The second element is the value at that index.
    kernel_indices_and_values = []
    for indices in itertools.product(*tuple([(0, _w) for _w in w])):
        kernel_indices_and_values.append(
            (indices, (-1) ** (image.ndim % 2 != np.sum(indices) % 2))
        )

    total_window_size = _misc.prod(w)
    kernel_shape = tuple(_w + 1 for _w in w)
    m = _correlate_sparse(integral, kernel_shape, kernel_indices_and_values)
    m = m.astype(dtype=float_dtype, copy=False)
    m /= total_window_size

    return m
