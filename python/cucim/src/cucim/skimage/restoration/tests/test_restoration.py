import cupy as cp
import numpy as np
import pytest
from cupyx.scipy import ndimage as ndi
from scipy import signal

from cucim.skimage import restoration
from cucim.skimage._shared.testing import expected_warnings, fetch
from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage.color import rgb2gray
from cucim.skimage.restoration import uft


def camera():
    import skimage
    import skimage.data

    return cp.asarray(skimage.img_as_float(skimage.data.camera()))


def astronaut():
    import skimage
    import skimage.data

    return cp.asarray(skimage.img_as_float(skimage.data.astronaut()))


test_img = camera()


def _get_rtol_atol(dtype):
    rtol = 1e-3
    atol = 0
    if dtype == np.float16:
        rtol = 1e-2
        atol = 1e-3
    elif dtype == np.float32:
        atol = 1e-5
    return rtol, atol


@pytest.mark.parametrize("dtype", [cp.float16, cp.float32, cp.float64])
def test_wiener(dtype):
    psf = np.ones((5, 5), dtype=dtype) / 25
    data = signal.convolve2d(cp.asnumpy(test_img), psf, "same")
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)

    psf = cp.asarray(psf, dtype=dtype)
    data = cp.asarray(data, dtype=dtype)

    deconvolved = restoration.wiener(data, psf, 0.05)
    assert deconvolved.dtype == _supported_float_type(dtype)

    rtol, atol = _get_rtol_atol(dtype)
    path = fetch("restoration/tests/camera_wiener.npy")
    cp.testing.assert_allclose(deconvolved, np.load(path), rtol=rtol, atol=atol)

    _, laplacian = uft.laplacian(2, data.shape)
    otf = uft.ir2tf(psf, data.shape, is_real=False)
    assert otf.real.dtype == _supported_float_type(dtype)
    deconvolved = restoration.wiener(
        data, otf, 0.05, reg=laplacian, is_real=False
    )
    assert deconvolved.real.dtype == _supported_float_type(dtype)
    cp.testing.assert_allclose(
        cp.real(deconvolved), np.load(path), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", [cp.float16, cp.float32, cp.float64])
def test_unsupervised_wiener(dtype):
    psf = np.ones((5, 5), dtype=dtype) / 25
    data = signal.convolve2d(cp.asnumpy(test_img), psf, "same")
    seed = 16829302
    # keep old-style RandomState here for compatibility with previously stored
    # reference data in camera_unsup.npy and camera_unsup2.npy
    rng = np.random.RandomState(seed)
    data += 0.1 * data.std() * rng.standard_normal(data.shape)

    psf = cp.asarray(psf, dtype=dtype)
    data = cp.asarray(data, dtype=dtype)
    deconvolved, _ = restoration.unsupervised_wiener(data, psf, rng=seed)
    float_type = _supported_float_type(dtype)
    assert deconvolved.dtype == float_type

    rtol, atol = _get_rtol_atol(dtype)

    # CuPy Backend: Cannot use the following comparison to scikit-image data
    #               due to different random values generated by cp.random
    #               within unsupervised_wiener.
    #               Verified similar appearance qualitatively.
    # path = fetch("restoration/tests/camera_unsup.npy")
    # cp.testing.assert_allclose(deconvolved, np.load(path), rtol=1e-3)

    _, laplacian = uft.laplacian(2, data.shape)
    otf = uft.ir2tf(psf, data.shape, is_real=False)
    assert otf.real.dtype == float_type

    np.random.seed(0)
    deconvolved2 = restoration.unsupervised_wiener(  # noqa
        data,
        otf,
        reg=laplacian,
        is_real=False,
        user_params={
            "callback": lambda x: None,
            "max_num_iter": 200,
            "min_num_iter": 30,
        },
        rng=seed,
    )[0]
    assert deconvolved2.real.dtype == float_type

    # CuPy Backend: Cannot use the following comparison to scikit-image data
    #               due to different random values generated by cp.random
    #               within unsupervised_wiener.
    #               Verified similar appearance qualitatively.
    # path = fetch("restoration/tests/camera_unsup2.npy")
    # cp.testing.assert_allclose(cp.real(deconvolved), np.load(path), rtol=1e-3)


def test_unsupervised_wiener_deprecated_user_param():
    psf = np.ones((5, 5), dtype=float) / 25
    data = signal.convolve2d(cp.asnumpy(test_img), psf, "same")
    data = cp.array(data)
    psf = cp.array(psf)
    otf = uft.ir2tf(psf, data.shape, is_real=False)
    _, laplacian = uft.laplacian(2, data.shape)
    with expected_warnings(
        [
            "`max_iter` is a deprecated key",
            "`min_iter` is a deprecated key",
            "`random_state` is a deprecated argument name",
        ]
    ):
        restoration.unsupervised_wiener(
            data,
            otf,
            reg=laplacian,
            is_real=False,
            user_params={"max_iter": 200, "min_iter": 30},
            random_state=5,
        )


def test_image_shape():
    """Test that shape of output image in deconvolution is same as input.

    This addresses issue #1172.
    """
    point = cp.zeros((5, 5), float)
    point[2, 2] = 1.0
    psf = ndi.gaussian_filter(point, sigma=1.0)
    # image shape: (45, 45), as reported in #1172
    image = cp.asarray(test_img[65:165, 215:315])  # just the face
    image_conv = ndi.convolve(image, psf)
    deconv_sup = restoration.wiener(image_conv, psf, 1)
    deconv_un = restoration.unsupervised_wiener(image_conv, psf)[0]
    # test the shape
    assert image.shape == deconv_sup.shape
    assert image.shape == deconv_un.shape
    # test the reconstruction error
    sup_relative_error = cp.abs(deconv_sup - image) / image
    un_relative_error = cp.abs(deconv_un - image) / image
    cp.testing.assert_array_less(cp.median(sup_relative_error), 0.1)
    cp.testing.assert_array_less(cp.median(un_relative_error), 0.1)


def test_richardson_lucy():
    rstate = np.random.RandomState(0)
    psf = np.ones((5, 5)) / 25
    data = signal.convolve2d(cp.asnumpy(test_img), psf, "same")
    np.random.seed(0)
    data += 0.1 * data.std() * rstate.standard_normal(data.shape)

    data = cp.asarray(data)
    psf = cp.asarray(psf)
    deconvolved = restoration.richardson_lucy(data, psf, 5)

    path = fetch("restoration/tests/camera_rl.npy")
    cp.testing.assert_allclose(deconvolved, np.load(path), rtol=1e-4)


@pytest.mark.parametrize("dtype_image", [cp.float16, cp.float32, cp.float64])
@pytest.mark.parametrize("dtype_psf", [cp.float32, cp.float64])
def test_richardson_lucy_filtered(dtype_image, dtype_psf):
    if dtype_image == cp.float64:
        atol = 1e-8
    else:
        atol = 1e-4

    test_img_astro = rgb2gray(astronaut())

    psf = cp.ones((5, 5), dtype=dtype_psf) / 25
    data = cp.array(
        signal.convolve2d(cp.asnumpy(test_img_astro), cp.asnumpy(psf), "same"),
        dtype=dtype_image,
    )
    deconvolved = restoration.richardson_lucy(data, psf, 5, filter_epsilon=1e-6)
    assert deconvolved.dtype == _supported_float_type(data.dtype)

    path = fetch("restoration/tests/astronaut_rl.npy")
    cp.testing.assert_allclose(deconvolved, np.load(path), rtol=1e-3, atol=atol)
