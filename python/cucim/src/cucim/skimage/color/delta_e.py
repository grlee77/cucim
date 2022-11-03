"""
Functions for calculating the "distance" between colors.

Implicit in these definitions of "distance" is the notion of "Just Noticeable
Distance" (JND).  This represents the distance between colors where a human can
perceive different colors.  Humans are more sensitive to certain colors than
others, which different deltaE metrics correct for with varying degrees of
sophistication.

The literature often mentions 1 as the minimum distance for visual
differentiation, but more recent studies (Mahy 1994) peg JND at 2.3

The delta-E notation comes from the German word for "Sensation" (Empfindung).

Reference
---------
https://en.wikipedia.org/wiki/Color_difference

"""

import warnings

import cupy as cp
import numpy as np

from .._shared.utils import _supported_float_type
from .colorconv import lab2lch


def _float_inputs(lab1, lab2, allow_float32=True):
    if allow_float32:
        float_dtype = _supported_float_type([lab1.dtype, lab2.dtype])
    else:
        float_dtype = cp.float64
    lab1 = lab1.astype(float_dtype, copy=False)
    lab2 = lab2.astype(float_dtype, copy=False)
    return lab1, lab2


@cp.memoize()
def _get_cie76_kernel():

    return cp.ElementwiseKernel(
        in_params='F L1, F a1, F b1, F L2, F a2, F b2',
        out_params='F out',
        operation="""
            out = (L2 - L1) * (L2 - L1);
            out += (a2 - a1) * (a2 - a1);
            out += (b2 - b1) * (b2 - b1);
            out = sqrt(out);
        """,
        name='cucim_skimage_cie76'
    )


def deltaE_cie76(lab1, lab2, channel_axis=-1):
    """Euclidean distance between two points in Lab color space

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

    Returns
    -------
    dE : array_like
        distance between colors `lab1` and `lab2`

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] A. R. Robertson, "The CIE 1976 color-difference formulae,"
           Color Res. Appl. 2, 7-11 (1977).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    L1, a1, b1 = cp.moveaxis(lab1, source=channel_axis, destination=0)[:3]
    L2, a2, b2 = cp.moveaxis(lab2, source=channel_axis, destination=0)[:3]
    kernel = _get_cie76_kernel()
    out = cp.empty_like(L1)
    return kernel(L1, a1, b1, L2, a2, b2, out)


@cp.memoize()
def _get_ciede94_kernel():

    return cp.ElementwiseKernel(
        in_params='F L1, F C1, F k1, F L2, F C2, F k2, F dH2, float64 kL, float64 kC, float64 kH',  # noqa
        out_params='F out',
        operation="""
            F dL = L1 - L2;
            F dC = C1 - C2;

            F SC = 1 + k1 * C1;
            F SH = 1 + k2 * C1;

            out = dL / kL;
            out *= out;
            F tmp = dC / (kC * SC);
            tmp *= tmp;
            out += tmp;
            tmp = kH * SH;
            tmp *= tmp;
            out += dH2 / tmp;
            out = max(out, static_cast<F>(0));
            out = sqrt(out);
        """,
        name='cucim_skimage_ciede94'
    )


def deltaE_ciede94(lab1, lab2, kH=1, kC=1, kL=1, k1=0.045, k2=0.015, *,
                   channel_axis=-1):
    """Color difference according to CIEDE 94 standard

    Accommodates perceptual non-uniformities through the use of application
    specific scale factors (`kH`, `kC`, `kL`, `k1`, and `k2`).

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    kH : float, optional
        Hue scale
    kC : float, optional
        Chroma scale
    kL : float, optional
        Lightness scale
    k1 : float, optional
        first scale parameter
    k2 : float, optional
        second scale parameter
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

    Returns
    -------
    dE : array_like
        color difference between `lab1` and `lab2`

    Notes
    -----
    deltaE_ciede94 is not symmetric with respect to lab1 and lab2.  CIEDE94
    defines the scales for the lightness, hue, and chroma in terms of the first
    color.  Consequently, the first color should be regarded as the "reference"
    color.

    `kL`, `k1`, `k2` depend on the application and default to the values
    suggested for graphic arts

    ==========  ==============  ==========
    Parameter    Graphic Arts    Textiles
    ==========  ==============  ==========
    `kL`         1.000           2.000
    `k1`         0.045           0.048
    `k2`         0.015           0.014
    ==========  ==============  ==========

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    lab1 = cp.moveaxis(lab1, source=channel_axis, destination=0)
    lab2 = cp.moveaxis(lab2, source=channel_axis, destination=0)
    L1, C1 = lab2lch(lab1, channel_axis=0)[:2]
    L2, C2 = lab2lch(lab2, channel_axis=0)[:2]

    dH2 = get_dH2(lab1, lab2, channel_axis=0)

    out = cp.empty_like(L1)
    kernel = _get_ciede94_kernel()
    return kernel(L1, C1, k1, L2, C2, k2, dH2, kL, kC, kH, out)


@cp.memoize()
def _get_ciede2000_kernel():

    return cp.ElementwiseKernel(
        in_params='F L1, F C1, F k1, F L2, F C2, F k2, F dH2, float64 kL, float64 kC, float64 kH',  # noqa
        out_params='F out',
        operation="""
            F dL = L1 - L2;
            F dC = C1 - C2;

            F SC = 1 + k1 * C1;
            F SH = 1 + k2 * C1;

            out = dL / kL;
            out *= out;
            F tmp = dC / (kC * SC);
            tmp *= tmp;
            out += tmp;
            tmp = kH * SH;
            tmp *= tmp;
            out += dH2 / tmp;
            out = max(out, static_cast<F>(0));
            out = sqrt(out);
        """,
        name='cucim_skimage_ciede2000'
    )


@cp.memoize()
def _get_ciede2000_kernel():

    return cp.ElementwiseKernel(
        in_params='F L1, F a1, F b1, F L2, F a2, F b2, float64 kL, float64 kC, float64 kH',  # noqa
        out_params='W out',
        operation="""

        // magnitude of (a, b) is the chroma
        // note: pow(25.0, 7.0) = 6103515625.0
        F Cbar, c7, scale;
        Cbar = 0.5 * (hypot(a1, b1) + hypot(a2, b2));
        c7 = pow(Cbar, static_cast<F>(7.0));
        scale = 0.5 * (1 - sqrt(c7 / (c7 + 6103515625.0)));
        scale += 1.0;

        // convert cartesian coordinates to polar (non-standard theta range!)
        // uses [0, 2*M_PI] instead of [-M_PI, M_PI]
        F h1, h2, C1, C2;
        C1 = hypot(a1 * scale, b1);
        h1 = atan2(b1, a1 * scale);
        if (h1 < 0) {
          h1 += 2 * M_PI;
        }
        C2 = hypot(a2 * scale, b2);
        h2 = atan2(b2, a2 * scale);
        if (h2 < 0) {
          h2 += 2 * M_PI;
        }

        // lightness term
        F tmpL, SL, L_term;
        tmpL = 0.5 * (L1 + L2) - 50;
        tmpL *= tmpL;
        SL = 1.0 + 0.015 * tmpL / sqrt(20.0 + tmpL);
        L_term = (L2 - L1) / (kL * SL);

        // chroma term
        F SC, C_term;
        Cbar = 0.5 * (C1 + C2);  // new coordiantes
        SC = 1 + 0.045 * Cbar;
        C_term = (C2 - C1) / (kC * SC);

        // hue term
        F h_diff, h_sum, CC;
        h_diff = h2 - h1;
        h_sum = h1 + h2;
        CC = C1 * C2;

        F dH, dH_term;
        dH = h_diff;
        if (CC == 0.) {
            dH = 0.;  // if r == 0, dtheta == 0
        } else {
            if (h_diff > M_PI) {
                dH -= 2 * M_PI;
            } else if (h_diff < -M_PI) {
                dH += 2 * M_PI;
            }
        }
        dH_term = 2 * sqrt(CC) * sin(dH / 2);

        F Hbar;
        Hbar = h_sum;
        if (CC == 0.0) {
            Hbar *= 2.0;
        } else if (abs(h_diff) > M_PI) {
            if (h_sum < 2 * M_PI) {
                Hbar += 2 * M_PI;
            } else if (h_sum >= 2 * M_PI) {
                Hbar -= 2 * M_PI;
            }
        }
        Hbar *= 0.5;

        F SH, H_term;
        F rad30 = 0.5235987755982988;
        F rad6 = 0.10471975511965978;
        F rad63 = 1.0995574287564276;
        SH = (1.0 -
              0.17 * cos(Hbar - rad30) +
              0.24 * cos(2.0 * Hbar) +
              0.32 * cos(3.0 * Hbar + rad6) -
              0.20 * cos(4.0 * Hbar - rad63));
        SH *= 0.015 * Cbar;
        SH += 1.0;
        H_term = dH_term / (kH * SH);

        // hue rotation
        F Rc, tmp, dtheta;
        // note: pow(25.0, 7.0) = 6103515625.0
        // recall that c, h are polar coordiantes.  c==r, h==theta

        F R_term;
        c7 = pow(Cbar, static_cast<F>(7.0));
        Rc = 2.0 * sqrt(c7 / (c7 + 6103515625.0));
        tmp = (Hbar * 180.0 / M_PI - 275.0) / 25.0;
        tmp *= tmp;
        dtheta = rad30 * exp(-tmp);
        R_term = -sin(2 * dtheta) * Rc * C_term * H_term;

        // put it all together
        F dE2;
        dE2 = L_term * L_term;
        dE2 += C_term * C_term;
        dE2 += H_term * H_term;
        dE2 += R_term;
        dE2 = max(dE2, 0.0);
        out = sqrt(dE2);

        """,
        name='cucim_skimage_ciede2000_kernel'
    )


def deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1, *, channel_axis=-1):
    """Color difference as given by the CIEDE 2000 standard.

    CIEDE 2000 is a major revision of CIDE94.  The perceptual calibration is
    largely based on experience with automotive paint on smooth surfaces.

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    kL : float (range), optional
        lightness scale factor, 1 for "acceptably close"; 2 for "imperceptible"
        see deltaE_cmc
    kC : float (range), optional
        chroma scale factor, usually 1
    kH : float (range), optional
        hue scale factor, usually 1
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

    Returns
    -------
    deltaE : array_like
        The distance between `lab1` and `lab2`

    Notes
    -----
    CIEDE 2000 assumes parametric weighting factors for the lightness, chroma,
    and hue (`kL`, `kC`, `kH` respectively).  These default to 1.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
           :DOI:`10.1364/AO.33.008069`
    .. [3] M. Melgosa, J. Quesada, and E. Hita, "Uniformity of some recent
           color metrics tested with an accurate color-difference tolerance
           dataset," Appl. Opt. 33, 8069-8077 (1994).
    """  # noqa
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    warnings.warn(
        "The numerical accuracy of this function on the GPU is reduced "
        "relative to the CPU version"
    )
    channel_axis = channel_axis % lab1.ndim
    unroll = False
    if lab1.ndim == 1 and lab2.ndim == 1:
        unroll = True
        if lab1.ndim == 1:
            lab1 = lab1[None, :]
        if lab2.ndim == 1:
            lab2 = lab2[None, :]
        channel_axis += 1
    L1, a1, b1 = cp.moveaxis(lab1, source=channel_axis, destination=0)[:3]
    L2, a2, b2 = cp.moveaxis(lab2, source=channel_axis, destination=0)[:3]

    # cide2000 has four terms to delta_e:
    # 1) Luminance term
    # 2) Hue term
    # 3) Chroma term
    # 4) hue Rotation term

    # core computation
    dE2 = cp.empty_like(L1)
    kernel = _get_ciede2000_kernel()
    kernel(L1, a1, b1, L2, a2, b2, kL, kC, kH, dE2)

    if unroll:
        dE2 = dE2[0]
    return dE2


@cp.memoize()
def _get_cmc_kernel():

    return cp.ElementwiseKernel(
        in_params='F L1, F C1, F L2, F C2, F h1, F dH2, float64 kL, float64 kC',  # noqa
        out_params='F dE2',
        operation="""

        F dC, dL, T, c1_4, f, SL, SC, SH_sq, tmp;
        F deg2rad164 = 2.8623399732707004;
        F deg2rad345 = 6.021385919380437;

        dC = C1 - C2;
        dL = L1 - L2;
        if ((h1 >= deg2rad164) && (h1 <= deg2rad345)) {
            // deg2rad(168) = 2.9321531433504737
            T = 0.56 + 0.2 * abs(cos(h1 + static_cast<F>(2.9321531433504737)));
        } else {
            // deg2rad(35) = 0.6108652381980153
            T = 0.36 + 0.4 * abs(cos(h1 + static_cast<F>(0.6108652381980153)));
        }
        c1_4 = C1 * C1;
        c1_4 *= c1_4;
        f = sqrt(c1_4 / (c1_4 + 1900.0));

        if (L1 < 16) {
            SL = 0.511;
        } else {
            SL = 0.040975 * L1 / (1.0 + 0.01765 * L1);
        }
        SC = 0.638 + 0.0638 * C1 / (1.0 + 0.0131 * C1);
        SH_sq = SC * (f * T + 1.0 - f);
        SH_sq *= SH_sq;

        dE2 = (dL / (kL * SL));
        dE2 *= dE2;
        tmp = (dC / (kC * SC));
        tmp *= tmp;
        dE2 += tmp;
        dE2 += dH2 / SH_sq;
        dE2 = max(dE2, 0.0);
        dE2 = sqrt(dE2);
        """,
        name='cucim_skimage_cmc_kernel'
    )


def deltaE_cmc(lab1, lab2, kL=1, kC=1, *, channel_axis=-1):
    """Color difference from the  CMC l:c standard.

    This color difference was developed by the Colour Measurement Committee
    (CMC) of the Society of Dyers and Colourists (United Kingdom). It is
    intended for use in the textile industry.

    The scale factors `kL`, `kC` set the weight given to differences in
    lightness and chroma relative to differences in hue.  The usual values are
    ``kL=2``, ``kC=1`` for "acceptability" and ``kL=1``, ``kC=1`` for
    "imperceptibility".  Colors with ``dE > 1`` are "different" for the given
    scale factors.

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

    Returns
    -------
    dE : array_like
        distance between colors `lab1` and `lab2`

    Notes
    -----
    deltaE_cmc the defines the scales for the lightness, hue, and chroma
    in terms of the first color.  Consequently
    ``deltaE_cmc(lab1, lab2) != deltaE_cmc(lab2, lab1)``

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html
    .. [3] F. J. J. Clarke, R. McDonald, and B. Rigg, "Modification to the
           JPC79 colour-difference formula," J. Soc. Dyers Colour. 100, 128-132
           (1984).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    lab1 = cp.moveaxis(lab1, source=channel_axis, destination=0)
    lab2 = cp.moveaxis(lab2, source=channel_axis, destination=0)
    L1, C1, h1 = lab2lch(lab1, channel_axis=0)[:3]
    L2, C2, h2 = lab2lch(lab2, channel_axis=0)[:3]

    dH2 = get_dH2(lab1, lab2, channel_axis=0)

    dE2 = cp.zeros_like(L1)
    kernel = _get_cmc_kernel()
    kernel(L1, C1, L2, C2, h1, dH2, kL, kC, dE2)

    return dE2


@cp.memoize()
def _get_dH2_kernel():

    return cp.ElementwiseKernel(
        in_params='F a1, F b1, F a2, F b2',  # noqa
        out_params='G out',
        operation="""
        // magnitude of (a, b) is the chroma
        double C1 = hypot(a1, b1);
        double C2 = hypot(a2, b2);

        // have to keep double here for out_temp for accuracy
        double out_temp = C1 * C2;
        out_temp -= a1 * a2;
        out_temp -= b1 * b2;
        out_temp *= 2;
        out = out_temp;
        """,
        name='cucim_skimage_ciede2000'
    )


def get_dH2(lab1, lab2, *, channel_axis=-1):
    """squared hue difference term occurring in deltaE_cmc and deltaE_ciede94

    Despite its name, "dH" is not a simple difference of hue values.  We avoid
    working directly with the hue value, since differencing angles is
    troublesome.  The hue term is usually written as:
        c1 = sqrt(a1**2 + b1**2)
        c2 = sqrt(a2**2 + b2**2)
        term = (a1-a2)**2 + (b1-b2)**2 - (c1-c2)**2
        dH = sqrt(term)

    However, this has poor roundoff properties when a or b is dominant.
    Instead, ab is a vector with elements a and b.  The same dH term can be
    re-written as:
        |ab1-ab2|**2 - (|ab1| - |ab2|)**2
    and then simplified to:
        2*|ab1|*|ab2| - 2*dot(ab1, ab2)
    """
    # This function needs double precision internally for accuracy
    float_dtype = _supported_float_type([lab1.dtype, lab2.dtype])
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=False)
    a1, b1 = cp.moveaxis(lab1, source=channel_axis, destination=0)[1:3]
    a2, b2 = cp.moveaxis(lab2, source=channel_axis, destination=0)[1:3]

    out = cp.empty(a1.shape, dtype=float_dtype)
    kernel = _get_dH2_kernel()
    return kernel(a1, b1, a2, b2, out)
