"""Lazuli Wide-field Context Camera (WCC) imaging configuration.

Lazuli is a 3.06 m unobscured (off-axis) space telescope, diffraction limited
with Strehl > 0.8 at 633 nm (Roy et al. 2026, arXiv:2601.02556). The WCC
images 350-1000 nm through notional Sloan-like filters onto Sony IMX455 CMOS
sensors at 17 mas per pixel.

Everything not stated in that paper is notional and chosen here:

* filter shapes are the SDSS curves (``sdss2010noatm``) scaled to peak at the
  quoted end-to-end throughput of 0.5;
* zero points follow from that throughput and the collecting area;
* sky is a typical zodiacal background at moderate ecliptic latitude;
* exposures are a short follow-up visit, 4 x 300 s per band.

``psf_type="PIXEL"`` gives a polychromatic Airy pattern averaged over the
filter and blurred by pointing jitter; the Gaussian alternative uses the Airy
core FWHM, 1.03 lambda / D.
"""

import copy

import lenstronomy.Util.util as util
import numpy as np
import speclite.filters
from scipy.special import j1

from . import _filters

__all__ = ["Lazuli", "BANDS", "configure_speclite_filters", "DIAMETER",
           "THROUGHPUT", "airy_kernel"]

DIAMETER = 3.06  # m
THROUGHPUT = 0.5  # telescope + instrument + detector, at filter peak
PIXEL_SCALE = 0.017  # arcsec, IMX455
JITTER = 0.004  # arcsec, 1-sigma per axis
OVERSAMPLE = 3
KERNEL_PIXELS = 75  # detector pixels, 1.3"

_SDSS = {"LZg": "g", "LZr": "r", "LZi": "i", "LZz": "z"}
_SKY = {"LZg": 23.0, "LZr": 22.8, "LZi": 22.7, "LZz": 22.6}

Lazuli_camera = {"read_noise": 2.0, "pixel_scale": PIXEL_SCALE, "ccd_gain": 1.0}

BANDS = list(_SDSS)


def configure_speclite_filters():
    """Register ``Lazuli-LZg`` ... ``Lazuli-LZz``."""
    for band, sdss in _SDSS.items():
        if not _filters._new("Lazuli", band):
            continue
        curve = speclite.filters.load_filter("sdss2010noatm-%s" % sdss)
        wavelength = np.asarray(curve.wavelength, dtype=float)
        _filters._register("Lazuli", band, wavelength,
                           THROUGHPUT * curve.response / curve.response.max())


def _curve(band):
    configure_speclite_filters()
    return speclite.filters.load_filter("Lazuli-%s" % band)


def zero_point(band):
    """AB magnitude giving 1 e-/s through the full aperture."""
    curve = _curve(band)
    wavelength = np.asarray(curve.wavelength, dtype=float)
    area = np.pi * (50 * DIAMETER) ** 2  # cm^2
    photons = 3631e-23 / 6.62607e-27 * np.trapezoid(curve.response / wavelength, wavelength)
    return 2.5 * np.log10(area * photons)


def fwhm(band):
    return 1.03 * _curve(band).effective_wavelength.value * 1e-10 / DIAMETER * 206264.806


def airy_kernel(wavelengths, weights, pixel_scale, size, jitter=JITTER):
    """Normalised polychromatic Airy PSF of the unobscured aperture.

    :param wavelengths: Angstrom
    :param weights: relative photon weight of each wavelength
    :param pixel_scale: arcsec per kernel pixel
    :param size: kernel width in pixels, odd
    :param jitter: Gaussian pointing jitter, arcsec 1-sigma
    """
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1] * pixel_scale
    radius = np.hypot(x, y) / 206264.806
    kernel = np.zeros_like(radius)
    for wavelength, weight in zip(np.atleast_1d(wavelengths), np.atleast_1d(weights)):
        u = np.pi * DIAMETER * radius / (wavelength * 1e-10)
        with np.errstate(invalid="ignore", divide="ignore"):
            pattern = np.where(u > 0, (2 * j1(u) / u) ** 2, 1.0)
        kernel += weight * pattern / pattern.sum()
    if jitter > 0:
        frequency = np.fft.fftfreq(size, d=pixel_scale)
        fx, fy = np.meshgrid(frequency, frequency)
        transfer = np.exp(-2 * (np.pi * jitter) ** 2 * (fx ** 2 + fy ** 2))
        kernel = np.fft.fftshift(np.real(np.fft.ifft2(
            np.fft.fft2(np.fft.ifftshift(kernel)) * transfer)))
    kernel = np.clip(kernel, 0, None)
    return kernel / kernel.sum()


def _band_kernel(band):
    curve = _curve(band)
    wavelength = np.linspace(*curve.wavelength[[0, -1]], 25)
    weight = curve(wavelength) * wavelength  # photon counting
    return airy_kernel(wavelength, weight, PIXEL_SCALE / OVERSAMPLE,
                       KERNEL_PIXELS * OVERSAMPLE | 1)


class Lazuli(object):
    """Lazuli WCC instrument and observation configuration."""

    def __init__(self, band="LZr", psf_type="GAUSSIAN"):
        if band not in _SDSS:
            raise ValueError("band %s not supported! Choose from %s." % (band, BANDS))
        self.obs = {
            "exposure_time": 300.0,
            "num_exposures": 4,
            "sky_brightness": _SKY[band],
            "magnitude_zero_point": zero_point(band),
            "seeing": fwhm(band),
            "psf_type": "GAUSSIAN",
        }
        if psf_type == "PIXEL":
            del self.obs["seeing"]
            self.obs.update(psf_type="PIXEL", kernel_point_source=_band_kernel(band),
                            point_source_supersampling_factor=OVERSAMPLE)
        self.camera = copy.deepcopy(Lazuli_camera)

    def kwargs_single_band(self):
        return util.merge_dicts(self.camera, self.obs)
