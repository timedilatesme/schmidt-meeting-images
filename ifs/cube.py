"""Simulate an IFS data cube of an slsim lens.

The cube is the sum of components, each a fixed spatial morphology times its
own spectrum, blurred by the wavelength-dependent PSF:

    cube(lambda) = PSF_lambda * [ lens(x) S_lens(lambda) + host(x) S_host(lambda)
                   + sum_k delta(x - x_k) |mu_k| (C(lambda) m_k(lambda) + L(lambda)) ]

* lens(x), host(x): slsim's unconvolved deflector light and lensed host, in
  maggies per pixel of the reference band, on a grid oversampled 5x.
* x_k, mu_k: slsim's point-source image positions and macro magnifications.
* C, L: the quasar continuum and emission lines; m_k(lambda) is the static
  microlensing of the continuum in image k, from slsim's accretion-disc
  morphology evaluated at each wavelength on one magnification map per image.
  The broad and narrow lines come from regions far larger than the disc and
  are left unmicrolensed; the partial microlensing of the broad-line region
  seen in some systems (e.g. Sluse et al. 2012) is neglected.

The PSF is evaluated at a few wavelengths and the convolved component images
interpolated between them. Spectra are smoothed to R(lambda) before sampling.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from slsim.ImageSimulation.image_quality_lenstronomy import _OBSERVATORY_REGISTRY
from slsim.ImageSimulation.image_simulation import sharp_image

from . import spectra

__all__ = ["Cube", "simulate", "chromatic_microlensing"]

REFERENCE_BAND = "LZr"


@dataclass
class Cube:
    wavelength: np.ndarray
    model: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray
    spaxel: float
    image_pixels: np.ndarray
    magnification: np.ndarray
    spectra: dict = field(default_factory=dict)
    microlensing: tuple = None

    @property
    def extent(self):
        ny, nx = self.flux.shape[1:]
        return (nx / 2 * self.spaxel, -nx / 2 * self.spaxel,
                -ny / 2 * self.spaxel, ny / 2 * self.spaxel)

    def collapse(self, low, high, cube=None):
        cube = self.flux if cube is None else cube
        inside = (self.wavelength >= low) & (self.wavelength <= high)
        return cube[inside].mean(axis=0)

    def aperture(self, row, col, radius=2.5, cube=None):
        cube = self.flux if cube is None else cube
        y, x = np.indices(cube.shape[1:])
        mask = np.hypot(y - row, x - col) <= radius
        return cube[:, mask].sum(axis=1)


def chromatic_microlensing(lens, wavelength_nm, band="i", seed=None):
    """Static microlensing Delta-mag of every image at each wavelength.

    One magnification map per image is drawn once; only the accretion-disc
    kernel changes with wavelength, so the chromatic signal is coherent.

    :return: array (n_images, n_wavelengths); negative is brighter
    """
    if seed is not None:
        np.random.seed(seed)
    lens._point_source_magnitude_microlensing(band=band, time=np.array([0.0]),
                                              source_index=0)
    model = lens.microlensing_model_class(0)
    kwargs = lens.source(0).prepare_microlensing_kwargs(
        band=band, cosmo=lens.cosmo)["kwargs_source_morphology"]
    delta = []
    for nm in wavelength_nm:
        kwargs.update(observing_wavelength_band=None,
                      observer_frame_wavelength_in_nm=float(nm))
        model.update_source_morphology(dict(kwargs))
        delta.append(model.generate_point_source_microlensing_magnitudes(time=0.0))
    return np.array(delta).T


def _resolution_matrix(fine, wavelength, resolution):
    sigma = wavelength / resolution / 2.3548
    weight = np.exp(-0.5 * ((fine[None] - wavelength[:, None]) / sigma[:, None]) ** 2)
    return weight / weight.sum(axis=1, keepdims=True)


def _bin(image, factor):
    ny, nx = image.shape
    return image.reshape(ny // factor, factor, nx // factor, factor).sum(axis=(1, 3))


def _deposit(shape, row, col):
    image = np.zeros(shape)
    r0, c0 = int(np.floor(row)), int(np.floor(col))
    dr, dc = row - r0, col - c0
    for r, c, w in ((r0, c0, (1 - dr) * (1 - dc)), (r0 + 1, c0, dr * (1 - dc)),
                    (r0, c0 + 1, (1 - dr) * dc), (r0 + 1, c0 + 1, dr * dc)):
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            image[r, c] += w
    return image


def simulate(lens, instrument, microlensing=True, microlensing_wavelengths=None,
             oversample=5, n_psf=16, psf_size=1.6, seed=None):
    """IFS cube of *lens* as *instrument* would record it.

    :param lens: slsim Lens with a quasar point source
    :param instrument: ``LazuliIFS``
    :param microlensing: include static chromatic microlensing of the continuum;
        True draws a realization, or pass ``(nm, delta_mag)`` from
        :func:`chromatic_microlensing` to reuse one
    :param microlensing_wavelengths: observed-frame nm at which to evaluate it
    :return: ``Cube``; x is East-left like ``util.render``
    """
    rng = np.random.default_rng(seed)
    wave = instrument.wavelengths()
    fine = np.arange(wave[0] - 300, wave[-1] + 300, 1.0)
    smooth = _resolution_matrix(fine, wave, instrument.resolution(wave))

    source = lens.source(0)
    bands = [b for info in _OBSERVATORY_REGISTRY.values() for b in info["bands"]]
    continuum, lines = spectra.quasar(source, fine, lens.cosmo)
    sed = {"lens": spectra.galaxy(lens.deflector_redshift, lens.deflector.magnitude, bands, fine),
           "host": spectra.galaxy(source.redshift, source.extended_source_magnitude, bands, fine)}
    for name in sed:
        maggies = 10 ** (-0.4 * spectra.ab_magnitude(REFERENCE_BAND, fine, sed[name]))
        sed[name] = smooth @ (sed[name] / maggies)
    continuum, lines = smooth @ continuum, smooth @ lines

    x, y = lens.point_source_image_positions()[0]
    magnification = np.abs(lens.point_source_magnification()[0])
    boost = np.ones((len(x), len(wave)))
    curve = None
    if microlensing is True:
        nm = (np.geomspace(400, 1700, 10) if microlensing_wavelengths is None
              else np.asarray(microlensing_wavelengths))
        microlensing = (nm, chromatic_microlensing(lens, nm, seed=seed))
    if microlensing:
        curve = nm, delta = microlensing
        for k in range(len(x)):
            boost[k] = 10 ** (-0.4 * np.interp(np.log(wave / 10), np.log(nm), delta[k]))

    pixel = instrument.spaxel / oversample
    ny, nx = (int(round(f / instrument.spaxel)) | 1 for f in instrument.field)
    shape = (ny * oversample, nx * oversample)
    side = shape[1]
    crop = slice((side - shape[0]) // 2, (side - shape[0]) // 2 + shape[0])
    morphology = {}
    for name, flags in (("lens", (False, True)), ("host", (True, False))):
        image = sharp_image(lens, REFERENCE_BAND, 0.0, pixel, side,
                            with_source=flags[0], with_deflector=flags[1])
        morphology[name] = image[crop, ::-1]
    centre = (np.array(shape) - 1) / 2
    rows, cols = y / pixel + centre[0], -x / pixel + centre[1]
    points = [_deposit(shape, r, c) for r, c in zip(rows, cols)]

    psf_wave = np.geomspace(wave[0], wave[-1], n_psf)
    size = int(psf_size / pixel) | 1
    stack = []
    for lam in psf_wave:
        kernel = instrument.psf(lam, pixel, size)
        stack.append([_bin(fftconvolve(image, kernel, mode="same"), oversample)
                      for image in [morphology["lens"], morphology["host"], *points]])
    stack = np.array(stack)  # (n_psf, n_component, ny, nx)
    position = np.interp(np.log(wave), np.log(psf_wave), np.arange(n_psf))
    low = np.clip(np.floor(position).astype(int), 0, n_psf - 2)
    frac = (position - low)[:, None, None, None]
    components = (1 - frac) * stack[low] + frac * stack[low + 1]

    model = (components[:, 0] * sed["lens"][:, None, None]
             + components[:, 1] * sed["host"][:, None, None])
    for k in range(len(x)):
        spectrum = magnification[k] * (continuum * boost[k] + lines)
        model += components[:, 2 + k] * spectrum[:, None, None]
    flux, sigma = instrument.observe(model, wave, rng=rng)

    return Cube(
        wavelength=wave, model=model, flux=flux, sigma=sigma, spaxel=instrument.spaxel,
        image_pixels=np.column_stack([rows, cols]) / oversample - 0.5 + 0.5 / oversample,
        magnification=magnification, microlensing=curve,
        spectra=dict(lens=sed["lens"], host=sed["host"], continuum=continuum,
                     lines=lines, boost=boost),
    )
