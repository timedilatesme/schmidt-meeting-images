"""Lazuli Integral Field Spectrograph (IFS) instrument model.

Quoted in Roy et al. 2026 (arXiv:2601.02556): image slicer with a prism,
continuous 400-1700 nm, R ~ 500 at the blue end falling to ~100 at 1 um and
rising to ~200 at 1.7 um; Narrow Field 2.3" x 4.6" at 40 mas, Wide Field
4.6" x 8.8" at 80 mas; instrument throughput > 40 % (400-1000 nm) and > 50 %
(1000-1700 nm); H4RG-10, QE > 50 % at 800 nm and > 70 % at 1200 nm, CDS read
noise <= 25 e-, dark <= 0.01 e-/s/pix.

Notional here: the exact R(lambda) curve, the telescope reflectivity, the QE
curve between the quoted points, the zodiacal sky spectrum, two detector
pixels per cube voxel, and 12 e- effective read noise for up-the-ramp
sampling of a 900 s exposure (the 25 e- CDS figure is for a single pair).
"""

from dataclasses import dataclass

import numpy as np

from observatories.lazuli import DIAMETER, JITTER, airy_kernel

__all__ = ["LazuliIFS", "MODES"]

H_C = 6.62607e-27 * 2.99792458e18  # erg Angstrom

MODES = {"NF": dict(spaxel=0.04, field=(2.3, 4.6)),
         "WF": dict(spaxel=0.08, field=(4.6, 8.8))}

_R = ([4000, 5000, 6500, 8000, 10000, 13000, 17000],
      [500, 330, 190, 130, 100, 135, 200])
_INSTRUMENT = ([4000, 10000, 10001, 17000], [0.42, 0.42, 0.52, 0.52])
_QE = ([4000, 6000, 8000, 12000, 17000], [0.35, 0.45, 0.55, 0.75, 0.75])
_TELESCOPE = 0.85
_SKY_AB = ([4000, 5000, 6000, 8000, 10000, 13000, 17000],
           [23.6, 23.2, 23.0, 22.7, 22.6, 22.6, 22.7])  # AB mag / arcsec^2


@dataclass
class LazuliIFS:
    mode: str = "NF"
    exposure_time: float = 900.0
    num_exposures: int = 4
    read_noise: float = 12.0
    dark_current: float = 0.01
    pixels_per_voxel: float = 2.0
    wavelength_range: tuple = (4000.0, 17000.0)

    @property
    def spaxel(self):
        return MODES[self.mode]["spaxel"]

    @property
    def field(self):
        return MODES[self.mode]["field"]

    @property
    def area(self):
        return np.pi * (50 * DIAMETER) ** 2

    @staticmethod
    def resolution(wavelength):
        return np.exp(np.interp(wavelength, _R[0], np.log(_R[1])))

    @staticmethod
    def throughput(wavelength):
        return (_TELESCOPE * np.interp(wavelength, *_INSTRUMENT)
                * np.interp(wavelength, *_QE))

    def wavelengths(self):
        """Channel centres, Nyquist sampling R(lambda)."""
        grid = [self.wavelength_range[0]]
        while grid[-1] < self.wavelength_range[1]:
            grid.append(grid[-1] * (1 + 0.5 / self.resolution(grid[-1])))
        return np.array(grid)

    def sky(self, wavelength):
        """Zodiacal f_lambda per spaxel, erg/s/cm^2/A."""
        f_nu = 3631e-23 * 10 ** (-0.4 * np.interp(wavelength, *_SKY_AB))
        return f_nu * 2.99792458e18 / wavelength ** 2 * self.spaxel ** 2

    def psf(self, wavelength, pixel_scale, size):
        return airy_kernel([wavelength], [1.0], pixel_scale, size, jitter=JITTER)

    def electrons_per_flux(self, wavelength, channel_width):
        """Electrons collected per unit f_lambda over the full exposure."""
        time = self.exposure_time * self.num_exposures
        return (wavelength / H_C * self.area * self.throughput(wavelength)
                * channel_width * time)

    def observe(self, flux, wavelength, rng=None):
        """Add sky, dark and read noise to a noiseless cube.

        :param flux: f_lambda cube (n_wave, ny, nx), erg/s/cm^2/A per spaxel
        :return: (noisy f_lambda cube, 1-sigma cube), sky subtracted
        """
        rng = np.random.default_rng(rng)
        width = np.gradient(wavelength)[:, None, None]
        gain = self.electrons_per_flux(wavelength[:, None, None], width)
        source = flux * gain
        sky = self.sky(wavelength)[:, None, None] * gain
        dark = self.dark_current * self.exposure_time * self.num_exposures * self.pixels_per_voxel
        read = self.num_exposures * self.pixels_per_voxel * self.read_noise ** 2
        variance = np.clip(source, 0, None) + sky + dark + read
        noisy = source + rng.normal(size=flux.shape) * np.sqrt(variance)
        return noisy / gain, np.sqrt(variance) / gain
