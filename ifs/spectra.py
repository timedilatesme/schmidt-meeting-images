"""Observed-frame spectra for the components of an slsim lens.

Every spectrum is f_lambda in erg/s/cm^2/A on a caller-supplied wavelength
grid, scaled so that it reproduces the slsim magnitude the imaging uses.

* quasar: qsogen with slsim's AGILE parameters, split into the accretion-disc
  continuum and the emission lines, host-galaxy term off (the host is its own
  component). Normalised to the unlensed LSST i magnitude, which is the band
  slsim anchors its quasar SEDs to.
* galaxies: the five kcorrect templates SkyPy draws its galaxies from,
  non-negatively fitted to the component's magnitudes in every band slsim
  carries for it.
"""

import numpy as np
import speclite.filters
from scipy.optimize import nnls
from skypy.galaxies.spectrum import kcorrect

from slsim.ImageSimulation.image_quality_lenstronomy import _OBSERVATORY_REGISTRY
from slsim.Sources.SourceCatalogues.QuasarCatalog.qsogen import Quasar_sed, params_agile

__all__ = ["quasar", "galaxy", "speclite_name", "ab_magnitude"]

ANCHOR = "i"


def speclite_name(band):
    for info in _OBSERVATORY_REGISTRY.values():
        if band in info["bands"] and info.get("speclite_fmt"):
            return info["speclite_fmt"](band)
    raise KeyError(band)


def ab_magnitude(band, wavelength, flux):
    curve = speclite.filters.load_filter(speclite_name(band))
    inside = (wavelength > curve.wavelength[0]) & (wavelength < curve.wavelength[-1])
    if not inside.any():
        return np.nan
    pad = curve.wavelength[[0, -1]]
    grid = np.concatenate([[pad[0] - 1], wavelength[inside], [pad[1] + 1]])
    values = np.concatenate([[0], flux[inside], [0]])
    return float(curve.get_ab_magnitude(values, grid))


def quasar(source, wavelength, cosmo):
    """(continuum, lines) of the point source, unlensed."""
    rest = np.logspace(2.0, 4.6, 40000)

    def sed(**kwargs):
        return Quasar_sed(z=source.redshift, params=params_agile, wavlen=rest,
                          cosmo=cosmo, gflag=False, **kwargs)

    full, bare = sed(), sed(scal_emline=0.0)
    target = source.point_source_magnitude(ANCHOR)
    scale = 10 ** (-0.4 * (target - ab_magnitude(ANCHOR, full.wavred, full.flux)))
    continuum = np.interp(wavelength, bare.wavred, bare.flux, left=0, right=0)
    total = np.interp(wavelength, full.wavred, full.flux, left=0, right=0)
    return continuum * scale, np.clip(total - continuum, 0, None) * scale


def galaxy(redshift, magnitude, bands, wavelength):
    """kcorrect-template spectrum fitted to ``magnitude(band)`` over *bands*."""
    rest = kcorrect.wavelength.to_value("Angstrom")
    observed = rest * (1 + redshift)
    templates = kcorrect.templates.value / (1 + redshift)
    used, target = [], []
    for band in bands:
        try:
            value = float(magnitude(band))
        except Exception:
            continue
        if np.isfinite(value):
            used.append(band)
            target.append(10 ** (-0.4 * value))
    target = np.array(target)
    design = np.array([[10 ** (-0.4 * ab_magnitude(band, observed, t)) for band in used]
                       for t in templates]).T
    weight = 1 / (0.05 * target)
    coeff, _ = nnls(design * weight[:, None], target * weight)
    return np.interp(wavelength, observed, coeff @ templates, left=0, right=0)
