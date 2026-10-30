"""Two ways of building a filter curve for speclite.

Which curve belongs to which band stays in the instrument's own module; only
the mechanics live here. Registration is idempotent.
"""

import numpy as np
import speclite.filters
from astropy import units as u

__all__ = ["tophat", "svo"]


def _new(group, band):
    try:
        speclite.filters.load_filter("%s-%s" % (group, band))
        return False
    except ValueError:
        return True


def _register(group, band, wavelength, response):
    speclite.filters.FilterResponse(
        wavelength=wavelength * u.Angstrom,
        response=response,
        meta=dict(group_name=group, band_name=band),
    )


def tophat(group, edges):
    """Square filters. *edges* maps band to ``(low, high)`` in Angstrom."""
    for band, (low, high) in edges.items():
        if not _new(group, band):
            continue
        wavelength = np.linspace(low - 100.0, high + 100.0, 1000)
        _register(group, band, wavelength,
                  ((wavelength >= low) & (wavelength <= high)).astype(float))


def svo(group, ids):
    """Measured curves from the SVO Filter Profile Service.

    *ids* maps band to an SVO filter id such as ``"HST/ACS_WFC.F606W"``.
    """
    from astroquery.svo_fps import SvoFps

    for band, filter_id in ids.items():
        if not _new(group, band):
            continue
        table = SvoFps.get_transmission_data(filter_id)
        wavelength = np.asarray(table["Wavelength"], dtype=float)
        response = np.asarray(table["Transmission"], dtype=float)
        # speclite requires the response to reach zero at both ends
        if response[0] != 0.0:
            wavelength = np.insert(wavelength, 0, wavelength[0] - 1.0)
            response = np.insert(response, 0, 0.0)
        if response[-1] != 0.0:
            wavelength = np.append(wavelength, wavelength[-1] + 1.0)
            response = np.append(response, 0.0)
        _register(group, band, wavelength, response)
