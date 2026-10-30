"""Observatory configurations, one module per instrument.

Call :func:`register_all` once, before any imaging. After that every band below
is usable through ``simulate_image(..., observatory=...)`` and through the
helpers in ``util``.

LSST, Roman and Euclid are already registered by slsim from lenstronomy's own
configs and are not repeated here.

Band names are unique across all observatories: slsim maps a band name to
exactly one observatory and takes the first match, so a duplicate would
silently resolve to the wrong instrument.

Every band also has a speclite filter curve, registered by
:func:`configure_all_filters`. Those are what the population pipeline
synthesises magnitudes through; static imaging never touches them.
"""

from slsim.ImageSimulation.image_quality_lenstronomy import register_observatory

from . import lazuli

__all__ = ["register_all", "configure_all_filters", "OBSERVATORIES"]

#: name, config class, bands, and the band -> speclite filter name map. slsim
#: inverts these with name.split("-")[1], so no band name may contain a hyphen.
OBSERVATORIES = [
    ("Lazuli", lazuli.Lazuli, lazuli.BANDS, lambda band: "Lazuli-%s" % band),
]

#: Modules in registration order, for :func:`configure_all_filters`.
_MODULES = [lazuli]


def register_all():
    """Register every observatory defined in this package."""
    for name, config_class, bands, speclite_fmt in OBSERVATORIES:
        register_observatory(name, config_class, bands, speclite_fmt=speclite_fmt)


def configure_all_filters():
    """Register every band's filter curve with speclite. Safe to call more
    than once.

    :return: list of speclite filter names, in OBSERVATORIES order
    """
    for module in _MODULES:
        module.configure_speclite_filters()
    return [fmt(band) for _, _, bands, fmt in OBSERVATORIES for band in bands]
