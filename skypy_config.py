"""Extend slsim's SkyPy galaxy config to cover every registered band.

slsim ships ``euclid-roman-lsst-like.yml``, which draws a galaxy population and
gives each galaxy a magnitude in Euclid, Roman and LSST bands. :func:`generate`
rewrites its two filter-dependent lines so the same population also carries
magnitudes for the instruments configured in ``observatories``.

The template's own bands are read from the template rather than listed here,
so a new slsim that adds a band (as it did LSST u) does not desynchronise the
filter list from the magnitude columns.

Only bands whose filter stays above the 600 A rest-frame floor of the kcorrect
templates across the whole ``z_range`` can be included; anything bluer makes
the pipeline fail outright.
"""

import os
import re

import slsim

__all__ = ["BASE_BANDS", "base_bands", "generate"]

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(slsim.__file__)),
                        "data/SkyPy/euclid-roman-lsst-like.yml")

_FILTERS = re.compile(r"filters: \[(.*?)\]", re.S)
# The line naming the magnitude columns, e.g. "    mag_VIS, mag_Y, ...: !...".
# Requiring a comma keeps it off scalar keys such as "mag_lim: 30".
_MAG_KEYS = re.compile(r"^(\s*)mag_\w+(?:, mag_\w+)+:", re.M)


def base_bands(template=TEMPLATE):
    """The (speclite filter, band name) pairs already in *template*."""
    config = open(template).read()
    filters = re.findall(r"'([^']+)'", _FILTERS.search(config).group(1))
    return [(name, name.split("-", 1)[1]) for name in filters]


#: The (speclite filter, band name) pairs in the installed slsim's config.
BASE_BANDS = base_bands()


def generate(path, extra_bands=()):
    """Write a SkyPy config covering :data:`BASE_BANDS` plus *extra_bands*.

    :param path: where to write the YAML
    :param extra_bands: iterable of ``(speclite filter, band name)``
    :return: the list of band names the config will produce ``mag_<band>`` for
    """
    bands = base_bands() + list(extra_bands)
    config = open(TEMPLATE).read()
    config = _FILTERS.sub(
        "filters: [%s]" % ", ".join("'%s'" % f for f, _ in bands), config, count=1)
    keys = ", ".join("mag_%s" % b for _, b in bands) + ":"
    config, n = _MAG_KEYS.subn(lambda m: m.group(1) + keys, config)
    if n == 0:
        raise RuntimeError("no magnitude column line found in %s" % TEMPLATE)
    with open(path, "w") as stream:
        stream.write(config)
    return [b for _, b in bands]
