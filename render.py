"""Pick the montage systems, image them with every observatory, and simulate
the Lazuli IFS cube of the hero system. Results go to ``cache/render.pkl`` for
``figure.py``.
"""

import os
import pickle
import sys

import numpy as np
from astropy.table import Table

from slsim.Lenses.lens import Lens
from slsim.Sources.source import Source

import ifs
import population
import util

CACHE = population.CACHE.parent / "render.pkl"
# HST COSMOS 23.5 training sample (https://zenodo.org/records/3242143): NERSC, then Mac
COSMOS_PATH = next(
    (path for path in ("/pscratch/sd/s/sharma/data/catalogs/COSMOS_23.5_training_sample",
                       "/Users/paras/Projects/datasets/COSMOS/COSMOS_23.5_training_sample")
     if os.path.isdir(path)),
    None,
)
FOV = 5.0
SEED = 7

RGB = {
    "LSST": ["i", "r", "g"],
    "Euclid": ["VIS", "Y", "J", "H"],  # combined Q1-style in figure.py
    "Roman": ["F158", "F129", "F106"],
    "Lazuli": ["LZz", "LZr", "LZg"],
}
CONFIG_KWARGS = {"Roman": {"psf_type": "PIXEL"}, "Lazuli": {"psf_type": "PIXEL"}}
BANDS = [band for bands in RGB.values() for band in bands]


def with_cosmos_host(lens):
    """The same lens with its Sersic host swapped for a real COSMOS galaxy."""
    source = lens.source(0)
    sersic = source._source._point_source.source_dict
    e1, e2 = source.ellipticity
    center_x, center_y = source.extended_source_position
    magnitudes = {}
    for band in util_bands():
        for prefix, magnitude in (("ps_mag_", source.point_source_magnitude),
                                  ("mag_", source.extended_source_magnitude)):
            try:
                magnitudes[prefix + band] = float(magnitude(band))
            except Exception:
                pass
    host = Source(
        cosmo=lens.cosmo, extended_source_type="catalog_source", point_source_type="quasar",
        catalog_type="HST_COSMOS", catalog_path=COSMOS_PATH, sersic_fallback=True,
        z=source.redshift, center_x=center_x, center_y=center_y, ra_off=0.0, dec_off=0.0,
        angular_size=source.angular_size, physical_size=sersic["physical_size"],
        n_sersic=sersic["n_sersic"], e1=e1, e2=e2, stellar_mass=source.stellar_mass,
        black_hole_mass_exponent=None, black_hole_spin=None, inclination_angle=None,
        eddington_ratio=None, accretion_disk=None, r_out=None, r_resolution=500,
        corona_height=10, random_seed=None, **magnitudes,
    )
    return Lens(source_class=host, deflector_class=lens.deflector, cosmo=lens.cosmo,
                los_class=lens.los_class)


def util_bands():
    from slsim.ImageSimulation.image_quality_lenstronomy import _OBSERVATORY_REGISTRY
    return [b for info in _OBSERVATORY_REGISTRY.values() for b in info["bands"]]


def max_separation(lens):
    ra, dec = lens.point_source_image_positions()[0]
    return max(np.hypot(ra[a] - ra[b], dec[a] - dec[b])
               for a in range(len(ra)) for b in range(a + 1, len(ra)))


def catalogue(lenses):
    return Table(rows=[
        (index, lens.image_number[0], max_separation(lens), lens.einstein_radius[0],
         lens.deflector_redshift, lens.source_redshift_list[0],
         np.sort(lens.point_source_magnitude("i", lensed=True)[0])[1],
         lens.deflector.magnitude("i"))
        for index, lens in enumerate(lenses)],
        names=["index", "n_images", "separation", "theta_E", "z_d", "z_s",
               "i_qso_2nd", "i_lens"])


# middle row, picked by eye from population.COMPACT: 0.47" quad (closest pair
# 0.25") that LSST sees as a point, Euclid/Roman blend and Lazuli resolves
COMPACT_QUAD = 81


def select(table):
    """Hero quad (the widest that fits the IFS narrow field, so the lens light
    is well separated from the images) and a double, from the main population."""
    bright = table[table["i_qso_2nd"] < 22.5]
    quads = bright[(bright["n_images"] == 4) & (bright["separation"] >= 1.2)]
    hero = quads[quads["separation"] <= 2.1]
    hero.sort("separation", reverse=True)
    doubles = bright[(bright["n_images"] == 2) & (bright["separation"] >= 1.2)
                     & (bright["separation"] <= 3.0)]
    doubles.sort("i_qso_2nd")
    return int(hero["index"][0]), int(doubles["index"][0])


def chosen():
    """(lens, catalogue row) for the hero, the compact quad and the double."""
    lenses = population.load()
    table = catalogue(lenses)
    hero, double = select(table)
    compact = population.load(population.COMPACT["path"])
    compact_row = catalogue([compact[COMPACT_QUAD]])[0]
    compact_row["index"] = COMPACT_QUAD
    return [(lenses[hero], table[table["index"] == hero][0]),
            (compact[COMPACT_QUAD], compact_row),
            (lenses[double], table[table["index"] == double][0])]


def image(lens, observatory):
    planes = []
    for band in RGB[observatory]:
        kwargs = util.band_kwargs(observatory, band, config_kwargs=CONFIG_KWARGS.get(observatory))
        counts = util.render(lens, observatory, band, fov_arcsec=FOV,
                             config_kwargs=CONFIG_KWARGS.get(observatory))
        planes.append((counts, kwargs["pixel_scale"], kwargs["magnitude_zero_point"],
                       kwargs.get("seeing")))
    return planes


def microlensing(lens, draws=4):
    """A mild static microlensing realization of *lens*: of a few independent
    star fields, the one with the smallest largest |Delta m|, so the figure
    is not dominated by a rare high-magnification event."""
    nm = np.geomspace(400, 1700, 10)
    draws = [ifs.chromatic_microlensing(pickle.loads(pickle.dumps(lens)), nm, seed=draw)
             for draw in range(draws)]
    delta = min(draws, key=lambda d: np.abs(d).max())
    print("microlensing: max |dm| %.2f mag" % np.abs(delta).max())
    return nm, delta


def cube_only():
    """Redo just the hero IFS cube, keeping the cached imaging."""
    lenses = population.load()
    data = pickle.load(open(CACHE, "rb"))
    hero = with_cosmos_host(lenses[select(catalogue(lenses))[0]])
    data["cube"] = ifs.simulate(hero, ifs.LazuliIFS(mode="NF", num_exposures=12),
                                microlensing=microlensing(hero), seed=SEED)
    pickle.dump(data, open(CACHE, "wb"))


def main():
    np.random.seed(SEED)
    systems = []
    for lens, row in chosen():
        lens, index = with_cosmos_host(lens), int(row["index"])
        systems.append(dict(
            index=index, lens=lens, info={k: row[k] for k in row.colnames},
            images={obs: image(lens, obs) for obs in RGB},
        ))
        print("system %d: %s" % (index, dict(systems[-1]["info"])))
    hero = systems[0]["lens"]
    cube = ifs.simulate(hero, ifs.LazuliIFS(mode="NF", num_exposures=12), microlensing=microlensing(hero),
                        seed=SEED)
    for system in systems:
        system.pop("lens")
    pickle.dump(dict(systems=systems, cube=cube, rgb=RGB, fov=FOV,
                     positions=hero.point_source_image_positions()[0]),
                open(CACHE, "wb"))


if __name__ == "__main__":
    cube_only() if "--cube-only" in sys.argv else main()
