"""Pick the montage systems, image them with every observatory, and simulate
the Lazuli IFS cube of the hero system. Results go to ``cache/render.pkl`` for
``figure.py``.
"""

import pickle

import numpy as np
from astropy.table import Table

from slsim.Lenses.lens import Lens
from slsim.Sources.source import Source

import ifs
import population
import util

CACHE = population.CACHE.parent / "render.pkl"
COSMOS_PATH = "/pscratch/sd/s/sharma/data/catalogs/COSMOS_23.5_training_sample"
FOV = 5.0
SEED = 7

RGB = {
    "LSST": ["i", "r", "g"],
    "Euclid": ["H", "Y", "VIS"],
    "Roman": ["F158", "F129", "F106"],
    "Lazuli": ["LZz", "LZr", "LZg"],
}
CONFIG_KWARGS = {"Roman": {"psf_type": "PIXEL"}, "Lazuli": {"psf_type": "PIXEL"}}
# NISP is resampled onto the VIS grid so the three planes stack
PIXEL_SCALE = {"Euclid": 0.101}
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


def select(table):
    """Hero quad that fits the IFS narrow field, a second quad, and a double."""
    bright = table[table["i_qso_2nd"] < 22.5]
    quads = bright[(bright["n_images"] == 4) & (bright["separation"] >= 1.2)]
    hero = quads[quads["separation"] <= 2.1]
    hero.sort("i_qso_2nd")
    others = quads[quads["index"] != hero["index"][0]]
    others.sort("separation", reverse=True)
    doubles = bright[(bright["n_images"] == 2) & (bright["separation"] >= 1.2)
                     & (bright["separation"] <= 3.0)]
    doubles.sort("i_qso_2nd")
    return [int(hero["index"][0]), int(others["index"][0]), int(doubles["index"][0])]


def image(lens, observatory):
    planes = []
    for band in RGB[observatory]:
        kwargs = util.band_kwargs(observatory, band, override_pixel_scale=PIXEL_SCALE.get(observatory),
                                  config_kwargs=CONFIG_KWARGS.get(observatory))
        counts = util.render(lens, observatory, band, fov_arcsec=FOV,
                             override_pixel_scale=PIXEL_SCALE.get(observatory),
                             config_kwargs=CONFIG_KWARGS.get(observatory))
        planes.append((counts, kwargs["pixel_scale"], kwargs["magnitude_zero_point"],
                       kwargs.get("seeing")))
    return planes


def main():
    lenses = population.load()
    table = catalogue(lenses)
    chosen = select(table)
    np.random.seed(SEED)
    systems = []
    for index in chosen:
        lens = with_cosmos_host(lenses[index])
        row = table[table["index"] == index][0]
        systems.append(dict(
            index=index, lens=lens, info={k: row[k] for k in table.colnames},
            images={obs: image(lens, obs) for obs in RGB},
        ))
        print("system %d: %s" % (index, dict(systems[-1]["info"])))
    hero = systems[0]["lens"]
    cube = ifs.simulate(hero, ifs.LazuliIFS(mode="NF"), microlensing=True, seed=SEED)
    for system in systems:
        system.pop("lens")
    pickle.dump(dict(systems=systems, cube=cube, rgb=RGB, fov=FOV,
                     positions=hero.point_source_image_positions()[0]),
                open(CACHE, "wb"))


if __name__ == "__main__":
    main()
