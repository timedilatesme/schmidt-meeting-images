"""Draw a lensed-AGN population with LSST, Euclid, Roman and Lazuli photometry.

Adapted from TDLDMC26 ``sims/extra/lensed_agn_montage.ipynb``. The result is
cached to ``cache/population.pkl``; run this once, ``montage.py`` loads it.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import vstack
from astropy.units import Quantity

import slsim.Pipelines as pipelines
from slsim.Deflectors.DeflectorPopulation.galaxy_deflectors import GalaxyDeflectors
from slsim.Lenses.LensPopulation.lens_pop import LensPop
from slsim.Pipelines import roman_speclite
from slsim.Sources.SourceCatalogues.QuasarCatalog.quasar_pop import QuasarRate
from slsim.Sources.SourcePopulation.point_plus_extended_sources import (
    PointPlusExtendedSources,
)

import observatories
import skypy_config

CACHE = Path(__file__).parent / "cache" / "population.pkl"
# extra draw for the compact middle-row quad: smaller separations, 4x the lens area
COMPACT = dict(path=CACHE.with_name("population_compact.pkl"), seed=20261031,
               separation=(0.3, 1.2), lens_area_factor=4)
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
SKY_AREA = Quantity(0.1, unit="deg2")
LENS_AREA = SKY_AREA * 5000
SPEED_FACTOR = 1000
Z_MAX = 3.0


def setup():
    observatories.register_all()
    roman_speclite.configure_roman_filters()
    observatories.configure_all_filters()


def draw(path=CACHE, seed=20261030, separation=(0.8, 4), lens_area_factor=1):
    setup()
    np.random.seed(seed)
    path.parent.mkdir(exist_ok=True)
    extra = [(fmt(band), band) for _, _, bands, fmt in observatories.OBSERVATORIES
             for band in bands]
    skypy_yml = str(path.with_suffix(".yml"))
    bands = skypy_config.generate(skypy_yml, extra)

    galaxies = pipelines.SkyPyPipeline(skypy_config=skypy_yml, sky_area=SKY_AREA,
                                       filters=None, cosmo=COSMO, z_min=0.0, z_max=Z_MAX)
    galaxies.red_galaxies["galaxy_type"] = "red"
    galaxies.blue_galaxies["galaxy_type"] = "blue"
    hosts = vstack([galaxies.red_galaxies, galaxies.blue_galaxies], join_type="exact")

    deflectors = GalaxyDeflectors(
        red_galaxy_list=galaxies.red_galaxies, blue_galaxy_list=galaxies.blue_galaxies,
        kwargs_cut={"band": "i", "band_max": 28, "z_min": 0.01, "z_max": 2.5},
        kwargs_mass2light={}, cosmo=COSMO, sky_area=SKY_AREA,
    )
    quasars = QuasarRate(
        skypy_config=skypy_yml, cosmo=COSMO, sky_area=SKY_AREA, noise=True,
        redshifts=np.linspace(0.001, Z_MAX + 0.01, 100), host_galaxy_candidate=hosts,
        use_qsogen_sed=True, use_sed_interpolator=True, qsogen_bands=bands,
    )
    sources = PointPlusExtendedSources(
        point_plus_extended_sources_list=quasars.quasar_sample(m_min=15, m_max=30, host_galaxy=True),
        cosmo=COSMO, sky_area=SKY_AREA,
        kwargs_cut={"band": "i", "band_max": 26, "z_min": 0.001, "z_max": 5.0},
        catalog_type="skypy", point_source_type="quasar", extended_source_type="single_sersic",
    )
    lenses = LensPop(deflector_population=deflectors, source_population=sources,
                     cosmo=COSMO, sky_area=LENS_AREA * lens_area_factor).draw_population(
        kwargs_lens_cuts={"min_image_separation": separation[0],
                          "max_image_separation": separation[1]},
        speed_factor=SPEED_FACTOR,
    )
    pickle.dump(lenses, open(path, "wb"))
    return lenses


def load(path=CACHE):
    setup()
    return pickle.load(open(path, "rb"))


if __name__ == "__main__":
    lenses = draw(**COMPACT) if "--compact" in sys.argv else draw()
    print("%d lenses, %d quads" % (len(lenses), sum(l.image_number[0] == 4 for l in lenses)))
