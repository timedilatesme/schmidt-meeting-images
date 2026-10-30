# Adding an observatory

An observatory is one module file in this folder plus one line in `__init__.py`.
There is no base class and no registry beyond that list — copy the closest
existing module and edit it.

## What the module must expose

```python
BANDS = ["MYBAND"]                     # every band this config can build

class MyInstrument:
    def __init__(self, band="MYBAND", psf_type="GAUSSIAN"):
        self.obs = copy.deepcopy(_BANDS[band])      # deep-copy, see below
        self.camera = copy.deepcopy(MY_camera)

    def kwargs_single_band(self):
        return lenstronomy.Util.util.merge_dicts(self.camera, self.obs)
```

That is the whole interface slsim needs. Then add it to `OBSERVATORIES` in
`__init__.py`:

```python
("MyInstrument", my_module.MyInstrument, my_module.BANDS, None),
```

The fourth entry is a `speclite_fmt` callable, only consulted for AGN
variability. Leave it `None` unless you have registered filter curves.

## The band dictionary

| key | units | notes |
|---|---|---|
| `exposure_time` | seconds | **per exposure**, not the total |
| `num_exposures` | count | coadds; total time is the product of the two |
| `sky_brightness` | AB mag / arcsec² | convert if your source quotes Vega |
| `magnitude_zero_point` | AB mag | magnitude giving 1 e⁻/s |
| `seeing` | arcsec | PSF FWHM. Required for `psf_type="GAUSSIAN"` |
| `psf_type` | `"GAUSSIAN"` or `"PIXEL"` | |

## The camera dictionary

| key | units |
|---|---|
| `pixel_scale` | arcsec / pixel — **native detector scale**, see below |
| `read_noise` | electrons |
| `ccd_gain` | electrons / ADU |

## Non-optional Rules

**Band names must be globally unique, and must not contain a hyphen.** slsim
maps a band name to exactly one observatory and takes the first match, so a
duplicate silently resolves to the wrong instrument. Bare `J`, `H`, `Y`, `VIS`
and the LSST `ugrizy` are taken. Prefix if you must — ELT uses `MIC`, with no
hyphen, because slsim inverts a speclite filter name with `name.split("-")[1]`:
a band called `MIC-J` comes back as `J` and collides with Euclid.

**Deep-copy the band and camera dicts in `__init__`.** lenstronomy's own `HST`,
`JWST` and `Roman` assign the module-level dict to `self.obs` and then mutate
it, so one request for a different `psf_type` permanently changes the default
for every later instance in the process. Do not repeat that.

**`pixel_scale` must be the native detector scale**, not a drizzled or resampled
product scale. A `PIXEL` kernel is interpreted as being sampled at
`pixel_scale / point_source_supersampling_factor`, so declaring anything else
silently renders the PSF at the wrong angular size.

**Magnitudes.** A band is only imaged if the source and
deflector dicts carry `ps_mag_<band>` and `mag_<band>` for it. Adding a band
here does not create photometry; missing bands are reported and skipped.

## Real PSFs

Gaussian is the default. To offer a real PSF, take `psf_type` in `__init__` and,
for `"PIXEL"`, write lenstronomy's three keys into `self.obs`:

```python
del self.obs["seeing"]
self.obs.update(
    psf_type="PIXEL",
    kernel_point_source=kernel,                 # 2d, odd axes, peak centred
    point_source_supersampling_factor=OVERSAMPLE,
)
```

The kernel must have an **odd** number of pixels on both axes so its peak sits
on a pixel, and must be sampled at exactly `pixel_scale / OVERSAMPLE`. Fetch or
build it inside this module — `hst.py` downloads an empirical library, `jwst.py`
computes one with `stpsf`, `keck_ao.py` and `elt_micado.py` build a Strehl-weighted double Gaussian analytically.

## Filter curves

Anything that synthesises magnitudes — the population pipeline, AGN variability
— works through speclite, not through the band dictionaries. Expose:

```python
def configure_speclite_filters():
    _filters.svo("HST", {"F606W": "HST/ACS_WFC.F606W"})     # measured curve
    _filters.tophat("MidEx", {"F625": (5000.0, 7500.0)})    # square passband/tophat
```

`_filters.svo` downloads from the SVO Filter Profile Service; `_filters.tophat`
builds a square passband from edges in Angstrom, for instruments SVO does not
carry. Both are idempotent. Add the module to `_MODULES` in `__init__.py` and
give it a `speclite_fmt` in `OBSERVATORIES`; registering a curve without the
format entry leaves slsim unable to find it.

The filter name is `<group>-<band>`, and `<band>` must equal the band name.

A band can only join the galaxy population if its filter stays above the 600 A
rest-frame floor of SkyPy's kcorrect templates over the whole redshift range,
`blue_edge / 600 - 1 >= z_max`. MidEx dropped F150 and F275 for this reason:
below that limit the pipeline raises rather than degrading.

## Testing

```bash
python -W ignore -c "import observatories, util; observatories.register_all(); util.observatory_table().pprint_all()"
```

A band that raises shows up as a `note` in that table rather than crashing.
