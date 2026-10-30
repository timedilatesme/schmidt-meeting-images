"""Helpers for imaging a single lens across several observatories.

Builds on ``slsim.ImageSimulation.image_simulation.simulate_image``:

* :func:`observatory_table` -- what each registered observatory can do
* :func:`available_bands`   -- which bands the source and deflector dicts
                               actually carry magnitudes for
* :func:`band_kwargs`       -- the instrument settings one image is made with
* :func:`numerics_kwargs`   -- the lenstronomy numerics it is rendered with
* :func:`pixel_grid`        -- the coordinates of a rendered image's pixels
* :func:`render`            -- one image, cutout sized in arcseconds
* :func:`render_all`        -- every available band of several observatories
* :func:`show_images`       -- display grid

``render`` and ``render_all`` return counts per second, untouched. Every
display choice -- stretch, colour map, annotation -- lives in ``show_images``.

Images are oriented like HST and JWST data: North up, East left, so RA
increases towards column 0. slsim renders with East right, and ``render``
mirrors the columns, which leaves every pixel at the same sky position.
"""

import lenstronomy.Util.util as lenstronomy_util
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.visualization import simple_norm
from matplotlib import patheffects
from scipy.ndimage import zoom

from slsim.ImageSimulation.image_quality_lenstronomy import (
    _OBSERVATORY_REGISTRY,
    kwargs_single_band,
)
from slsim.ImageSimulation.image_simulation import simulate_image

__all__ = [
    "observatory_table",
    "available_bands",
    "band_kwargs",
    "numerics_kwargs",
    "pixel_grid",
    "render",
    "render_all",
    "show_images",
]


def observatory_table(observatories=None):
    """Instrument properties of every registered band.

    The ``band`` column is the reference for which ``mag_<band>`` and
    ``ps_mag_<band>`` keys a source or deflector dictionary needs.

    :param observatories: optional list of names to restrict to
    :return: astropy Table
    """
    rows = []
    for name, info in _OBSERVATORY_REGISTRY.items():
        if observatories is not None and name not in observatories:
            continue
        for band in info["bands"]:
            row = {"observatory": name, "band": band, "note": ""}
            try:
                kwargs = kwargs_single_band(band, observatory=name)
            except Exception as exc:
                row["note"] = str(exc).split("\n")[0][:60]
                rows.append(row)
                continue
            row.update(
                pixel_scale=kwargs["pixel_scale"],
                exp_time=kwargs["exposure_time"],
                n_exp=kwargs.get("num_exposures", 1),
                zero_point=kwargs["magnitude_zero_point"],
                sky=kwargs.get("sky_brightness", np.nan),
                fwhm=kwargs.get("seeing") or np.nan,
            )
            rows.append(row)
    table = Table(rows)
    for column, fmt in (("pixel_scale", "%.4f"), ("exp_time", "%.1f"),
                        ("zero_point", "%.3f"), ("sky", "%.2f"), ("fwhm", "%.4f")):
        table[column].info.format = fmt
    return table


def available_bands(source_dict, deflector_dict, observatory, with_source=True,
                    with_deflector=True, with_point_source=True):
    """Bands of *observatory* for which every enabled component has a magnitude.

    A band with no magnitude is skipped rather than guessed. The point source
    needs ``ps_mag_<band>`` in the source dict, the host needs ``mag_<band>``
    there, and the deflector needs ``mag_<band>`` in its ``kwargs_light``.

    :return: list of band names
    """
    light = deflector_dict.get("kwargs_light") or {}
    bands = []
    for band in _OBSERVATORY_REGISTRY[observatory]["bands"]:
        if with_point_source and source_dict.get("point_source_type"):
            if "ps_mag_%s" % band not in source_dict:
                continue
        if with_source and source_dict.get("extended_source_type"):
            if "mag_%s" % band not in source_dict:
                continue
        if with_deflector and "mag_%s" % band not in light:
            continue
        bands.append(band)
    return bands


def band_kwargs(observatory, band, override_pixel_scale=None, psf_kernel=None,
                psf_supersampling=1, config_kwargs=None, exposure_time=None,
                num_exposures=None, kwargs_psf=None):
    """The ``kwargs_single_band`` that :func:`render` images a band with.

    :param override_pixel_scale: resample to this scale instead of the
        detector's, for combining bands from different cameras
    :param psf_kernel: a measured kernel to use instead of the config's PSF,
        2d with an odd number of pixels on both axes
    :param psf_supersampling: supersampling of ``psf_kernel`` relative to the
        detector pixel scale
    :param config_kwargs: extra kwargs for the observatory config, e.g.
        ``{"psf_type": "PIXEL"}`` for a real PSF, or
        ``{"survey_mode": "time_domain_deep"}`` for Roman
    :param exposure_time, num_exposures, kwargs_psf: overrides, as
        ``simulate_image`` applies them
    :return: dict
    """
    kwargs_band = kwargs_single_band(
        band, observatory=observatory, **(config_kwargs or {})
    )
    if override_pixel_scale is not None:
        # A pixel kernel is sampled at pixel_scale / supersampling, so changing
        # the scale without rescaling the kernel would resize the PSF.
        if kwargs_band.get("psf_type") == "PIXEL":
            kernel = kwargs_band["kernel_point_source"]
            n = len(kernel)
            m = int(round(n * kwargs_band["pixel_scale"] / override_pixel_scale)) | 1
            kernel = zoom(kernel, m / n)
            kwargs_band["kernel_point_source"] = kernel / kernel.sum()
        kwargs_band["pixel_scale"] = override_pixel_scale

    if psf_kernel is not None:
        kernel = np.asarray(psf_kernel, dtype=float)
        if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError(
                "PSF kernel must be 2d with an odd number of pixels on both "
                "axes so that it is centred on a pixel; got %r" % (kernel.shape,)
            )
        kwargs_band.pop("seeing", None)
        kwargs_band.update(
            psf_type="PIXEL",
            kernel_point_source=kernel / kernel.sum(),
            point_source_supersampling_factor=int(psf_supersampling),
        )
    if exposure_time is not None:
        kwargs_band["exposure_time"] = exposure_time
    if num_exposures is not None:
        kwargs_band["num_exposures"] = num_exposures
    kwargs_band.update(kwargs_psf or {})
    return kwargs_band


def numerics_kwargs(kwargs_band):
    """lenstronomy numerics for a band: a supersampled PSF kernel is convolved
    at its own resolution, with point sources placed on the same subgrid, so
    the kernel is not first binned to detector pixels. Other bands keep
    slsim's default."""
    factor = int(kwargs_band.get("point_source_supersampling_factor", 1))
    if kwargs_band.get("psf_type") == "PIXEL" and factor > 1:
        return {"supersampling_factor": factor, "supersampling_convolution": True,
                "point_source_supersampling_factor": factor}
    return {"supersampling_factor": 3, "point_source_supersampling_factor": 1}


def pixel_grid(num_pix, pixel_scale):
    """``(ra_at_xy_0, dec_at_xy_0, transform_pix2angle)`` of a rendered image,
    in arcsec from the coordinate origin at its central pixel."""
    grid = lenstronomy_util.make_grid_with_coordtransform(num_pix, pixel_scale,
                                                         inverse=True)
    return grid[2], grid[3], grid[6]


def render(lens_class, observatory, band, fov_arcsec=3.0, override_pixel_scale=None,
           psf_kernel=None, psf_supersampling=1, config_kwargs=None,
           **simulate_kwargs):
    """Render one lens image, sizing the cutout from the instrument's own scale.

    :param lens_class: an ``slsim.Lenses.lens.Lens`` instance
    :param observatory: registered observatory name, e.g. ``"HST"``
    :param band: band name owned by that observatory
    :param fov_arcsec: field of view per axis [arcsec]
    :param override_pixel_scale, psf_kernel, psf_supersampling, config_kwargs:
        see :func:`band_kwargs`
    :param simulate_kwargs: passed to ``simulate_image``, e.g. ``with_source``,
        ``with_deflector``, ``with_point_source``, ``add_noise``, ``t_obs``,
        ``kwargs_numerics`` (default :func:`numerics_kwargs`)
    :return: 2d image [counts/second], East left, on :func:`pixel_grid`
    """
    kwargs_band = band_kwargs(
        observatory, band, override_pixel_scale, psf_kernel, psf_supersampling,
        config_kwargs, **{key: simulate_kwargs.pop(key) for key in
                          ("exposure_time", "num_exposures", "kwargs_psf")
                          if key in simulate_kwargs})
    simulate_kwargs.setdefault("kwargs_numerics", numerics_kwargs(kwargs_band))
    num_pix = int(round(fov_arcsec / kwargs_band["pixel_scale"]))
    num_pix = max(num_pix + 1 - num_pix % 2, 3)  # odd, so the lens sits on a pixel
    image = simulate_image(
        lens_class,
        band=band,
        num_pix=num_pix,
        observatory=observatory,
        kwargs_single_band=kwargs_band,
        **simulate_kwargs
    )
    return image[:, ::-1]


def render_all(lens_class, source_dict, deflector_dict, observatories,
               fov_arcsec=3.0, config_kwargs=None, verbose=True, **render_kwargs):
    """Render every band of every observatory that has the magnitudes it needs.

    :param observatories: list of registered observatory names
    :param fov_arcsec: scalar, or a dict keyed by observatory name with an
        optional ``"default"`` entry
    :param config_kwargs: ``{observatory: {...}}`` of per-observatory config
        options, e.g. ``{"HST": {"psf_type": "PIXEL"}}``
    :param render_kwargs: passed to :func:`render`, including the
        ``with_source`` / ``with_deflector`` / ``with_point_source`` flags
    :return: ``{(observatory, band): image}``
    """
    components = {k: render_kwargs[k] for k in
                  ("with_source", "with_deflector", "with_point_source")
                  if k in render_kwargs}
    images = {}
    for observatory in observatories:
        fov = (fov_arcsec.get(observatory, fov_arcsec.get("default"))
               if isinstance(fov_arcsec, dict) else fov_arcsec)
        wanted = available_bands(source_dict, deflector_dict, observatory,
                                 **components)
        skipped = [b for b in _OBSERVATORY_REGISTRY[observatory]["bands"]
                   if b not in wanted]
        done, failed = [], []
        for band in wanted:
            try:
                images[(observatory, band)] = render(
                    lens_class, observatory, band, fov_arcsec=fov,
                    config_kwargs=(config_kwargs or {}).get(observatory),
                    **render_kwargs
                )
                done.append(band)
            except Exception as exc:
                failed.append("%s: %s" % (band, str(exc).split("\n")[0][:70]))
        if verbose:
            print("%-8s %s" % (observatory, ", ".join(done) or "nothing"))
            if skipped:
                print("%-8s   no magnitudes: %s" % ("", ", ".join(skipped)))
            for message in failed:
                print("%-8s   not available: %s" % ("", message))
    return images


_BAR_STEPS = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]


def _scale_bar(fov_arcsec):
    """Largest round bar that fits comfortably in a panel, and its label."""
    length = next((s for s in _BAR_STEPS if s <= 0.6 * fov_arcsec), _BAR_STEPS[-1])
    label = '%g"' % length if length >= 1 else "%g mas" % round(length * 1000)
    return length, label


def _annotation(kwargs, n_pix):
    """The three lines of instrument text drawn over a panel."""
    scale = kwargs["pixel_scale"]
    seeing, psf_type = kwargs.get("seeing"), kwargs.get("psf_type")
    lines = [
        '%.4f"/px  %d px  %.2f" FoV' % (scale, n_pix, n_pix * scale),
        "%g s total (%d x %g s)" % (
            kwargs["exposure_time"] * kwargs.get("num_exposures", 1),
            kwargs.get("num_exposures", 1), kwargs["exposure_time"]),
    ]
    if "sky_brightness" in kwargs:
        lines.append("sky %.2f mag/arcsec²" % kwargs["sky_brightness"])
    lines.append('FWHM %.3f"' % seeing if psf_type == "GAUSSIAN" and seeing
                 else "pixel PSF")
    return "\n".join(lines)


def show_images(images, title=None, max_percent=99.5, cmap="magma",
                config_kwargs=None):
    """Grid of rendered images, one row per observatory.

    An asinh stretch is used throughout so the host galaxy stays visible
    against the point sources. Each panel is annotated with its pixel scale,
    cutout size, total exposure time, sky brightness and PSF, and carries a
    scale bar whose length is chosen to fit -- so check the label when
    comparing across rows.

    :param images: ``{(observatory, band): image}`` as returned by
        :func:`render_all`
    :param config_kwargs: ``{observatory: {...}}``, matching what was rendered
    """
    rows = {}
    for (observatory, band), image in images.items():
        rows.setdefault(observatory, []).append((band, image))
    if not rows:
        print("nothing to show")
        return None

    outline = [patheffects.withStroke(linewidth=1.8, foreground="black")]
    n_cols = max(len(entries) for entries in rows.values())
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(2.9 * n_cols, 3.2 * len(rows)), squeeze=False)
    for row, (observatory, entries) in enumerate(rows.items()):
        for col in range(n_cols):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col >= len(entries):
                ax.axis("off")
                continue
            band, image = entries[col]
            ax.imshow(image, origin="lower", cmap=cmap,
                      norm=simple_norm(image, stretch="asinh", min_percent=1,
                                       max_percent=max_percent))
            kwargs = kwargs_single_band(
                band, observatory=observatory,
                **((config_kwargs or {}).get(observatory) or {}))
            n_pix = image.shape[0]
            ax.text(0.04, 0.96, _annotation(kwargs, n_pix), color="w",
                    fontsize=8.5, va="top", ha="left", linespacing=1.35,
                    transform=ax.transAxes, path_effects=outline)
            bar, label = _scale_bar(n_pix * kwargs["pixel_scale"])
            x0, y0 = 0.06 * n_pix, 0.08 * n_pix
            ax.plot([x0, x0 + bar / kwargs["pixel_scale"]], [y0, y0],
                    color="w", lw=2.2, path_effects=outline)
            ax.text(x0, y0 + 0.035 * n_pix, label, color="w", fontsize=8,
                    path_effects=outline)
            ax.set_title("%s  %s" % (observatory, band), fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13)
        # tight_layout ignores the suptitle, so reserve the space explicitly
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.2 * len(rows))))
    else:
        fig.tight_layout()
    return fig
