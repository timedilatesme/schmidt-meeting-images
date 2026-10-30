"""Export a simulated lens in dolphin's input layout, with a separate truth file.

* :func:`export`         -- write the data files and the truth file for one lens
* :func:`read_band`      -- one band's ``kwargs_data``, ``kwargs_psf`` and metadata
* :func:`read_truth`     -- the truth file, as a nested dict
* :func:`to_lenstronomy` -- ``kwargs_data``, ``kwargs_psf`` and ``kwargs_numerics``
                            for one band, as a modeller has them or exactly as
                            the image was simulated

Layout under ``out_dir``, following https://github.com/ajshajib/dolphin::

    lens_list.txt                       lens names, one per line
    data/<lens>/image_<lens>_<band>.h5
    data/<lens>/psf_<lens>_<band>.h5
    truth/<lens>_truth.h5               not part of a blind release

Every dataset in the two data files is a keyword of lenstronomy's ``ImageData``
or ``PSF``, which is how dolphin reads them, so they must hold nothing else.

``image_<lens>_<band>.h5``
    datasets   image_data [counts/s], background_rms [counts/s],
               exposure_time [s, effective: all exposures, times the gain if
               the camera counts ADU], ra_at_xy_0, dec_at_xy_0 [arcsec],
               transform_pix2angle [arcsec/pixel]
    attributes format_version, observatory, band, pixel_scale [arcsec],
               total_exposure_time [s], num_exposures, magnitude_zero_point
               [AB mag for 1 count/s], sky_brightness [AB mag/arcsec2],
               read_noise [e-], ccd_gain [e-/ADU], psf_type, fwhm [arcsec],
               psf_supersampling

``psf_<lens>_<band>.h5``
    datasets   kernel_point_source: the instrument PSF binned to detector
               pixels and normalised. It is not the kernel the image was made
               with when the instrument supplies a finer one; that is in the
               truth.

``truth/<lens>_truth.h5``
    system         z_lens, z_source, theta_E [arcsec], lens_center_x,
                   lens_center_y [arcsec], kappa_ext, gamma1_ext, gamma2_ext,
                   t_obs [days], H0 [km/s/Mpc], Om0
    point_source   image_x, image_y [arcsec], magnification, arrival_time [days]
    kwargs_model   lenstronomy's, shared by every band
    bands/<band>   observatory, kwargs_params (magnitudes, not amplitudes),
                   kwargs_psf and kwargs_numerics the image was simulated with,
                   model_image and model_lens_light, model_source,
                   model_point_source: noiseless, on the data's pixel grid
    inputs         source_dict, deflector_dict, config_kwargs, render_kwargs

Images are North up, East left, on ``util.pixel_grid``: coordinates are arcsec
from the origin at the central pixel, the frame of every truth position. The
deflector is at the origin only if the deflector dict puts it there.
"""

from pathlib import Path

import h5py
import numpy as np
from lenstronomy.Data.psf import PSF
from lenstronomy.SimulationAPI.data_api import DataAPI

import util

__all__ = ["export", "read_band", "read_truth", "to_lenstronomy"]

FORMAT_VERSION = 1

_BAND_KWARGS = ("override_pixel_scale", "psf_kernel", "psf_supersampling",
                "exposure_time", "num_exposures", "kwargs_psf")
_COMPONENTS = {"model_lens_light": "with_deflector", "model_source": "with_source",
               "model_point_source": "with_point_source"}


def export(lens_class, images, out_dir, lens_name, config_kwargs=None,
           source_dict=None, deflector_dict=None, include_truth=True,
           **render_kwargs):
    """Write the data files for every band of *images*, and the truth file.

    :param images: ``{(observatory, band): image}`` from ``util.render_all``
    :param config_kwargs: ``{observatory: {...}}``, as passed to ``render_all``
    :param source_dict, deflector_dict: the inputs, kept in the truth
    :param include_truth: False for a blind release
    :param render_kwargs: the other keyword arguments *images* were rendered
        with, so the files describe the images as they were made
    """
    for key in ("image_units_counts", "add_background_counts"):
        if render_kwargs.get(key):
            raise ValueError("%s=True is not supported: the data files assume "
                             "background-subtracted counts/s" % key)
    bands = [band for _, band in images]
    if len(set(bands)) < len(bands):
        raise ValueError("dolphin names files by band alone, so band names "
                         "must be unique across observatories")

    out_dir = Path(out_dir)
    data_dir = out_dir / "data" / lens_name
    data_dir.mkdir(parents=True, exist_ok=True)
    lens_list = out_dir / "lens_list.txt"
    names = lens_list.read_text().split() if lens_list.exists() else []
    if lens_name not in names:
        lens_list.write_text("".join(name + "\n" for name in names + [lens_name]))

    truth_bands = {}
    for (observatory, band), image in images.items():
        kwargs_band = util.band_kwargs(
            observatory, band, config_kwargs=(config_kwargs or {}).get(observatory),
            **{key: render_kwargs[key] for key in _BAND_KWARGS if key in render_kwargs})
        kwargs_psf = _write_band(data_dir, lens_name, observatory, band, image,
                                 kwargs_band)
        if include_truth:
            truth_bands[band] = _truth_band(
                lens_class, observatory, band, image, kwargs_band, kwargs_psf,
                config_kwargs, render_kwargs)

    if include_truth:
        truth_dir = out_dir / "truth"
        truth_dir.mkdir(exist_ok=True)
        truth = _truth(lens_class, bands, render_kwargs)
        truth["bands"] = truth_bands
        truth["inputs"] = {"source_dict": source_dict, "deflector_dict": deflector_dict,
                           "config_kwargs": config_kwargs, "render_kwargs": render_kwargs}
        with h5py.File(truth_dir / ("%s_truth.h5" % lens_name), "w") as file:
            file.attrs["format_version"] = FORMAT_VERSION
            _write_h5(file, truth)


def _write_band(data_dir, lens_name, observatory, band, image, kwargs_band):
    num_pix = image.shape[0]
    data = DataAPI(num_pix, **kwargs_band)
    kwargs_psf = data.kwargs_psf
    ra_at_xy_0, dec_at_xy_0, transform = util.pixel_grid(num_pix, data.pixel_scale)
    detector_kernel = PSF(pixel_size=data.pixel_scale, **kwargs_psf).kernel_point_source
    kwargs_data = {
        "image_data": image,
        # floats: ImageData treats a numpy integer exposure time as an array
        "background_rms": float(data.kwargs_data["background_rms"]),
        "exposure_time": float(data.kwargs_data["exposure_time"]),
        "ra_at_xy_0": ra_at_xy_0,
        "dec_at_xy_0": dec_at_xy_0,
        "transform_pix2angle": transform,
    }
    num_exposures = kwargs_band.get("num_exposures", 1)
    meta = {
        "format_version": FORMAT_VERSION,
        "observatory": observatory,
        "band": band,
        "pixel_scale": data.pixel_scale,
        "total_exposure_time": kwargs_band["exposure_time"] * num_exposures,
        "num_exposures": num_exposures,
        "magnitude_zero_point": kwargs_band["magnitude_zero_point"],
        "sky_brightness": kwargs_band.get("sky_brightness"),
        "read_noise": kwargs_band.get("read_noise"),
        "ccd_gain": kwargs_band.get("ccd_gain"),
        "psf_type": kwargs_psf["psf_type"],
        "fwhm": kwargs_psf.get("fwhm"),
        "psf_supersampling": kwargs_psf.get("point_source_supersampling_factor", 1),
    }
    with h5py.File(data_dir / ("image_%s_%s.h5" % (lens_name, band)), "w") as file:
        for key, value in kwargs_data.items():
            file.create_dataset(key, data=value)
        file.attrs.update({key: value for key, value in meta.items() if value is not None})
    with h5py.File(data_dir / ("psf_%s_%s.h5" % (lens_name, band)), "w") as file:
        file.create_dataset("kernel_point_source", data=_east_left(detector_kernel))
    return kwargs_psf


def _east_left(kernel):
    kernel = np.asarray(kernel, dtype=float)[:, ::-1]
    return kernel / kernel.sum()


def _truth_band(lens_class, observatory, band, image, kwargs_band, kwargs_psf,
                config_kwargs, render_kwargs):
    kwargs_psf = {key: value for key, value in kwargs_psf.items()
                  if key != "kernel_point_source_normalisation"}
    if "kernel_point_source" in kwargs_psf:
        kwargs_psf["kernel_point_source"] = _east_left(kwargs_psf["kernel_point_source"])
    else:
        kwargs_psf["pixel_size"] = kwargs_band["pixel_scale"]
    entry = {
        "observatory": observatory,
        "kwargs_params": lens_class.lenstronomy_kwargs(
            band, time=render_kwargs.get("t_obs"),
            microlensing=render_kwargs.get("microlensing", False),
            kwargs_microlensing=render_kwargs.get("kwargs_microlensing"))[1],
        "kwargs_psf": kwargs_psf,
        "kwargs_numerics": render_kwargs.get("kwargs_numerics")
                           or util.numerics_kwargs(kwargs_band),
    }

    flags = {flag: render_kwargs.get(flag, True) for flag in _COMPONENTS.values()}
    noiseless = {key: value for key, value in render_kwargs.items()
                 if key not in flags and key != "add_noise"}

    def model(**components):
        return util.render(lens_class, observatory, band,
                           fov_arcsec=image.shape[0] * kwargs_band["pixel_scale"],
                           config_kwargs=(config_kwargs or {}).get(observatory),
                           add_noise=False, **components, **noiseless)

    entry["model_image"] = model(**flags)
    for name, flag in _COMPONENTS.items():
        if flags[flag]:
            entry[name] = model(**{other: other == flag for other in flags})
    return entry


def _truth(lens_class, bands, render_kwargs):
    ra, dec = lens_class.point_source_image_positions()[0]
    kappa, gamma1, gamma2 = lens_class.los_linear_distortions
    lens_x, lens_y = lens_class.deflector.deflector_center
    kwargs_model = lens_class.lenstronomy_kwargs(bands[0])[0]
    return {
        "system": {
            "z_lens": lens_class.deflector_redshift,
            "z_source": lens_class.source_redshift_list[0],
            "theta_E": lens_class.einstein_radius[0],
            "lens_center_x": lens_x,
            "lens_center_y": lens_y,
            "kappa_ext": kappa,
            "gamma1_ext": gamma1,
            "gamma2_ext": gamma2,
            "t_obs": render_kwargs.get("t_obs"),
            "H0": lens_class.cosmo.H0.value,
            "Om0": lens_class.cosmo.Om0,
        },
        "point_source": {
            "image_x": ra,
            "image_y": dec,
            "magnification": lens_class.point_source_magnification()[0],
            "arrival_time": lens_class.point_source_arrival_times()[0],
        },
        "kwargs_model": {key: [str(v) for v in value] if isinstance(value, list)
                         and not all(isinstance(v, (int, float)) for v in value)
                         else value for key, value in kwargs_model.items()},
    }


def read_band(out_dir, lens_name, band):
    """``(kwargs_data, kwargs_psf, meta)`` from one band's data files."""
    data_dir = Path(out_dir) / "data" / lens_name
    with h5py.File(data_dir / ("image_%s_%s.h5" % (lens_name, band)), "r") as file:
        kwargs_data = {key: file[key][()] for key in file}
        meta = dict(file.attrs)
    with h5py.File(data_dir / ("psf_%s_%s.h5" % (lens_name, band)), "r") as file:
        kwargs_psf = {"psf_type": "PIXEL", **{key: file[key][()] for key in file}}
    return kwargs_data, kwargs_psf, meta


def read_truth(out_dir, lens_name):
    with h5py.File(Path(out_dir) / "truth" / ("%s_truth.h5" % lens_name), "r") as file:
        return {"format_version": file.attrs["format_version"], **_read_h5(file)}


def to_lenstronomy(out_dir, lens_name, band, exact=False):
    """``(kwargs_data, kwargs_psf, kwargs_numerics)`` for one band.

    By default the PSF is the data file's detector-sampled kernel, which is all
    a modeller is given. ``exact=True`` takes the kernel and numerics the image
    was simulated with from the truth, which reproduce ``model_image`` exactly.
    """
    kwargs_data, kwargs_psf, _ = read_band(out_dir, lens_name, band)
    if not exact:
        return kwargs_data, kwargs_psf, util.numerics_kwargs({})
    truth = read_truth(out_dir, lens_name)["bands"][band]
    return kwargs_data, truth["kwargs_psf"], truth["kwargs_numerics"]


def _write_h5(group, tree):
    for key, value in tree.items():
        if value is None:
            continue
        if isinstance(value, dict):
            _write_h5(group.create_group(key, track_order=True), value)
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            # lenstronomy's per-profile kwargs lists
            subgroup = group.create_group(key)
            subgroup.attrs["is_list"] = True
            _write_h5(subgroup, {str(i): v for i, v in enumerate(value)})
        elif isinstance(value, (np.ndarray, list, tuple)) and np.asarray(value).dtype.kind in "biuf":
            value = np.asarray(value)
            group.create_dataset(key, data=value,
                                 compression="gzip" if value.ndim > 1 else None)
        else:
            try:
                group.attrs[key] = value
            except TypeError:  # objects with no HDF5 type, e.g. a microlensing map
                group.attrs[key] = str(value)


def _read_h5(group):
    # plain Python scalars: lenstronomy rejects numpy integers in places
    tree = {key: value.tolist() if isinstance(value, (np.ndarray, np.generic)) else value
            for key, value in group.attrs.items() if key != "is_list"}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            tree[key] = _read_h5(item)
        else:
            tree[key] = item[()].item() if item.ndim == 0 else item[()]
    if group.attrs.get("is_list"):
        return [tree[str(i)] for i in range(len(tree))]
    return tree
