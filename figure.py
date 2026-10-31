"""Slide figure: lensed AGN discovered by LSST / Euclid / Roman, characterised
by Lazuli imaging and integral field spectroscopy.

Reads ``cache/render.pkl`` (from ``render.py``) and writes
``figures/lazuli_lensed_agn.{png,pdf}`` at 16:9.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import speclite.filters
from scipy.ndimage import gaussian_filter
from astropy.visualization import make_lupton_rgb
from slsim.Util.euclid_rgb_util import euclid_rgb_from_image_list
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D

import population
from ifs import spectra
from observatories import lazuli
from observatories.lazuli import airy_kernel

FIGURES = Path(__file__).parent / "figures"
WIDTH, HEIGHT = 16.0, 9.0
BACKGROUND, INK, MUTED = "black", "white", "0.62"
ACCENT = "#f28c28"  # Lazuli column and IFS block
CHANNEL = ["#ff5a4e", "#5ad46a", "#5a9bff"]
IMAGE_COLOURS = ["#ffd166", "#ef476f", "#06d6a0", "#4cc9f0"]
LENS_COLOUR = "#c3a6ff"
OUTLINE = [patheffects.withStroke(linewidth=2.2, foreground="black")]

TITLES = {"LSST": "Rubin LSST", "Euclid": "Euclid", "Roman": "Roman", "Lazuli": "Lazuli WCC"}
DEPTH = {"LSST": "10-yr", "Euclid": "wide", "Roman": "HLWAS", "Lazuli": "4×300 s"}

# rest-frame vacuum wavelengths [A]; [O III] 5007 is blended with Hbeta at R ~ 150
QSO_LINES = {"Lyα": 1215.67, "C IV": 1549.06, "C III]": 1908.73, "Mg II": 2798.75,
             "Hγ": 4340.47, "Hβ": 4861.33}

Q, STRETCH, WHITE_PERCENTILE, WHITE_REFERENCE = 10, 0.3, 99.5, "Roman"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": MUTED,
    "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
    "savefig.facecolor": BACKGROUND, "mathtext.default": "regular",
})


#: Keynote build order; each step's elements are exported as their own layer.
STEPS = ["0_base", "1_surveys", "2_lazuli", "3_ifs_field", "4a_slice_blue", "4b_slice_mid",
         "4c_slice_red", "5_quasar_spectra", "6_caption", "7_lens"]


class Steps:
    """Assigns every artist drawn between two :meth:`start` calls to a build step."""

    def __init__(self, fig):
        self.fig, self.current, self.groups = fig, None, {}
        self.seen = self._everything()

    def _everything(self):
        found = set(self.fig.axes) | set(self.fig.texts) | set(self.fig.patches)
        for ax in self.fig.axes:
            found |= set(ax.get_children())
        return found

    def start(self, name):
        now = self._everything()
        new = now - self.seen
        new_axes = {a for a in new if isinstance(a, plt.Axes)}
        new = {a for a in new if a in new_axes or getattr(a, "axes", None) not in new_axes}
        if self.current is not None:
            self.groups.setdefault(self.current, set()).update(new)
        self.seen, self.current = now, name

    def show_until(self, last):
        self.start(self.current)
        for name, artists in self.groups.items():
            for artist in artists:
                artist.set_visible(STEPS.index(name) <= STEPS.index(last))


STEP = None


def step(name):
    if STEP is not None:
        STEP.start(name)


def axes_at(fig, x, y, w, h, **kwargs):
    return fig.add_axes([x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT], **kwargs)


def angle(*values):
    values = sorted(set(values))
    if max(values) >= 0.1:
        return "–".join("%.2g" % v for v in values) + "″"
    return "–".join("%d" % round(1e3 * v) for v in values) + " mas"


# ----------------------------------------------------------------- montage


def surface_brightness(counts, pixel_scale, zero_point, _seeing):
    return counts * 10 ** (-0.4 * (zero_point - 31.4)) / pixel_scale ** 2


EUCLID_COLOURS = ["#5a9bff", "#5ad46a", "#ffb04e", "#ff5a4e"]


def colour_composite(images, observatory):
    if observatory == "Euclid":
        # Euclid Q1 display: VIS luminance, NISP Y/J/H colour (arXiv:2503.15324)
        return euclid_rgb_from_image_list([p[0] for p in images["Euclid"]],
                                          colour="VIS_WEIGHTED_Y_J_H", use_luminance=True,
                                          stretch="arcsinh", arcsinh_scale=200.0,
                                          white_percentile=99.995)
    white = np.percentile(np.mean([surface_brightness(*p) for p in images[WHITE_REFERENCE]],
                                  axis=0), WHITE_PERCENTILE)
    planes = [surface_brightness(*p) / white for p in images[observatory]]
    return make_lupton_rgb(*planes, minimum=0, stretch=STRETCH, Q=Q)


def filter_strip(ax, bands, colours=CHANNEL):
    for band, colour in zip(bands, colours):
        curve = speclite.filters.load_filter(spectra.speclite_name(band))
        response = curve.response / curve.response.max()
        ax.fill_between(curve.wavelength / 1e4, response, color=colour, alpha=0.35, lw=0)
        ax.plot(curve.wavelength / 1e4, response, color=colour, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.35, 2.15)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([0.4, 0.6, 1, 2])
    ax.set_xticklabels(["0.4", "0.6", "1", "2 µm"], fontsize=9, color=MUTED)
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2, pad=1.5, colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def coloured_bands(fig, x, y, bands, colours, suffix):
    """One centred line of text with each band name in its filter colour."""
    muted = dict(color=MUTED, fontsize=9.5)
    parts = [TextArea(band, textprops=dict(color=colour, fontsize=10.5, fontweight="bold"))
              for band, colour in zip(bands, colours)]
    parts.append(TextArea("·  %s" % suffix, textprops=muted))
    ax = axes_at(fig, x - 0.01, y - 0.01, 0.02, 0.02)
    ax.axis("off")
    ax.add_artist(AnchoredOffsetbox(loc="center", child=HPacker(children=parts, sep=4, pad=0),
                                    frameon=False, bbox_to_anchor=(0.5, 0.5),
                                    bbox_transform=ax.transAxes, borderpad=0))


def draw_montage(fig, data, x0=0.72, top=7.7, panel=2.05, gap=0.06):
    systems, rgb, fov = data["systems"], data["rgb"], data["fov"]
    columns = {}
    for col, (observatory, bands) in enumerate(rgb.items()):
        x = x0 + col * (panel + gap)
        columns[observatory] = x
        column_step = "2_lazuli" if observatory == "Lazuli" else "1_surveys"
        step(column_step)
        colour = ACCENT if observatory == "Lazuli" else INK
        planes = systems[0]["images"][observatory]
        fig.text((x + panel / 2) / WIDTH, (top + 0.34) / HEIGHT, TITLES[observatory],
                 ha="center", va="bottom", fontsize=15, fontweight="bold", color=colour)
        fwhm = ([lazuli.fwhm(b) for b in bands] if observatory == "Lazuli"
                else [p[3] for p in planes if p[3]])
        colours = EUCLID_COLOURS if observatory == "Euclid" else CHANNEL
        detail = "PSF FWHM %s" % angle(min(fwhm), max(fwhm))
        fig.text((x + panel / 2) / WIDTH, (top + 0.12) / HEIGHT, detail, ha="center",
                 va="bottom", fontsize=10.5, color=MUTED)
        filter_strip(axes_at(fig, x + 0.1, top - 0.52, panel - 0.2, 0.34), bands, colours)
        # labels run blue -> red like the curves; RGB lists are red-first except Euclid's
        order = slice(None) if observatory == "Euclid" else slice(None, None, -1)
        coloured_bands(fig, x + panel / 2, top - 0.08, bands[order], colours[order],
                       DEPTH[observatory])

        for row, system in enumerate(systems):
            y = top - 0.8 - (row + 1) * panel - row * gap
            ax = axes_at(fig, x, y, panel, panel)
            counts, pixel_scale = system["images"][observatory][0][:2]
            half = counts.shape[0] * pixel_scale / 2
            ax.imshow(colour_composite(system["images"], observatory), origin="lower",
                      extent=(half, -half, -half, half), interpolation="antialiased")
            ax.set_xlim(fov / 2, -fov / 2)
            ax.set_ylim(-fov / 2, fov / 2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(ACCENT if observatory == "Lazuli" else "0.25")
                spine.set_linewidth(1.6 if observatory == "Lazuli" else 0.8)
            if col == 0:
                ax.plot([2.2, 1.2], [-2.15, -2.15], color="w", lw=2, path_effects=OUTLINE)
                ax.text(1.7, -2.05, "1″", color="w", ha="center", va="bottom", fontsize=10,
                        path_effects=OUTLINE)
                info = system["info"]
                kind = "Quad" if info["n_images"] == 4 else "Double"
                ax.text(-0.17, 0.5, kind, transform=ax.transAxes, rotation=90, ha="center",
                        va="center", fontsize=14, fontweight="bold")
                ax.text(-0.07, 0.5, r"$z_d$=%.2f  $z_s$=%.2f" % (info["z_d"], info["z_s"]),
                        transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=10.5, color=MUTED)
            if observatory == "Lazuli" and row == 0:
                step("3_ifs_field")
                field = data["cube"].extent
                ax.add_patch(Rectangle((field[1], field[2]), field[0] - field[1],
                                       field[3] - field[2], fill=False, lw=1.2, ls="--",
                                       ec=ACCENT))
                ax.text(field[0] - 0.08, field[3] + 0.06, "IFS field", color=ACCENT,
                        fontsize=10.5, ha="left", va="bottom", path_effects=OUTLINE)
                hero = (x + panel * (fov / 2 - field[1]) / fov, y + panel / 2)
                for name, colour, lx, ly in labels(data["cube"]):
                    step("7_lens" if name == "G" else "5_quasar_spectra")
                    ax.text(lx, ly, name, color=colour, fontsize=13, fontweight="bold",
                            ha="center", va="center", path_effects=OUTLINE)
                step(column_step)

    span = (columns["LSST"], columns["Roman"] + panel)
    step("1_surveys")
    bracket(fig, *span, top + 0.82, r"DISCOVER  ·  wide-field surveys find $\mathcal{O}(1000)$ lensed AGN", INK)
    step("2_lazuli")
    bracket(fig, columns["Lazuli"], columns["Lazuli"] + panel, top + 0.82,
            "CHARACTERIZE", ACCENT)
    return hero


def bracket(fig, x_start, x_end, y, label, colour):
    ax = axes_at(fig, 0, 0, WIDTH, HEIGHT)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    ax.patch.set_alpha(0)
    ax.plot([x_start, x_start, x_end, x_end], [y - 0.1, y, y, y - 0.1], color=colour, lw=1.2)
    ax.text((x_start + x_end) / 2, y + 0.06, label, ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=colour)


# --------------------------------------------------------------------- IFS


def image_order(cube):
    return list(np.argsort(-cube.magnification))


def labels(cube):
    """``(name, colour, x, y)`` in arcsec, East-left like the panels, for the
    quasar images (A brightest) and the lens G, placed where the label is
    clear of the images."""
    ny, nx = cube.flux.shape[1:]
    out = []
    angles = []
    for k, index in enumerate(image_order(cube)):
        row, col = cube.image_pixels[index]
        x, y = -(col - (nx - 1) / 2) * cube.spaxel, (row - (ny - 1) / 2) * cube.spaxel
        radius = np.hypot(x, y)
        angles.append(np.arctan2(y, x))
        out.append(("ABCD"[k], IMAGE_COLOURS[k], x * (1 + 0.3 / radius), y * (1 + 0.3 / radius)))
    angles = np.sort(angles)
    gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
    middle = angles[np.argmax(gaps)] + gaps.max() / 2
    out.append(("G", LENS_COLOUR, 0.3 * np.cos(middle), 0.3 * np.sin(middle)))
    return out


def near_line(wave, z_s, width=0.025):
    return np.any([np.abs(wave / (rest * (1 + z_s)) - 1) < width for rest in QSO_LINES.values()],
                  axis=0)


def continuum_slice(cube, z_s, low, high):
    """Mean over [low, high] Angstrom, skipping channels near quasar lines."""
    use = (cube.wavelength >= low) & (cube.wavelength <= high) & ~near_line(cube.wavelength, z_s)
    return cube.flux[use].mean(axis=0)


def stretch(image, percent=99.8, soft=0.08):
    scale = np.percentile(image, percent)
    return np.arcsinh(np.clip(image, 0, None) / (soft * scale)) / np.arcsinh(1 / soft)


# blue -> neutral -> red, hues not used elsewhere on the slide
SLICE_COLOURS = ["#4361ee", "#e0e0e0", "#d62828"]


def tinted(colour):
    return LinearSegmentedColormap.from_list("tint", ["black", colour, "white"], N=256)


def draw_cube(fig, cube, z_s, z_d, rect):
    """Three plain continuum slices stacked by wavelength, bluest at the bottom.

    Windows avoid the quasar's lines, so each slice is just the light at that
    wavelength: the blue quasar images dominate short of the lens's 4000 A
    break, and the lens's old stars take over towards the red.

    :return: figure position (inches) of the stack's left edge, for the arrow
    """
    step("4a_slice_blue")
    ax = axes_at(fig, *rect)
    ax.axis("off")
    ny, nx = cube.flux.shape[1:]
    windows = [((0.60, 0.70), "quasar images"),
               ((0.85, 1.00), "lens galaxy appears"),
               ((1.20, 1.50), "lens galaxy: red starlight")]
    slices = []
    for ((low, high), note), colour in zip(windows, SLICE_COLOURS):
        image = gaussian_filter(continuum_slice(cube, z_s, low * 1e4, high * 1e4), 0.7)
        edge = np.concatenate([image[:6].ravel(), image[-6:].ravel()])
        image = image - 1.5 * 1.4826 * np.median(np.abs(edge - np.median(edge)))
        slices.append((stretch(image, soft=0.03), colour, "%.2f–%.2f µm" % (low, high), note))
    slices = slices[::-1]  # top of the stack is the reddest
    skew = Affine2D().scale(1, 0.55).skew_deg(35, 0)
    right = skew.transform([[nx, ny]])[0][0] + 8  # lambda arrow, then labels
    spacing = -0.95 * ny
    corners = {}
    for depth, (image, colour, label, note) in enumerate(slices):
        step(["4c_slice_red", "4b_slice_mid", "4a_slice_blue"][depth])
        place = skew + Affine2D().translate(0, depth * spacing)
        ax.imshow(image, origin="lower", cmap=tinted(colour), vmin=0, vmax=1,
                  extent=(0, nx, 0, ny), transform=place + ax.transData,
                  interpolation="nearest", zorder=depth)
        corners[depth] = place.transform([[0, 0], [nx, 0], [nx, ny], [0, ny]])
        ax.add_patch(Polygon(corners[depth], closed=True, fill=False, ec=colour, lw=1.4,
                             zorder=depth + 0.5))
        mid = corners[depth][1:3].mean(axis=0)[1]
        ax.text(right + 20, mid + 3, label, color=colour, fontsize=13, fontweight="bold",
                va="bottom")
        ax.text(right + 20, mid - 3, note, color=INK, fontsize=11.5, va="top")
    step("4a_slice_blue")
    low, high = corners[2][:, 1].mean(), corners[0][:, 1].mean()
    ax.add_patch(FancyArrowPatch((right, low), (right, high), arrowstyle="-|>",
                                 mutation_scale=20, color="0.85", lw=2.4))
    ax.text(right, high + 3, "λ", color="0.85", fontsize=18, fontweight="bold",
            ha="center", va="bottom", clip_on=False)
    ax.set_xlim(-4, right + 4)
    ax.set_ylim(corners[2][:, 1].min() - 4, corners[0][:, 1].max() + 0.6 * ny)
    ax.set_aspect("equal", anchor="W")
    ax.apply_aspect()
    edge = ax.transData.transform([corners[1][[0, 3]].mean(axis=0)])
    return fig.transFigure.inverted().transform(edge)[0] * [WIDTH, HEIGHT]


def binned(wavelength, values, factor):
    n = len(values) // factor * factor
    return (wavelength[:n].reshape(-1, factor).mean(axis=1),
            values[:n].reshape(-1, factor).mean(axis=1))


def rebinned(wavelength, values, resolution):
    """Mean of *values* in bins of constant width d(ln lambda) = 1 / resolution."""
    edges = np.exp(np.arange(np.log(wavelength[0]), np.log(wavelength[-1]), 1 / resolution))
    index = np.digitize(wavelength, edges)
    keep = [i for i in np.unique(index) if 0 < i < len(edges)]
    return (np.array([wavelength[index == i].mean() for i in keep]),
            np.array([values[index == i].mean() for i in keep]),
            np.array([(index == i).sum() for i in keep]))


def aperture_correction(cube, radius=2.5, oversample=5, samples=12):
    """Fraction of a point source's light inside the extraction aperture, vs
    wavelength, from the same diffraction PSF the cube was built with."""
    pixel = cube.spaxel / oversample
    size = 41 * oversample  # 41 spaxels, centred on a spaxel
    y, x = np.indices((41, 41))
    inside = np.hypot(y - 20, x - 20) <= radius
    wave = np.geomspace(cube.wavelength[0], cube.wavelength[-1], samples)
    fraction = []
    for lam in wave:
        kernel = airy_kernel([lam], [1.0], pixel, size)
        binned_kernel = kernel.reshape(41, oversample, 41, oversample).sum(axis=(1, 3))
        fraction.append(binned_kernel[inside].sum())
    return np.interp(np.log(cube.wavelength), np.log(wave), fraction)


def lens_spectrum(cube, radius=10, mask=4):
    """Lens light and its variance in a 0.4" aperture, quasar images masked."""
    ny, nx = cube.flux.shape[1:]
    y, x = np.indices((ny, nx))
    keep = np.hypot(y - (ny - 1) / 2, x - (nx - 1) / 2) <= radius
    for row, col in cube.image_pixels:
        keep &= np.hypot(y - row, x - col) > mask
    return cube.flux[:, keep].sum(axis=1), (cube.sigma[:, keep] ** 2).sum(axis=1)


def draw_spectra(fig, cube, z_s, z_d, rect):
    """Observed f_lambda of each quasar image and the lens, one panel each."""
    wave = cube.wavelength
    unit = 1e-18
    correction = aperture_correction(cube)
    rows = [("ABCD"[k], IMAGE_COLOURS[k], cube.aperture(*cube.image_pixels[index]) / correction,
             cube.aperture(*cube.image_pixels[index], cube=cube.sigma ** 2) / correction ** 2, 90)
            for k, index in enumerate(image_order(cube))]
    rows.append(("G", LENS_COLOUR, *lens_spectrum(cube), 40))
    step("5_quasar_spectra")
    x, y, width, height = rect
    gap = 0.06
    panel = (height - gap * (len(rows) - 1)) / len(rows)
    for k, (name, colour, flux, variance, resolution) in enumerate(rows):
        ax = axes_at(fig, x, y + (len(rows) - 1 - k) * (panel + gap), width, panel)
        step("7_lens" if name == "G" else "5_quasar_spectra")
        w, f, count = rebinned(wave, flux / unit, resolution)
        noise = np.sqrt(rebinned(wave, variance, resolution)[1] / count) / unit
        detected = f > 2 * noise if name == "G" else np.ones(len(f), bool)
        blue = (f < 2 * noise) & (w < 1e4) & (name == "G")
        if blue.sum() > 2:  # hide the lens wherever bluer than its last non-detection
            detected &= w > w[blue].max()
        w = w / 1e4
        blue = blue & (w > 0)
        good = f[detected & (f > 0)]
        low, high = np.percentile(good, 2) / 1.6, good.max() * 1.6
        # noise can push faint blue bins below zero: pin them to the axis floor
        ax.plot(w, np.where(detected, np.clip(f, low * 1.05, None), np.nan), color=colour, lw=1.1)
        ax.set_yscale("log")
        ax.set_ylim(low, high)
        ticks = [10.0 ** p for p in range(-2, 3) if low <= 10.0 ** p <= high]
        ax.set_yticks(ticks or [float("%.1g" % np.median(good))])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%g" % v))
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(labelsize=9.5, length=3, pad=2, width=1.1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
        if blue.sum() > 2:
            ax.axvspan(0.42, w[blue].max(), color="0.25", alpha=0.35, lw=0)
            ax.text(0.43, 0.15, "not detected in 3 h", color="0.75", fontsize=10.5,
                    transform=ax.get_xaxis_transform(), va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(0.42, 1.7)
        ax.set_xticks([0.5, 0.7, 1.0, 1.3, 1.7])
        ax.set_xticks([], minor=True)
        ax.set_xticklabels(["0.5", "0.7", "1.0", "1.3", "1.7"] if k == len(rows) - 1 else [])
        ax.text(1.0, 0.5, "  " + name, color=colour, fontsize=15, fontweight="bold",
                va="center", transform=ax.transAxes)
        step("5_quasar_spectra")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for rest in QSO_LINES.values():
            observed = rest * (1 + z_s) / 1e4
            if 0.42 < observed < 1.6 and name != "G":
                ax.axvline(observed, color="#bdbdbd", lw=0.6, ls=":", alpha=0.6)
        if k == 0:
            for line, rest in QSO_LINES.items():
                observed = rest * (1 + z_s) / 1e4
                if 0.42 < observed < 1.6:
                    ax.text(observed, 1.04, line, color="#e0e0e0", fontsize=11.5,
                            ha="center", va="bottom", transform=ax.get_xaxis_transform())
        if k == 0:
            ax.text(1.0, 1.04, "binned to R ≈ 90", color=MUTED, fontsize=9.5, ha="right",
                    va="bottom", transform=ax.transAxes)
        if name == "G":
            ax.set_xlabel("observed wavelength [µm]", fontsize=12.5, labelpad=3)
            ax.tick_params(axis="x", labelsize=11)
    fig.text((x - 0.5) / WIDTH, (y + height / 2) / HEIGHT,
             "flux  $f_\\lambda$  [10$^{-18}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]",
             rotation=90, ha="center", va="center", fontsize=11)


def draw_ifs(fig, data, x0=9.85):
    cube = data["cube"]
    info = data["systems"][0]["info"]
    z_s, z_d = float(info["z_s"]), float(info["z_d"])
    step("3_ifs_field")
    fig.text((x0 + 2.95) / WIDTH, 8.5 / HEIGHT, "LAZULI IFS  ·  one pointing, every image",
             ha="center", va="bottom", fontsize=16, fontweight="bold", color=ACCENT)
    fig.text((x0 + 2.95) / WIDTH, 8.18 / HEIGHT,
             "0.4–1.7 µm  ·  R ≈ 100–500  ·  40 mas spaxels  ·  3 h",
             ha="center", va="bottom", fontsize=12, color=MUTED)
    target = draw_cube(fig, cube, z_s, z_d, (x0 - 0.1, 5.0, 6.1, 3.0))
    draw_spectra(fig, cube, z_s, z_d, (x0 + 0.65, 0.72, 5.0, 3.35))
    step("6_caption")
    fig.text((x0 + 3.0) / WIDTH, 4.68 / HEIGHT,
             "Same quasar lines in every image  →  lensing confirmed",
             ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=INK)
    return target


def build(path=None):
    global STEP
    population.setup()
    data = pickle.load(open(path or population.CACHE.parent / "render.pkl", "rb"))
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    STEP = Steps(fig)
    step("0_base")
    fig.text(0.04 / WIDTH, 0.1 / HEIGHT,
             "Simulated with SLSim; quasar hosts from HST COSMOS. Lensed-AGN counts: Oguri & Marshall 2010.\n"
             "Lazuli: Roy et al. 2026 (arXiv:2601.02556). Modelled, not official: diffraction-limited "
             "3.06 m Airy PSF + 4 mas jitter, SDSS-shaped filters, throughput, noise, exposure times.",
             fontsize=9, color="0.55", ha="left", va="bottom")
    hero = draw_montage(fig, data)
    target = draw_ifs(fig, data)
    step("3_ifs_field")
    ax = axes_at(fig, 0, 0, WIDTH, HEIGHT)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    ax.patch.set_alpha(0)
    ax.add_patch(FancyArrowPatch(hero, target - [0.08, 0], arrowstyle="-|>", mutation_scale=14,
                                 color=ACCENT, lw=1.4, connectionstyle="arc3,rad=-0.25"))
    step(None)
    return fig


def main(path=None, name="lazuli_lensed_agn"):
    FIGURES.mkdir(exist_ok=True)
    fig = build(path)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / (name + "." + suffix), dpi=200)
    return fig


def keynote(path=None, dpi=200):
    """One transparent full-slide PNG per build step, holding only that step's
    new pixels, so the layers stack exactly in Keynote; plus cumulative
    previews on black."""
    out = FIGURES / "keynote"
    out.mkdir(parents=True, exist_ok=True)
    fig = build(path)
    previous = None
    for number, name in enumerate(STEPS):
        STEP.show_until(name)
        fig.savefig(out / ("preview_%02d_%s.png" % (number, name)), dpi=dpi)
        buffer = out / "_cumulative.png"
        fig.savefig(buffer, dpi=dpi, transparent=True)
        current = plt.imread(buffer)
        layer = current.copy()
        if previous is not None:
            layer[np.all(np.isclose(current, previous, atol=1 / 512), axis=-1)] = 0
        plt.imsave(out / ("build_%02d_%s.png" % (number, name)), layer)
        previous = current
    buffer.unlink()
    STEP.show_until(STEPS[-1])


if __name__ == "__main__":
    if "--keynote" in sys.argv:
        keynote()
    else:
        main(None, *[a for a in sys.argv[1:] if not a.startswith("--")])
