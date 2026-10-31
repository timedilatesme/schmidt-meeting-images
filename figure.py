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
from astropy.visualization import make_lupton_rgb
from slsim.Util.euclid_rgb_util import euclid_rgb_from_image_list
from matplotlib import patheffects
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D

import population
from ifs import spectra
from observatories import lazuli

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
    ax.set_xticks([0.4, 0.6, 1, 1.5, 2])
    ax.set_xticklabels(["0.4", "0.6", "1", "1.5", "2 µm"], fontsize=8, color=MUTED)
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2, pad=1.5, colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def coloured_bands(fig, x, y, bands, colours, suffix):
    """One centred line of text with each band name in its filter colour."""
    muted = dict(color=MUTED, fontsize=8.5)
    parts = [TextArea(band, textprops=dict(color=colour, fontsize=9, fontweight="bold"))
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
        colour = ACCENT if observatory == "Lazuli" else INK
        planes = systems[0]["images"][observatory]
        fig.text((x + panel / 2) / WIDTH, (top + 0.34) / HEIGHT, TITLES[observatory],
                 ha="center", va="bottom", fontsize=15, fontweight="bold", color=colour)
        fwhm = ([lazuli.fwhm(b) for b in bands] if observatory == "Lazuli"
                else [p[3] for p in planes if p[3]])
        colours = EUCLID_COLOURS if observatory == "Euclid" else CHANNEL
        pixels = sorted({p[1] for p in planes})
        detail = "pixel %s  ·  PSF %s" % (angle(*pixels), angle(min(fwhm), max(fwhm)))
        fig.text((x + panel / 2) / WIDTH, (top + 0.12) / HEIGHT, detail, ha="center",
                 va="bottom", fontsize=9, color=MUTED)
        filter_strip(axes_at(fig, x + 0.1, top - 0.52, panel - 0.2, 0.34), bands, colours)
        coloured_bands(fig, x + panel / 2, top - 0.08, bands, colours, DEPTH[observatory])

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
                        va="center", fontsize=12, fontweight="bold")
                ax.text(-0.07, 0.5, r"$z_d$=%.2f  $z_s$=%.2f" % (info["z_d"], info["z_s"]),
                        transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=9, color=MUTED)
            if observatory == "Lazuli" and row == 0:
                field = data["cube"].extent
                ax.add_patch(Rectangle((field[1], field[2]), field[0] - field[1],
                                       field[3] - field[2], fill=False, lw=1.2, ls="--",
                                       ec=ACCENT))
                ax.text(field[0] - 0.08, field[3] + 0.06, "IFS field", color=ACCENT,
                        fontsize=8.5, ha="left", va="bottom", path_effects=OUTLINE)
                hero = (x + panel * (fov / 2 - field[1]) / fov, y + panel / 2)
                for name, colour, lx, ly in labels(data["cube"]):
                    ax.text(lx, ly, name, color=colour, fontsize=11, fontweight="bold",
                            ha="center", va="center", path_effects=OUTLINE)

    span = (columns["LSST"], columns["Roman"] + panel)
    bracket(fig, *span, top + 0.82, r"DISCOVER  ·  wide-field surveys find $\mathcal{O}(1000)$ lensed AGN", INK)
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
            fontsize=11, fontweight="bold", color=colour)


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


def line_slice(cube, z_s):
    """Continuum-subtracted cube slice on the broad line detected best."""
    def subtracted(observed, cube_):
        width = observed * 0.012
        offset = np.abs(cube.wavelength - observed)
        on, side = offset < width, (offset > 2.5 * width) & (offset < 5 * width)
        return cube_[on].mean(axis=0) - np.median(cube_[side], axis=0), on

    best = None
    for label, rest in QSO_LINES.items():
        observed = rest * (1 + z_s)
        if not cube.wavelength[10] < observed < cube.wavelength[-10]:
            continue
        signal, on = subtracted(observed, cube.model)
        row, col = cube.image_pixels[image_order(cube)[0]]
        y, x = np.indices(signal.shape)
        aperture = np.hypot(y - row, x - col) <= 2.5
        noise = np.sqrt((cube.sigma[on][:, aperture] ** 2).sum()) / on.sum()
        score = signal[aperture].sum() / noise
        if best is None or score > best[0]:
            best = (score, label, observed)
    _, name, observed = best
    return subtracted(observed, cube.flux)[0], name, observed


def continuum_slice(cube, z_s, low, high):
    """Mean over [low, high] Angstrom, skipping channels near quasar lines."""
    use = (cube.wavelength >= low) & (cube.wavelength <= high) & ~near_line(cube.wavelength, z_s)
    return cube.flux[use].mean(axis=0)


def stretch(image, percent=99.8, soft=0.08):
    scale = np.percentile(image, percent)
    return np.arcsinh(np.clip(image, 0, None) / (soft * scale)) / np.arcsinh(1 / soft)


def draw_cube(fig, cube, z_s, z_d, rect):
    """Three slices stacked by observed wavelength, shortest at the bottom.

    Continuum windows are set by the lens redshift: one blueward of the lens's
    4000 A break, where its old stars are faint, and one at rest 0.6-0.85 um,
    where they are brightest.

    :return: figure position (inches) of the stack's left edge, for the arrow
    """
    ax = axes_at(fig, *rect)
    ax.axis("off")
    ny, nx = cube.flux.shape[1:]
    line, line_name, line_wave = line_slice(cube, z_s)
    blue = (max(0.30 * (1 + z_d), 0.42), 0.38 * (1 + z_d))
    red = (0.60 * (1 + z_d), min(0.85 * (1 + z_d), 1.65))
    slices = sorted([
        (line_wave / 1e4, stretch(line),
         "%.2f µm: %s line (rest %.3f µm)" % (line_wave / 1e4, line_name,
                                              QSO_LINES[line_name] / 1e4),
         "continuum removed: quasar broad lines only"),
        (np.mean(blue), stretch(continuum_slice(cube, z_s, *np.multiply(blue, 1e4)), soft=0.02),
         "%.2f–%.2f µm continuum" % blue,
         "lens rest %.2f–%.2f µm (old stars faint): quasars dominate"
         % (blue[0] / (1 + z_d), blue[1] / (1 + z_d))),
        (np.mean(red), stretch(continuum_slice(cube, z_s, *np.multiply(red, 1e4)), soft=0.01),
         "%.2f–%.2f µm continuum" % red,
         "lens rest %.2f–%.2f µm (old stars bright): lens G shows"
         % (red[0] / (1 + z_d), red[1] / (1 + z_d))),
    ], key=lambda item: -item[0])  # top of the stack is the longest wavelength
    skew = Affine2D().scale(1, 0.55).skew_deg(35, 0)
    right = skew.transform([[nx, ny]])[0][0] + 8  # lambda arrow, then labels
    step = -0.95 * ny
    corners = {}
    for depth, (_, image, label, note) in enumerate(slices):
        place = skew + Affine2D().translate(0, depth * step)
        ax.imshow(image, origin="lower", cmap="magma", vmin=0, vmax=1, extent=(0, nx, 0, ny),
                  transform=place + ax.transData, interpolation="nearest", zorder=depth)
        corners[depth] = place.transform([[0, 0], [nx, 0], [nx, ny], [0, ny]])
        ax.add_patch(Polygon(corners[depth], closed=True, fill=False, ec="0.7", lw=1.0,
                             zorder=depth + 0.5))
        mid = corners[depth][1:3].mean(axis=0)[1]
        ax.text(right + 10, mid + 4, label, color=INK, fontsize=10, fontweight="bold",
                va="bottom")
        ax.text(right + 10, mid - 2, note, color=MUTED, fontsize=8, va="top")
    low, high = corners[2][:, 1].mean(), corners[0][:, 1].mean()
    ax.add_patch(FancyArrowPatch((right, low), (right, high), arrowstyle="-|>",
                                 mutation_scale=12, color=MUTED, lw=1.2))
    ax.text(right, high + 2, "λ", color=MUTED, fontsize=12, ha="center", va="bottom",
            clip_on=False)
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
    rows = [("ABCD"[k], IMAGE_COLOURS[k], cube.aperture(*cube.image_pixels[index]),
             cube.aperture(*cube.image_pixels[index], cube=cube.sigma ** 2), 90)
            for k, index in enumerate(image_order(cube))]
    rows.append(("G", LENS_COLOUR, *lens_spectrum(cube), 40))
    x, y, width, height = rect
    gap = 0.06
    panel = (height - gap * (len(rows) - 1)) / len(rows)
    for k, (name, colour, flux, variance, resolution) in enumerate(rows):
        ax = axes_at(fig, x, y + (len(rows) - 1 - k) * (panel + gap), width, panel)
        w, f, count = rebinned(wave, flux / unit, resolution)
        noise = np.sqrt(rebinned(wave, variance, resolution)[1] / count) / unit
        detected = f > (2 * noise if name == "G" else 0)
        blue = (f < 2 * noise) & (w < 1e4) & (name == "G")
        if blue.sum() > 2:  # hide the lens wherever bluer than its last non-detection
            detected &= w > w[blue].max()
        w = w / 1e4
        blue = blue & (w > 0)
        ax.plot(w, np.where(detected, f, np.nan), color=colour, lw=1.1)
        ax.set_yscale("log")
        good = f[detected]
        low, high = np.percentile(good, 2) / 1.6, good.max() * 1.6
        ax.set_ylim(low, high)
        ticks = [10.0 ** p for p in range(-2, 3) if low <= 10.0 ** p <= high]
        ax.set_yticks(ticks or [float("%.1g" % np.median(good))])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%g" % v))
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(labelsize=7.5, length=2, pad=1.5)
        if blue.sum() > 2:
            ax.axvspan(0.42, w[blue].max(), color="0.25", alpha=0.35, lw=0)
            ax.text(0.425, 0.12, "below noise in 1 h", color="0.7", fontsize=8,
                    transform=ax.get_xaxis_transform(), va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(0.42, 1.7)
        ax.set_xticks([0.5, 0.7, 1.0, 1.3, 1.7])
        ax.set_xticks([], minor=True)
        ax.set_xticklabels(["0.5", "0.7", "1.0", "1.3", "1.7"] if k == len(rows) - 1 else [])
        ax.text(1.0, 0.5, "  " + name, color=colour, fontsize=12, fontweight="bold",
                va="center", transform=ax.transAxes)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for rest in QSO_LINES.values():
            observed = rest * (1 + z_s) / 1e4
            if 0.42 < observed < 1.7 and name != "G":
                ax.axvline(observed, color="#bdbdbd", lw=0.6, ls=":", alpha=0.6)
        if k == 0:
            rest_axis = ax.secondary_xaxis("top", functions=(lambda v: v / (1 + z_s),
                                                             lambda v: v * (1 + z_s)))
            rest_axis.set_xticks([0.12, 0.15, 0.2, 0.3, 0.4])
            rest_axis.set_xticklabels(["0.12", "0.15", "0.2", "0.3", "0.4"], fontsize=7.5)
            rest_axis.set_xticks([], minor=True)
            rest_axis.tick_params(colors=MUTED, length=2, pad=1)
            rest_axis.set_xlabel("quasar rest-frame wavelength [µm]", fontsize=8.5,
                                 color=MUTED, labelpad=2)
            for line, rest in QSO_LINES.items():
                observed = rest * (1 + z_s) / 1e4
                if 0.42 < observed < 1.7:
                    ax.text(observed * 1.008, 0.95, line, color="#d0d0d0", fontsize=8.5,
                            ha="left", va="top", transform=ax.get_xaxis_transform())
        if name == "G":
            ax.set_xlabel("observed wavelength [µm]", fontsize=10)
    fig.text((x - 0.42) / WIDTH, (y + height / 2) / HEIGHT,
             "$f_\\lambda$  [10$^{-18}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]",
             rotation=90, ha="center", va="center", fontsize=9)


def draw_ifs(fig, data, x0=9.85):
    cube = data["cube"]
    info = data["systems"][0]["info"]
    z_s, z_d = float(info["z_s"]), float(info["z_d"])
    fig.text((x0 + 2.95) / WIDTH, 8.52 / HEIGHT, "LAZULI IFS  ·  one pointing, every image",
             ha="center", va="bottom", fontsize=11, fontweight="bold", color=ACCENT)
    fig.text((x0 + 2.95) / WIDTH, 8.28 / HEIGHT,
             "0.4–1.7 µm  ·  R ≈ 100–500  ·  40 mas spaxels  ·  4 × 900 s",
             ha="center", va="bottom", fontsize=9, color=MUTED)
    target = draw_cube(fig, cube, z_s, z_d, (x0 - 0.1, 4.95, 6.1, 3.15))
    draw_spectra(fig, cube, z_s, z_d, (x0 + 0.55, 0.62, 5.15, 3.55))
    fig.text((x0 + 3.05) / WIDTH, 4.72 / HEIGHT,
             "Same broad lines at the same redshift in every image  →  lensed quasar confirmed",
             ha="center", va="bottom", fontsize=9.5, color=INK)
    return target


def main(path=None, name="lazuli_lensed_agn"):
    population.setup()
    data = pickle.load(open(path or population.CACHE.parent / "render.pkl", "rb"))
    FIGURES.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    hero = draw_montage(fig, data)
    target = draw_ifs(fig, data)
    ax = axes_at(fig, 0, 0, WIDTH, HEIGHT)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    ax.patch.set_alpha(0)
    ax.add_patch(FancyArrowPatch(hero, target - [0.08, 0], arrowstyle="-|>", mutation_scale=14,
                                 color=ACCENT, lw=1.4, connectionstyle="arc3,rad=-0.25"))
    fig.text(0.04 / WIDTH, 0.12 / HEIGHT,
             "Simulated with SLSim: COSMOS hosts, qsogen quasar SEDs, static microlensing. "
             "Lazuli: 3.06 m, Strehl > 0.8 at 633 nm; specs from Roy et al. 2026 (arXiv:2601.02556). "
             "Lazuli throughput, sky, noise and exposure times are notional. Lensed-AGN counts: Oguri & Marshall 2010.",
             fontsize=7.5, color="0.5", ha="left", va="bottom")
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / (name + "." + suffix), dpi=200)
    return fig


if __name__ == "__main__":
    main(None, *sys.argv[1:])
