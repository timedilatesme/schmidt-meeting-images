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
from matplotlib.colors import LinearSegmentedColormap
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
QSO_LINES = {"C IV": 1549.06, "C III]": 1908.73, "Mg II": 2798.75, "Hγ": 4340.47,
             "Hβ": 4861.33}

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
    ax.set_xlim(0.35, 2.0)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([0.4, 0.6, 1, 1.5])
    ax.set_xticklabels(["0.4", "0.6", "1", "1.5 µm"], fontsize=8, color=MUTED)
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2, pad=1.5, colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


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
        composite = ("Q1 style: VIS + Y J H" if observatory == "Euclid"
                     else "RGB = %s" % " ".join(bands))
        fig.text((x + panel / 2) / WIDTH, (top - 0.08) / HEIGHT,
                 "%s  ·  %s" % (composite, DEPTH[observatory]),
                 ha="center", va="center", fontsize=8.5, color=MUTED)

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

    span = (columns["LSST"], columns["Roman"] + panel)
    bracket(fig, *span, top + 0.82, "DISCOVER  ·  wide-field surveys find ~10³–10⁴ lensed AGN", INK)
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


def line_slice(cube, z_s):
    """Continuum-subtracted cube slice on the strongest broad line."""
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


def stretch(image, percent=99.8, soft=0.08):
    scale = np.percentile(image, percent)
    return np.arcsinh(np.clip(image, 0, None) / (soft * scale)) / np.arcsinh(1 / soft)


def tinted(colour):
    return LinearSegmentedColormap.from_list("t", ["black", colour, "white"], N=256)


def draw_cube(fig, cube, z_s, rect):
    """Three slices stacked by observed wavelength, shortest at the bottom.

    :return: figure position (inches) of the stack's left edge, for the arrow
    """
    ax = axes_at(fig, *rect)
    ax.axis("off")
    ny, nx = cube.flux.shape[1:]
    line, line_name, line_wave = line_slice(cube, z_s)
    rest = QSO_LINES[line_name] / 1e4
    slices = [  # top to bottom
        (stretch(line), "#7ee081", "%.2f µm: %s line only" % (line_wave / 1e4, line_name),
         "rest %.3f µm × (1 + $z_s$): quasar images only" % rest),
        (stretch(cube.collapse(10000, 12000), soft=0.01), "#ff6b5a",
         "1.00–1.20 µm continuum", "quasar images + lens galaxy"),
        (stretch(cube.collapse(5800, 7600), soft=0.02), "#6fa8ff",
         "0.58–0.76 µm continuum", "quasar images dominate"),
    ]
    skew = Affine2D().scale(1, 0.55).skew_deg(35, 0)
    right = skew.transform([[nx, ny]])[0][0] + 8  # lambda arrow, then labels
    step = -0.95 * ny
    corners = {}
    for depth, (image, colour, label, note) in enumerate(slices):
        place = skew + Affine2D().translate(0, depth * step)
        ax.imshow(image, origin="lower", cmap=tinted(colour), vmin=0, vmax=1,
                  extent=(0, nx, 0, ny), transform=place + ax.transData,
                  interpolation="nearest", zorder=depth)
        corners[depth] = place.transform([[0, 0], [nx, 0], [nx, ny], [0, ny]])
        ax.add_patch(Polygon(corners[depth], closed=True, fill=False, ec=colour, lw=1.2,
                             zorder=depth + 0.5))
        mid = corners[depth][1:3].mean(axis=0)[1]
        ax.text(right + 10, mid + 4, label, color=colour, fontsize=10,
                fontweight="bold", va="bottom")
        ax.text(right + 10, mid - 2, note, color=MUTED, fontsize=8.5, va="top")
        front = place
    for k, index in enumerate(image_order(cube)):
        row, col = cube.image_pixels[index]
        px, py = front.transform([[col + 0.5, row + 0.5]])[0]
        ax.text(px + 3.5, py + 2.5, "ABCD"[k], color=IMAGE_COLOURS[k], fontsize=10,
                fontweight="bold", path_effects=OUTLINE, zorder=10)
    cx, cy = front.transform([[nx / 2, ny / 2]])[0]
    ax.text(cx - 9, cy + 3, "G", color=LENS_COLOUR, fontsize=10, fontweight="bold",
            path_effects=OUTLINE, zorder=10)
    low, high = corners[2][:, 1].mean(), corners[0][:, 1].mean()
    ax.add_patch(FancyArrowPatch((right, low), (right, high),
                                 arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))
    ax.text(right, high + 2, "λ", color=MUTED, fontsize=12, ha="center",
            va="bottom", clip_on=False)
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


def label_lines(ax, lines, redshift, colour, y, wavelength, text=True):
    for name, rest in lines.items():
        observed = rest * (1 + redshift) / 1e4
        if wavelength[0] / 1e4 < observed < wavelength[-1] / 1e4:
            ax.axvline(observed, color=colour, lw=0.6, ls=":", alpha=0.7)
            if text:
                ax.text(observed, y, name, color=colour, fontsize=8.5, rotation=90,
                        ha="right", va="top", transform=ax.get_xaxis_transform())


def draw_spectra(fig, cube, z_s, rect):
    """Spectrum of each quasar image, normalised and offset, f_nu."""
    order = image_order(cube)
    wave = cube.wavelength
    ax = axes_at(fig, *rect)
    window = (wave > 10000) & (wave < 12000)
    offset = 1.1
    for k, index in enumerate(order):
        row, col = cube.image_pixels[index]
        f_nu = cube.aperture(row, col) * wave ** 2
        f_nu /= np.median(f_nu[window])
        w, f = binned(wave, f_nu, 4)
        shift = (len(order) - 1 - k) * offset
        ax.plot(w / 1e4, f + shift, color=IMAGE_COLOURS[k], lw=1.1)
        ax.text(1.72, 1 + shift, "ABCD"[k], color=IMAGE_COLOURS[k], fontsize=11,
                fontweight="bold", va="center", transform=ax.transData, clip_on=False)
    ax.set_xscale("log")
    ax.set_xlim(0.42, 1.7)
    ax.set_ylim(0, 1 + (len(order) - 1) * offset + 1.6)
    ax.set_xticks([0.5, 0.7, 1.0, 1.3, 1.7])
    ax.set_xticklabels(["0.5", "0.7", "1.0", "1.3", "1.7"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("observed wavelength [µm]", fontsize=10)
    ax.set_yticks([])
    ax.set_ylabel("normalised $f_\\nu$ + offset", fontsize=9.5)
    rest_axis = ax.secondary_xaxis("top", functions=(lambda x: x / (1 + z_s),
                                                     lambda x: x * (1 + z_s)))
    rest_axis.set_xticks([0.15, 0.2, 0.3, 0.4, 0.5])
    rest_axis.set_xticklabels(["0.15", "0.2", "0.3", "0.4", "0.5"], fontsize=8, color=MUTED)
    rest_axis.set_xticks([], minor=True)
    rest_axis.tick_params(colors=MUTED)
    rest_axis.set_xlabel("rest-frame wavelength at $z_s$ = %.2f [µm]" % z_s, fontsize=9,
                         color=MUTED)
    for name, rest in QSO_LINES.items():
        observed = rest * (1 + z_s) / 1e4
        if 0.42 < observed < 1.7:
            ax.axvline(observed, color="#bdbdbd", lw=0.6, ls=":", alpha=0.7)
            ax.text(observed, 0.985, name, color="#d0d0d0", fontsize=9, rotation=90,
                    ha="right", va="top", transform=ax.get_xaxis_transform())
    ax.tick_params(labelsize=8.5)
    for spine in ("right", "left"):
        ax.spines[spine].set_visible(False)

def draw_ifs(fig, data, x0=9.85):
    cube = data["cube"]
    z_s = float(data["systems"][0]["info"]["z_s"])
    fig.text((x0 + 2.95) / WIDTH, 8.52 / HEIGHT, "LAZULI IFS  ·  one pointing, every image",
             ha="center", va="bottom", fontsize=11, fontweight="bold", color=ACCENT)
    fig.text((x0 + 2.95) / WIDTH, 8.28 / HEIGHT,
             "0.4–1.7 µm  ·  R ≈ 100–500  ·  40 mas spaxels  ·  4 × 900 s",
             ha="center", va="bottom", fontsize=9, color=MUTED)
    target = draw_cube(fig, cube, z_s, (x0 - 0.1, 4.95, 6.1, 3.15))
    draw_spectra(fig, cube, z_s, (x0 + 0.45, 0.75, 5.25, 3.35))
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
