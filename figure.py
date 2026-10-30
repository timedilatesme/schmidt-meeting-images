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
LENS_COLOUR = "#e0a96d"
OUTLINE = [patheffects.withStroke(linewidth=2.2, foreground="black")]

TITLES = {"LSST": "Rubin LSST", "Euclid": "Euclid", "Roman": "Roman", "Lazuli": "Lazuli WCC"}
DEPTH = {"LSST": "10-yr", "Euclid": "wide", "Roman": "HLWAS", "Lazuli": "4×300 s"}

QSO_LINES = {"Lyα": 1216, "C IV": 1549, "C III]": 1909, "Mg II": 2798,
             "Hβ": 4861, "[O III]": 5007, "Hα": 6563}
LENS_LINES = {"Ca H&K": 3950, "Mg b": 5175, "Na D": 5893}

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
    if max(values) >= 0.1:
        return "–".join("%.2g" % v for v in values) + "″"
    return "–".join("%d" % round(1e3 * v) for v in values) + " mas"


# ----------------------------------------------------------------- montage


def surface_brightness(counts, pixel_scale, zero_point, _seeing):
    return counts * 10 ** (-0.4 * (zero_point - 31.4)) / pixel_scale ** 2


def colour_composite(images, observatory):
    white = np.percentile(np.mean([surface_brightness(*p) for p in images[WHITE_REFERENCE]],
                                  axis=0), WHITE_PERCENTILE)
    planes = [surface_brightness(*p) / white for p in images[observatory]]
    return make_lupton_rgb(*planes, minimum=0, stretch=STRETCH, Q=Q)


def filter_strip(ax, bands):
    for band, colour in zip(bands, CHANNEL):
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
        detail = "pixel %s" % angle(planes[-1][1])
        detail += "  ·  PSF %s" % angle(min(fwhm), max(fwhm))
        fig.text((x + panel / 2) / WIDTH, (top + 0.12) / HEIGHT, detail, ha="center",
                 va="bottom", fontsize=9, color=MUTED)
        filter_strip(axes_at(fig, x + 0.1, top - 0.42, panel - 0.2, 0.42), bands)
        fig.text((x + panel / 2) / WIDTH, (top - 0.62) / HEIGHT,
                 "RGB = %s  ·  %s" % (" ".join(bands), DEPTH[observatory]),
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
                hero = (x + panel, y + panel / 2)

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
        noise = np.sqrt((cube.sigma[on] ** 2).sum(axis=0)) / on.sum()
        score = np.max(signal / noise)
        if best is None or score > best[0]:
            best = (score, label, observed)
    _, name, observed = best
    return subtracted(observed, cube.flux)[0], name, observed


def stretch(image, percent=99.8, soft=0.08):
    scale = np.percentile(image, percent)
    return np.arcsinh(np.clip(image, 0, None) / (soft * scale)) / np.arcsinh(1 / soft)


def tinted(colour):
    return LinearSegmentedColormap.from_list("t", ["black", colour, "white"], N=256)


def draw_cube(fig, cube, z_s, z_d, rect):
    ax = axes_at(fig, *rect)
    ax.axis("off")
    ny, nx = cube.flux.shape[1:]
    line, line_name, line_wave = line_slice(cube, z_s)
    slices = [
        (stretch(cube.collapse(11000, 16500), soft=0.01), "#ff6b5a",
         "1.10–1.65 µm continuum", "quasar images + lens galaxy"),
        (stretch(line), "#7ee081", "%s  %.2f µm, continuum-subtracted" % (line_name, line_wave / 1e4),
         "quasar images only"),
        (stretch(cube.collapse(5500, 7500), soft=0.02), "#6fa8ff",
         "0.55–0.75 µm continuum", "quasar images + lensed host"),
    ]
    skew = Affine2D().scale(1, 0.55).skew_deg(35, 0)
    dx, dy = 0.0, -0.95 * ny
    for depth, (image, colour, label, note) in enumerate(slices):
        offset = Affine2D().translate(depth * dx, depth * dy)
        transform = skew + offset + ax.transData
        ax.imshow(image, origin="lower", cmap=tinted(colour), vmin=0, vmax=1,
                  extent=(0, nx, 0, ny), transform=transform, interpolation="nearest",
                  zorder=depth)
        corners = (skew + offset).transform([[0, 0], [nx, 0], [nx, ny], [0, ny]])
        ax.add_patch(Polygon(corners, closed=True, fill=False, ec=colour, lw=1.2,
                             zorder=depth + 0.5))
        right = corners[1:3].mean(axis=0)
        ax.text(right[0] + 12, right[1] + 4, label, color=colour, fontsize=10,
                fontweight="bold", va="bottom")
        ax.text(right[0] + 12, right[1] - 2, note, color=MUTED, fontsize=9, va="top")
        if depth == len(slices) - 1:
            front = (skew + offset)
    for k, index in enumerate(image_order(cube)):
        row, col = cube.image_pixels[index]
        px, py = front.transform([[col + 0.5, row + 0.5]])[0]
        ax.text(px + 3.5, py + 2.5, "ABCD"[k], color=IMAGE_COLOURS[k], fontsize=10,
                fontweight="bold", path_effects=OUTLINE, zorder=10)
    cx, cy = front.transform([[nx / 2, ny / 2]])[0]
    ax.text(cx + 3, cy - 6, "G", color=LENS_COLOUR, fontsize=10, fontweight="bold",
            path_effects=OUTLINE, zorder=10)
    top = skew.transform([[0, ny]])[0]
    bottom = (skew + Affine2D().translate(2 * dx, 2 * dy)).transform([[0, 0]])[0]
    ax.add_patch(FancyArrowPatch((bottom[0] - 10, bottom[1]), (top[0] - 10, top[1]),
                                 arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))
    ax.text(min(bottom[0], top[0]) - 16, (top[1] + bottom[1]) / 2, "λ", color=MUTED,
            fontsize=13, ha="right", va="center")
    ax.set_xlim(bottom[0] - 30, nx * 2.55)
    ax.set_ylim(bottom[1] - 4, top[1] + 4)
    ax.set_aspect("equal")
    return line_name


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


def draw_spectra(fig, cube, z_s, z_d, rect_spectra, rect_ratio):
    order = image_order(cube)
    wave = cube.wavelength
    ny, nx = cube.flux.shape[1:]
    unit = 1e-18
    ax = axes_at(fig, *rect_spectra)
    extracted = {}
    for k, index in enumerate(order):
        row, col = cube.image_pixels[index]
        flux = cube.aperture(row, col)
        model = cube.aperture(row, col, cube=cube.model)
        extracted["ABCD"[k]] = (flux, model)
        w, f = binned(wave, flux / unit, 3)
        ax.plot(w / 1e4, f, color=IMAGE_COLOURS[k], lw=0.9, alpha=0.95,
                label="image %s" % "ABCD"[k])
    lens = cube.aperture((ny - 1) / 2, (nx - 1) / 2, radius=6)
    w, f = binned(wave, lens / unit, 6)
    ax.plot(w / 1e4, f, color=LENS_COLOUR, lw=1.1, label="lens galaxy G")
    ax.set_xscale("log")
    ax.set_yscale("log")
    top = max(np.percentile(extracted["A"][1] / unit, 99.5), 1)
    ax.set_ylim(top / 300, top * 3)
    ax.set_xlim(0.4, 1.7)
    ax.set_xticks([0.4, 0.5, 0.7, 1.0, 1.3, 1.7])
    ax.set_xticklabels([])
    ax.set_xticks([], minor=True)
    ax.set_ylabel("$f_\\lambda$  [10$^{-18}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=9)
    ax.tick_params(labelsize=8)
    label_lines(ax, QSO_LINES, z_s, "#bdbdbd", 0.97, wave)
    label_lines(ax, LENS_LINES, z_d, LENS_COLOUR, 0.30, wave)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=3, handlelength=1.2, columnspacing=1.0, labelcolor=INK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # chromatic microlensing: pick the pair whose continuum ratio changes most
    boost = cube.spectra["boost"][order]
    spread = [(np.ptp(np.log(boost[a] / boost[b])), a, b)
              for a in range(len(order)) for b in range(len(order)) if a < b]
    _, a, b = max(spread)
    first, second = "ABCD"[a], "ABCD"[b]
    ratio_ax = axes_at(fig, *rect_ratio)
    macro = cube.magnification[order[a]] / cube.magnification[order[b]]
    w, numerator = binned(wave, extracted[first][0], 6)
    _, denominator = binned(wave, extracted[second][0], 6)
    _, model_num = binned(wave, extracted[first][1], 2)
    wm, model_den = binned(wave, extracted[second][1], 2)
    ratio_ax.plot(w / 1e4, numerator / denominator / macro, "o", ms=2.2, color=INK,
                  alpha=0.75, label="observed")
    ratio_ax.plot(wm / 1e4, model_num / model_den / macro, color=ACCENT, lw=1.6,
                  label="truth")
    ratio_ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ratio_ax.text(0.41, 1.0, "no microlensing", color=MUTED, fontsize=8, ha="left",
                  va="bottom")
    ratio_ax.set_xscale("log")
    ratio_ax.set_xlim(0.4, 1.7)
    ratio_ax.set_xticks([0.4, 0.5, 0.7, 1.0, 1.3, 1.7])
    ratio_ax.set_xticklabels(["0.4", "0.5", "0.7", "1.0", "1.3", "1.7"])
    ratio_ax.set_xticks([], minor=True)
    ratio_ax.set_xlabel("observed wavelength [µm]", fontsize=9.5)
    truth = model_num / model_den / macro
    pad = 0.15 * np.ptp(truth) + 0.05
    ratio_ax.set_ylim(min(truth.min(), 1) - pad, max(truth.max(), 1) + pad)
    ratio_ax.set_ylabel("%s / %s\n÷ macro" % (first, second), fontsize=9)
    ratio_ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ratio_ax.spines[spine].set_visible(False)
    label_lines(ratio_ax, QSO_LINES, z_s, "#bdbdbd", 0.97, wave, text=False)
    ratio_ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2, labelcolor=INK)
    return first, second


def draw_ifs(fig, data, x0=9.85):
    cube = data["cube"]
    info = data["systems"][0]["info"]
    z_s, z_d = info["z_s"], info["z_d"]
    fig.text((x0 + 2.95) / WIDTH, 8.82 / HEIGHT, "LAZULI IFS  ·  one pointing, every image",
             ha="center", va="bottom", fontsize=11, fontweight="bold", color=ACCENT)
    fig.text((x0 + 2.95) / WIDTH, 8.56 / HEIGHT,
             "0.4–1.7 µm  ·  R ≈ 100–500  ·  40 mas spaxels  ·  4 × 900 s",
             ha="center", va="bottom", fontsize=9, color=MUTED)
    line_name = draw_cube(fig, cube, z_s, z_d, (x0 - 0.2, 4.95, 6.2, 3.5))
    first, second = draw_spectra(fig, cube, z_s, z_d, (x0 + 0.55, 2.5, 5.35, 2.3),
                                 (x0 + 0.55, 0.8, 5.35, 1.3))
    fig.text((x0 + 3.2) / WIDTH, 4.86 / HEIGHT,
             "Spectra of each image and the lens: both redshifts and lensing confirmed in one visit",
             ha="center", va="bottom", fontsize=9, color=INK)
    fig.text((x0 + 3.2) / WIDTH, 2.36 / HEIGHT,
             "Continuum ratio %s/%s varies with λ but not in broad lines → microlensing of the accretion disk"
             % (first, second), ha="center", va="top", fontsize=8.5, color=INK)


def main(path=None):
    population.setup()
    data = pickle.load(open(path or population.CACHE.parent / "render.pkl", "rb"))
    FIGURES.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    hero = draw_montage(fig, data)
    draw_ifs(fig, data)
    ax = axes_at(fig, 0, 0, WIDTH, HEIGHT)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    ax.patch.set_alpha(0)
    ax.add_patch(FancyArrowPatch(hero, (9.75, 6.6), arrowstyle="-|>", mutation_scale=14,
                                 color=ACCENT, lw=1.4, connectionstyle="arc3,rad=-0.25"))
    fig.text(0.04 / WIDTH, 0.12 / HEIGHT,
             "Simulated with SLSim (COSMOS hosts, qsogen quasar SEDs, chromatic microlensing). "
             "Lazuli: 3.06 m, diffraction-limited; specs from Roy et al. 2026 (arXiv:2601.02556); "
             "throughput, sky and noise notional.",
             fontsize=7.5, color="0.5", ha="left", va="bottom")
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / ("lazuli_lensed_agn." + suffix), dpi=200)
    return fig


if __name__ == "__main__":
    main(*sys.argv[1:])
