# Lazuli lensed-AGN slide

One 16:9 slide figure for a Schmidt Sciences talk: wide-field surveys (Rubin
LSST, Euclid, Roman) **discover** lensed AGN; the Lazuli Space Observatory
**characterizes** them — diffraction-limited WCC imaging plus one IFS pointing
that resolves every quasar image and the lens galaxy.

**Status:** the figure is final (`figures/lazuli_lensed_agn.{png,pdf}`) and the
Keynote deck is built (`figures/lazuli_lensed_agn.key`, layers stacked by
`keynote.applescript`) with the user's hand-added click builds: footer and
survey columns show from the start, then 8 clicks (Lazuli WCC → lens). Builds
are not scriptable, so rebuilding the deck loses them.

## Pipeline

```bash
python population.py           # SLSim lensed-AGN population incl. Lazuli bands -> cache/population.pkl (~10 min)
python population.py --compact # extra draw, separations 0.3-1.2", 4x area -> cache/population_compact.pkl (~8 min)
python render.py               # imaging for 4 observatories + hero IFS cube    -> cache/render.pkl (~20 min, CPU)
python render.py --cube-only   # redo only the hero IFS cube, keep imaging      (~6 min)
python figure.py               # figures/lazuli_lensed_agn.{png,pdf}
python figure.py --keynote     # figures/keynote/build_NN_*.png + preview_NN_*.png
osascript keynote.applescript "$PWD"   # Mac only: new figures/lazuli_lensed_agn.key from the layers
```

Rebuilding the `.key` creates a fresh deck, so the hand-added builds must be
redone afterwards.

`cache/` is committed (so `figure.py` runs on a fresh clone); `scratch/` is
gitignored. Rerunning `population.py`/`render.py` draws a new realization, so
the hero system changes. They need SLSim (the user's editable install) and the
HST COSMOS 23.5 catalog (zenodo.org/records/3242143); `render.py` picks the
NERSC path or `/Users/paras/Projects/datasets/COSMOS/` on the Mac, whichever
exists. On the Mac use `~/miniconda3/bin/python` (not `/opt/miniconda3`, whose
lenstronomy is too old); lenstronomy there is an editable install of
`/Users/paras/Projects/repos/other/lenstronomy` (pip wheels lack the Roman
`PSF_models/*.fits` that Roman's PIXEL PSF needs).

## Files

- `observatories/lazuli.py` – Lazuli WCC config for SLSim: 17 mas pixels,
  SDSS-shaped filters `LZg/LZr/LZi/LZz` at 0.5 peak throughput, polychromatic
  Airy PSF (3.06 m unobscured) + 4 mas jitter.
- `ifs/` – IFS cube simulator using only SLSim's public API (no SLSim changes):
  `instrument.py` (R(λ), 40 mas NF spaxels, throughput, zodi, H4RG noise),
  `spectra.py` (qsogen quasar split into continuum + lines; kcorrect-template
  fits for lens and host), `cube.py` (morphology × spectrum per component,
  λ-dependent PSF, static chromatic microlensing of the continuum only).
- `render.py` – system selection (hero = widest quad that fits the IFS narrow
  field, and the double, from the main population; middle row = `COMPACT_QUAD`
  from the compact population), COSMOS-host swap, imaging, cube.
- `figure.py` – layout. Every artist is assigned to a Keynote build step via
  `step("...")`; `STEPS` lists the order. New elements must be drawn after the
  right `step()` call or they land in the wrong layer.

## Decisions already made with the user (don't relitigate)

- Layout: 4-column montage (LSST, Euclid, Roman, Lazuli WCC) × 3 systems on the
  left; Lazuli IFS on the right. Hero (top row) quad: z_d = 0.89, z_s = 2.74.
- Middle row is a compact quad (#81 of the compact population: θ_E = 0.23″,
  widest pair 0.47″, closest pair 0.25″, z_d = 1.16, z_s = 2.64): Rubin sees a
  point, Euclid/Roman a blended clump, Lazuli four images — the point is
  Lazuli's resolution. That row is zoomed to 2.5″ (`ROW_FOV` in `figure.py`)
  with its own longer "1 arcsec" bar; hero (IFS field is 4.6″ wide) and
  double stay at 5″. Row labels give z_d, z_s; θ_E sits inside each Rubin panel
  (top left, "θ_E = 0.84 arcsec").
- Filter labels run blue → red, matching the response curves.
- Footer: two short lines (SLSim + COSMOS hosts + OM10; Lazuli refs and what
  is modelled). Bottom right: "Credits: Paras Sharma (Stony Brook)".
- Euclid uses SLSim's Q1-style RGB (`slsim.Util.euclid_rgb_util`, VIS luminance
  + NISP Y/J/H colour, native pixel scales), arcsinh stretch, scale 200.
- IFS slices: three plain continuum slices (0.60–0.70, 0.85–1.00,
  1.20–1.50 µm) avoiding quasar lines, coloured blue → grey → red, bluest at
  the bottom. No continuum-subtracted line slice. A–D/G labels live on the WCC
  image, not on the slices.
- Spectra: true observed f_λ, log axis, one panel per image + lens, binned to
  R ≈ 90, PSF aperture-corrected (quasars only). IFS exposure 3 h (12 × 900 s).
  Lens shown only where detected (> 2σ); bluer is shaded "not detected in 3 h".
- Microlensing stays on in the simulation but is not featured on this slide.
- Header says 𝒪(1000) lensed AGN (Oguri & Marshall 2010).

## Accuracy notes

- Lazuli numbers come from Roy et al. 2026 (arXiv:2601.02556). Not published,
  so modelled here: PSF, WCC filter curves, R(λ) shape between quoted points,
  throughput/QE curves, zodi spectrum, 12 e⁻ up-the-ramp read noise, 4 detector
  pixels per voxel, exposure times. The footer says so.
- R ≤ 500 cannot measure the lens velocity dispersion; in 3 h the lens redshift
  is not measured either — don't claim it.
- SLSim microlensing magnification maps are not reproducible from a seed, so
  `--cube-only` redraws a slightly different (mild) realization each run.

## Keynote plan (agreed build order)

Background black; place every `build_NN_*.png` full-slide at (0, 0), in order:

0. base (footer) — on slide from the start
1. LSST/Euclid/Roman columns + DISCOVER bracket
2. Lazuli WCC column + CHARACTERIZE bracket
3. IFS field box, "LAZULI IFS" title, orange arrow
4. slices one at a time, bottom → top (4a blue + λ arrow, 4b, 4c)
5. A–D labels on the WCC image + A–D spectra
6. caption "Same quasar lines in every image → lensing confirmed"
7. G label + lens spectrum

`preview_NN_*.png` are the cumulative frames on black, for checking.
`keynote.applescript` does the stacking (Basic Black theme, 1920×1080, Blank
layout). Then by hand: select layers 1–9 in order → Animate → Build In →
Appear (or Dissolve), On Click; check Build Order lists them 1 → 9.

## Working conventions for this user

- Never recursively traverse shared filesystems (`/global`, `/pscratch`, …);
  stay inside this directory. The session scratchpad may be unwritable — use
  `scratch/` (gitignored) for throwaway scripts.
- Verify every number that appears on the slide against the code/model before
  reporting; the user checks them.
- Keep slide text large and the message simple; explain physics in notes.
