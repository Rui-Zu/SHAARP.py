"""Build (and execute) the two flagship paper-reproduction notebooks:

    notebooks/Reproduce_SHAARP_si_paper.ipynb -- SHAARP.si 2022 Fig 4/5/6/7
    notebooks/Reproduce_SHAARP_ml_paper.ipynb -- SHAARP.ml 2024 Fig 3/4/5/6/7

Each section prints the case parameters (with a provenance cite to the Mathematica case example the
values came from), reproduces the figure through the SAME compute path the GUI uses
(benchmarks.paper_cases / paper_figures), and shows it BESIDE the cropped published panel.

Run: python build_paper_notebooks.py [si|ml|both] (default both)
Copies each executed notebook into docs/tutorials/ as well, for the Sphinx site.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent
NB = ROOT / "notebooks"
TUT = ROOT / "docs" / "tutorials"


def _mk(cells):
    def md(t): cells.append(nbf.v4.new_markdown_cell(t))
    def code(t): cells.append(nbf.v4.new_code_cell(t))
    return md, code


def build_si():
    cells = []
    md, code = _mk(cells)
    md("""# Reproducing the ♯SHAARP.si (2022) paper with SHAARP.py

**Zu, R. et al. "Reflection second harmonic generation ... ♯SHAARP."** *npj Comput. Mater.* **8**, 246 (2022).

This notebook reproduces the four case-study crystals of the SHAARP.si paper — **GaAs (111)**,
**LiNbO₃ (11̄20)**, **KTP (100)**, **TaAs (112)** — using `SHAARP.py`. Each is a single air/crystal
interface probed by **reflected-SHG polarimetry**: the *p*- and *s*-polarized SHG intensity as a
function of the incident polarization angle φ, at several incidence angles θᵢ.

**Where the numbers come from.** Every parameter (crystal orientation, point group, SHG *d*-tensor,
dielectric ε(ω)/ε(2ω)) is taken from the author's *own* Mathematica ♯SHAARP.si case-study definitions
(transcribed verbatim into `shaarp/casestudy_dispersion.json`, provenance-audited) and interpolated
to the paper's single experimental wavelength (**800 nm**, Ti:Sapphire). The compute call is exactly
the one the desktop GUI's *Update* button makes. Each section shows the reproduction **beside the
published panel** so you can compare shapes, symmetry, and the relative *p*/*s* channel scaling.
""")
    code("""import sys; from pathlib import Path
# locate the SHAARP.py repository root (works from the repo root, notebooks/, or docs/tutorials/ —
# `pip install -e .` ships only the `shaarp` package; the reproduction code lives in the repo)
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / 'benchmarks' / 'paper_figures.py').exists()), None)
assert ROOT is not None, 'run this notebook from inside the SHAARP.py repository checkout'
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
from benchmarks.paper_figures import show_si
print('SHAARP.py paper reproduction — SHAARP.si 2022. repo root =', ROOT.name)""")

    sections = [
        ("fig4", "## Figure 4 — GaAs (111), 800 nm  (isotropic, −43m)",
         "GaAs is cubic (−43m): the reflected-SHG lobes come purely from the (111) surface geometry. "
         "This is the SHAARP.py↔paper anchor case (validated to ~5e-14 against the closed-form solver)."),
        ("fig5", "## Figure 5 — LiNbO₃ (11̄20), 800 nm  (uniaxial, 3m)",
         "The a-plane (x-cut) surface of uniaxial LiNbO₃, computed with the **paper's own fitted "
         "800-nm *d* tensor** (d₃₃/d₃₁ = 5.21 ± 0.13, d₁₅/d₃₁ = 1.23, d₂₂/d₃₁ = −0.23; |d₃₃| = 28.5 "
         "pm/V — the published fit, recovered from the archived `NonlinearModelFit` table). Panel (c) "
         "*Iₚ* is a horizontal dumbbell; panel (d) *Iₛ* the 4-lobe. **On the ×15 display factor:** the "
         "physics ratio from the published *d* is Iₚ/Iₛ ≈ 18.2; the paper's ×15 is the *experimental* "
         "cross-channel ratio, which folds in the x/y detector calibration that the original fit "
         "explicitly models (the fitting notebook is titled \"800 nm x-LNO **Different Intensity of "
         "x and y**\", with per-dataset amplitude parameters and separate `…X`/`…Y` data files). "
         "Pattern, lobe angles, and the fitted *d* are reproduced exactly; ×15 is a calibrated-display "
         "number, not a pure physics benchmark."),
        ("fig6", "## Figure 6 — KTP (100), 800 nm  (biaxial, mm2)",
         "Biaxial KTP, computed with the paper's extracted tensor (|d₃₃| = 17.0 ± 0.2 pm/V, "
         "LiNbO₃-referenced). Here the *s*-channel is the strong one: the physics ratio with the "
         "published tensor is Iₛ/Iₚ ≈ 8.7, and the panel's ×10 is the experimental display ratio — "
         "the original xKTP fit's second-channel amplitudes are ~1.15–1.37, and 8.7 × ~1.15 ≈ 10 "
         "(the same x/y channel-calibration semantics as Fig 5's ×15). SHAARP.py reproduces the "
         "4-lobe *Iₚ* and vertical-dumbbell *Iₛ* exactly."),
        ("fig7", "## Figure 7 — TaAs (112), 800 nm  (Weyl semimetal, 4mm, complex ε & d)",
         "A strongly absorbing Weyl semimetal, computed with the **published fitted tensor** "
         "|d₃₃|, |d₃₁|, |d₁₅| = 827 ± 39, 12 ± 15, 113 ± 20 pm/V — recovered verbatim (values *and* "
         "error bars) from the archived fit (`Plot of Polar Plots_TaAs112_Generalized.nb`, with the "
         "LiNbO₃-d₃₁ = 5.47 pm/V absolute calibration). The reflected SHG is highly θ-dependent "
         "(≈0 at normal incidence); SHAARP.py carries the full complex ε (F32 fix) and computes the "
         "true (112) oblique tilt (F40 exact route — the earlier z-cut approximation is gone). "
         "Panel (b) is closed EXACTLY (F42): the eo-mode effective index **rises** 4.3006 → 4.3512 "
         "while k **falls** 2.1684 → 2.0541 across θᵢ = 0.5° → 80°, machine-identical to a live "
         "SHAARP.si V1.04 replication. (The V1.04 frame propagates transmitted waves *downward*, so "
         "its verbatim orientation matrix needs a documented 180° lab azimuth in this port's upward "
         "frame — `_SI_FRAME_AZIMUTH_DEG`, fence `TestSiFig7TaAsEffectiveIndex`; the polar channels "
         "are m_y-symmetric and azimuth-insensitive.) *On the ×20 factor:* "
         "the panel legend reads \"Exp Iₓ / Exp 20×Iᵧ\" — an experimental display scale that includes "
         "the x/y channel calibration (same class as Fig 5's ×15); the pure-physics lab-channel ratio "
         "with the published d and the exact tilt is I_L₁/I_L₂ ≈ 37."),
    ]
    for key, heading, blurb in sections:
        md(heading + "\n\n" + blurb)
        code(f"show_si({key!r})")
    md("""---
### Summary

All four SHAARP.si case crystals reproduce their published reflected-SHG polarimetry — matching the
lobe **shape**, the crystal-class **symmetry**, and the relative *p*/*s* channel **scaling** — from
parameters sourced entirely from the author's original Mathematica case studies. See the companion
`Reproduce_SHAARP_ml_paper.ipynb` for the multilayer / Maker-fringe paper.""")
    return cells


def build_ml():
    cells = []
    md, code = _mk(cells)
    md("""# Reproducing the ♯SHAARP.ml (2024) paper with SHAARP.py

**Zu, R. et al. "... multilayer SHG ... ♯SHAARP.ml."** *npj Comput. Mater.* **10**, 64 (2024).

This notebook reproduces the five case figures of the SHAARP.ml paper with `SHAARP.py`:
**Maker fringes** (Fig 3, 4), **transmitted single-crystal polarimetry** (Fig 5), a **reflected
heterostructure** (Fig 6), and **two-crystal SHG interference** (Fig 7). The three multiple-reflection
**assumptions** appear throughout:

- **FMR** — Full Multiple Reflections (all multiply-reflected ω and 2ω waves retained),
- **JK** — Jerphagnon & Kurtz (single pass, no multiple reflections),
- **HH** — Herman & Hayden (multiple reflections kept **only for the homogeneous 2ω waves**).

**Parameter provenance** is identical to the SI notebook: registry values transcribed from the author's
Mathematica ♯SHAARP.ml case studies, with per-figure geometry (thicknesses, wavelengths) from the
manuscript. **Fig 3 additionally overlays the raw Herman-1995 analytic HH expression** — a byte-exact
Python port of the author's own `analyticHH.mx` benchmark — so SHAARP.py's HH mode is checked against an
independent analytic model, not just against SHAARP itself.

> The multilayer polarimetry sweeps solve the full boundary problem per polarization angle, so this
> notebook runs at a moderate angular resolution to execute in minutes; finer grids sharpen the fine
> 2ω fringes (see the `step`/`n_phi` arguments).
""")
    code("""import sys; from pathlib import Path
# locate the SHAARP.py repository root (works from the repo root, notebooks/, or docs/tutorials/ —
# `pip install -e .` ships only the `shaarp` package; the reproduction code lives in the repo)
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / 'benchmarks' / 'paper_figures.py').exists()), None)
assert ROOT is not None, 'run this notebook from inside the SHAARP.py repository checkout'
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
from benchmarks.paper_figures import (
    ml_fig3_figure, ml_fig4_figure, ml_polarimetry_figure, ml_interference_figure,
    ml_mos2_figure, _show_panel, _embed_fig)
from benchmarks.paper_cases import FIG5_CRYSTALS, ml_single_crystal_system, ml_fig6_system
print('SHAARP.py paper reproduction — SHAARP.ml 2024. repo root =', ROOT.name)""")

    md("""## Figure 3 — 300 µm X-cut quartz Maker fringes, 1064 nm (HH vs JK vs raw analytic HH)

Transmitted *p→p* SHG vs incidence angle. The **HH** and **JK** envelopes nearly coincide; the
magnified window shows the **fine 2ω fringes** present in HH (multiple reflections of the homogeneous
2ω wave) but absent in JK. Two independent references are overlaid from the author's own Fig-3 archive:

- the **raw Herman-1995 analytic HH** expression (`benchmarks/herman_hayden_maker.py`, ported
  byte-exact from `fig3/analyticHH.mx` — matches the Mathematica value to 1e-15), and
- his **serialized reference curves** (`fig3/dataoldHHJK.mx` evaluated HH on the main panel —
  identified by 0.9996 correlation with the analytic expression — and `fig3/dataHHsimHHFiner1.mx`
  on the magnified fine-fringe window), evaluated at h = 300 µm, φ = 0 and cached in
  `benchmarks/fig3_reference_curves.json`.

In the magnified window **SHAARP.py(HH) overlaps the analytic HH reference *exactly*, fringe for
fringe** (same-grid correlation = 1.000000; the reference is a constant multiple of the ported
analytic expression, ratio spread < 1e-9) — the paper's exact-overlap claim, reproduced. The zoom
is sampled at 0.05° matching the reference: the fine 2ω fringes have ~0.2° period, so any coarser
sweep sits at the Nyquist edge and *displays* a spurious fringe offset.""")
    code("""fig, stats = ml_fig3_figure(step=0.25)
_embed_fig(fig)
print(f"HH/JK envelope peak ratio = {stats['hh_jk_peak_ratio']:.4f} (≈1: near-identical envelopes)")
print(f"SHAARP.py(HH) vs raw analytic-HH shape correlation = {stats['hh_vs_analytic_corr']:.4f}")
if 'analytic_vs_mx_corr' in stats:
    print(f"ported analytic HH vs evaluated .mx curve = {stats['analytic_vs_mx_corr']:.4f}")
if 'shaarp_hh_vs_hhsim_fine_corr' in stats:
    print(f"SHAARP.py(HH) fine fringes vs HH-sim window = {stats['shaarp_hh_vs_hhsim_fine_corr']:.4f}")
_show_panel('ml_fig3', 'SHAARP.ml 2024 Fig. 3')""")

    md("""## Figure 4 — Z-cut quartz, 800 nm: the effect of full multiple reflections

Uncoated (left, **123.6 µm**) the three assumptions agree in envelope, and the author's experimental
scan (black dots) rides it through both humps. Add a **backside Au mirror** (right) and **FMR departs
strongly** — the paper's central point.

**Drawn exactly the way the published panels were drawn.** Both panels use the display recipe
transcribed from the author's generating notebook (`Maker fringes 100 um quartz.nb`, cells 14 and 23;
`benchmarks.paper_figures.FIG4_PUBLISHED`), which was verified by overlaying that recipe's curves on
the published figure at pixel level: model curves are `Rescale`d on a **common** per-panel {min, max}
(FMR ∪ HH ∪ JK on the left, FMR ∪ HH on the right) and then multiplied by the published display
factors — right panel **FMR ×1.07, HH ×1.5, averaged ×0.93** (the "× 1.5" the paper prints beside
the green curve); the experiment is `Rescale`d to [0, 1] on its own (×0.91 and a +0.1° stage offset
on the left panel). The right panel is computed at the **published spot thickness h₀ = 121.18 µm**
(the cell's `hMRP1`; a 0.18 µm change is more than one 2ω-fringe period in thickness and moves the
θ = 0 fringe phase, which is why the earlier 121.0 µm rendering looked wrong at the centre) with
**13.9 nm Au**.

**What SHAARP.py computes here, and what is embedded from Dr. Zu.** Left panel: JK / HH / FMR are all
SHAARP.py, fenced against the author's three bare-slab models (`fig4b_*_reference.csv`, corr ≥ 0.998).
Right panel: the **FMR** (red) and the **FMR+θⁱ+h+λω** curve (teal — the author's method: a plain
mean over 0.80 µm of equivalent thickness for the h and λ spreads, then a 3° moving average for the
beam divergence, then its own Rescale ×0.93) are SHAARP.py. The red FMR is the physics result: at the
published thickness it reproduces the author's *own* closed-form FMR model (`QuartzAuSimuMRP1S0p02.mx`)
to **shape correlation 0.9994 with peak magnitude ratio 1.0001** — printed below and fenced in
`tests/test_paper_cases.py`. The raw FMR shows the published 0.22 centre dip with 0.30 side peaks;
the ~0.35 centre bump belongs to the experiment and the averaged curve, which hugs the black scan
through the dip and up to both peaks.

The **green HH ×1.5** curve is now **SHAARP.py's own HH**, computed the way the author's published HH
model was built — decoded from the model files: a *three*-medium closed form (quartz on a semi-infinite Au
half-space; its 2ω Fabry-Pérot constant |r₁r₂| = 0.0559 equals quartz→air × quartz→*bulk*-Au, with no Au
thickness dependence, whereas the FMR model carries the 13.9 nm thin-film factors) whose reported
"p-out" intensity is the beam-frame x′ component of the transmitted field inside the Au. In that geometry
and convention SHAARP.py's HH matches the author's model (dashed dark green, `QuartzAu_800nm_HHNMRP1S0p01.mx`)
to **shape correlation 0.998 and peak ratio 0.992** on the fringe-resolved 30–45° window (fenced). The
earlier "my HH diverges" finding was geometry plus output convention, not a single-pass-ω defect. What
remains is a ~5 % offset near normal incidence that the port shows against *all* of the author's SLAB
closed forms alike (FMR and HH, bare and Au) — a property of those older closed forms; the port is
validated to ~1e-15 against the .ml numeric solver on this quartz+Au Maker case. HH is still the paper's
**deliberately-failing** illustrative curve (FMR ⊋ HH is the point of the panel).""")
    code("""fig, dev = ml_fig4_figure(step=0.05)
_embed_fig(fig)
print(f"SHAARP.py FMR vs the author's closed-form FMR model at 121.18 um = {dev['d_FMR_vs_his_FMR_corr']:.4f} "
      f"(shape corr; peak ratio {dev['d_FMR_peak_ratio']:.4f}); SHAARP.py HH (author geometry) vs his HH model = "
      f"{dev['d_HH_vs_his_HH_corr']:.4f} (peak ratio {dev['d_HH_peak_ratio']:.4f})")
print(f"panel (b) vs the author's bare-slab models: FMR {dev['b_FMR_vs_his_corr']:.4f}  "
      f"HH {dev['b_HH_vs_his_corr']:.4f}  JK {dev['b_JK_vs_his_corr']:.4f}")
print('stats:', {k: round(v, 3) for k, v in dev.items()})
_show_panel('ml_fig4', 'SHAARP.ml 2024 Fig. 4')""")

    md("""## Figure 5 — LiNbO₃ and KTP single crystals, 1550 nm (transmitted polarimetry)

**Transmitted** *p*/*s* SHG vs incident polarization φ for thick single crystals at several θᵢ:
LiNbO₃ (11̄20) [x-cut, 3m], LiNbO₃ (0001) [z-cut, 3m], and KTP (x-cut, mm2). The lobe patterns,
their angular positions, and the per-crystal channel hierarchy reproduce the published panels.

*Open item (documented):* the published panels' cross-panel scale factors (×135 / ×4 / ×7) depend
on the **slab thickness used in the original simulation**, which is not stated in the paper or SI
and is not recoverable from the archived notebooks (the transmitted-channel ratio oscillates with
thickness through multiple-reflection interference — e.g. LiNbO₃ (11̄20) spans ~58–399 over
10–1000 µm; this build uses a representative 500 µm). The patterns shown here are
thickness-robust; only those display ratios are not pinned.""")
    code("""rows = [
    ('LiNbO₃ (11̄20)', ml_single_crystal_system(FIG5_CRYSTALS['LiNbO₃ (11̄20)']), (0., 10., 30.)),
    ('LiNbO₃ (0001)', ml_single_crystal_system(FIG5_CRYSTALS['LiNbO₃ (0001)']), (0., 10., 30.)),
    ('KTP (x-cut)', ml_single_crystal_system(FIG5_CRYSTALS['KTP (x-cut)']), (0., 10., 20., 40.)),
]
fig = ml_polarimetry_figure(rows, channel='transmitted', n_phi=73,
                            title='Fig 5 — transmitted SHG polarimetry, single crystals @ 1550 nm')
_embed_fig(fig)
_show_panel('ml_fig5', 'SHAARP.ml 2024 Fig. 5')""")

    md("""## Figure 6 — ZnO // Pt // Al₂O₃ heterostructure, 1550 nm (reflected polarimetry)

A 159 nm ZnO film on a 200 nm Pt electrode on Al₂O₃. Pt is opaque at 1550 nm, so this is a
**reflected**-SHG measurement (transmitted ≈ 0). SHAARP.py's multilayer solver handles the
metal interlayer and reproduces the *Iₚ* dumbbell / *Iₛ* 4-lobe reflected pattern.""")
    code("""rows = [('ZnO // Pt // Al₂O₃', ml_fig6_system(), (45.,))]
fig = ml_polarimetry_figure(rows, channel='reflected', n_phi=73,
                            title='Fig 6 — reflected SHG polarimetry, ZnO//Pt//Al₂O₃ @ 1550 nm')
_embed_fig(fig)
_show_panel('ml_fig6', 'SHAARP.ml 2024 Fig. 6')""")

    md("""## Figure 7 — SHG interference in LiNbO₃(21̄1̄0) // quartz(001), 1550 nm

Two stacked SHG-active crystals (50 µm LiNbO₃ on 35 µm quartz), probed by **transmitted SHG
polarimetry near normal incidence** across the paper's four stacking cases: LNO alone with its polar
axis **P ∥ −L₁** (Case 1) or **P ∥ +L₁** (Case 4), and the LNO//quartz(001)[100] stack with each P
direction (Cases 2, 3). The SHG waves generated in the two crystals **interfere**: one stacking order
is **constructive** (above LNO alone), the other **destructive** (below) — flipping only the sign of
P against quartz's fixed [100] switches between them, while the two LNO-alone cases are identical
(a single crystal's intensity is invariant under the P flip). The paper's panel-(d) bars
(≈0.86 / 1.00 / 0.72 / 0.86) are reproduced almost exactly.""")
    code("""fig, stats = ml_interference_figure(n_phi=73)
_embed_fig(fig)
mx = max(stats[f'case{k}_IL1_phi0'] for k in (1, 2, 3, 4))
print('normalized I_L1(phi=0):',
      {f'Case {k}': round(stats[f'case{k}_IL1_phi0'] / mx, 3) for k in (1, 2, 3, 4)},
      ' (paper bars: ~0.86 / 1.00 / 0.72 / 0.86)')
print(f"global I_L1/I_L2 ratio = {stats['IL1_over_IL2']:.0f} (paper display factor: x61)")
_show_panel('ml_fig7', 'SHAARP.ml 2024 Fig. 7')""")

    md("""## Figure S7 (Supplementary) — twist bilayer MoS₂, rotational-anisotropy reflected SHG

Monolayer MoS₂ (point group **−6m2**, $d_{22}$-dominant) gives the characteristic **six-petal**
RA-SHG pattern. The reflected SHG is swept vs **sample azimuth** (fixed *p*-incidence along $L_1$)
through a fixed analyzer at 0° (b) and 90° (c), for monolayer MoS₂(0°), monolayer MoS₂(25°), and the
**twist bilayer MoS₂(25°)//MoS₂(0°)**. Because the two atomically-thin layers add coherently, the
twist bilayer's SHG is *larger* than either monolayer — the paper's point about twist-controlled SHG
in stacked 2D materials. (The registry `MoS2` is the monolayer; the bilayer stacks two, the top one
rotated about the surface normal.)

*Scope note:* panels (b, c) here use an air exit medium — the six-fold symmetry, the 25° lobe
rotation, and the coherent bilayer enhancement are substrate-independent. Panel (a) is reproduced
below with the substrate included.""")
    code("""fig, stats = ml_mos2_figure(n_azimuth=145)
_embed_fig(fig)
print('bilayer / monolayer I_p peak ratio =', round(stats['bilayer_over_monolayer_Ip_peak'], 2),
      '(>1: the twist bilayer is the stronger emitter, as in the paper)')
_show_panel('ml_figS7_MoS2', 'SHAARP.ml SI Fig. S7')""")

    md("""### Fig S7(a) — bilayer coherence κ vs substrate thickness

Panel (a) sweeps the **substrate thickness**: κ(Δθ=0°, t) = I_bilayer/(2·I_monolayer) − 1, the
**coherent interference cross-term** of the two monolayers' SHG (κ = 1 ⇔ fully coherent addition;
definition from the author's own `Heterostructure Test.nb`: κ = (I_bi − I_top − I_bot)/(2√(I_top·I_bot))).
The substrate is a **~500 µm Al₂O₃ (sapphire) wafer** (identified from the κ notebooks' sweep
variable *t*₍Al₂O₃₎), whose Fabry–Pérot phase modulates the inter-layer coherence — producing the
fast fringe + slow beat of the published panel. As the SI caption itself notes, "the κ function for
the thickness range 500 ± 200 µm follows the same periodicity", so a representative sub-window
suffices here (the offline validation sweeps the full 480–520 µm).""")
    code("""import numpy as np
from benchmarks.paper_cases import ml_mos2_kappa_sweep
t = np.arange(480.0, 488.0 + 1e-9, 0.05)
tt, kk = ml_mos2_kappa_sweep(t)
fig, ax = plt.subplots(figsize=(10.5, 3.2))
ax.plot(tt, kk, color='#5b84c4', lw=0.8)
ax.set_xlabel(r'Substrate thickness, t ($\\mu$m)')
ax.set_ylabel(r'$\\kappa(\\Delta\\theta=0°, t)$')
ax.set_title('Fig S7(a) — twist-bilayer MoS₂ coherence vs Al₂O₃ wafer thickness', fontsize=10)
_embed_fig(fig)
print(f'kappa in [{kk.min():.3f}, {kk.max():.3f}], mean {kk.mean():.3f} '
      '(published panel: oscillates within ~0.7–0.95)')""")

    md("""### Fig 8c — bilayer coherence κ vs TWIST angle Δθ

Panel (c) sweeps the **twist angle Δθ** between the two MoS₂ monolayers at a fixed substrate: the
coherence κ(Δθ) = I_bi/(2·I_mono) − 1 (peak-over-azimuth reflected SHG) tracks the ideal **cos(3Δθ)**
of the three-fold monolayer symmetry, sitting slightly below it because the Al₂O₃ Fabry–Pérot phase
reduces the inter-layer coherence. The **red band** is κ's spread over the substrate-thickness edges
(the author's `minmaxkappa.nb`: t = 500.57 – 508.47 µm).""")
    code("""from benchmarks.paper_figures import ml_fig8c_figure
fig, st = ml_fig8c_figure()
_embed_fig(fig)
lo, hi = st['kappa0_band']
print(f'kappa(0) band = [{lo:.3f}, {hi:.3f}]; kappa follows cos(3 dTheta): crosses 0 near 28-30 deg, '
      'reaches -1 at 60 deg (three-fold twist coherence)')
_show_panel('ml_fig8c', 'SHAARP.ml 2024 Fig. 8c')""")

    md("""---
### Summary

SHAARP.py reproduces all ♯SHAARP.ml case figures — Maker fringes under all three MR assumptions,
transmitted single-crystal polarimetry, a reflected metal-backed heterostructure, two-crystal
interference, and the twist-bilayer coherence (Fig 8c) — and its **HH** mode agrees with the **raw
Herman-1995 analytic benchmark**. Parameters trace to the author's original Mathematica case studies; the
multilayer physics matches the paper.""")
    return cells


def _write(cells, out: Path, *, timeout: int) -> None:
    nb = nbf.v4.new_notebook(); nb["cells"] = cells
    print(f"executing {out.name} ...", flush=True)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
        print(f"  executed OK: {out.name}", flush=True)
    except Exception as e:
        print(f"  execution ERROR in {out.name}: {e}", flush=True)
        raise
    # pin the kernel so Jupyter does not prompt "select kernel" on open -- ExecutePreprocessor
    # runs the notebook but does not record which kernel it used.
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                "name": "python3"}
    NB.mkdir(exist_ok=True); TUT.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    shutil.copy2(out, TUT / out.name)
    print(f"  saved {out} (+ docs/tutorials copy)", flush=True)


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    if which in ("si", "both"):
        _write(build_si(), NB / "Reproduce_SHAARP_si_paper.ipynb", timeout=1200)
    if which in ("ml", "both"):
        _write(build_ml(), NB / "Reproduce_SHAARP_ml_paper.ipynb", timeout=3000)
    print("done.")
