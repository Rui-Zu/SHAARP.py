# SHAARP.py — Validation against the Mathematica SHAARP package

This document is the canonical record of how the Python re-implementation
(**SHAARP.py**) is validated against the original Wolfram / Mathematica
**SHAARP** package (`SHAARP.si` single-interface, `SHAARP.ml` multilayer). Every
agreement claim here is backed by an automated, tolerance-gated test that
compares Python output **value-by-value** against output exported from a live
Mathematica kernel — not against a Python re-derivation.

> **Bottom line:** SHAARP.py reproduces the original Mathematica SHAARP package to
> machine precision (typical max error 10⁻¹³ – 10⁻¹⁷) across solver stages,
> geometries, the Full / JK / HH assumption modes, single-interface reflected SHG,
> crystal orientation, and the symbolic engine. The full automated suite (216
> modules / 1,269 tests) passes, including an offscreen pass over every selectable
> case × functionality on both GUI tabs.

## Additions since the evidence table below

- **SHAARP.ml docs quartz+Au Maker case** (Z-cut quartz 121.2 um + Au 13.9 nm,
  lambda 0.8 um, p-in/p-out, 20-30 deg at 0.1 deg): live-Mathematica MFList
  reproduced to max abs err 3.6e-9 (rel 5.2e-10), zero spurious nulls. Gated by
  `tests/test_quartz_au_docs_reference.py`; case in `benchmarks/quartz_au_docs_case.py`.
- **Degenerate-eigenmode slot fixes** (6 instances found via the everything-must-
  be-continuous lens): Maker per-channel slot swap (verified vs live SHAARP.ml,
  3.4e-11 in the swap window), GUI Fresnel tp/ts, SI compat omega+2omega branch
  collision for isotropic crystals — the isotropic fix verified against classical
  Fresnel (exact) and the published-GaAs(111)-validated closed form (scale 1.0,
  agreement ~1.5e-7; s-incidence selection-rule zero pinned as exact physics).
- **Natural-units convention enforced** for the multilayer path (mu=eps0=1 defaults
  + dispersion-contract warning); suite logs warning-free.
- **Merged SHAARP.si + .ml GUI** (`shaarp.make_shaarp_gui()`) with all controls on
  the validated backends; 49 GUI tests including continuity gates.
- Suite total: **216 modules, 1,269 tests, 0 failures**, measured by running every module in its
  own subprocess from a clean checkout copied outside the development tree — so this is the figure
  a user reproduces after cloning, not one that holds only on the author's machine.
  Earlier editions of this file quoted **1,250 / 1,257 / 1,258** tests. Those were UNDERCOUNTS
  rather than a larger suite: per-module totals were summed with a pattern requiring the plural
  `"Ran N tests"`, while a module with a single test prints `"Ran 1 test"` and therefore counted as
  zero. Only the reported number was affected; no module's pass/fail status ever was.

## Methodology

1. **Live Mathematica export.** The original `SHAARP.ml` / `SHAARP.si` notebooks
   are executed in Wolfram (`wolframscript`); their reference outputs are
   persisted byte-faithfully as JSON under `benchmarks/mathematica_reference/`.
   Each Wolfram exporter is derived from a proven-parsing canonical script by
   exact surgical string-replacement (verified by unified diff) so it cannot
   introduce a silent syntax error.
2. **Value-by-value comparison.** Python output is compared element-wise to those
   references at **explicit numerical tolerances** (atol 10⁻⁹ – 10⁻¹²).
3. **Un-fakeable gated tests.** Each comparison is guarded by a test that *errors*
   (rather than spuriously passing) when a feature or reference is absent.
4. **Honest diagnostics.** Phase-matching / singular points, where the
   inhomogeneous solve is intrinsically near-singular, are retained as **labelled
   diagnostics** and excluded from the agreement count — they reflect shared
   ill-conditioning, not a port error.

## Evidence — agreement across the validated surface

| Validation category | Cases | Tolerance | Max \|error\| vs. Mathematica | Gating test |
|---|---|---|---|---|
| Maker fringes — single film (ext1) | 12 | 10⁻⁹ | 2.1×10⁻¹⁴ | `test_maker_fringes_reference_comparison.py` |
| Maker fringes — multilayer, 2–9 layers (ml1–ml8) | 30 | 10⁻⁹ | 1.0×10⁻¹³ | `test_maker_fringes_*_reference*.py` |
| Maker fringes — point-group classes (pg1/2/3, pgml) | 18 | 10⁻⁹ | 2.7×10⁻¹⁴ | `test_maker_fringes_pointgroup_*` |
| Sample-rotation azimuth (ext1 + ml4) | 16 | 10⁻⁹ | 4.5×10⁻¹⁵ | `test_sample_rotation_ext_reference_comparison.py`, `test_sample_rotation_multilayer_ml4_reference_comparison.py` |
| SHG assumption modes — Full / JK / HH | 12 | 5×10⁻⁴ | 2.0×10⁻¹⁵ | `test_jkhh_*_agreement.py` |
| Multilayer polarimetry solve (forward-only) | 20 | 10⁻¹⁰ | 2.0×10⁻¹⁴ | `test_polarimetry_reference_comparison.py` |
| Fresnel coefficients (multilayer `listFresnel`) | 3 | 10⁻¹⁰ | ~1×10⁻¹⁵ | `test_fresnel_sweep_python_comparison.py`, `test_figuredata.py` |
| Crystal orientation (`hklConvert`/QC/QP, eps-lab, d-lab) | 10 | 10⁻¹¹ | ≤10⁻¹¹ (asserted) | `test_mathematica_orientation_reference.py` |
| Single-interface solver stage values | — | 10⁻¹² | 1.2×10⁻¹⁶ | `test_shaarp_si_stage_*` |
| Single-interface reflected — point groups (sipg1/2) | 14 | 2×10⁻⁶ | match | `test_shaarp_si_*` |
| Single-interface reflected — grazing incidence, θ = 75–89.5° (sigraze) | 6 | 2×10⁻⁶ | 3.6×10⁻¹⁵ | `test_shaarp_si_compat_signal_agreement_sigraze.py` |
| Multilayer Maker — grazing incidence, 9-layer, θ = 75–83° (mlgraze) | 4 | 10⁻⁹ | 1.0×10⁻¹⁴ | `test_maker_fringes_multilayer_mlgraze_reference_comparison.py` |
| Multilayer Maker — **non-air incident medium**, n₀ = 1.33/1.34, 9-layer (mlamb) | 2 | 10⁻⁹ | 5.6×10⁻¹⁴ | `test_maker_fringes_multilayer_mlamb_reference_comparison.py` |
| Symbolic nonlinear polarization Pᴺᴸ (live Wolfram) | 36 | 10⁻⁹ | 2.8×10⁻¹⁷ | `test_*symbolic*pnl*` |
| Symbolic inhomogeneous wave — `solveInhom` (live Wolfram) | 20 | 10⁻⁹ | 9.8×10⁻¹⁷ | `test_*solve_inhom*` |

All non-singular cases pass at the stated tolerance. Source values:
`benchmarks/mathematica_reference/*_comparison_summary*.json` and the per-family
reference JSON files.

### Selected like-for-like values (Python vs. Mathematica are identical to all printed digits)

**Maker fringes**, single-film LiNbO₃, transmitted-SHG \|MF_para\| at θ = 0/20/40/60°:

| Mode | Source | 0° | 20° | 40° | 60° |
|---|---|---|---|---|---|
| Full | Mathematica | 0.052139 | 0.054415 | 0.036081 | 0.008876 |
| Full | SHAARP.py | 0.052139 | 0.054415 | 0.036081 | 0.008876 |
| JK | both | 0.054221 | 0.056948 | 0.038364 | 0.009295 |
| HH | both | 0.047766 | 0.048828 | 0.032616 | 0.008250 |

(max \|error\| ≈ 2×10⁻¹⁵ for all three modes.)

## Scope — what is, and isn't, claimed

- **Phase-matching singularities are diagnostics, not failures.** A small number of
  phase-matched points (7 in Maker, 1 in sample-rotation) are labelled and excluded
  from the agreement count.
- **Transmitted-2ω wave selection is a documented convention.** Agreement is
  asserted under the SHAARP-selected transmitted-wave policy (matching the original
  package); a physically-summed alternative is provided but explicitly not labelled
  the Mathematica default.
- **Polarimetry source-policy convention.** The polarimetry *solve* matches
  SHAARP.ml `SampleRotate` to ~2×10⁻¹⁴ under the **forward-only** inhomogeneous-source
  policy (the SHAARP.ml convention). The convenience wrapper
  `solve_multilayer_shg_polarimetry_sweep` defaults to `inhomogeneous_source_policy="all"`,
  which is a different (non-SHAARP.ml) convention — pass `"forward_only"` to match
  the original package.
- **The numerical + symbolic core is the validated surface.** Full interactive-GUI
  parity (live widgets, 2D/3D render) is display-bound and not claimed headlessly.
  (An older edition of this note claimed a closed-form symbolic multilayer thickness
  `h` was "absent from the source itself". That was retracted for the README once
  symbolic-h shipped, and is now doubly wrong: the original DOES carry it per layer
  — `SHAARP.ml.nb:5135-5146` stores a distinct symbol h1, h2, … per interior layer and
  `setup.nb:11011-11017` consumes the thicknesses as a list — and the port reproduces
  it for N layers, mixed symbolic/numeric, gated by
  `tests/test_multilayer_shg_symbolic_nlayer.py`. The SHAARP.si single-interface path
  genuinely has no thickness to symbolize, being semi-infinite.)

## Reproducing the validation

```bash
# from the repo root, with the project's Python interpreter
python -m unittest discover -s tests -p 'test_*.py'
```

Live-Wolfram references are committed under `benchmarks/mathematica_reference/`, so
the comparison tests run without a Wolfram kernel.

### Regenerating the Mathematica references

Regenerating a reference is optional — the committed JSON is what the suite compares against — and
requires a licensed Wolfram kernel plus the original Mathematica packages, which are separate
repositories and are not vendored here. Each exporter resolves its paths from the environment:

| Variable | Points at | Needed by |
|---|---|---|
| `SHAARP_REF_DIR` | this checkout's `benchmarks/mathematica_reference` | optional; defaults to the directory holding the script |
| `SHAARP_ML_DIR` | a checkout of [SHAARP.ml](https://github.com/bzw133/SHAARP.ml) (provides `setup.nb`) | the multilayer exporters |
| `SHAARP_SI_DIR` | a checkout of [SHAARP](https://github.com/Rui-Zu/SHAARP) (provides `SHAARP_V1.03`) | four single-interface exporters |

```bash
SHAARP_ML_DIR=/path/to/SHAARP.ml \
  wolframscript -script benchmarks/mathematica_reference/export_polarimetry_reference.wl
```

An exporter that needs one of the external packages and cannot find it stops immediately with a
message naming the variable to set, rather than failing part-way through.

## Additions (v1.0.0 evaluation ladder)

The release now carries a four-layer evaluation ladder on top of the evidence table:

- **T1 input-sensitivity matrix** (`tests/test_input_sensitivity_matrix.py`): every output must
  respond to every input that should matter (or declare its invariance). Includes the metal-film
  full-range Fresnel fence and the GaAs(111) all-angle finiteness + absolute-agreement fences.
- **T2 fidelity matrix**: row-by-row feature parity vs the original notebooks/docs — all rows
  check.
- **T3 published-figure replication** (`scripts/replicate_paper_figures.py` →
  `build/paper_replication/`): both papers' validation figures reproduced through the GUI compute
  path — SHAARP.si 2022 Fig. 4(b–e) for GaAs (111) at 800 nm (including the quantitative
  ε_R-approximation claim: d-ratio 0.818 vs the paper's 0.809) and SHAARP.ml 2024 Figs. 3(b,c) /
  4(b,d) for X-/Z-cut quartz (HH-vs-JK fine-fringe distinction; Au-coating FMR amplification — the
  Fig. 4(d) FMR reproduces the author's own closed-form model to correlation 0.9993 including the central
  bump, with its HH curve embedded from his published HH model).
  This layer immediately caught two latent defects the whole gate suite had passed (surprise): float-noise-broken isotropic degeneracy (NaN GaAs polarimetry) and a silent real-cast of
  complex ε/d in the curve/expression paths — both fixed, verified vs the numeric reference to
  5×10⁻¹⁴, and fenced.
- **T4 d-extraction noise characterization**: the Monte-Carlo study (`benchmarks/dextraction_noise_benchmark.py`, fenced): the phase-resolved
  field method degrades gracefully (median error ≈ noise level); the phase-less intensity method
  amplifies catastrophically at realistic conditioning — its noisy-data output is an initial guess.

All five tutorial notebooks execute cleanly against the current API
(`jupyter nbconvert --execute`).

## Honesty discipline

The port is **released as v1.0.0** (`RELEASE GATE: PASS`, with the suite total recorded above, plus the
312-cell GUI matrix sweep `CLEAN`). The old planning-era "~91%" figure is retired: the numerical
core and end-to-end pipelines are fully validated as tabulated above, and the remaining
NOT-verified surface is stated in the tables above, claim by claim, instead of being summarized as
a percentage. No number here is inflated for breadth — every agreement traces to a live Wolfram
export and an un-fakeable, tolerance-gated test.
