# How SHAARP.py is tested

This page is the evidence behind the one-line statement elsewhere in these docs that "the solvers
are checked against the published equations and the reference output of the original packages".
It says what is compared against what, at which tolerance, and which test guards each comparison.

Three kinds of reference are used:

- the equations published with the two ♯SHAARP papers and their supplementary notes,
- numerical output exported from the original Mathematica packages (`SHAARP.si` for the single
  interface, `SHAARP.ml` for multilayers) at fixed reference points, and
- the papers' own figures, reproduced through the same compute path the desktop app uses.

In short: SHAARP.py agrees with the reference output of the original packages to machine precision
(typical maximum error 10⁻¹³ – 10⁻¹⁷) across solver stages, geometries, the Full / JK / HH
assumption modes, single-interface reflected SHG, crystal orientation and the symbolic engine. The
full automated suite (221 modules / 1,320 tests) passes, including an offscreen pass over every
selectable case and functionality on both GUI tabs.

## Method

1. **Reference export.** The original `SHAARP.ml` / `SHAARP.si` notebooks are run in Wolfram
   (`wolframscript`) and their outputs are stored as JSON under
   `benchmarks/mathematica_reference/`. Each exporter is derived from one canonical script by exact
   string replacement, checked by diff, so an exporter cannot introduce a silent syntax error.
2. **Element-wise comparison.** Python output is compared element by element with those references
   at explicit numerical tolerances (atol 10⁻⁹ – 10⁻¹²).
3. **Tests that cannot pass vacuously.** Each comparison is guarded by a test that errors, rather
   than passing, when a feature or a reference file is absent.
4. **Singular points are labelled, not hidden.** At phase-matching points the inhomogeneous solve
   is intrinsically near-singular in both implementations; these points are kept as labelled
   diagnostics and excluded from the agreement count.

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
| Quartz + Au Maker case from the docs (Z-cut quartz 121.2 µm + Au 13.9 nm, 0.8 µm, p-in/p-out, 20–30° at 0.1°) | 1 | 10⁻⁸ | 3.6×10⁻⁹ (rel 5.2×10⁻¹⁰, zero spurious nulls) | `test_quartz_au_docs_reference.py` |
| Degenerate-eigenmode slot handling (Maker per-channel swap window; single-interface ω/2ω branch collision for isotropic crystals) | 6 | 10⁻⁶ | 3.4×10⁻¹¹ (Maker); ~1.5×10⁻⁷ vs. classical Fresnel and the GaAs(111) closed form | `test_maker_fringes_*`, `test_shaarp_si_compat_*` |

All non-singular cases pass at the stated tolerance. Source values:
`benchmarks/mathematica_reference/*_comparison_summary*.json` and the per-family
reference JSON files.

Two further properties are enforced by the suite rather than compared against a reference: the
multilayer path runs in natural units (μ = ε₀ = 1 defaults, with a dispersion-contract warning) and
the suite logs warning-free; and the merged SHAARP.si + .ml GUI (`shaarp.make_shaarp_gui()`) drives
only the validated backends, with 49 GUI tests including continuity gates.

Suite total: **221 modules, 1,320 tests, 0 failures**, measured by running every module in its own
subprocess from a clean checkout copied outside the development tree, so this is the figure you
reproduce after cloning.

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
- **Symbolic film thickness.** The closed-form multilayer expressions carry a distinct symbolic
  thickness per interior layer, for N layers and mixed symbolic/numeric input, as the original
  package does; gated by `tests/test_multilayer_shg_symbolic_nlayer.py`. The single-interface path
  has no thickness to symbolize, being semi-infinite.

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

## Paper figures and further checks

Beyond the evidence table, four further checks are part of the suite:

- **Input sensitivity** (`tests/test_input_sensitivity_matrix.py`): every output must respond to
  every input that should matter, or declare its invariance. This includes the metal-film
  full-range Fresnel check and the GaAs(111) all-angle finiteness and absolute-agreement checks.
- **Control-by-control comparison with the original GUIs**: every control of the two original
  notebooks has a counterpart or a documented, deliberate difference.
- **Published figures** (`scripts/replicate_paper_figures.py`, output under
  `build/paper_replication/`): both papers' validation figures are reproduced through the app's own
  compute path. SHAARP.si 2022 Fig. 4(b–e), GaAs (111) at 800 nm, including the paper's
  quantitative claim about the real-ε approximation (d-ratio 0.818 against the paper's 0.809), and
  SHAARP.ml 2024 Figs. 3(b,c) and 4(b,d) for X- and Z-cut quartz, including the fine-fringe
  difference between the Herman–Hayden and Jerphagnon–Kurtz treatments and the amplification with an
  Au coating (the Fig. 4(d) full-multiple-reflection curve matches the authors' closed-form model to
  correlation 0.9993, central feature included). Adding this check found two defects the rest of
  the suite had not: a float-noise-broken isotropic degeneracy (NaN GaAs polarimetry) and a silent
  real-cast of complex ε and d in the curve and expression paths. Both are fixed, verified against
  the numeric reference to 5×10⁻¹⁴, and now guarded by tests.
- **d-extraction under noise** (`benchmarks/dextraction_noise_benchmark.py`, a seeded Monte-Carlo
  study, guarded by a test): the phase-resolved field method degrades gracefully, with median error
  about equal to the noise level; the phase-less intensity method amplifies noise strongly at
  realistic conditioning, so its output on noisy data is an initial guess, not an estimate.

All five tutorial notebooks execute cleanly against the current API
(`jupyter nbconvert --execute`).

## What is not covered

v1.0.0 was released with the release gate passing at the suite total recorded above, plus a
312-cell GUI matrix sweep. Rather than summarising coverage as a percentage, the tables above state
what is verified and at what tolerance, and the Scope section states what is not: display-bound GUI
behaviour, the phase-matching singular points that are labelled rather than compared, and the
convention choices that are documented rather than asserted as agreement.
