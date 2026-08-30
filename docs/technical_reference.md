# Technical reference

The full solver-stage, facade, and benchmark reference. This is the deep documentation moved out
of the repository README so the landing page stays short; nothing here is required
for everyday use — start with {doc}`usage` and the {doc}`api/index` instead.

## Public facade

The stable user-facing entry points are:

```python
from shaarp import (
    run_si_numeric,
    run_ml_numeric,
    run_fresnel_sweep,
    run_maker_fringes,
    run_sample_rotation,
    run_si_full_analytical,
    run_ml_partial_analytical,
    compare_with_mathematica,
    export_result,
)
```

Each returns a `SHAARPResult` with:

- `numeric`: arrays/scalars intended for plotting or export
- `stages`: intermediate records when available
- `conventions`: field, boundary, intensity, phase-matching, and Maker policy metadata
- `validation`: whether the path is reduced, staged, partially Mathematica-validated, or still pending

`compare_with_mathematica(reference, result, tolerance)` compares selected
Mathematica reference values against a `SHAARPResult`. Reference keys can be
plain numeric names such as `intensity`, or dotted intermediate and metadata
values such as `numeric.boundary_residual_norm`,
`stages.normal_incidence_2omega_branch_policy`, `validation.status`, and
`conventions.field_convention`. This is intended for value-by-value benchmark
files that check intermediate values and policy labels, not only final plots.

`run_si_numeric` keeps the legacy reduced model as its default. To use the
stage-validated SHAARP.si reflected-SHG compatibility path:

```python
from shaarp import Polarimetry, run_si_numeric
from shaarp.interactive import default_interactive_material

result = run_si_numeric(
    default_interactive_material(),
    {
        "workflow": "shaarp_si_compat",
        "polarimetry": Polarimetry(theta_deg=30.0),
        "incident_polarization": "s",
        "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
    },
)
```

This path is validated against the exported 36-case SHAARP.si reflected `ER2w`
stage reference when the SHAARP-compatible branch policy is selected. It is
still marked as stage validation, not a full notebook-execution claim.

Analytical facades are present as explicit scoped workflows:
`run_si_full_analytical(..., {"workflow": "isotropic_symbolic_scaffold"})` and
`run_ml_partial_analytical(..., {"workflow": "pnl_symbolic_scaffold"})` return
SymPy analytical results. Both facades also accept `{"workflow": "polarimetry"}`,
which returns the **d-extraction polarimetry** (reflected SHG as a closed form in the
input polarization φ with d_ijk as the symbolic coefficient): the SI facade returns the
full-analytical `E_s/E_p^2ω(φ, …)` plus the analyzed intensity `I(φ, ψ)`, and the ML
facade the partial-analytical film polarimetry in (φ, d, h). These workflows wrap the
validated `solve_si_shg_full_analytical_symbolic` / `solve_single_film_shg_symbolic_polarimetry`
(test asserts the facade output is byte-identical to the solver); the scaffold workflows
remain the defaults. For the **isotropic** class (e.g. GaAs), the SI
full-analytical chain is now FORM-VALIDATED at every stage against the
**published** SHAARP supplementary (npj Comput. Mater. s41524-022-00930-4,
Suppl. Note 5, eq 9-32 for GaAs(111)): the omega-boundary fundamental fields
(eq 29-32), the nonlinear polarization P_NL (eq 20-28, all nine ee/oo/eo
components), the inhomogeneous `solveInhom` coefficients C_Li (eq 11-19), and the
final 2omega-boundary signal E_p/E_s^2w (eq 9-10) — verified in BOTH the real and
the COMPLEX (absorbing) domain, with the physically-correct decaying branch (see
`tests/test_published_gaas111_pnl_supplementary.py` and
`tests/test_published_gaas111_stages_supplementary.py`).

**Closed-form SHG across crystal classes and the SHG coefficient d_ijk.** Beyond the
isotropic published case, the SI full-analytical reflected SHG is now assembled in closed
form for **uniaxial and biaxial (principal-axes-aligned) crystals** too
(`solve_{uniaxial,biaxial}_single_interface_shg_symbolic`, uniaxial delegating to biaxial),
each validated end-to-end vs the numeric `solve_single_interface_shg` to ~1e-14..1e-15. The
**SHG coefficient tensor `d_ijk` (Voigt `d_il`) is symbolic**: these SI functions accept a
symbolic `d_voigt_lab` directly (the chain is fully symbolic and computePNL is linear in d),
giving the SHG as a closed form in `d_ijk` (and angles) — see `tests/test_si_shg_symbolic_d.py`.
The ML **partial-analytical** counterpart is closed-form in **both** the film thickness `h`
and `d_ijk` (`shaarp.multilayer_shg_symbolic.solve_single_film_shg_symbolic_thickness_and_d`,
the Maker fringe), validated to ~1e-18. Honest boundary: the **general rotated** (non
principal-aligned) biaxial closed form is a full Booker quartic in `k_z` (impractical as a
radical); the numeric path covers all orientations. When comparing any symbolic SHG to the
numeric solver, pass **identical `mu, eps0`** to both (`MU0*EPS0 = 1/c^2` rescales the
inhomogeneous field per polarization row otherwise). All boundary solvers (SI/ML,
numeric/symbolic) share one tangential-continuity convention via `shaarp.waves.tangential_eh`
(`[Ex, Ey, Hx, Hy]`, matching SHAARP `solveFresnelN`).

**SHG polarimetry — the d-extraction expression (input-polarization scan).** The analytical
output SHAARP is built for is the **SHG polarimetry**: the reflected SHG as a closed form in the
**input polarization angle φ** with the `d_ijk` tensor as the symbolic coefficient — the curve an
experimentalist fits to a measured polarimetry scan to *extract d*. (Computing SHG only at fixed
s/p, φ=0/90°, misses the curve — those are often the polarimetry *zeros*.)
`shaarp.symbolic.solve_si_shg_full_analytical_symbolic` is the **full** analytical form — symbolic
in φ, analyzer ψ, sample azimuth, incidence θ, refractive indices, and `d_ijk` — validated against
the **published GaAs(111)** closed form composed from eq 9–32 over a φ sweep (ratio 1.000 to the
published 5-decimal rounding floor, both channels, including the eo cross term), the s/p endpoints
vs the numeric `solve_single_interface_shg`, and the C3v **3-fold** sample-azimuth symmetry — see
`tests/test_si_shg_polarimetry_symbolic.py`.
`shaarp.multilayer_shg_symbolic.solve_single_film_shg_symbolic_polarimetry` is the ML **partial**
analytical form — symbolic in φ, `d_ijk`, and thickness `h` only, with eigenmodes/angles/indices
NUMERIC (that split is what "partial" means) — validated vs the numeric arbitrary-Jones
`solve_multilayer_shg_from_tensors_jones` over a (φ, h) sweep to ~1e-10 — see
`tests/test_ml_shg_polarimetry_symbolic.py`. The omega eigenmode selection is
symbolic-index-safe (the in-plane mode is chosen structurally by `E_y = 0`, not by a magnitude
comparison that cannot order symbolic indices), so the SI full form is genuinely symbolic in the
refractive indices. For the **anisotropic** classes (uniaxial/biaxial), where the published GaAs
closed form does not apply, the SI polarimetry curve — including the eo **cross term**, not just
the s/p endpoints — is validated against the numeric `solve_single_interface_shg(incident_jones=…)`
path (an arbitrary input-polarization SI solve, added for exactly this purpose and shown to reduce
to the discrete s/p path at the endpoints) over a φ sweep to ~1e-12..1e-16, real and complex
absorbing — see the anisotropic tests in `tests/test_si_shg_polarimetry_symbolic.py`.

Both SHG **channels** are available: `SymbolicSIFullPolarimetry` exposes the reflected
`reflected_s`/`reflected_p` (and `intensity`) and the total **transmitted** field
`transmitted_field` / `transmitted_intensity` `E_t(φ)` — the Maker-fringe observable —
validated vs the numeric total transmitted field for uniaxial and biaxial over a φ sweep
(the per-mode split uses a different basis than the numeric fast/slow modes, but the total
field is basis-independent and agrees to ~1e-12).

**d-extraction (the closed form fulfilling its purpose).** Because the reflected field is linear
in the d-components, the closed form `E(φ) = Σ_k d_k · B_k(φ)` (basis `B_k = ∂E/∂d_k`) is exactly
the expression an experimentalist inverts to recover d from a polarimetry scan.
`tests/test_si_polarimetry_d_extraction.py` demonstrates this end-to-end: a known six-component
d-tensor is recovered to ~1e-12 from synthetic measurements (generated by the independent numeric
solver) via a linear least-squares fit of the closed form. It also pins the **identifiability
physics**: a single geometry is rank-deficient (one scan cannot determine the whole tensor, rank
4 of 6), while a multi-angle × multi-azimuth scan restores full rank — which is why such scans are
run. (Read the linear basis with `∂/∂d_k`, not `.coeff`, on the unexpanded symbolic expression.)
The same test also recovers d from **intensity-only** measurements `I(φ)=|E_p(φ)|²` (what a
polarimeter actually records): since `I` is linear in the Gram matrix `D_kl=d_k d_l`, a linear
solve for `D` followed by a rank-1 factorization returns d to ~1e-12 (up to the physical overall
sign), and the recovered `D` being exactly rank-1 is itself a data-consistency check.
This is packaged as a reusable feature: `shaarp.polarimetry_extraction.extract_si_d_voigt(...,
method="field"|"intensity")` takes the measurement geometries, the φ samples, and a
`measure(theta, azimuth, phi) -> (E_s, E_p)` forward output, and returns a `DExtractionResult`
with the recovered `d_voigt` plus an honest report (`rank`, `identifiable`, `condition_number`,
`residual`, `sign_ambiguous`) — see `tests/test_polarimetry_d_extraction_api.py`. It also accepts
`observable="transmitted"` (the **Maker-fringe** geometry): `measure` then returns the total
transmitted SHG field 3-vector, giving 3 observables per scan point, so the tensor is often
identifiable with fewer geometries (3 vs 6 in the validated test). The same module
provides `extract_ml_film_d_voigt(...)` for **thin films** (SHAARP.ml): it uses the ML
partial-analytical film polarimetry as the model and recovers a film's d-tensor from a
θ × sample-azimuth × φ scan (validated to recover a known six-component film tensor to ~1e-9 in
field mode, up to sign in intensity mode). It also accepts `observable="transmitted"` (the
standard thin-film **Maker-fringe** geometry, using the substrate-transmitted SHG amplitudes).
So d-extraction now spans both channels (reflected / transmitted) for both SI interfaces and ML
films, and all three optical classes with real crystals (isotropic GaAs, uniaxial LiNbO₃,
biaxial **KTP** at the 2022 paper's Table 1 values). For near-degenerate biaxial crystals
(n_x≈n_y, e.g. KTP) the symbolic basis is ill-conditioned, so `extract_si_d_voigt` offers
`basis="numeric"` — the design column is the numeric unit-d SHG response, exact for any crystal.
The numeric basis also rotates **both** ε and d under a sample azimuth (the symbolic basis rotates
only d), so it additionally recovers a full biaxial tensor from a **rotated-biaxial azimuth scan** —
the case the symbolic closed form (a full Booker quartic) cannot reach.

Runnable, plotted examples: `examples/gaas111_shg_polarimetry.py` (GaAs(111) closed-form
polarimetry with the C3v 3-fold sample-azimuth signature), `examples/d_extraction_demo.py`
(simulate a Maker-geometry scan of a known crystal → recover its d-tensor to ~1e-9), and
`examples/maker_fringes_dense.py` (Full/JK/HH Maker fringes at the 0.1° angular-sampling rigor of
npj Comput. Mater. 10, 64 (2024), with a 0.02° zoom resolving the 2ω multiple-reflection fine
fringes).

**Benchmark monitoring.** `notebooks/SHAARP_py_MASTER_BENCHMARK.ipynb` (built by
`build_master_benchmark_notebook.py`) is the single dashboard for all benchmark activity — it
embeds the dense figures, reads every live-Mathematica comparison-summary artifact, counts the
gated test suite, and presents one master agreement table. Attribution kept straight:
**2022 (♯SHAARP.si)** = reflected-SHG **polarimetry** vs literature; **2024 (♯SHAARP.ml)** =
**Maker fringes** (JK/HH/FMR) + polarimetry at 0.1°.

## Jupyter session

The notebook-first interactive entry point is `notebooks/SHAARP_py_interactive_session.ipynb`.
It uses `shaarp.make_interactive_session()` and requires the optional interactive dependencies
(`pip install ".[interactive]"`).

For the faithful two-in-one replica of the original SHAARP.si + SHAARP.ml GUIs
(tab navigation, constrained d-tensor entry, Miller orientation, assumptions,
presets, copyable closed-form expressions, 2D/3D schematics), use:

```python
import shaarp
gui = shaarp.make_shaarp_gui()
gui
```

The legacy session controls expose ML numeric, GUI-shaped Fresnel sweep, Maker
fringes, sample rotation, polarimetry angles, inhomogeneous solution policy,
Maker transmitted-wave policy, and a validation-reference toggle. The default system
uses an active nonlinear film so ML/Maker/SampleRotate modes exercise nonzero
nonlinear sources. The output panel reports validation status and physical
conventions for each run.

## Validated building blocks

Mathematica-verified staged building blocks include:

- `shaarp.anisotropic.modes_for_direction`
- `shaarp.anisotropic.solve_snell_modes`
- `shaarp.anisotropic.identify_uniaxial_modes`
- `shaarp.anisotropic.track_mode_branches`
- `shaarp.boundary.solve_fresnel_boundary`
- `shaarp.boundary.solve_boundary_continuity`
- `shaarp.linear.solve_linear_interface`
- `shaarp.linear.solve_linear_interface_sweep`
- `shaarp.nonlinear.compute_pnl_voigt`
- `shaarp.nonlinear.single_interface_sources`
- `shaarp.nonlinear.solve_inhomogeneous_field`
- `shaarp.nonlinear.solve_single_interface_inhomogeneous_fields`
- `shaarp.multilayer_basis.build_multilayer_omega_basis`
- `shaarp.multilayer_basis.build_multilayer_2omega_basis`
- `shaarp.multilayer_boundary.solve_multilayer_boundary`
- `shaarp.multilayer_shg_boundary.solve_multilayer_shg_boundary`
- `shaarp.multilayer_shg_boundary.solve_multilayer_shg_from_fundamental`
- `shaarp.multilayer_shg_boundary.solve_multilayer_shg_from_tensors`
- `shaarp.multilayer_shg_boundary.solve_multilayer_shg_from_tensors_jones`
- `shaarp.multilayer_shg_boundary.solve_multilayer_shg_from_system`
- `shaarp.multilayer_shg_boundary.solve_multilayer_shg_from_system_jones`
- `shaarp.multilayer_shg_boundary.solve_multilayer_shg_from_system_polarimetry`
- `shaarp.multilayer_shg_boundary.solve_multilayer_shg_system_sweep`
- `shaarp.multilayer_shg_boundary.solve_multilayer_shg_polarimetry_sweep`
- `shaarp.shg.solve_single_interface_shg`
- `shaarp.symbolic.d_voigt_symbolic`
- `shaarp.symbolic.rotate_d_voigt_symbolic`
- `shaarp.symbolic.compute_pnl_voigt_symbolic`
- `shaarp.symbolic.partial_pnl_expressions`
- `shaarp.symbolic.solve_fresnel_boundary_symbolic`
- `shaarp.symbolic.solve_boundary_continuity_symbolic`
- `shaarp.multilayer_shg_symbolic.solve_single_film_shg_symbolic_thickness`
- `shaarp.multilayer_shg_symbolic.solve_single_film_shg_symbolic_thickness_and_d`
- `shaarp.multilayer_shg_symbolic.solve_single_film_shg_symbolic_polarimetry`
- `shaarp.symbolic.solve_si_shg_full_analytical_symbolic`
- `shaarp.symbolic.fresnel_quadratic_eigenmodes_symbolic`
- `shaarp.symbolic.solve_biaxial_linear_interface_symbolic`
- `shaarp.symbolic.solve_biaxial_single_interface_shg_symbolic`
- `shaarp.symbolic.solve_uniaxial_linear_interface_symbolic`
- `shaarp.symbolic.solve_uniaxial_single_interface_shg_symbolic`
- `shaarp.symbolic.solve_isotropic_linear_interface_symbolic`
- `shaarp.symbolic.solve_inhomogeneous_field_symbolic`
- `shaarp.symbolic.single_interface_sources_symbolic`
- `shaarp.symbolic.solve_single_interface_inhomogeneous_fields_symbolic`
- `shaarp.symbolic.solve_single_interface_shg_boundary_symbolic`
- `shaarp.symbolic.solve_single_interface_shg_known_waves_symbolic`
- `shaarp.symbolic.solve_isotropic_single_interface_shg_symbolic`

This ports the numerical core of Mathematica `solveSnell`: solve `L E = eta epsilon E`, identify `eta = 1/n^2`, and solve Snell's law separately for fast and slow modes. The current Python path covers transmitted and reflected/backward roots for real/lossless and complex absorbing anisotropic dielectric tensors. `identify_uniaxial_modes` adds a guarded ordinary/extraordinary classifier for uniaxial media so nonlinear SHAARP source labels do not silently assume `fast == extraordinary` and `slow == ordinary`. For biaxial or otherwise non-uniaxial sweeps, `track_mode_branches` keeps two eigenmode branches continuous by electric-field overlap and marks ambiguous assignments, without forcing ordinary/extraordinary labels. This is verified by equation residuals, Python regression fingerprints, and exported Mathematica `solveSnell` reference values (20 cases; `test_mathematica_solve_snell_reference.py`).

`CrystalOrientation.from_miller_surface(structure, (h, k, l))` builds a real orientation from a general Miller plane by using the reciprocal lattice vectors from `CrystalStructure`. `CrystalOrientation.from_cubic_miller_surface((h, k, l))` is the narrower cubic/orthonormal-axis convenience path. The benchmark suite includes both an all-nonzero cubic `[1, 2, 3]` orientation and a non-cubic reciprocal-lattice `[1, 2, 3]` case whose normal is not the naive Cartesian triple. Complex spatial rotation matrices are intentionally rejected; absorption/complex response belongs in the dielectric and nonlinear tensors, not in the real-space orientation matrix.

`CrystalOrientation.with_lab_azimuth_deg(...)` and
`with_sample_azimuth_deg(system, ...)` apply a physical sample/crystal azimuth
rotation about lab `L3`. This is intentionally separate from
`Polarimetry.phi_deg`, which remains the incident polarizer angle.

`solve_fresnel_boundary` ports the tangential `E`/`H` continuity structure of Mathematica `solveFresnel` for four unknown reflected/transmitted wave amplitudes. It is verified against closed-form isotropic Fresnel coefficients for s and p polarization.

`solve_boundary_continuity` is the lower-level interface solver used by `solve_fresnel_boundary`. It supports known waves on both sides of an interface, which is needed for SHG boundary stages with known inhomogeneous source fields. It is verified by residual tests that include a known bottom-side source wave.

`solve_multilayer_boundary` ports the numeric boundary-condition structure of Mathematica `solveFresnelN`: top interface, internal interfaces with `exp(I k_z thickness)` propagation, and bottom interface. It solves the linear amplitude system for reflected, internal forward/backward, and substrate waves. It also accepts known layer/substrate waves — the hook for inhomogeneous nonlinear source fields in the SHAARP.ml workflow, which is validated end-to-end against live Mathematica (see {doc}`validation`).

`solve_multilayer_shg_boundary` wraps the multilayer boundary kernel for the
2omega nonlinear stage once inhomogeneous layer source waves are known. It
solves for reflected, homogeneous in-layer, and substrate SH amplitudes with the
known source waves included in the layer boundary residuals.

`shaarp.multilayer_basis` builds homogeneous omega and 2omega basis waves for
the multilayer boundary kernel from lab-frame dielectric tensors. Layer
bases are ordered `[F1, F2, B1, B2]`, matching the SHAARP.ml forward/backward
source construction. `solve_multilayer_shg_from_fundamental` solves the omega
multilayer boundary problem, builds the ten layer source terms, solves their
inhomogeneous fields, and runs the 2omega boundary solve. The higher-level
`solve_multilayer_shg_from_tensors` additionally generates the omega and 2omega
bases from supplied lab-frame tensors for an isotropic top medium. The
multilayer SHG workflow is validated value-by-value against live Mathematica
SHAARP.ml Maker fringes (single film through nine layers, plus point-group
classes) to ~10⁻¹³–10⁻¹⁴ (see {doc}`validation`).

The lower-level `solve_multilayer_shg_from_fundamental` path accepts supplied
incident/reflected top waves, so anisotropic top media can be tested when the
incident mode and reflected basis are built explicitly. The high-level
`solve_multilayer_shg_from_system` helper still requires an isotropic top
medium because automatic selection of the incident anisotropic top mode and its
polarimetry convention has not been Mathematica-validated.

`solve_multilayer_shg_from_system` connects that staged workflow to
`MultilayerSystem`, rotating each configured material's dielectric and SHG
tensors into the lab frame and using layer thicknesses in microns with
`omega = 2*pi / wavelength_um`.

`solve_multilayer_shg_system_sweep` runs the same system-level workflow across
a one-dimensional list of incidence angles, returning per-angle solver results
plus fundamental and SH boundary residual norms.

The Jones/polarimetry variants support arbitrary complex incident amplitudes in
explicit `(s, p)` order. `solve_multilayer_shg_from_system_polarimetry` follows
the Python convention `s = sin(phi) exp(i ellipticity)`, `p = cos(phi)`, and is
validated value-by-value against live Mathematica SHAARP.ml `SampleRotate` to
~2×10⁻¹⁴ across 20 polarimetry settings, under the SHAARP.ml-matching
forward-only inhomogeneous-source policy (see `test_polarimetry_reference_comparison.py`).
Note: the convenience sweep wrapper `solve_multilayer_shg_polarimetry_sweep`
defaults to `inhomogeneous_source_policy="all"`, a different (non-SHAARP.ml)
convention — pass `"forward_only"` to match the original package.

Reflected 2omega output helpers are available for analyzer-style postprocessing:
`reflected_2omega_jones_sp`, `analyze_reflected_2omega`,
`analyzer_jones_from_polarimetry`, and
`analyze_reflected_2omega_polarimetry`. They extract reflected `(s, p)`
components and compute analyzer-projected complex amplitude/intensity.

`solve_multilayer_shg_polarimetry_sweep` runs the staged system workflow across
broadcast-compatible `theta_deg`, `phi_deg`, `psi_deg`, and `ellipticity_deg`
arrays, returning analyzer amplitudes, intensities, and residual norms.

`solve_multilayer_shg_sample_azimuth_sweep` sweeps physical sample/crystal
azimuth about lab `L3` while keeping it separate from the polarizer angle
`phi_deg`. It broadcasts the requested sample azimuth values with the existing
polarimetry arrays and returns analyzer amplitudes, intensities, and residual
norms.

`solve_linear_interface` connects the Snell-mode and boundary solvers for an isotropic incident medium and anisotropic transmitted medium. It is verified to reduce to closed-form isotropic Fresnel coefficients and has anisotropic smoke tests.

`solve_linear_interface_sweep` wraps the linear interface solver for angle sweeps and attaches continuity-tracked transmitted branch waves — for biaxial/non-uniaxial sweeps where per-angle fast/slow sorting can swap identities.

`compute_pnl_voigt` ports Mathematica `computePNL`, including the mixed-mode factor used for `eo` source polarization. `single_interface_sources` generates the `ee`, `oo`, and `eo` source wavevectors/polarizations.

`solve_inhomogeneous_field` ports the coefficient-level linear equation from Mathematica `solveInhom`:

```text
Curl[Curl[E_inh]] = omega^2 mu0 eps0 (epsilon. E_inh + PNL)
```

for one nonlinear source phase. `solve_single_interface_inhomogeneous_fields` applies this to the `ee`, `oo`, and `eo` sources. Each returned inhomogeneous field includes `operator_condition`, `ill_conditioned`, and `solution_method` diagnostics so near phase-matched singular systems are visible instead of being treated as ordinary residual-pass cases. The default `solution_policy="solve"` preserves the direct-solve behavior used for SHAARP compatibility checks; exact singular operators fall back to an explicit minimum-norm least-squares diagnostic. For phase-matched studies where a deterministic truncated-SVD convention is preferred, use `solution_policy="minimum_norm_if_ill_conditioned"` or `solution_policy="minimum_norm"`. These policies make the arbitrary near-null component explicit; they are not a substitute for any finite-thickness or resonant phase-matching treatment Mathematica may use.

`solve_single_interface_shg` ports the numeric structure of Mathematica `f1NL` for one interface: solve the linear omega boundary problem, construct homogeneous reflected/transmitted 2omega waves, solve the three inhomogeneous 2omega source fields, then enforce the 2omega tangential E/H boundary condition with the inhomogeneous waves included on the transmitted side. For detectably uniaxial omega tensors, it classifies the fundamental modes and builds `ee`, `oo`, and `eo` from extraordinary/ordinary identity rather than from fast/slow sort order.

`shaarp.symbolic` provides the analytical-expression building blocks. It ports
the SHAARP.si partial-analytical `doldExp` point-group SHG tensor patterns, the
`dnewExp` crystal-to-lab Voigt rotation loop, and the symbolic
`computePNL`/`P2e`, `P2o`, `Peo` nonlinear source expressions (optional SymPy
dependency). Symbolic tangential `E`/`H` boundary continuity, isotropic
linear-interface solving, symbolic `solveInhom`, the symbolic single-interface
source connector, and the symbolic final 2omega boundary assembly are each
tested against their numerical counterparts with complex tensor values. The
complete closed-form output is the FULL analytical polarimetry
(`run_si_full_analytical`, `workflow="polarimetry"`), validated against the
published GaAs(111) form; a comparison against a Mathematica analytical PRINTOUT
is not possible because SHAARP.si has no symbolic anisotropic eigenmode
closed-form to print. The test suite intentionally includes a slower dense
complex symbolic known-waves case rather than skipping it for speed.

`solve_isotropic_single_interface_shg_symbolic` provides a scoped automatic
symbolic SHG path for isotropic incident/transmitted media: symbolic omega
Fresnel interface, phase-matched isotropic 2omega s/p bases, then the symbolic
source, `solveInhom`, and final boundary pipeline.

## Numeric benchmarks

Regression benchmarks are stored in:

```text
benchmarks/numeric_benchmarks_v1.json
benchmarks/solver_stage_benchmarks_v1.json
benchmarks/complex_inhomogeneous_benchmarks_v1.json
benchmarks/reflected_snell_benchmarks_v1.json
benchmarks/biaxial_branch_benchmarks_v1.json
benchmarks/multilayer_system_benchmarks_v1.json
benchmarks/multilayer_polarimetry_benchmarks_v1.json
benchmarks/comprehensive_coverage_matrix_v1.json
benchmarks/coverage_report_v1.json
benchmarks/mathematica_reference/reference_manifest_v1.json
benchmarks/mathematica_reference/reference_template_v1.json
```

`comprehensive_coverage_matrix_v1.json` is the coverage contract for the
Python regression/residual suite (normal and high-angle incidence, s/p
polarization, identity/random/cubic/non-cubic orientations,
isotropic/uniaxial/biaxial/non-diagonal/complex dielectric tensors, zero and
dense complex nonlinear tensors, branch tracking, symbolic cases, phase
matching, reflected roots, and Mathematica reference-comparison
infrastructure). `coverage_report_v1.json`
(`benchmarks/generate_coverage_report.py`) summarizes the actual counts. The
`mathematica_reference/` manifest + template
(`benchmarks/generate_mathematica_reference_manifest.py` / `_template.py`)
define the export request list for Mathematica reference data — intermediate
values such as wavevectors, fields, nonlinear source polarizations, and
boundary coefficients, not only final intensities. The benchmark families
sweep incidence angles, orientations, s/p polarization, complex anisotropic
(including non-diagonal) tensors, and all ten forward/backward nonlinear layer
source terms; each case stores full numeric arrays under `outputs` so
Mathematica exports are compared value-by-value
(`benchmarks/compare_mathematica_reference.py`).

Current verification policy:

- Include complex-valued refractive indices, dielectric tensors, SHG
  coefficients, fields, and wavevectors wherever the solver stage supports them.
- Include non-diagonal dielectric permittivity where the solver stage accepts
  full tensors.
- Include simple limits such as normal incidence.
- Include extreme checks such as the same dielectric permittivity at omega and
  2omega. Exact phase matching can make the inhomogeneous particular-field
  equation ill-conditioned or resonant, so those cases are detected and
  documented (`solution_method="least_squares_singular"`,
  `solution_policy="minimum_norm_if_ill_conditioned"`) rather than silently
  counted as ordinary residual passes.
- Keep the slower dense complex symbolic case active; do not skip it only for
  speed.

Command-line Wolfram evaluation is only needed to REGENERATE reference data; every shipped test
runs against the frozen JSON exports in `benchmarks/` and needs no Mathematica installation.

## Legacy reduced CLI demos

These commands use the legacy reduced plotting paths. Use the public facade
functions above when you need validation metadata and SHAARP-compatible staged
workflows.

```bash
shaarp --mode si --theta 45 --save si_demo.png
shaarp --mode ml --theta 45 --save ml_demo.png
```

## Notes

Angles are in degrees at the public API boundary, matching the Mathematica notebooks. Internally,
vector operations use radians and NumPy arrays.
