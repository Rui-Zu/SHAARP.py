# Conventions

SHAARP.py keeps the conventions of the original ♯SHAARP packages and the published *npj Computational
Materials* equations (see {doc}`references`), except where this page says otherwise. There are two
departures, both under *Reflection / transmission* below: the transmittance is a power transmittance,
and SHG intensities carry the exit medium's index. The exact, machine-readable conventions attached to each
run are exposed via {py:class}`~shaarp.PhysicsConventions` (on the result), so a computation is always
self-describing.

## Coordinate frames

- **Crystal-physics frame** — the material's intrinsic axes; $\varepsilon$ and the $d$ tensor are
  entered here.
- **Lab frame** — L1, L2, L3 with **L3 along the surface normal** and the **plane of incidence = the
  L1–L3 plane**. The {py:class}`~shaarp.CrystalOrientation` rotates the crystal frame into the lab frame (z-cut = identity;
  Miller `(hkl)`+`[uvw]`; or explicit Z1/Z2/Z3 axes). See {doc}`guide/si_tab`.
- **Vertical propagation frame** — SHAARP.py (like the original ♯SHAARP.ml `solveSnell` it is
  validated against) propagates transmitted waves along **+L3**; the original ♯SHAARP.si numeric
  notebook propagates along −L3. The two frames differ by a z-mirror, observable only when an
  anisotropy axis is tilted *obliquely* in the plane of incidence. Orientations authored in the
  ♯SHAARP.si frame therefore carry a documented 180° lab-azimuth correction when built as case-study
  materials (`TaAs (112)` — see `_SI_FRAME_AZIMUTH_DEG` in `shaarp/casestudy_materials.py`), which
  reproduces the published si-2022 Fig 7(b) effective-index trend exactly.

## Angles

- **Incident angle $\theta_i$** is measured from the surface normal (0° = normal incidence).
- **Input polarization $\varphi$** parameterizes the incident Jones vector
  $E = E_0(\cos\varphi,\ \sin\varphi\,e^{i\Delta\delta},\ 0)$; $\varphi=0$ is $p$, $\varphi=90°$ is $s$.
- **Analyzer angle $\psi$** selects the detected polarization. A *rotating* analyzer returns the
  $p$ and $s$ channels $I_p$, $I_s$, referred to the plane of incidence. A *co-rotating* analyzer,
  $\psi = \varphi + \text{offset}$, returns the parallel and perpendicular channels, referred to the
  input polarization. The two pairs coincide only at $\varphi = 0$.

## Units, and degrees vs radians

Both units appear, at different layers of the API. The rule is simple: **input classes and the GUI
use degrees; the low-level solvers use radians.**

| Quantity | Unit | Where |
|---|---|---|
| Incidence / polarization / analyzer angle | **degrees** | GUI fields, {py:class}`~shaarp.Polarimetry`, `run_maker_fringes` angle grids |
| The same angles | **radians** | `solve_*` solvers (`incident_theta_rad`), `extract_si_d_voigt` (`geometries`, `phi_values`), symbolic `phi` |
| Wavelength $\lambda$ | **µm** | `MultilayerSystem(wavelength_um=…)`, GUI wavelength field |
| Layer thickness $h$ | **µm** | `Layer(thickness_um=…)`; `None` marks a half-space |
| Dielectric tensor $\varepsilon$ | dimensionless relative permittivity, complex | everywhere ($\varepsilon = n^2$ for an isotropic medium) |
| Lattice constants | Å (lengths), degrees (angles) | GUI *Crystal Structure* group |
| SHG coefficients $d_{ij}$ | pm/V | GUI *SHG Tensor* group, `d_voigt` |
| Intensities | arbitrary units | all outputs — relative shape is meaningful, absolute scale is not |

```{warning}
**$\mu$ and $\varepsilon_0$ are a convention, not a constant of nature, in this code.** The
low-level solvers default to SI values (`mu=MU0`, `eps0=EPS0`), while every Mathematica validation
and every shipped benchmark runs in **natural units, `mu=1, eps0=1`**. Two consequences:

- When comparing a symbolic result to a numeric one, pass **identical** `mu, eps0` to both —
  otherwise `MU0*EPS0 = 1/c²` rescales the inhomogeneous field per polarization row.
- Some stiff configurations (thick slabs with a metal coating, for instance) are numerically
  unstable at SI constants and stable at `mu=1, eps0=1`. Reproduce benchmarks in natural units.
```

## The SHG $d$ tensor (Voigt notation)

The second-order nonlinear response is the $3\times6$ Voigt matrix $d_{i\mu}$ ($i=1..3$, $\mu=1..6$),
with $P_i^{2\omega} = \varepsilon_0 \sum_\mu d_{i\mu}\,(E E)_\mu$ and the contracted index
$\mu \in \{xx, yy, zz, yz, xz, xy\}$. **The mixed terms carry a factor of two**, as usual in this
convention:

$$(E E)_\mu = \left(E_x^2,\ E_y^2,\ E_z^2,\ 2E_yE_z,\ 2E_xE_z,\ 2E_xE_y\right).$$

Check this before substituting $d$ values taken from another source. The crystal **point group**
fixes which components are
independent; the rest follow by symmetry. See {py:func}`shaarp.d_voigt_symbolic` and
{py:func}`shaarp.rotate_d_voigt_crystal_to_lab`. Centrosymmetric/isotropic groups have $d \equiv 0$
(SHG forbidden).

## Reflection / transmission

- Linear Fresnel coefficients are returned as power $R_p, R_s, T_p, T_s$; for a lossless stack they
  satisfy $R+T=1$ per polarization, and at normal incidence $p$ and $s$ are degenerate.
- $R = |r|^2$, and the transmittance carries the obliquity factor

  $$T = \frac{\mathrm{Re}(n_\text{exit}\cos\theta_\text{exit})}{\mathrm{Re}(n_\text{inc}\cos\theta_\text{inc})}\,|t|^2 .$$

  Refraction changes the beam's width, so the incident and transmitted beams do not share a
  cross-section even though they cross the same patch of interface. Without this factor, $R+T=1$
  holds only when the exit medium is index-matched to the incident one. Pass
  `transmittance="amplitude"` to {py:func}`shaarp.run_fresnel_sweep` for bare $|t|^2$, which is what
  ♯SHAARP.ml's `listFresnel` emits.
- SHG intensities carry the exit medium's index at $2\omega$, $I_{2\omega} \propto n_\text{exit}\,|E_{2\omega}|^2$,
  because a plane wave carries $I = \tfrac{1}{2}c\varepsilon_0 n |E|^2$. The exit medium is the
  *incident* medium for reflected SHG and the *substrate* for transmitted SHG. There is deliberately
  no $\cos\theta$ here: that factor belongs to $R$ and $T$, which are ratios between two beams of
  different widths, not to a single beam's intensity. Every published case exits into air, where
  $n_\text{exit}=1$ and the factor is invisible.
- The **reflected $\omega$ and reflected $2\omega$ are collinear and specular** (same medium, angle
  $=\theta_i$); only the *transmitted* $\omega$/$2\omega$ split by crystal dispersion. (This is what
  the optical-setup schematic draws — see {doc}`guide/interface`.)

## Multiple-reflection assumptions (multilayer)

The Maker-fringe sweep supports **Full multiple reflections (FMR)** (with forward/backward/standing
sub-modes), **Jerphagnon–Kurtz** (no multiple reflections), and **Herman–Hayden** (multiple
reflections only for the homogeneous $2\omega$ waves). See {doc}`guide/ml_tab`.

## Wavelength

The wavelength you set is the **fundamental**. The permittivity at the second harmonic comes from
each material's own second-harmonic data at that fundamental, so you never enter a second
wavelength.

That second-harmonic data is the material's index at **half the wavelength**, $\lambda/2$. A
material given by an index table is read there directly, so a table covering $\lambda_\text{lo}$
to $\lambda_\text{hi}$ can answer an SHG sweep only from $2\lambda_\text{lo}$ up to
$\lambda_\text{hi}$. The same halving is why a wavelength inside a material's data range can still
reach an ultraviolet pole at the second harmonic. Fresnel coefficients are linear optics at the
fundamental, so neither applies to them: a Fresnel sweep uses the whole table. The ranges of the
shipped tables are listed in {doc}`usage`.

Where the wavelength reaches the answer differs between the two methods, and it follows from what
each problem contains rather than from a modelling choice:

- A **single interface** has no thickness in it, so it has no length to compare a wavelength
  against. The wavelength reaches the answer only through the dispersion of the permittivity.
- A **layer stack** has the layer thicknesses, so the wavelength reaches the answer twice: through
  the same permittivity dispersion, and through the optical thickness of each layer. A stack whose
  materials carry one index at every wavelength therefore still gives a curve that moves, through
  the optical thickness alone. That curve is a thickness sweep rather than a spectrum.

Across a wavelength sweep the SHG tensor is held at its tabulated value, so a computed spectrum
carries the dispersion of the linear optics and a wavelength-independent nonlinearity. See
{doc}`technical_reference` for what that costs and when it matters.

## Validation metadata

Each {py:class}`~shaarp.SHAARPResult` carries a {py:class}`~shaarp.ValidationStatus` /
{py:class}`~shaarp.PhysicsConventions` describing whether the result came from a Mathematica-validated
path and any reduced-model assumptions in effect.
