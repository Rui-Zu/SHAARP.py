# Conventions

SHAARP.py follows the conventions of the original ♯SHAARP package and the published *npj Computational
Materials* equations (see {doc}`references`). The exact, machine-readable conventions attached to each
run are exposed via {py:class}`~shaarp.PhysicsConventions` (on the result), so a computation is always
self-describing.

## Coordinate frames

- **Crystal-physics frame** — the material's intrinsic axes; $\varepsilon$ and the $d$ tensor are
  entered here.
- **Lab frame** — L1, L2, L3 with **L3 along the surface normal** and the **plane of incidence = the
  L1–L3 plane**. The {py:class}`~shaarp.CrystalOrientation` rotates crystal → lab (z-cut = identity;
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
- **Analyzer angle $\psi$** selects the detected polarization (rotating analyzer returns the parallel
  $I_p$ and perpendicular $I_s$ channels).

## The SHG $d$ tensor (Voigt notation)

The second-order nonlinear response is the $3\times6$ Voigt matrix $d_{i\mu}$ ($i=1..3$, $\mu=1..6$),
with $P_i^{2\omega} = \varepsilon_0 \sum_\mu d_{i\mu}\,(E E)_\mu$ and the contracted index
$\mu \in \{xx, yy, zz, yz, xz, xy\}$. The crystal **point group** fixes which components are
independent; the rest follow by symmetry. See {py:func}`shaarp.d_voigt_symbolic` and
{py:func}`shaarp.rotate_d_voigt_crystal_to_lab`. Centrosymmetric/isotropic groups have $d \equiv 0$
(SHG forbidden).

## Reflection / transmission

- Linear Fresnel coefficients are returned as **power** $R_p, R_s, T_p, T_s$; for a lossless stack they
  satisfy $R+T=1$ per polarization, and at normal incidence $p$ and $s$ are degenerate.
- The **reflected $\omega$ and reflected $2\omega$ are collinear and specular** (same medium, angle
  $=\theta_i$); only the *transmitted* $\omega$/$2\omega$ split by crystal dispersion. (This is what
  the optical-setup schematic draws — see {doc}`guide/interface`.)

## Multiple-reflection assumptions (multilayer)

The Maker-fringe sweep supports **Full multiple reflections (FMR)** (with forward/backward/standing
sub-modes), **Jerphagnon–Kurtz** (no multiple reflections), and **Herman–Hayden** (multiple
reflections only for the homogeneous $2\omega$ waves). See {doc}`guide/ml_tab`.

## Validation metadata

Each {py:class}`~shaarp.SHAARPResult` carries a {py:class}`~shaarp.ValidationStatus` /
{py:class}`~shaarp.PhysicsConventions` describing whether the result came from a Mathematica-validated
path and any reduced-model assumptions in effect.
