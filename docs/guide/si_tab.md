# SHAARP.si — single-interface tab

Models **reflected SHG** from a single air/crystal interface (semi-infinite crystal). Set the material,
geometry, and polarimetry on the left, click **Update**, and read the polar plots / closed form on the
right.

## Functionality

The **Functionality** dropdown lists **compute modes only** — each one runs physics and produces a
result (the help page is on the **Help → User Guide** menu and the startup *Guide* tab; the optical
schematic is a persistent banner; the crystal-axes view sits in the orientation input group):

```{list-table}
:header-rows: 1

* - Option
  - What it does
* - **SHG Simulation**
  - Validated numeric reflected-SHG polarimetry (the `shaarp_si_compat` workflow). Produces the polar plots.
* - **Partial Analytical Expression**
  - Closed-form reflected-SHG polarimetry, symbolic in the input polarization $\varphi$ and the $d_{ij}$ components (numeric angle/indices, taken from the selected material — complex $\varepsilon$ included for absorbing crystals).
* - **Full Analytical Expression**
  - Closed-form symbolic in $\varphi$, $d_{ij}$, angle, **and** indices. Fully-symbolic forms exist for isotropic/uniaxial crystals in the identity or pure-$R_z$ orientations; other orientations fall back to the substituted (Partial-style) form **with a declared note** in the output.
```

```{tip}
The analytical modes run a computer-algebra solve. The progress bar says so: the **first** run for
a configuration can take from seconds up to a few minutes (rotated or multilayer cases); the result
is cached, so repeats are instant.
```

## Material setup

**Case-study material.** The dropdown is the original ♯SHAARP.si palette, with each case's
constants transcribed verbatim from the notebook. The *Cases in DOI* group — the four cases worked
in the ♯SHAARP.si paper — is **all at 800 nm** (the header says so); the other groups follow the
original panel:

> **Cases in DOI (all 800 nm):** GaAs (111), LiNbO₃ (112̄0) MTI X-cut, KTP (100), TaAs (112)
> **Complex SHG Coefficients:** GaAs (111) @1064 nm
> **Deep UV NLO:** LiB₃O₅ (LBO), KBBF · **Polar Metals:** LiOsO₃

Choosing one auto-fills the point group, lattice, $\varepsilon(\omega)$, $\varepsilon(2\omega)$,
and the $d$ tensor. Select **"Custom (use fields)"** to enter everything yourself.

At the end of the same dropdown, under the header **— My Materials —**, sit any materials you
have saved yourself with the **My Materials** group at the bottom of the input column (name field +
**Save current as new**); they are persistent, single-wavelength, and shared with the ML tab's layer
editor. See {doc}`my_materials`.

```{note}
The ♯SHAARP.ml palette materials (Air, quartz, ZnO, MoS₂, …) live on the **SHAARP.ml tab** as film
choices, matching the original packages' separation. Centrosymmetric / isotropic materials there
are SHG-forbidden by symmetry; the app shows **"SHG ≈ 0 (symmetry-forbidden)"** rather than a
spurious signal.
```

**Point group.** The dropdown reproduces the original package's two popups
(SHAARP.ml.nb:5191 / :5630) as two header-separated sections; the headers themselves cannot be
selected:

> **— Noncentrosymmetric (SHG-active) —** 1, 2, m, mm2, 222, 3, 32, 3m, 4, 6, -4, 4mm, 6mm, 422,
> 622, -42m, -6, -6m2, -43m, 23, ∞, ∞m, ∞2
> **— Centrosymmetric (SHG-inactive) —** -1, 2/m, mmm, 4/m, 4/mmm, -3, -3m, 6/m, 6/mmm, m3, m3m,
> 432, ∞/m, ∞/mm, ∞∞, ∞∞m

Whether a crystal is an SHG source is decided by this choice alone — there is no separate
"SHG active" switch. Selecting a group in the first section constrains which $d_{ij}$ are
independent; the rest follow by symmetry (e.g. 3m: $d_{16}=-d_{22}$, $d_{24}=d_{15}$; -43m: only
$d_{14}=d_{25}=d_{36}$). Selecting a group in the second section sets $d \equiv 0$: the *SHG
Tensor dᵢⱼ* group collapses and its title gains the suffix **"— not used: SHG-inactive point group
(d ≡ 0)"**; choosing an active group again restores the pattern. 432 sits in the inactive section
because its $d$ vanishes under Kleinman symmetry, as in the original.

**Lattice constants.** $a, b, c$ (Å) and $\alpha, \beta, \gamma$ (deg). The point group's crystal
system locks the dependent cells (greyed out and coerced to the rule): triclinic — all free;
monoclinic — $\alpha=\gamma=90°$; orthorhombic — all angles $90°$; tetragonal — $b=a$, angles
$90°$; trigonal / hexagonal (and the Curie groups ∞, ∞m, ∞2, ∞/m, ∞/mm) — $b=a$,
$\alpha=\beta=90°$, $\gamma=120°$; cubic (and ∞∞, ∞∞m) — $a=b=c$, angles $90°$. Editing $a$ under a
locked system propagates to the locked lengths. Every palette material's cell already satisfies
its rule, so presets load unchanged.

**Crystal orientation.** Three modes:

- **z-cut (identity)** — crystal-physics axes aligned with the lab frame.
- **Miller (hkl + in-plane uvw)** — surface plane $(hkl)$ → surface normal (lab L3); in-plane $[uvw]$
  → lab L2 (must lie in the surface plane).
- **Crystal Physics Directions (Z1,Z2,Z3)** — enter the crystal axes as rows in the lab frame
  (validated for orthonormality).

The group is compact: only the selected mode's input rows are shown — **Surface (hkl)** and
**In-plane [uvw]** in Miller mode, **Z1 (lab)** / **Z2 (lab)** / **Z3 (lab)** in Crystal Physics
mode, and neither for z-cut. Beneath them sits the small live **crystal-axes ($Z_i$) vs lab-axes
($L_i$)** figure, its title above the plot and the *lab $L_i$* / *crystal $Z_i$* legend along the
bottom.

**Dielectric tensors.** Full $3\times3$ complex, symmetric $\varepsilon(\omega)$ and $\varepsilon(2\omega)$
(editing an off-diagonal mirrors its partner). The scalar $n_\omega / n_{2\omega}$ (and extraordinary
indices) feed the figure's effective-index curve.

**SHG tensor.** The full $3\times6$ Voigt $d$ matrix. Enter the independent components; the point-group
symmetry relations fill the rest.

**Wavelength.** Fundamental $\lambda$ (µm); it sets the dielectric interpolation for case-study materials.
If $\lambda$ falls **outside** a selected material's exported dispersion grid, the tensors clamp to the
nearest tabulated value and an amber note appears under the field naming the material and its
tabulated range — no silent extrapolation.

## Polarimetry settings

- **Incident angle $\theta_i$** — spin box (0–89°) + slider + quick-angle buttons (0, 15, 30, 45, 60, 75).
- **Incident field / polarization $\varphi$** — the incident Jones vector $E = E_0(\cos\varphi,\ \sin\varphi\,e^{i\Delta\delta},\ 0)$.
  *Rotate Polarizer* sweeps $\varphi$; *Fix Polarizer* holds $\varphi$ and sweeps the analyzer.
- **Ellipticity $\Delta\delta$** — incident-field phase between components.
- **Analyzer** — three modes:
  *Rotating* (returns the standard $I_p$ and $I_s$ channels), *Fixed* at angle $\psi$, or
  *Rotating with polarizer* ($\psi = \varphi + \text{offset}$) — the original's co-rotating
  analyzer/polarizer mode; enter the offset in degrees and the plot shows the parallel and
  perpendicular channels tracked at that offset.

## Outputs

- **2D & 3D schematics** — the optical setup at the entered $\theta_i$ (see {doc}`interface`).
- **Polar Plots** — reflected $I_p^{2\omega}(\varphi)$ and $I_s^{2\omega}(\varphi)$, plus the
  effective-refractive-index curve $n(\theta_i)$ and the incident-ellipticity locus.
- **Analytical Expression** — the closed form (Partial/Full Analytical modes), TYPESET like the original package: real super/subscripts and Greek symbols (n_ω², θᵢ, d₁₄, φ). The **Copy** button and the `.txt` export deliver the machine-readable SymPy text (paste-able into Python/Mathematica), not the typeset view.

See {doc}`outputs_export` for Copy/Export. Next: {doc}`ml_tab`.
