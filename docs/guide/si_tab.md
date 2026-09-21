# SHAARP.si — single-interface tab

Models **reflected SHG** from a single air/crystal interface (semi-infinite crystal). Set the material,
geometry, and polarimetry on the left, click **Update**, and read the polar plots / closed form on the
right.

## Functionality

The **Functionality** dropdown lists **compute modes only** — each one runs physics and produces a
result (the help page is under **Help**, **User Guide** and on the startup *Guide* tab; the optical
schematic is a persistent banner; the crystal-axes view sits in the orientation input group):

```{list-table}
:header-rows: 1

* - Option
  - What it does
* - **SHG Simulation**
  - Numeric reflected-SHG polarimetry. Produces the polar plots.
* - **Partial Analytical Expressions**
  - Closed-form reflected-SHG polarimetry, symbolic in the input polarization $\varphi$ and the $d_{ij}$ components (numeric angle/indices, taken from the selected material — complex $\varepsilon$ included for absorbing crystals).
* - **Full Analytical Expressions**
  - Closed-form symbolic in $\varphi$, $d_{ij}$, angle, **and** indices. Available whenever the crystal's lab-frame $\varepsilon$ is diagonal, meaning it is aligned with its principal axes — isotropic, uniaxial and biaxial alike. A rotated crystal whose lab $\varepsilon$ picks up off-diagonal terms, such as TaAs (112), has no principal-aligned closed form; the output says so and points you to *SHG Simulation*.
```

```{tip}
The analytical modes run a computer-algebra solve. The progress bar says so: the **first** run for
a configuration can take from seconds up to a few minutes (rotated or multilayer cases); the result
is cached, so repeats are instant.
```

## Material setup

**Case-study material.** The dropdown lists the published ♯SHAARP.si case studies, each with its
constants as published. The *Cases in DOI* group — the four cases worked in the ♯SHAARP.si paper —
is **all at 800 nm** (the header says so); the other groups are:

> **Cases in DOI (all 800 nm):** GaAs (111), LiNbO₃ (11-20) MTI X-cut, KTP (100), TaAs (112)
> **Complex SHG Coefficients:** GaAs (111) @1064 nm
> **Deep UV NLO:** LiB₃O₅ (LBO), KBBF · **Polar Metals:** LiOsO₃
> **Dispersive (published index data):** LiNbO₃, KTP, LBO, GaAs and TaAs, each name ending with
> the wavelength range its index data covers

Choosing one auto-fills the point group, lattice, $\varepsilon(\omega)$, $\varepsilon(2\omega)$,
and the $d$ tensor. Select **"Custom (use fields)"** to enter everything yourself.

The **Dispersive** group sits after the published case studies. It holds five of these crystals
again, each given a published index curve so that a wavelength sweep over it means something; see
[Wavelength Scan Range](#wavelength-scan-range) below.

At the end of the same dropdown, under the header **— My Materials —**, sit any materials you
have saved yourself with the **My Materials** group at the bottom of the input column (name field +
**Save current as new**); they are persistent, single-wavelength, and shared with the ML tab's layer
editor. See {doc}`my_materials`.

```{note}
The ♯SHAARP.ml palette materials (Air, quartz, ZnO, MoS₂, …) live on the **SHAARP.ml tab** as film
choices. Centrosymmetric / isotropic materials there
are SHG-forbidden by symmetry; the app shows **"SHG ≈ 0 (symmetry-forbidden)"** rather than a
spurious signal.
```

**Point group.** The dropdown has two sections, under headers you cannot select:

> **— Noncentrosymmetric (SHG-active) —** 1, 2, m, mm2, 222, 3, 32, 3m, 4, 6, -4, 4mm, 6mm, 422,
> 622, -42m, -6, -6m2, -43m, 23, ∞, ∞m, ∞2
> **— Centrosymmetric (SHG-inactive) —** -1, 2/m, mmm, 4/m, 4/mmm, -3, -3m, 6/m, 6/mmm, m3, m3m,
> 432, ∞/m, ∞/mm, ∞∞, ∞∞m

This choice alone decides whether a crystal is an SHG source; there is no separate "SHG active"
switch.

- A group from the first section constrains which $d_{ij}$ are independent, and the rest follow by
  symmetry. For 3m that is $d_{16}=-d_{22}$, $d_{24}=d_{15}$; for -43m only $d_{14}=d_{25}=d_{36}$.
- A group from the second section sets $d \equiv 0$: the *SHG Tensor dᵢⱼ* group collapses, its
  title gaining the suffix **"— not used: SHG-inactive point group (d ≡ 0)"**. Choosing an active
  group again restores the pattern.
- 432 is listed under the centrosymmetric heading although it is not centrosymmetric, because it
  has no SHG either. Its own symmetry cancels $d$: the cubic axes require $d_{14}=d_{25}$, the
  four-fold axis requires $d_{14}=-d_{25}$, so both are zero, with no Kleinman assumption needed.
  It is the only noncentrosymmetric class where this happens.

**Lattice constants.** $a, b, c$ in Å, $\alpha, \beta, \gamma$ in degrees. The point group's
crystal system locks the dependent cells — they grey out and are coerced to the rule:

| Crystal system | Locked to |
|---|---|
| Triclinic | nothing — all six free |
| Monoclinic | $\alpha=\gamma=90°$ |
| Orthorhombic | all angles $90°$ |
| Tetragonal | $b=a$; all angles $90°$ |
| Trigonal / hexagonal, and the Curie groups ∞, ∞m, ∞2, ∞/m, ∞/mm | $b=a$; $\alpha=\beta=90°$, $\gamma=120°$ |
| Cubic, and ∞∞, ∞∞m | $a=b=c$; all angles $90°$ |

Editing $a$ under a locked system propagates to the locked lengths. Every palette material's cell
already satisfies its rule, so presets load unchanged.

**Crystal orientation.** Three modes:

- **z-cut (identity)** — crystal-physics axes aligned with the lab frame.
- **Miller (hkl + in-plane uvw)** — the normal of the surface plane $(hkl)$ lies along lab L3, and
  the in-plane direction $[uvw]$ along lab L2 (it must lie in the surface plane).
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

**Wavelength.** Fundamental $\lambda$ (µm); it sets the dielectric tensors of a case-study
material. A case defined at one wavelength holds the field at that wavelength, and the row label
says so. If $\lambda$ falls **outside** the selected material's built-in index data, the tensors
are held at the nearest tabulated value and an amber note under the field names the material and
its range, so nothing is extrapolated without your knowing. For a crystal from the **Dispersive**
group the note also covers the second harmonic, which reads the index at λ/2: when λ/2 falls below
the table, the note says ε(2ω) is held at the table's first value and names the wavelength from
which both harmonics are covered.

## Wavelength Scan Range

This section describes the sweep on both tabs; {doc}`ml_tab` adds what is particular to a layer
stack.

Tick **sweep the wavelength** to compute across a band instead of at one wavelength, and set
**λ min (µm)**, **λ max (µm)** and **λ step (µm)**. Press **Update** and the result appears on the
**Spectrum** output tab. The status line under the plots says what was computed, for example
"SHG Simulation: spectrum over 26 wavelengths, 0.55–0.8 µm". The grid stops at the last step that
fits, so its end can fall short of λ max. The fields open at 0.55–0.80 µm in 0.01 µm steps, a
quick first look; the chips beside each field offer the common alternatives, and any value can be
typed.

```{figure} ../_static/screens/spectrum.png
:width: 100%
:alt: The SHAARP.si tab after a wavelength sweep: the scan-range box ticked above a greyed-out single-wavelength field, and the Spectrum output tab showing the s and p reflected SHG channels both rising across the band

A sweep on the **Spectrum** tab: KTP from the **Dispersive** group, its scan range fitted to
0.86–1.6 µm, with the polarizer held at φ = 30°.
```

A spectrum needs one number per wavelength, so turning the sweep on sets the polarizer and the
analyzer to *Fix* and greys out their selectors; the fixed $\varphi$ and $\psi$ you enter set the
geometry. Turning the sweep off gives the selectors back the settings they had before. The sweep
also greys out the **wavelength (µm)** field, which it does not use, and the single-wavelength note
under that field stands down; both come back when you untick the box, or when an analytical mode
greys the sweep out. Setting λ min equal to λ max collapses the sweep back to a single wavelength.

The sweep works with SHG Simulation. The analytical modes return a closed-form expression rather
than a curve, and each wavelength would give a different one, so there the switch greys out and a
note says where a spectrum comes from instead.

A sweep costs one full solve per wavelength. When **Update** estimates that a sweep will take more
than about a minute, it asks before starting, with the size of the run and the expected time. If
you decline, nothing is computed: the progress bar reads "Not computed", and a banner above the
plots says they still show the previous result, whose notes stay with them. When a sweep is
refused or fails, the same banner appears and the progress bar reads "Did not complete".

For a real spectrum, pick a crystal from the **Dispersive** group. Each carries a published index
curve, and its name ends with the range that data covers. A sweep can use a little less: the second
harmonic reads the index at half the wavelength, so the usable range starts at twice the low end.
When you pick one of these, the app fits the scan range into that span, rounded inward to 0.01 µm.
When it moves the range, the status bar says so; a range that already fits is left alone. The
usable range of each crystal is in the
[materials table](../usage.md#materials-that-already-carry-dispersion) of the usage page.

After a sweep, the amber note under the wavelength field says what the result cannot show. It
lists every distinct note, up to three, and hovering over it shows them all. The two you meet on
this tab:

- The permittivity does not move with wavelength. Most of the palette is defined at one
  wavelength, and on this tab that makes the spectrum a flat line. The note names the material and
  what to pick instead: the table-backed version of the same crystal when there is one (for
  KTP (100), "KTP (dispersive) 0.43-3.54 um"), otherwise a crystal from the Dispersive group.
- The sweep ran past the end of an index table. The note gives the range the table can answer.
  Outside it the index is held at the nearest tabulated value rather than extrapolated.

The SHG tensor is held at its tabulated value across the sweep, as the plot says beneath its title;
{doc}`../conventions` explains what that leaves out of a spectrum.

## Polarimetry settings

- **Incident angle $\theta_i$** — spin box (0–89.9°) + slider + quick-angle buttons (0, 15, 30, 45, 60, 75).
- **Incident field / polarization $\varphi$** — the incident Jones vector $E = E_0(\cos\varphi,\ \sin\varphi\,e^{i\Delta\delta},\ 0)$.
  *Rotate Polarizer* sweeps $\varphi$; *Fix Polarizer* holds $\varphi$ and sweeps the analyzer.
- **Ellipticity $\Delta\delta$** — incident-field phase between components.
- **Analyzer** — two choices, *Rotate Analyzer* and *Fix Analyzer*. Together with the polarizer
  setting they decide what the polar plots draw:

  | Polarizer | Analyzer | Polar Plots |
  |---|---|---|
  | Rotate Polarizer | Fix Analyzer | $I^{2\omega}(\varphi)$ with the analyzer at $\psi$ and at $\psi + 90°$; with $\psi = 0°$ these are the $p$ and $s$ channels |
  | Rotate Polarizer | Rotate Analyzer | the analyzer turns with the polarizer, $\psi = \varphi + \text{offset}$: the parallel channel $I_\parallel$ and the perpendicular one $I_\perp$. Set the offset in **analyzer–polarizer offset (deg)**. This is the original's co-rotating mode. |
  | Fix Polarizer | either | one panel, $I^{2\omega}(\psi)$ against the analyzer angle, at the $\varphi$ you enter |

## Outputs

- **Optical setup schematic** — a 2D drawing of the setup at the entered $\theta_i$, above the
  plots (see {doc}`interface`). The 3D view is the crystal-axes view in the **Crystal Orientation**
  group, where **View** switches it to a 2D top view.
- **Polar Plots** — the reflected SHG patterns chosen by the polarizer and analyzer settings
  above, plus the effective-refractive-index curve $n(\theta_i)$ and the incident-ellipticity locus.
- **Spectrum** — with the wavelength sweep on, $I_s^{2\omega}(\lambda)$ and $I_p^{2\omega}(\lambda)$
  against the fundamental wavelength, at the fixed geometry you set. An **Update** with the sweep
  off empties this tab.
- **Analytical Expression** — the closed form from the two analytical modes, typeset with real
  super/subscripts and Greek symbols (n_ω², θᵢ, d₁₄, φ). **Copy closed form (Python/SymPy)** and
  the `.txt` export deliver machine-readable SymPy text rather than the typeset view; **Copy closed
  form (Mathematica)** gives the same expression in Wolfram Language syntax.

See {doc}`outputs_export` for copying and exporting. Next: {doc}`ml_tab`.
