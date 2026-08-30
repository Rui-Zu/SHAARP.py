# SHAARP.ml — multilayer tab

Models SHG from an **N-layer thin-film stack** (air / film(s) / substrate): Maker-fringe angle scans,
linear Fresnel curves, reflected+transmitted polarimetry, and the closed-form film expression.

```{figure} ../_static/screens/ml_tab.png
:width: 100%
:alt: SHAARP.py SHAARP.ml tab
```

## Functionality

The dropdown lists **compute modes only** (help is on the **Help** menu + startup tab; the schematic
is a persistent banner; the crystal-axes view is in the orientation input group):

```{list-table}
:header-rows: 1

* - Option
  - What it does
* - **SHG Simulation**
  - Validated numeric multilayer SHG polarimetry.
* - **Maker Fringes**
  - Transmitted/reflected SHG intensity vs incidence angle $\theta_i$ (a $\theta$-sweep).
* - **Fresnel Coefficients**
  - Linear reflection/transmission power coefficients $R_p, R_s, T_p, T_s$ vs $\theta_i$.
* - **Partial Analytical Expressions**
  - Closed-form polarimetry for the stack, symbolic in $\varphi$ plus whichever thicknesses and
    SHG tensors you mark analytical.
```

Rotational anisotropy (the 0–360° sample-azimuth polar scan) is not a separate mode: it is the
**Sample rotation** toggle inside *Polarimetry settings* (see below), computed within
SHG Simulation exactly as the original's `samplerotationcontrol`, and available under
Partial Analytical Expressions too.

The functionality combo is the single control for each plot — there are no separate "generate
plot" checkboxes (the former *Generate Fresnel Coefficients Plot* / *Generate Maker Fringes Plot*
boxes duplicated the *Fresnel Coefficients* / *Maker Fringes* modes and were removed). Pressing
**Update / Run** in one of those two modes computes that $\theta$-sweep, fills the matching
*Maker Fringes* / *Fresnel Coefficients* output tab, and switches the output panel to it;
**Export data** then writes that mode's curves (the $\theta$ grid with the fringe intensities, or
$R_p, R_s, T_p, T_s$). The expensive sweeps therefore run only in their own modes, and the
*Maker Fringes Scan Range* / *Fresnel Coefficients Scan Range* groups are read only by their
own mode (see below).

Two per-layer checkboxes decide what stays a symbol, and each sits with the input it governs:
**analytical h (this layer's thickness stays symbolic)** in the *Layer Selection (N-layer stack)*
group, under the thickness row, and **analytical dᵢⱼ (this layer's SHG tensor stays symbolic)** in
the *SHG Tensor dᵢⱼ (full 3×6 Voigt, pm/V)* group, below the grid. Both apply to the layer
currently selected in **Edit layer**, and any subset of layers may be symbolic while the rest are
substituted. Neither box appears while a semi-infinite row (the first or the last layer) is
selected: their thickness is not a variable, and the solver never treats a half-space as an SHG
source. Whether an interior layer is an SHG source is decided by its **Point group** (see
*System / layer setup* below), not by a switch: for a layer whose group is in the
*Centrosymmetric (SHG-inactive)* section the **analytical dᵢⱼ** box is unchecked and disabled,
and the layer's analytical-$d$ flag and known values are cleared.

A symbol is named after the row you are editing: with air / quartz / Au / air, flagging the quartz
row gives $h_2$ and $d_{11}m2$, $d_{14}m2$. While a flag is set the panel shows those symbols
instead of numbers — the thickness field displays `h2`, and the $d$ grid displays the point group's
symmetry-allowed components by name (for Z-cut quartz on row 2: `d11m2, -d11m2, 0, d14m2, 0, 0` on
the first row, `-d14m2, -d11m2` in the last two cells of the second, all else `0`), with a hard `0`
wherever symmetry forbids one. The flag belongs to the layer, not to the display: re-selecting the
row, changing its material, or reloading the stack brings the symbolic grid back, and only clearing
the box restores the numbers. The symbolic grid is a *Partial Analytical Expressions* display,
though: under any numeric functionality (SHG Simulation, Maker Fringes, …) the flags are
inert, the computation uses the stored numbers, and the grid shows those numbers even with the box
still ticked — switching the functionality combo re-mirrors the grid either way.

Only part of a tensor needs to be unknown. To declare a component **known**, type a **number** over
its symbol while the layer is flagged: the value is stored on that layer and the grid shows it
again every time the row is displayed, with the dependent entries following by symmetry (typing
`0.3` over `d11m2` on the quartz row gives `0.3, -0.3, 0, d14m2, 0, 0` on the first row and `-0.3`
in the last cell of the second; `d14m2` stays a symbol). A typed `0` counts as known-zero. To make
a component unknown again, clear the cell or type its symbol name back. Typing a number declares a
known component only while the grid is symbolic (*Partial Analytical Expressions*); in a numeric
functionality a $d$ edit is an ordinary tensor edit and converts a palette row to *Custom (fields)*
as usual. Declaring a known value is
not an edit of the material's tensor — the row keeps its palette material and does not become
*Custom (fields)* — and the value survives switching to another row and back. Update reads the
flags and the known values stored on the stack, not the grid on screen: a flagged layer stays
symbolic in the closed form whichever row is selected when you press it, and every flagged layer's
known values are substituted and listed in the `# symbols:` header as `known d = d11m2=0.3+0j`.
Editing anything else in a flagged row — its dielectric tensors, for instance — still converts it
to *Custom (fields)*; the converted layer keeps its analytical flags and known values, and its
snapshot carries the material's real numeric $d$ rather than the symbolic grid.

Ticking either box switches the functionality combo to *Partial Analytical Expressions* for you;
the `# symbols:` header line of the expression declares exactly which symbols survived and what
was substituted (e.g. `thickness = h2 = 121.2 um (substituted)` when analytical h is off for that
layer).

```{tip}
The analytical mode runs a computer-algebra solve; the progress bar announces that the **first** run
for a configuration can take minutes (some multilayer cases up to ~6) and that results are cached —
repeat Updates are instant.
```

## System / layer setup

**System presets.** The paper's demonstrated heterostructures at their published wavelengths:
*"Quartz + Au (Fig 4, 800 nm)"* (the documentation example, validated to ~1e-9 against the
original), *"ZnO / Pt / Al₂O₃ (Fig 6, 1550 nm)"*, and *"LiNbO₃ / Quartz (Fig 7, 1550 nm)"*.
Below them, the **Single film in air** section lists the original ♯SHAARP.ml palette — each
material labelled with its provenance wavelength (multi-wavelength materials are grouped under a
master title, e.g. *Quartz → x-cut · 1064 nm / z-cut · 800 nm*) — plus **"N-layer stack
(editor)"** and **"Custom film (use fields)"**.

**N-layer stack editor.** Set the **number of layers** (layer 1 = ambient/air, layers 2…N-1 = films,
layer N = substrate half-space). For the selected layer choose the **material**, an optional **layer
name** (shown in the selector and the schematic; leave blank for the automatic "role: material"
label), and the **thickness** (µm; disabled for half-spaces). Each layer can carry its own custom
crystal (point group, orientation, $\varepsilon$, $d$) via the same controls as {doc}`si_tab`;
stack labels also show the material's point group and surface $(hkl)$. The **substrate** is
isotropic — enter scalar $n_\omega, n_{2\omega}$.

**SHG activity is decided by the point group.** As in the original ♯SHAARP.ml, there is no
per-layer "SHG active" checkbox. A layer is an SHG source if and only if it is an interior layer,
not air / isotropic, and its **Point group** (Crystal Structure group) is in the
*— Noncentrosymmetric (SHG-active) —* section of the dropdown; the
*— Centrosymmetric (SHG-inactive) —* section (-1, 2/m, mmm, 4/m, 4/mmm, -3, -3m, 6/m, 6/mmm, m3,
m3m, 432, ∞/m, ∞/mm, ∞∞, ∞∞m) makes it a linear layer with $d \equiv 0$. The palette materials
show their real group there — the palette *Air* and *Au coating* are ∞∞m, *Pt (111)* and *Blank
linear* are m3m, *Al2O3 (0001)* is 6/mmm. Choosing an inactive group collapses the *SHG Tensor dᵢⱼ
(full 3×6 Voigt, pm/V)* group with the title suffix **"— not used: SHG-inactive point group
(d ≡ 0)"**, unchecks and disables the **analytical dᵢⱼ** box, and clears that layer's
analytical-$d$ flag and known values; choosing an active group restores the pattern and the box.
The generic ambient rows (*air*, *isotropic n (set below)*) keep their own
"— not used: isotropic medium" hint instead. The point group's crystal system also locks the dependent
lattice cells (e.g. hexagonal/trigonal $b=a$, $\gamma=120°$; cubic $a=b=c$) exactly as on the SI
tab. Session files saved before this change that still carry a `shg_active` key load normally;
the key is ignored.

**Crystal-axes view.** The orientation input group draws the selected layer's **crystal-physics
axes ($Z_i$) against the lab axes ($L_i$)** — a quick visual check of the entered orientation, live
as you edit it (for GaAs (111) the $Z$ triad tilts so that $\sum Z_i \parallel L_3$). The figure is
compact, with its title above the plot and the *lab $L_i$* / *crystal $Z_i$* legend along the
bottom, and the group shows only the rows of the selected orientation mode — Surface (hkl) /
In-plane [uvw] in Miller mode, Z1/Z2/Z3 in Crystal Physics mode, neither for z-cut — exactly as on
the SI tab (see {doc}`si_tab`).

**My Materials.** The layer-material list also offers, under the header **— My Materials —** just
before *Custom (fields)*, the materials you have saved yourself (the group at the bottom of the input
column: name field + **Save current as new** / **Update selected** / **Rename…** / **Delete
selected**). They persist in `~/.shaarp/user_materials.json`, are single-wavelength, and are not
listed in the *Single film in air* section — assign them to a layer in the stack editor. The former
per-session *Layer Properties Preset Values* group is no longer shown. See {doc}`my_materials`.

## Assumptions

The multiple-reflection treatment for the Maker sweep:

```{list-table}
:header-rows: 1

* - Assumption
  - Meaning
* - **Full Multiple Reflections (FMR)**
  - All multiply-reflected waves retained. Has a sub-mode: *Forward only* / *Forward + Backward* / *Forward + Backward + Standing*.
* - **Jerphagnon & Kurtz (No MR)**
  - No multiple reflections (single-pass).
* - **Herman & Hayden (MR only for 2ω homogeneous waves)**
  - Multiple reflections kept only for the homogeneous $2\omega$ waves.
```

The **2D schematic mirrors the selected assumption**: FMR draws the multiple-reflection ladder for
both the fundamental (red) and SHG (blue) waves inside the film, plus the inhomogeneous source waves
(orange, with a backward leg when the sub-mode includes it); JK draws single-pass rays; HH draws a
single-pass fundamental with the $2\omega$ ladder only. An italic caption names the assumption.
With a backside metal coating (e.g. the quartz + Au case) the backward waves matter strongly — the
FMR fine-fringe amplitude grows visibly relative to HH/JK.

```{note}
The assumption also applies to **SHG Simulation**, not just the Maker sweep: the same FMR/JK/HH
policy and FMR sub-mode are passed to the numeric multilayer solve.
```

## Maker Fringes Scan Range / Fresnel Coefficients Scan Range

Each sweep mode has its **own** scan section, controlled and toggled separately.
**$\theta_{\min}$, $\theta_{\max}$, $\theta_{\text{step}}$** (deg) set the incidence-angle grid; a finer
step gives smoother curves at the cost of compute time. The *Maker Fringes Scan Range* defaults to
0–45° at 0.5°; the *Fresnel Coefficients Scan Range* defaults to the original's full 0–89.9° at a
finer **0.1°** step (the original fixed the Fresnel range at 0–90° and exposed only the step; the
separate min/max here is a deliberate extension). Each group clears its "— not used by this
mode" hint only in its own mode.

## Polarimetry settings

Incident angle $\theta_i$, incident polarization $\varphi$, ellipticity $\Delta\delta$, analyzer —
as on the SI tab, applied to the stack — plus (multilayer only) **Sample rotation**.

**Rotate/fix × 3.** The polarizer, the analyzer, and the sample each carry an independent
*rotate / fix* choice, and **any of the 8 combinations is legal** — nothing is pinned. Every
element set to *rotate* follows one common scan angle $t$ (0–360°): the polarizer at
$\varphi(t)=t$, the analyzer at $\psi(t)=t+\text{offset}$ (the analyzer–polarizer offset
participates only when **both** rotate, exactly the original's gate), and the sample at
$\psi_s(t)=\pm t$ per the **rotation direction** (CW/CCW, looking at the sample from the beam
side) with the **step size** setting the grid. A fixed element holds its fixed angle. With the
sample fixed this is the ordinary polarimetry; with the sample rotating the polar RA figure is
drawn over the sample azimuth. Self-consistency across the combinations is fenced in
`tests/test_polarimetry_combinations.py` (e.g. at normal incidence, rotating the sample by $t$
equals co-rotating polarizer and analyzer by $t$ with the sample fixed, to ~5e-10).

**Maker Fringes uses this panel too:** the sweep's *input* polarization ($\varphi$,
$\Delta\delta$) and *detection* polarization (analyzer $\psi$; the perpendicular channel sits at
$\psi+90°$) are the panel's fixed values — the rotate/fix selectors grey out in that mode. Fresnel
Coefficients is linear ($R_p, R_s, T_p, T_s$ per angle) and reads no polarimetry.

## Outputs

- **Maker Fringes** — $I(\theta_i)$ with the assumption shown in the subtitle; removable
  eigenmode-degeneracy singularities are interpolated (the title notes how many).
- **Fresnel Coefficients** — $R_p, R_s, T_p, T_s$ over the original's full 0–90° range at the
  panel's step (isolated singular boundary solves for metallic films are interpolated, like the
  Maker singularities; lossless stacks satisfy $R+T=1$).
- **Polar Plots** — reflected/transmitted $I_p$, $I_s$ panels, plus a **beam-ellipticity tile**
  showing the polarization ellipses of the incident, reflected, and transmitted fundamental beams.
  The co-rotating analyzer mode ($\psi = \varphi +$ offset) is available on this tab too.
- **Analytical Expression** — the closed form $I(\varphi, d, h)$, typeset with real
  super/subscripts and Greek symbols like the original package; the Copy button and the `.txt`
  export stay machine-readable SymPy.

See {doc}`outputs_export`. Common questions are in the {doc}`faq`.
