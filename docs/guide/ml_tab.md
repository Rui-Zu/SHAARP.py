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

**The functionality combo is the single control for each plot.** There are no separate "generate
plot" checkboxes — the former *Generate Fresnel Coefficients Plot* / *Generate Maker Fringes Plot*
boxes duplicated the corresponding modes and were removed. So, in the two $\theta$-sweep modes:

- **Update / Run** computes that sweep, fills the matching *Maker Fringes* / *Fresnel
  Coefficients* output tab, and switches the output panel to it.
- **Export data** then writes that mode's curves — the $\theta$ grid with the fringe intensities,
  or $R_p, R_s, T_p, T_s$.
- Each *Scan Range* group is read only by its own mode, so the expensive sweeps never run behind
  your back (see below).

### Choosing what stays symbolic

Two per-layer checkboxes decide what the closed form keeps as a symbol. Each sits with the input it
governs, and both apply to the layer currently selected in **Edit layer**:

| Checkbox | Where it is | What stays symbolic |
|---|---|---|
| **analytical h** | *Layer Selection (N-layer stack)*, under the thickness row | that layer's thickness |
| **analytical dᵢⱼ** | *SHG Tensor dᵢⱼ (full 3×6 Voigt, pm/V)*, below the grid | that layer's SHG tensor |

Any subset of layers may be symbolic while the rest are substituted. Ticking either box switches
the functionality combo to *Partial Analytical Expressions* for you.

**When a box is missing or greyed out** — that is deliberate, in two cases:

- **A semi-infinite row is selected** (the first or the last layer): neither box appears. A
  half-space has no thickness variable, and the solver never treats one as an SHG source.
- **The layer's point group is centrosymmetric**: the **analytical dᵢⱼ** box is unchecked and
  disabled, and that layer's analytical-$d$ flag and known values are cleared. SHG activity follows
  the **Point group** alone (see *System / layer setup* below) — there is no separate switch.

### How the symbols are named

A symbol is named after the row you are editing. With a stack of air / quartz / Au / air, flagging
the quartz row (row 2) gives $h_2$, $d_{11}m2$, $d_{14}m2$.

While a flag is set, the panel shows those symbols in place of numbers:

- the thickness field displays `h2`;
- the $d$ grid displays the point group's symmetry-allowed components by name, with a hard `0`
  wherever symmetry forbids one. For Z-cut quartz on row 2 that is `d11m2, -d11m2, 0, d14m2, 0, 0`
  across the first row, `-d14m2, -d11m2` in the last two cells of the second, and `0` everywhere
  else.

The flag belongs to the **layer**, not to the display. Re-selecting the row, changing its material,
or reloading the stack all bring the symbolic grid back; only clearing the box restores the numbers.

```{note}
The symbolic grid is a *Partial Analytical Expressions* display. Under any numeric functionality
(SHG Simulation, Maker Fringes, …) the flags are inert: the computation uses the stored numbers,
and the grid shows those numbers even with the box still ticked. Switching the functionality combo
re-mirrors the grid either way.
```

### Declaring part of a tensor known

Only part of a tensor needs to be unknown — useful when you know some coefficients and want to fit
the rest.

1. **Flag the layer** with **analytical dᵢⱼ**, so its grid is symbolic.
2. **Type a number over a symbol.** It is stored on that layer, and the dependent entries follow by
   symmetry. Typing `0.3` over `d11m2` on the quartz row gives `0.3, -0.3, 0, d14m2, 0, 0` on the
   first row and `-0.3` in the last cell of the second — `d14m2` stays a symbol. A typed `0` counts
   as known-zero.
3. **To make it unknown again**, clear the cell or type its symbol name back.

Three things worth knowing about this:

- **It only works while the grid is symbolic.** Under a numeric functionality, a $d$ edit is an
  ordinary tensor edit and converts a palette row to *Custom (fields)* as usual.
- **It is not an edit of the material.** The row keeps its palette material, does not become
  *Custom (fields)*, and the value survives switching to another row and back.
- **Editing anything else in a flagged row still converts it** to *Custom (fields)* — its
  dielectric tensors, for instance. The converted layer keeps its analytical flags and known
  values, and its snapshot carries the material's real numeric $d$ rather than the symbolic grid.

### What Update actually reads

**Update reads the flags and known values stored on the stack, not the grid on screen.** A flagged
layer stays symbolic in the closed form whichever row happens to be selected when you press it.

The expression's `# symbols:` header then declares exactly what survived and what was substituted —
`known d = d11m2=0.3+0j` for a declared value, or
`thickness = h2 = 121.2 um (substituted)` when analytical h is off for that layer.

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
per-layer "SHG active" checkbox. A layer radiates SHG if and only if all three hold:

1. it is an **interior** layer (not the first or last half-space),
2. it is not air / isotropic, and
3. its **Point group** (Crystal Structure group) is in the *— Noncentrosymmetric (SHG-active) —*
   section of the dropdown.

Anything in the *— Centrosymmetric (SHG-inactive) —* section (-1, 2/m, mmm, 4/m, 4/mmm, -3, -3m,
6/m, 6/mmm, m3, m3m, 432, ∞/m, ∞/mm, ∞∞, ∞∞m) is a linear layer with $d \equiv 0$. The palette
materials show their real group: *Air* and *Au coating* are ∞∞m, *Pt (111)* and *Blank linear* are
m3m, *Al₂O₃ (0001)* is 6/mmm.

Selecting an inactive group changes three things at once:

- the *SHG Tensor dᵢⱼ (full 3×6 Voigt, pm/V)* group collapses, its title gaining the suffix
  **"— not used: SHG-inactive point group (d ≡ 0)"**;
- the **analytical dᵢⱼ** box is unchecked and disabled;
- that layer's analytical-$d$ flag and known values are cleared.

Choosing an active group again restores the symmetry pattern and the box. The generic ambient rows
(*air*, *isotropic n (set below)*) show their own "— not used: isotropic medium" hint instead.

The point group's crystal system also locks the dependent lattice cells (hexagonal/trigonal
$b=a$, $\gamma=120°$; cubic $a=b=c$), exactly as on the SI tab.

```{note}
Session files saved before SHG activity moved to the point group still carry a `shg_active` key.
They load normally; the key is ignored.
```

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
