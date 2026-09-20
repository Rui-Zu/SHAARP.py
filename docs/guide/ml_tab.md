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
  - Numeric multilayer SHG polarimetry.
* - **Maker Fringes**
  - Transmitted SHG intensity vs incidence angle $\theta_i$ (a $\theta$-sweep), in the parallel and
    perpendicular channels.
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
`Quartz + Au (Fig 4, 800 nm)` (the documentation example), `ZnO / Pt / Al2O3 (Fig 6, 1550 nm)`, and
`LiNbO3 / Quartz (Fig 7, 1550 nm)`. Those are the exact strings, spelled with plain digits; pass one
as `system_preset=` to {py:func}`shaarp.compute_ml_gui_result` to run the same case from Python.
Below them, the **Single film in air** section lists the original ♯SHAARP.ml palette, each
material labelled with its provenance wavelength. Materials with more than one variant are grouped
under a master title, such as *Quartz* with *x-cut · 1064 nm* and *z-cut · 800 nm* beneath it.
After the palette comes the **Dispersive** group, the five crystals that carry published index data
for a wavelength sweep (see [Wavelength Scan Range](#wavelength-scan-range)), and then
**"N-layer stack (editor)"** and **"Custom film (use fields)"**.

**N-layer stack editor.** Set the **number of layers** (layer 1 = ambient/air, layers 2…N-1 = films,
layer N = substrate half-space). For the selected layer choose the **material**, an optional **layer
name** (shown in the selector and the schematic; leave blank for the automatic "role: material"
label), and the **thickness** (µm; disabled for half-spaces). Each layer can carry its own custom
crystal (point group, orientation, $\varepsilon$, $d$) via the same controls as {doc}`si_tab`;
stack labels also show the material's point group and surface $(hkl)$. An interior layer's
material list holds the palette, the Dispersive crystals, your saved materials and *Custom
(fields)*. The first and last layers are half-spaces, and their list offers only *air* and the
isotropic option, where you enter scalar $n_\omega, n_{2\omega}$. In the simple single-film mode
above, the substrate is always the isotropic one.

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
policy and FMR sub-mode are passed to the numeric multilayer solve — including the
**sample-rotation** sweep below, where every azimuth point is solved under the selected
assumption and the plot's subtitle names it. Under JK and HH the source waves are forward-only,
as in the original's own JK/HH branches.
```

## Maker Fringes Scan Range / Fresnel Coefficients Scan Range

Each sweep mode has its **own** scan section, controlled and toggled separately.
**$\theta_{\min}$, $\theta_{\max}$, $\theta_{\text{step}}$** (deg) set the incidence-angle grid; a finer
step gives smoother curves at the cost of compute time. Both scan groups default to a **0.05°**
step: the *Maker Fringes Scan Range* over 0–45°, the *Fresnel Coefficients Scan Range* over the
original's full 0–89.9° (the original fixed the Fresnel range at 0–90° and exposed only the step;
the separate min/max here is a deliberate extension). Each group clears its "— not used by this
mode" hint only in its own mode.

The Maker default is fine on purpose. A 121.2 µm quartz slab, the default preset, puts its fringes
about 0.56° apart, so a 0.5° step lands barely one sample on each: the curve still follows the
envelope, but not the individual fringes. At 0.05° the fringes are
drawn smoothly and the full 0–45° sweep takes about a minute. The quick-preset buttons beside the
step (0.1, 0.5, 1, 2, 5) coarsen it for a fast look at the envelope. The Fresnel step is fine for
the same reason: a coated slab's reflectance and transmittance carry the slab's own interference.

## Wavelength Scan Range

The switch, the λ fields, the wavelength field greying out while the sweep is on, the question
before a long run, and the notes under the wavelength field work as on the SHAARP.si tab, described
under [Wavelength Scan Range](si_tab.md#wavelength-scan-range) there. Turning the sweep on here
also sets **Sample rotation** to *Fix*, and turning it off gives back the previous setting. A layer
stack adds the following.

All three compute modes sweep. SHG Simulation plots one spectrum. With **Maker Fringes** or
**Fresnel Coefficients** selected, the wavelength sweep and that mode's angle scan run together, and
the **Spectrum** tab shows a wavelength-by-angle map. The incident-angle field greys out there,
because the angles come from the mode's own scan range. With the sweep on, θ min may equal θ max:
the map then has a single angle and draws as a spectrum line. Without the sweep, equal values are
still an error.

A sweep uses the same settings as a run at one wavelength: the **Assumptions** panel (full
multiple reflections and its sub-mode, Jerphagnon–Kurtz, or Herman–Hayden), the fixed
**sample azimuth ψₛ (deg)**, and for Maker Fringes the **Maker Fringes Δδ (deg)**. At a single
wavelength, a map row is the Maker Fringes result for that wavelength.

Picking a crystal from the **Dispersive** group under *Case Study and Examples* fits the scan range
to what its data can answer, as on the SHAARP.si tab. With **Fresnel Coefficients** selected, the
fit starts at the table's own low end instead of twice it, because a Fresnel map reads the index at
the fundamental only. Picking one in the stack editor's layer list leaves the range as it is; take
the sweep range from the [materials table](../usage.md#materials-that-already-carry-dispersion).

A stack of materials whose index does not change with wavelength still gives a curve that moves,
through each layer's optical thickness alone, and the note under the wavelength field calls it a
thickness sweep rather than a spectrum; {doc}`../conventions` explains why.

A stack can disperse in part. A layer whose data is a single number stays at that value while the
other layers move. In the Quartz + Au preset that layer is the Au coating, and the note reads "Only
part of this stack disperses: Au coating keeps a single index at every wavelength, so the spectrum
carries the other layers' dispersion but not its own."

A thick layer needs a fine λ step. A stack's interference fringes run along the wavelength axis
too, closer together the thicker the layer. When the λ step is too coarse for them, the note says
the curve is "undersampled and can show spikes that are not spectral features" and suggests a
step. The 121.2 µm quartz plate of the Quartz + Au preset sets it off at the default step by a wide
margin: narrow the range and use the step the note gives. A 1 µm quartz film sets it off at the
default step too, narrowly, except in a Fresnel map, whose fringes along the wavelength sit about
twice as far apart. How the fringe spacing is estimated is on the {doc}`../technical_reference`
page.

Five palette films have a floor: their index data runs into an ultraviolet pole at half the
wavelength inside their own range (the list is on the
[usage page](../usage.md#materials-that-already-carry-dispersion)). At a single wavelength below the
floor, the note says any result there is not physical and the solve may fail. For the LiNbO₃ films
at 1550 nm it does fail below about 0.5 µm, and a banner says the plots still show the previous
result. A spectrum or Maker map that reaches below the floor refuses and names the window to use.

A Fresnel map reads no second harmonic, so neither the half-wavelength rule nor the floor applies to
it: it runs over the whole range, and its exported assumptions carry no SHG-tensor line.

Maps take a while. A map costs the product of its two grids, and with the default grids a Maker map
runs for tens of minutes, so **Update** asks first. For a first look, raise **λ step (µm)** to 0.1
and the Maker **θ step (deg)** to 1. That brings the default range to 11 wavelengths by 46 angles,
which runs in under a minute on the Quartz + Au preset without the question. On that preset a map
this coarse shows the envelope rather than the individual fringes, and the note will call it
undersampled; for a first look at the envelope, that is expected.

## Polarimetry settings

Incident angle $\theta_i$, incident polarization $\varphi$, ellipticity $\Delta\delta$ and the
analyzer work as on the SI tab, applied to the stack. One limit differs: the incident-angle spin box
here stops at 89.0°, where the SI tab's goes to 89.9°. The slider and the quick-angle buttons are
the same on both tabs. This tab adds **Sample rotation**.

**Rotate/fix × 3.** The polarizer, the analyzer, and the sample each carry an independent
*rotate / fix* choice, and **any of the 8 combinations is legal** — nothing is pinned. Every
element set to *rotate* follows one common scan angle $t$ (0–360°): the polarizer at
$\varphi(t)=t$, the analyzer at $\psi(t)=t+\text{offset}$ (the analyzer–polarizer offset
participates only when **both** rotate, exactly the original's gate), and the sample at
$\psi_s(t)=\pm t$ per the **rotation direction** (CW/CCW, looking at the sample from the beam
side) with the **step size** setting the grid. A fixed element holds its fixed angle. With the
sample fixed this is the ordinary polarimetry; with the sample rotating the polar RA figure is
drawn over the sample azimuth. The combinations are checked against one another in the test
suite; {doc}`../validation` has the detail.

A rotating sample is a **physical crystal rotation**, not a relabelled polarizer sweep: at each
$\psi_s$ the layer orientations are turned about the surface normal, $\varepsilon(\omega)$,
$\varepsilon(2\omega)$ and the $d$ tensor are re-derived in the lab frame, and the multilayer
boundary-value problem is solved afresh. That is what the original's `SampleRotate` does, and away
from normal incidence it does **not** reduce to co-rotating the polarizer and analyzer. The
selected assumption (FMR/JK/HH and the FMR sub-mode) applies to every point.

**Why a fine step is cheap.** Where every rotating layer's $\varepsilon$ is unchanged by a rotation
about the surface normal, the whole *linear* problem — eigenmodes, $k_z$, Fresnel coefficients,
propagation phases, the $2\omega$ boundary matrix — does not depend on $\psi_s$ at all, and only
the nonlinear source turns. Because the SHG fields are exactly linear in every $d$ component, the
sweep is then a linear combination of **one solve per touched $d$ component** (six for z-cut
quartz) instead of one solve per azimuth point. The cost is therefore flat in the number of points:
on the quartz + gold case a $0.5°$ step takes 0.7 s instead of 54 s. It is the same solver with the
same assumption support, checked against the per-point loop ({doc}`../validation`), and it falls
back to that loop automatically whenever a rotating layer's $\varepsilon$ does turn with the sample
(a rotated biaxial) or the polarizer and analyzer vary per point.

**Maker Fringes uses this panel too:** the sweep's *input* polarization ($\varphi$,
$\Delta\delta$) and *detection* polarization (analyzer $\psi$; the perpendicular channel sits at
$\psi+90°$) are the panel's fixed values — the rotate/fix selectors grey out in that mode. Fresnel
Coefficients is linear ($R_p, R_s, T_p, T_s$ per angle) and reads no polarimetry.

## Outputs

- **Maker Fringes** — $I(\theta_i)$ with the assumption shown in the subtitle, plotted straight from
  the solver.
- **Fresnel Coefficients** — $R_p, R_s, T_p, T_s$ over the *Fresnel Coefficients Scan Range*
  (0–89.9° by default). An isolated angle where the boundary solve is singular, which happens for
  metallic films, is interpolated from its neighbours. $T$ is a true power transmittance, so lossless
  stacks give $R+T=1$ whatever the substrate (see {doc}`../conventions`).
- **Polar Plots** — reflected/transmitted $I_p$, $I_s$ panels, plus a **beam-ellipticity tile**
  showing the polarization ellipses of the incident, reflected, and transmitted fundamental beams.
  The co-rotating analyzer mode ($\psi = \varphi +$ offset) is available on this tab too.
- **Spectrum** — with the wavelength sweep on. Under SHG Simulation, one curve: the reflected
  $I^{2\omega}(\lambda)$ at the analyzer angle you set, against the fundamental wavelength. Under
  Maker Fringes, two maps over $\theta_i$ and $\lambda$, one for each analyzer channel
  ($I_\parallel$, $I_\perp$). Under Fresnel Coefficients, four maps, $R_p, R_s, T_p, T_s$, on a
  shared 0–1 scale. With λ min equal to λ max a map becomes the ordinary angle scan at that
  wavelength, named once in its title; with θ min equal to θ max it becomes a spectrum line. In a
  Maker map, a channel that vanishes by symmetry, such as the perpendicular channel for p-polarized
  input on z-cut quartz, is drawn on the other channel's colour scale and its panel title carries
  "(≈ 0)". An **Update** with the sweep off empties this tab.
- **Analytical Expression** — the closed form $I(\varphi, d, h)$, typeset with real
  super/subscripts and Greek symbols like the original package; **Copy closed form (Python/SymPy)**
  and the `.txt` export stay machine-readable SymPy.

See {doc}`outputs_export`. Common questions are in the {doc}`faq`.
