# The interface

```{figure} ../_static/screens/si_tab.png
:width: 100%
:alt: SHAARP.py main window, SHAARP.si tab

The SHAARP.py main window (SHAARP.si tab): input panel on the left, output panel on the right.
```

The window has two top-level tabs — **SHAARP.si (single interface)** and **SHAARP.ml (multilayer)** —
and, within each tab, the same overall layout:

## Layout

- **Top:** a progress bar (idle text "Ready") and the global **Update** button. Click **Update** to
  (re)compute and redraw with the current inputs.
- **Left — input panel** (scrollable): grouped, collapsible sub-panels for the material, geometry,
  polarimetry, and (on the ML tab) layer stack / assumptions / scan range. The **Crystal
  Orientation** group also shows a small live **crystal-axes ($Z_i$) vs lab-axes ($L_i$)** view.
- **Right — output panel:** a draggable splitter with three regions, top to bottom:
  1. **Optical-setup schematic** — a compact 2D banner showing the reflection/transmission geometry
     and (on ML) the multiple-reflection assumption.
  2. **Plot tabs** — *Polar Plots*, plus *Maker Fringes* / *Fresnel Coefficients* (ML), and *Guide*.
  3. **Analytical Expression** — the copyable closed-form output.
- **Bottom:** **Copy** (closed form → clipboard) and **Export** (numeric data + closed form → JSON),
  and a **Time-Used** readout.

## The Functionality dropdown

Each tab's **Functionality** dropdown selects what **Update** computes (and which output tab is
shown). It lists **compute modes only** — help is on the **Help → User Guide** menu and the startup
*Guide* tab, the optical schematic is a persistent banner, and the crystal-axes view lives in the
orientation input group. The per-tab lists are detailed in {doc}`si_tab` and {doc}`ml_tab`.

## Tooltips & help

Every control carries a tooltip taken from the original ♯SHAARP documentation — hover to read it. The
status bar shows "Hover any control for help."

## Reading the optical-setup schematic

The 2D schematic draws, at the **true** incident angle $\theta_i$ you entered:

- the **incident** $\omega$ beam (solid red, arriving at the surface),
- the **reflected** $\omega$ beam (solid red, specular — same angle $\theta_i$),
- the **reflected** $2\omega$ beam (dashed navy, collinear with the reflected $\omega$), and
- the **transmitted** $2\omega$ beam (dashed navy, refracted into the crystal).

At **normal incidence ($\theta_i = 0$)** all beams are drawn **vertical** along the surface normal; as
$\theta_i$ increases they fan out to the entered angle (45° looks like 45°, near-90° looks grazing).

Next: {doc}`si_tab`.
