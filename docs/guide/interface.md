# The interface

```{figure} ../_static/screens/si_tab.png
:width: 100%
:alt: SHAARP.py main window, SHAARP.si tab

The SHAARP.py main window (SHAARP.si tab): input panel on the left, output panel on the right.
```

The window has two top-level tabs — **SHAARP.si (single interface)** and **SHAARP.ml (multilayer)** —
and, within each tab, the same overall layout:

## Layout

- **Top:** a progress bar (idle text "Ready"), the global **Update** button, and four buttons for
  keeping your inputs. Click **Update** to (re)compute and redraw with the current inputs.
  - **Save Session…** writes every input on both tabs to a JSON file, and **Load Session…** reads
    one back. Attach a session file to a bug report and the problem can be reproduced exactly.
  - **Save Full Setup…** keeps the same full set of inputs in your personal collection under a
    name, and **Saved Setups…** loads or deletes one. For a single material, use the
    **My Materials** group instead ({doc}`my_materials`).
- **Left — input panel** (scrollable): grouped, collapsible sub-panels for the material, geometry,
  polarimetry, and (on the ML tab) layer stack / assumptions / scan range. The **Crystal
  Orientation** group also shows a small live **crystal-axes ($Z_i$) vs lab-axes ($L_i$)** view.
- **Right — output panel:** a draggable splitter with three regions, top to bottom:
  1. **Optical-setup schematic** — a compact 2D banner showing the reflection/transmission geometry
     and (on ML) the multiple-reflection assumption.
  2. **Plot tabs** — *Guide*, *Polar Plots*, plus *Fresnel Coefficients* / *Maker Fringes* (ML),
     *Spectrum* for a wavelength sweep, and *Analytical Expression* for the closed form.
  3. **A strip along the bottom** — the **Copy closed form** buttons, a second **Update**, the
     check-status line and the **Time Used** readout.

  When the plots no longer match the inputs, a banner above the output panel says so: after you
  change an input, and after an **Update** that was declined, refused or failed, which leaves the
  previous result on screen.
- **Foot of the input panel:** **Update / Run**, **Export data** (writes the numeric data to a JSON
  file), **Export figure** and **Copy figure**.

The app remembers where you left off. It saves your inputs after each **Update** and when you quit,
and the next launch restores them; the status bar then reads "Restored your last session (N
inputs) — press Update to recompute." The functionality always opens on SHG Simulation, so a slow
analytical run never starts by itself.

## The Functionality dropdown

Each tab's **Functionality** dropdown selects what **Update** computes (and which output tab is
shown). It lists **compute modes only** — help is under **Help**, **User Guide** and on the startup
*Guide* tab, the optical schematic is a persistent banner, and the crystal-axes view lives in the
orientation input group. The per-tab lists are detailed in {doc}`si_tab` and {doc}`ml_tab`.

## Tooltips & help

Every control carries a tooltip; hover to read it. At launch the status bar reads "Hover any control
for help (tooltips from the original SHAARP documentation)."

## Reading the optical-setup schematic

The 2D schematic draws, at the **true** incident angle $\theta_i$ you entered:

- the **incident** $\omega$ beam (solid red, arriving at the surface),
- the **reflected** $\omega$ beam (solid red, specular — same angle $\theta_i$),
- the **reflected** $2\omega$ beam (dashed navy, collinear with the reflected $\omega$), and
- the **transmitted** $2\omega$ beam (dashed navy, refracted into the crystal).

At **normal incidence ($\theta_i = 0$)** all beams are drawn **vertical** along the surface normal; as
$\theta_i$ increases they fan out to the entered angle (45° looks like 45°, near-90° looks grazing).

The beams inside the sample bend by the **material's own refractive index**, so a high-index crystal
such as GaAs ($n \approx 3.7$) draws them close to the surface normal even at a steep $\theta_i$.

Next: {doc}`si_tab`.
