# Your first calculation

Pick a published crystal and press one button. This is the quickest way to confirm your install
works.

## 1. Open the app

Double-click `SHAARP_py.exe` (Windows) or `SHAARP_py.app` (macOS). If nothing appears, give the
first launch a moment; if it still does not open, see {doc}`faq`.

The first time, it opens on the **SHAARP.si (single interface)** tab with the paper's first case
study already selected:

| Setting | Default value |
|---|---|
| Functionality | SHG Simulation |
| Case Study and Examples | GaAs (111) |
| Point group | `-43m` (the GaAs class — one independent coefficient, $d_{14}$), from the case |
| Orientation | Crystal Physics Directions, the (111) face, from the case |
| $\varepsilon(\omega)$, $\varepsilon(2\omega)$, $d$ | filled from the case |
| Incident angle $\theta_i$ | 0° |
| Wavelength | 1.064 µm |

Later launches reopen with the inputs you used last.

## 2. Check that GaAs (111) is selected

Under **Case Study and Examples** the dropdown reads **GaAs (111)**. If a previous session left
something else there, pick it again. The point group, lattice, dielectric tensors and $d$ tensor
fill in from the paper's case study.

## 3. Press **Update**

Press the blue **Update** button, top right, or **Update / Run** at the foot of the input panel;
they do the same thing. Nothing computes until you press it — that is true throughout the app.

## 4. Read the result

```{figure} ../_static/screens/first_run_result.png
:width: 100%
:alt: The SHAARP.si tab after an Update on GaAs (111) at normal incidence: the optical setup schematic above two four-lobed reflected SHG polar plots, the left one with its lobes on the diagonals and the right one with its lobes on the axes

The *Polar Plots* tab after an **Update**, with GaAs (111) at θᵢ = 0°.
```

You should see, within a few seconds:

- **Polar Plots** — two polar panels, the reflected $I^{2\omega}(\varphi)$ with the analyzer at
  $\psi = 0°$ and at $\psi = 90°$. At normal incidence each is a four-lobed pattern: along the
  diagonals at $\psi = 0°$, along the axes at $\psi = 90°$. Below them sit the effective-index and
  incident-ellipticity panels.
- **Optical setup schematic** (top) — the incident and reflected $\omega$ beams in red, the
  reflected and transmitted $2\omega$ beams dashed navy. At 0° they run along the surface normal.
- **Time Used** (bottom) — a few seconds.
- **The check-status line** (bottom) — how this solver path is checked, for example
  *"Checked: this solver path matches the original package on its reference cases."*
- The status bar reads **Run complete.**

If you got that, the install is good and every other page in this guide is now just a variation on
these steps.

## Now change one thing

Each of these is a single control followed by **Update**:

1. **A different angle** — drag the $\theta_i$ slider, or press one of the quick-angle buttons
   (0, 15, 30, 45, 60, 75). At 45° the lobes grow unequal.
2. **The closed form** — set *Functionality* to **Partial Analytical Expressions**. The result is an
   equation rather than a curve, in the *Analytical Expression* tab. The first analytical run of a
   configuration can take seconds to minutes (a computer-algebra solve); repeats are instant.
3. **Keep the numbers** — **Export data** writes the curves to a JSON file, and the closed form
   beside it as text.

```{tip}
Hover any control for a tooltip explaining it. Nothing in the app recomputes on its own: when you
change an input, a banner above the plots says so until you press **Update**.
```

Next: {doc}`interface` for the full tour of the window, or jump to {doc}`si_tab` /
{doc}`ml_tab` for what each control does.
