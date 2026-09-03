# Your first calculation

Three clicks, no typing. The app opens with a complete example already loaded, so the fastest way
to confirm your install works is simply to run it.

## 1. Open the app

Double-click `SHAARP_py.exe` (Windows) or `SHAARP_py.app` (macOS). The first launch takes 10–30
seconds with nothing on screen while the system scans the bundle; later launches are quick.

It opens on the **SHAARP.si (single interface)** tab, already filled in with a default crystal:

| Setting | Default value |
|---|---|
| Functionality | SHG Simulation |
| Point group | `-43m` (the GaAs class — one independent coefficient, $d_{14}$) |
| Orientation | z-cut (identity) |
| $\varepsilon(\omega)$, $\varepsilon(2\omega)$ | 4.00, 4.84 |
| Incident angle $\theta_i$ | 45° |

## 2. Press **Update**

The blue **Update** button, top right. Nothing computes until you press it — that is true
throughout the app.

## 3. Read the result

```{figure} ../_static/screens/si_tab.png
:width: 100%
:alt: SHAARP.py after the first Update — reflected SHG polar plots for point group -43m

After **Update**: the *Polar Plots* tab holds the reflected SHG polarimetry.
```

You should see, within a few seconds:

- **Polar Plots** — $I_p^{2\omega}(\varphi)$ as a **four-lobed clover**, $I_s^{2\omega}(\varphi)$ as
  a **figure-eight**. That shape is the signature of the $-43m$ class at 45° incidence.
- **Optical setup schematic** (top) — the incident and reflected $\omega$ beams in red, the
  reflected and transmitted $2\omega$ beams dashed navy, drawn at the true 45°.
- **Time Used** (bottom) — a few seconds.
- **validation status** (bottom) — names the Mathematica-validated path the numbers came from.
- The status bar reads **Run complete.**

If you got that, the install is good and every other page in this guide is now just a variation on
these three clicks.

## Now change one thing

Each of these is a single control followed by **Update**:

1. **A real published crystal** — *Case Study and Examples* → **GaAs (111)**. The point group,
   lattice, dielectric tensors, and $d$ tensor all fill in from the paper's case study, and the
   polar pattern changes with them.
2. **A different angle** — drag the $\theta_i$ slider, or press one of the quick-angle buttons
   (0, 15, 30, 45, 60, 75).
3. **The closed form** — *Functionality* → **Partial Analytical Expression**. The result is an
   equation rather than a curve, in the *Analytical Expression* tab. The first analytical run of a
   configuration can take seconds to minutes (a computer-algebra solve); repeats are instant.
4. **Keep the numbers** — **Export** writes the curves and the closed form to a JSON file.

```{tip}
Hover any control for a tooltip taken from the original ♯SHAARP documentation. Nothing in the app
recomputes on its own — if a plot looks stale, press **Update**.
```

Next: {doc}`interface` for the full tour of the window, or jump to {doc}`si_tab` /
{doc}`ml_tab` for what each control does.
