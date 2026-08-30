# Outputs & export

## Plot tabs

```{figure} ../_static/screens/polar.png
:width: 90%
:alt: Reflected SHG polar plots

Reflected SHG polar plots for LiNbO₃ (point group 3m) at θᵢ = 45°.
```

- **Polar Plots** — reflected (and, on ML, transmitted) SHG intensity vs polarization angle, plus the
  effective-index and ellipticity panels (SI). The radial axis starts at the centre (intensity ≥ 0);
  a symmetry-forbidden material shows a centred *"SHG ≈ 0"* note instead of a misleading circle.
- **Maker Fringes** (ML) — $I(\theta_i)$; the subtitle states the multiple-reflection assumption.
- **Fresnel Coefficients** (ML) — $R_p, R_s, T_p, T_s$ vs $\theta_i$.
- **Analytical Expression** — the closed-form SHG expression as copyable text.
- **Guide** — the in-app help page.

## Copy & Export

- **Copy** — copies the analytical closed form to the clipboard (an analytical run auto-switches to
  that tab).
- **Export** — writes the last result to **JSON**: the numeric data (curves / intensities), the result
  `kind` metadata, and — for analytical runs — the closed-form expression. This is the same payload the
  Python API returns as a {py:class}`~shaarp.SHAARPResult` (see {doc}`../usage`).

## Time-Used

The output panel shows the wall-clock time of the last **Update**, so you can gauge the cost of finer
angle sweeps or symbolic (analytical) runs.
