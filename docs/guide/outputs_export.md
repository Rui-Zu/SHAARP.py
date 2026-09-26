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
- **Spectrum** — the result of a wavelength sweep. SHG Simulation gives curves against the
  fundamental wavelength: $I_s$ and $I_p$ on SI, one analyzed curve on ML. Maker Fringes and Fresnel
  Coefficients give maps over incidence angle and wavelength: two for the Maker channels
  ($I_\parallel$, $I_\perp$), four for $R_p, R_s, T_p, T_s$ on a shared 0–1 scale. A map with one
  wavelength draws as the angle scan at that wavelength, and a map with one angle as a spectrum
  line. A map or a spectrum with one wavelength names it in its title, for example "Spectral
  Response at λ = 0.8 µm". Over more than one wavelength, a caption under the title says that the
  SHG tensor is held constant; on the Fresnel maps, which involve no SHG tensor, it says only that
  the dielectric tensors disperse. An **Update** with the sweep off empties this tab, so an old
  spectrum never sits beside a newer result.
- **Analytical Expression** — the closed-form SHG expression as copyable text.
- **Guide** — the in-app help page.

## Copy & Export

- **Copy closed form (Python/SymPy)** and **Copy closed form (Mathematica)** — copy the analytical
  closed form to the clipboard, as SymPy text or in Wolfram Language syntax (an analytical run
  auto-switches to that tab). Both stay greyed until an analytical run has produced a closed form.
- **Export data** — writes the last result to JSON: the numeric data, the result `kind`, its
  check status, and under `provenance` the version, the export time and, in `inputs`, every setting
  that produced it. If you changed an input after the run, `provenance` says so: its `inputs` are
  then the current settings. For a spectrum or a map the numeric data are the wavelength grid, the
  angle grid where there is one, and the computed values, and the JSON also carries the assumptions
  the result was computed under, in a top-level `assumptions` block. For an SHG result, a spectrum
  or a Maker map, those include that the SHG tensor was held constant across the wavelength; a
  Fresnel map's assumptions say instead that it is linear optics at the fundamental. For an
  analytical run the closed form is also written beside the JSON as `.txt`, once in SymPy and once
  in Mathematica syntax. To get a spectrum or a map as CSV, run it from Python and pass
  `format="csv"` to `export_result` (see
  [the usage page](../usage.md#wavelength-against-incidence-angle)).
- **Export figure** — saves the plot on the tab you are looking at as PNG, SVG or PDF.
- **Copy figure** — copies that plot to the clipboard as an image.

## Time-Used

The output panel shows the wall-clock time of the last **Update**, so you can gauge the cost of finer
angle sweeps, wavelength sweeps or symbolic (analytical) runs.
