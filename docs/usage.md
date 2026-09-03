# Usage (Python API)

Everything the desktop app computes is available as a Python call — the GUI's **Update** button
runs these same solvers.

```{admonition} Before you start
:class: note

Python ≥ 3.10, then one command:

`pip install "shaarp-py[desktop,interactive] @ git+https://github.com/Rui-Zu/SHAARP.py"`

SHAARP.py is **not on PyPI** — a bare `pip install shaarp-py` will not find it. Full instructions,
including the packaged app for people who do not want Python at all, are in
{doc}`guide/install_launch`.
```

## Which function do I call?

```{list-table}
:header-rows: 1
:widths: 34 40 26

* - I want…
  - Call
  - Notes
* - a polarimetry curve $I(\varphi)$
  - `run_si_full_analytical(case, {"workflow": "polarimetry"})`
  - validated closed form; symbolic in $\varphi$ and $d_{ij}$
* - Maker fringes $I(\theta_i)$
  - `run_maker_fringes(system, angles)`
  - validated; returns plottable arrays
* - one reflected-SHG configuration
  - `run_si_numeric(mat, {"workflow": "shaarp_si_compat", …})`
  - validated; **one** $\theta_i$, s- or p-input
* - multilayer polarimetry at one setting
  - `run_ml_numeric(system)`
  - validated
* - linear Fresnel curves
  - `run_fresnel_sweep(system, angles)`
  - $R_p, R_s, T_p, T_s$ vs $\theta_i$
* - $d_{ij}$ from a measured scan
  - `extract_si_d_voigt(...)`
  - see [d-tensor extraction](#extracting-the-d-tensor) below
* - a rough sketch, in two lines
  - `single_interface_intensity`, `multilayer_shg`
  - **reduced model** — see [the warning](#the-two-shortcut-functions)
```

All angles in the public input classes are **degrees**; the low-level solvers take **radians**.
Units and sign conventions are collected in {doc}`conventions`.

(first-plot)=
## First plot: a validated polarimetry curve

This is the closed form the GUI plots, and the expression you would fit measured data with. It is
symbolic in the input polarization $\varphi$ and in the $d_{ij}$, so you substitute your own
coefficients and evaluate.

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from shaarp import run_si_full_analytical

res = run_si_full_analytical(
    {"point_group": "3m", "incident_theta_rad": np.deg2rad(45),
     "eps_omega_principal": (4.88, 4.88, 4.55),
     "eps_2omega_principal": (5.28, 5.28, 4.87)},
    {"workflow": "polarimetry"},
)

phi = sp.Symbol("phi", real=True)
d = {"d15": -4.4, "d22": 2.5, "d31": -4.4, "d33": -27.0}          # LiNbO3, pm/V
I_s = sp.lambdify(phi, sp.Abs(sp.sympify(res.stages["reflected_s_2omega"]).subs(d))**2, "numpy")
I_p = sp.lambdify(phi, sp.Abs(sp.sympify(res.stages["reflected_p_2omega"]).subs(d))**2, "numpy")

g = np.linspace(0, 2 * np.pi, 361)
ax = plt.subplot(projection="polar")
ax.plot(g, I_s(g), label=r"$I_s^{2\omega}$")
ax.plot(g, I_p(g), label=r"$I_p^{2\omega}$")
ax.legend()
plt.show()
```

About half a second. `res.stages` also carries the intermediate derivation steps
(`deriv_1_omega`, `deriv_2_pnl`, `deriv_3_inhom`) and `analyzed_intensity` — the
$I^{2\omega}(\varphi,\psi)$ fit expression.

## Maker fringes from a layer stack

```python
import numpy as np
import matplotlib.pyplot as plt
from shaarp import Layer, MultilayerSystem, Polarimetry, presets, run_maker_fringes

system = MultilayerSystem(
    wavelength_um=1.064,
    polarimetry=Polarimetry(theta_deg=45, phi_deg=0.0, psi_deg=0.0),
    layers=[
        Layer("air in", presets.air(), thickness_um=None),
        Layer("LiNbO3", presets.linbo3_zcut_1064(), thickness_um=10.0),
        Layer("air out", presets.air(), thickness_um=None),
    ],
)

res = run_maker_fringes(system, np.arange(0.0, 45.1, 0.5))
plt.plot(res.numeric["theta_deg"], res.numeric["parallel_intensity"])
plt.xlabel(r"incidence angle $\theta_i$ (deg)")
plt.ylabel(r"transmitted $I^{2\omega}$ (parallel channel)")
plt.show()
```

`res.numeric` holds `theta_deg`, `parallel_intensity`, `perpendicular_intensity` and the two
amplitudes. First and last layers are half-spaces (`thickness_um=None`); an interior layer
radiates SHG if its point group is noncentrosymmetric.

## One reflected-SHG configuration

The validated single-interface workflow — what the GUI's SHAARP.si *SHG Simulation* runs:

```python
from shaarp import Polarimetry, presets, run_si_numeric

res = run_si_numeric(
    presets.linbo3_1120_xcut(),
    {"workflow": "shaarp_si_compat",
     "polarimetry": Polarimetry(theta_deg=45),
     "incident_polarization": "s"},          # "s" or "p"
)
print(res.numeric["reflected_intensity"])
```

```{warning}
**This workflow reads only `theta_deg` from the `Polarimetry` you pass.** `phi_deg`, `psi_deg` and
`ellipticity_deg` are ignored: `Polarimetry(theta_deg=45, phi_deg=np.linspace(0, 360, 361))`
returns **one** number here, and looping over $\varphi$ gives a flat line rather than a
polarimetry pattern. Passing any of them raises a `RuntimeWarning` saying so.

The input polarization is set by the separate `incident_polarization` option (`"s"` or `"p"`), and
the choice matters: for LiNbO₃ (11-20) at 45° the two differ by a factor of ~500.

For a curve in $\varphi$ use the closed form in [First plot](#first-plot), or drive
`solve_single_interface_shg(..., incident_jones=(sin φ, cos φ))` yourself.
```

## Case-study materials

The same palette the GUI offers, with wavelength-interpolated dielectric tensors:

```python
from shaarp.casestudy_materials import GUI_ML_CASES, build_casestudy_material

print([label for label, _key in GUI_ML_CASES])          # the names offered in the GUI
mat = build_casestudy_material("LiNbO3 z-cut (1550 nm)", wavelength_um=1.55)
```

`GUI_ML_CASES` is the curated palette (the original ♯SHAARP case studies at their published
wavelengths). `CASE_STUDY_ORDER` is the full registry, which additionally holds a few entries kept
only as numerical fixtures and deliberately not offered in the GUI.

(extracting-the-d-tensor)=
## Extracting the $d$ tensor

`extract_si_d_voigt` recovers chosen $d_{ij}$ from a polarimetry scan, using the closed form as the
fit model. It is keyword-only, and its geometry angles are in **radians**:

```python
from shaarp import extract_si_d_voigt

res = extract_si_d_voigt(
    eps_omega_principal=[2.2**2, 2.2**2, 2.5**2],
    eps_2omega_principal=[2.4**2, 2.4**2, 2.7**2],
    d_positions=[(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 0)],   # Voigt (row, col), 0-based
    geometries=[(0.3, 0.0), (0.6, 0.0), (0.9, 0.0)],                # (theta, azimuth), RADIANS
    phi_values=[0.25, 0.8, 1.4, 1.9, 2.5, 2.9],                     # input polarizations, radians
    measure=my_measurement_function,        # (theta, azimuth, phi) -> field or intensity
    method="field",                         # or "intensity" (phase-less)
    observable="transmitted",               # or "reflected"
)
print(res.values, res.identifiable, res.rank, res.condition_number, res.residual)
```

`measure` is your data: a callable returning the measured quantity at each
$(\theta, \text{azimuth}, \varphi)$. A complete runnable version — it simulates a scan from a known
tensor and recovers it to ~1e-9 — is `examples/d_extraction_demo.py` in {doc}`examples/index`.
Start from that file rather than from scratch.

Not every geometry constrains every component. The result reports `identifiable`, `rank` and
`condition_number`; if the fit is rank-deficient, add incidence angles, sample azimuths, or the
other observable channel. Noise behaviour differs sharply between `method="field"` and
`method="intensity"` — see {doc}`guide/faq`.

(the-two-shortcut-functions)=
## The two shortcut functions

`single_interface_intensity` and `multilayer_shg` are two-line conveniences returning an object you
can `.plot()` directly:

```python
import matplotlib.pyplot as plt
import numpy as np
from shaarp import Polarimetry, presets, single_interface_intensity

sample = presets.linbo3_1120_xcut()
result = single_interface_intensity(sample, Polarimetry(theta_deg=45, phi_deg=np.linspace(0, 360, 361)))
result.plot()
plt.show()
```

```{warning}
**These two are a reduced model, and say so at runtime.** They collapse the anisotropic dielectric
tensors to a single effective refractive index, and use the crystal orientation only to rotate the
tensor — not to solve anisotropic propagation directions. Both emit `RuntimeWarning`s saying
exactly that.

They are fine for a quick shape sketch and are kept for backwards compatibility. **For anything
you intend to publish or compare against a reference, use the validated routes above** — those are
the ones benchmarked against the ♯SHAARP papers and the Mathematica originals.
```

## Where to go next

- {doc}`examples/index` — six runnable scripts, including the d-extraction demo.
- {doc}`tutorials/index` — notebooks reproducing both published papers figure by figure.
- {doc}`api/index` — the full reference.
- {doc}`conventions` — frames, angles, Voigt ordering, units.
