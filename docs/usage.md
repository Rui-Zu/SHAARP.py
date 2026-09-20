# Usage (Python API)

Everything the desktop app computes is available as a Python call. `compute_si_gui_result` and
`compute_ml_gui_result` run exactly what the app's **Update** button runs, with the same arguments
the GUI controls take; the `run_*` functions below are the solver stages underneath them, for when
you want to build the inputs yourself.

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
* - exactly what the app's Update button computes
  - `compute_si_gui_result(functionality, ...)`, `compute_ml_gui_result(functionality, ...)`
  - same arguments as the GUI controls, e.g. `system_preset="Quartz + Au (Fig 4, 800 nm)"`; see {doc}`api/facade`
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
* - a spectrum $I(\lambda)$, or a $\lambda \times \theta$ map
  - `run_si_spectrum(factory, options=…)`, `run_ml_spectrum(factory, options=…)`, `run_spectral_map(factory, options=…)`
  - pass a function of wavelength; see [Sweeping the wavelength](#sweeping-the-wavelength)
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

(sweeping-the-wavelength)=
## Sweeping the wavelength

A spectral sweep recomputes one observable at each fundamental wavelength, rebuilding the dielectric
tensors at that wavelength. Pass a factory, a function of wavelength, rather than a material or a
system, because the permittivities have to be rebuilt at every point:

```python
import matplotlib.pyplot as plt
from shaarp import run_si_spectrum
from shaarp.spectral import casestudy_spectrum

res = run_si_spectrum(
    casestudy_spectrum("Quartz z-cut (800 nm)"),
    options={"lambda_min_um": 0.6, "lambda_max_um": 1.6, "lambda_step_um": 0.02,
             "theta_deg": 45.0, "phi_deg": 30.0},
)
plt.plot(res.numeric["wavelength_um"], res.numeric["intensity_s"])
plt.xlabel(r"fundamental wavelength $\lambda$ (µm)")
plt.ylabel(r"reflected $I^{2\omega}$ (s channel)")
plt.show()
```

Quartz z-cut (800 nm) carries index data that moves with wavelength across 0.40–2.00 µm, so this is
a real spectrum. Most palette materials do not; see
[Materials that already carry dispersion](#materials-that-already-carry-dispersion).

`run_ml_spectrum` does the same for a layer stack, and
`shaarp.spectral.casestudy_ml_spectrum(name, thickness_um=…)` builds the factory for an ambient /
film / substrate stack around one palette material or one of the five crystals with published index
data. For a stack of your own, write the factory yourself. It returns the whole system at the
wavelength it is given:

```python
import matplotlib.pyplot as plt
from shaarp import Layer, MultilayerSystem, Polarimetry, presets, run_ml_spectrum
from shaarp.dispersion import dispersive_spectrum

film = dispersive_spectrum("LiNbO3 (dispersive)")   # also a factory: wavelength -> Material


def my_stack(lam):
    return MultilayerSystem(
        wavelength_um=lam,
        polarimetry=Polarimetry(theta_deg=45, phi_deg=0.0, psi_deg=0.0),
        layers=[
            Layer("air in", presets.air(), thickness_um=None),
            Layer("LiNbO3 film", film(lam), thickness_um=0.5),
            Layer("air out", presets.air(), thickness_um=None),
        ],
    )


res = run_ml_spectrum(my_stack, options={"lambda_min_um": 0.8, "lambda_max_um": 1.6,
                                         "lambda_step_um": 0.02})
plt.plot(res.numeric["wavelength_um"], res.numeric["intensity"])
plt.xlabel(r"fundamental wavelength $\lambda$ (µm)")
plt.ylabel(r"reflected $I^{2\omega}$")
plt.show()
```

Build every layer inside the function. A factory that changes only `wavelength_um` and keeps the
same materials gives a curve that moves for the wrong reason, and the sweep warns about it.

The geometry is fixed on purpose. The result is a spectrum only if each wavelength gives one
number, so the polarizer, the analyzer and the incidence angle are single values here rather than
the swept quantities they are elsewhere.

What moves, and what does not: the dielectric tensors at both the fundamental and the second
harmonic move with wavelength. The SHG tensor is held at its tabulated value, and the result says
so in `res.stages["assumptions"]`. On a layer stack the wavelength also sets each layer's optical
thickness, so a stack responds to wavelength through both; see {doc}`conventions`.

When the data cannot give a spectrum, the sweep tells you with a `RuntimeWarning`, or refuses:

- A material whose index does not change with wavelength. On a single interface the spectrum comes
  out flat. On a stack the curve still moves, but only through the optical thickness, so it is a
  thickness sweep rather than a spectrum. The warning names the material and what to use instead:
  the table-backed version of the same crystal when there is one, otherwise a material from the
  list [below](#materials-that-already-carry-dispersion) or an index table of your own. Pass
  `allow_constant_dispersion=True` in the options to leave such a material silent; the other
  warnings still come.
- A stack where only some layers disperse. A layer whose data is a single number, such as the Au
  coating in the Quartz + Au preset, stays at that value while the other layers move. The warning
  names it: "Only part of this stack disperses: Au coating keeps a single index at every
  wavelength".
- A wavelength step coarser than the stack's interference fringes. The fringes along the wavelength
  axis come closer together the thicker the layer, and a coarse step samples them into spikes that
  are not spectral features. The warning names the layer and suggests a step.
- A range that reaches where a material's index data is no longer physical. Five palette films run
  into an ultraviolet pole at half the wavelength inside their own data range (listed
  [below](#materials-that-already-carry-dispersion)). A sweep of the second harmonic that reaches
  there refuses and names the window that is usable. A Fresnel map reads no second harmonic, so it
  is exempt.

### Giving a material its own dispersion

Most palette materials carry one index at every wavelength, because the original notebooks defined
them at a single wavelength. To compute a spectrum that means something, give the material a
refractive-index table, the form an index is normally published in:

```text
wavelength_um, nx, ny, nz, kx, ky, kz
```

Two shorthands expand into the same shape: `wavelength_um, n, k` for an isotropic medium, and
`wavelength_um, no, ne, ko, ke` for a uniaxial crystal. The extinction columns are optional and
default to zero. Save these lines as `my_crystal.csv`; the numbers are made up, to show the format:

```text
wavelength_um,no,ne
0.30,1.576,1.587
0.80,1.538,1.547
1.60,1.528,1.536
```

Then build a factory from it:

```python
from shaarp.casestudy_materials import build_casestudy_material
from shaarp.dispersion import load_index_table, table_spectrum

table = load_index_table("my_crystal.csv", source="my ellipsometry, 2026")
spectrum = table_spectrum(table, build_casestudy_material("Quartz z-cut (800 nm)"))
```

`spectrum` goes to `run_si_spectrum` like any other factory. The table supplies the linear optics
only. The point group, lattice, orientation and SHG tensor come from the template material, so a
table gives an existing case its dispersion without changing anything else about it.

A table has to reach down to half your shortest wavelength. The permittivity at the second harmonic
is the table read at $\lambda/2$, so sweeping from 0.6 µm needs index data at 0.3 µm, as this one
has. A table that covers `lo` to `hi` µm can answer an SHG sweep from `2 * lo` up to `hi`, and a
Fresnel map, which reads no second harmonic, over the whole `lo` to `hi`. Past either end the index
is held at the nearest tabulated value rather than extrapolated, and the sweep says so once per
table, naming the range the table can answer. A spectrum that went quietly flat at one end could not
be told apart from physics.

### Materials that already carry dispersion

Five crystals ship with published index data. In the app they sit in the **Dispersive** group of
the material list on both tabs. From Python, pass the name to `casestudy_spectrum`,
`casestudy_ml_spectrum` or `dispersive_spectrum`. The full name ends with the range its source
covers; the short name without the range works too:

```python
from shaarp import run_si_spectrum
from shaarp.dispersion import dispersive_material_names
from shaarp.spectral import casestudy_spectrum

print(dispersive_material_names())   # the full names, each ending with its table's range

res = run_si_spectrum(
    casestudy_spectrum("LiNbO3 (dispersive)"),
    options={"lambda_min_um": 0.8, "lambda_max_um": 2.0, "lambda_step_um": 0.02,
             "theta_deg": 45.0, "phi_deg": 30.0},
)
```

| Material | Table range | Sweep range | Crystal data from | Source |
| --- | --- | --- | --- | --- |
| `LiNbO3 (dispersive)` | 0.40–5.00 µm | 0.80–5.00 µm | LiNbO₃ (11-20) MTI X-cut | Zelmon, Small & Jundt, *J. Opt. Soc. Am. B* **14**, 3319 (1997) |
| `KTP (dispersive)` | 0.43–3.54 µm | 0.86–3.54 µm | KTP (100) | Kato & Takaoka, *Appl. Opt.* **41**, 5040 (2002) |
| `LiB3O5 / LBO (dispersive)` | 0.29–1.06 µm | 0.58–1.06 µm | LiB₃O₅ (LBO) | Chen et al., *J. Opt. Soc. Am. B* **6**, 616 (1989); dispersion formula from Hanson & Dick, *Opt. Lett.* **16**, 205 (1991) |
| `GaAs (dispersive)` | 0.21–12.40 µm | 0.42–12.39 µm | GaAs (111) @1064 nm | Rakić & Majewski, *J. Appl. Phys.* **80**, 5909 (1996) |
| `TaAs (dispersive)` | 0.21–1.03 µm | 0.42–1.03 µm | TaAs (112) | Zu et al., *Phys. Rev. B* **103**, 165137 (2021) |

The table range is what the name carries. The sweep range is the part an SHG sweep can use, from
twice the table's low end, rounded inward to 0.01 µm; a Fresnel map can use the whole table range.
In the app, picking one of these under **Case Study and Examples** fits the Wavelength Scan Range
into it.

Each pairs a published index curve with the case-study crystal under *Crystal data from*, named as
the SHAARP.si case list shows it, so the
point group, orientation and SHG tensor are the ones that case already uses; only the linear optics
change. GaAs and TaAs carry an extinction coefficient as well, so their second harmonic absorbs
where the real material absorbs.

Six films of the SHAARP.ml palette also carry index data that moves with wavelength, over a
narrower window. Quartz z-cut (800 nm) has data over 0.40–2.00 µm; a sweep that goes past either
end holds the index at the end value and says so. The other five have a floor: LiNbO₃ z-cut and
x-cut (1550 nm) and KTP x-cut and y-cut are physical over 0.54–2.00 µm, and ZnO (001) over
0.42–2.00 µm. Below those floors their data runs into the ultraviolet pole, and a sweep that
reaches there refuses. Every other palette material keeps a single index at every wavelength.

### Wavelength against incidence angle

Maker fringes and Fresnel coefficients already sweep the incidence angle, so sweeping the
wavelength as well gives a map:

```python
from shaarp import export_result, run_spectral_map
from shaarp.spectral import casestudy_ml_spectrum

res = run_spectral_map(
    casestudy_ml_spectrum("Quartz z-cut (800 nm)", thickness_um=2.0),
    options={"lambda_min_um": 0.8, "lambda_max_um": 1.4, "lambda_step_um": 0.02,
             "theta_min_deg": 0.0, "theta_max_deg": 45.0, "theta_step_deg": 1.5,
             "kind": "maker"},
)
rows, cols = res.stages["shape"]
surface = res.numeric["parallel_intensity"].reshape(rows, cols)
export_result(res, "map.csv", format="csv")
```

This is 31 wavelengths by 31 angles, under a minute. The film is 2 µm thick so that a 0.02 µm step
resolves its interference fringes along the wavelength; a thicker layer needs a finer step, and
the sweep warns when the step is too coarse.

Both axes are stored expanded to the full length, so the CSV has one row per point: a wavelength
column, an angle column, then one column per channel. `export_result` writes JSON unless you pass
`format="csv"`; the JSON also keeps `stages`, including the held-constant SHG tensor, which the CSV
does not. Pass `"kind": "fresnel"` for $R_p, R_s, T_p, T_s$ instead of Maker fringes.

A map costs the product of its two grids, so keep the wavelength grid coarse and the angle grid
fine. Collapsing either axis to a single value is allowed: with one wavelength the map is an
ordinary angle scan, and with one angle it is an ordinary spectrum.

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
`solve_single_interface_shg(..., incident_jones=(sin φ, cos φ))` yourself. The tuple is ordered
$(J_s, J_p)$, so with $\varphi = 0$ meaning $p$ the sine comes first.
```

## Case-study materials

The same palette the GUI offers, with wavelength-interpolated dielectric tensors:

```python
from shaarp.casestudy_materials import GUI_ML_CASES, build_casestudy_material

print([key for _label, key in GUI_ML_CASES])   # the names build_casestudy_material accepts
mat = build_casestudy_material("LiNbO3 z-cut (1550 nm)", wavelength_um=1.55)
```

Each entry of `GUI_ML_CASES` is a pair: the label the dropdown shows, and the key you pass to
`build_casestudy_material`. They are spelled differently, so print the keys, not the labels.

`GUI_ML_CASES` is the SHAARP.ml film palette and `GUI_SI_GROUPS` the SHAARP.si case-study palette,
both at the published wavelengths. `CASE_STUDY_ORDER` is the full list of case-study materials
behind them, plus one entry kept only as a numerical fixture. The five crystals with published
index data are separate; see [Materials that already carry dispersion](#materials-that-already-carry-dispersion).

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
    measure=my_measurement_function,        # transmitted: -> (Ex, Ey, Ez); reflected: -> (Es, Ep)
    method="field",                         # or "intensity" (phase-less)
    observable="transmitted",               # or "reflected"
)
print(res.values, res.identifiable, res.rank, res.condition_number, res.residual)
```

`measure` is your data: a callable returning the measured quantity at each
$(\theta, \text{azimuth}, \varphi)$. Its return type follows `observable`: the total transmitted SHG
field as $(E_x, E_y, E_z)$ for `"transmitted"`, and the complex pair $(E_s, E_p)$ for `"reflected"`.
Build it with the same `mu` and `eps0` the extractor uses, both 1 by default; a mismatch rescales the
recovered $d$ while leaving the residual small. A complete runnable version — it simulates a scan
from a known tensor and recovers it to ~1e-9 — is `examples/d_extraction_demo.py` in
{doc}`examples/index`. Start from that file rather than from scratch.

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

- {doc}`examples/index` — seven runnable scripts, including the d-extraction demo and a
  wavelength sweep.
- {doc}`tutorials/index` — notebooks reproducing both published papers figure by figure.
- {doc}`api/index` — the full reference.
- {doc}`conventions` — frames, angles, Voigt ordering, units.
