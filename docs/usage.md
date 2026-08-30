# Usage (Python API)

SHAARP.py can be scripted directly — the desktop GUI's **Update** button simply calls the same solvers
documented here. There are two layers:

- **High-level convenience functions** — {py:func}`~shaarp.single_interface_intensity`,
  {py:func}`~shaarp.multilayer_shg` — take a sample / system + a {py:class}`~shaarp.Polarimetry` and
  return a result you can `.plot()`.
- **Facades** mirroring the GUI — {py:func}`~shaarp.run_si_numeric`, {py:func}`~shaarp.run_ml_numeric`,
  {py:func}`~shaarp.run_maker_fringes`, {py:func}`~shaarp.run_fresnel_sweep`,
  {py:func}`~shaarp.run_si_full_analytical`, {py:func}`~shaarp.run_ml_partial_analytical` — return a
  {py:class}`~shaarp.SHAARPResult` (the same payload the GUI exports as JSON).

```{warning}
**The two convenience functions are a reduced model, and they say so at runtime.** Calling
`single_interface_intensity` emits two `RuntimeWarning`s:

> Reduced model warning: anisotropic dielectric tensors are collapsed to one effective refractive
> index. This is not the full SHAARP anisotropic eigenmode and boundary-condition solver.

> Reduced model warning: crystal orientation is used for tensor rotation, but not for solving
> anisotropic propagation directions.

They are kept for quick sketches and backwards compatibility. **The validated solvers — the ones
benchmarked against the ♯SHAARP papers and the Mathematica originals, and the ones the GUI's
Update button calls — are the `run_*` facades.** Use those for anything you intend to publish or
compare against a reference — they are the `run_*` entries listed just above.
```

## Single-interface reflected SHG

```python
import numpy as np
import matplotlib.pyplot as plt
from shaarp import Polarimetry, presets, single_interface_intensity

sample = presets.linbo3_1120_xcut() # a built-in crystal preset
polarimetry = Polarimetry(theta_deg=45,
                          phi_deg=np.linspace(0, 360, 361), # rotate the input polarization
                          psi_deg=0)
result = single_interface_intensity(sample, polarimetry)
result.plot() # reflected SHG polar plot
plt.show()
```

## Multilayer (Maker fringes / thin film)

```python
import numpy as np
import matplotlib.pyplot as plt
from shaarp import Layer, MultilayerSystem, Polarimetry, multilayer_shg, presets

system = MultilayerSystem(
    wavelength_um=1.064,
    polarimetry=Polarimetry(theta_deg=45, phi_deg=np.linspace(0, 360, 181), psi_deg=0),
    layers=[
        Layer("air in", presets.air(), thickness_um=None, shg_active=False),
        Layer("LNO", presets.linbo3_zcut_1064(), thickness_um=1.0, shg_active=True),
        Layer("air out", presets.air(), thickness_um=None, shg_active=False),
    ],
)
result = multilayer_shg(system)
result.plot()
plt.show()
```

## Case-study materials

The same materials offered in the GUI dropdown are available programmatically with
wavelength-interpolated dielectric tensors:

```python
from shaarp.casestudy_materials import GUI_ML_CASES, build_casestudy_material

print([label for label, _key in GUI_ML_CASES]) # the names offered in the GUI
mat = build_casestudy_material("LiNbO3 z-cut (1550 nm)", wavelength_um=1.55)
```

`GUI_ML_CASES` is the curated palette (the original ♯SHAARP case studies, each at its published
wavelength). `CASE_STUDY_ORDER` is the full registry: it additionally contains a few entries kept
only as numerical fixtures, which the GUI deliberately does not offer.

## Extracting the d tensor from a polarimetry scan

```python
from shaarp import extract_si_d_voigt
# given measured/simulated intensity vs polarization angle, recover the d_ij components:
result = extract_si_d_voigt(...) # see the API reference for the exact signature & options
print(result.d_voigt) # recovered tensor + identifiability diagnostics
```

See {doc}`api/index` for the full reference and {doc}`conventions` for the field / Voigt / Fresnel
conventions. Runnable scripts are in {doc}`examples/index`; step-by-step notebooks in
{doc}`tutorials/index`.
