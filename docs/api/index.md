# API reference

The public surface of the `shaarp` package, grouped by role. Everything below is importable directly
from `shaarp` (e.g. `from shaarp import run_si_numeric`).

```{toctree}
:maxdepth: 2

facade
config
results
solvers
anisotropic
symbolic
extraction
materials
```

## At a glance

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {doc}`facade`
The GUI-mirroring entry points: `run_si_numeric`, `run_ml_numeric`, `run_maker_fringes`,
`run_fresnel_sweep`, `run_si_full_analytical`, `run_ml_partial_analytical`, plus the high-level
`single_interface_intensity` / `multilayer_shg` and the `extract_*` d-extraction.
:::

:::{grid-item-card} {doc}`config`
Input data classes: `Material`, `MultilayerSystem`, `Layer`, `CrystalStructure`,
`CrystalOrientation`, `Polarimetry`.
:::

:::{grid-item-card} {doc}`solvers`
The validated numeric SHG solvers (single-interface and multilayer boundary problems).
:::

:::{grid-item-card} {doc}`symbolic`
Closed-form / analytical SHG (symbolic in polarization, $d_{ij}$, thickness).
:::

:::{grid-item-card} {doc}`anisotropic`
Eigenmodes, Snell's law, and branch tracking in anisotropic media.
:::

:::{grid-item-card} {doc}`extraction`
Recovering the $d_{ij}$ tensor from polarimetry (`extract_si_d_voigt`, `extract_ml_film_d_voigt`).
:::
::::
