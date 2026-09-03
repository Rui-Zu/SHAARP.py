# API reference

The public surface of the `shaarp` package, grouped by role. Everything below is importable directly
from `shaarp` (e.g. `from shaarp import run_si_numeric`).

## How the layers fit together

A call enters at the top and falls through:

| Layer | Module | What it does |
|---|---|---|
| **Facade** | `shaarp.api` | `run_si_numeric`, `run_ml_numeric`, `run_maker_fringes`, `run_fresnel_sweep`, the two `*_analytical` entry points. Assembles inputs, picks a workflow, wraps the answer in a `SHAARPResult`. This is what the GUI's **Update** calls. |
| **Numeric solvers** | `shaarp.shg`, `shaarp.multilayer_shg_boundary` | The boundary-value problem itself: nonlinear source terms, homogeneous/inhomogeneous field split, continuity across each interface. |
| **Anisotropic optics** | `shaarp.anisotropic` | Eigenmodes, Snell's law in anisotropic media, ordinary/extraordinary identification, complex-branch tracking — the layer the solvers stand on. |
| **Symbolic** | `shaarp.symbolic`, `shaarp.multilayer_shg_symbolic` | The same physics carried through SymPy instead of NumPy, giving closed forms in $\varphi$, $d_{ij}$ and $h$. |
| **Inputs / outputs** | `shaarp.config`, `shaarp.api` | `Material`, `Layer`, `MultilayerSystem`, `Polarimetry` going in; `SHAARPResult` and the typed per-solver results coming back. |

Each level is usable on its own: the facades are the supported surface, but the solvers underneath
take plain tensors and radians if you would rather drive them directly (as
`examples/d_extraction_demo.py` does).

```{note}
`shaarp.__all__` exports ~158 names, most of them solver internals reached through the facades. The
pages below document the surface intended for direct use; treat anything not listed here as
internal and subject to change.
```

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
