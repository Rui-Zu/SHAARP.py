[![GitHub release version](https://img.shields.io/github/v/release/Rui-Zu/SHAARP.py?color=%2350C878&include_prereleases)](https://github.com/Rui-Zu/SHAARP.py/releases)
[![License](https://img.shields.io/github/license/Rui-Zu/SHAARP.py)](https://github.com/Rui-Zu/SHAARP.py/blob/master/LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![CI](https://github.com/Rui-Zu/SHAARP.py/actions/workflows/ci.yml/badge.svg)](https://github.com/Rui-Zu/SHAARP.py/actions/workflows/ci.yml)

# SHAARP.py documentation

SHAARP.py simulates and fits optical second-harmonic generation (SHG) in anisotropic crystals and
thin-film multilayers. It is a free, open-source Python package and a standalone desktop app:
nothing else to install, and no Mathematica or other commercial license needed.

It brings the two published ♯SHAARP methods together:

- **SHAARP.si** — reflected SHG polarimetry from a single crystal surface, for any SHG-active
  point group and orientation.
- **SHAARP.ml** — SHG from a multilayer stack: Maker fringes, Fresnel curves and polarimetry, under
  the multiple-reflection treatment you choose — complete multiple reflections, Jerphagnon–Kurtz
  (single pass), or Herman–Hayden (the $2\omega$ homogeneous waves only).

On top of the two methods it adds closed-form symbolic SHG expressions, $d$-tensor extraction from
polarimetry scans, an N-layer stack editor, a palette for your own materials, and a Python API that
runs exactly what the app's Update button runs. The solvers are checked against the published
equations and the reference output of the original packages; see {doc}`validation`.

```{figure} screenshot_si.png
:width: 100%
:alt: The SHAARP.py desktop app on its single-interface tab, showing the optical setup schematic beside the reflected SHG polar plots

The desktop app: a case study picked, Update pressed, and the reflected SHG polarimetry plotted
beside a schematic of the sample.
```

## What it computes

- Reflected SHG **polarimetry** $I_s(\varphi)$, $I_p(\varphi)$ for any crystal point group and orientation.
- Multilayer **Maker fringes** $I(\theta_i)$ under the *Full multiple-reflection*, *Jerphagnon–Kurtz*
  and *Herman–Hayden* treatments.
- Linear **Fresnel** reflection and transmission coefficients.
- **Closed-form analytical** SHG expressions, symbolic in the input polarization $\varphi$, the $d_{ij}$
  tensor and the film thickness $h$.
- **$d$-tensor extraction**: recover $d_{ij}$ from a simulated or measured polarimetry scan.

## Start here

Pick the row that describes you. Each is a complete path through these docs.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} I want to run it, not code it
**No Python needed.** Download the app, press one button, read the polar plots. Choose a
case-study crystal or type in your own, and export the curves and closed-form expressions.

**[Download](https://github.com/Rui-Zu/SHAARP.py/releases/latest)**, then {doc}`guide/first_run`
and {doc}`guide/si_tab`.
:::

:::{grid-item-card} I want to script it
Call the same solvers the app calls, from your own Python: build a {py:class}`~shaarp.Material`
or {py:class}`~shaarp.MultilayerSystem`, run it, plot it, fit your own measured data.

{doc}`usage`, then {doc}`examples/index` and {doc}`api/index`.
:::

:::{grid-item-card} I want to check the physics
Governing equations, sign and Voigt conventions, what was compared against what and to what
tolerance, and the known limitations.

{doc}`conventions`, then {doc}`technical_reference` and {doc}`validation`.
:::
::::

## The ♯SHAARP family

SHAARP stands for Second Harmonic Analysis of Anisotropic Rotational Polarimetry. It is a family of
open-source packages from the same group, each adding a piece of the problem. SHAARP.py is the
current member, and it carries both earlier methods.

```{list-table}
:header-rows: 1
:widths: 14 44 20 22

* - Package
  - What it introduced
  - Released as
  - Where
* - **♯SHAARP.si**
  - Reflected SHG polarimetry from a **single interface**, solved without the usual slab and
    transparency approximations: full anisotropic eigenmodes, absorbing media, arbitrary point group
    and surface orientation, and closed-form expressions alongside the numerics.
  - Mathematica package, v1.0.0 – v1.0.3 (2022)
  - [repo](https://github.com/Rui-Zu/SHAARP) ·
    [paper](https://doi.org/10.1038/s41524-022-00930-4)
* - **♯SHAARP.ml**
  - The same physics extended to an **N-layer stack**, with complete multireflection of both the
    fundamental and the second-harmonic waves, so Maker fringes, Fresnel curves and multilayer
    polarimetry come out of one boundary-value solve.
  - Mathematica package, v1.0.0 – v1.0.2 (2024)
  - [repo](https://github.com/bzw133/SHAARP.ml) ·
    [paper](https://doi.org/10.1038/s41524-024-01229-2)
* - **SHAARP.py**
  - Both methods in one place, free of any commercial licence: a Python library, a standalone
    desktop app, symbolic closed forms, *d*-tensor extraction from a polarimetry scan, an N-layer
    stack editor, and a palette for your own materials.
  - Python package + desktop app, v1.0.0 (2026)
  - [releases](https://github.com/Rui-Zu/SHAARP.py/releases) · this site
```

The two Mathematica packages remain the reference implementations of their methods, and SHAARP.py's
solvers are checked against their published equations and reference output; {doc}`validation` says
what that covers.

If you use SHAARP.py, please cite the papers that introduced the methods:

1. Zu, R., Wang, B., He, J. *et al.* Analytical and numerical modeling of optical second harmonic
   generation in anisotropic crystals using ♯SHAARP package. *npj Comput. Mater.* **8**, 246
   (2022). [doi:10.1038/s41524-022-00930-4](https://doi.org/10.1038/s41524-022-00930-4)
2. Zu, R., Wang, B., He, J. *et al.* Optical second harmonic generation in anisotropic multilayers
   with complete multireflection of linear and nonlinear waves using ♯SHAARP.ml package.
   *npj Comput. Mater.* **10**, 64 (2024). [doi:10.1038/s41524-024-01229-2](https://doi.org/10.1038/s41524-024-01229-2)

More detail, including the figures reproduced from each paper, is on {doc}`references`.

```{toctree}
:maxdepth: 2
:caption: GUI Guide

guide/overview
guide/install_launch
guide/first_run
guide/interface
guide/si_tab
guide/ml_tab
guide/my_materials
guide/outputs_export
guide/faq
```

```{toctree}
:maxdepth: 2
:caption: Python package

usage
api/index
conventions
sample_rotation
technical_reference
```

```{toctree}
:maxdepth: 2
:caption: Tutorials & examples

tutorials/index
examples/index
```

```{toctree}
:maxdepth: 1
:caption: Project

validation
references
```
