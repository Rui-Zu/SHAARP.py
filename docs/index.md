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
- **SHAARP.ml** — SHG from a multilayer stack: Maker fringes, Fresnel curves and polarimetry with
  complete multiple reflections (Full, Jerphagnon–Kurtz and Herman–Hayden treatments).

On top of the two methods it adds closed-form symbolic SHG expressions, $d$-tensor extraction from
polarimetry scans, an N-layer stack editor, a palette for your own materials, and a Python API that
runs exactly what the app's Update button runs. The solvers are checked against the published
equations and the reference output of the original packages; see {doc}`validation`.

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
