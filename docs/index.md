[![GitHub release version](https://img.shields.io/github/v/release/-Zu/SHAARP.py?color=%2350C878&include_prereleases)](https://github.com/-Zu/SHAARP.py/releases)
[![License](https://img.shields.io/github/license/-Zu/SHAARP.py)](https://github.com/-Zu/SHAARP.py/blob/master/LICENSE)
![GitHub Size](https://img.shields.io/github/repo-size/-Zu/SHAARP.py)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![CI](https://github.com/-Zu/SHAARP.py/actions/workflows/ci.yml/badge.svg)](https://github.com/-Zu/SHAARP.py/actions/workflows/ci.yml)
[![HitCount](https://hits.dwyl.com/-Zu/SHAARP.py.svg?style=flat-square&show=unique)](http://hits.dwyl.com/-Zu/SHAARP.py)

# SHAARP.py documentation

**SHAARP.py** is a validated Python reproduction of the Mathematica **♯SHAARP** package — the
*Second Harmonic Analysis of Anisotropic Rotational Polarimetry* toolset — covering both:

- **♯SHAARP.si** — single-interface **reflected** second-harmonic generation (SHG) — original
  Mathematica package: [github.com/Rui-Zu/SHAARP](https://github.com/-Zu/SHAARP), and
- **♯SHAARP.ml** — **multilayer** (Maker-fringe / transmitted + reflected) SHG — original
  Mathematica package: [github.com/bzw133/SHAARP.ml](https://github.com/bzw133/SHAARP.ml).

Both are merged into **one desktop application** (two tabs) and exposed as a **Python package** you can
script. Every solver is checked against the original Mathematica package and the published *npj
Computational Materials* equations (see {doc}`references`).

```{toctree}
:maxdepth: 2
:caption: GUI Guide

guide/overview
guide/install_launch
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
residual_risks
references
```

## Two ways to use SHAARP.py

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🖥️ Desktop GUI
The point-and-click app: choose a crystal or case-study material, set the geometry and polarimetry,
and read off polar plots, Maker fringes, Fresnel curves, and copyable closed-form expressions.

Start with {doc}`guide/overview` → {doc}`guide/install_launch`.
:::

:::{grid-item-card} 🐍 Python API
Call the validated solvers directly: build a {py:class}`~shaarp.Material` /
{py:class}`~shaarp.MultilayerSystem`, run a `run_*` facade, and read the
{py:class}`~shaarp.SHAARPResult`.

Start with {doc}`usage` → {doc}`api/index`.
:::
::::

## What the package computes

- Reflected SHG **polarimetry** $I_s(\varphi)$, $I_p(\varphi)$ for any crystal point group and orientation.
- Multilayer **Maker fringes** $I(\theta_i)$ under the *Full multiple-reflection*, *Jerphagnon–Kurtz*,
  and *Herman–Hayden* assumptions.
- Linear **Fresnel** reflection/transmission coefficients.
- **Closed-form analytical** SHG expressions (symbolic in input polarization $\varphi$, the $d_{ij}$
  tensor, and film thickness $h$).
- **$d$-tensor extraction** — recover $d_{ij}$ from a simulated/measured polarimetry scan.

See {doc}`conventions` for the field, Voigt, and Fresnel conventions used throughout.
