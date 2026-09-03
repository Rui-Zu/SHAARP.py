[![GitHub release version](https://img.shields.io/github/v/release/Rui-Zu/SHAARP.py?color=%2350C878&include_prereleases)](https://github.com/Rui-Zu/SHAARP.py/releases)
[![License](https://img.shields.io/github/license/Rui-Zu/SHAARP.py)](https://github.com/Rui-Zu/SHAARP.py/blob/master/LICENSE)
![GitHub Size](https://img.shields.io/github/repo-size/Rui-Zu/SHAARP.py)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![CI](https://github.com/Rui-Zu/SHAARP.py/actions/workflows/ci.yml/badge.svg)](https://github.com/Rui-Zu/SHAARP.py/actions/workflows/ci.yml)
[![HitCount](https://hits.dwyl.com/Rui-Zu/SHAARP.py.svg?style=flat-square&show=unique)](http://hits.dwyl.com/Rui-Zu/SHAARP.py)

# SHAARP.py documentation

**SHAARP.py** is a validated Python reproduction of the Mathematica **♯SHAARP** package — the
*Second Harmonic Analysis of Anisotropic Rotational Polarimetry* toolset — covering both:

- **♯SHAARP.si** — single-interface **reflected** second-harmonic generation (SHG) — original
  Mathematica package: [github.com/Rui-Zu/SHAARP](https://github.com/Rui-Zu/SHAARP), and
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
residual_risks
references
```

## Start here

Pick the row that describes you — each is a complete path through these docs.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🖥️ I want to run it, not code it
**No Python needed.** Download the app, press one button, read the polar plots. Choose a
case-study crystal or type in your own, and export the curves and closed-form expressions.

**[Download](https://github.com/Rui-Zu/SHAARP.py/releases/latest)** → {doc}`guide/first_run` →
{doc}`guide/si_tab`
:::

:::{grid-item-card} 🐍 I want to script it
Call the same validated solvers the GUI calls, from your own Python: build a
{py:class}`~shaarp.Material` or {py:class}`~shaarp.MultilayerSystem`, run it, plot it, fit your own
measured data.

{doc}`usage` → {doc}`examples/index` → {doc}`api/index`
:::

:::{grid-item-card} 🔬 I want to check the physics
Governing equations, sign and Voigt conventions, what was compared against what and to what
tolerance, and the known limitations — stated, not implied.

{doc}`conventions` → {doc}`technical_reference` → {doc}`validation` → {doc}`residual_risks`
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
