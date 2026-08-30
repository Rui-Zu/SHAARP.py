# Overview

SHAARP.py is a faithful Python port of the Mathematica **♯SHAARP** package, with the two original
programs merged into a single desktop application:

| Original program | SHAARP.py tab | What it models |
|---|---|---|
| ♯SHAARP.si | **SHAARP.si (single interface)** | Reflected SHG from one air/crystal interface |
| ♯SHAARP.ml | **SHAARP.ml (multilayer)** | SHG from an N-layer thin-film stack (Maker fringes, transmitted + reflected) |

The desktop app reproduces the original GUIs' functionality, inputs, plots, and analytical-expression
output, while every numeric result is validated against the original Mathematica package (see
{doc}`../references`).

## Who it is for

Experimentalists and theorists working on **nonlinear optics** and **ferroelectric / polar materials**
who need to:

- simulate the **reflected SHG polarimetry** of a crystal of known symmetry and orientation,
- model **Maker-fringe** thickness/angle scans of a thin film,
- obtain the **closed-form** SHG expression to fit experimental polarimetry, and
- **extract the $d_{ij}$ tensor** from a measured polarimetry scan.

## How the two tabs differ

```{list-table}
:header-rows: 1

* - Aspect
  - SHAARP.si
  - SHAARP.ml
* - Geometry
  - one semi-infinite interface (air → crystal)
  - air / film(s) / substrate stack (2–N layers)
* - Primary observable
  - reflected SHG polar plots $I_s(\varphi)$, $I_p(\varphi)$
  - Maker fringes $I(\theta_i)$, Fresnel curves, polarimetry
* - Functionalities
  - SHG Simulation, Partial/Full Analytical
  - SHG Simulation, Maker Fringes, Fresnel Coefficients, Partial Analytical
* - Multiple reflections
  - n/a (single interface)
  - Full (FMR) / Jerphagnon–Kurtz / Herman–Hayden
```

Continue to {doc}`install_launch` to start the app, then {doc}`interface` for a tour of the window.
