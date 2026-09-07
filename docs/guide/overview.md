# Overview

SHAARP.py is for experimentalists and theorists working on **nonlinear optics** and
**ferroelectric / polar materials** who need to:

- simulate the **reflected SHG polarimetry** of a crystal of known symmetry and orientation,
- model **Maker-fringe** thickness and angle scans of a thin film or a layer stack,
- obtain the **closed-form** SHG expression to fit experimental polarimetry, and
- **extract the $d_{ij}$ tensor** from a measured polarimetry scan.

The desktop app puts the two SHAARP methods side by side as two tabs:

| Method | SHAARP.py tab | What it models |
|---|---|---|
| SHAARP.si | **SHAARP.si (single interface)** | Reflected SHG from one air/crystal interface |
| SHAARP.ml | **SHAARP.ml (multilayer)** | SHG from an N-layer thin-film stack (Maker fringes, transmitted + reflected) |

Every number the app shows comes from the same solvers the Python API exposes, so anything you set
up by clicking can be repeated from a script (see {doc}`../usage`).

## How the two tabs differ

```{list-table}
:header-rows: 1

* - Aspect
  - SHAARP.si
  - SHAARP.ml
* - Geometry
  - one semi-infinite interface (air over crystal)
  - air / film(s) / substrate stack (2–N layers)
* - Primary observable
  - reflected SHG polar plots $I_s(\varphi)$, $I_p(\varphi)$
  - Maker fringes $I(\theta_i)$, Fresnel curves, polarimetry
* - Functionalities
  - SHG Simulation, Partial Analytical Expressions, Full Analytical Expressions
  - SHG Simulation, Maker Fringes, Fresnel Coefficients, Partial Analytical Expressions
* - Multiple reflections
  - n/a (single interface)
  - Full (FMR) / Jerphagnon–Kurtz / Herman–Hayden
```

Both methods began as Mathematica packages from the same group; the papers and the original
repositories are listed under {doc}`../references`.

Continue to {doc}`install_launch` to start the app, then {doc}`interface` for a tour of the window.
