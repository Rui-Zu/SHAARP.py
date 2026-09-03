# Tutorials

Five Jupyter notebooks, rendered here with their stored outputs. The same files are in the
repository's `notebooks/` folder, ready to run.

## Which one should I open?

```{list-table}
:header-rows: 1
:widths: 26 40 34

* - Notebook
  - Open it if you want to…
  - Size
* - {doc}`SHAARP_py_step_by_step`
  - **Start here.** Learn the package by building one single-interface and one multilayer
    calculation from scratch, one small step at a time.
  - 19 code cells, 2 figures
* - {doc}`SHAARP_py_interactive_session`
  - Drive the solvers interactively — build materials and stacks, change a parameter, re-run,
    plot — the scripting equivalent of clicking around the GUI.
  - 8 code cells
* - {doc}`Reproduce_SHAARP_si_paper`
  - See the single-interface paper reproduced case by case, each SHAARP.py panel beside the
    published one.
  - 5 code cells, 8 figures
* - {doc}`Reproduce_SHAARP_ml_paper`
  - Same for the multilayer paper — Maker fringes, heterostructures, twist bilayers.
  - 9 code cells, 14 figures
* - {doc}`SHAARP_py_MASTER_BENCHMARK`
  - Audit the port: the dense case-by-case comparison against the original Mathematica package.
  - 6 code cells, 3 figures
```

## What each one covers

**Step by step** — a guided walk through one single-interface and one multilayer calculation:
define the crystal, set the geometry, run the solver, read the result. The gentlest entry point if
you have just installed the package.

**Interactive session** — the package used conversationally: building materials and systems,
running the solvers, and plotting, with the state kept between cells.

**Reproduce the SHAARP.si paper** — every case figure of *npj Comput. Mater.* **8**, 246 (2022),
reproduced with SHAARP.py and shown beside the published panel:

- GaAs (111)
- LiNbO₃ (112̄0)
- KTP (100)
- TaAs (112)

Parameters come from the author's original Mathematica case studies, not from re-fitted values.

**Reproduce the SHAARP.ml paper** — every case figure of *npj Comput. Mater.* **10**, 64 (2024):

- quartz Maker fringes under all three assumptions (HH / JK / FMR), plus the raw Herman-1995
  analytic benchmark
- single-crystal and ZnO//Pt//Al₂O₃ heterostructure polarimetry
- LiNbO₃//quartz interference
- the twist bilayer MoS₂ rotational-anisotropy case from Supplementary Fig S7

**Master benchmark** — the dense validation notebook comparing SHAARP.py against the original
Mathematica package across cases. This is the monitoring artifact behind {doc}`../validation` and
{doc}`../references`.

```{toctree}
:maxdepth: 1
:hidden:

Reproduce_SHAARP_si_paper
Reproduce_SHAARP_ml_paper
SHAARP_py_step_by_step
SHAARP_py_interactive_session
SHAARP_py_MASTER_BENCHMARK
```

```{note}
Notebooks are rendered from their saved outputs — they are not re-executed when these docs are
built. Re-run them locally to regenerate the figures.
```
