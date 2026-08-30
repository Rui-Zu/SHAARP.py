# Tutorials

Step-by-step Jupyter notebooks (rendered with their stored outputs). They are also runnable from the
repository's `notebooks/` folder.

```{toctree}
:maxdepth: 1

Reproduce_SHAARP_si_paper
Reproduce_SHAARP_ml_paper
SHAARP_py_step_by_step
SHAARP_py_interactive_session
SHAARP_py_MASTER_BENCHMARK
```

- **Reproduce the SHAARP.si paper** — every case figure of *npj Comput. Mater.* **8**, 246 (2022)
  (GaAs (111), LiNbO₃ (112̄0), KTP (100), TaAs (112)) reproduced with SHAARP.py and shown beside the
  published panel, with parameters sourced from the author's original Mathematica case studies.
- **Reproduce the SHAARP.ml paper** — every case figure of *npj Comput. Mater.* **10**, 64 (2024)
  (quartz Maker fringes with HH/JK/FMR **+ the raw Herman-1995 analytic benchmark**, single-crystal
  and ZnO//Pt//Al₂O₃ heterostructure polarimetry, LiNbO₃//quartz interference, and the twist bilayer
  MoS₂ rotational-anisotropy case from Supplementary Fig S7).
- **Step by step** — a guided walk through a single-interface and a multilayer calculation.
- **Interactive session** — using the package interactively (building materials/systems, running the
  solvers, plotting).
- **Master benchmark** — the dense validation notebook comparing SHAARP.py against the original
  Mathematica package across cases (the monitoring artifact behind {doc}`../references`).
```{note}
Notebooks are rendered from their saved outputs (not re-executed at doc-build time). Re-run them
locally to reproduce the figures.
```
