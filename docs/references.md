# References & validation

SHAARP.py is a Python reproduction of the Mathematica **♯SHAARP** package. Please cite the original
SHAARP publications when using it:

- **♯SHAARP.si** (single-interface reflected-SHG polarimetry) — *npj Computational Materials*,
  s41524-022-00930-4 (2022). Original Mathematica package:
  [github.com/Rui-Zu/SHAARP](https://github.com/Rui-Zu/SHAARP).
- **♯SHAARP.ml** (multilayer / Maker-fringe SHG) — *npj Computational Materials* **10**, 64 (2024).
  Original Mathematica package: [github.com/bzw133/SHAARP.ml](https://github.com/bzw133/SHAARP.ml).

## Validation

Every solver in SHAARP.py is checked against the original Mathematica package and the published
supplementary equations:

- The reproduced multilayer Maker `MFList` matches the live ♯SHAARP.ml output to ~1×10⁻¹⁵.
- The single-interface stages are validated against the 2022 npj supplementary equations; the
  multilayer Maker sweep against the 2024 npj equations at the paper's **0.1° angular-sampling rigor**
  (Full / Jerphagnon–Kurtz / Herman–Hayden assumptions).

The full validation evidence (per-stage comparisons, benchmark artifacts, and the release gate) lives
in the repository:

- {doc}`validation` — the validation evidence table and methodology.
- `benchmarks/` — the Mathematica reference data and comparison reports.
- `notebooks/SHAARP_py_MASTER_BENCHMARK.ipynb` — the dense monitoring notebook
  (see {doc}`tutorials/index`).

The machine-readable conventions and validation status attached to each computation are exposed via
{py:class}`~shaarp.PhysicsConventions` and {py:class}`~shaarp.ValidationStatus` (see {doc}`conventions`).

## Published-figure replication

Beyond numerical agreement, the **published validation figures of both papers are replicated
end-to-end through the GUI compute path** (`scripts/replicate_paper_figures.py` — the same pure
calls the desktop **Update** button makes, with the case-study materials exported verbatim from the
original notebooks).

**Every case figure of both papers** is reproduced and shown **beside the actual paper panel** in the
two flagship notebooks {doc}`tutorials/Reproduce_SHAARP_si_paper` (SI Fig 4/5/6/7) and
{doc}`tutorials/Reproduce_SHAARP_ml_paper` (ML Fig 3/4/5/6/7 **plus the supplementary twist-bilayer
MoS₂ rotational-anisotropy case, SI Fig S7**). Each parameter set is sourced from the author's original Mathematica case studies (`benchmarks/paper_cases.py`); the ML Fig-3 section overlays
the **raw Herman-1995 analytic HH benchmark** (`benchmarks/herman_hayden_maker.py`, a byte-exact port
of `fig3/analyticHH.mx`), which SHAARP.py's HH mode tracks to a shape-correlation of **0.999**. The
reproductions are gated by `tests/test_paper_cases.py` and `tests/test_herman_hayden_benchmark.py`.
A representative selection follows.

**♯SHAARP.si 2022, Fig. 4(b–d)** — GaAs (111) at 800 nm: flat effective complex indices, the
$I_p^{2\omega}(\varphi)$ / $I_s^{2\omega}(\varphi)$ polar patterns at $\theta_i = 0/15/30/45°$, and
the panel-(e) quantitative claim (fitting with the real-$\varepsilon$ approximation underestimates
$d_{36}$ by ~20%: replicated 0.818 vs the paper's 216/267 = 0.809).

```{figure} _static/replication/si2022_fig4_gaas111.png
:width: 100%
:alt: Replication of SHAARP.si 2022 Fig. 4(b-d)
```

**♯SHAARP.ml 2024, Fig. 3(b,c)** — 300 µm X-cut quartz Maker fringes at 1064 nm: envelope peaks,
nulls, and relative heights match, including the paper's specific observation that ♯SHAARP(HH)
carries fine fringes at 20–30° that are absent for ♯SHAARP(JK).

```{figure} _static/replication/ml2024_fig3_xcut_quartz.png
:width: 100%
:alt: Replication of SHAARP.ml 2024 Fig. 3(b,c)
```

**♯SHAARP.ml 2024, Fig. 4(b,d)** — 123.6 µm Z-cut quartz at 800 nm, uncoated vs 13.9 nm backside
Au: the FMR fine-fringe amplification with the Au mirror is replicated, including the central
feature near $\theta_i = 0$, which the FMR model reproduces to correlation 0.9994 at the published spot
thickness (121.18 µm); both panels use the author's own published display recipe, and the Au
panel's HH is SHAARP.py's HH computed in the geometry the author's HH model used (quartz on an Au
half-space, beam-frame projection), matching that model to correlation 0.998. (An earlier
edition of this page attributed that central bump to the authors' journal-SI parameters. That
diagnosis came from comparing against the wrong reference curve and is retracted.)

```{figure} _static/replication/ml2024_fig4_zcut_quartz_au.png
:width: 100%
:alt: Replication of SHAARP.ml 2024 Fig. 4(b,d)
```
