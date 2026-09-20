# References

If you use SHAARP.py, please cite the two papers that introduced the methods it implements:

- **♯SHAARP.si** (single-interface reflected-SHG polarimetry) — Zu, R. *et al.*, *npj Computational
  Materials* **8**, 246 (2022),
  [doi:10.1038/s41524-022-00930-4](https://doi.org/10.1038/s41524-022-00930-4).
- **♯SHAARP.ml** (multilayer / Maker-fringe SHG) — Zu, R. *et al.*, *npj Computational Materials*
  **10**, 64 (2024),
  [doi:10.1038/s41524-024-01229-2](https://doi.org/10.1038/s41524-024-01229-2).

SHAARP stands for Second Harmonic Analysis of Anisotropic Rotational Polarimetry. Both methods
began as Mathematica packages from the same group, and SHAARP.py grew out of them:
[github.com/Rui-Zu/SHAARP](https://github.com/Rui-Zu/SHAARP) (single interface) and
[github.com/bzw133/SHAARP.ml](https://github.com/bzw133/SHAARP.ml) (multilayer).

## How SHAARP.py is checked

Every solver in SHAARP.py is checked against the published equations of the two papers and against
reference output exported from the original packages, and the published figures of both papers are
reproduced through the same compute path the desktop app uses. The linear multilayer stage is
additionally cross-checked against two outside codes, so that stage does not rest on a single
formalism:

- **`tmm`** — S. J. Byrnes, *Multilayer optical calculations*,
  [arXiv:1603.02720](https://arxiv.org/abs/1603.02720);
  [github.com/sbyrnes321/tmm](https://github.com/sbyrnes321/tmm) (MIT).
- **`inkstone`** — RCWA from the Fan group at Stanford;
  [github.com/alexysong/inkstone](https://github.com/alexysong/inkstone) (AGPL-3.0). Method
  background: Song, Catrysse and Fan, *Physical Review Letters* **120**, 193903 (2018),
  [doi:10.1103/PhysRevLett.120.193903](https://doi.org/10.1103/PhysRevLett.120.193903).
- A third, analytic reference follows M. Born and E. Wolf, *Principles of Optics*, 7th ed., §1.6, and
  H. A. Macleod, *Thin-Film Optical Filters*, 4th ed., ch. 2.

Neither package is a runtime dependency; they regenerate the stored reference numbers. The
evidence, with tolerances and the test that guards each comparison, is on {doc}`validation`. The machine-readable conventions and
validation status attached to each computation are exposed via {py:class}`~shaarp.PhysicsConventions`
and {py:class}`~shaarp.ValidationStatus` (see {doc}`conventions`).

## Published figures, reproduced

Every case figure of both papers is reproduced and shown beside the published panel in the two
notebooks {doc}`tutorials/Reproduce_SHAARP_si_paper` (SI Fig 4/5/6/7) and
{doc}`tutorials/Reproduce_SHAARP_ml_paper` (ML Fig 3/4/5/6/7 plus the supplementary twist-bilayer
MoS₂ rotational-anisotropy case, SI Fig S7). Each parameter set comes from the published case
studies (`benchmarks/paper_cases.py`), and the ML Fig-3 section also overlays the analytic
Herman–Hayden expression from the original benchmark (`benchmarks/herman_hayden_maker.py`). A
selection follows.

**♯SHAARP.si 2022, Fig. 4(b–d)** — GaAs (111) at 800 nm: flat effective complex indices and the
$I_p^{2\omega}(\varphi)$ / $I_s^{2\omega}(\varphi)$ polar patterns at $\theta_i = 0/15/30/45°$,
including the paper's observation that fitting with the real-$\varepsilon$ approximation
underestimates $d_{36}$ by about 20%.

```{figure} _static/replication/si2022_fig4_gaas111.png
:width: 100%
:alt: Replication of SHAARP.si 2022 Fig. 4(b-d)
```

**♯SHAARP.ml 2024, Fig. 3(b,c)** — 300 µm X-cut quartz Maker fringes at 1064 nm: envelope peaks,
nulls and relative heights match, including the paper's observation that the Herman–Hayden
treatment carries fine fringes at 20–30° that are absent under Jerphagnon–Kurtz.

```{figure} _static/replication/ml2024_fig3_xcut_quartz.png
:width: 100%
:alt: Replication of SHAARP.ml 2024 Fig. 3(b,c)
```

**♯SHAARP.ml 2024, Fig. 4(b,d)** — Z-cut quartz at 800 nm: the uncoated 123.6 µm slab in panel (b),
and the 121.18 µm slab with a 13.9 nm backside Au mirror in panel (d). The fine-fringe amplification
with the mirror is reproduced under the full multiple-reflection treatment, including the central
feature near $\theta_i = 0$; both panels use the paper's own display recipe.

```{figure} _static/replication/ml2024_fig4_zcut_quartz_au.png
:width: 100%
:alt: Replication of SHAARP.ml 2024 Fig. 4(b,d)
```
