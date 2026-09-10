<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/shaarp_si_logo_dark.png">
  <img src="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/shaarp_si_logo.png" alt="♯SHAARP.si" height="52">
</picture>
&nbsp;&nbsp;&nbsp;
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/shaarp_ml_logo_dark.png">
  <img src="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/shaarp_ml_logo.png" alt="♯SHAARP.ml" height="52">
</picture>

# SHAARP.py

**Optical second-harmonic generation in anisotropic crystals and multilayers.** Both ♯SHAARP methods, in one free Python library and one desktop app.

<!-- Badges MUST stay on one source line: GitHub renders a README's soft line breaks as <br>, so one
     badge per line comes out as one badge per ROW. -->
[![Release](https://img.shields.io/github/v/release/Rui-Zu/SHAARP.py?color=%2350C878&include_prereleases&label=download)](https://github.com/Rui-Zu/SHAARP.py/releases) [![Docs](https://img.shields.io/readthedocs/shaarp-py?label=docs)](https://shaarp-py.readthedocs.io/en/latest/) [![CI](https://github.com/Rui-Zu/SHAARP.py/actions/workflows/ci.yml/badge.svg)](https://github.com/Rui-Zu/SHAARP.py/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/) [![License](https://img.shields.io/github/license/Rui-Zu/SHAARP.py)](https://github.com/Rui-Zu/SHAARP.py/blob/master/LICENSE) [![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FRui-Zu%2FSHAARP.py&countColor=%23263759&style=flat)](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FRui-Zu%2FSHAARP.py&countColor=%23263759&style=flat)

</div>

SHAARP.py simulates and fits optical second-harmonic generation (SHG) in anisotropic crystals and
thin-film multilayers. It is a free, open-source Python package and a standalone desktop app:
nothing else to install, and no Mathematica or other commercial license needed.

It brings the two published ♯SHAARP methods together:

- **SHAARP.si** — reflected SHG polarimetry from a single crystal surface, for any SHG-active
  point group and orientation.
- **SHAARP.ml** — SHG from a multilayer stack: Maker fringes, Fresnel curves and polarimetry, under
  the multiple-reflection treatment you choose — complete multiple reflections, Jerphagnon–Kurtz
  (single pass), or Herman–Hayden (the 2ω homogeneous waves only).

On top of the two methods it adds closed-form symbolic SHG expressions, d-tensor extraction from
polarimetry scans, an N-layer stack editor, a palette for your own materials, and a Python API that
runs exactly what the app's Update button runs. The solvers are checked against the published
equations and the reference output of the original packages, and the test suite runs on every
commit; see [how it is tested](https://shaarp-py.readthedocs.io/en/latest/validation.html).

<!-- The showcase strip. Every tile is drawn by the app's OWN figure builders, so what a visitor
     sees here is literally what the app produces. Regenerate with scripts/make_readme_figures.py.
     Each has a dark twin; the <picture>/<source> pair is the same mechanism the logos use. -->
<table>
<tr>
<td width="33%" align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/readme/polarimetry_dark.png">
  <img src="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/readme/polarimetry.png" alt="Reflected SHG polarimetry of a 3m crystal at 45 degrees incidence: the p and s channels drawn as four-lobed polar patterns, the s channel rotated against the p channel" width="100%">
</picture>
<br><sub><b>Reflected SHG polarimetry</b><br>any point group, any orientation</sub>
</td>
<td width="33%" align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/readme/maker_dark.png">
  <img src="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/readme/maker.png" alt="Maker fringes of quartz with a backside gold mirror: a dense fringe train rising under its envelope toward 40 degrees" width="100%">
</picture>
<br><sub><b>Multilayer Maker fringes</b><br>quartz + gold, 800 nm</sub>
</td>
<td width="34%" align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/readme/fresnel_dark.png">
  <img src="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/readme/fresnel.png" alt="Linear Fresnel reflectance and transmittance of the same stack, carrying the slab's own interference" width="100%">
</picture>
<br><sub><b>Linear Fresnel curves</b><br>reflectance and transmittance</sub>
</td>
</tr>
</table>

<!-- These cards are deliberately NOT markdown headings. A `###` here would take the
     `#use-it-from-python` anchor away from the real section below, and release.yml publishes that
     anchor in every release's notes. Bold links keep the look and leave the anchors alone. -->
<table>
<tr>
<td width="33%" valign="top" align="center">

**[Download the app](https://github.com/Rui-Zu/SHAARP.py/releases/latest)**

Windows and Apple-Silicon macOS. Extract, double-click, press Update. No Python, no installation.

</td>
<td width="33%" valign="top" align="center">

**[Use it from Python](#use-it-from-python)**

One pip line, then script it, batch it, and fit your own measured data.

</td>
<td width="34%" valign="top" align="center">

**[Read the documentation](https://shaarp-py.readthedocs.io/en/latest/)**

Guide, worked examples, API reference, conventions, and how it is tested.

</td>
</tr>
</table>

---

## Run the app (no Python needed)

Pick the file for your computer. Each name below is a direct download; they are also listed on the
[Releases](https://github.com/Rui-Zu/SHAARP.py/releases) page.

| Your computer | Download | Then |
|---|---|---|
| Windows (64-bit) | [`SHAARP_py_v1.0.0_win64.zip`](https://github.com/Rui-Zu/SHAARP.py/releases/download/v1.0.0/SHAARP_py_v1.0.0_win64.zip) (115 MB) | extract the zip, then double-click `SHAARP_py\SHAARP_py.exe` |
| macOS, Apple Silicon (M-series) | [`SHAARP_py_v1.0.0_macos_arm64.zip`](https://github.com/Rui-Zu/SHAARP.py/releases/download/v1.0.0/SHAARP_py_v1.0.0_macos_arm64.zip) (79 MB) | extract the zip, then open `SHAARP_py/SHAARP_py.app` |

Extract the zip before running. The app is not code-signed, so your system will ask you to allow
it the first time, and the first launch takes a moment; if it does not open, the
[FAQ](https://shaarp-py.readthedocs.io/en/latest/guide/faq.html) has the fix. On an Intel Mac or
on Linux there is no packaged build, so use the Python route below.

<div align="center">

![The SHAARP.py app computing reflected SHG polarimetry of GaAs (111): pressing Update at normal incidence, then again at 45 degrees, with the polar lobes changing between them](https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/readme/first_run.gif)

<sub>One calculation: pick the case, press Update, read the polar plots. Then change the incidence angle and press it again.</sub>

</div>

<table>
<tr>
<td width="50%" align="center"><img src="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/screens/si_tab.png" alt="SHAARP.si tab, single-interface polarimetry"><br><sub><b>SHAARP.si tab</b>: single-interface reflected SHG</sub></td>
<td width="50%" align="center"><img src="https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/screens/ml_tab.png" alt="SHAARP.ml tab, multilayers and Maker fringes"><br><sub><b>SHAARP.ml tab</b>: multilayers and Maker fringes</sub></td>
</tr>
</table>

Your first calculation takes three clicks:

1. Under **Case Study and Examples**, pick **GaAs (111)** and press **Update / Run**. The reflected
   SHG polar plots appear next to a schematic of the sample.
2. Switch to the **SHAARP.ml** tab, set Functionality to **Maker Fringes**, and press
   **Update / Run** to see the transmitted fringes of the quartz + Au preset. It sweeps finely
   enough to draw them smoothly, so give it about a minute; raise **θ step** for a quicker look
   at the envelope.
3. Hover any control for an explanation. The
   [first-calculation guide](https://shaarp-py.readthedocs.io/en/latest/guide/first_run.html)
   takes it from here.

---

## Use it from Python

```bash
pip install "shaarp-py[desktop,interactive] @ git+https://github.com/Rui-Zu/SHAARP.py"
```

Python 3.10 or newer. SHAARP.py is not on PyPI yet, so use the whole line; it installs from
GitHub, which needs git on your PATH. New to Python, or want a lighter install? The
[installation guide](https://shaarp-py.readthedocs.io/en/latest/guide/install_launch.html) walks
through it from scratch.

Save this as `first_shg.py` and run it with `python first_shg.py`:

```python
import matplotlib.pyplot as plt

import shaarp

# Maker fringes of the paper's Fig-4 quartz + Au heterostructure (the SHAARP.ml workflow).
# compute_ml_gui_result is the app's Update button, headless: same arguments, same result.
# The step has to resolve the fringes: this slab's are 0.56 deg apart, so 0.05 draws them smoothly.
ml = shaarp.compute_ml_gui_result(
    "Maker Fringes", system_preset="Quartz + Au (Fig 4, 800 nm)",
    theta_min_deg=0.0, theta_max_deg=45.0, theta_step_deg=0.05)

# transmitted 2ω intensity (arbitrary units) vs incidence angle in degrees
plt.plot(ml.numeric["theta_deg"], ml.numeric["parallel_intensity"])
plt.xlabel("incidence angle θᵢ (deg)"); plt.ylabel("transmitted 2ω intensity (a.u.)")
plt.show()
```

The functions behind the app's Update button are the same ones you call here, so anything you can
click you can script. The [usage page](https://shaarp-py.readthedocs.io/en/latest/usage.html)
shows the rest: polarimetry closed forms, Maker fringes from your own layer stack, and d-tensor
extraction.

---

## What it does

- Reflected SHG polarimetry, *I*ₛ(φ) and *I*ₚ(φ), for any crystal point group and orientation.
- Multilayer Maker fringes *I*(θᵢ) under the Full, Jerphagnon–Kurtz and Herman–Hayden treatments.
- Linear Fresnel reflection and transmission coefficients.
- Closed-form analytical SHG expressions, symbolic in the input polarization, the *d* tensor and
  the film thickness.
- *d*-tensor extraction: recover *d*ᵢⱼ from a simulated or measured polarimetry scan.

## Where to go next

| I want to… | Go to |
|---|---|
| Install and run the app | [Installation guide](https://shaarp-py.readthedocs.io/en/latest/guide/install_launch.html) · [First calculation](https://shaarp-py.readthedocs.io/en/latest/guide/first_run.html) |
| Script the solvers | [Usage](https://shaarp-py.readthedocs.io/en/latest/usage.html) · [API reference](https://shaarp-py.readthedocs.io/en/latest/api/index.html) |
| Learn by example | [Tutorial notebooks](https://shaarp-py.readthedocs.io/en/latest/tutorials/index.html) · [Example scripts](https://shaarp-py.readthedocs.io/en/latest/examples/index.html) |
| Reproduce the papers | [the two notebooks below](#reproduce-the-papers) |
| See what is tested, and how | [How it is tested](https://shaarp-py.readthedocs.io/en/latest/validation.html) |
| Fix a problem | [FAQ](https://shaarp-py.readthedocs.io/en/latest/guide/faq.html) |
| Understand the physics | [Conventions](https://shaarp-py.readthedocs.io/en/latest/conventions.html) · [Technical reference](https://shaarp-py.readthedocs.io/en/latest/technical_reference.html) |

The full documentation lives at **[shaarp-py.readthedocs.io](https://shaarp-py.readthedocs.io/en/latest/)**.

## Reproduce the papers

Two notebooks reproduce every case figure of both papers beside the published panels:

| Notebook | Paper |
|---|---|
| [`notebooks/Reproduce_SHAARP_si_paper.ipynb`](https://github.com/Rui-Zu/SHAARP.py/blob/master/notebooks/Reproduce_SHAARP_si_paper.ipynb) | ♯SHAARP.si (2022): GaAs (111), LiNbO₃ (112̄0), KTP (100), TaAs (112) polarimetry |
| [`notebooks/Reproduce_SHAARP_ml_paper.ipynb`](https://github.com/Rui-Zu/SHAARP.py/blob/master/notebooks/Reproduce_SHAARP_ml_paper.ipynb) | ♯SHAARP.ml (2024): quartz Maker fringes, quartz + Au, LiNbO₃/KTP and ZnO / Pt / Al₂O₃ polarimetry, LiNbO₃ / quartz interference, twisted-bilayer MoS₂ |

<div align="center">

![Published SHAARP.si 2022 Fig. 4 above, SHAARP.py below, GaAs (111) at 800 nm](https://raw.githubusercontent.com/Rui-Zu/SHAARP.py/master/docs/_static/replication/si2022_fig4_gaas111_vs_paper.png)

<sub>The published figure and ours, same case, same axes: GaAs (111) at 800 nm.</sub>

</div>

## The ♯SHAARP family

SHAARP stands for Second Harmonic Analysis of Anisotropic Rotational Polarimetry. It is a family of
open-source packages from the same group, each adding a piece of the problem. SHAARP.py is the
current member, and it carries both earlier methods.

| Package | What it introduced | Released as | Where |
|---|---|---|---|
| **♯SHAARP.si** | Reflected SHG polarimetry from a **single interface**, solved without the usual slab and transparency approximations: full anisotropic eigenmodes, absorbing media, arbitrary point group and surface orientation, and closed-form expressions alongside the numerics. | Mathematica package, v1.0.0 – v1.0.3 (2022) | [repo](https://github.com/Rui-Zu/SHAARP) · [paper](https://doi.org/10.1038/s41524-022-00930-4) |
| **♯SHAARP.ml** | The same physics extended to an **N-layer stack**, with complete multireflection of both the fundamental and the second-harmonic waves, so Maker fringes, Fresnel curves and multilayer polarimetry come out of one boundary-value solve. | Mathematica package, v1.0.0 – v1.0.2 (2024) | [repo](https://github.com/bzw133/SHAARP.ml) · [paper](https://doi.org/10.1038/s41524-024-01229-2) |
| **SHAARP.py** *(this repo)* | Both methods in one place, free of any commercial licence: a Python library, a standalone desktop app, symbolic closed forms, *d*-tensor extraction from a polarimetry scan, an N-layer stack editor, and a palette for your own materials. | Python package + desktop app, v1.0.0 (2026) | [releases](https://github.com/Rui-Zu/SHAARP.py/releases) · [docs](https://shaarp-py.readthedocs.io/en/latest/) |

The two Mathematica packages remain the reference implementations of their methods, and SHAARP.py's
solvers are checked against their published equations and reference output. See
[how it is tested](https://shaarp-py.readthedocs.io/en/latest/validation.html) for what that covers.

If you use SHAARP.py, please cite the papers that introduced the methods:

1. Zu, R., Wang, B., He, J. *et al.* Analytical and numerical modeling of optical second harmonic
   generation in anisotropic crystals using ♯SHAARP package. *npj Comput. Mater.* **8**, 246
   (2022). [doi:10.1038/s41524-022-00930-4](https://doi.org/10.1038/s41524-022-00930-4)
2. Zu, R., Wang, B., He, J. *et al.* Optical second harmonic generation in anisotropic multilayers
   with complete multireflection of linear and nonlinear waves using ♯SHAARP.ml package.
   *npj Comput. Mater.* **10**, 64 (2024). [doi:10.1038/s41524-024-01229-2](https://doi.org/10.1038/s41524-024-01229-2)

## License

GNU General Public License v3 — see [LICENSE](https://github.com/Rui-Zu/SHAARP.py/blob/master/LICENSE).
