"""Dense Maker-fringe benchmark figure (SHAARP.ml), at the angular-sampling rigor of
Zu et al., npj Comput. Mater. 10, 64 (2024): transmitted-SHG incidence-angle (theta_i)
scans compared across the three assumption modes:

  * Full (mrassumption 0) -- full multiple reflections of linear + nonlinear waves,
  * HH (mrassumption 2) -- Herman-Hayden (multiple reflections of homogeneous 2w waves),
  * JK (mrassumption 1) -- Jerphagnon-Kurtz (forward-only).

Panel (a): the broad Maker-fringe pattern of a STRONG real crystal (LiNbO3 z-cut, d33,
10 um slab) at 0.1 deg step -- Full/HH oscillate from 2-omega multiple-reflection
interference while forward-only JK is smooth, the textbook assumption-mode distinction.
Panel (b): GENUINE fine fringes -- the SHAARP.ml docs' canonical THICK-slab example (Z-cut
quartz 121.2 um + Au 13.9 nm, lambda=0.8 um, p-in/p-out; ~6e3-rad phase budget -> ~0.9 deg
fringe period), VALIDATED point-by-point against live SHAARP.ml over 20-30 deg at 0.1 deg
(max err 3.6e-9 abs / 5.2e-10 rel; tests/test_quartz_au_docs_reference), resampled at
0.02 deg. Fringe-period sanity is part of the lens: a 0.16 um film (phase budget < 10 rad)
has no resolvable fringes and a 10 um slab cannot oscillate faster than ~9 deg period.

UNITS (standing lesson): all sweeps run at the CANONICAL natural units
mu=1, eps0=1 -- the units every live-Mathematica validation used. Running this config at
physical SI constants (MU0/EPS0) excites a units-scale-induced solver instability (the
transmitted field telegraphs between exact fractions E/n of the correct value at scattered
angles), which is documented as a known issue and is NOT a physical result.

Run: python examples/maker_fringes_dense.py
"""

from __future__ import annotations

import numpy as np

from shaarp import presets
from shaarp.config import CrystalOrientation, CrystalStructure, Layer, Material, MultilayerSystem, Polarimetry
from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

THICKNESS_UM = 10.0  # thin enough that 0.1 deg fully resolves the full-range fringes (~13 pts/fringe)
WAVELENGTH_UM = 1.064
MODES = [(0, "Full (FMR)", "navy", "-"), (2, "HH", "crimson", "--"), (1, "JK", "seagreen", ":")]


def _film():
    """A STRONG, allowed nonlinear slab: LiNbO3 z-cut (3m, d33=27 pm/V). NOTE: the
    earlier 'quartz-like' choice (point group 32, d11=0.3, p-in/p-out) was a near symmetry-NULL, so
    its true SHG sat at the ~1e-6 cancellation-noise floor and the 'fringes' were numerical noise.
    A benchmark figure must use a configuration whose signal is well above the noise floor."""
    return presets.linbo3_zcut_1064()


def _system():
    return MultilayerSystem(
        wavelength_um=WAVELENGTH_UM, polarimetry=Polarimetry(theta_deg=0.0, phi_deg=0.0, psi_deg=0.0),
        layers=[Layer("air in", presets.air(), shg_active=False),
                Layer("LiNbO3 z-cut", _film(), thickness_um=THICKNESS_UM, shg_active=True),
                Layer("air out", presets.air(), shg_active=False)],
    )


# `python examples/<name>.py` puts examples/ on sys.path, NOT the repo root, so the
# `benchmarks` imports below fail in a fresh clone -- which is exactly how this file's
# own docstring says to run it. Make the root importable first.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

def _fine_system():
    """The SHAARP.ml docs' canonical THICK-slab Maker example (Z-cut quartz 121.2 um + Au
    13.9 nm at lambda=0.8 um, p-in/p-out), VALIDATED point-by-point against live SHAARP.ml over
    20-30 deg at 0.1 deg (max err 3.6e-9 abs / 5.2e-10 rel; tests/test_quartz_au_docs_reference).
    The 121.2 um slab carries a ~6e3-rad phase budget -> GENUINE fine fringes (~0.9 deg period),
    the honest subject for the fine-fringe panel."""
    from benchmarks.quartz_au_docs_case import build_quartz_au_system

    return build_quartz_au_system()


def _sweep(system, theta_min, theta_max, step, mrassumption):
    # mu=1, eps0=1: the canonical natural units of every live-Mathematica validation. Physical SI
    # constants excite a scale-induced solver instability here (see module docstring).
    r = solve_multilayer_maker_fringes_sweep(
        system, theta_min_deg=theta_min, theta_max_deg=theta_max, theta_step_deg=step,
        mrassumption=mrassumption, mu=1.0, eps0=1.0,
    )
    return r.theta_deg, r.parallel_intensity


def compute(full_range=(0.0, 50.0), full_step=0.1, zoom=(24.0, 28.0), zoom_step=0.02):
    """Return {label: (theta_full, I_full, theta_zoom, I_zoom)} for each mode. Panel (a) sweeps
    the strong LiNbO3 slab; panel (b) sweeps the validated pilot configuration."""
    broad, fine = _system(), _fine_system()
    out = {}
    for mr, label, _c, _s in MODES:
        tf, If = _sweep(broad, full_range[0], full_range[1], full_step, mr)
        tz, Iz = _sweep(fine, zoom[0], zoom[1], zoom_step, mr)
        out[label] = (tf, If, tz, Iz)
    return out


def build_figure(data=None):
    import matplotlib.pyplot as plt

    data = data or compute()
    # The transmitted Maker channels are natively continuous: the eigenmode-ordering bug (which put
    # spurious exact-0 dives in the per-channel split) is fixed at the source -- the sweep selects
    # the transmitted wave by dominant amplitude, matching live SHAARP.ml. Plot the raw output.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    # Normalize each panel independently (different systems): each to its own Full-mode peak.
    norm_a = max(np.max(np.abs(data[label][1])) for label in data) or 1.0
    norm_b = max(np.max(np.abs(data[label][3])) for label in data) or 1.0
    for mr, label, color, style in MODES:
        tf, If, tz, Iz = data[label]
        ax1.plot(tf, np.abs(If) / norm_a, color=color, ls=style, lw=1.3, label=f"♯SHAARP({label})")
        ax2.plot(tz, np.abs(Iz) / norm_b, color=color, ls=style, lw=1.6, label=f"♯SHAARP({label})")
    ax1.set_xlabel(r"incidence angle $\theta_i$ (deg)")
    ax1.set_ylabel(r"transmitted SHG $|MF_\parallel|$ (norm.)")
    ax1.set_title(f"(a) Maker fringes, LiNbO₃ z-cut {THICKNESS_UM:.0f} µm slab, 0.1° step", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    ax2.set_xlabel(r"incidence angle $\theta_i$ (deg)")
    ax2.set_title("(b) genuine fine fringes, VALIDATED docs config\n(Z-cut quartz 121.2 µm + Au, 0.02° step; Full/HH vs JK)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)
    fig.suptitle("Dense Maker-fringe benchmark (♯SHAARP.ml, λ=1064 nm, p-in/p-out)  ·  paper-rigor angular sampling", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    build_figure()
    plt.show()
