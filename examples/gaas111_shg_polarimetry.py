"""GaAs(111) reflected SHG polarimetry from the FULL ANALYTICAL closed form.

GaAs is cubic -43m (one independent constant d14). This example computes the closed-form
reflected SHG polarimetry E_s/E_p(phi) -- symbolic in the input polarization angle phi with
d14 as the coefficient -- with ``shaarp.solve_si_shg_full_analytical_symbolic``, then plots:

  * the reflected SHG polarimetry I_s(phi), I_p(phi) (polar), and
  * the C3v 3-fold signature: rotating the sample azimuth by 120 deg leaves the pattern
    unchanged, while 60 deg does not.

The closed form is the one validated against the published GaAs(111) supplementary
(npj Comput. Mater. 8, 246 (2022), eq 9-32) and the numeric solver.

Run: python examples/gaas111_shg_polarimetry.py
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp

from shaarp import solve_si_shg_full_analytical_symbolic
from shaarp.config import CrystalOrientation
from shaarp.tensors import rotate_d_voigt_crystal_to_lab

N_OMEGA, N_2OMEGA, THETA = 3.4, 3.7, math.pi / 4  # GaAs-like indices, 45 deg incidence


def _gaas111_lab_d(d14):
    d = np.zeros((3, 6))
    d[0, 3] = d[1, 4] = d[2, 5] = 1.0  # -43m: d14 = d25 = d36
    ori = CrystalOrientation.from_cubic_miller_surface(np.array([1.0, 1.0, 1.0]), in_plane_hint=np.array([1.0, -1.0, 0.0]))
    geom = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d, np.asarray(ori.rotation_matrix(), dtype=float)), dtype=complex))
    return sp.Matrix(geom.tolist()) * d14


def _channel_funcs(sample_azimuth_rad=0.0):
    """Return fast numpy callables I_s(phi_deg), I_p(phi_deg) (for d14 = 1) from the closed form."""
    d14 = sp.Symbol("d14")
    phi = sp.Symbol("phi", real=True)
    sol = solve_si_shg_full_analytical_symbolic(
        eps_x_omega=N_OMEGA**2, eps_y_omega=N_OMEGA**2, eps_z_omega=N_OMEGA**2,
        eps_x_2omega=N_2OMEGA**2, eps_y_2omega=N_2OMEGA**2, eps_z_2omega=N_2OMEGA**2,
        d_voigt_lab=_gaas111_lab_d(d14), incident_theta_rad=THETA, phi_symbol=phi,
        sample_azimuth_symbol=(sp.Float(sample_azimuth_rad) if sample_azimuth_rad else None),
        omega=1, mu=1, eps0=1, simplify=False,
    )
    e_s = sp.lambdify(phi, sol.reflected_s.subs(d14, 1.0), "numpy")
    e_p = sp.lambdify(phi, sol.reflected_p.subs(d14, 1.0), "numpy")
    return (lambda deg: np.abs(e_s(np.radians(deg))) ** 2,
            lambda deg: np.abs(e_p(np.radians(deg))) ** 2)


def build_figure():
    import matplotlib.pyplot as plt

    phi = np.linspace(0.0, 360.0, 361)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), subplot_kw={"projection": "polar"})

    # --- panel 1: reflected SHG polarimetry I_s(phi), I_p(phi) at azimuth 0 ---
    i_s, i_p = _channel_funcs(0.0)
    ax1.plot(np.radians(phi), i_p(phi), color="navy", lw=2, label=r"$I_p^{2\omega}(\varphi)$")
    ax1.plot(np.radians(phi), i_s(phi), color="darkorange", lw=2, label=r"$I_s^{2\omega}(\varphi)$")
    ax1.set_title("GaAs(111) reflected SHG polarimetry\n(closed form, $\\theta_i=45°$)", fontsize=10)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10), fontsize=9)

    # --- panel 2: C3v 3-fold signature (I_p at sample azimuths) ---
    for az, color, style in [(0.0, "navy", "-"), (120.0, "crimson", "--"), (240.0, "seagreen", ":"), (60.0, "gray", "-.")]:
        _, i_p_az = _channel_funcs(math.radians(az))
        lbl = f"azimuth {int(az)}°" + ("  (≡ 0°, 3-fold)" if az in (120.0, 240.0) else ("  (60° differs)" if az == 60.0 else ""))
        ax2.plot(np.radians(phi), i_p_az(phi), color=color, lw=1.8, ls=style, label=lbl)
    ax2.set_title("C3v 3-fold: azimuth 0°/120°/240° coincide,\n60° differs (rotating the sample)", fontsize=10)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)

    fig.suptitle("GaAs(111) full-analytical SHG polarimetry  ·  shaarp.solve_si_shg_full_analytical_symbolic", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    build_figure()
    plt.show()
