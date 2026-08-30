"""End-to-end d-extraction demo -- the package's purpose: recover the SHG coefficient tensor
d_ijk from a (simulated) polarimetry scan using the closed-form analytical model.

We simulate a transmitted-SHG (Maker-geometry) polarimetry scan of a known crystal with the
numeric solver, then recover its d-tensor with ``shaarp.extract_si_d_voigt`` (the closed form
is the fit model). The figure shows (left) the measured vs fitted transmitted-SHG curve and
(right) the recovered vs true d-components -- they coincide to ~1e-9.

Run: python examples/d_extraction_demo.py
"""

from __future__ import annotations

import math

import numpy as np

from shaarp import extract_si_d_voigt
from shaarp.shg import solve_single_interface_shg
from shaarp.tensors import rotate_d_voigt_crystal_to_lab

EPS_W = [2.2**2, 2.2**2, 2.5**2]      # uniaxial (optic axis || normal): azimuth rotates only d
EPS_2W = [2.4**2, 2.4**2, 2.7**2]
POSITIONS = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 0)]
TRUE = np.array([3.0, 5.0, 6.0, 8.0, 4.0, 2.0])   # the "unknown" d-components (pm/V-ish)
GEOMETRIES = [(0.3, 0.0), (0.6, 0.0), (0.9, 0.0)]   # (theta, azimuth); 3 incidence angles, no
# azimuth -> the transmitted channel (3 observables/point) is already full-rank, and avoiding the
# symbolic sample-azimuth rotation keeps it fast (~2 s vs ~minutes).
PHIS = list(np.linspace(0.25, math.pi - 0.25, 6))


def _rz(az):
    c, s = math.cos(az), math.sin(az)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _transmitted_field(d_components, theta, az, phi):
    """Numeric total transmitted SHG field 3-vector for a crystal with the given d-components,
    sample rotated by Rz(az). (The 'instrument' in this demo.)"""
    d = np.zeros((3, 6))
    for (m, ell), v in zip(POSITIONS, d_components):
        d[m, ell] = v
    d_rot = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d, _rz(az)), dtype=complex))
    r = solve_single_interface_shg(
        np.diag(EPS_W).astype(complex), np.diag(EPS_2W).astype(complex), d_rot,
        incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
        incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=1.0, eps0=1.0,
    )
    return np.asarray(r.transmitted_fast_2omega.electric) + np.asarray(r.transmitted_slow_2omega.electric)


def run_extraction():
    """Simulate the scan with TRUE d, then recover d from it. Returns (result, fit-curve data)."""
    res = extract_si_d_voigt(
        eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
        geometries=GEOMETRIES, phi_values=PHIS,
        measure=lambda th, az, ph: _transmitted_field(TRUE, th, az, ph),
        method="field", observable="transmitted", mu=1, eps0=1,
    )
    # fit-curve data at the first geometry: measured (TRUE d) vs fitted (recovered d)
    th, az = GEOMETRIES[0]
    dense = np.linspace(0.0, math.pi, 121)
    measured = np.array([np.linalg.norm(_transmitted_field(TRUE, th, az, p)) for p in PHIS])
    fitted = np.array([np.linalg.norm(_transmitted_field(res.values, th, az, p)) for p in dense])
    return res, (np.degrees(PHIS), measured, np.degrees(dense), fitted, th, az)


def build_figure():
    import matplotlib.pyplot as plt

    res, (phi_meas, measured, phi_fit, fitted, th, az) = run_extraction()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    ax1.plot(phi_fit, fitted, color="navy", lw=2, label="fitted (recovered d)")
    ax1.plot(phi_meas, measured, "o", color="crimson", ms=7, label="measured (simulated scan)")
    ax1.set_xlabel(r"input polarization $\varphi$ (deg)")
    ax1.set_ylabel(r"$|E_t^{2\omega}|$  (transmitted SHG)")
    ax1.set_title(f"Maker-geometry polarimetry fit\n($\\theta_i$={math.degrees(th):.0f}°, azimuth {math.degrees(az):.0f}°)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    x = np.arange(len(POSITIONS))
    labels = [f"d[{m+1}{ell+1}]" for (m, ell) in POSITIONS]
    ax2.bar(x - 0.2, TRUE, width=0.4, color="seagreen", label="true d")
    ax2.bar(x + 0.2, np.real(res.values), width=0.4, color="navy", label="recovered d")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("d-component (pm/V, model units)")
    err = float(np.max(np.abs(np.real(res.values) - TRUE)))
    ax2.set_title(f"Recovered vs true d-tensor\n(identifiable={res.identifiable}, max err={err:.1e})", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("d-extraction from a simulated SHG polarimetry scan  ·  shaarp.extract_si_d_voigt", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result, _ = run_extraction()
    print("recovered d:", np.round(np.real(result.values), 4))
    print("true      d:", TRUE)
    print(f"identifiable={result.identifiable}  rank={result.rank}/{result.n_components}  "
          f"cond={result.condition_number:.1f}  residual={result.residual:.1e}")
    build_figure()
    plt.show()
