"""Correctness checks (NOT just crash-free) for three physical inputs:
POLARIZATION, the COMPLEX permittivity tensor (eps), and the SHG d_ijk tensor.

These exercise the REAL public solvers (the same ones the GUI feeds) and assert the inputs are used
correctly -- complex parts preserved, tensor symmetry honored, point-group constraints right, and the
forward model invertible (d-extraction round-trip). Run alone:

    QT_QPA_PLATFORM=offscreen python scripts/verify_inputs_correctness.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure the working source tree wins over any stale pip-installed `shaarp` in site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from shaarp import compute_pnl_voigt, extract_si_d_voigt
from shaarp.shaarp_gui import build_constrained_d_voigt, build_custom_si_material, point_group_free_components
from shaarp.shg import solve_single_interface_shg
from shaarp.tensors import rotate_d_voigt_crystal_to_lab

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


# Voigt column mu -> (j, k) index pair (the standard contraction)
MU_JK = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (1, 2), 4: (0, 2), 5: (0, 1)}


def voigt_to_full_d(dv):
    """Expand a 3x6 Voigt d to the full 3x3x3 d_ijk (symmetric in the last two indices)."""
    full = np.zeros((3, 3, 3), dtype=complex)
    for i in range(3):
        for mu, (j, k) in MU_JK.items():
            full[i, j, k] = dv[i, mu]
            full[i, k, j] = dv[i, mu]
    return full


# =====================================================================================
# B. d_ijk TENSOR
# =====================================================================================
def b1_point_group_patterns():
    """Symmetry-constrained d for representative classes must match the textbook nonzero pattern."""
    expected = {
        "-43m": {(0, 3), (1, 4), (2, 5)},
        "4mm": {(0, 4), (1, 3), (2, 0), (2, 1), (2, 2)},                       # d15,d24,d31,d32,d33
        "mm2": {(0, 4), (1, 3), (2, 0), (2, 1), (2, 2)},                       # d15,d24,d31,d32,d33
        "32": {(0, 0), (0, 1), (0, 3), (1, 4), (1, 5)},                        # d11,d12=-d11,d14,d25=-d14,d26=-d11
    }
    # also assert EXACT equality for the trigonal classes (catches spurious extra components)
    exact = {
        "32": {(0, 0), (0, 1), (0, 3), (1, 4), (1, 5)},
        "3m": {(0, 4), (0, 5), (1, 0), (1, 1), (1, 3), (2, 0), (2, 1), (2, 2)},  # d15,d16,d21,d22,d24,d31,d32,d33
    }
    all_ok = True
    detail = []
    for pg, exp in expected.items():
        free = point_group_free_components(pg)
        d = build_constrained_d_voigt(pg, {(r, c): float(i + 1) for i, (r, c, _n) in enumerate(free)})
        nz = {(r, c) for r in range(3) for c in range(6) if abs(complex(d[r, c])) > 1e-12}
        ok = exp.issubset(nz)
        if pg in exact:
            ok = ok and (nz == exact[pg])  # no missing AND no spurious extra components
        all_ok = all_ok and ok
        detail.append(f"{pg}:{'ok' if ok else f'got {sorted(nz)}'}")
    check("B1 point-group d patterns (representative classes)", all_ok, "; ".join(detail))


def b2_full_tensor_jk_symmetry():
    """The expanded full d_ijk must obey the intrinsic SHG permutation symmetry d_ijk == d_ikj."""
    d = build_constrained_d_voigt("3m", {(r, c): complex(i + 1, 0.3 * (i + 1))
                                         for i, (r, c, _n) in enumerate(point_group_free_components("3m"))})
    full = voigt_to_full_d(np.asarray(d, dtype=complex))
    asym = float(np.max(np.abs(full - np.transpose(full, (0, 2, 1)))))
    check("B2 full d_ijk is jk-symmetric (d_ijk == d_ikj)", asym < 1e-12, f"max|d_ijk - d_ikj| = {asym:.2e}")


def b3_pnl_contraction():
    """compute_pnl_voigt must equal the full-tensor contraction P_i = sum_jk d_ijk E_j E_k."""
    rng = np.random.default_rng(0)
    dv = (rng.standard_normal((3, 6)) + 1j * rng.standard_normal((3, 6)))
    E = (rng.standard_normal(3) + 1j * rng.standard_normal(3))
    p_voigt = np.asarray(compute_pnl_voigt(dv, E, E), dtype=complex)
    full = voigt_to_full_d(dv)
    p_full = np.einsum("ijk,j,k->i", full, E, E)
    err = float(np.max(np.abs(p_voigt - p_full)))
    check("B3 P_NL Voigt contraction == full d_ijk:EE", err < 1e-10, f"max|dP| = {err:.2e}")


def b4_complex_d_passthrough():
    """The GUI custom-material builder must preserve a COMPLEX d tensor (not drop the imaginary part)."""
    d_full = np.zeros((3, 6), dtype=complex)
    d_full[0, 3] = d_full[1, 4] = d_full[2, 5] = 2.5 - 1.3j     # complex d14=d25=d36
    mat = build_custom_si_material("-43m", n_omega=2.0, n_2omega=2.2, d_full=d_full)
    got = np.asarray(mat.d_voigt_pm_v, dtype=complex)
    ok = abs(got[0, 3] - (2.5 - 1.3j)) < 1e-9 and abs(got[1, 4] - (2.5 - 1.3j)) < 1e-9
    check("B4 complex d preserved through GUI material builder", ok, f"d14 stored = {got[0,3]}")


# =====================================================================================
# A. COMPLEX PERMITTIVITY TENSOR (eps)
# =====================================================================================
def a1_complex_symmetric_passthrough():
    """A complex SYMMETRIC eps entered in the GUI full-matrix fields must round-trip unchanged."""
    eps = np.array([[5.0 + 0.2j, 0.3 + 0.1j, 0.0],
                    [0.3 + 0.1j, 5.2 + 0.25j, 0.0],
                    [0.0, 0.0, 5.5 + 0.3j]], dtype=complex)
    eps2 = eps * 1.1
    mat = build_custom_si_material("3m", eps_omega_full=eps, eps_2omega_full=eps2, n_omega=2.0, n_2omega=2.2)
    got = np.asarray(mat.epsilon_omega, dtype=complex)
    preserved = np.max(np.abs(got - eps)) < 1e-9
    symmetric = np.max(np.abs(got - got.T)) < 1e-12
    check("A1 complex symmetric eps preserved + symmetric through builder", preserved and symmetric,
          f"max|deps|={np.max(np.abs(got-eps)):.2e}, asym={np.max(np.abs(got-got.T)):.2e}")


def _shg_trans_norm(eps_w, eps_2w, d, theta=0.5, phi=0.7):
    r = solve_single_interface_shg(
        np.asarray(eps_w, complex), np.asarray(eps_2w, complex), np.asarray(d, complex),
        incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
        incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=1.0, eps0=1.0,
    )
    e = np.asarray(r.transmitted_fast_2omega.electric) + np.asarray(r.transmitted_slow_2omega.electric)
    return float(np.linalg.norm(e))


def a2_imag_eps_is_used():
    """Adding loss (Im eps > 0) must change the SHG result -- i.e. the imaginary part is actually used."""
    d = np.zeros((3, 6)); d[2, 2] = 5.0; d[2, 0] = d[2, 1] = 2.0; d[0, 4] = d[1, 3] = 3.0  # mm2-like
    lossless = _shg_trans_norm(np.diag([4.0, 4.2, 4.5]), np.diag([4.6, 4.8, 5.1]), d)
    lossy = _shg_trans_norm(np.diag([4.0 + 0.6j, 4.2 + 0.6j, 4.5 + 0.6j]),
                            np.diag([4.6 + 0.6j, 4.8 + 0.6j, 5.1 + 0.6j]), d)
    rel = abs(lossless - lossy) / max(lossless, 1e-30)
    check("A2 Im(eps) changes the result (loss is used)", rel > 1e-3,
          f"|lossless-lossy|/lossless = {rel:.3e}")


def a3_offdiagonal_eps_is_used():
    """Off-diagonal eps must change the result -- i.e. the FULL 3x3 tensor is used, not just the diagonal."""
    d = np.zeros((3, 6)); d[2, 2] = 5.0; d[2, 0] = d[2, 1] = 2.0; d[0, 4] = d[1, 3] = 3.0
    diag = _shg_trans_norm(np.diag([4.0, 4.2, 4.5]), np.diag([4.6, 4.8, 5.1]), d)
    off_w = np.array([[4.0, 0.5, 0.0], [0.5, 4.2, 0.0], [0.0, 0.0, 4.5]], complex)
    off = _shg_trans_norm(off_w, np.diag([4.6, 4.8, 5.1]), d)
    rel = abs(diag - off) / max(diag, 1e-30)
    check("A3 off-diagonal eps changes the result (full tensor used)", rel > 1e-3,
          f"|diag-offdiag|/diag = {rel:.3e}")


# =====================================================================================
# C. POLARIZATION
# =====================================================================================
def c1_s_vs_p_differ():
    """s-polarized vs p-polarized input must give different SHG (polarization actually drives the field)."""
    d = np.zeros((3, 6)); d[2, 2] = 5.0; d[2, 0] = d[2, 1] = 2.0; d[0, 4] = d[1, 3] = 3.0
    ew, e2w = np.diag([4.0, 4.2, 4.5]).astype(complex), np.diag([4.6, 4.8, 5.1]).astype(complex)

    def trans(js, jp):
        r = solve_single_interface_shg(ew, e2w, d, incident_index_omega=1.0, incident_index_2omega=1.0,
                                       incident_theta_rad=0.5, incident_jones=(js, jp), omega=1.0, mu=1.0, eps0=1.0)
        return np.asarray(r.transmitted_fast_2omega.electric) + np.asarray(r.transmitted_slow_2omega.electric)
    s_only = np.linalg.norm(trans(1.0, 0.0))
    p_only = np.linalg.norm(trans(0.0, 1.0))
    check("C1 s-pol vs p-pol input give distinct SHG", abs(s_only - p_only) / max(s_only + p_only, 1e-30) > 1e-3,
          f"|Es|={s_only:.4g}, |Ep|={p_only:.4g}")


def c2_d_extraction_roundtrip():
    """THE end-to-end correctness test: simulate a polarimetry scan (varying phi -> incident_jones) of a
    known crystal+d, then recover d. Only succeeds if polarization, eps, and d are ALL applied correctly."""
    EPS_W = [2.2 ** 2, 2.2 ** 2, 2.5 ** 2]
    EPS_2W = [2.4 ** 2, 2.4 ** 2, 2.7 ** 2]
    POSITIONS = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 0)]
    TRUE = np.array([3.0, 5.0, 6.0, 8.0, 4.0, 2.0])
    GEOMS = [(0.3, 0.0), (0.6, 0.0), (0.9, 0.0)]
    PHIS = list(np.linspace(0.25, math.pi - 0.25, 6))

    def trans_field(dc, theta, az, phi):
        d = np.zeros((3, 6))
        for (m, ell), v in zip(POSITIONS, dc):
            d[m, ell] = v
        r = solve_single_interface_shg(
            np.diag(EPS_W).astype(complex), np.diag(EPS_2W).astype(complex), d,
            incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
            incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=1.0, eps0=1.0)
        return np.asarray(r.transmitted_fast_2omega.electric) + np.asarray(r.transmitted_slow_2omega.electric)

    res = extract_si_d_voigt(eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
                             geometries=GEOMS, phi_values=PHIS,
                             measure=lambda th, az, ph: trans_field(TRUE, th, az, ph),
                             method="field", observable="transmitted", mu=1, eps0=1)
    rec = np.asarray(res.values, dtype=float)
    err = float(np.max(np.abs(rec - TRUE)))
    check("C2 d-extraction round-trip recovers TRUE d (polarization+eps+d end-to-end)", err < 1e-6,
          f"max|recovered - true| = {err:.2e}")


if __name__ == "__main__":
    print("=== INPUT CORRECTNESS: polarization, complex eps, d_ijk ===\n")
    b1_point_group_patterns()
    b2_full_tensor_jk_symmetry()
    b3_pnl_contraction()
    b4_complex_d_passthrough()
    a1_complex_symmetric_passthrough()
    a2_imag_eps_is_used()
    a3_offdiagonal_eps_is_used()
    c1_s_vs_p_differ()
    c2_d_extraction_roundtrip()
    n_fail = sum(1 for _n, ok, _d in results if not ok)
    print(f"\n=== {len(results)-n_fail}/{len(results)} checks passed ===")
    raise SystemExit(1 if n_fail else 0)
