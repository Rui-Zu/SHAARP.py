"""CAPSTONE: extract the SHG coefficient tensor d_ijk from a (synthetic) polarimetry scan
using the closed-form analytical polarimetry -- the actual PURPOSE of the d-extraction
expression ("the expression an experimentalist fits to a polarimetry scan to extract d").

The reflected SHG field E_s/E_p is LINEAR in the d-components, so for a fixed measurement
geometry the closed form is E(phi) = sum_k d_k * B_k(phi), where the basis functions
B_k(phi) = d E / d d_k are pure functions of the scan angle. Stacking several geometries
(incidence angle + sample azimuth + input polarization phi) gives an over-determined LINEAR
system M d = y that recovers d by least squares. This test:

  1. Recovers a KNOWN six-component d-tensor to ~machine precision from synthetic measurements
     produced by the INDEPENDENT numeric arbitrary-Jones solver
     (solve_single_interface_shg(incident_jones=...)) -- i.e. the closed form really is the
     correct d-extraction expression, and the design matrix is consistent with the data
     (M @ d_true == y to ~1e-13).
  2. Documents the IDENTIFIABILITY physics honestly: a SINGLE geometry is rank-deficient
     (one polarimetry scan cannot determine the whole tensor -- here rank 4 of 6), while a
     multi-geometry scan (incidence angles x sample azimuths) restores FULL rank and recovers
     every component -- which is exactly why an azimuthal/multi-angle polarimetry experiment
     is run to determine the tensor.
  3. Is noise-robust for a well-conditioned identifiable set.

LESSON (encoded, important): use ``sympy.diff(E, d_k)`` -- NOT ``E.coeff(d_k)`` -- to read the
linear basis functions. ``.coeff`` is unreliable on the UNEXPANDED rotated-tensor expression
and silently drops terms: it made M @ d_true disagree with the data AND produced a spurious
"rank 5 of 6 zzz/zxx degeneracy" that does NOT actually exist. ``.diff`` is exact for the
degree-1-in-d form, and with it the full six-component tensor is identifiable.
"""

import math
import unittest

import numpy as np
import sympy as sp

from shaarp.shg import solve_single_interface_shg
from shaarp.symbolic import solve_si_shg_full_analytical_symbolic
from shaarp.tensors import rotate_d_voigt_crystal_to_lab

# uniaxial crystal (optic axis || surface normal z): a sample azimuth about z rotates ONLY
# the d-tensor and leaves eps invariant -> a physically consistent azimuthal scan.
EPS_W = [2.2**2, 2.2**2, 2.5**2]
EPS_2W = [2.4**2, 2.4**2, 2.7**2]
# a generic six-component d (xxx, yyy, zzz, xyz, yxz, zxx)
FULL_POSITIONS = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 0)]
FULL_VALUES = [0.30, 0.50, 0.60, 0.80, 0.40, 0.20]
MULTI_GEOMETRY = [(0.35, 0.0), (0.35, 0.6), (0.35, 1.2), (0.75, 0.0), (0.75, 0.6), (0.75, 1.2)]


def _rz(az):
    c, s = math.cos(az), math.sin(az)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _numeric_field(d_numeric, theta, az, phi_val):
    """'Measured' complex reflected SHG (E_s, E_p) at one (theta, azimuth, phi) from the
    independent numeric arbitrary-Jones solver, for a sample rotated by `az` about z."""
    d_rot = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d_numeric, _rz(az)), dtype=complex))
    r = solve_single_interface_shg(
        np.diag(EPS_W).astype(complex), np.diag(EPS_2W).astype(complex), d_rot,
        incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
        incident_jones=(math.sin(phi_val), math.cos(phi_val)), omega=1.0, mu=1.0, eps0=1.0,
    )
    c = np.asarray(r.coefficients)
    return complex(c[0]), complex(c[1])


def _matrices(positions, values, geometries, phis):
    """Build the linear d-extraction system M d = y: M from the closed-form basis functions
    B_k(phi)=dE/dd_k (exact for the degree-1 form), y from the numeric measurements."""
    syms = sp.symbols(f"d0:{len(positions)}")
    d_sym = sp.zeros(3, 6)
    for (m, ell), s in zip(positions, syms):
        d_sym[m, ell] = s
    d_num = np.zeros((3, 6))
    for (m, ell), v in zip(positions, values):
        d_num[m, ell] = v
    phi = sp.Symbol("phi", real=True)
    rows, rhs = [], []
    for theta, az in geometries:
        sol = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=EPS_W[0], eps_y_omega=EPS_W[1], eps_z_omega=EPS_W[2],
            eps_x_2omega=EPS_2W[0], eps_y_2omega=EPS_2W[1], eps_z_2omega=EPS_2W[2],
            d_voigt_lab=d_sym, incident_theta_rad=theta, phi_symbol=phi,
            sample_azimuth_symbol=sp.Float(az), omega=1, mu=1, eps0=1, simplify=False,
        )
        basis_s = [sp.diff(sol.reflected_s, s) for s in syms]  # .diff (exact), NOT .coeff
        basis_p = [sp.diff(sol.reflected_p, s) for s in syms]
        for v in phis:
            ms, mp = _numeric_field(d_num, theta, az, v)
            rows.append([complex(b.subs(phi, v).evalf()) for b in basis_s])
            rhs.append(ms)
            rows.append([complex(b.subs(phi, v).evalf()) for b in basis_p])
            rhs.append(mp)
    return np.array(rows), np.array(rhs), np.array(values, dtype=float)


class DExtractionFromPolarimetryTests(unittest.TestCase):
    def test_recovers_full_six_component_tensor_to_machine_precision(self):
        phis = np.linspace(0.25, math.pi - 0.25, 5)
        M, y, d_true = _matrices(FULL_POSITIONS, FULL_VALUES, MULTI_GEOMETRY, phis)

        self.assertGreater(np.max(np.abs(y)), 1e-4, "near-zero data -> vacuous")
        # the closed-form design matrix is CONSISTENT with the independent numeric data
        self.assertLess(np.max(np.abs(M @ d_true - y)), 1e-11, "design matrix inconsistent with data")

        d_rec, _, rank, sv = np.linalg.lstsq(M, y, rcond=None)
        self.assertEqual(rank, len(FULL_POSITIONS), "full tensor not identifiable from the multi-geometry scan")
        self.assertLess(sv[0] / sv[-1], 1e4, "design matrix ill-conditioned")
        self.assertLess(np.max(np.abs(d_rec - d_true)), 1e-9, "d not recovered from the scan")

    def test_single_geometry_cannot_identify_the_full_tensor(self):
        """Honest identifiability physics: ONE polarimetry scan (single theta + azimuth) is
        rank-deficient -- it cannot determine all six d-components, however many phi points."""
        M, _, _ = _matrices(FULL_POSITIONS, FULL_VALUES, [(0.5, 0.0)], np.linspace(0.15, math.pi - 0.15, 9))
        rank = np.linalg.matrix_rank(M, tol=1e-9)
        self.assertLess(rank, len(FULL_POSITIONS), "a single geometry unexpectedly identified the full tensor")

    def test_multi_geometry_restores_full_rank(self):
        """Stacking incidence angles x sample azimuths lifts the single-geometry deficiency to
        FULL rank -- the basis for why a multi-angle/azimuth polarimetry scan determines d."""
        M, _, _ = _matrices(FULL_POSITIONS, FULL_VALUES, MULTI_GEOMETRY, np.linspace(0.25, math.pi - 0.25, 5))
        self.assertEqual(np.linalg.matrix_rank(M, tol=1e-9), len(FULL_POSITIONS), "multi-geometry not full rank")

    def test_recovery_is_noise_robust(self):
        phis = np.linspace(0.25, math.pi - 0.25, 5)
        M, y, d_true = _matrices(FULL_POSITIONS, FULL_VALUES, MULTI_GEOMETRY, phis)
        rng = np.random.default_rng(0)
        y_noisy = y * (1 + 0.01 * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y))))
        d_rec, _, _, _ = np.linalg.lstsq(M, y_noisy, rcond=None)
        rel = np.max(np.abs(d_rec - d_true) / np.abs(d_true))
        self.assertLess(rel, 0.5, f"1% measurement noise blew up the d recovery ({rel:.1%})")


def _intensity_gram_system(positions, values, geometries, phis):
    """Build the INTENSITY d-extraction system. A real polarimetry experiment measures
    I(phi) = |E_p(phi)|^2 (intensity), not the complex field. Since E_p = sum_k d_k B_k(phi)
    is linear in d, the intensity is a quadratic form
        I = sum_{k<=l} w_kl(phi) * D_kl, D_kl = d_k d_l (real d),
        w_kk = |B_k|^2, w_{k<l} = 2 Re(B_k* B_l),
    i.e. LINEAR in the Gram matrix D. Returns (W, I_meas, pairs, d_true): solve W D = I_meas
    for the Gram entries, then rank-1 factor D = d d^T to recover d (up to overall sign)."""
    n = len(positions)
    syms = sp.symbols(f"d0:{n}")
    d_sym = sp.zeros(3, 6)
    for (m, ell), s in zip(positions, syms):
        d_sym[m, ell] = s
    d_num = np.zeros((3, 6))
    for (m, ell), v in zip(positions, values):
        d_num[m, ell] = v
    phi = sp.Symbol("phi", real=True)
    pairs = [(k, l) for k in range(n) for l in range(k, n)]
    rows, rhs = [], []
    for theta, az in geometries:
        sol = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=EPS_W[0], eps_y_omega=EPS_W[1], eps_z_omega=EPS_W[2],
            eps_x_2omega=EPS_2W[0], eps_y_2omega=EPS_2W[1], eps_z_2omega=EPS_2W[2],
            d_voigt_lab=d_sym, incident_theta_rad=theta, phi_symbol=phi,
            sample_azimuth_symbol=sp.Float(az), omega=1, mu=1, eps0=1, simplify=False,
        )
        basis = [sp.diff(sol.reflected_p, s) for s in syms]
        for v in phis:
            b = np.array([complex(expr.subs(phi, v).evalf()) for expr in basis])
            rows.append([
                (abs(b[k]) ** 2 if k == l else 2.0 * np.real(np.conj(b[k]) * b[l])) for (k, l) in pairs
            ])
            _, ep = _numeric_field(d_num, theta, az, v)  # measured |E_p|^2
            rhs.append(abs(ep) ** 2)
    return np.array(rows), np.array(rhs), pairs, np.array(values, dtype=float)


class DExtractionFromIntensityTests(unittest.TestCase):
    """The PHYSICALLY HONEST inverse: recover d from measured INTENSITY |E_p(phi)|^2 (what a
    polarimeter actually records), via the linear-in-Gram solve + rank-1 factorization."""

    def test_recovers_d_from_intensity_only(self):
        W, i_meas, pairs, d_true = _intensity_gram_system(
            FULL_POSITIONS, FULL_VALUES, MULTI_GEOMETRY, np.linspace(0.2, math.pi - 0.2, 6)
        )
        self.assertGreater(np.max(np.abs(i_meas)), 1e-6, "near-zero intensity -> vacuous")
        d_vec, _, rank, sv = np.linalg.lstsq(W, i_meas, rcond=None)
        self.assertEqual(rank, len(pairs), "Gram system not full rank")

        n = len(d_true)
        gram = np.zeros((n, n))
        for (k, l), val in zip(pairs, d_vec):
            gram[k, l] = gram[l, k] = val
        eig, vecs = np.linalg.eigh(gram)
        # the recovered Gram of a single coherent d MUST be rank-1 (consistency check)
        self.assertGreater(abs(eig[-1]) / max(abs(eig[-2]), 1e-30), 1e6, "Gram not rank-1 -> inconsistent data")
        d_rec = math.sqrt(max(eig[-1], 0.0)) * vecs[:, -1]
        if np.dot(d_rec, d_true) < 0:  # intensity is invariant under d -> -d (physical sign ambiguity)
            d_rec = -d_rec
        self.assertLess(np.max(np.abs(d_rec - d_true)), 1e-9, "d not recovered from intensity")

    def test_intensity_recovery_is_sign_ambiguous(self):
        """Intensity |E|^2 is invariant under d -> -d, so the inverse determines d only up to
        an overall sign -- documented here as a property, not a defect."""
        W1, i1, _, _ = _intensity_gram_system(FULL_POSITIONS, FULL_VALUES, [(0.5, 0.0), (0.5, 0.7)], np.linspace(0.3, 2.8, 5))
        W2, i2, _, _ = _intensity_gram_system(FULL_POSITIONS, [-v for v in FULL_VALUES], [(0.5, 0.0), (0.5, 0.7)], np.linspace(0.3, 2.8, 5))
        self.assertLess(np.max(np.abs(i1 - i2)), 1e-12, "intensity changed under d -> -d (should be invariant)")


if __name__ == "__main__":
    unittest.main()
