"""Rotated-biaxial d-extraction via the NUMERIC basis -- the case the symbolic basis cannot do.

A sample azimuth about the surface normal physically rotates BOTH the d-tensor AND the
dielectric tensor. For a biaxial crystal (eps_x != eps_y) eps is NOT azimuth-invariant, so a
correct azimuth-scan extractor must rotate eps too. ``extract_si_d_voigt(basis="numeric")``
rotates eps with the SAME Rz convention as d (``rotate_rank2_crystal_to_lab``); this test proves
it recovers the full six-component biaxial tensor from an AZIMUTH-varying scan, and that the eps
rotation is necessary (omitting it -- the prior d-only-rotation bug -- changes the response, so
the fix is non-vacuous).

This closes the documented rotated-biaxial gap: the symbolic SI closed form is impractical for a
general rotated biaxial (full Booker quartic), but the numeric-basis extractor handles it exactly.
"""

import math
import unittest

import numpy as np

from shaarp import extract_si_d_voigt
from shaarp.shg import solve_single_interface_shg
from shaarp.tensors import rotate_d_voigt_crystal_to_lab, rotate_rank2_crystal_to_lab

EPS_W = [2.0**2, 2.3**2, 2.6**2]   # biaxial: three distinct principal indices (eps_x != eps_y != eps_z)
EPS_2W = [2.2**2, 2.5**2, 2.8**2]
POSITIONS = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 0)]
TRUE = np.array([0.30, 0.50, 0.60, 0.80, 0.40, 0.20])
# AZIMUTH-varying geometries (az != 0 at several theta) -- exercises the eps-rotation path.
GEOMETRIES = [(0.5, 0.4), (0.5, 1.0), (0.7, 0.7), (0.9, 1.3)]
PHIS = list(np.linspace(0.2, math.pi - 0.2, 8))


def _rz(az):
    c, s = math.cos(az), math.sin(az)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _true_d():
    d = np.zeros((3, 6))
    for (m, ell), v in zip(POSITIONS, TRUE):
        d[m, ell] = v
    return d


def _measure(theta, az, phi):
    """Physical reflected SHG of a biaxial sample rotated by Rz(az): eps AND d both rotate."""
    rz = _rz(az)
    ew = rotate_rank2_crystal_to_lab(np.diag(EPS_W).astype(complex), rz)
    e2 = rotate_rank2_crystal_to_lab(np.diag(EPS_2W).astype(complex), rz)
    dlab = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(_true_d(), rz), dtype=complex))
    r = solve_single_interface_shg(
        ew, e2, dlab, incident_index_omega=1.0, incident_index_2omega=1.0,
        incident_theta_rad=theta, incident_jones=(math.sin(phi), math.cos(phi)),
        omega=1.0, mu=1.0, eps0=1.0,
    )
    c = np.asarray(r.coefficients)
    return complex(c[0]), complex(c[1])


class RotatedBiaxialNumericBasisTests(unittest.TestCase):
    def test_numeric_basis_recovers_rotated_biaxial(self):
        res = extract_si_d_voigt(
            eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
            geometries=GEOMETRIES, phi_values=PHIS, measure=_measure,
            method="field", observable="reflected", basis="numeric", mu=1, eps0=1,
        )
        self.assertTrue(
            res.identifiable,
            f"rotated-biaxial not identifiable (rank {res.rank}/{res.n_components})",
        )
        self.assertLess(res.residual, 1e-9)
        self.assertLess(
            np.max(np.abs(np.real(res.values) - TRUE)), 1e-8,
            "rotated-biaxial d not recovered by the numeric basis",
        )

    def test_numeric_basis_intensity_recovers_rotated_biaxial(self):
        """The fix flows through to the INTENSITY measurement model too: the intensity Gram
        system is built from the SAME complex field basis (now eps-rotated), so a |E|^2
        polarimeter recovers the rotated-biaxial tensor up to the physical overall sign."""
        res = extract_si_d_voigt(
            eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
            geometries=GEOMETRIES, phi_values=PHIS, measure=_measure,
            method="intensity", observable="reflected", basis="numeric", mu=1, eps0=1,
        )
        self.assertLess(res.residual, 1e-9)
        v = np.real(res.values)
        err = min(np.max(np.abs(v - TRUE)), np.max(np.abs(v + TRUE)))  # intensity: d up to global sign
        self.assertLess(err, 1e-7, "rotated-biaxial d not recovered (intensity, up to sign)")

    def test_eps_rotation_is_necessary(self):
        """The fix is non-vacuous: at az != 0 a biaxial crystal's response with eps rotated
        differs from leaving eps at its principal diagonal (the prior d-only-rotation bug). If
        these matched, rotating eps would be a no-op and the fix meaningless."""
        az, theta, phi = 1.0, 0.6, 0.7
        rz = _rz(az)
        dlab = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(_true_d(), rz), dtype=complex))
        ewp = np.diag(EPS_W).astype(complex)
        e2p = np.diag(EPS_2W).astype(complex)

        def coef(ew, e2):
            r = solve_single_interface_shg(
                ew, e2, dlab, incident_index_omega=1.0, incident_index_2omega=1.0,
                incident_theta_rad=theta, incident_jones=(math.sin(phi), math.cos(phi)),
                omega=1.0, mu=1.0, eps0=1.0,
            )
            return np.asarray(r.coefficients)[:2]

        rotated = coef(rotate_rank2_crystal_to_lab(ewp, rz), rotate_rank2_crystal_to_lab(e2p, rz))
        principal = coef(ewp, e2p)
        self.assertGreater(
            np.max(np.abs(rotated - principal)), 1e-3,
            "eps rotation must change the biaxial response (else the numeric-basis fix is vacuous)",
        )


if __name__ == "__main__":
    unittest.main()
