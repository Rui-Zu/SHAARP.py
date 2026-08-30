"""d-extraction (mu, eps0) units contract -- a regression guard from the correctness
audit. The extractor builds a linear design ``M(mu,eps0) . d = y`` whose ``y`` comes from the
user's ``measure``; the two MUST use the same (mu, eps0) or the recovered d is silently
rescaled/corrupted. This test pins three things:

  1. CONSISTENT clean units (1,1) recover d to machine precision (relative_residual ~ 1e-12).
  2. CONSISTENT physical SI constants (MU0, EPS0) ALSO recover to machine precision -- i.e. the
     value doesn't matter, only that BOTH sides agree (MU0/EPS0 is NOT ill-conditioned; an
     earlier audit mis-attributed a non-identifiable-geometry failure to the units).
  3. MIXED units (measure clean, extractor physical) give WRONG d -- and crucially this is
     DETECTABLE via ``result.relative_residual`` (large on a full-rank fit), which is exactly the
     scale-free reliability signal added so the footgun cannot pass silently.

It also locks that the SI and ML extractors share the SAME default units (they were inconsistent
-- SI=(1,1), ML=(MU0,EPS0) -- which made the mixed-units footgun easy to hit; now both (1,1)).
"""

import inspect
import math
import unittest

import numpy as np

from shaarp import extract_ml_film_d_voigt, extract_si_d_voigt
from shaarp.shg import solve_single_interface_shg
from shaarp.tensors import rotate_d_voigt_crystal_to_lab
from shaarp.waves import EPS0, MU0

# Known-IDENTIFIABLE uniaxial config (mirrors test_polarimetry_d_extraction_api.py).
EPS_W = [2.2**2, 2.2**2, 2.5**2]
EPS_2W = [2.4**2, 2.4**2, 2.7**2]
POSITIONS = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 0)]
TRUE = np.array([0.30, 0.50, 0.60, 0.80, 0.40, 0.20])
GEOMETRIES = [(0.35, 0.0), (0.35, 0.6), (0.35, 1.2), (0.75, 0.0), (0.75, 0.6), (0.75, 1.2)]
PHIS = list(np.linspace(0.25, math.pi - 0.25, 6))


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _measure(mu, eps0):
    def f(theta, az, phi):
        d = np.zeros((3, 6))
        for (m, ell), v in zip(POSITIONS, TRUE):
            d[m, ell] = v
        d_rot = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d, _rz(az)), dtype=complex))
        r = solve_single_interface_shg(
            np.diag(EPS_W).astype(complex), np.diag(EPS_2W).astype(complex), d_rot,
            incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
            incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=mu, eps0=eps0,
        )
        c = np.asarray(r.coefficients)
        return complex(c[0]), complex(c[1])
    return f


def _extract(mu_measure, eps_measure, mu_x, eps_x):
    return extract_si_d_voigt(
        eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
        geometries=GEOMETRIES, phi_values=PHIS, measure=_measure(mu_measure, eps_measure),
        method="field", basis="numeric", mu=mu_x, eps0=eps_x,
    )


class DExtractionUnitsContractTests(unittest.TestCase):
    def test_consistent_clean_units_recovers_to_machine_precision(self):
        res = _extract(1.0, 1.0, 1.0, 1.0)
        self.assertTrue(res.identifiable)
        self.assertLess(res.relative_residual, 1e-9, "clean-units fit should be machine-precision")
        self.assertLess(np.max(np.abs(np.real(res.values) - TRUE)), 1e-9)

    def test_consistent_physical_units_recovers_to_machine_precision(self):
        # MU0/EPS0 is NOT ill-conditioned -- only mismatch matters.
        res = _extract(MU0, EPS0, MU0, EPS0)
        self.assertTrue(res.identifiable)
        self.assertLess(res.relative_residual, 1e-9, "physical-units (matched) fit should be machine-precision")
        self.assertLess(np.max(np.abs(np.real(res.values) - TRUE)), 1e-9)

    def test_mixed_units_is_wrong_AND_detectable(self):
        # measure uses clean units, extractor uses physical constants -> the footgun.
        res = _extract(1.0, 1.0, MU0, EPS0)
        # The recovery is corrupted...
        self.assertGreater(
            np.max(np.abs(np.real(res.values) - TRUE)), 1e-2,
            "mixed units should corrupt the recovered d",
        )
        # ...and -- the point of the fix -- it is DETECTABLE: relative_residual is large on a
        # fit that is nominally full-rank, so a user checking it cannot be silently misled.
        self.assertGreater(
            res.relative_residual, 1e-4,
            "mixed units must be flagged by a large relative_residual (the silent-failure guard)",
        )

    def test_si_and_ml_extractors_share_default_units(self):
        # Locks the unified default (both clean units); they were inconsistent before the audit
        # (SI=(1,1), ML=(MU0,EPS0)), which is what made the mixed-units footgun easy to hit.
        si = inspect.signature(extract_si_d_voigt).parameters
        ml = inspect.signature(extract_ml_film_d_voigt).parameters
        self.assertEqual(si["mu"].default, ml["mu"].default)
        self.assertEqual(si["eps0"].default, ml["eps0"].default)
        self.assertEqual(float(si["mu"].default), 1.0)
        self.assertEqual(float(si["eps0"].default), 1.0)


if __name__ == "__main__":
    unittest.main()
