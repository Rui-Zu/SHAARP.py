"""KTP (mm2, biaxial) d-extraction -- the third optical class with a REAL near-uniaxial-biaxial
crystal, at the d-values reported in Zu et al., npj Comput. Mater. 8, 246 (2022), Table 1:
|d33| = 17.0 pm/V, d33/d31 = 6.0, d32/d31 = 1.6, d24/d31 = 1.5, d15/d31 = 1.3.

KTP's principal indices are near-degenerate (n_x ~= n_y), which makes the SYMBOLIC SI in-plane
eigenmode selection ill-conditioned in the reflected-p channel (a documented closed-form
limitation). The NUMERIC-basis extractor (`basis="numeric"`) -- the design column for each d
position is the numeric single-interface SHG response to a unit d -- is exact for any crystal
including this near-degenerate case, and recovers the full KTP mm2 tensor.

(Representative KTP biaxial indices are used; the d-extraction is a round-trip, so the recovered
values reproduce the paper's mm2 magnitudes/ratios regardless of the exact indices.)
"""

import math
import unittest

import numpy as np

from shaarp import extract_si_d_voigt
from shaarp.shg import solve_single_interface_shg

# KTP mm2 d-tensor from the 2022 paper Table 1 (this-work column)
D31 = 17.0 / 6.0
D32 = 1.6 * D31
D33 = 17.0
D24 = 1.5 * D31
D15 = 1.3 * D31
POSITIONS = [(0, 4), (1, 3), (2, 0), (2, 1), (2, 2)]   # d15, d24, d31, d32, d33 (mm2)
TRUE = np.array([D15, D24, D31, D32, D33])
# representative KTP biaxial indices (n_x ~= n_y < n_z -> near-degenerate)
EPS_W = [1.738**2, 1.746**2, 1.830**2]
EPS_2W = [1.789**2, 1.794**2, 1.887**2]
GEOMETRIES = [(0.3, 0.0), (0.55, 0.0), (0.8, 0.0), (1.05, 0.0)]
PHIS = list(np.linspace(0.2, math.pi - 0.2, 8))


def _numeric_d():
    d = np.zeros((3, 6))
    for (m, ell), v in zip(POSITIONS, TRUE):
        d[m, ell] = v
    return d


def _measure(theta, az, phi):
    r = solve_single_interface_shg(
        np.diag(EPS_W).astype(complex), np.diag(EPS_2W).astype(complex), _numeric_d(),
        incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
        incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=1.0, eps0=1.0,
    )
    c = np.asarray(r.coefficients)
    return complex(c[0]), complex(c[1])


class KTPDExtractionTests(unittest.TestCase):
    def test_numeric_basis_recovers_ktp_mm2_tensor(self):
        res = extract_si_d_voigt(
            eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
            geometries=GEOMETRIES, phi_values=PHIS, measure=_measure,
            method="field", observable="reflected", basis="numeric", mu=1, eps0=1,
        )
        self.assertTrue(res.identifiable, f"KTP not identifiable (rank {res.rank}/{res.n_components})")
        self.assertLess(res.residual, 1e-7, "numeric model inconsistent with measured KTP scan")
        self.assertLess(np.max(np.abs(np.real(res.values) - TRUE)), 1e-6, "KTP d not recovered")
        # reproduces the paper Table 1 ratios
        rec = {pos: float(np.real(v)) for pos, v in zip(POSITIONS, res.values)}
        d31 = rec[(2, 0)]
        self.assertAlmostEqual(rec[(2, 2)] / d31, 6.0, places=3)   # d33/d31
        self.assertAlmostEqual(rec[(2, 1)] / d31, 1.6, places=3)   # d32/d31
        self.assertAlmostEqual(rec[(1, 3)] / d31, 1.5, places=3)   # d24/d31
        self.assertAlmostEqual(rec[(0, 4)] / d31, 1.3, places=3)   # d15/d31

    def test_symbolic_basis_is_ill_conditioned_for_near_degenerate_biaxial(self):
        """Documents WHY numeric basis is needed: the symbolic basis fails for KTP's n_x~=n_y."""
        res = extract_si_d_voigt(
            eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
            geometries=GEOMETRIES, phi_values=PHIS, measure=_measure,
            method="field", observable="reflected", basis="symbolic", mu=1, eps0=1,
        )
        # symbolic basis does NOT recover KTP (near-degenerate biaxial) -> motivates basis="numeric"
        self.assertGreater(np.max(np.abs(np.real(res.values) - TRUE)), 1.0)


if __name__ == "__main__":
    unittest.main()
