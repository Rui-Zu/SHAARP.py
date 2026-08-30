"""Real-crystal d-extraction: recover the LiNbO3 (z-cut, point group 3m) SHG tensor from a
simulated reflected-SHG polarimetry scan, at the LITERATURE d-values carried by the package
preset (d33=27, d31=4.6, d15=4.6, d22=2.2 pm/V) -- the kind of crystal benchmarked by
SHAARP.si (Zu et al., npj Comput. Mater. 8, 246 (2022)).

This goes beyond the generic / single-component d-extraction tests: it recovers the FULL
eight-position lab Voigt tensor of an actual uniaxial crystal, and checks that the recovered
values obey the 3m symmetry relations (d[0,0]=-d22, d[0,1]=d22, d[2,0]=d[2,1]=d31, d[2,2]=d33).
The reflected scan needs incidence-angle AND sample-azimuth diversity (z-cut's 3-fold
in-plane degeneracy makes a single azimuth rank-deficient by one).
"""

import math
import unittest

import numpy as np

from shaarp import extract_si_d_voigt, presets
from shaarp.shg import solve_single_interface_shg
from shaarp.tensors import rotate_d_voigt_crystal_to_lab


def _rz(az):
    c, s = math.cos(az), math.sin(az)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class LiNbO3RealCrystalDExtractionTests(unittest.TestCase):
    def setUp(self):
        self.m = presets.linbo3_zcut_1064()
        self.d_lab = np.real(np.asarray(self.m.d_voigt(), dtype=complex))
        self.positions = [(r, c) for r in range(3) for c in range(6) if abs(self.d_lab[r, c]) > 1e-12]
        self.true = np.array([self.d_lab[r, c] for (r, c) in self.positions])
        self.ew = [float(self.m.eps_w()[i, i].real) for i in range(3)]
        self.e2 = [float(self.m.eps_2w()[i, i].real) for i in range(3)]

    def _measure(self, theta, az, phi):
        d_rot = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(self.d_lab, _rz(az)), dtype=complex))
        r = solve_single_interface_shg(
            np.diag(self.ew).astype(complex), np.diag(self.e2).astype(complex), d_rot,
            incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
            incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=1.0, eps0=1.0,
        )
        c = np.asarray(r.coefficients)
        return complex(c[0]), complex(c[1])

    def test_recovers_linbo3_3m_tensor_from_reflected_polarimetry(self):
        res = extract_si_d_voigt(
            eps_omega_principal=self.ew, eps_2omega_principal=self.e2, d_positions=self.positions,
            geometries=[(0.3, 0.0), (0.6, 0.0), (0.5, math.radians(30.0))],
            phi_values=list(np.linspace(0.2, math.pi - 0.2, 8)),
            measure=self._measure, method="field", observable="reflected", mu=1, eps0=1,
        )
        self.assertTrue(res.identifiable, f"LiNbO3 tensor not identifiable (rank {res.rank}/{res.n_components})")
        self.assertLess(res.residual, 1e-9, "model inconsistent with simulated LiNbO3 scan")
        self.assertLess(np.max(np.abs(np.real(res.values) - self.true)), 1e-7, "LiNbO3 d not recovered")

        # the recovered tensor obeys the 3m symmetry relations (and the literature magnitudes)
        rec = {pos: float(np.real(v)) for pos, v in zip(self.positions, res.values)}
        d22, d31, d33, d15 = 2.2, 4.6, 27.0, 4.6
        self.assertAlmostEqual(rec[(2, 2)], d33, places=5)            # d33
        self.assertAlmostEqual(rec[(2, 0)], d31, places=5)            # d31
        self.assertAlmostEqual(rec[(2, 1)], d31, places=5)            # d32 = d31
        self.assertAlmostEqual(rec[(0, 4)], d15, places=5)            # d15
        self.assertAlmostEqual(rec[(1, 3)], d15, places=5)            # d24 = d15
        self.assertAlmostEqual(rec[(0, 1)], d22, places=5)            # +d22
        self.assertAlmostEqual(rec[(0, 0)], -d22, places=5)           # -d22
        self.assertAlmostEqual(rec[(1, 5)], -d22, places=5)           # -d22


if __name__ == "__main__":
    unittest.main()
