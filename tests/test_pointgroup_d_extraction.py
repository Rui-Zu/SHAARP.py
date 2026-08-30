"""d-extraction across MORE crystal point-group symmetry classes -- breadth beyond the three
optical classes already covered (isotropic GaAs -43m, uniaxial LiNbO3 3m, biaxial KTP mm2).

Here two further uniaxial NLO classes whose d-tensor symmetry forms (Boyd, *Nonlinear Optics*,
the standard contracted d-matrix tables) differ structurally from 3m:

  * -42m (D2d) -- KDP, AgGaSe2, ZnGeP2: d[0,3]=d[1,4]=d14, d[2,5]=d36
  * 32 (D3) -- alpha-quartz: d[0,0]=d11, d[0,1]=-d11, d[1,5]=-d11, d[0,3]=d14, d[1,4]=-d14

Each is recovered from a simulated reflected-SHG polarimetry scan (numeric basis; uniaxial eps is
azimuth-invariant, so a sample azimuth rotates only d -- consistent), and the recovered values are
checked to OBEY the point group's symmetry relations -- i.e. the inverse reproduces the crystal
class, not just isolated numbers. Magnitudes are representative (the symmetry FORM is the point).
"""

import math
import unittest

import numpy as np

from shaarp import extract_si_d_voigt
from shaarp.shg import solve_single_interface_shg
from shaarp.tensors import rotate_d_voigt_crystal_to_lab

# Rich theta + azimuth diversity (uniaxial in-plane structure needs azimuth to lift degeneracy).
GEOMETRIES = [(0.3, 0.0), (0.6, 0.0), (0.5, 0.4), (0.5, 0.9), (0.8, 0.6)]
PHIS = list(np.linspace(0.2, math.pi - 0.2, 8))


def _rz(az):
    c, s = math.cos(az), math.sin(az)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _make_measure(d_lab, ew, e2):
    def measure(theta, az, phi):
        d_rot = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d_lab, _rz(az)), dtype=complex))
        r = solve_single_interface_shg(
            np.diag(ew).astype(complex), np.diag(e2).astype(complex), d_rot,
            incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
            incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=1.0, eps0=1.0,
        )
        c = np.asarray(r.coefficients)
        return complex(c[0]), complex(c[1])
    return measure


def _recover(d_lab, ew, e2):
    positions = [(r, c) for r in range(3) for c in range(6) if abs(d_lab[r, c]) > 1e-12]
    true = np.array([d_lab[r, c] for (r, c) in positions])
    res = extract_si_d_voigt(
        eps_omega_principal=ew, eps_2omega_principal=e2, d_positions=positions,
        geometries=GEOMETRIES, phi_values=PHIS, measure=_make_measure(d_lab, ew, e2),
        method="field", observable="reflected", basis="numeric", mu=1, eps0=1,
    )
    rec = {pos: float(np.real(v)) for pos, v in zip(positions, res.values)}
    return res, rec, positions, true


class PointGroupDExtractionTests(unittest.TestCase):
    def test_recovers_minus42m_tensor_and_symmetry(self):
        # -42m (KDP-like), uniaxial. d14=d25, d36.
        d14, d36 = 0.40, 0.39
        d = np.zeros((3, 6))
        d[0, 3] = d14
        d[1, 4] = d14   # d25 = d14
        d[2, 5] = d36
        ew = [1.494**2, 1.494**2, 1.460**2]   # uniaxial (representative KDP @ 1064 nm)
        e2 = [1.514**2, 1.514**2, 1.470**2]   # @ 532 nm
        res, rec, _, true = _recover(d, ew, e2)
        self.assertTrue(res.identifiable, f"-42m not identifiable (rank {res.rank}/{res.n_components})")
        self.assertLess(res.relative_residual, 1e-9, "model inconsistent with -42m scan")
        self.assertLess(np.max(np.abs(np.real(res.values) - true)), 1e-7, "-42m d not recovered")
        # symmetry relations of -42m
        self.assertAlmostEqual(rec[(0, 3)], rec[(1, 4)], places=6, msg="-42m requires d14 = d25")
        self.assertAlmostEqual(rec[(0, 3)], d14, places=6)
        self.assertAlmostEqual(rec[(2, 5)], d36, places=6)

    def test_recovers_32_tensor_and_symmetry(self):
        # 32 (alpha-quartz-like), uniaxial. d11 (= -d12 = -d26), d14 (= -d25).
        d11, d14 = 0.30, 0.05
        d = np.zeros((3, 6))
        d[0, 0] = d11
        d[0, 1] = -d11   # d12 = -d11
        d[1, 5] = -d11   # d26 = -d11
        d[0, 3] = d14
        d[1, 4] = -d14   # d25 = -d14
        ew = [1.534**2, 1.534**2, 1.543**2]   # uniaxial (representative quartz @ 1064 nm)
        e2 = [1.547**2, 1.547**2, 1.556**2]   # @ 532 nm
        res, rec, _, true = _recover(d, ew, e2)
        self.assertTrue(res.identifiable, f"32 not identifiable (rank {res.rank}/{res.n_components})")
        self.assertLess(res.relative_residual, 1e-9, "model inconsistent with 32 scan")
        self.assertLess(np.max(np.abs(np.real(res.values) - true)), 1e-7, "32 d not recovered")
        # symmetry relations of 32
        self.assertAlmostEqual(rec[(0, 1)], -rec[(0, 0)], places=6, msg="32 requires d12 = -d11")
        self.assertAlmostEqual(rec[(1, 5)], -rec[(0, 0)], places=6, msg="32 requires d26 = -d11")
        self.assertAlmostEqual(rec[(1, 4)], -rec[(0, 3)], places=6, msg="32 requires d25 = -d14")
        self.assertAlmostEqual(rec[(0, 0)], d11, places=6)
        self.assertAlmostEqual(rec[(0, 3)], d14, places=6)


if __name__ == "__main__":
    unittest.main()
