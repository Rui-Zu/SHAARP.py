"""Biaxial d-extraction (the third optical class). The SHAARP.si benchmark crystals span the
three optical classes -- isotropic (GaAs), uniaxial (LiNbO3), biaxial (KTP) -- Zu et al.,
npj Comput. Mater. 8, 246 (2022). This test covers the BIAXIAL case: recover the full
six-component d-tensor of a generic biaxial crystal (eps_x != eps_y != eps_z) from a simulated
SHG polarimetry scan.

Key (physically meaningful) result: for a biaxial crystal the full tensor is identifiable from
INCIDENCE-ANGLE diversity ALONE (no sample azimuth), in BOTH the reflected and the transmitted
channel. The broken in-plane degeneracy (eps_x != eps_y) supplies the constraints that the
uniaxial/isotropic cases needed an azimuth scan for -- so biaxial extraction never requires the
(numeric-only) rotated-biaxial closed form.
"""

import math
import unittest

import numpy as np

from shaarp import extract_si_d_voigt
from shaarp.shg import solve_single_interface_shg

EPS_W = [2.0**2, 2.3**2, 2.6**2]      # biaxial: three distinct principal indices
EPS_2W = [2.2**2, 2.5**2, 2.8**2]
POSITIONS = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 0)]
TRUE = np.array([0.30, 0.50, 0.60, 0.80, 0.40, 0.20])
GEOMETRIES = [(0.3, 0.0), (0.55, 0.0), (0.8, 0.0), (1.05, 0.0)]  # theta-only (no azimuth)
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


def _measure_transmitted(theta, az, phi):
    r = solve_single_interface_shg(
        np.diag(EPS_W).astype(complex), np.diag(EPS_2W).astype(complex), _numeric_d(),
        incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
        incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=1.0, eps0=1.0,
    )
    return np.asarray(r.transmitted_fast_2omega.electric) + np.asarray(r.transmitted_slow_2omega.electric)


class BiaxialDExtractionTests(unittest.TestCase):
    def test_reflected_theta_only_recovers_full_biaxial_tensor(self):
        res = extract_si_d_voigt(
            eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
            geometries=GEOMETRIES, phi_values=PHIS, measure=_measure,
            method="field", observable="reflected", mu=1, eps0=1,
        )
        self.assertTrue(res.identifiable, f"biaxial reflected not identifiable theta-only (rank {res.rank}/{res.n_components})")
        self.assertLess(res.residual, 1e-9)
        self.assertLess(np.max(np.abs(np.real(res.values) - TRUE)), 1e-9, "biaxial reflected d not recovered")

    def test_transmitted_theta_only_recovers_full_biaxial_tensor(self):
        res = extract_si_d_voigt(
            eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
            geometries=GEOMETRIES[:3], phi_values=PHIS, measure=_measure_transmitted,
            method="field", observable="transmitted", mu=1, eps0=1,
        )
        self.assertTrue(res.identifiable, f"biaxial transmitted not identifiable theta-only (rank {res.rank}/{res.n_components})")
        self.assertLess(res.residual, 1e-9)
        self.assertLess(np.max(np.abs(np.real(res.values) - TRUE)), 1e-9, "biaxial transmitted d not recovered")


if __name__ == "__main__":
    unittest.main()
