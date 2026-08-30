"""End-to-end symbolic UNIAXIAL single-interface reflected SHG (gap #1, uniaxial-first)
validated against the numeric ``shaarp.shg.solve_single_interface_shg``.

This exercises the WHOLE uniaxial symbolic full-analytical chain:
omega birefringent boundary -> ee/oo/eo sources -> solveInhom -> 2omega birefringent
boundary -> reflected SHG. Distinct omega vs 2omega indices (physical dispersion) are
used so the bound source is off-shell (equal indices => phase-matching singularity).
"""

import math
import unittest

import numpy as np

from shaarp.shg import solve_single_interface_shg
from shaarp.symbolic import solve_uniaxial_single_interface_shg_symbolic


def _d_uniaxial_allowed():
    d = np.zeros((3, 6))
    d[0, 3] = d[1, 4] = 0.9
    d[2, 0] = d[2, 1] = 0.3
    d[2, 2] = 0.5
    return d


class UniaxialSHGSymbolicTests(unittest.TestCase):
    def _compare(self, no, ne, no2, ne2, thi, pol):
        import sympy as sp

        d = _d_uniaxial_allowed()
        res = solve_uniaxial_single_interface_shg_symbolic(
            eps_ordinary_omega=no**2, eps_extraordinary_omega=ne**2,
            eps_ordinary_2omega=no2**2, eps_extraordinary_2omega=ne2**2,
            d_voigt_lab=sp.Matrix(d.tolist()), incident_theta_rad=thi,
            incident_polarization=pol, omega=1, mu=1, eps0=1, simplify=False,
        )
        rs = complex(sp.N(res.boundary.coefficients[0]))
        rp = complex(sp.N(res.boundary.coefficients[1]))
        epsW = np.diag([no**2, no**2, ne**2]).astype(complex)
        eps2 = np.diag([no2**2, no2**2, ne2**2]).astype(complex)
        num = solve_single_interface_shg(
            epsW, eps2, d, incident_index_omega=1.0, incident_index_2omega=1.0,
            incident_theta_rad=thi, incident_polarization=pol, omega=1.0, mu=1.0, eps0=1.0,
        )
        nc = np.asarray(num.coefficients)
        # return (worst diff, max |numeric output|) so callers can guard against vacuous 0==0
        return max(abs(rs - nc[0]), abs(rp - nc[1])), max(abs(complex(nc[0])), abs(complex(nc[1])))

    def test_real_uniaxial_shg_matches_numeric(self):
        worst, max_out = 0.0, 0.0
        for thi in (0.3, 0.5, 0.7, 0.9):
            for pol in ("s", "p"):
                d, m = self._compare(2.2, 2.5, 2.4, 2.7, thi, pol)
                worst, max_out = max(worst, d), max(max_out, m)
        self.assertGreater(max_out, 1e-4, "near-zero output -> vacuous 0==0 check")
        self.assertLess(worst, 1e-10, f"uniaxial SHG deviates from numeric: {worst:.2e}")

    def test_negative_uniaxial_shg_matches_numeric(self):
        worst, max_out = 0.0, 0.0
        for thi in (0.4, 0.8):
            for pol in ("s", "p"):
                d, m = self._compare(2.5, 2.2, 2.7, 2.35, thi, pol)
                worst, max_out = max(worst, d), max(max_out, m)
        self.assertGreater(max_out, 1e-4, "near-zero output -> vacuous 0==0 check")
        self.assertLess(worst, 1e-10)

    def test_complex_absorbing_uniaxial_shg_matches_numeric(self):
        import sympy as sp

        d = _d_uniaxial_allowed()
        no, ne = complex(2.2, 0.04), complex(2.5, 0.09)
        no2, ne2 = complex(2.4, 0.06), complex(2.7, 0.11)
        worst, max_out = 0.0, 0.0
        for thi in (0.4, 0.7):
            for pol in ("s", "p"):
                res = solve_uniaxial_single_interface_shg_symbolic(
                    eps_ordinary_omega=no**2, eps_extraordinary_omega=ne**2,
                    eps_ordinary_2omega=no2**2, eps_extraordinary_2omega=ne2**2,
                    d_voigt_lab=sp.Matrix(d.tolist()), incident_theta_rad=thi,
                    incident_polarization=pol, omega=1, mu=1, eps0=1, simplify=False,
                )
                rs = complex(sp.N(res.boundary.coefficients[0]))
                rp = complex(sp.N(res.boundary.coefficients[1]))
                epsW = np.diag([no**2, no**2, ne**2]).astype(complex)
                eps2 = np.diag([no2**2, no2**2, ne2**2]).astype(complex)
                num = solve_single_interface_shg(
                    epsW, eps2, d, incident_index_omega=1.0, incident_index_2omega=1.0,
                    incident_theta_rad=thi, incident_polarization=pol, omega=1.0, mu=1.0, eps0=1.0,
                )
                nc = np.asarray(num.coefficients)
                worst = max(worst, abs(rs - nc[0]), abs(rp - nc[1]))
                max_out = max(max_out, abs(complex(nc[0])), abs(complex(nc[1])))
        self.assertGreater(max_out, 1e-4, "near-zero output -> vacuous 0==0 check")
        self.assertLess(worst, 1e-9, f"complex uniaxial SHG deviates: {worst:.2e}")


if __name__ == "__main__":
    unittest.main()
