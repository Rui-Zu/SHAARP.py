"""End-to-end symbolic BIAXIAL (principal-axes-aligned) single-interface reflected SHG
validated against the numeric ``shaarp.shg.solve_single_interface_shg``.

Geometry: air -> biaxial crystal, principal dielectric axes aligned to the lab
(eps=diag(ex,ey,ez)), surface normal z, plane of incidence x-z. The two transmitted
eigenmodes are the y-mode (E||y) and the xz-mode (E in x-z). Uniaxial is the ex=ey
special case (covered by test_uniaxial_shg_symbolic via delegation). Distinct omega vs
2omega indices (dispersion) are used so the bound source is off-shell.
"""

import math
import unittest

import numpy as np

from shaarp.shg import solve_single_interface_shg
from shaarp.symbolic import solve_biaxial_single_interface_shg_symbolic


def _d():
    d = np.zeros((3, 6))
    d[0, 3] = 0.8
    d[1, 4] = 0.6
    d[2, 0] = 0.3
    d[2, 2] = 0.5
    d[0, 0] = 0.2
    return d


class BiaxialSHGSymbolicTests(unittest.TestCase):
    def _worst(self, epsW, eps2, thetas):
        import sympy as sp

        d = _d()
        worst, max_out = 0.0, 0.0
        for thi in thetas:
            for pol in ("s", "p"):
                res = solve_biaxial_single_interface_shg_symbolic(
                    eps_x_omega=epsW[0], eps_y_omega=epsW[1], eps_z_omega=epsW[2],
                    eps_x_2omega=eps2[0], eps_y_2omega=eps2[1], eps_z_2omega=eps2[2],
                    d_voigt_lab=sp.Matrix(d.tolist()), incident_theta_rad=thi,
                    incident_polarization=pol, omega=1, mu=1, eps0=1, simplify=False,
                )
                rs = complex(sp.N(res.boundary.coefficients[0]))
                rp = complex(sp.N(res.boundary.coefficients[1]))
                num = solve_single_interface_shg(
                    np.diag(epsW).astype(complex), np.diag(eps2).astype(complex), d,
                    incident_index_omega=1.0, incident_index_2omega=1.0,
                    incident_theta_rad=thi, incident_polarization=pol, omega=1.0, mu=1.0, eps0=1.0,
                )
                nc = np.asarray(num.coefficients)
                worst = max(worst, abs(rs - nc[0]), abs(rp - nc[1]))
                max_out = max(max_out, abs(complex(nc[0])), abs(complex(nc[1])))
        return worst, max_out

    def test_real_biaxial_shg_matches_numeric(self):
        epsW = [2.0**2, 2.3**2, 2.6**2]
        eps2 = [2.2**2, 2.5**2, 2.8**2]
        worst, max_out = self._worst(epsW, eps2, (0.3, 0.5, 0.7, 0.9))
        self.assertGreater(max_out, 1e-4, "near-zero output -> vacuous 0==0 check")
        self.assertLess(worst, 1e-10)

    def test_complex_absorbing_biaxial_shg_matches_numeric(self):
        epsW = [complex(2.0, 0.03) ** 2, complex(2.3, 0.06) ** 2, complex(2.6, 0.09) ** 2]
        eps2 = [complex(2.2, 0.05) ** 2, complex(2.5, 0.08) ** 2, complex(2.8, 0.12) ** 2]
        worst, max_out = self._worst(epsW, eps2, (0.4, 0.7))
        self.assertGreater(max_out, 1e-4, "near-zero output -> vacuous 0==0 check")
        self.assertLess(worst, 1e-9)

    def test_uniaxial_is_special_case_ex_equals_ey(self):
        """ex=ey must reproduce the uniaxial result (the delegation contract)."""
        from shaarp.symbolic import solve_uniaxial_single_interface_shg_symbolic
        import sympy as sp

        d = _d()
        for thi in (0.4, 0.8):
            for pol in ("s", "p"):
                bi = solve_biaxial_single_interface_shg_symbolic(
                    eps_x_omega=2.2**2, eps_y_omega=2.2**2, eps_z_omega=2.5**2,
                    eps_x_2omega=2.4**2, eps_y_2omega=2.4**2, eps_z_2omega=2.7**2,
                    d_voigt_lab=sp.Matrix(d.tolist()), incident_theta_rad=thi,
                    incident_polarization=pol, omega=1, simplify=False,
                )
                uni = solve_uniaxial_single_interface_shg_symbolic(
                    eps_ordinary_omega=2.2**2, eps_extraordinary_omega=2.5**2,
                    eps_ordinary_2omega=2.4**2, eps_extraordinary_2omega=2.7**2,
                    d_voigt_lab=sp.Matrix(d.tolist()), incident_theta_rad=thi,
                    incident_polarization=pol, omega=1, simplify=False,
                )
                for i in range(2):
                    self.assertAlmostEqual(
                        complex(sp.N(bi.boundary.coefficients[i])),
                        complex(sp.N(uni.boundary.coefficients[i])), places=12,
                    )


if __name__ == "__main__":
    unittest.main()
