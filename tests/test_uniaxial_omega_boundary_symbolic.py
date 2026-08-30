"""Validate the symbolic UNIAXIAL omega Fresnel boundary (first sub-step of the
uniaxial symbolic full-analytical SI assembly, gap #1) against the numeric
``shaarp.linear.solve_linear_interface``.

Geometry: isotropic incident medium (air) -> uniaxial crystal with optic axis along
the surface normal (z), plane of incidence x-z. The reflected s/p amplitudes are
convention-independent (air s/p basis is standard), so they are compared directly.
Covers real and complex (absorbing) uniaxial, ordinary (s) and extraordinary (p)
coupling. The eigenmode degeneracy fix is what makes the p/extraordinary path correct.
"""

import cmath
import math
import unittest

import numpy as np

from shaarp.linear import solve_linear_interface
from shaarp.symbolic import solve_uniaxial_linear_interface_symbolic


class UniaxialOmegaBoundarySymbolicTests(unittest.TestCase):
    def _r_sym(self, no2, ne2, thi, pol):
        import sympy as sp

        coeffs, _, _ = solve_uniaxial_linear_interface_symbolic(
            eps_ordinary=no2, eps_extraordinary=ne2, incident_index=1,
            incident_theta_rad=thi, incident_polarization=pol, omega=1, simplify=False,
        )
        return complex(sp.N(coeffs[0])), complex(sp.N(coeffs[1]))

    def _r_num(self, no2, ne2, thi, pol):
        eps = np.diag([no2, no2, ne2]).astype(complex)
        num = solve_linear_interface(eps, incident_index=1.0, incident_theta_rad=thi, incident_polarization=pol, omega=1.0)
        return complex(num.r_s), complex(num.r_p)

    def test_real_uniaxial_reflection_matches_numeric(self):
        no2, ne2 = 2.2**2, 2.5**2
        worst = 0.0
        for thi in (0.2, 0.4, 0.6, 0.8, 1.0):
            for pol in ("s", "p"):
                rs, rp = self._r_sym(no2, ne2, thi, pol)
                rsn, rpn = self._r_num(no2, ne2, thi, pol)
                worst = max(worst, abs(rs - rsn), abs(rp - rpn))
        self.assertLess(worst, 1e-10, f"uniaxial reflection deviates from numeric: {worst:.2e}")

    def test_negative_uniaxial_reflection_matches_numeric(self):
        no2, ne2 = 2.5**2, 2.2**2  # ne < no
        worst = 0.0
        for thi in (0.3, 0.7, 1.1):
            for pol in ("s", "p"):
                rs, rp = self._r_sym(no2, ne2, thi, pol)
                rsn, rpn = self._r_num(no2, ne2, thi, pol)
                worst = max(worst, abs(rs - rsn), abs(rp - rpn))
        self.assertLess(worst, 1e-10)

    def test_complex_absorbing_uniaxial_reflection_matches_numeric(self):
        no2, ne2 = complex(2.2, 0.05) ** 2, complex(2.5, 0.12) ** 2
        worst = 0.0
        for thi in (0.3, 0.6, 0.9):
            for pol in ("s", "p"):
                rs, rp = self._r_sym(no2, ne2, thi, pol)
                rsn, rpn = self._r_num(no2, ne2, thi, pol)
                worst = max(worst, abs(rs - rsn), abs(rp - rpn))
        self.assertLess(worst, 1e-9)

    def test_isotropic_limit_recovers_isotropic_fresnel(self):
        """no == ne -> the uniaxial solver must reduce to the ordinary isotropic
        Fresnel reflection (a consistency check on the birefringent construction)."""
        n2 = 2.3**2
        for thi in (0.25, 0.75):
            for pol in ("s", "p"):
                rs, rp = self._r_sym(n2, n2, thi, pol)
                rsn, rpn = self._r_num(n2, n2, thi, pol)
                self.assertAlmostEqual(rs, rsn, places=9)
                self.assertAlmostEqual(rp, rpn, places=9)


if __name__ == "__main__":
    unittest.main()
