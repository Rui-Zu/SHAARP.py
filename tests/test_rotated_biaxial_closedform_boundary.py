"""Document (and pin) the PRACTICAL BOUNDARY of the closed-form symbolic SI SHG.

The transmitted-mode kz solve det(k^2 I - k k^T - eps_lab) = 0 (k = (kx, 0, kz)) is a
degree-4 polynomial in kz. Whether a *clean* closed form exists depends on whether it is
BIQUADRATIC (only even powers of kz -> a quadratic in kz^2, the form solve_biaxial_* uses):

  * PRINCIPAL-AXES-ALIGNED eps (and any rotation preserving the z surface-normal mirror
    for k in the x-z plane, e.g. rotation ABOUT x) -> BIQUADRATIC -> clean closed form.
  * GENERAL rotation (about y, or full Euler -> optic axes tilted out of the mirror) ->
    FULL QUARTIC (odd powers kz^3, kz^1 nonzero) -> roots are the quartic-formula radicals,
    impractical as a symbolic closed form.

The NUMERIC ``solve_single_interface_shg`` handles ALL orientations (it solves the quartic
numerically), so nothing is lost physically -- only the closed-form symbolic convenience
stops at the biquadratic cases. This test pins that characterization so the boundary is
understood (not mistaken for a missing capability), per the intention-first practice.
"""

import math
import unittest

import numpy as np
import sympy as sp


def _kz_poly_coeffs(eps_lab, kx):
    kz = sp.symbols("kz")
    k = sp.Matrix([kx, 0, kz])
    M = (k.T * k)[0] * sp.eye(3) - k * k.T - sp.Matrix(eps_lab)
    return [complex(c) for c in sp.Poly(sp.expand(M.det()), kz).all_coeffs()]  # [kz^4..kz^0]


def _R(a, b, c):
    ca, sa, cb, sb, cc, sc = (math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c))
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return Rz @ Ry @ Rx


EPS = np.diag([2.0**2, 2.3**2, 2.6**2])
KX = 0.5


class ClosedFormBoundaryTests(unittest.TestCase):
    def _odd_power_magnitude(self, eps_lab):
        c = _kz_poly_coeffs(eps_lab, KX)  # [kz^4, kz^3, kz^2, kz^1, kz^0]
        return abs(c[1]) + abs(c[3])

    def test_principal_aligned_is_biquadratic(self):
        self.assertLess(self._odd_power_magnitude(EPS.tolist()), 1e-9)

    def test_mirror_preserving_rotation_about_x_stays_biquadratic(self):
        eps = _R(0, 0, math.radians(30)) @ EPS @ _R(0, 0, math.radians(30)).T
        self.assertLess(self._odd_power_magnitude(eps.tolist()), 1e-9)

    def test_general_rotation_is_full_quartic(self):
        for ang in [(0, math.radians(30), 0), tuple(map(math.radians, (20, 35, 40)))]:
            eps = _R(*ang) @ EPS @ _R(*ang).T
            self.assertGreater(self._odd_power_magnitude(eps.tolist()), 1e-3)

    def test_numeric_path_covers_general_rotated_biaxial(self):
        """The numeric SI SHG must still work for a general-rotated biaxial crystal
        (so the closed-form boundary is a convenience limit, not a capability gap)."""
        from shaarp.shg import solve_single_interface_shg

        d = np.zeros((3, 6))
        d[0, 3] = 0.8
        d[1, 4] = 0.6
        d[2, 2] = 0.5
        R = _R(*map(math.radians, (20, 35, 40)))
        epsW = R @ EPS @ R.T
        eps2 = R @ np.diag([2.2**2, 2.5**2, 2.8**2]) @ R.T
        res = solve_single_interface_shg(
            epsW.astype(complex), eps2.astype(complex), d,
            incident_index_omega=1.0, incident_index_2omega=1.0,
            incident_theta_rad=0.5, incident_polarization="p", omega=1.0,
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(res.coefficients))))


if __name__ == "__main__":
    unittest.main()
