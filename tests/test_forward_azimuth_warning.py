"""Forward-solver azimuth guard ( correctness audit, same class as the d-extraction
eps-rotation bug). ``solve_si_shg_full_analytical_symbolic`` rotates ONLY the d-tensor under a
sample azimuth, not eps -- physical only for azimuth-invariant eps (isotropic / uniaxial-z). For
a BIAXIAL crystal under a non-trivial azimuth the closed form is unphysical (the eigenmodes would
need the rotated eps = the intractable rotated-biaxial quartic). This was previously silent and
untested; the solver now emits a RuntimeWarning for the clearly-detectable numeric case. This
test locks: (1) biaxial + concrete azimuth WARNS; (2) uniaxial + concrete azimuth does NOT warn.
"""

import unittest
import warnings

import sympy as sp

from shaarp.symbolic import solve_si_shg_full_analytical_symbolic


def _solve(eps_xy, az):
    """One fully-numeric (concrete phi/analyzer) SI full-analytical solve; eps_xy sets eps_x=eps_y
    vs distinct (biaxial)."""
    d = sp.zeros(3, 6)
    d[2, 2] = 1.0
    d[0, 0] = 0.5
    ex, ey = eps_xy
    return solve_si_shg_full_analytical_symbolic(
        eps_x_omega=ex, eps_y_omega=ey, eps_z_omega=6.76,
        eps_x_2omega=ex + 0.84, eps_y_2omega=ey + 0.84, eps_z_2omega=7.84,
        d_voigt_lab=d, incident_theta_rad=0.5, phi_symbol=sp.Float(0.5),
        analyzer_symbol=sp.Float(0.3), sample_azimuth_symbol=sp.Float(az),
        omega=1, mu=1, eps0=1, simplify=False,
    )


class ForwardAzimuthWarningTests(unittest.TestCase):
    def test_biaxial_azimuth_emits_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _solve((4.0, 5.29), 0.6)  # eps_x != eps_y -> biaxial, az not a multiple of pi
            msgs = [str(x.message).lower() for x in caught if issubclass(x.category, RuntimeWarning)]
        self.assertTrue(
            any("biaxial" in m and "azimuth" in m for m in msgs),
            "biaxial crystal under a non-trivial azimuth must warn (eps not rotated)",
        )

    def test_uniaxial_azimuth_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _solve((4.0, 4.0), 0.6)  # eps_x == eps_y -> uniaxial-z, azimuth-invariant eps
            msgs = [str(x.message).lower() for x in caught if issubclass(x.category, RuntimeWarning)]
        self.assertFalse(
            any("biaxial" in m for m in msgs),
            "uniaxial-z (azimuth-invariant eps) must NOT trigger the biaxial-azimuth warning",
        )


if __name__ == "__main__":
    unittest.main()
