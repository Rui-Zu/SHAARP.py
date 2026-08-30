"""Form-validate the EIGENVALUE / eigenmode stage for UNIAXIAL crystals against the
closed-form (textbook) index surface -- extending the analytical-form validation
beyond the isotropic GaAs case.

the author's directive named the eigenvalue (principal-axis/eigendecomposition) stage as one
that must be validated in closed form, not only numerically. For an OPTICALLY UNIAXIAL
crystal (epsilon = diag(no^2, no^2, ne^2), optic axis along z) the two transverse modes
have exact closed forms for a wave at angle theta to the optic axis:

    ordinary: n_o(theta) = no (independent of theta)
                     E_o perpendicular to the k-optic-axis plane (along y here)
    extraordinary: 1/n_e(theta)^2 = cos^2(theta)/no^2 + sin^2(theta)/ne^2
                     E_e in the k-optic-axis plane (x-z here)

The numeric SHAARP `modes_for_direction` (the `solveSnell` `L E = eta eps E` port) must
reproduce these closed forms exactly. This is the uniaxial analogue of the isotropic
GaAs full-analytical eigenvalue validation; the general BIAXIAL closed form exists too
(the 3x3 characteristic polynomial is a cubic with Cardano roots) but is intentionally
out of scope here -- it is messy and is the documented remaining `not_full` item.

Real AND complex (absorbing uniaxial) indices are covered, per the complex-coefficient
discipline.
"""

import cmath
import math
import unittest

import numpy as np

from shaarp.anisotropic import modes_for_direction


def _ne_of_theta(no, ne, theta):
    """Closed-form extraordinary index for optic axis along z, wave at angle theta."""
    return 1.0 / cmath.sqrt(math.cos(theta) ** 2 / no**2 + math.sin(theta) ** 2 / ne**2)


def _sorted_indices(eps, theta):
    d = np.array([math.sin(theta), 0.0, math.cos(theta)])
    m1, m2 = modes_for_direction(eps, d)
    n1 = complex(m1.refractive_index)
    n2 = complex(m2.refractive_index)
    return sorted([n1, n2], key=lambda z: z.real), (m1, m2)


class UniaxialEigenmodeClosedFormTests(unittest.TestCase):
    ANGLES_DEG = (5, 20, 35, 50, 70, 85)

    def test_real_uniaxial_indices_match_closed_form(self):
        no, ne = 2.20, 2.45  # positive uniaxial
        eps = np.diag([no**2, no**2, ne**2]).astype(complex)
        for th_deg in self.ANGLES_DEG:
            th = math.radians(th_deg)
            (n_lo, n_hi), _ = _sorted_indices(eps, th)
            closed = sorted([complex(no), _ne_of_theta(no, ne, th)], key=lambda z: z.real)
            self.assertAlmostEqual(n_lo, closed[0], places=12, msg=f"ordinary @ {th_deg} deg")
            self.assertAlmostEqual(n_hi, closed[1], places=12, msg=f"extraordinary @ {th_deg} deg")

    def test_negative_uniaxial_indices_match_closed_form(self):
        no, ne = 2.45, 2.20  # negative uniaxial (ne < no)
        eps = np.diag([no**2, no**2, ne**2]).astype(complex)
        for th_deg in self.ANGLES_DEG:
            th = math.radians(th_deg)
            (n_lo, n_hi), _ = _sorted_indices(eps, th)
            closed = sorted([complex(no), _ne_of_theta(no, ne, th)], key=lambda z: z.real)
            self.assertAlmostEqual(n_lo, closed[0], places=12, msg=f"ordinary @ {th_deg} deg")
            self.assertAlmostEqual(n_hi, closed[1], places=12, msg=f"extraordinary @ {th_deg} deg")

    def test_complex_absorbing_uniaxial_indices_match_closed_form(self):
        no, ne = complex(2.20, 0.05), complex(2.45, 0.12)  # absorbing uniaxial
        eps = np.diag([no**2, no**2, ne**2]).astype(complex)
        for th_deg in self.ANGLES_DEG:
            th = math.radians(th_deg)
            d = np.array([math.sin(th), 0.0, math.cos(th)])
            m1, m2 = modes_for_direction(eps, d)
            got = {complex(m1.refractive_index), complex(m2.refractive_index)}
            # the closed-form set {n_o, n_e(theta)} must be reproduced (match each to the nearest)
            for target in (no, _ne_of_theta(no, ne, th)):
                nearest = min(got, key=lambda z: abs(z - target))
                self.assertLess(abs(nearest - target), 1e-10, f"abs uniaxial index @ {th_deg} deg")

    def test_mode_field_directions_are_ordinary_y_and_extraordinary_in_plane(self):
        no, ne = 2.20, 2.45
        eps = np.diag([no**2, no**2, ne**2]).astype(complex)
        th = math.radians(40.0)
        d = np.array([math.sin(th), 0.0, math.cos(th)])
        m1, m2 = modes_for_direction(eps, d)
        # one mode is ordinary (E along y); the other extraordinary (E in x-z plane)
        labels = []
        for m in (m1, m2):
            e = np.asarray(m.electric_direction, dtype=complex)
            e = e / np.linalg.norm(e)
            ord_like = abs(e[1])
            inplane = abs(e[0]) + abs(e[2])
            labels.append("o" if ord_like > 0.99 else ("e" if inplane > 0.99 else "?"))
        self.assertEqual(set(labels), {"o", "e"}, f"mode field directions not o/e: {labels}")

    def test_isotropic_limit_both_indices_equal(self):
        """When no == ne the uniaxial crystal is isotropic: both modes give the same n."""
        n0 = 2.3
        eps = np.diag([n0**2, n0**2, n0**2]).astype(complex)
        for th_deg in (10, 45, 80):
            (n_lo, n_hi), _ = _sorted_indices(eps, math.radians(th_deg))
            self.assertAlmostEqual(n_lo, complex(n0), places=12)
            self.assertAlmostEqual(n_hi, complex(n0), places=12)


if __name__ == "__main__":
    unittest.main()
