import math
import unittest

import numpy as np

from shaarp.boundary import boundary_residual, isotropic_fresnel_coefficients, solve_boundary_continuity, solve_fresnel_boundary
from shaarp.waves import Wave
from shaarp.waves import make_isotropic_wave


class BoundaryTests(unittest.TestCase):
    def test_boundary_residual_order_matches_mathematica_threaded_equations(self):
        top = Wave(
            k=np.array([0.2, 0.0, 1.1], dtype=complex),
            electric=np.array([1.0 + 1.5j, -2.0j, 0.25], dtype=complex),
            omega=2.0,
        )
        bottom = Wave(
            k=np.array([0.2, 0.0, 1.3], dtype=complex),
            electric=np.array([0.5 - 0.25j, 0.75j, -0.1], dtype=complex),
            omega=2.0,
        )
        expected_e = top.electric - bottom.electric
        expected_h = top.magnetic(mu=1.0) - bottom.magnetic(mu=1.0)

        np.testing.assert_allclose(
            boundary_residual([top], [bottom], mu=1.0),
            np.array([expected_e[0], expected_e[1], expected_h[0], expected_h[1]], dtype=complex),
            atol=1e-12,
        )

    def test_s_polarized_boundary_matches_closed_form_fresnel(self):
        n1, n2 = 1.0, 1.7
        theta_i = math.radians(35.0)
        closed = isotropic_fresnel_coefficients(n1, n2, theta_i)
        theta_t = float(np.real(closed["theta_t"]))

        incident = [make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="s", amplitude=1.0)]
        reflected_basis = [
            make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="s", direction_sign_z=-1.0),
            make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="p", direction_sign_z=-1.0),
        ]
        transmitted_basis = [
            make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="s"),
            make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="p"),
        ]
        _, _, coeffs = solve_fresnel_boundary(incident, reflected_basis, transmitted_basis)
        self.assertTrue(np.isclose(coeffs[0], closed["rs"], atol=1e-12))
        self.assertTrue(np.isclose(coeffs[2], closed["ts"], atol=1e-12))
        self.assertTrue(abs(coeffs[1]) < 1e-12)
        self.assertTrue(abs(coeffs[3]) < 1e-12)

    def test_p_polarized_boundary_matches_closed_form_fresnel(self):
        n1, n2 = 1.0, 1.7
        theta_i = math.radians(35.0)
        closed = isotropic_fresnel_coefficients(n1, n2, theta_i)
        theta_t = float(np.real(closed["theta_t"]))

        incident = [make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="p", amplitude=1.0)]
        reflected_basis = [
            make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="s", direction_sign_z=-1.0),
            make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="p", direction_sign_z=-1.0),
        ]
        transmitted_basis = [
            make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="s"),
            make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="p"),
        ]
        _, _, coeffs = solve_fresnel_boundary(incident, reflected_basis, transmitted_basis)
        self.assertTrue(abs(coeffs[0]) < 1e-12)
        self.assertTrue(abs(coeffs[2]) < 1e-12)
        self.assertTrue(np.isclose(coeffs[1], closed["rp"], atol=1e-12))
        self.assertTrue(np.isclose(coeffs[3], closed["tp"], atol=1e-12))

    def test_general_boundary_continuity_allows_known_bottom_source(self):
        n1, n2 = 1.0, 1.5
        theta_i = math.radians(20.0)
        theta_t = float(np.real(isotropic_fresnel_coefficients(n1, n2, theta_i)["theta_t"]))

        top_known = []
        bottom_known = [
            make_isotropic_wave(
                n=n2,
                theta_rad=theta_t,
                polarization="s",
                amplitude=0.13 + 0.07j,
                omega=2.0,
            )
        ]
        top_basis = [
            make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="s", direction_sign_z=-1.0, omega=2.0),
            make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="p", direction_sign_z=-1.0, omega=2.0),
        ]
        bottom_basis = [
            make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="s", omega=2.0),
            make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="p", omega=2.0),
        ]

        top_unknown, bottom_unknown, coeffs = solve_boundary_continuity(
            top_known=top_known,
            bottom_known=bottom_known,
            top_unknown_basis=top_basis,
            bottom_unknown_basis=bottom_basis,
        )
        residual = boundary_residual(top_known + top_unknown, bottom_known + bottom_unknown)
        self.assertTrue(np.linalg.norm(residual) < 1e-12)
        self.assertTrue(np.all(np.isfinite(coeffs)))


if __name__ == "__main__":
    unittest.main()
