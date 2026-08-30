import math
import unittest

import numpy as np

from shaarp.shg import solve_single_interface_shg


class SingleInterfaceSHGTests(unittest.TestCase):
    def test_zero_nonlinearity_gives_zero_2omega_fields(self):
        result = solve_single_interface_shg(
            np.eye(3) * 1.6**2,
            np.eye(3) * 1.8**2,
            np.zeros((3, 6), dtype=complex),
            incident_index_omega=1.0,
            incident_index_2omega=1.0,
            incident_theta_rad=math.radians(20.0),
            incident_polarization="s",
            mu=1.0,
            eps0=1.0,
        )
        self.assertTrue(np.linalg.norm(result.coefficients) < 1e-12)
        self.assertTrue(np.linalg.norm(result.boundary_residual) < 1e-12)
        self.assertTrue(np.linalg.norm(result.reflected_total_2omega.electric) < 1e-12)

    def test_nonzero_nonlinearity_solves_2omega_boundary_residual(self):
        d = np.array(
            [
                [0.4, 0.1j, -0.2, 0.3, 0.0, 0.1],
                [0.2j, -0.3, 0.1, 0.0, 0.5, -0.4j],
                [0.1, 0.2, 0.3j, -0.2, 0.4j, 0.6],
            ],
            dtype=complex,
        )
        result = solve_single_interface_shg(
            np.diag([1.55**2, 1.65**2, 1.9**2]),
            np.diag([1.7**2, 1.82**2, 2.05**2]),
            d,
            incident_index_omega=1.0,
            incident_index_2omega=1.05,
            incident_theta_rad=math.radians(18.0),
            incident_polarization="p",
            mu=1.0,
            eps0=1.0,
        )
        self.assertEqual(result.coefficients.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.coefficients)))
        self.assertTrue(np.linalg.norm(result.boundary_residual) < 1e-10)
        total_inhomogeneous = (
            np.linalg.norm(result.inhomogeneous.ee.electric)
            + np.linalg.norm(result.inhomogeneous.oo.electric)
            + np.linalg.norm(result.inhomogeneous.eo.electric)
        )
        self.assertGreater(total_inhomogeneous, 0.0)

    def test_uniaxial_sources_use_extraordinary_ordinary_identity(self):
        d = _dense_complex_d(scale=0.4)
        eps_omega = np.diag([1.5**2, 1.5**2, 2.1**2])
        eps_2omega = np.diag([1.68**2, 1.68**2, 2.25**2])
        result = solve_single_interface_shg(
            eps_omega,
            eps_2omega,
            d,
            incident_index_omega=1.0,
            incident_index_2omega=1.0,
            incident_theta_rad=math.radians(30.0),
            incident_polarization="p",
            mu=1.0,
            eps0=1.0,
        )

        self.assertEqual(result.source_mode_basis, "uniaxial_ordinary_extraordinary")
        self.assertEqual(result.source_mode_order, ("slow", "fast"))
        self.assertTrue(np.allclose(result.sources.ee.k, 2 * result.linear_omega.transmitted_slow.k))
        self.assertTrue(np.allclose(result.sources.oo.k, 2 * result.linear_omega.transmitted_fast.k))
        self.assertTrue(np.allclose(result.sources.eo.k, result.linear_omega.transmitted_slow.k + result.linear_omega.transmitted_fast.k))

    def test_inhomogeneous_sources_share_boundary_phase_tangential_component(self):
        result = solve_single_interface_shg(
            np.eye(3) * 1.5**2,
            np.eye(3) * 1.7**2,
            np.eye(3, 6, dtype=complex),
            incident_index_omega=1.0,
            incident_index_2omega=1.0,
            incident_theta_rad=math.radians(12.0),
            incident_polarization="s",
            mu=1.0,
            eps0=1.0,
        )
        kx_values = [
            result.inhomogeneous.ee.k[0],
            result.inhomogeneous.oo.k[0],
            result.inhomogeneous.eo.k[0],
            result.reflected_s_2omega.k[0],
            result.transmitted_fast_2omega.k[0],
            result.transmitted_slow_2omega.k[0],
        ]
        for kx in kx_values[1:]:
            self.assertTrue(np.isclose(kx, kx_values[0], atol=1e-12))

    def test_complex_reflected_2omega_angle_solves_boundary_residual(self):
        d = np.array(
            [
                [0.3 + 0.1j, 0.0, -0.1j, 0.2, 0.05j, -0.04],
                [0.02, -0.25j, 0.11, 0.08j, -0.06, 0.03],
                [0.07j, 0.04, 0.19 + 0.02j, -0.05, 0.09, -0.12j],
            ],
            dtype=complex,
        )
        result = solve_single_interface_shg(
            np.diag([(1.48 + 0.02j) ** 2, (1.61 + 0.03j) ** 2, (1.82 + 0.04j) ** 2]),
            np.diag([(1.66 + 0.03j) ** 2, (1.79 + 0.04j) ** 2, (2.04 + 0.05j) ** 2]),
            d,
            incident_index_omega=1.0 + 0.01j,
            incident_index_2omega=1.08 + 0.04j,
            incident_theta_rad=math.radians(25.0),
            incident_polarization="p",
            mu=1.0,
            eps0=1.0,
        )
        self.assertEqual(result.coefficients.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.coefficients)))
        self.assertGreater(abs(np.imag(result.reflected_s_2omega.k[2])), 0.0)
        self.assertTrue(np.linalg.norm(result.boundary_residual) < 1e-9)

    def test_normal_incidence_all_complex_properties_solves_boundary_residual(self):
        d = _dense_complex_d()
        eps_omega = _complex_nondiagonal_epsilon([(1.46 + 0.02j), (1.63 + 0.03j), (1.91 + 0.04j)])
        eps_2omega = _complex_nondiagonal_epsilon([(1.59 + 0.05j), (1.78 + 0.06j), (2.08 + 0.07j)], scale=1.3)
        result = solve_single_interface_shg(
            eps_omega,
            eps_2omega,
            d,
            incident_index_omega=1.0 + 0.015j,
            incident_index_2omega=1.04 + 0.025j,
            incident_theta_rad=0.0,
            incident_polarization="s",
            mu=1.0,
            eps0=1.0,
        )
        self.assertTrue(np.all(np.isfinite(result.coefficients)))
        self.assertTrue(np.linalg.norm(result.boundary_residual) < 1e-9)
        self.assertGreater(np.max(np.abs(np.imag(d))), 0.0)
        self.assertGreater(np.max(np.abs(np.imag(eps_omega))), 0.0)
        self.assertGreater(np.max(np.abs(np.imag(eps_2omega))), 0.0)
        self.assertGreater(np.max(np.abs(eps_omega - np.diag(np.diag(eps_omega)))), 0.0)
        self.assertGreater(np.max(np.abs(eps_2omega - np.diag(np.diag(eps_2omega)))), 0.0)
        self.assertTrue(abs(result.reflected_s_2omega.k[0]) < 1e-12)
        self.assertTrue(abs(result.transmitted_fast_2omega.k[0]) < 1e-12)

    def test_phase_matched_same_complex_dielectric_at_both_frequencies_is_ill_conditioned(self):
        d = _dense_complex_d(scale=0.7)
        epsilon = np.array(
            [
                [(1.72 + 0.04j) ** 2, 0.03 + 0.012j, -0.02 + 0.008j],
                [0.03 + 0.012j, (1.93 + 0.05j) ** 2, 0.018 - 0.006j],
                [-0.02 + 0.008j, 0.018 - 0.006j, (2.18 + 0.06j) ** 2],
            ],
            dtype=complex,
        )
        result = solve_single_interface_shg(
            epsilon,
            epsilon.copy(),
            d,
            incident_index_omega=1.0 + 0.02j,
            incident_index_2omega=1.0 + 0.02j,
            incident_theta_rad=math.radians(22.0),
            incident_polarization="p",
            mu=1.0,
            eps0=1.0,
        )
        self.assertTrue(np.all(np.isfinite(result.coefficients)))
        self.assertGreater(np.max(np.abs(np.imag(epsilon))), 0.0)
        self.assertGreater(np.max(np.abs(epsilon - np.diag(np.diag(epsilon)))), 0.0)
        self.assertTrue(np.isclose(result.transmitted_2omega_modes.fast.theta_rad, result.linear_omega.transmitted_modes.fast.theta_rad, atol=1e-8))
        self.assertTrue(np.isclose(result.transmitted_2omega_modes.slow.theta_rad, result.linear_omega.transmitted_modes.slow.theta_rad, atol=1e-8))
        conditions = [
            result.inhomogeneous.ee.operator_condition,
            result.inhomogeneous.oo.operator_condition,
            result.inhomogeneous.eo.operator_condition,
        ]
        self.assertGreater(max(conditions), 1e12)
        self.assertTrue(any(field.ill_conditioned for field in (result.inhomogeneous.ee, result.inhomogeneous.oo, result.inhomogeneous.eo)))
        self.assertGreater(np.linalg.norm(result.boundary_residual), 1e-9)


def _dense_complex_d(scale=1.0):
    return scale * np.array(
        [
            [0.31 + 0.07j, -0.12 + 0.03j, 0.05 - 0.02j, 0.21 + 0.09j, -0.18j, 0.14 + 0.01j],
            [-0.04 + 0.11j, 0.17 - 0.06j, -0.09 + 0.04j, 0.08j, 0.13 - 0.05j, -0.07 + 0.02j],
            [0.16 - 0.03j, -0.10j, 0.24 + 0.08j, -0.15 + 0.05j, 0.06 + 0.12j, -0.04j],
        ],
        dtype=complex,
    )


def _complex_nondiagonal_epsilon(indices, scale=1.0):
    diag = np.diag([n**2 for n in indices]).astype(complex)
    offdiag = scale * np.array(
        [
            [0.0, 0.018 + 0.006j, -0.012 + 0.004j],
            [0.018 + 0.006j, 0.0, 0.010 - 0.003j],
            [-0.012 + 0.004j, 0.010 - 0.003j, 0.0],
        ],
        dtype=complex,
    )
    return diag + offdiag


if __name__ == "__main__":
    unittest.main()
