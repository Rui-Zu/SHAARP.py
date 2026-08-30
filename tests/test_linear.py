import math
import unittest

import numpy as np

from shaarp.boundary import isotropic_fresnel_coefficients
from shaarp.anisotropic import track_mode_branches
from shaarp.linear import solve_fresnel_coefficient_sweep, solve_linear_interface, solve_linear_interface_sweep


class LinearInterfaceTests(unittest.TestCase):
    def test_isotropic_s_incidence_matches_fresnel(self):
        n1, n2 = 1.0, 1.7
        theta = math.radians(35.0)
        result = solve_linear_interface(
            np.eye(3) * n2**2,
            incident_index=n1,
            incident_theta_rad=theta,
            incident_polarization="s",
        )
        closed = isotropic_fresnel_coefficients(n1, n2, theta)
        self.assertTrue(np.isclose(result.r_s, closed["rs"], atol=1e-12))
        self.assertTrue(abs(result.r_p) < 1e-12)
        # In isotropic media, fast/slow polarization labels are degenerate.
        transmitted_total = result.transmitted_fast.electric + result.transmitted_slow.electric
        self.assertTrue(np.isclose(transmitted_total[1], closed["ts"], atol=1e-12))

    def test_isotropic_p_incidence_matches_fresnel(self):
        n1, n2 = 1.0, 1.7
        theta = math.radians(35.0)
        result = solve_linear_interface(
            np.eye(3) * n2**2,
            incident_index=n1,
            incident_theta_rad=theta,
            incident_polarization="p",
        )
        closed = isotropic_fresnel_coefficients(n1, n2, theta)
        self.assertTrue(abs(result.r_s) < 1e-12)
        self.assertTrue(np.isclose(result.r_p, closed["rp"], atol=1e-12))

    def test_anisotropic_lossless_interface_runs(self):
        eps = np.diag([2.0**2, 2.3**2, 2.8**2])
        result = solve_linear_interface(
            eps,
            incident_index=1.0,
            incident_theta_rad=math.radians(20.0),
            incident_polarization="s",
        )
        self.assertEqual(result.coefficients.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.coefficients)))
        self.assertGreater(np.real(result.transmitted_modes.slow.refractive_index), 0)

    def test_linear_interface_sweep_attaches_branch_tracked_waves(self):
        eps = np.array(
            [
                [1.57**2, 0.021, -0.015],
                [0.021, 1.76**2, 0.018],
                [-0.015, 0.018, 2.05**2],
            ],
            dtype=complex,
        )
        angles = [math.radians(value) for value in (12.0, 18.0, 24.0, 30.0)]

        sweep = solve_linear_interface_sweep(
            eps,
            incident_index=1.0,
            incident_theta_rad=angles,
            incident_polarization="p",
            omega=1.2,
        )
        expected = track_mode_branches([item.linear.transmitted_modes for item in sweep])

        self.assertEqual(len(sweep), len(angles))
        for item, branches in zip(sweep, expected):
            self.assertEqual(item.transmitted_branches.branch_1_source_label, branches.branch_1_source_label)
            self.assertEqual(item.transmitted_branches.branch_2_source_label, branches.branch_2_source_label)
            if branches.branch_1_source_label == "fast":
                self.assertTrue(np.allclose(item.transmitted_branch_1.k, item.linear.transmitted_fast.k))
            else:
                self.assertTrue(np.allclose(item.transmitted_branch_1.k, item.linear.transmitted_slow.k))
            self.assertTrue(np.all(np.isfinite(item.linear.coefficients)))

    def test_fresnel_coefficient_sweep_matches_isotropic_power_coefficients(self):
        n1, n2 = 1.0, 1.7
        angles = [0.0, 20.0, 40.0]

        sweep = solve_fresnel_coefficient_sweep(
            np.eye(3) * n2**2,
            incident_index=n1,
            theta_deg=angles,
        )

        np.testing.assert_allclose(sweep.theta_deg, angles)
        for index, theta_deg in enumerate(angles):
            closed = isotropic_fresnel_coefficients(n1, n2, math.radians(theta_deg))
            sin_t = n1 / n2 * math.sin(math.radians(theta_deg))
            cos_i = math.cos(math.radians(theta_deg))
            cos_t = np.sqrt(1 - sin_t**2 + 0j)
            transmission_scale = np.real(n2 * cos_t) / (np.real(n1) * cos_i)
            self.assertTrue(np.isclose(sweep.rp[index], abs(closed["rp"]) ** 2, atol=1e-12))
            self.assertTrue(np.isclose(sweep.rs[index], abs(closed["rs"]) ** 2, atol=1e-12))
            self.assertTrue(np.isclose(sweep.tp[index], transmission_scale * abs(closed["tp"]) ** 2, atol=1e-12))
            self.assertTrue(np.isclose(sweep.ts[index], transmission_scale * abs(closed["ts"]) ** 2, atol=1e-12))
            self.assertTrue(np.isclose(sweep.rp[index] + sweep.tp[index], 1.0, atol=1e-12))
            self.assertTrue(np.isclose(sweep.rs[index] + sweep.ts[index], 1.0, atol=1e-12))

    def test_fresnel_coefficient_sweep_exposes_shaarp_ml_copy_order(self):
        sweep = solve_fresnel_coefficient_sweep(
            np.eye(3) * 1.5**2,
            incident_index=1.0,
            theta_deg=[0.0, 15.0],
        )

        rp, rs, tp, ts = sweep.shaarp_ml_copy_lists()
        self.assertEqual(len(sweep.list_fresnel), 4)
        np.testing.assert_allclose(rp[:, 0], [0.0, 15.0])
        np.testing.assert_allclose(rs[:, 0], [0.0, 15.0])
        np.testing.assert_allclose(tp[:, 0], [0.0, 15.0])
        np.testing.assert_allclose(ts[:, 0], [0.0, 15.0])
        np.testing.assert_allclose(rp[:, 1], sweep.rp)
        np.testing.assert_allclose(rs[:, 1], sweep.rs)
        np.testing.assert_allclose(tp[:, 1], sweep.tp)
        np.testing.assert_allclose(ts[:, 1], sweep.ts)

    def test_fresnel_coefficient_sweep_default_grid_matches_shaarp_ml_range(self):
        sweep = solve_fresnel_coefficient_sweep(
            np.eye(3) * 1.4**2,
            incident_index=1.0,
            theta_step_deg=30.0,
        )

        np.testing.assert_allclose(sweep.theta_deg, [0.0, 30.0, 60.0])
        self.assertEqual(sweep.rp.shape, (3,))
        self.assertTrue(np.all(np.isfinite(sweep.tp)))

    def test_fresnel_coefficient_sweep_rejects_invalid_angle_grid(self):
        with self.assertRaisesRegex(ValueError, "theta_step_deg"):
            solve_fresnel_coefficient_sweep(np.eye(3), incident_index=1.0, theta_step_deg=0.0)
        with self.assertRaisesRegex(ValueError, "0 <= theta_deg < 90"):
            solve_fresnel_coefficient_sweep(np.eye(3), incident_index=1.0, theta_deg=[0.0, 90.0])
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            solve_fresnel_coefficient_sweep(np.eye(3), incident_index=1.0, theta_deg=[[0.0, 10.0]])


if __name__ == "__main__":
    unittest.main()
