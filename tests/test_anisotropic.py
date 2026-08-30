import math
import unittest

import numpy as np

from shaarp.anisotropic import (
    AnisotropicMode,
    SnellModes,
    identify_uniaxial_modes,
    modes_for_direction,
    solve_snell_modes,
    track_mode_branches,
)


class AnisotropicTests(unittest.TestCase):
    def test_modes_for_isotropic_direction(self):
        eps = np.eye(3) * 4.0
        fast, slow = modes_for_direction(eps, np.array([0.0, 0.0, 1.0]))
        self.assertTrue(np.isclose(fast.refractive_index, 2.0))
        self.assertTrue(np.isclose(slow.refractive_index, 2.0))
        self.assertTrue(np.isclose(np.vdot(fast.direction, fast.electric_direction), 0.0))
        self.assertTrue(np.isclose(np.vdot(slow.direction, slow.electric_direction), 0.0))


    def test_snell_for_isotropic_medium_matches_scalar_formula(self):
        eps = np.eye(3) * 4.0
        theta_i = math.radians(30.0)
        modes = solve_snell_modes(eps, incident_index=1.0, incident_theta_rad=theta_i)
        expected = math.asin(math.sin(theta_i) / 2.0)
        self.assertTrue(np.isclose(modes.fast.theta_rad, expected, atol=1e-10))
        self.assertTrue(np.isclose(modes.slow.theta_rad, expected, atol=1e-10))
        self.assertTrue(np.isclose(modes.fast.refractive_index, 2.0))
        self.assertTrue(np.isclose(modes.slow.refractive_index, 2.0))

    def test_reflected_snell_for_isotropic_medium_matches_scalar_formula(self):
        eps = np.eye(3) * 4.0
        theta_i = math.radians(30.0)
        modes = solve_snell_modes(eps, incident_index=1.0, incident_theta_rad=theta_i, reflected=True)
        expected = math.pi - math.asin(math.sin(theta_i) / 2.0)
        for mode in (modes.fast, modes.slow):
            self.assertTrue(np.isclose(mode.theta_rad, expected, atol=1e-10))
            self.assertTrue(np.isclose(mode.refractive_index * np.sin(mode.theta_rad), np.sin(theta_i), atol=1e-10))
            self.assertLess(np.real(mode.direction[2]), 0.0)

    def test_negative_incidence_snell_preserves_tangential_sign(self):
        eps = np.eye(3) * 4.0
        theta_i = math.radians(-30.0)
        modes = solve_snell_modes(eps, incident_index=1.0, incident_theta_rad=theta_i)
        reflected = solve_snell_modes(eps, incident_index=1.0, incident_theta_rad=theta_i, reflected=True)
        expected = math.asin(math.sin(theta_i) / 2.0)
        reflected_expected = -math.pi - expected
        for mode in (modes.fast, modes.slow):
            self.assertTrue(np.isclose(mode.theta_rad, expected, atol=1e-10))
            self.assertTrue(np.isclose(mode.refractive_index * np.sin(mode.theta_rad), np.sin(theta_i), atol=1e-10))
            self.assertGreater(np.real(mode.direction[2]), 0.0)
        for mode in (reflected.fast, reflected.slow):
            self.assertTrue(np.isclose(mode.theta_rad, reflected_expected, atol=1e-10))
            self.assertTrue(np.isclose(mode.refractive_index * np.sin(mode.theta_rad), np.sin(theta_i), atol=1e-10))
            self.assertLess(np.real(mode.direction[2]), 0.0)

    def test_complex_isotropic_snell_matches_scalar_formula(self):
        n2 = 1.8 + 0.12j
        theta_i = math.radians(32.0)
        modes = solve_snell_modes(np.eye(3) * n2**2, incident_index=1.0, incident_theta_rad=theta_i)
        expected_theta = np.arcsin(np.sin(theta_i) / n2)
        for mode in (modes.fast, modes.slow):
            self.assertTrue(np.isclose(mode.theta_rad, expected_theta, atol=1e-10))
            self.assertTrue(np.isclose(mode.refractive_index, n2, atol=1e-10))
            self.assertTrue(np.isclose(mode.refractive_index * np.sin(mode.theta_rad), np.sin(theta_i), atol=1e-10))

    def test_reflected_complex_isotropic_snell_matches_scalar_formula(self):
        n2 = 1.8 + 0.12j
        theta_i = math.radians(32.0)
        modes = solve_snell_modes(np.eye(3) * n2**2, incident_index=1.0, incident_theta_rad=theta_i, reflected=True)
        expected_theta = np.pi - np.arcsin(np.sin(theta_i) / n2)
        for mode in (modes.fast, modes.slow):
            self.assertTrue(np.isclose(mode.theta_rad, expected_theta, atol=1e-10))
            self.assertTrue(np.isclose(mode.refractive_index, n2, atol=1e-10))
            self.assertTrue(np.isclose(mode.refractive_index * np.sin(mode.theta_rad), np.sin(theta_i), atol=1e-10))
            self.assertLess(np.real(mode.direction[2]), 0.0)

    def test_complex_anisotropic_snell_satisfies_equation(self):
        eps = np.array(
            [
                [2.4 + 0.08j, 0.03 - 0.01j, 0.02 + 0.015j],
                [0.03 - 0.01j, 2.9 + 0.11j, -0.025 + 0.02j],
                [0.02 + 0.015j, -0.025 + 0.02j, 3.5 + 0.16j],
            ],
            dtype=complex,
        )
        theta_i = math.radians(24.0)
        modes = solve_snell_modes(eps, incident_index=1.0, incident_theta_rad=theta_i)
        target = np.sin(theta_i)
        for mode in (modes.fast, modes.slow):
            self.assertTrue(np.isclose(mode.refractive_index * np.sin(mode.theta_rad), target, atol=1e-9))
            projector = np.eye(3, dtype=complex) - np.outer(mode.direction, mode.direction)
            lhs = projector @ mode.electric_direction
            rhs = mode.eta * (eps @ mode.electric_direction)
            self.assertTrue(np.linalg.norm(lhs - rhs) < 1e-8)

    def test_positive_uniaxial_identity_is_not_fast_slow_order(self):
        no = 1.5
        ne = 2.1
        eps = np.diag([no**2, no**2, ne**2])
        modes = solve_snell_modes(eps, incident_index=1.0, incident_theta_rad=math.radians(30.0))
        identified = identify_uniaxial_modes(eps, modes)

        self.assertEqual(identified.ordinary_branch, "fast")
        self.assertEqual(identified.extraordinary_branch, "slow")
        self.assertGreater(identified.ordinary_alignment, 0.999)
        self.assertTrue(np.isclose(identified.ordinary.refractive_index, no, atol=1e-10))

    def test_rotated_uniaxial_identity_uses_optic_axis_not_index_sort_only(self):
        no = 1.55
        ne = 2.05
        tilt = math.radians(28.0)
        rotation_y = np.array(
            [
                [math.cos(tilt), 0.0, math.sin(tilt)],
                [0.0, 1.0, 0.0],
                [-math.sin(tilt), 0.0, math.cos(tilt)],
            ]
        )
        eps_principal = np.diag([no**2, no**2, ne**2])
        eps_lab = rotation_y @ eps_principal @ rotation_y.T
        optic_axis = rotation_y @ np.array([0.0, 0.0, 1.0])
        modes = solve_snell_modes(eps_lab, incident_index=1.0, incident_theta_rad=math.radians(22.0))

        inferred = identify_uniaxial_modes(eps_lab, modes)
        supplied = identify_uniaxial_modes(eps_lab, modes, optic_axis_lab=optic_axis)

        self.assertEqual(inferred.ordinary_branch, supplied.ordinary_branch)
        self.assertEqual(inferred.extraordinary_branch, supplied.extraordinary_branch)
        self.assertGreater(inferred.ordinary_alignment, 0.999)

    def test_reflected_complex_anisotropic_snell_satisfies_equation(self):
        eps = np.array(
            [
                [2.4 + 0.08j, 0.03 - 0.01j, 0.02 + 0.015j],
                [0.03 - 0.01j, 2.9 + 0.11j, -0.025 + 0.02j],
                [0.02 + 0.015j, -0.025 + 0.02j, 3.5 + 0.16j],
            ],
            dtype=complex,
        )
        theta_i = math.radians(24.0)
        modes = solve_snell_modes(eps, incident_index=1.0, incident_theta_rad=theta_i, reflected=True)
        target = np.sin(theta_i)
        for mode in (modes.fast, modes.slow):
            self.assertTrue(np.isclose(mode.refractive_index * np.sin(mode.theta_rad), target, atol=1e-9))
            self.assertLess(np.real(mode.direction[2]), 0.0)
            projector = np.eye(3, dtype=complex) - np.outer(mode.direction, mode.direction)
            lhs = projector @ mode.electric_direction
            rhs = mode.eta * (eps @ mode.electric_direction)
            self.assertTrue(np.linalg.norm(lhs - rhs) < 1e-8)

    def test_biaxial_branch_tracking_preserves_continuity_through_swapped_order(self):
        eps = np.array(
            [
                [1.58**2, 0.025, -0.018],
                [0.025, 1.74**2, 0.021],
                [-0.018, 0.021, 2.03**2],
            ],
            dtype=complex,
        )
        first = solve_snell_modes(eps, incident_index=1.0, incident_theta_rad=math.radians(18.0))
        second_actual = solve_snell_modes(eps, incident_index=1.0, incident_theta_rad=math.radians(19.0))
        second_swapped = SnellModes(fast=second_actual.slow, slow=second_actual.fast)

        tracked = track_mode_branches([first, second_swapped])

        self.assertFalse(tracked[1].ambiguous)
        self.assertIs(tracked[1].branch_1, second_actual.fast)
        self.assertIs(tracked[1].branch_2, second_actual.slow)
        self.assertEqual(tracked[1].branch_1_source_label, "fast")
        self.assertEqual(tracked[1].branch_2_source_label, "slow")
        self.assertGreater(tracked[1].branch_1_alignment, 0.99)
        self.assertGreater(tracked[1].branch_2_alignment, 0.99)

    def test_branch_tracking_marks_ambiguous_polarization_assignment(self):
        first = SnellModes(
            fast=_test_mode("fast", [1.0, 0.0, 0.0]),
            slow=_test_mode("slow", [0.0, 1.0, 0.0]),
        )
        root2 = 1 / np.sqrt(2)
        second = SnellModes(
            fast=_test_mode("fast", [root2, root2, 0.0]),
            slow=_test_mode("slow", [root2, -root2, 0.0]),
        )

        tracked = track_mode_branches([first, second])

        self.assertTrue(tracked[1].ambiguous)
        self.assertTrue(np.isclose(tracked[1].branch_1_alignment, root2))
        self.assertTrue(np.isclose(tracked[1].branch_2_alignment, root2))

def _test_mode(label, electric):
    return AnisotropicMode(
        label=label,
        refractive_index=1.0,
        theta_rad=0.0,
        direction=np.array([0.0, 0.0, 1.0], dtype=complex),
        electric_direction=np.array(electric, dtype=complex),
        eta=1.0,
    )


if __name__ == "__main__":
    unittest.main()
