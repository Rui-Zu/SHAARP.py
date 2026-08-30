import unittest

import numpy as np

from shaarp.nonlinear import (
    NonlinearSource,
    compute_pnl_voigt,
    curl_curl_matrix,
    multilayer_sources,
    single_interface_sources,
    solve_inhomogeneous_field,
    solve_multilayer_inhomogeneous_fields,
    solve_single_interface_inhomogeneous_fields,
)
from shaarp.waves import Wave


class NonlinearTests(unittest.TestCase):
    def test_compute_pnl_voigt_matches_manual_formula(self):
        d = np.arange(18, dtype=float).reshape(3, 6) + 1j * 0.5
        e1 = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j])
        e2 = np.array([0.25 - 0.5j, -1.5 + 0.2j, 3 + 0.1j])
        products = np.array(
            [
                e1[0] * e2[0],
                e1[1] * e2[1],
                e1[2] * e2[2],
                e1[1] * e2[2] + e1[2] * e2[1],
                e1[0] * e2[2] + e1[2] * e2[0],
                e1[1] * e2[0] + e1[0] * e2[1],
            ]
        )
        expected = 2.0 * d @ products
        actual = compute_pnl_voigt(d, e1, e2, factor=2.0)
        self.assertTrue(np.allclose(actual, expected))

    def test_single_interface_sources_have_expected_wavevectors_and_mix_factor(self):
        d = np.eye(3, 6, dtype=complex)
        fast = Wave(k=np.array([1.0, 0.0, 2.0]), electric=np.array([1.0, 2.0, 3.0]), omega=1.0)
        slow = Wave(k=np.array([0.5, 0.0, 1.5]), electric=np.array([4.0, 5.0, 6.0]), omega=1.0)
        sources = single_interface_sources(d, fast, slow)
        self.assertTrue(np.allclose(sources.ee.k, 2 * fast.k))
        self.assertTrue(np.allclose(sources.oo.k, 2 * slow.k))
        self.assertTrue(np.allclose(sources.eo.k, fast.k + slow.k))
        self.assertTrue(np.allclose(sources.eo.polarization, 2 * compute_pnl_voigt(d, fast.electric, slow.electric)))
        self.assertEqual(sources.ee.omega, 2.0)

    def test_curl_curl_matrix_matches_plane_wave_sign_convention(self):
        k = np.array([0.4, -0.2, 1.3])
        e = np.array([1.0 + 0.5j, -2.0, 0.25j])
        expected = -np.cross(k, np.cross(k, e))
        self.assertTrue(np.allclose(curl_curl_matrix(k) @ e, expected))

    def test_solve_inhomogeneous_field_has_small_residual(self):
        epsilon = np.array(
            [
                [2.4 + 0.10j, 0.07 - 0.02j, -0.03 + 0.04j],
                [0.07 - 0.02j, 2.9 + 0.05j, 0.05 + 0.01j],
                [-0.03 + 0.04j, 0.05 + 0.01j, 3.3 + 0.08j],
            ],
            dtype=complex,
        )
        source = NonlinearSource(
            label="test",
            k=np.array([0.31, 0.0, 1.17], dtype=complex),
            polarization=np.array([0.2 + 0.5j, -0.1 + 0.2j, 0.3 - 0.05j]),
            omega=1.7,
        )
        field = solve_inhomogeneous_field(epsilon, source, mu=1.0, eps0=1.0)
        self.assertEqual(field.label, "test")
        self.assertTrue(np.all(np.isfinite(field.electric)))
        self.assertTrue(np.linalg.norm(field.residual) < 1e-12)
        self.assertTrue(np.isfinite(field.operator_condition))
        self.assertFalse(field.ill_conditioned)

    def test_near_phase_matching_marks_inhomogeneous_operator_ill_conditioned(self):
        epsilon = np.diag([1.0 - 1e-13, 1.0 - 2e-13, 1.0]).astype(complex)
        source = NonlinearSource(
            label="near_phase_match",
            k=np.array([0.0, 0.0, 1.0], dtype=complex),
            polarization=np.array([0.1, 0.0, 0.0], dtype=complex),
            omega=1.0,
        )
        field = solve_inhomogeneous_field(epsilon, source, mu=1.0, eps0=1.0)
        self.assertGreater(field.operator_condition, 1e12)
        self.assertTrue(field.ill_conditioned)
        self.assertEqual(field.solution_method, "solve_ill_conditioned")

    def test_near_phase_matching_can_use_explicit_minimum_norm_policy(self):
        epsilon = np.diag([1.0 - 1e-13, 1.0 - 2e-13, 1.0]).astype(complex)
        source = NonlinearSource(
            label="near_phase_match",
            k=np.array([0.0, 0.0, 1.0], dtype=complex),
            polarization=np.array([0.1, 0.0, 0.0], dtype=complex),
            omega=1.0,
        )

        default = solve_inhomogeneous_field(epsilon, source, mu=1.0, eps0=1.0)
        minimum_norm = solve_inhomogeneous_field(
            epsilon,
            source,
            mu=1.0,
            eps0=1.0,
            solution_policy="minimum_norm_if_ill_conditioned",
        )

        self.assertEqual(default.solution_method, "solve_ill_conditioned")
        self.assertEqual(minimum_norm.solution_method, "minimum_norm_ill_conditioned")
        self.assertGreater(np.linalg.norm(default.electric), 1e10)
        self.assertLess(np.linalg.norm(minimum_norm.electric), 1e-9)
        self.assertGreater(np.linalg.norm(minimum_norm.residual), 0.0)

    def test_solve_inhomogeneous_field_rejects_unknown_solution_policy(self):
        source = NonlinearSource(
            label="bad_policy",
            k=np.array([0.0, 0.0, 1.0], dtype=complex),
            polarization=np.array([0.1, 0.0, 0.0], dtype=complex),
            omega=1.0,
        )

        with self.assertRaisesRegex(ValueError, "solution_policy"):
            solve_inhomogeneous_field(np.eye(3), source, mu=1.0, eps0=1.0, solution_policy="guess")

    def test_exact_phase_matching_singular_operator_uses_explicit_least_squares_diagnostic(self):
        epsilon = np.eye(3, dtype=complex)
        source = NonlinearSource(
            label="exact_phase_match",
            k=np.array([0.0, 0.0, 1.0], dtype=complex),
            polarization=np.array([0.1 + 0.2j, -0.3j, 0.0], dtype=complex),
            omega=1.0,
        )

        field = solve_inhomogeneous_field(epsilon, source, mu=1.0, eps0=1.0)

        self.assertTrue(field.ill_conditioned)
        self.assertFalse(np.isfinite(field.operator_condition))
        self.assertEqual(field.solution_method, "least_squares_singular")
        self.assertTrue(np.all(np.isfinite(field.electric)))
        self.assertGreater(np.linalg.norm(field.residual), 0.0)

    def test_single_interface_inhomogeneous_fields_solve_all_three_sources(self):
        d = np.array(
            [
                [1 + 0.1j, 0.2, -0.3j, 0.4, 0.5j, -0.6],
                [0.1, -0.2j, 0.3, -0.4j, 0.5, 0.6j],
                [-0.2, 0.3j, 0.4, 0.5j, -0.6, 0.7],
            ],
            dtype=complex,
        )
        fast = Wave(k=np.array([0.2, 0.0, 0.9]), electric=np.array([0.8, 0.1j, -0.2]), omega=1.1)
        slow = Wave(k=np.array([0.15, 0.0, 1.1]), electric=np.array([0.05j, 0.7, 0.3]), omega=1.1)
        sources = single_interface_sources(d, fast, slow)
        epsilon = np.diag([2.0 + 0.2j, 2.3 + 0.1j, 2.7 + 0.15j])
        fields = solve_single_interface_inhomogeneous_fields(epsilon, sources, mu=1.0, eps0=1.0)
        self.assertTrue(np.linalg.norm(fields.ee.residual) < 1e-12)
        self.assertTrue(np.linalg.norm(fields.oo.residual) < 1e-12)
        self.assertTrue(np.linalg.norm(fields.eo.residual) < 1e-12)

    def test_multilayer_sources_have_all_shaarp_ml_wavevectors_and_mix_factors(self):
        d = np.array(
            [
                [1 + 0.2j, -0.3j, 0.4, 0.05 + 0.1j, -0.2, 0.7j],
                [-0.2 + 0.3j, 0.6, 0.1j, -0.45, 0.25j, 0.8],
                [0.15j, -0.35, 0.55 + 0.05j, 0.2, -0.1j, 0.9],
            ],
            dtype=complex,
        )
        f1 = Wave(k=np.array([0.2, 0.0, 1.1]), electric=np.array([0.8 + 0.1j, 0.05, -0.2j]), omega=1.3)
        f2 = Wave(k=np.array([0.2, 0.0, 1.4]), electric=np.array([0.04j, 0.9 - 0.2j, 0.12]), omega=1.3)
        b1 = Wave(k=np.array([0.2, 0.0, -1.05]), electric=np.array([-0.3, 0.2j, 0.7 + 0.1j]), omega=1.3)
        b2 = Wave(k=np.array([0.2, 0.0, -1.35]), electric=np.array([0.6j, -0.25, 0.5 - 0.05j]), omega=1.3)

        sources = multilayer_sources(d, f1, f2, b1, b2)
        self.assertEqual([source.label for source in sources.as_list()], ["F1F1", "F2F2", "F1F2", "B1B1", "B2B2", "B1B2", "F1B1", "F2B2", "F1B2", "F2B1"])
        self.assertTrue(np.allclose(sources.f1f1.k, 2 * f1.k))
        self.assertTrue(np.allclose(sources.f2f2.k, 2 * f2.k))
        self.assertTrue(np.allclose(sources.f1f2.k, f1.k + f2.k))
        self.assertTrue(np.allclose(sources.b1b1.k, 2 * b1.k))
        self.assertTrue(np.allclose(sources.b2b2.k, 2 * b2.k))
        self.assertTrue(np.allclose(sources.b1b2.k, b1.k + b2.k))
        self.assertTrue(np.allclose(sources.f1b1.k, f1.k + b1.k))
        self.assertTrue(np.allclose(sources.f2b2.k, f2.k + b2.k))
        self.assertTrue(np.allclose(sources.f1b2.k, f1.k + b2.k))
        self.assertTrue(np.allclose(sources.f2b1.k, f2.k + b1.k))
        self.assertTrue(np.allclose(sources.f1f1.polarization, compute_pnl_voigt(d, f1.electric, f1.electric)))
        self.assertTrue(np.allclose(sources.f1f2.polarization, 2 * compute_pnl_voigt(d, f1.electric, f2.electric)))
        self.assertTrue(np.allclose(sources.b1b2.polarization, 2 * compute_pnl_voigt(d, b1.electric, b2.electric)))
        self.assertTrue(np.allclose(sources.f2b1.polarization, 2 * compute_pnl_voigt(d, f2.electric, b1.electric)))
        self.assertTrue(all(source.omega == 2.6 for source in sources.as_list()))

    def test_multilayer_sources_reject_omega_mismatch(self):
        wave = Wave(k=np.array([0.1, 0.0, 0.8]), electric=np.array([1.0, 0.0, 0.0]), omega=1.0)
        mismatch = Wave(k=np.array([0.1, 0.0, -0.8]), electric=np.array([0.0, 1.0, 0.0]), omega=1.1)
        with self.assertRaisesRegex(ValueError, "same omega"):
            multilayer_sources(np.eye(3, 6), wave, wave, wave, mismatch)

    def test_multilayer_inhomogeneous_fields_solve_all_ten_complex_nondiagonal_sources(self):
        d = np.array(
            [
                [0.3 + 0.2j, -0.4j, 0.7, 0.2 - 0.1j, 0.05j, -0.6],
                [0.1j, 0.5 + 0.15j, -0.25, 0.45j, 0.8, -0.2 + 0.05j],
                [-0.15, 0.25j, 0.35 + 0.2j, -0.5, 0.65j, 0.4],
            ],
            dtype=complex,
        )
        f1 = Wave(k=np.array([0.31, 0.0, 1.12]), electric=np.array([0.8 + 0.1j, 0.04j, -0.22]), omega=1.4)
        f2 = Wave(k=np.array([0.31, 0.0, 1.37]), electric=np.array([0.03, 0.91 - 0.18j, 0.11j]), omega=1.4)
        b1 = Wave(k=np.array([0.31, 0.0, -1.09]), electric=np.array([-0.28j, 0.24, 0.68 + 0.08j]), omega=1.4)
        b2 = Wave(k=np.array([0.31, 0.0, -1.41]), electric=np.array([0.58, -0.21j, 0.47 - 0.04j]), omega=1.4)
        epsilon = np.array(
            [
                [2.6 + 0.15j, 0.08 - 0.03j, -0.05 + 0.02j],
                [0.04 + 0.07j, 3.1 + 0.11j, 0.06 - 0.01j],
                [-0.02 + 0.05j, 0.03 + 0.04j, 3.7 + 0.09j],
            ],
            dtype=complex,
        )

        sources = multilayer_sources(d, f1, f2, b1, b2)
        fields = solve_multilayer_inhomogeneous_fields(epsilon, sources, mu=1.0, eps0=1.0)
        self.assertEqual([field.label for field in fields.as_list()], [source.label for source in sources.as_list()])
        self.assertTrue(all(np.linalg.norm(field.residual) < 1e-11 for field in fields.as_list()))
        self.assertTrue(all(np.isfinite(field.operator_condition) for field in fields.as_list()))


if __name__ == "__main__":
    unittest.main()
