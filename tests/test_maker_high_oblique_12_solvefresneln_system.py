import json
import unittest

from benchmarks.compare_maker_high_oblique_12_solvefresneln_system import (
    SYSTEM_PATH,
    build_summary,
)


class MakerHighOblique12SolveFresnelNSystemTests(unittest.TestCase):
    def test_system_artifact_targets_theta_12(self):
        payload = json.loads(SYSTEM_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_maker_high_oblique_12_solvefresneln_system_exported")
        self.assertEqual(payload["case_id"], "maker_fringes_moderate_high_oblique")
        self.assertEqual(payload["theta_deg"], 12.0)
        self.assertEqual(payload["solvefresneln_system"]["nsolve_solution_count"], 1)

    def test_system_matches_after_shaarp_standard_eigenbasis(self):
        payload = json.loads(SYSTEM_PATH.read_text(encoding="utf-8"))
        summary = build_summary(payload, atol=1e-10)

        self.assertEqual(summary["status"], "theta12_solvefresneln_matrix_and_coefficients_match")
        self.assertTrue(summary["agreement_claimed"])
        self.assertLess(summary["max_abs_B_plus_python_rhs"], 1e-12)
        self.assertLess(summary["max_aligned_matrix_relative_error"], 1e-12)
        self.assertLess(summary["max_mathematica_direct_solve_vs_nsolve_coeff_error"], 1e-12)
        self.assertLess(summary["max_basis_electric_relative_error"], 1e-12)
        self.assertLess(summary["max_basis_k_abs_error"], 1e-12)
        self.assertLess(summary["max_aligned_coefficient_abs_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
