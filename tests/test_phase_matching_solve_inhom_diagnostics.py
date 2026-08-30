import json
import unittest
from pathlib import Path


DIAGNOSTIC_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "phase_matching_solve_inhom_diagnostics_v1.json"
)


class PhaseMatchingSolveInhomDiagnosticsTests(unittest.TestCase):
    def test_phase_matching_diagnostic_records_singular_near_null_difference(self):
        payload = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["case_id"], "maker_fringes_phase_matching_like")
        self.assertEqual(payload["status"], "phase_matching_solve_inhom_diagnostic")
        self.assertEqual(len(payload["points"]), 3)

        failing_forward_sources = [
            source
            for point in payload["points"]
            for source in point["sources"][:2]
        ]
        self.assertTrue(failing_forward_sources)
        for source in failing_forward_sources:
            with self.subTest(theta=source["theta_deg"], label=source["label"]):
                self.assertGreater(source["operator_condition"], 1e14)
                self.assertEqual(source["operator_rank_tol_1e_minus_12"], 2)
                self.assertGreater(source["field_delta_norm"], 1e12)
                self.assertGreater(source["field_delta_near_null_fraction"], 0.999999)
                self.assertLess(source["python_relative_residual"], 1e-14)
                self.assertLess(source["mathematica_relative_residual"], 1e-12)
                self.assertEqual(source["minimum_norm_policy_solution_method"], "minimum_norm_ill_conditioned")
                self.assertLess(source["minimum_norm_policy_field_norm"], 10.0)
                self.assertGreater(source["default_to_minimum_norm_delta_norm"], 1e12)

        mixed_sources = [
            source
            for point in payload["points"]
            for source in point["sources"][2:3]
        ]
        for source in mixed_sources:
            with self.subTest(theta=source["theta_deg"], label=source["label"]):
                self.assertLess(source["operator_condition"], 100.0)
                self.assertLess(source["field_delta_norm"], 1e-12)


if __name__ == "__main__":
    unittest.main()
