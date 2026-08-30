import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_v1.json"


class MakerFringesBenchmarkTests(unittest.TestCase):
    def test_maker_fringes_benchmarks_are_small_slow_path_regressions(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "python_staged_maker_fringes_regression_not_mathematica_validation")
        self.assertEqual(payload["case_count"], 3)
        self.assertEqual(payload["total_angle_count"], 10)
        self.assertTrue(payload["slow_path_policy"]["maker_fringes_can_be_slow"])
        self.assertTrue(payload["slow_path_policy"]["default_saved_grid_is_intentionally_small"])
        self.assertTrue(payload["slow_path_policy"]["phase_matching_like_case_is_diagnostic_not_small_residual_claim"])
        self.assertTrue(payload["slow_path_policy"]["mathematica_reference_still_required_for_validation_claims"])

    def test_maker_fringes_benchmarks_cover_requested_angle_extremes(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        all_angles = {theta for case in payload["cases"] for theta in case["theta_deg"]}
        self.assertIn(0.0, all_angles)
        self.assertTrue(any(theta < 0.0 for theta in all_angles))
        self.assertGreaterEqual(max(all_angles), 60.0)
        self.assertTrue(any(case["phase_matching_like"] for case in payload["cases"]))

    def test_maker_fringes_benchmarks_have_complex_nondiagonal_tensors(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            metrics = case["metrics"]
            self.assertGreater(metrics["max_abs_epsilon_omega_imag"], 0.0)
            self.assertGreater(metrics["max_abs_epsilon_2omega_imag"], 0.0)
            self.assertGreater(metrics["max_abs_epsilon_omega_offdiag"], 0.0)
            self.assertGreater(metrics["max_abs_d_imag"], 0.0)

    def test_maker_fringes_benchmark_residuals_are_small(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            metrics = case["metrics"]
            self.assertLess(metrics["max_fundamental_residual_l2"], 1e-8)
            if metrics["small_shg_residual_expected"]:
                self.assertLess(metrics["max_shg_residual_l2"], 1e-7)
            else:
                self.assertTrue(case["phase_matching_like"])
                self.assertGreater(metrics["max_shg_residual_l2"], 1e-7)

    def test_maker_fringes_benchmarks_store_copy_shaped_values(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            self.assertEqual(len(case["outputs"]["list_mf_para"]), case["angle_count"])
            self.assertEqual(len(case["outputs"]["list_mf_perp"]), case["angle_count"])
            self.assertEqual(len(case["outputs"]["parallel_amplitude"]), case["angle_count"])
            self.assertEqual(len(case["outputs"]["perpendicular_amplitude"]), case["angle_count"])
            self.assertEqual(len(case["outputs"]["transmitted_2omega_jones_sp"]), case["angle_count"])
            self.assertEqual(len(case["outputs"]["transmitted_2omega_jones_sp"][0]), 2)
            self.assertEqual(len(case["outputs"]["list_mf_para"][0]), 2)
            self.assertEqual(len(case["outputs"]["list_mf_perp"][0]), 2)


if __name__ == "__main__":
    unittest.main()
