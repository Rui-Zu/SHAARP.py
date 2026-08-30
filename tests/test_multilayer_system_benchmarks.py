import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks" / "multilayer_system_benchmarks_v1.json"


class MultilayerSystemBenchmarkTests(unittest.TestCase):
    def test_multilayer_system_benchmarks_cover_angles_orientations_and_polarizations(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "python_staged_multilayer_system_regression_not_mathematica_validation")
        self.assertEqual(payload["case_count"], 20)
        cases = payload["cases"]
        self.assertEqual({case["theta_deg"] for case in cases}, {0.0, 8.0, 18.0, 33.0, 52.0})
        self.assertEqual({case["orientation_index"] for case in cases}, {0, 1, 2, 3})
        self.assertEqual({case["incident_polarization"] for case in cases}, {"s", "p"})

    def test_multilayer_system_benchmarks_have_complex_nondiagonal_tensors_and_ten_sources(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            metrics = case["metrics"]
            self.assertGreater(metrics["max_abs_epsilon_omega_imag"], 0.0)
            self.assertGreater(metrics["max_abs_epsilon_2omega_imag"], 0.0)
            self.assertGreater(metrics["max_abs_epsilon_omega_offdiag"], 0.0)
            self.assertGreater(metrics["max_abs_d_imag"], 0.0)
            self.assertEqual(metrics["source_count"], 10)

    def test_multilayer_system_benchmark_residuals_are_small(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            metrics = case["metrics"]
            self.assertLess(metrics["fundamental_residual_l2"], 1e-8)
            self.assertLess(metrics["shg_residual_l2"], 1e-7)
            self.assertLess(metrics["max_inhomogeneous_residual_l2"], 1e-8)

    def test_multilayer_system_benchmarks_store_value_outputs(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        first = payload["cases"][0]
        self.assertEqual(len(first["outputs"]["fundamental_coefficients"]), 8)
        self.assertEqual(len(first["outputs"]["shg_coefficients"]), 8)
        self.assertEqual(len(first["outputs"]["first_layer_source_wavevectors"]), 10)
        self.assertEqual(len(first["outputs"]["first_layer_inhomogeneous_electric"]), 10)


if __name__ == "__main__":
    unittest.main()
