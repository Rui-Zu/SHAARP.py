import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks" / "multilayer_polarimetry_benchmarks_v1.json"


class MultilayerPolarimetryBenchmarkTests(unittest.TestCase):
    def test_multilayer_polarimetry_benchmarks_cover_angles_orientations_and_angles(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "python_staged_multilayer_polarimetry_regression_not_mathematica_validation")
        self.assertEqual(payload["case_count"], 20)
        cases = payload["cases"]
        self.assertEqual({case["theta_deg"] for case in cases}, {0.0, 8.0, 18.0, 33.0, 52.0})
        self.assertEqual({case["orientation_index"] for case in cases}, {0, 1, 2, 3})
        self.assertEqual({case["phi_deg"] for case in cases}, {0.0, 30.0, 67.5, 90.0})
        self.assertEqual({case["psi_deg"] for case in cases}, {0.0, 22.5, 45.0, 90.0})
        self.assertEqual({case["ellipticity_deg"] for case in cases}, {-35.0, 0.0, 15.0})
        self.assertEqual({case["sample_azimuth_deg"] for case in cases}, {0.0, 15.0, 37.5, 72.0})

    def test_multilayer_polarimetry_benchmarks_have_complex_nondiagonal_tensors_and_ten_sources(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            metrics = case["metrics"]
            self.assertGreater(metrics["max_abs_epsilon_omega_imag"], 0.0)
            self.assertGreater(metrics["max_abs_epsilon_2omega_imag"], 0.0)
            self.assertGreater(metrics["max_abs_epsilon_omega_offdiag"], 0.0)
            self.assertGreater(metrics["max_abs_d_imag"], 0.0)
            self.assertEqual(metrics["source_count"], 10)

    def test_multilayer_polarimetry_benchmark_residuals_are_small(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            metrics = case["metrics"]
            self.assertLess(metrics["fundamental_residual_l2"], 1e-8)
            self.assertLess(metrics["shg_residual_l2"], 1e-7)
            self.assertLess(metrics["max_inhomogeneous_residual_l2"], 1e-8)

    def test_multilayer_polarimetry_benchmarks_store_value_outputs(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        first = payload["cases"][0]
        self.assertEqual(len(first["inputs"]["incident_jones_sp"]), 2)
        self.assertEqual(len(first["inputs"]["analyzer_jones_sp"]), 2)
        self.assertEqual(len(first["outputs"]["fundamental_coefficients"]), 8)
        self.assertEqual(len(first["outputs"]["shg_coefficients"]), 8)
        self.assertEqual(len(first["outputs"]["reflected_2omega_jones_sp"]), 2)
        self.assertIn("analyzer_amplitude", first["outputs"])
        self.assertIn("analyzer_intensity", first["outputs"])
        self.assertEqual(len(first["outputs"]["first_layer_source_wavevectors"]), 10)
        self.assertEqual(len(first["outputs"]["first_layer_source_polarizations"]), 10)
        self.assertEqual(len(first["outputs"]["first_layer_inhomogeneous_electric"]), 10)

    def test_multilayer_polarimetry_benchmarks_include_true_mixed_complex_incidence(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        mixed = [case for case in payload["cases"] if case["ellipticity_deg"] != 0.0]
        self.assertGreater(len(mixed), 0)
        for case in mixed:
            s_value = case["inputs"]["incident_jones_sp"][0]
            p_value = case["inputs"]["incident_jones_sp"][1]
            self.assertNotEqual(s_value["imag"], 0.0)
            self.assertNotEqual(abs(s_value["real"]) + abs(s_value["imag"]), 0.0)
            self.assertNotEqual(abs(p_value["real"]) + abs(p_value["imag"]), 0.0)


if __name__ == "__main__":
    unittest.main()
