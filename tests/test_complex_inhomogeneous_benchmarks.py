import json
import unittest
from pathlib import Path

from benchmarks.generate_complex_inhomogeneous_benchmarks import build_cases, fingerprint_record, run_case


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "complex_inhomogeneous_benchmarks_v1.json"
)


class ComplexInhomogeneousBenchmarkTests(unittest.TestCase):
    def test_complex_inhomogeneous_benchmark_file_covers_angles_and_orientations(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["case_count"], 24)
        self.assertEqual(len(payload["cases"]), 24)
        self.assertGreaterEqual(len({case["theta_deg"] for case in payload["cases"]}), 6)
        self.assertGreaterEqual(len({case["orientation_index"] for case in payload["cases"]}), 4)
        for case in payload["cases"]:
            self.assertGreater(case["metrics"]["max_abs_epsilon_imag"], 0.0)
            self.assertGreater(case["metrics"]["max_abs_epsilon_offdiag"], 0.0)
            self.assertGreater(case["metrics"]["max_abs_source_imag"], 0.0)

    def test_complex_inhomogeneous_benchmark_residuals_are_small(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            self.assertLess(case["metrics"]["residual_l2"], 1e-10)
            self.assertGreater(case["metrics"]["electric_l2"], 0.0)

    def test_complex_inhomogeneous_benchmark_fingerprints_match_current_python(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        expected = {case["id"]: case["fingerprint"] for case in payload["cases"]}
        for case in build_cases():
            record = run_case(case)
            self.assertEqual(fingerprint_record(record), expected[record["id"]])


if __name__ == "__main__":
    unittest.main()
