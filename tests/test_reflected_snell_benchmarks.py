import json
import unittest
from pathlib import Path

from benchmarks.generate_reflected_snell_benchmarks import build_cases, fingerprint_record, run_case


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "reflected_snell_benchmarks_v1.json"
)


class ReflectedSnellBenchmarkTests(unittest.TestCase):
    def test_reflected_snell_benchmark_file_covers_angles_and_orientations(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["case_count"], 20)
        self.assertEqual(len(payload["cases"]), 20)
        self.assertGreaterEqual(len({case["theta_deg"] for case in payload["cases"]}), 5)
        self.assertGreaterEqual(len({case["orientation_index"] for case in payload["cases"]}), 4)
        for case in payload["cases"]:
            self.assertGreater(case["metrics"]["max_abs_epsilon_imag"], 0.0)
            self.assertGreater(case["metrics"]["max_abs_epsilon_offdiag"], 0.0)

    def test_reflected_snell_benchmark_residuals_are_small_and_backward(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            self.assertLess(case["metrics"]["max_residual"], 1e-8)
            self.assertLess(case["metrics"]["max_real_direction_z"], 1e-10)

    def test_reflected_snell_benchmark_fingerprints_match_current_python(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        expected = {case["id"]: case["fingerprint"] for case in payload["cases"]}
        for case in build_cases():
            record = run_case(case)
            self.assertEqual(fingerprint_record(record), expected[record["id"]])


if __name__ == "__main__":
    unittest.main()
