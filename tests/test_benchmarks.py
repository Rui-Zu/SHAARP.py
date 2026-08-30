import json
import unittest
from pathlib import Path

from benchmarks.generate_numeric_benchmarks import build_cases, fingerprint_record, run_case


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "numeric_benchmarks_v1.json"
)


class NumericBenchmarkTests(unittest.TestCase):
    def test_numeric_benchmark_file_has_twenty_cases(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["case_count"], 20)
        self.assertEqual(len(payload["cases"]), 20)
        for case in payload["cases"]:
            self.assertGreater(case["material_summary"]["max_abs_offdiag_epsilon_omega"], 0.0)
            self.assertGreater(case["material_summary"]["max_abs_offdiag_epsilon_2omega"], 0.0)
            self.assertGreater(case["material_summary"]["max_abs_d_voigt_imag"], 0.0)

    def test_numeric_benchmark_fingerprints_match_current_python(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        expected = {case["id"]: case["fingerprint"] for case in payload["cases"]}
        for case in build_cases():
            record = run_case(case)
            self.assertEqual(fingerprint_record(record), expected[record["id"]])


if __name__ == "__main__":
    unittest.main()
