import json
import unittest
from pathlib import Path

from benchmarks.compare_sample_rotation_reference import build_sample_rotation_summary, compare_sample_rotation_payloads


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "sample_rotation_reference_v1.json"
PYTHON_PATH = ROOT / "benchmarks" / "sample_rotation_benchmarks_v1.json"


class SampleRotationReferenceComparisonTests(unittest.TestCase):
    def test_real_mathematica_sample_rotation_summary_passes_nonsingular_cases(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        results = compare_sample_rotation_payloads(reference, python, atol=1e-10, rtol=1e-10)
        summary = build_sample_rotation_summary(results, reference, atol=1e-10, rtol=1e-10)

        self.assertEqual(summary["status"], "sample_rotation_nonsingular_match_with_phase_matching_diagnostic")
        self.assertFalse(summary["full_sample_rotation_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], 2)
        self.assertEqual(summary["nonsingular_fail_count"], 0)
        self.assertEqual(summary["diagnostic_case_ids"], ["sample_rotation_phase_matching_like"])
        self.assertEqual(summary["diagnostic_fail_count"], 1)

        nonsingular = [result for result in summary["results"] if result["case_id"] != "sample_rotation_phase_matching_like"]
        self.assertTrue(nonsingular)
        for result in nonsingular:
            self.assertEqual(result["comparison_status"], "passed")
            self.assertLessEqual(result["max_abs_error"], 1e-10)


if __name__ == "__main__":
    unittest.main()
