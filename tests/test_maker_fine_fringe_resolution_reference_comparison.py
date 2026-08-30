import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import (
    build_maker_reference_agreement_summary,
    compare_maker_reference_payloads,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_fringe_resolution_reference_v1.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_fine_sampling_fringe_resolution_output_v1.json"


class MakerFineFringeResolutionReferenceComparisonTests(unittest.TestCase):
    def test_dense_fringe_resolution_reference_matches_mathematica(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        results = compare_maker_reference_payloads(reference, python, atol=1e-10, rtol=1e-10)
        summary = build_maker_reference_agreement_summary(reference, python, atol=1e-10, rtol=1e-10)

        self.assertEqual(summary["status"], "maker_outputs_match_all_compared_cases")
        self.assertTrue(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_ids"], ["fine_maker_fringe_resolution"])
        self.assertEqual(len(results), 7)
        self.assertTrue(all(result.passed for result in results))
        self.assertTrue(all(result.shape[0] == 481 for result in results))
        self.assertLess(max(result.max_abs_error for result in results), 3e-14)


if __name__ == "__main__":
    unittest.main()
