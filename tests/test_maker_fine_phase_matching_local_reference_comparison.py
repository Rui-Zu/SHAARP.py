import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import build_maker_reference_agreement_summary


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_phase_matching_local_reference_v1.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_fine_sampling_phase_matching_local_output_v1.json"


class MakerFinePhaseMatchingLocalReferenceComparisonTests(unittest.TestCase):
    def test_phase_matching_like_local_grid_is_diagnostic_not_pass_claim(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        summary = build_maker_reference_agreement_summary(reference, python, atol=1e-10, rtol=1e-10)

        self.assertEqual(summary["status"], "nonsingular_maker_outputs_match_with_phase_matching_diagnostic")
        self.assertFalse(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(summary["diagnostic_case_ids"], ["fine_maker_phase_matching_like_local"])
        self.assertEqual(summary["diagnostic_fail_count"], 7)
        self.assertEqual(summary["nonsingular_case_count"], 0)
        self.assertTrue(all(item["shape"][0] == 101 for item in summary["diagnostic_outputs"]))
        self.assertGreater(max(item["max_abs_error"] for item in summary["diagnostic_outputs"]), 1.0)


if __name__ == "__main__":
    unittest.main()
