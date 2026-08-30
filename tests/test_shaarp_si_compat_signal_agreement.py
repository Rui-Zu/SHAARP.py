import json
import unittest
from pathlib import Path

from benchmarks.compare_shaarp_si_compat_signal_agreement import build_compat_signal_agreement


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "benchmarks" / "mathematica_reference" / "shaarp_si_compat_signal_agreement_v1.json"


class SHAARPSICompatSignalAgreementTests(unittest.TestCase):
    def test_si_compat_solver_matches_all_exported_signal_stage_er2w_values(self):
        summary = build_compat_signal_agreement(atol=2e-6, rtol=0.0)

        self.assertEqual(summary["status"], "si_compat_reflected_signal_matches_exported_shaarp_si_stage_reference")
        self.assertFalse(summary["full_notebook_end_to_end_agreement_claimed"])
        self.assertEqual(summary["case_count"], 36)
        self.assertEqual(summary["passing_er2w_case_count"], 36)
        self.assertEqual(summary["failing_er2w_case_count"], 0)
        self.assertLessEqual(summary["max_er2w_abs_error"], 2e-6)
        self.assertEqual(summary["normal_incidence_2omega_branch_policy"], "shaarp_reference_like")

    def test_saved_compat_signal_agreement_matches_generator(self):
        saved = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

        self.assertEqual(saved, build_compat_signal_agreement(atol=2e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
