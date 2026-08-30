import json
import unittest
from pathlib import Path

from benchmarks.compare_shaarp_si_compat_signal_agreement_ext import build_agreement


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "benchmarks" / "mathematica_reference" / "shaarp_si_compat_signal_agreement_ext1.json"


class SHAARPSICompatSignalAgreementExtTests(unittest.TestCase):
    def test_si_compat_solver_matches_all_exported_ext1_signal_stage_er2w_values(self):
        summary = build_agreement(tag="ext1", atol=2e-6, rtol=0.0)

        self.assertEqual(
            summary["status"],
            "si_compat_reflected_signal_matches_extension_shaarp_si_stage_reference",
        )
        self.assertEqual(summary["case_count"], 84)
        self.assertEqual(summary["passing_er2w_case_count"], 84)
        self.assertEqual(summary["failing_er2w_case_count"], 0)
        self.assertLessEqual(summary["max_er2w_abs_error"], 2e-6)
        self.assertEqual(summary["normal_incidence_2omega_branch_policy"], "shaarp_reference_like")

    def test_saved_compat_signal_agreement_ext1_matches_generator(self):
        saved = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

        self.assertEqual(saved, build_agreement(tag="ext1", atol=2e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
