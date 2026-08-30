import json
import unittest
from pathlib import Path

from benchmarks.compare_shaarp_si_end_to_end_signal_diagnostic import build_diagnostic_summary


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "benchmarks" / "mathematica_reference" / "shaarp_si_end_to_end_signal_diagnostic_v1.json"


class SHAARPSIEndToEndSignalDiagnosticTests(unittest.TestCase):
    def test_current_python_si_solver_does_not_claim_end_to_end_signal_agreement(self):
        summary = build_diagnostic_summary(atol=1e-10, rtol=1e-10)

        self.assertEqual(summary["status"], "diagnostic_current_python_si_solver_not_full_shaarp_si_agreement")
        self.assertFalse(summary["full_shaarp_si_agreement_claimed"])
        self.assertEqual(summary["case_count"], 36)
        self.assertEqual(summary["passing_er2w_case_count"], 5)
        self.assertEqual(summary["failing_er2w_case_count"], 31)
        self.assertGreater(summary["max_er2w_abs_error"], 1.0)
        self.assertIn("ordinary/extraordinary", summary["likely_first_gap"])

    def test_saved_diagnostic_matches_generator(self):
        saved = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

        self.assertEqual(saved, build_diagnostic_summary(atol=1e-10, rtol=1e-10))


if __name__ == "__main__":
    unittest.main()
