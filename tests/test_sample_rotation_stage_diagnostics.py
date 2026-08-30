import json
import unittest
from pathlib import Path

from benchmarks.compare_sample_rotation_stage_diagnostics import (
    SAMPLE_ROTATION_STAGE_PATH,
    build_sample_rotation_stage_summary,
)


class SampleRotationStageDiagnosticsTests(unittest.TestCase):
    def test_stage_diagnostics_are_mathematica_backed_and_machine_readable(self):
        payload = json.loads(SAMPLE_ROTATION_STAGE_PATH.read_text(encoding="utf-8"))

        self.assertIn("Mathematica", payload["source"])
        self.assertEqual(payload["status"], "mathematica_sample_rotation_stage_diagnostics_exported")
        self.assertEqual(payload["case_count"], 3)
        case = payload["cases"][0]
        point = case["points"][0]
        self.assertIn("nonlinear_reflected", point)
        self.assertIn("nonlinear_transmitted", point)
        self.assertIn("sample_rotate_amplitudes", point)

    def test_stage_summary_keeps_nonsingular_mismatch_visible(self):
        payload = json.loads(SAMPLE_ROTATION_STAGE_PATH.read_text(encoding="utf-8"))
        summary = build_sample_rotation_stage_summary(payload, atol=1e-10)

        self.assertEqual(summary["status"], "sample_rotation_stage_nonsingular_match_with_phase_matching_diagnostic")
        self.assertFalse(summary["full_sample_rotation_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], 2)
        failed = [stage for stage in summary["stages"] if stage["comparison_status"] == "failed"]
        self.assertEqual(failed, [])
        diagnostic = [stage for stage in summary["stages"] if stage["comparison_status"] == "diagnostic_only"]
        self.assertEqual([stage["stage_id"] for stage in diagnostic], ["phase_matching_case"])


if __name__ == "__main__":
    unittest.main()
