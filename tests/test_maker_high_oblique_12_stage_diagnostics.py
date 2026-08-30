import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_high_oblique_12_stage_diagnostics import (
    STAGE_PATH,
    build_summary,
)


class MakerHighOblique12StageDiagnosticsTests(unittest.TestCase):
    def test_stage_artifact_targets_residual_angle(self):
        payload = json.loads(STAGE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_high_oblique_12_stage_diagnostics_exported")
        self.assertEqual(payload["case_count"], 1)
        self.assertEqual(payload["cases"][0]["id"], "maker_fringes_moderate_high_oblique")
        self.assertEqual(payload["cases"][0]["points"][0]["theta_deg"], 12.0)

    def test_summary_passes_after_shaarp_standard_eigenbasis_fix(self):
        payload = json.loads(STAGE_PATH.read_text(encoding="utf-8"))
        summary = build_summary(payload, atol=1e-10)

        self.assertEqual(summary["status"], "high_oblique_12_stage_passes")
        self.assertTrue(summary["agreement_claimed"])
        self.assertIsNone(summary["first_understood_residual_stage"])
        self.assertIsNone(summary["first_understood_residual_max_abs"])
        self.assertLess(summary["max_understood_residual_abs"], 2e-13)


if __name__ == "__main__":
    unittest.main()
