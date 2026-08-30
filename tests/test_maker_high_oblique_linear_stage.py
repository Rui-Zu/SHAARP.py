import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_high_oblique_linear_stage import (
    LINEAR_STAGE_PATH,
    build_linear_stage_summary,
)


class MakerHighObliqueLinearStageTests(unittest.TestCase):
    def test_mathematica_linear_stage_artifact_targets_failing_endpoint(self):
        payload = json.loads(LINEAR_STAGE_PATH.read_text(encoding="utf-8"))

        self.assertIn("Mathematica", payload["source"])
        self.assertEqual(payload["status"], "mathematica_maker_high_oblique_linear_stage_exported")
        self.assertEqual(payload["case_id"], "fine_maker_high_oblique")
        self.assertEqual(payload["theta_deg"], 80.0)
        self.assertIn("linear_reflected", payload)
        self.assertIn("linear_transmitted", payload)
        self.assertIn("forward_layer_waves", payload)
        self.assertIn("backward_layer_waves", payload)

    def test_direct_python_linear_stage_mismatch_is_visible(self):
        payload = json.loads(LINEAR_STAGE_PATH.read_text(encoding="utf-8"))
        summary = build_linear_stage_summary(payload, atol=1e-10)

        self.assertEqual(summary["status"], "python_direct_linear_stage_matches_mathematica")
        self.assertTrue(summary["agreement_claimed"])
        self.assertLess(summary["max_abs_error"], 1e-10)


if __name__ == "__main__":
    unittest.main()
