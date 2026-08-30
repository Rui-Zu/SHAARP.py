import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.compare_maker_stage_diagnostics import build_stage_agreement_summary, write_stage_agreement_summary


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_stage_diagnostics_v1.json"


class MakerStageDiagnosticsComparisonTests(unittest.TestCase):
    def test_summary_records_value_matches_and_expected_caveats(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        summary = build_stage_agreement_summary(payload, atol=1e-10)

        self.assertEqual(
            summary["status"],
            "partial_stage_agreement_with_selected_wave_and_phase_matching_caveats",
        )
        self.assertFalse(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], 2)
        self.assertEqual(summary["singular_case_ids"], ["maker_fringes_phase_matching_like"])

        stages = {stage["stage_id"]: stage for stage in summary["stages"]}
        for stage_id in (
            "linear_reflected_best_match",
            "linear_transmitted_selected_wave",
            "inhomogeneous_forward_sources",
            "nonlinear_transmitted_selected_wave",
            "homogeneous_layer_2omega_boundary",
        ):
            self.assertEqual(stages[stage_id]["comparison_status"], "passed")
            self.assertLessEqual(stages[stage_id]["max_abs_error"], 1e-10)

        self.assertEqual(
            stages["transmitted_absent_slots"]["comparison_status"],
            "documented_caveat",
        )
        self.assertEqual(
            stages["phase_matching_case"]["comparison_status"],
            "diagnostic_only",
        )
        self.assertEqual(
            stages["homogeneous_layer_2omega_boundary"]["notes"],
            "Forward/backward homogeneous 2omega layer waves match Mathematica after comparing the nonlinear boundary records.",
        )

    def test_summary_writer_creates_machine_readable_artifact(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "summary.json"
            summary = write_stage_agreement_summary(output_path, payload, atol=1e-10)

            self.assertTrue(output_path.exists())
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved, summary)
            self.assertEqual(saved["selected_transmitted_wave_policy"], "shaarp_ml_selected")


if __name__ == "__main__":
    unittest.main()
