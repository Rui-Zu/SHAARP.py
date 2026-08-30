import unittest

from benchmarks.generate_shaarp_si_stage_comparison_summary import build_summary


class SHAARPSIStageComparisonSummaryTests(unittest.TestCase):
    def test_summary_records_passed_and_diagnostic_stage_counts(self):
        summary = build_summary()

        self.assertEqual(summary["status"], "partial_stage_value_comparison_not_full_shaarp_si_agreement")
        self.assertEqual(summary["passed_value_comparison_count"], 1116)
        self.assertEqual(summary["diagnostic_only_count"], 144)
        self.assertFalse(summary["full_shaarp_si_agreement_claimed"])

        stages = {stage["stage_id"]: stage for stage in summary["stages"]}
        self.assertEqual(stages["pnl"]["comparison_status"], "passed")
        self.assertEqual(stages["inhomogeneous_croots"]["comparison_status"], "passed")
        self.assertEqual(stages["omega_fresnel_boundary"]["comparison_status"], "passed")
        self.assertEqual(stages["two_omega_boundary"]["comparison_status"], "passed")
        self.assertEqual(stages["signal_intensity"]["comparison_status"], "passed")
        self.assertEqual(stages["angle_root_equations"]["comparison_status"], "passed")
        self.assertEqual(stages["angle_root_labels"]["comparison_status"], "diagnostic_only")


if __name__ == "__main__":
    unittest.main()
