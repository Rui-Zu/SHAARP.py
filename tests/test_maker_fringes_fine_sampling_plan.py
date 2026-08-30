import json
import unittest
from pathlib import Path
from dataclasses import replace

from benchmarks.generate_maker_fringes_fine_sampling_plan import build_plan
from benchmarks.generate_maker_fringes_benchmarks import build_cases
from shaarp.multilayer_shg_boundary import solve_multilayer_shg_from_system_polarimetry


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "benchmarks" / "maker_fringes_fine_sampling_plan_v1.json"


class MakerFringesFineSamplingPlanTests(unittest.TestCase):
    def test_saved_plan_matches_generator(self):
        saved = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_plan())

    def test_plan_is_marked_as_slow_not_routine_validation(self):
        plan = build_plan()
        self.assertEqual(plan["status"], "slow_path_fine_sampling_plan_not_routine_test_not_mathematica_validation")
        self.assertTrue(plan["routine_suite_should_not_run_fine_grid"])
        self.assertTrue(plan["mathematica_reference_required_for_agreement_claim"])
        self.assertIn("--confirm-slow", plan["slow_runner"])

    def test_plan_has_dense_angle_grids_and_extremes(self):
        plan = build_plan()
        self.assertGreaterEqual(plan["total_planned_angle_count"], 800)
        all_angles = [theta for case in plan["cases"] for theta in case["theta_deg"]]
        self.assertIn(0.0, all_angles)
        self.assertTrue(any(theta < 0.0 for theta in all_angles))
        self.assertGreaterEqual(max(all_angles), 80.0)
        self.assertTrue(all(case["theta_step_deg"] <= 0.5 for case in plan["cases"]))

    def test_plan_keeps_phase_matching_as_diagnostic_case(self):
        plan = build_plan()
        diagnostic = [
            case for case in plan["cases"] if case["expected_residual_behavior"] == "diagnostic_extreme_not_small_residual_claim"
        ]
        self.assertEqual(len(diagnostic), 1)
        self.assertEqual(diagnostic[0]["source_case_id"], "maker_fringes_phase_matching_like")

    def test_negative_normal_positive_fine_grid_includes_regressed_complex_snell_angle(self):
        case = next(case for case in build_cases() if case["id"] == "maker_fringes_negative_normal_positive")
        system = replace(case["system"], polarimetry=replace(case["system"].polarimetry, theta_deg=-7.5))

        result = solve_multilayer_shg_from_system_polarimetry(
            system,
            mu=1.0,
            eps0=1.0,
            inhomogeneous_source_policy="forward_only",
            inhomogeneous_solution_policy="solve",
        )

        self.assertLess(max(abs(value) for value in result.shg.residual), 1e-8)


if __name__ == "__main__":
    unittest.main()
