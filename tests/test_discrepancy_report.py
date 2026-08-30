import copy
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.discrepancy_report import (
    ALLOWED_CLASSIFICATIONS,
    blank_discrepancy_report,
    draft_report_from_comparison_result,
    validate_discrepancy_report,
    write_draft_reports_from_comparison_results,
)
from benchmarks.compare_mathematica_reference import ComparisonResult


class DiscrepancyReportTests(unittest.TestCase):
    def test_blank_report_contains_all_allowed_classification_context(self):
        report = blank_discrepancy_report()
        self.assertEqual(report["classification"], "unresolved")
        self.assertIn("case_id", report)
        self.assertIn("first_failing_stage", report)
        self.assertIn("expected_mathematica_value", report)
        self.assertIn("python_value", report)
        self.assertIn("next_action", report)
        self.assertIn("python_bug", ALLOWED_CLASSIFICATIONS)
        self.assertIn("mathematica_shaarp_bug_or_ambiguity", ALLOWED_CLASSIFICATIONS)

    def test_valid_python_bug_report_passes(self):
        report = _filled_report(classification="python_bug")

        validate_discrepancy_report(report)

    def test_missing_first_failing_stage_is_rejected(self):
        report = _filled_report()
        report["first_failing_stage"] = None

        with self.assertRaisesRegex(ValueError, "first_failing_stage"):
            validate_discrepancy_report(report)

    def test_unknown_classification_is_rejected(self):
        report = _filled_report()
        report["classification"] = "looks_wrong"

        with self.assertRaisesRegex(ValueError, "looks_wrong"):
            validate_discrepancy_report(report)

    def test_shaarp_bug_report_requires_extra_evidence(self):
        report = _filled_report(classification="mathematica_shaarp_bug_or_ambiguity")
        report["minimal_reproducible_case"] = None

        with self.assertRaisesRegex(ValueError, "minimal_reproducible_case"):
            validate_discrepancy_report(report)

    def test_shaarp_bug_report_with_extra_evidence_passes(self):
        report = _filled_report(classification="mathematica_shaarp_bug_or_ambiguity")

        validate_discrepancy_report(report)

    def test_tolerance_requires_atol_and_rtol(self):
        report = _filled_report()
        report["tolerance"] = {"atol": 1e-10}

        with self.assertRaisesRegex(ValueError, "rtol"):
            validate_discrepancy_report(report)

    def test_report_template_round_trips_as_json(self):
        report = blank_discrepancy_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, report)

    def test_draft_report_from_failed_comparison_result_is_valid_and_unresolved(self):
        result = ComparisonResult(
            case_id="boundary_system_01",
            key="solveFresnelN_matrix",
            passed=False,
            max_abs_error=1.2e-4,
            max_rel_error=2.5e-5,
            shape=(8, 8),
        )

        report = draft_report_from_comparison_result(
            result,
            source_notebook_or_package="Github/SHAARP.ml/setup.nb",
            mathematica_version_or_runtime="Wolfram 14.3",
            shaarp_file_path_version_or_date="Github/SHAARP.ml/setup.nb",
            python_git_or_worktree_state="not a git repository",
            first_failing_stage="2omega boundary matrix/RHS",
            tolerance={"atol": 1e-10, "rtol": 1e-10},
        )

        self.assertEqual(report["case_id"], "boundary_system_01")
        self.assertEqual(report["classification"], "unresolved")
        self.assertEqual(report["absolute_error"], 1.2e-4)
        self.assertEqual(report["intermediate_values_needed_to_reproduce"]["comparison_key"], "solveFresnelN_matrix")
        self.assertEqual(report["intermediate_values_needed_to_reproduce"]["comparison_shape"], [8, 8])
        validate_discrepancy_report(report)

    def test_draft_report_does_not_allow_shaarp_bug_without_extra_evidence(self):
        result = ComparisonResult(
            case_id="case_001",
            key="final_intensity",
            passed=False,
            max_abs_error=0.2,
            max_rel_error=0.1,
            shape=(5,),
        )

        report = draft_report_from_comparison_result(
            result,
            source_notebook_or_package="Github/SHAARP/SHAARP_V1.03.nb",
            mathematica_version_or_runtime="Wolfram 14.3",
            shaarp_file_path_version_or_date="Github/SHAARP/SHAARP_V1.03.nb",
            python_git_or_worktree_state="not a git repository",
            first_failing_stage="final intensity",
            tolerance={"atol": 1e-7, "rtol": 1e-6},
            classification="mathematica_shaarp_bug_or_ambiguity",
        )

        with self.assertRaisesRegex(ValueError, "minimal_reproducible_case"):
            validate_discrepancy_report(report)

    def test_write_draft_reports_from_failed_comparison_results(self):
        results = [
            ComparisonResult("case/001", "matrix[0,0]", False, 1e-4, 2e-5, (8, 8)),
            ComparisonResult("case_002", "rhs", True, 0.0, 0.0, (8,)),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_draft_reports_from_comparison_results(
                results,
                Path(tmpdir),
                source_notebook_or_package="Github/SHAARP.ml/setup.nb",
                mathematica_version_or_runtime="Wolfram 14.3",
                shaarp_file_path_version_or_date="Github/SHAARP.ml/setup.nb",
                python_git_or_worktree_state="not a git repository",
                first_failing_stage="2omega boundary matrix/RHS",
                tolerance={"atol": 1e-10, "rtol": 1e-10},
            )
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].name.startswith("001_case_001_matrix_0_0"))
            report = json.loads(paths[0].read_text(encoding="utf-8"))

        self.assertEqual(report["classification"], "unresolved")
        self.assertEqual(report["case_id"], "case/001")
        self.assertEqual(report["intermediate_values_needed_to_reproduce"]["comparison_key"], "matrix[0,0]")
        validate_discrepancy_report(report)

    def test_committed_mathematica_reference_discrepancy_reports_are_valid(self):
        report_dir = Path(__file__).resolve().parents[1] / "benchmarks" / "mathematica_reference" / "discrepancy_reports"
        reports = sorted(report_dir.glob("*.json"))

        self.assertTrue(reports)
        for path in reports:
            with self.subTest(path=path.name):
                validate_discrepancy_report(json.loads(path.read_text(encoding="utf-8")))


def _filled_report(*, classification: str = "unresolved"):
    report = copy.deepcopy(blank_discrepancy_report())
    report.update(
        {
            "status": "triaged",
            "case_id": "case_001",
            "source_notebook_or_package": "Github/SHAARP.ml/setup.nb",
            "mathematica_version_or_runtime": "Wolfram 14.3",
            "shaarp_file_path_version_or_date": "Github/SHAARP.ml/setup.nb",
            "python_git_or_worktree_state": "not a git repository",
            "first_failing_stage": "omega Snell roots and branch identity",
            "expected_mathematica_value": {"real": 1.0, "imag": 0.0},
            "python_value": {"real": 1.1, "imag": 0.0},
            "absolute_error": 0.1,
            "relative_error": 0.1,
            "tolerance": {"atol": 1e-10, "rtol": 1e-10},
            "intermediate_values_needed_to_reproduce": {
                "theta_deg": 33.0,
                "epsilon_omega": "stored in reference payload",
            },
            "classification": classification,
            "next_action": "Inspect exact Mathematica solveSnell branch export.",
            "minimal_reproducible_case": "case_001 with one layer and theta=33 deg",
            "equation_or_residual_check": "Python eigen residual is below 1e-12.",
            "why_python_result_follows_documented_equations": "Python root satisfies exported Snell/eigen equations.",
        }
    )
    return report


if __name__ == "__main__":
    unittest.main()
