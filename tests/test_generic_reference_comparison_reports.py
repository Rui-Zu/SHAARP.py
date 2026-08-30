import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.compare_mathematica_reference import compare_reference_files, compare_payloads
from benchmarks.discrepancy_report import validate_discrepancy_report, write_draft_reports_from_comparison_results


class GenericReferenceComparisonReportTests(unittest.TestCase):
    def test_generic_comparator_file_results_feed_discrepancy_report_writer(self):
        python_payload = _python_payload()
        reference = _reference_payload(value=1.25)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            reference_path = tmp / "reference.json"
            python_path = tmp / "python.json"
            reports_dir = tmp / "reports"
            reference_path.write_text(json.dumps(reference), encoding="utf-8")
            python_path.write_text(json.dumps(python_payload), encoding="utf-8")

            results = compare_reference_files(
                reference_path,
                python_path,
                keys=["single_interface_intensity"],
                atol=1e-12,
                rtol=1e-12,
            )
            paths = write_draft_reports_from_comparison_results(
                results,
                reports_dir,
                source_notebook_or_package="Github/SHAARP/SHAARP_V1.03.nb",
                mathematica_version_or_runtime="Wolfram 14.3",
                shaarp_file_path_version_or_date="Github/SHAARP/SHAARP_V1.03.nb",
                python_git_or_worktree_state="not a git repository",
                first_failing_stage="final intensity",
                tolerance={"atol": 1e-12, "rtol": 1e-12},
            )
            report = json.loads(paths[0].read_text(encoding="utf-8"))

        self.assertEqual(len(paths), 1)
        self.assertEqual(report["case_id"], "case_001")
        self.assertEqual(report["classification"], "unresolved")
        self.assertEqual(report["intermediate_values_needed_to_reproduce"]["comparison_key"], "single_interface_intensity")
        validate_discrepancy_report(report)

    def test_generic_comparator_strict_mode_rejects_python_regression_status(self):
        python_payload = _python_payload()
        reference = _reference_payload(value=1.0)
        reference["status"] = "python_regression"

        with self.assertRaisesRegex(ValueError, "python_regression"):
            compare_payloads(
                reference,
                python_payload,
                keys=["single_interface_intensity"],
                atol=1e-12,
                rtol=1e-12,
                strict_reference=True,
            )


def _python_payload():
    return {
        "cases": [
            {
                "id": "case_001",
                "outputs": {
                    "single_interface_intensity": [1.0, 2.0],
                },
            }
        ]
    }


def _reference_payload(*, value: float):
    return {
        "source": "Mathematica SHAARP synthetic comparator fixture",
        "status": "mathematica_reference_exported",
        "cases": [
            {
                "id": "case_001",
                "outputs": {
                    "single_interface_intensity": [value, 2.0],
                },
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
