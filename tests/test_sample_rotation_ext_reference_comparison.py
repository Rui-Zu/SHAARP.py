import json
import unittest
from pathlib import Path

from benchmarks.compare_sample_rotation_reference import (
    build_sample_rotation_summary,
    compare_sample_rotation_payloads,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "sample_rotation_reference_ext1.json"
PYTHON_PATH = ROOT / "benchmarks" / "sample_rotation_benchmarks_ext1.json"

ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 12


class SampleRotationExtReferenceComparisonTests(unittest.TestCase):
    """SHAARP.ml end-to-end reflected + transmitted 2omega SHG (SampleRotate)
    value-validated against live Wolfram 14.3 over an extended, all-nonsingular
    grid that spans every orientation index and many build indices."""

    def test_extended_sample_rotation_grid_matches_live_mathematica(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        results = compare_sample_rotation_payloads(reference, python, atol=ATOL, rtol=RTOL)
        summary = build_sample_rotation_summary(results, reference, atol=ATOL, rtol=RTOL)

        # Every ext case is nonsingular, so full agreement is expected.
        self.assertEqual(summary["status"], "sample_rotation_outputs_match")
        self.assertTrue(summary["full_sample_rotation_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["nonsingular_fail_count"], 0)
        self.assertEqual(summary["diagnostic_fail_count"], 0)
        self.assertEqual(len(summary["results"]), EXPECTED_CASE_COUNT)
        for result in summary["results"]:
            self.assertEqual(result["comparison_status"], "passed")
            self.assertEqual(result["shape"], [3, 5])
            self.assertLessEqual(result["max_abs_error"], ATOL)

    def test_extended_reference_is_live_wolfram_sourced(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        source = str(reference.get("source", "")).lower()
        self.assertTrue("mathematica" in source or "wolfram" in source)
        self.assertEqual(reference.get("tag"), "ext1")
        self.assertEqual(reference.get("case_count"), EXPECTED_CASE_COUNT)
        # SampleRotate/f4NL must be the actual loaded SHAARP.ml definitions.
        self.assertEqual(reference["functionDownValues"]["SampleRotate"], 1)
        self.assertEqual(reference["functionDownValues"]["f4NL"], 1)


if __name__ == "__main__":
    unittest.main()
