import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import (
    build_maker_reference_agreement_summary,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_ext1.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_ext1.json"

ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 12


class MakerFringesExtReferenceComparisonTests(unittest.TestCase):
    """SHAARP.ml transmitted 2omega Maker-fringe SHG swept over incidence angle,
    value-validated against live Wolfram 14.3 over an extended, all-nonsingular
    grid spanning every orientation index and many build indices. This covers
    the theta-dependence that the SampleRotate ext grid (fixed theta) does not."""

    def test_extended_maker_grid_matches_live_mathematica(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        summary = build_maker_reference_agreement_summary(
            reference, python, atol=ATOL, rtol=RTOL
        )

        # Every ext case is nonsingular, so full agreement is expected.
        self.assertEqual(summary["status"], "maker_outputs_match_all_compared_cases")
        self.assertTrue(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["nonsingular_fail_count"], 0)
        self.assertEqual(summary["diagnostic_case_ids"], [])
        self.assertEqual(summary["diagnostic_fail_count"], 0)
        # Each compared output key must pass with a roundoff-level abs error.
        self.assertTrue(summary["nonsingular_outputs"])
        for key_result in summary["nonsingular_outputs"]:
            self.assertEqual(key_result["comparison_status"], "passed")
            self.assertEqual(key_result["case_count"], EXPECTED_CASE_COUNT)
            self.assertLessEqual(key_result["max_abs_error"], ATOL)

    def test_extended_reference_is_live_wolfram_sourced(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        source = str(reference.get("source", "")).lower()
        self.assertTrue("mathematica" in source or "wolfram" in source)
        self.assertEqual(reference.get("case_count"), EXPECTED_CASE_COUNT)
        # f4NL must be the actual loaded SHAARP.ml orchestrator.
        self.assertEqual(reference["functionDownValues"]["f4NL"], 1)
        # No case may have failed to resolve in the exporter.
        items = reference["suites"][0]["items"]
        self.assertFalse([item for item in items if "error" in item])


if __name__ == "__main__":
    unittest.main()
