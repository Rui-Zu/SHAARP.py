import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import build_maker_reference_agreement_summary

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_pg3.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_pg3.json"
ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 2


@unittest.skipUnless(REFERENCE_PATH.exists(), "pg3 live Wolfram reference not present")
class MakerFringesPointGroupPg3ReferenceComparisonTests(unittest.TestCase):
    """Closes the last distinct buildable SHG d-pattern end-to-end: the '4'/'6'
    class (differs from '-4' at one Voigt position). Validates the full Maker
    pipeline for tetragonal '4' and hexagonal '6' vs live Wolfram f4NL."""

    def setUp(self):
        self.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

    def test_pg3_grid_matches_live_mathematica(self):
        summary = build_maker_reference_agreement_summary(self.reference, self.python, atol=ATOL, rtol=RTOL)
        self.assertEqual(summary["status"], "maker_outputs_match_all_compared_cases")
        self.assertTrue(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["nonsingular_fail_count"], 0)
        for kr in summary["nonsingular_outputs"]:
            self.assertEqual(kr["comparison_status"], "passed")
            self.assertLessEqual(kr["max_abs_error"], ATOL)

    def test_cases_are_4_or_6_and_constrained(self):
        self.assertEqual(self.reference["case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(self.reference["functionDownValues"]["f4NL"], 1)
        pgs = {c["point_group"] for c in self.python["cases"]}
        self.assertEqual(pgs, {"4", "6"})
        for c in self.python["cases"]:
            self.assertEqual(c["nonzero_d_entries"], 7)
