import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import build_maker_reference_agreement_summary

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_pg2.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_pg2.json"

ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 8
EXPECTED_POINT_GROUPS = {"2", "m", "-4", "422", "3", "32", "-6", "-6m2"}


@unittest.skipUnless(REFERENCE_PATH.exists(), "pg2 live Wolfram reference not present")
class MakerFringesPointGroupPg2ReferenceComparisonTests(unittest.TestCase):
    """POINT-GROUP end-to-end Maker SHG (pg2): the active film carries a
    SYMMETRY-CONSTRAINED d-tensor for eight further crystal classes
    (2/m/-4/422/3/32/-6/-6m2), not a generic 18/18 tensor. Extends pg1
    (3m/-42m/mm2/4mm) to broader crystallographic coverage of the full
    incidence-swept Maker pipeline vs live Wolfram f4NL -- the same physical
    axis as pg1, distinct from multilayer depth (ml1-ml8) and from the isolated
    symbolic doldExp validation."""

    def setUp(self):
        self.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

    def test_pg2_grid_matches_live_mathematica(self):
        summary = build_maker_reference_agreement_summary(self.reference, self.python, atol=ATOL, rtol=RTOL)
        self.assertEqual(summary["status"], "maker_outputs_match_all_compared_cases")
        self.assertTrue(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["nonsingular_fail_count"], 0)
        for key_result in summary["nonsingular_outputs"]:
            self.assertEqual(key_result["comparison_status"], "passed")
            self.assertLessEqual(key_result["max_abs_error"], ATOL)

    def test_cases_use_symmetry_constrained_d_and_real_point_groups(self):
        self.assertEqual(self.reference["case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(self.reference["functionDownValues"]["f4NL"], 1)
        pgs = {c["point_group"] for c in self.python["cases"]}
        self.assertEqual(pgs, EXPECTED_POINT_GROUPS)
        # every case must be genuinely symmetry-constrained (fewer than 18 nonzero d)
        for c in self.python["cases"]:
            self.assertLess(c["nonzero_d_entries"], 18)
            self.assertGreater(c["nonzero_d_entries"], 0)


if __name__ == "__main__":
    unittest.main()
