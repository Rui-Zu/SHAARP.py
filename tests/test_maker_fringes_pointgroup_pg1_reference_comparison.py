import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import build_maker_reference_agreement_summary

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_pg1.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_pg1.json"

ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 4


@unittest.skipUnless(REFERENCE_PATH.exists(), "pg1 live Wolfram reference not present")
class MakerFringesPointGroupPg1ReferenceComparisonTests(unittest.TestCase):
    """POINT-GROUP end-to-end Maker SHG: the active film carries a SYMMETRY-
    CONSTRAINED d-tensor (3m/-42m/mm2/4mm), not a generic 18/18 tensor. Validates
    the full incidence-swept Maker pipeline for real crystal classes vs live
    Wolfram f4NL -- a NEW physical axis distinct from multilayer depth (ml1-ml8)
    and from the isolated symbolic doldExp validation."""

    def setUp(self):
        self.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

    def test_pg1_grid_matches_live_mathematica(self):
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
        self.assertEqual(pgs, {"3m", "-42m", "mm2", "4mm"})
        # every case must be genuinely symmetry-constrained (fewer than 18 nonzero d)
        for c in self.python["cases"]:
            self.assertLess(c["nonzero_d_entries"], 18)
            self.assertGreater(c["nonzero_d_entries"], 0)


if __name__ == "__main__":
    unittest.main()
