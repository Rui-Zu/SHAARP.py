import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import build_maker_reference_agreement_summary

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_pgml.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_pgml.json"

ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 4
POINT_GROUPS = {"3m", "-42m", "32", "-6m2"}


@unittest.skipUnless(REFERENCE_PATH.exists(), "pgml live Wolfram reference not present")
class MakerFringesPointGroupAtDepthPgmlReferenceComparisonTests(unittest.TestCase):
    """POINT-GROUP-AT-DEPTH end-to-end Maker SHG: the active film carries a
    SYMMETRY-CONSTRAINED d-tensor (3m/-42m/32/-6m2) AND sits inside a 4-layer
    stack (air / point-group-film / isotropic-interlayer / substrate). Validates
    the full incidence-swept Maker pipeline when BOTH the crystallographic
    point-group axis (pg1/pg2) and the multilayer-depth axis (ml1-ml8) are present
    simultaneously vs live Wolfram f4NL -- a combination never tested before."""

    def setUp(self):
        self.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

    def test_pgml_grid_matches_live_mathematica(self):
        summary = build_maker_reference_agreement_summary(self.reference, self.python, atol=ATOL, rtol=RTOL)
        self.assertEqual(summary["status"], "maker_outputs_match_all_compared_cases")
        self.assertTrue(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["nonsingular_fail_count"], 0)
        for key_result in summary["nonsingular_outputs"]:
            self.assertEqual(key_result["comparison_status"], "passed")
            self.assertLessEqual(key_result["max_abs_error"], ATOL)

    def test_cases_combine_depth_and_symmetry_constrained_d(self):
        self.assertEqual(self.reference["case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(self.reference["functionDownValues"]["f4NL"], 1)
        pgs = {c["point_group"] for c in self.python["cases"]}
        self.assertEqual(pgs, POINT_GROUPS)
        # BOTH axes must be present in every case: 4-layer depth AND a genuine
        # symmetry-constrained point-group d-tensor (fewer than 18 nonzero d).
        for c in self.python["cases"]:
            self.assertEqual(c["layer_count"], 4)
            self.assertIn(c["point_group"], POINT_GROUPS)
            self.assertLess(c["nonzero_d_entries"], 18)
            self.assertGreater(c["nonzero_d_entries"], 0)


if __name__ == "__main__":
    unittest.main()
