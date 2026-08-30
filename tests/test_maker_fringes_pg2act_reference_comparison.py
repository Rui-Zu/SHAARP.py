import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import build_maker_reference_agreement_summary

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_pg2act.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_pg2act.json"

ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 4


@unittest.skipUnless(REFERENCE_PATH.exists(), "pg2act live Wolfram reference not present")
class MakerFringesPointGroupTwoActivePg2actReferenceComparisonTests(unittest.TestCase):
    """POINT-GROUP x COUPLED-SOURCE end-to-end Maker SHG: TWO adjacent active films
    (air / pg-film-1 / pg-film-2 / substrate), EACH carrying a SYMMETRY-CONSTRAINED
    point-group d-tensor. The transmitted 2omega signal is the coherent superposition
    of two symmetry-constrained nonlinear sources. Generalizes ml3 (two coupled
    generic-d sources) and pgml (one point-group film at depth) to the intersection,
    validated vs live Wolfram f4NL -- a combination never tested before."""

    def setUp(self):
        self.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

    def test_pg2act_grid_matches_live_mathematica(self):
        summary = build_maker_reference_agreement_summary(self.reference, self.python, atol=ATOL, rtol=RTOL)
        self.assertEqual(summary["status"], "maker_outputs_match_all_compared_cases")
        self.assertTrue(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["nonsingular_fail_count"], 0)
        for key_result in summary["nonsingular_outputs"]:
            self.assertEqual(key_result["comparison_status"], "passed")
            self.assertLessEqual(key_result["max_abs_error"], ATOL)

    def test_cases_combine_two_active_and_symmetry_constrained_d(self):
        self.assertEqual(self.reference["case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(self.reference["functionDownValues"]["f4NL"], 1)
        # BOTH axes must be present in every case: two coupled active layers AND each
        # active film a genuine symmetry-constrained point-group d (fewer than 18 nonzero).
        for c in self.python["cases"]:
            self.assertEqual(c["layer_count"], 4)
            self.assertEqual(c["active_layer_count"], 2)
            self.assertEqual(len(c["point_groups"]), 2)
            self.assertEqual(len(c["nonzero_d_entries"]), 2)
            for nz in c["nonzero_d_entries"]:
                self.assertGreater(nz, 0)
                self.assertLess(nz, 18)


if __name__ == "__main__":
    unittest.main()
