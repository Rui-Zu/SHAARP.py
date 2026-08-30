import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import (
    build_maker_reference_agreement_summary,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_ml1.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_ml1.json"

ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 6


class MakerFringesMultilayerReferenceComparisonTests(unittest.TestCase):
    """Genuine MULTILAYER (4-layer) SHAARP.ml Maker SHG validation: a stack
    air / active-film / isotropic-interlayer / substrate, incidence-swept,
    validated value-by-value against live Wolfram 14.3 f4NL.

    This exercises the multilayer fundamental + 2omega boundary solve with a
    DEGENERATE (isotropic) internal layer. That degeneracy previously broke the
    solve (the two independent per-mode Snell solves picked an arbitrary
    degenerate-plane eigenbasis that collapsed two forward basis waves onto the
    same vector -> rank-deficient fundamental matrix at scattered high angles).
    Fixed by building the degenerate mode pair from a SINGLE
    `modes_for_direction` call with a fixed s/p fallback (mirroring SHAARP.ml's
    solveSnell 'Iso or quasi Iso' branch). See the project notes
    ' Multilayer Discrepancy RESOLVED'."""

    def setUp(self):
        self.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

    def test_reference_is_live_wolfram_sourced_4_layer(self):
        source = str(self.reference.get("source", "")).lower()
        self.assertTrue("mathematica" in source or "wolfram" in source)
        self.assertEqual(self.reference.get("case_count"), EXPECTED_CASE_COUNT)
        self.assertEqual(self.reference["functionDownValues"]["f4NL"], 1)
        items = self.reference["suites"][0]["items"]
        self.assertFalse([item for item in items if "error" in item])
        for case in self.python["cases"]:
            self.assertEqual(case["layer_count"], 4)

    def test_multilayer_grid_matches_live_mathematica(self):
        """All 6 four-layer cases must now match live Wolfram f4NL to roundoff,
        across every compared output (MFList cols, listMFpara/perp, transmitted
        Jones, analyzer amplitudes). This is the post-fix full-agreement state."""
        summary = build_maker_reference_agreement_summary(
            self.reference, self.python, atol=ATOL, rtol=RTOL
        )
        self.assertEqual(summary["status"], "maker_outputs_match_all_compared_cases")
        self.assertTrue(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["nonsingular_fail_count"], 0)
        self.assertEqual(summary["diagnostic_case_ids"], [])
        self.assertTrue(summary["nonsingular_outputs"])
        for key_result in summary["nonsingular_outputs"]:
            self.assertEqual(key_result["comparison_status"], "passed")
            self.assertEqual(key_result["case_count"], EXPECTED_CASE_COUNT)
            self.assertLessEqual(key_result["max_abs_error"], ATOL)


if __name__ == "__main__":
    unittest.main()
