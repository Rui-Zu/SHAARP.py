import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import (
    build_maker_reference_agreement_summary,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_ml8.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_ml8.json"

ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 4


@unittest.skipUnless(REFERENCE_PATH.exists(), "ml8 live Wolfram reference not yet exported")
class MakerFringesMultilayerMl8ReferenceComparisonTests(unittest.TestCase):
    """9-LAYER SHAARP.ml Maker SHG depth validation: a stack
    air / active-film / ANISOTROPIC-interlayer / isotropic-interlayer /
    isotropic-interlayer-2 / isotropic-interlayer-3 / ANISOTROPIC-interlayer-2 /
    isotropic-interlayer-4 / substrate, incidence-swept, validated value-by-value
    against live Wolfram 14.3 f4NL.

    Extends the ml7 8-layer grid one internal layer deeper (a seventh internal
    layer + a sixth internal interface, with six distinct passive interlayers:
    two anisotropic + four isotropic). Only the film is SHG-active. See
    the project notes ' Multilayer Depth: 9-Layer ml8'."""

    def setUp(self):
        self.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

    def test_reference_is_live_wolfram_sourced_9_layer(self):
        source = str(self.reference.get("source", "")).lower()
        self.assertTrue("mathematica" in source or "wolfram" in source)
        self.assertEqual(self.reference.get("case_count"), EXPECTED_CASE_COUNT)
        self.assertEqual(self.reference["functionDownValues"]["f4NL"], 1)
        items = self.reference["suites"][0]["items"]
        self.assertFalse([item for item in items if "error" in item])
        for case in self.python["cases"]:
            self.assertEqual(case["layer_count"], 9)

    def test_ml8_grid_matches_live_mathematica(self):
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
