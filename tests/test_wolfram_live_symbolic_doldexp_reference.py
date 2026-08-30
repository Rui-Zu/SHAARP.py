import json
import unittest
from pathlib import Path

from benchmarks.compare_symbolic_reference import compare_symbolic_payloads
from benchmarks.generate_symbolic_doldexp_python_reference import build_payload


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    ROOT
    / "benchmarks"
    / "mathematica_reference"
    / "wolfram_live_symbolic_doldexp_reference_v1.json"
)
PYTHON_PATH = ROOT / "benchmarks" / "python_symbolic_doldexp_reference_v1.json"

EXPECTED_POINT_GROUP_COUNT = 16


class WolframLiveSymbolicDoldExpReferenceTests(unittest.TestCase):
    """Python's d_voigt_symbolic point-group SHG Voigt templates are validated to
    be EXACTLY symbolically equal to SHAARP.si's doldExp templates, extracted
    live from the SHAARP_V1.03.nb GUI/analytical cell (box-tree extraction, since
    the 1 MB Manipulate cell will not ToExpression as a whole)."""

    def test_saved_python_symbolic_payload_matches_generator(self):
        saved = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_payload())

    def test_reference_is_live_wolfram_sourced_with_all_point_groups(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertIn("Wolfram", reference["source"])
        self.assertIn("SHAARP.si", reference["source"])
        self.assertEqual(reference["status"], "mathematica_symbolic_reference_exported")
        self.assertEqual(reference["record_count"], EXPECTED_POINT_GROUP_COUNT)

    def test_live_wolfram_doldexp_matches_python_for_every_point_group(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)

        # 16 point groups must all be present and every check must pass.
        expression_ids = {result.expression_id for result in results}
        self.assertEqual(len(expression_ids), EXPECTED_POINT_GROUP_COUNT)
        self.assertTrue(
            all(result.passed for result in results),
            msg=str([(r.expression_id, r.check, r.detail) for r in results if not r.passed]),
        )
        # Every point group must clear an EXACT symbolic residual and matching symbol set.
        for expression_id in expression_ids:
            checks = {r.check for r in results if r.expression_id == expression_id and r.passed}
            self.assertIn("symbolic_residual", checks)
            self.assertIn("symbol_set", checks)
            self.assertIn("numeric_substitution[0]", checks)


if __name__ == "__main__":
    unittest.main()
