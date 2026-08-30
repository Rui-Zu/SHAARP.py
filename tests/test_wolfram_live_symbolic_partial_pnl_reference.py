import json
import unittest
from pathlib import Path

from benchmarks.compare_symbolic_reference import compare_symbolic_payloads
from benchmarks.generate_symbolic_partial_pnl_python_reference import build_payload


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    ROOT
    / "benchmarks"
    / "mathematica_reference"
    / "wolfram_live_symbolic_partial_pnl_reference_v1.json"
)
PYTHON_PATH = ROOT / "benchmarks" / "python_symbolic_partial_pnl_reference_v1.json"

EXPECTED_IDS = {"partial_pnl_P2o", "partial_pnl_P2eo", "partial_pnl_Peoo"}


class WolframLiveSymbolicPartialPNLReferenceTests(unittest.TestCase):
    """SHAARP.si's partial-analytical nonlinear-source polarizations (P2o same-mode,
    Peoo mixed factor-2), extracted live from the SHAARP_V1.03.nb GUI cell with a
    symbolic d-tensor and symbolic eigenmode fields, are validated to be EXACTLY
    symbolically equal to Python's compute_pnl_voigt_symbolic assembly."""

    def test_saved_python_symbolic_payload_matches_generator(self):
        saved = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_payload())

    def test_reference_is_live_wolfram_sourced(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertIn("Wolfram", reference["source"])
        self.assertIn("SHAARP.si", reference["source"])
        self.assertEqual(reference["status"], "mathematica_symbolic_reference_exported")
        self.assertEqual(reference["record_count"], 3)

    def test_live_wolfram_partial_pnl_matches_python(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        results = compare_symbolic_payloads(reference, python, atol=1e-9, rtol=1e-9)

        self.assertEqual({r.expression_id for r in results}, EXPECTED_IDS)
        self.assertTrue(
            all(r.passed for r in results),
            msg=str([(r.expression_id, r.check, r.detail) for r in results if not r.passed]),
        )
        for expression_id in EXPECTED_IDS:
            checks = {r.check for r in results if r.expression_id == expression_id and r.passed}
            self.assertIn("symbolic_residual", checks)
            self.assertIn("symbol_set", checks)
            self.assertIn("numeric_substitution[0]", checks)
            self.assertIn("numeric_substitution[1]", checks)


if __name__ == "__main__":
    unittest.main()
