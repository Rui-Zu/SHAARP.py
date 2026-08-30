import json
import unittest
from pathlib import Path

from benchmarks.compare_symbolic_reference import compare_symbolic_payloads
from benchmarks.generate_symbolic_ml_partial_pnl_python_reference import SOURCE_SPECS, build_payload


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    ROOT
    / "benchmarks"
    / "mathematica_reference"
    / "wolfram_live_symbolic_ml_partial_pnl_reference_v1.json"
)
PYTHON_PATH = ROOT / "benchmarks" / "python_symbolic_ml_partial_pnl_reference_v1.json"


class WolframLiveSymbolicMLPartialPNLReferenceTests(unittest.TestCase):
    def test_saved_python_symbolic_payload_matches_generator(self):
        saved = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(saved, build_payload())

    def test_live_wolfram_symbolic_ten_source_partial_pnl_matches_python_sympy(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)
        expected_ids = {f"ml_partial_{spec[1]}" for spec in SOURCE_SPECS}

        self.assertEqual(len(reference["expressions"]), 10)
        self.assertEqual({result.expression_id for result in results}, expected_ids)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(sum(result.check == "symbolic_residual" for result in results), 10)
        self.assertEqual(sum(result.check == "numeric_substitution[1]" for result in results), 10)


if __name__ == "__main__":
    unittest.main()
