import json
import unittest
from pathlib import Path

from benchmarks.compare_symbolic_reference import compare_symbolic_payloads
from benchmarks.generate_symbolic_compute_pnl_python_reference import build_payload


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    ROOT
    / "benchmarks"
    / "mathematica_reference"
    / "wolfram_live_symbolic_compute_pnl_reference_v1.json"
)
PYTHON_PATH = ROOT / "benchmarks" / "python_symbolic_compute_pnl_reference_v1.json"


class WolframLiveSymbolicComputePNLReferenceTests(unittest.TestCase):
    def test_saved_python_symbolic_payload_matches_generator(self):
        saved = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(saved, build_payload())

    def test_live_wolfram_symbolic_compute_pnl_matches_python_sympy(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertIn("Wolfram", reference["source"])
        self.assertEqual(reference["status"], "mathematica_symbolic_reference_exported")
        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)

        self.assertEqual({result.expression_id for result in results}, {"compute_pnl_same_symbolic", "compute_pnl_mixed_symbolic"})
        self.assertTrue(all(result.passed for result in results))
        self.assertIn("symbolic_residual", {result.check for result in results})
        self.assertIn("numeric_substitution[1]", {result.check for result in results})


if __name__ == "__main__":
    unittest.main()
