import json
import unittest
from pathlib import Path

from benchmarks.compare_symbolic_reference import compare_symbolic_payloads
from benchmarks.generate_symbolic_rotated_d_python_reference import build_payload


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    ROOT
    / "benchmarks"
    / "mathematica_reference"
    / "wolfram_live_symbolic_rotated_d_reference_v1.json"
)
PYTHON_PATH = ROOT / "benchmarks" / "python_symbolic_rotated_d_reference_v1.json"


class WolframLiveSymbolicRotatedDReferenceTests(unittest.TestCase):
    """The SHAARP crystal->lab Voigt-d rotation (dnewExp / extMater dL) is an
    analytical building block of SHAARP.si full / SHAARP.ml partial expressions.
    This validates Python's symbolic rotate_d_voigt_symbolic against live
    Wolfram 14.3 extMater dL for an exact-rational orientation, checking both an
    exact symbolic residual and numeric substitution."""

    def test_saved_python_symbolic_payload_matches_generator(self):
        saved = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_payload())

    def test_live_wolfram_symbolic_rotated_d_matches_python_sympy(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertIn("Wolfram", reference["source"])
        self.assertEqual(reference["status"], "mathematica_symbolic_reference_exported")
        # Both exact-rational orientations must be orthonormal in Wolfram.
        self.assertTrue(reference["qcA_orthonormal"])
        self.assertTrue(reference["qcB_orthonormal"])

        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)

        expected_ids = {"rotated_d_voigt_symbolic", "rotated_d_voigt_symbolic_b"}
        self.assertEqual({result.expression_id for result in results}, expected_ids)
        self.assertTrue(all(result.passed for result in results), msg=str([r for r in results if not r.passed]))
        # Each orientation must pass an EXACT symbolic residual and both numeric subs.
        for expression_id in expected_ids:
            checks = {r.check for r in results if r.expression_id == expression_id and r.passed}
            self.assertIn("symbolic_residual", checks)
            self.assertIn("symbol_set", checks)
            self.assertIn("numeric_substitution[0]", checks)
            self.assertIn("numeric_substitution[1]", checks)


if __name__ == "__main__":
    unittest.main()
