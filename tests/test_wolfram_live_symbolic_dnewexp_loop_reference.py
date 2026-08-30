import json
import unittest
from pathlib import Path

from benchmarks.compare_symbolic_reference import compare_symbolic_payloads
from benchmarks.generate_symbolic_dnewexp_loop_python_reference import build_payload


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    ROOT
    / "benchmarks"
    / "mathematica_reference"
    / "wolfram_live_symbolic_dnewexp_loop_reference_v1.json"
)
PYTHON_PATH = ROOT / "benchmarks" / "python_symbolic_dnewexp_loop_reference_v1.json"


class WolframLiveSymbolicDnewExpLoopReferenceTests(unittest.TestCase):
    """SHAARP.si's OWN GUI dnewExp rotation For-loop (extracted live and run with a
    rational crystal->lab matrix and a symbolic crystal SHG tensor) is validated
    to be EXACTLY symbolically equal to Python's rotate_d_voigt_symbolic. This is
    the SI-specific lab-frame d rotation, distinct from the ML extMater dL form."""

    def test_saved_python_symbolic_payload_matches_generator(self):
        saved = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_payload())

    def test_reference_is_live_wolfram_sourced_and_orthonormal(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertIn("Wolfram", reference["source"])
        self.assertIn("SHAARP.si", reference["source"])
        self.assertEqual(reference["status"], "mathematica_symbolic_reference_exported")
        self.assertTrue(reference["a_orthonormal"])

    def test_live_wolfram_dnewexp_loop_matches_python(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        results = compare_symbolic_payloads(reference, python, atol=1e-12, rtol=1e-12)

        self.assertEqual({r.expression_id for r in results}, {"dnewExp_loop_rotated_d_voigt"})
        self.assertTrue(
            all(r.passed for r in results),
            msg=str([(r.check, r.detail) for r in results if not r.passed]),
        )
        checks = {r.check for r in results if r.passed}
        self.assertIn("symbolic_residual", checks)
        self.assertIn("symbol_set", checks)
        self.assertIn("numeric_substitution[0]", checks)


if __name__ == "__main__":
    unittest.main()
