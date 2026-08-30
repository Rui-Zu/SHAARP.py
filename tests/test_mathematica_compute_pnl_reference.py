import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.nonlinear import compute_pnl_voigt


REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "mathematica_reference"
    / "compute_pnl_reference_v1.json"
)


class MathematicaComputePNLReferenceTests(unittest.TestCase):
    def test_mathematica_compute_pnl_reference_matches_python_values(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertIn("Mathematica", payload["source"])
        self.assertEqual(payload["status"], "mathematica_reference_exported")
        self.assertEqual(payload["inputCellCount"], 49)
        self.assertEqual(payload["functionDownValues"]["computePNL"], 1)
        self.assertEqual(len(payload["cases"]), 20)

        for case in payload["cases"]:
            with self.subTest(case=case["id"]):
                inputs = case["inputs"]
                d = numeric_array(inputs["d_voigt_lab"])
                e1 = numeric_array(inputs["e1"])
                e2 = numeric_array(inputs["e2"])
                factor = numeric_array(inputs["factor"]).item()
                expected = numeric_array(case["outputs"]["pnl"])

                actual = compute_pnl_voigt(d, e1, e2, factor=factor)

                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
