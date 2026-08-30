import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.nonlinear import compute_pnl_voigt


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "wolfram_live_compute_pnl_smoke_v1.json"


class WolframLiveComputePNLSmokeTests(unittest.TestCase):
    def test_live_wolfram_compute_pnl_smoke_matches_python_values(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertIn("Wolfram", payload["source"])
        self.assertEqual(payload["status"], "mathematica_reference_exported")
        self.assertEqual(payload["batch_execution"], "file_export_confirmed")
        self.assertEqual(len(payload["cases"]), 1)

        case = payload["cases"][0]
        inputs = case["inputs"]
        d = numeric_array(inputs["d_voigt_lab"])
        e1 = numeric_array(inputs["e1"])
        e2 = numeric_array(inputs["e2"])
        factor = numeric_array(inputs["factor"]).item()
        expected = numeric_array(case["outputs"]["pnl"])

        actual = compute_pnl_voigt(d, e1, e2, factor=factor)

        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
