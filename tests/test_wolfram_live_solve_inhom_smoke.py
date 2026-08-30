import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.nonlinear import NonlinearSource, solve_inhomogeneous_field


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "wolfram_live_solve_inhom_smoke_v1.json"


class WolframLiveSolveInhomSmokeTests(unittest.TestCase):
    def test_live_wolfram_solve_inhom_smoke_matches_python_values(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertIn("Wolfram", payload["source"])
        self.assertEqual(payload["status"], "mathematica_reference_exported")
        self.assertEqual(payload["batch_execution"], "file_export_confirmed")
        self.assertEqual(len(payload["cases"]), 1)

        case = payload["cases"][0]
        inputs = case["inputs"]
        epsilon = numeric_array(inputs["epsilon_2omega_lab"])
        k = numeric_array(inputs["k"])
        polarization = numeric_array(inputs["polarization"])
        omega = numeric_array(inputs["omega"]).item()
        expected_electric = numeric_array(case["outputs"]["electric"])
        expected_residual = numeric_array(case["outputs"]["residual"])

        actual = solve_inhomogeneous_field(
            epsilon,
            NonlinearSource(label="live", k=k, polarization=polarization, omega=omega),
            mu=1.0,
            eps0=1.0,
        )

        np.testing.assert_allclose(actual.electric, expected_electric, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(actual.residual, expected_residual, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
