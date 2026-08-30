import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.nonlinear import (
    NonlinearSource,
    compute_pnl_voigt,
    solve_inhomogeneous_field,
)


REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "mathematica_reference"
    / "solve_inhom_reference_v1.json"
)


class MathematicaSolveInhomReferenceTests(unittest.TestCase):
    def test_mathematica_solve_inhom_reference_matches_python_values(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertIn("Mathematica", payload["source"])
        self.assertEqual(payload["status"], "mathematica_reference_exported")
        self.assertEqual(payload["inputCellCount"], 49)
        self.assertEqual(payload["functionDownValues"]["solveInhom"], 2)
        self.assertEqual(payload["selectedKeys"]["epsilon2omegaLCharacterCodes"], [949, 50, 969, 76])
        self.assertEqual(len(payload["cases"]), 20)

        for case in payload["cases"]:
            with self.subTest(case=case["id"]):
                inputs = case["inputs"]
                omega = numeric_array(inputs["omega"]).item()
                epsilon = numeric_array(inputs["epsilon_2omega_lab"])
                d = numeric_array(inputs["d_voigt_lab"])
                e1 = numeric_array(inputs["e1"])
                e2 = numeric_array(inputs["e2"])
                k1 = numeric_array(inputs["k1"])
                k2 = numeric_array(inputs["k2"])
                expected_ee = numeric_array(case["outputs"]["ee"])
                expected_oo = numeric_array(case["outputs"]["oo"])
                expected_eo = numeric_array(case["outputs"]["eo"])

                sources = [
                    NonlinearSource(
                        label="ee",
                        k=2 * k1,
                        polarization=compute_pnl_voigt(d, e1, e1),
                        omega=2 * omega,
                    ),
                    NonlinearSource(
                        label="oo",
                        k=2 * k2,
                        polarization=compute_pnl_voigt(d, e2, e2),
                        omega=2 * omega,
                    ),
                    NonlinearSource(
                        label="eo",
                        k=k1 + k2,
                        polarization=compute_pnl_voigt(d, e1, e2, factor=2),
                        omega=2 * omega,
                    ),
                ]
                expected = [expected_ee, expected_oo, expected_eo]

                for source, expected_field in zip(sources, expected):
                    actual = solve_inhomogeneous_field(epsilon, source, mu=1.0, eps0=1.0)
                    np.testing.assert_allclose(actual.electric, expected_field, atol=1e-10, rtol=1e-10)
                    self.assertLess(np.linalg.norm(actual.residual), 1e-9)


if __name__ == "__main__":
    unittest.main()
