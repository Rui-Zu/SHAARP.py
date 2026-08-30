import json
import math
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.boundary import solve_fresnel_boundary
from shaarp.waves import make_isotropic_wave


REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "mathematica_reference"
    / "solve_fresnel_reference_v1.json"
)


class MathematicaSolveFresnelReferenceTests(unittest.TestCase):
    def test_mathematica_solve_fresnel_reference_matches_python_values(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertIn("Mathematica", payload["source"])
        self.assertEqual(payload["status"], "mathematica_reference_exported")
        self.assertEqual(payload["inputCellCount"], 49)
        self.assertEqual(payload["functionDownValues"]["solveFresnel"], 1)
        self.assertEqual(payload["unknown_order"], ["rS", "rP", "tS", "tP"])
        self.assertEqual(payload["case_count"], 20)
        self.assertEqual(len(payload["cases"]), 20)

        for case in payload["cases"]:
            with self.subTest(case=case["id"]):
                inputs = case["inputs"]
                n1 = numeric_array(inputs["n1"]).item()
                n2 = numeric_array(inputs["n2"]).item()
                theta_i = math.radians(float(inputs["theta_deg"]))
                theta_t = numeric_array(inputs["theta_transmitted_rad"]).item()
                pol = str(inputs["incident_polarization"])

                incident = [make_isotropic_wave(n=n1, theta_rad=theta_i, polarization=pol, amplitude=1.0)]
                reflected_basis = [
                    make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="s", direction_sign_z=-1.0),
                    make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="p", direction_sign_z=-1.0),
                ]
                transmitted_basis = [
                    make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="s"),
                    make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="p"),
                ]
                reflected, transmitted, coeffs = solve_fresnel_boundary(
                    incident,
                    reflected_basis,
                    transmitted_basis,
                    mu=1.0,
                )

                np.testing.assert_allclose(coeffs, numeric_array(case["outputs"]["coefficients"]), atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(
                    np.array([wave.electric for wave in reflected]),
                    numeric_array(case["outputs"]["reflected_electric"]),
                    atol=1e-10,
                    rtol=1e-10,
                )
                np.testing.assert_allclose(
                    np.array([wave.electric for wave in transmitted]),
                    numeric_array(case["outputs"]["transmitted_electric"]),
                    atol=1e-10,
                    rtol=1e-10,
                )


if __name__ == "__main__":
    unittest.main()
