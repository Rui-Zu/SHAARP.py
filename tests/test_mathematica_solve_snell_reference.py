import json
import math
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.anisotropic import solve_snell_modes


REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "mathematica_reference"
    / "solve_snell_reference_v1.json"
)


class MathematicaSolveSnellReferenceTests(unittest.TestCase):
    def test_mathematica_solve_snell_reference_matches_python_values(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertIn("Mathematica", payload["source"])
        self.assertEqual(payload["status"], "mathematica_reference_exported")
        self.assertEqual(payload["inputCellCount"], 49)
        self.assertEqual(payload["functionDownValues"]["solveSnell"], 1)
        self.assertEqual(payload["selectedKeys"]["epsilonOmegaL"], [949, 969, 76])
        self.assertEqual(payload["case_count"], 20)
        self.assertEqual(len(payload["cases"]), 20)

        for case in payload["cases"]:
            with self.subTest(case=case["id"]):
                inputs = case["inputs"]
                epsilon = numeric_array(inputs["epsilon_lab"])
                theta = math.radians(float(inputs["theta_deg"]))
                reflected = bool(inputs["reflected"])
                omega = numeric_array(inputs["omega"]).item()

                modes = solve_snell_modes(epsilon, incident_index=1.0, incident_theta_rad=theta, reflected=reflected)

                self._assert_mode_matches(case, "fast", modes.fast, omega)
                self._assert_mode_matches(case, "slow", modes.slow, omega)

    def _assert_mode_matches(self, case, key, mode, omega):
        outputs = case["outputs"]
        expected_theta = numeric_array(outputs[f"{key}_theta"]).item()
        expected_n = numeric_array(outputs[f"{key}_refractive_index"]).item()
        expected_k = numeric_array(outputs[f"{key}_wavevector"])
        expected_e = numeric_array(outputs[f"{key}_electric_direction"])

        self.assertAlmostEqual(mode.theta_rad.real, expected_theta.real, places=10)
        self.assertAlmostEqual(mode.theta_rad.imag, expected_theta.imag, places=10)
        np.testing.assert_allclose(mode.refractive_index, expected_n, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(mode.refractive_index * omega * mode.direction, expected_k, atol=1e-9, rtol=1e-9)
        self.assertLess(_phase_aligned_error(mode.electric_direction, expected_e), 1e-9)


def _phase_aligned_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = np.vdot(expected, actual) / np.vdot(expected, expected)
    return float(np.linalg.norm(actual - scale * expected))


if __name__ == "__main__":
    unittest.main()
