import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.anisotropic import solve_snell_modes


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "solve_snell_smoke_output.json"


class WolframLiveSolveSnellSmokeTests(unittest.TestCase):
    def test_live_shaarp_ml_solve_snell_smoke_matches_python_modes(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertIn("Mathematica SHAARP.ml", payload["source"])
        self.assertEqual(payload["functionDownValues"]["solveSnell"], 1)
        self.assertEqual(payload["inputCellCount"], 49)

        inputs = payload["inputs"]
        outputs = payload["outputs"]
        epsilon = numeric_array(inputs["epsilon_lab"])
        theta = float(np.real(numeric_array(inputs["theta_incident_rad"]).item()))
        omega = 2 * np.pi / 0.8
        modes = solve_snell_modes(epsilon, incident_index=1.0, incident_theta_rad=theta)

        self._assert_mode("fast", modes.fast, outputs, omega)
        self._assert_mode("slow", modes.slow, outputs, omega)

    def _assert_mode(self, key, actual, outputs, omega):
        expected_theta = numeric_array(outputs[f"{key}_theta"]).item()
        expected_n = numeric_array(outputs[f"{key}_refractive_index"]).item()
        expected_k = numeric_array(outputs[f"{key}_wavevector"])
        expected_e = numeric_array(outputs[f"{key}_electric_direction"])

        np.testing.assert_allclose(actual.theta_rad, expected_theta, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(actual.refractive_index, expected_n, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(actual.refractive_index * omega * actual.direction, expected_k, atol=1e-9, rtol=1e-9)
        self.assertLess(_phase_aligned_error(actual.electric_direction, expected_e), 1e-9)


def _phase_aligned_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = np.vdot(expected, actual) / np.vdot(expected, expected)
    return float(np.linalg.norm(actual - scale * expected))


if __name__ == "__main__":
    unittest.main()
