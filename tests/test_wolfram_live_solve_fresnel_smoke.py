import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.linear import make_isotropic_wave, solve_fresnel_boundary


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "solve_fresnel_smoke_output.json"


class WolframLiveSolveFresnelSmokeTests(unittest.TestCase):
    def test_live_shaarp_ml_solve_fresnel_smoke_matches_python_boundary_values(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertIn("Mathematica SHAARP.ml", payload["source"])
        self.assertEqual(payload["functionDownValues"]["solveFresnel"], 1)
        self.assertEqual(payload["inputCellCount"], 49)

        n1 = 1.0
        n2 = 1.7
        theta_i = np.deg2rad(35.0)
        theta_t = np.arcsin((n1 / n2) * np.sin(theta_i))
        incident = [make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="s")]
        reflected_basis = [
            make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="s", direction_sign_z=-1),
            make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="p", direction_sign_z=-1),
        ]
        transmitted_basis = [
            make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="s"),
            make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="p"),
        ]

        reflected, transmitted, coeffs = solve_fresnel_boundary(incident, reflected_basis, transmitted_basis, mu=1.0)

        np.testing.assert_allclose(coeffs, numeric_array(payload["solution"]), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray([wave.electric for wave in reflected]),
            numeric_array(payload["reflected_electric"]),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray([wave.electric for wave in transmitted]),
            numeric_array(payload["transmitted_electric"]),
            atol=1e-12,
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
