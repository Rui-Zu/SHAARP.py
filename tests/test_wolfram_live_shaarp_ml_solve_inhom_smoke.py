import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.nonlinear import NonlinearSource, compute_pnl_voigt, solve_inhomogeneous_field


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "solve_inhom_smoke_output.json"


class WolframLiveShaarpMLSolveInhomSmokeTests(unittest.TestCase):
    def test_live_shaarp_ml_solve_inhom_smoke_matches_python_sources(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertIn("Mathematica SHAARP.ml", payload["source"])
        self.assertEqual(payload["functionDownValues"]["solveInhom"], 2)
        self.assertEqual(payload["inputCellCount"], 49)

        d = np.array(
            [
                [0.3 + 0.2j, -0.4j, 0.7, 0.2 - 0.1j, 0.05j, -0.6],
                [0.1j, 0.5 + 0.15j, -0.25, 0.45j, 0.8, -0.2 + 0.05j],
                [-0.15, 0.25j, 0.35 + 0.2j, -0.5, 0.65j, 0.4],
            ],
            dtype=complex,
        )
        epsilon = np.array(
            [
                [2.6 + 0.15j, 0.08 - 0.03j, -0.05 + 0.02j],
                [0.04 + 0.07j, 3.1 + 0.11j, 0.06 - 0.01j],
                [-0.02 + 0.05j, 0.03 + 0.04j, 3.7 + 0.09j],
            ],
            dtype=complex,
        )
        e1 = np.array([0.8 + 0.1j, 0.04j, -0.22], dtype=complex)
        e2 = np.array([0.03, 0.91 - 0.18j, 0.11j], dtype=complex)
        k1 = np.array([0.31, 0.0, 1.12], dtype=complex)
        k2 = np.array([0.31, 0.0, 1.37], dtype=complex)

        sources = {
            "ee": NonlinearSource("ee", 2 * k1, compute_pnl_voigt(d, e1, e1), 2.8),
            "oo": NonlinearSource("oo", 2 * k2, compute_pnl_voigt(d, e2, e2), 2.8),
            "eo": NonlinearSource("eo", k1 + k2, compute_pnl_voigt(d, e1, e2, factor=2), 2.8),
        }

        for label, source in sources.items():
            with self.subTest(label=label):
                actual = solve_inhomogeneous_field(epsilon, source, mu=1.0, eps0=1.0)
                expected = numeric_array(payload["outputs"][label])
                np.testing.assert_allclose(actual.electric, expected, atol=1e-10, rtol=1e-10)
                self.assertLess(np.linalg.norm(actual.residual), 1e-9)


if __name__ == "__main__":
    unittest.main()
