"""Smoke test for the d-extraction demo example: the simulated-scan -> recover-d workflow
runs, recovers the known tensor (identifiable, to ~1e-9), and the figure builds."""

import unittest

import numpy as np


class DExtractionDemoExampleTests(unittest.TestCase):
    def test_run_extraction_recovers_true_tensor(self):
        from examples.d_extraction_demo import TRUE, run_extraction

        res, _ = run_extraction()
        self.assertTrue(res.identifiable, f"not identifiable (rank {res.rank}/{res.n_components})")
        self.assertLess(res.residual, 1e-10, "model inconsistent with simulated data")
        self.assertLess(float(np.max(np.abs(np.real(res.values) - TRUE))), 1e-9, "d not recovered")

    def test_build_figure_returns_two_panel_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        from examples.d_extraction_demo import build_figure

        fig = build_figure()
        self.assertEqual(len(fig.axes), 2, "expected a 2-panel figure")
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
