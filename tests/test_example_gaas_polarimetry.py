"""Smoke test for the GaAs(111) SHG-polarimetry example, so it does not bit-rot: its
closed-form compute path runs and produces a nonzero, phi-varying polarimetry curve, and
the figure builder returns a 2-panel matplotlib figure."""

import math
import unittest

import numpy as np


class GaAsPolarimetryExampleTests(unittest.TestCase):
    def test_channel_funcs_give_nonzero_phi_varying_curve(self):
        from examples.gaas111_shg_polarimetry import _channel_funcs

        i_s, i_p = _channel_funcs(0.0)
        phi = np.linspace(0.0, 360.0, 73)
        cp = i_p(phi)
        cs = i_s(phi)
        self.assertGreater(float(np.max(cp)), 1e-8, "p-curve ~ 0 (vacuous)")
        self.assertGreater(float(np.max(cs)), 1e-8, "s-curve ~ 0 (vacuous)")
        self.assertGreater(float(np.max(cp) - np.min(cp)), 0.3 * float(np.max(cp)), "p-curve does not vary with phi")

    def test_threefold_azimuth_in_example(self):
        from examples.gaas111_shg_polarimetry import _channel_funcs

        phi = np.array([20.0, 75.0, 130.0])
        _, p0 = _channel_funcs(0.0)
        _, p120 = _channel_funcs(math.radians(120.0))
        _, p60 = _channel_funcs(math.radians(60.0))
        self.assertLess(float(np.max(np.abs(p0(phi) - p120(phi)))), 1e-9, "azimuth 0 vs 120 not equal (3-fold broken)")
        self.assertGreater(float(np.max(np.abs(p0(phi) - p60(phi)))), 1e-6, "azimuth 0 vs 60 equal (should differ)")

    def test_build_figure_returns_two_panel_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        from examples.gaas111_shg_polarimetry import build_figure

        fig = build_figure()
        self.assertEqual(len(fig.axes), 2, "expected a 2-panel figure")
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
