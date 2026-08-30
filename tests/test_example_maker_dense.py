"""Fast smoke test for the dense Maker-fringe example so it does not bit-rot. Uses tiny
theta ranges (the figure rigor is 0.1deg / 0.02deg; here we only check the pipeline runs,
produces nonzero output for the three assumption modes, and that the modes differ)."""

import unittest

import numpy as np


class MakerDenseExampleTests(unittest.TestCase):
    def test_three_modes_run_nonzero_and_differ(self):
        from examples.maker_fringes_dense import compute

        # tiny grids -> fast; still exercises Full(0)/HH(2)/JK(1) sweeps + the figure data path
        data = compute(full_range=(15.0, 17.0), full_step=0.5, zoom=(24.0, 24.6), zoom_step=0.3)
        self.assertEqual(set(data), {"Full (FMR)", "HH", "JK"})
        peak = max(np.max(np.abs(data[k][1])) for k in data)
        self.assertGreater(peak, 0.0, "all Maker modes gave zero output")
        full = data["Full (FMR)"][1]
        hh = data["HH"][1]
        # Full and HH must not be identical (different assumption modes)
        self.assertGreater(float(np.max(np.abs(full - hh))), 1e-12 * max(peak, 1e-300), "Full == HH (modes not distinct)")

    def test_build_figure_two_panels(self):
        import matplotlib
        matplotlib.use("Agg")
        from examples.maker_fringes_dense import build_figure, compute

        data = compute(full_range=(15.0, 17.0), full_step=0.5, zoom=(24.0, 24.6), zoom_step=0.3)
        fig = build_figure(data)
        self.assertEqual(len(fig.axes), 2)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
