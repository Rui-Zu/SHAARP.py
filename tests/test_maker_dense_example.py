"""Continuity lens on the dense Maker-fringe benchmark example (examples/maker_fringes_dense.py).

Two real artifacts were caught in this figure by the author's standing lens ("every plot should be
continuous; a non-continuous plot means something is off"):
  1. the original quartz-like config was a near symmetry-NULL whose 'fringes' were cancellation
     noise at the ~1e-6 floor;
  2. at physical SI units (MU0/EPS0) the transmitted field TELEGRAPHS between exact fractions
     E/n (E, E/2, E/3, E/4, 0) of the correct value at scattered angles -- a units-scale-induced
     solver instability (the same window at natural units mu=1, eps0=1 is perfectly smooth, and
     natural units are what every live-Mathematica validation used).

The example now sweeps at natural units with a strong real crystal (panel a) and uses the
VALIDATED pilot configuration for the fine-fringe zoom (panel b). These tests pin both.
"""

import unittest

import numpy as np

from examples.maker_fringes_dense import _fine_system, _sweep, _system


class MakerDenseExampleTests(unittest.TestCase):
    def test_broad_sweep_is_continuous_at_natural_units(self):
        # the exact window where the SI-units telegraph appeared (26.0-26.12 deg): at the
        # canonical natural units the curve must be smooth -- no exact zeros, tiny steps.
        theta, intensity = _sweep(_system(), 26.0, 26.12, 0.01, 0)
        i_par = np.abs(np.asarray(intensity, dtype=float))
        self.assertEqual(int(np.sum(i_par == 0.0)), 0, "no exact-0 sentinel in the natural-units sweep")
        steps = np.abs(np.diff(i_par))
        self.assertLess(
            float(steps.max()), 0.01 * float(i_par.mean()),
            "natural-units Maker curve must be smooth (the SI-units telegraph must not appear)",
        )

    def test_fine_panel_uses_validated_docs_config(self):
        # panel (b)'s fine fringes must come from a configuration that was actually compared
        # point-by-point to live SHAARP.ml AND that physically HAS fine fringes: the docs'
        # canonical thick-slab case (Z-cut quartz 121.2 um + Au 13.9 nm, lambda=0.8 um,
        # validated to 3.6e-9 in tests/test_quartz_au_docs_reference.py).
        fine = _fine_system()
        self.assertEqual(len(fine.layers), 4)  # air / quartz / Au / air
        self.assertAlmostEqual(float(fine.layers[1].thickness_um), 121.2, places=6)
        self.assertAlmostEqual(float(fine.layers[2].thickness_um), 0.0139, places=6)
        self.assertAlmostEqual(float(fine.wavelength_um), 0.8, places=9)


if __name__ == "__main__":
    unittest.main()
