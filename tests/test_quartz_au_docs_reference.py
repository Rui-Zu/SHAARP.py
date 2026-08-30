"""Gated agreement test: the SHAARP.ml docs' canonical Maker example (Z-cut quartz 121.2 um +
Au 13.9 nm, lambda=0.8 um, p-in/p-out) vs the LIVE SHAARP.ml reference exported by
benchmarks/verify_quartz_au_docs_vs_mathematica.py (window 20-30 deg at the paper's 0.1 deg).

This is the THICK-slab configuration with genuine sub-degree fine fringes (~0.9 deg period,
~10 fringes in the window). Live result at export time: max|SHAARP.ml - py| parallel = 3.6e-9
(rel 5.2e-10; the 121.2 um slab carries ~6e3-rad phase factors, so ~1e-9 relative is the
phase-wrap precision floor, NOT a model discrepancy), perpendicular ~1e-32, and BOTH sides
continuous with zero exact-0 dropouts (the continuity lens as a cross-code check).
"""

import json
import unittest
from pathlib import Path

import numpy as np

REF = Path(__file__).resolve().parents[1] / "benchmarks" / "mathematica_reference" / "quartz_au_docs_reference.json"


def _find_mflist(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "MFList":
                return v
            f = _find_mflist(v)
            if f is not None:
                return f
    if isinstance(o, list):
        for x in o:
            f = _find_mflist(x)
            if f is not None:
                return f
    return None


def _cval(x):
    return complex(x["real"], x["imag"]) if isinstance(x, dict) else complex(x)


@unittest.skipUnless(REF.exists(), "live SHAARP.ml quartz+Au docs reference not exported")
class QuartzAuDocsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from benchmarks.quartz_au_docs_case import build_quartz_au_system
        from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

        mf = _find_mflist(json.loads(REF.read_text(encoding="utf-8")))
        cls.theta = [float(_cval(r[0]).real) for r in mf]
        cls.mma_par = np.abs([_cval(r[1]) for r in mf])
        cls.mma_perp = np.abs([_cval(r[2]) for r in mf])
        sw = solve_multilayer_maker_fringes_sweep(
            build_quartz_au_system(), theta_deg=cls.theta, mrassumption=0, mu=1.0, eps0=1.0
        )
        cls.py_par = np.abs(np.asarray(sw.parallel_intensity))
        cls.py_perp = np.abs(np.asarray(sw.perpendicular_intensity))

    def test_window_is_the_paper_rigor_grid(self):
        self.assertEqual(len(self.theta), 101)
        steps = np.diff(np.asarray(self.theta))
        self.assertTrue(np.allclose(steps, 0.1, atol=1e-9), "0.1 deg sampling expected")

    def test_parallel_channel_matches_live_mathematica(self):
        err = float(np.max(np.abs(self.mma_par - self.py_par)))
        rel = err / max(float(np.max(self.mma_par)), 1e-300)
        self.assertLess(err, 1e-7, f"abs err {err:.3e} beyond the phase-wrap floor margin")
        self.assertLess(rel, 1e-8, f"rel err {rel:.3e} beyond the phase-wrap floor margin")

    def test_perpendicular_channel_matches_live_mathematica(self):
        self.assertLess(float(np.max(np.abs(self.mma_perp - self.py_perp))), 1e-12)

    def test_both_sides_are_continuous_with_genuine_fine_fringes(self):
        # the continuity lens as a CROSS-CODE check: zero exact-0 dropouts on either side, and
        # the window really contains the thick-slab fine fringes (several interior maxima).
        self.assertEqual(int(np.sum(self.mma_par == 0.0)), 0)
        self.assertEqual(int(np.sum(self.py_par == 0.0)), 0)
        y = self.py_par
        interior_maxima = int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])))
        self.assertGreaterEqual(interior_maxima, 8, "expected ~10 fine fringes in the 20-30 deg window")


if __name__ == "__main__":
    unittest.main()
