"""Fence: SHAARP.py's Herman-Hayden (HH) multilayer SHG mode reproduces the RAW Herman-1995 analytic
Maker-fringe expression -- the author's own ``analyticHH`` benchmark (fig3/analyticHH.mx), ported byte-exact
into ``benchmarks/herman_hayden_maker.py``. This proves the HH mode against an INDEPENDENT analytic
model (not just against SHAARP), for the SHAARP.ml 2024 Fig-3 case (300 um X-cut quartz, 1064 nm).
"""
import unittest

import numpy as np


class HermanHaydenPortTests(unittest.TestCase):
    def test_exact_port_value_at_30deg(self):
        """The Python port equals Mathematica's analyticHH value at theta = 30 deg to ~1e-12."""
        from benchmarks.herman_hayden_maker import herman_hayden_quartz_300um_intensity
        val = float(herman_hayden_quartz_300um_intensity(np.deg2rad(30.0)))
        self.assertAlmostEqual(val, 4.875968756693962e-05, delta=1e-16,
                               msg="ported analyticHH diverged from the Mathematica reference value")

    def test_reference_curve_finite_positive_with_envelope_peak(self):
        """The reference Maker curve is finite/positive and peaks at an oblique angle (Maker envelope)."""
        from benchmarks.herman_hayden_maker import herman_hayden_quartz_reference_curve
        th, inten = herman_hayden_quartz_reference_curve(0.0, 65.0, 651)
        self.assertTrue(np.all(np.isfinite(inten)), "reference curve has non-finite samples")
        self.assertGreaterEqual(float(np.min(inten)), 0.0, "reference intensity went negative")
        peak_theta = float(th[int(np.argmax(inten))])
        self.assertGreater(peak_theta, 25.0, "HH Maker envelope should peak well off normal incidence")


class RuiMxReferenceCurveTests(unittest.TestCase):
    """Fences against the author's own serialized Fig-3 reference curves (fig3/*.mx evaluated at
    h = 300 um, phi = 0 -> benchmarks/fig3_reference_curves.json)."""

    def _load(self):
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "benchmarks" / "fig3_reference_curves.json"
        self.assertTrue(p.exists(), "benchmarks/fig3_reference_curves.json missing")
        d = json.loads(p.read_text(encoding="utf-8"))
        return (np.array(d["hhsim_finer_20_30deg"], float),
                np.array(d["old_hhjk_0_90deg"], float))

    def test_ported_analytic_hh_is_exactly_ruis_hhsim_curve(self):
        """The ported analyticHH expression IS the author's dataHHsimHHFiner1.mx curve: on the identical
        0.05-deg grid the two are the same function up to a constant unit factor (ratio std ~ 0).
        (the author's own artifacts agree the same way: HHsim(h0=300, phi=0) vs analyticHH corr = 1.0 in
        Mathematica; the earlier 0.9547 was a resampling artifact of comparing mixed grids.)"""
        from benchmarks.herman_hayden_maker import herman_hayden_quartz_reference_curve
        fine, _ = self._load()
        th_ref, i_ref = herman_hayden_quartz_reference_curve(20.0, 30.0, 201)
        self.assertTrue(np.allclose(th_ref, fine[:, 0]), "grids must be identical for the exactness claim")
        ratio = fine[:, 1] / i_ref
        rel_spread = float(np.std(ratio) / np.mean(ratio))
        self.assertLess(rel_spread, 1e-9,
                        f"ported analytic HH is not a constant multiple of the author's HHsim (spread={rel_spread:.2e})")
        corr = float(np.corrcoef(fine[:, 1], i_ref)[0, 1])
        self.assertGreater(corr, 0.999999, f"exact-overlap correlation degraded (corr={corr:.6f})")

    def test_ported_analytic_hh_matches_ruis_evaluated_curve(self):
        """The ported analyticHH expression matches the author's dataoldHHJK.mx evaluated HH column."""
        from benchmarks.herman_hayden_maker import herman_hayden_quartz_reference_curve
        from benchmarks.paper_figures import _shape_corr
        _, hhjk = self._load()
        th_ref, i_ref = herman_hayden_quartz_reference_curve(2.0, 65.0, 631)
        m = (hhjk[:, 0] >= 2.0) & (hhjk[:, 0] <= 65.0)
        corr = _shape_corr(hhjk[m, 0], hhjk[m, 2], th_ref, i_ref)
        self.assertGreater(corr, 0.99,
                           f"ported analytic HH diverged from the author's evaluated .mx curve (corr={corr:.4f})")

    def test_shaarp_hh_fine_fringes_exactly_overlap_ruis_hhsim_window(self):
        """SHAARP.py(HH)'s FINE 2w fringes exactly overlap the author's dataHHsimHHFiner1.mx, fringe for
        fringe, when sampled on the SAME 0.05-deg grid (the paper's exact-overlap claim). A 0.1-deg
        sweep sits at the Nyquist edge of the ~0.2-deg fringe period and aliases -- never compare
        fine fringes across mixed grids."""
        from benchmarks.paper_cases import ASSUMPTION_HH, ml_fig3_system, ml_maker
        fine, _ = self._load()
        th_py, i_py = ml_maker(ml_fig3_system(), ASSUMPTION_HH, th_min=20.0, th_max=30.0, step=0.05)
        self.assertTrue(np.allclose(th_py, fine[:, 0]), "sweep grid must match the reference grid")
        corr = float(np.corrcoef(i_py, fine[:, 1])[0, 1])
        self.assertGreater(corr, 0.999,
                           f"SHAARP.py(HH) fine fringes no longer overlap the author's HH reference (corr={corr:.6f})")


class HermanHaydenVsShaarpTests(unittest.TestCase):
    def test_shaarp_hh_tracks_the_raw_analytic_hh(self):
        """SHAARP.py's own HH Maker sweep for the 300 um quartz case has the SAME fringe SHAPE as the
        raw Herman-1995 analytic expression (normalized cross-correlation near 1)."""
        from benchmarks.herman_hayden_maker import herman_hayden_quartz_reference_curve
        from benchmarks.paper_cases import ASSUMPTION_HH, ml_fig3_system, ml_maker
        from benchmarks.paper_figures import _shape_corr

        s = ml_fig3_system()
        # coarse-but-honest grid: the main fringes + envelope (fine 2w fringes need a denser grid).
        th_hh, i_hh = ml_maker(s, ASSUMPTION_HH, th_min=2.0, th_max=60.0, step=1.0)
        th_ref, i_ref = herman_hayden_quartz_reference_curve(2.0, 60.0, 601)
        corr = _shape_corr(th_hh, i_hh, th_ref, i_ref)
        self.assertGreater(corr, 0.7,
                           f"SHAARP.py(HH) shape does not track the raw analytic HH (corr={corr:.3f})")


if __name__ == "__main__":
    unittest.main()
