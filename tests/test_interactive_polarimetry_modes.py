"""Interactive GUI exposure of the session's validated polarimetry + d-extraction.

The ipywidgets control panel (shaarp/interactive.py) gained three modes backed by PURE, testable
helpers so the logic is verifiable without rendering widgets:
  * "SI polarimetry" -> compute_polarimetry_result("si_polarimetry") -> the VALIDATED SHAARP.si
    full-analytical reflected polarimetry closed form (run_si_full_analytical workflow="polarimetry")
  * "ML polarimetry" -> compute_polarimetry_result("ml_polarimetry") -> the VALIDATED SHAARP.ml
    partial-analytical film polarimetry (run_ml_partial_analytical workflow="polarimetry")
  * "d-extraction (demo)" -> compute_dextraction_demo() -> simulate a scan and recover d via the
    public extract_si_d_voigt (the inverse capability, end to end)

These route the VALIDATED workflows, not the symbolic scaffolds the older si_analytical /
ml_partial_analytical modes use.
"""

import unittest

import numpy as np

from shaarp.interactive import (
    build_dextraction_figure,
    build_polarimetry_figure,
    compute_dextraction_demo,
    compute_polarimetry_curve,
    compute_polarimetry_result,
)


class PolarimetryHelperTests(unittest.TestCase):
    def test_si_polarimetry_routes_validated_workflow(self):
        res = compute_polarimetry_result("si_polarimetry", point_group="-43m", theta_deg=20.0)
        self.assertEqual(res.kind, "si_full_analytical_polarimetry")
        self.assertEqual(res.stages["workflow"], "polarimetry")

    def test_ml_polarimetry_routes_validated_workflow(self):
        res = compute_polarimetry_result("ml_polarimetry", point_group="-43m", theta_deg=20.0)
        self.assertIn("polarimetry", res.kind)
        self.assertEqual(res.stages["workflow"], "polarimetry")

    def test_unknown_polarimetry_mode_raises(self):
        with self.assertRaises(ValueError):
            compute_polarimetry_result("bogus", point_group="1", theta_deg=10.0)


class DExtractionDemoTests(unittest.TestCase):
    def test_demo_recovers_known_tensor(self):
        out = compute_dextraction_demo()
        self.assertEqual(out["kind"], "d_extraction_demo")
        self.assertTrue(out["identifiable"], "demo scan should be identifiable (rank 6/6)")
        self.assertLess(out["relative_residual"], 1e-9, "consistent-units demo should fit to machine precision")
        self.assertLess(out["max_abs_error_vs_true"], 1e-7, "demo d not recovered")
        # the reported recovered/true dicts are aligned and complete
        self.assertEqual(set(out["recovered_d"]), set(out["true_d"]))
        self.assertEqual(len(out["recovered_d"]), 6)

    def test_figure_builds_with_recovered_and_true_bars(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = build_dextraction_figure()
        ax = fig.axes[0]
        # 2 bar groups (true + recovered) x 6 components = 12 bars
        self.assertEqual(len(ax.patches), 12)
        plt.close(fig)


class PolarimetryCurveTests(unittest.TestCase):
    def test_curve_is_a_real_phi_varying_curve(self):
        c = compute_polarimetry_curve(theta_deg=45.0, n_phi=181)
        self.assertEqual(c["phi_deg"].shape, (181,))
        self.assertEqual(c["intensity_s"].shape, (181,))
        self.assertEqual(c["intensity_p"].shape, (181,))
        self.assertTrue(np.all(np.isfinite(c["intensity_p"])) and np.all(np.isfinite(c["intensity_s"])))
        # a genuine polarimetry curve varies with phi (not a flat/selection-rule-zero artifact)
        self.assertGreater(float(np.ptp(c["intensity_p"])), 0.0)
        self.assertGreater(float(np.ptp(c["intensity_s"])), 0.0)

    def test_curve_obeys_c3v_threefold_azimuth_symmetry(self):
        # GaAs(111) -43m is C3v: rotating the sample 120 deg leaves the pattern unchanged, 60 does not.
        i0 = compute_polarimetry_curve(theta_deg=45.0, sample_azimuth_deg=0.0)["intensity_p"]
        i120 = compute_polarimetry_curve(theta_deg=45.0, sample_azimuth_deg=120.0)["intensity_p"]
        i60 = compute_polarimetry_curve(theta_deg=45.0, sample_azimuth_deg=60.0)["intensity_p"]
        scale = max(float(np.max(np.abs(i0))), 1e-30)
        self.assertLess(float(np.max(np.abs(i0 - i120))) / scale, 1e-9, "120 deg must coincide (3-fold)")
        self.assertGreater(float(np.max(np.abs(i0 - i60))) / scale, 1e-3, "60 deg must differ")

    def test_figure_builds_with_two_curves(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = build_polarimetry_figure(theta_deg=45.0)
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.axes[0].lines), 2)  # I_s and I_p
        plt.close(fig)


class PanelWiringTests(unittest.TestCase):
    def _find_mode_dropdown(self, panel):
        import ipywidgets as widgets
        stack = [panel]
        while stack:
            w = stack.pop()
            if isinstance(w, widgets.Dropdown) and w.description == "Mode":
                return w
            stack.extend(list(getattr(w, "children", []) or []))
        return None

    def test_panel_exposes_the_new_modes(self):
        try:
            import ipywidgets  # noqa: F401
        except ImportError:
            self.skipTest("ipywidgets not installed")
        from shaarp.interactive import make_interactive_session

        panel = make_interactive_session()
        dd = self._find_mode_dropdown(panel)
        self.assertIsNotNone(dd, "Mode dropdown not found in the panel")
        values = {v for (_label, v) in dd.options}
        for mode_id in ("si_polarimetry", "ml_polarimetry", "si_polarimetry_plot", "d_extraction", "d_extraction_plot"):
            self.assertIn(mode_id, values, f"GUI panel must expose the {mode_id!r} mode")


if __name__ == "__main__":
    unittest.main()
