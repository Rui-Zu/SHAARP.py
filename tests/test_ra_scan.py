"""SAMPLE ROTATION — the ML rotational-anisotropy (azimuth) sweep, driven by the
Polarimetry toggle rather than by a functionality (F47-3a's "RA Scan" mode was a port invention;
the original .ml GUI carries `samplerotationcontrol` — "Rotate Sample"/"Fix Sample" — inside
Polarimetry Settings, forcing RotatePolarizer = RotateAnalyzer = False, with a STEP SIZE).

The physics is the already-Mathematica-validated ``run_sample_rotation`` path (SampleRotate azimuth
sweep, ~4.7e-15 in test_jkhh_samplerotate_agreement.py), so these fences pin the GUI WIRING:
the toggle produces a sample_rotation result and pins the two polarization combos; the step size
drives the point count; the direction mirrors the curve; the FIXED polarizer/analyzer/ellipticity
reach every azimuth point (the dead-control fix — pre- the sweep was handed phi=psi=0); and
the plotted result equals a fresh ``run_sample_rotation`` on the exact system the GUI built.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets
    HAVE_QT = True
except Exception:  # pragma: no cover
    HAVE_QT = False

import numpy as np


def _ml_page(win):
    return win.findChild(QtWidgets.QTabWidget).widget(1)


def _func_combo(page):
    return next(c for c in page.findChildren(QtWidgets.QComboBox)
                if c.toolTip().startswith("Choose what to calculate"))


def _tipped(page, cls, key):
    from shaarp.desktop_app import TOOLTIPS
    return [w for w in page.findChildren(cls) if w.toolTip() == TOOLTIPS[key]]


def _sample_widgets(page):
    """(mode combo, step spin, direction combo) — all three carry TOOLTIPS['sample_rotation']."""
    combos = _tipped(page, QtWidgets.QComboBox, "sample_rotation")
    spins = _tipped(page, QtWidgets.QDoubleSpinBox, "sample_rotation")
    mode = next(c for c in combos if c.findText("Rotate Sample") >= 0)
    direction = next(c for c in combos if c.findText("CW (clockwise)") >= 0)
    return mode, spins[0], direction


def _films(page):
    from . import gui_harness as gh
    return gh.ml_film_labels(page)


# Non-centrosymmetric films with strong RA SHG (verified: LiNbO3 x-cut reflected≈37 /
# transmitted≈66; KTP x-cut reflected≈0.45). Both are also azimuth-DEPENDENT in eps, so their RA
# curve is not mirror-trivial — which is what the direction fence needs.
GOOD_FILM = "LiNbO3 x-cut · 1550 nm"
GOOD_FILM2 = "KTP x-cut · 1550 nm"
CHANNELS = ("reflected_parallel_intensity", "reflected_perpendicular_intensity",
            "transmitted_parallel_intensity", "transmitted_perpendicular_intensity")


def _good(page, name):
    if name not in _films(page):
        raise unittest.SkipTest(f"{name!r} not in the film rows")
    return name


def _chan(res, key="reflected_parallel_intensity"):
    return np.real(np.asarray(res.numeric[key])).astype(float)


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class SampleRotationSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from shaarp.desktop_app import build_main_window

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = build_main_window()
        cls.win._gui_smoke_errors = []

    def _run(self, film, *, step=20.0, ccw=True, phi=None, psi=None, ellipticity=None):
        from . import gui_harness as gh

        ml = _ml_page(self.win)
        self.win.findChild(QtWidgets.QTabWidget).setCurrentIndex(1)
        gh.select_ml_film(ml, self.app, film)
        _func_combo(ml).setCurrentText("SHG Simulation")
        mode, step_spin, direction = _sample_widgets(ml)
        # no pinning -- these sweep tests exercise the FIXED polarizer/analyzer combination,
        # so set it explicitly (the default polarizer state is Rotate Polarizer).
        _tipped(ml, QtWidgets.QComboBox, "polarizer")[0].setCurrentText("Fix Polarizer")
        _tipped(ml, QtWidgets.QComboBox, "analyzer")[0].setCurrentText("Fix Analyzer")
        mode.setCurrentText("Rotate Sample")
        step_spin.setValue(float(step))
        direction.setCurrentText("CCW (counter-clockwise)" if ccw else "CW (clockwise)")
        if phi is not None:
            _tipped(ml, QtWidgets.QDoubleSpinBox, "polarizer")[0].setValue(float(phi))
        if psi is not None:
            _tipped(ml, QtWidgets.QDoubleSpinBox, "analyzer")[0].setValue(float(psi))
        if ellipticity is not None:
            _tipped(ml, QtWidgets.QDoubleSpinBox, "ellipticity")[0].setValue(float(ellipticity))
        self.app.processEvents()
        del self.win._gui_smoke_errors[:]
        gh.click_update(ml, self.app)
        self.assertFalse(self.win._gui_smoke_errors,
                         f"sample-rotation Update errored: {self.win._gui_smoke_errors[:2]}")
        return ml._last_ra_result()

    def test_toggle_produces_a_sample_rotation_result(self):
        res = self._run(_good(_ml_page(self.win), GOOD_FILM), step=10.0)
        self.assertIsNotNone(res, "sample-rotation result hook empty")
        self.assertEqual(res.kind, "sample_rotation")
        az = np.asarray(res.stages["sample_rotation"]["azimuth_deg_user"], dtype=float)
        self.assertEqual(len(az), 37, "step size not honored (10 deg -> 0..360 inclusive = 37)")
        self.assertAlmostEqual(az[0], 0.0, places=6)
        self.assertAlmostEqual(az[-1], 360.0, places=6)
        peak = 0.0
        for key in CHANNELS:
            ch = _chan(res, key)
            self.assertTrue(np.all(np.isfinite(ch)), f"non-finite intensity in {key}")
            peak = max(peak, float(np.max(np.abs(ch))))
        self.assertGreater(peak, 0.0, "identically-zero SHG in every channel")

    def test_step_size_drives_the_point_count(self):
        film = _good(_ml_page(self.win), GOOD_FILM)
        for step, want in ((10.0, 37), (20.0, 19), (30.0, 13)):
            res = self._run(film, step=step)
            got = len(np.asarray(res.stages["sample_rotation"]["azimuth_deg_user"]))
            self.assertEqual(got, want, f"step {step} deg should give {want} points, got {got}")

    def test_direction_mirrors_the_curve(self):
        """The solver's +azimuth reads CW from the beam side, so CCW is the negated grid.
        The two sweeps must be each other's reversal on the closed 0..360 grid."""
        film = _good(_ml_page(self.win), GOOD_FILM)
        ccw = self._run(film, step=20.0, ccw=True)
        cw = self._run(film, step=20.0, ccw=False)
        self.assertEqual(ccw.stages["sample_rotation"]["direction"], "CCW")
        self.assertEqual(cw.stages["sample_rotation"]["direction"], "CW")
        np.testing.assert_allclose(
            np.asarray(ccw.stages["sample_rotation"]["azimuth_deg_user"]),
            np.asarray(cw.stages["sample_rotation"]["azimuth_deg_user"]), atol=1e-12,
            err_msg="the PLOTTED azimuth axis must stay 0..360 in both directions")
        np.testing.assert_allclose(
            np.asarray(ccw.stages["sample_rotation"]["azimuth_deg_solver"]),
            -np.asarray(cw.stages["sample_rotation"]["azimuth_deg_solver"]), atol=1e-12,
            err_msg="CCW must hand the solver the negated grid")
        a, b = _chan(ccw), _chan(cw)
        self.assertFalse(np.allclose(a, b), "direction did not change the curve at all")
        np.testing.assert_allclose(a, b[::-1], rtol=1e-9, atol=1e-12,
                                   err_msg="CW is not the reversal of CCW about psi_s = 0")

    def test_fixed_polarizer_analyzer_ellipticity_reach_every_point(self):
        """THE DEAD-CONTROL FENCE. The sweep was once handed `sketch_sys`, whose polarimetry
        is phi_deg=0/psi_deg=0 (layer_stack.build_system_from_stack), so these three panel controls
        were inert — unlike the original, which passes its fixed phi and psi into every point."""
        film = _good(_ml_page(self.win), GOOD_FILM)
        base = _chan(self._run(film, step=30.0, phi=0.0, psi=0.0, ellipticity=0.0))
        moved_phi = _chan(self._run(film, step=30.0, phi=45.0, psi=0.0, ellipticity=0.0))
        moved_psi = _chan(self._run(film, step=30.0, phi=45.0, psi=30.0, ellipticity=0.0))
        moved_del = _chan(self._run(film, step=30.0, phi=45.0, psi=30.0, ellipticity=25.0))
        self.assertFalse(np.allclose(base, moved_phi), "fixed polarizer phi never reached the sweep")
        self.assertFalse(np.allclose(moved_phi, moved_psi), "fixed analyzer psi never reached the sweep")
        self.assertFalse(np.allclose(moved_psi, moved_del), "ellipticity never reached the sweep")
        # the stage record must agree with the panel (provenance, not just causality)
        st = _ml_page(self.win)._last_ra_result().stages["sample_rotation"]
        self.assertAlmostEqual(st["fixed_phi_deg"], 45.0, places=6)
        self.assertAlmostEqual(st["analyzer_psi_deg"], 30.0, places=6)
        self.assertAlmostEqual(st["ellipticity_deg"], 25.0, places=6)

    def test_material_drives_the_curve(self):
        ml = _ml_page(self.win)
        a = _chan(self._run(_good(ml, GOOD_FILM), step=30.0))
        b = _chan(self._run(_good(ml, GOOD_FILM2), step=30.0))
        self.assertFalse(np.allclose(a, b),
                         "different films gave identical curves (material not fed through)")

    def test_plotted_equals_fresh_run_sample_rotation(self):
        from shaarp.api import run_sample_rotation

        ml = _ml_page(self.win)
        res = self._run(_good(ml, GOOD_FILM), step=20.0, phi=15.0, psi=40.0)
        sysm, grid = ml._last_ra_system(), ml._last_ra_grid()
        self.assertIsNotNone(sysm, "system hook empty")
        self.assertIsNotNone(grid, "solver-grid hook empty")
        fresh = run_sample_rotation(sysm, np.asarray(grid, dtype=float))
        for key in CHANNELS:
            np.testing.assert_allclose(_chan(res, key), _chan(fresh, key), rtol=1e-9, atol=1e-12,
                                       err_msg=f"GUI {key} != fresh run_sample_rotation")


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class SampleRotationControlContract(unittest.TestCase):
    """Polarizer, analyzer, and sample each carry an INDEPENDENT rotate/fix
    choice and any of the 8 combinations is legal — the pinning (Rotate Sample forcing both
    polarization controls to fixed) is gone. One function (`_sync_pol_enabled`) owns every
    polarimetry widget's enabled state."""

    @classmethod
    def setUpClass(cls):
        from shaarp.desktop_app import build_main_window

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = build_main_window()

    def _polarization_combos(self, page):
        return (_tipped(page, QtWidgets.QComboBox, "polarizer")[0],
                _tipped(page, QtWidgets.QComboBox, "analyzer")[0])

    def setUp(self):
        ml = _ml_page(self.win)
        _func_combo(ml).setCurrentText("SHG Simulation")
        mode, _s, _d = _sample_widgets(ml)
        pol, ana = self._polarization_combos(ml)
        mode.setCurrentText("Fix Sample")
        pol.setCurrentText("Rotate Polarizer")
        ana.setCurrentText("Fix Analyzer")
        self.app.processEvents()

    def test_default_is_fix_sample_and_leaves_the_panel_alone(self):
        ml = _ml_page(self.win)
        mode, step, direction = _sample_widgets(ml)
        pol, ana = self._polarization_combos(ml)
        self.assertEqual(mode.currentText(), "Fix Sample", "the original's default is False")
        self.assertFalse(step.isEnabled(), "step size is meaningless while the sample is fixed")
        self.assertFalse(direction.isEnabled())
        self.assertTrue(pol.isEnabled() and ana.isEnabled())

    def test_no_pinning_all_eight_combinations_stay_settable(self):
        import itertools

        ml = _ml_page(self.win)
        mode, step, direction = _sample_widgets(ml)
        pol, ana = self._polarization_combos(ml)
        for p_, a_, s_ in itertools.product(
                ("Rotate Polarizer", "Fix Polarizer"),
                ("Rotate Analyzer", "Fix Analyzer"),
                ("Rotate Sample", "Fix Sample")):
            pol.setCurrentText(p_)
            ana.setCurrentText(a_)
            mode.setCurrentText(s_)
            self.app.processEvents()
            # nothing overrides anything: the choice sticks and every combo stays live
            self.assertEqual((pol.currentText(), ana.currentText(), mode.currentText()),
                             (p_, a_, s_), "a combination was silently overridden")
            self.assertTrue(pol.isEnabled() and ana.isEnabled() and mode.isEnabled())
            rotating = s_.startswith("Rotate")
            self.assertEqual(step.isEnabled(), rotating)
            self.assertEqual(direction.isEnabled(), rotating)
            st = ml._sample_rotation_state()
            self.assertEqual(st["on"], rotating)
            self.assertEqual(st["rotate_polarizer"], p_.startswith("Rotate"))
            self.assertEqual(st["rotate_analyzer"], a_.startswith("Rotate Analyzer"))

    def test_offset_enabled_only_when_both_rotate(self):
        """The original's gate: `Dynamic[If[RotateAnalyzer, If[RotatePolarizer, offset, ""]]]` —
        the offset exists only when BOTH the polarizer and the analyzer rotate (the fidelity audit
        finding,; the pre-F70 port consulted the analyzer alone)."""
        ml = _ml_page(self.win)
        pol, ana = self._polarization_combos(ml)
        offset = next(s for s in ml.findChildren(QtWidgets.QDoubleSpinBox)
                      if "offset" in s.toolTip().lower())
        for p_, a_, want in (("Rotate Polarizer", "Rotate Analyzer", True),
                             ("Fix Polarizer", "Rotate Analyzer", False),
                             ("Rotate Polarizer", "Fix Analyzer", False),
                             ("Fix Polarizer", "Fix Analyzer", False)):
            pol.setCurrentText(p_)
            ana.setCurrentText(a_)
            self.app.processEvents()
            self.assertEqual(offset.isEnabled(), want, f"offset gate wrong for {p_} / {a_}")

    def test_fixed_fields_follow_their_combo(self):
        ml = _ml_page(self.win)
        pol, ana = self._polarization_combos(ml)
        phi = _tipped(ml, QtWidgets.QDoubleSpinBox, "polarizer")[0]
        psi = _tipped(ml, QtWidgets.QDoubleSpinBox, "analyzer")[0]
        pol.setCurrentText("Rotate Polarizer")
        ana.setCurrentText("Rotate Analyzer")
        self.app.processEvents()
        self.assertFalse(phi.isEnabled(), "fixed φ is dead while the polarizer rotates")
        self.assertFalse(psi.isEnabled(), "fixed ψ is dead while the analyzer rotates")
        pol.setCurrentText("Fix Polarizer")
        ana.setCurrentText("Fix Analyzer")
        self.app.processEvents()
        self.assertTrue(phi.isEnabled() and psi.isEnabled())

    def test_relevance_gating(self):
        """Sample rotation is an SHG-Simulation / Partial-Analytical measurement: each sweep
        mode gets its OWN scan section, and Maker Fringes keeps Polarimetry OPEN (its input φ and
        detection ψ live there) with the rotate/fix selectors greyed."""
        ml = _ml_page(self.win)
        mode, _step, _dir = _sample_widgets(ml)
        pol, ana = self._polarization_combos(ml)
        phi = _tipped(ml, QtWidgets.QDoubleSpinBox, "polarizer")[0]
        psi = _tipped(ml, QtWidgets.QDoubleSpinBox, "analyzer")[0]
        for func, want in (("SHG Simulation", True), ("Partial Analytical Expressions", True),
                           ("Maker Fringes", False), ("Fresnel Coefficients", False)):
            _func_combo(ml).setCurrentText(func)
            self.app.processEvents()
            self.assertEqual(mode.isEnabled(), want, f"sample rotation enabled-state wrong in {func}")
            self.assertEqual(pol.isEnabled(), want, f"polarizer rotate/fix wrong in {func}")
            self.assertEqual(ana.isEnabled(), want, f"analyzer rotate/fix wrong in {func}")
        _func_combo(ml).setCurrentText("Maker Fringes")
        self.app.processEvents()
        self.assertTrue(phi.isEnabled(), "Maker Fringes input polarization φ must be live (F70)")
        self.assertTrue(psi.isEnabled(), "Maker Fringes detection polarization ψ must be live (F70)")
        groups = {g.title(): g for g in ml.findChildren(QtWidgets.QGroupBox)}
        g_pol = next(g for t, g in groups.items() if t.startswith("Polarimetry Settings"))
        g_mk = next(g for t, g in groups.items() if t.startswith("Maker Fringes Scan Range"))
        g_fr = next(g for t, g in groups.items() if t.startswith("Fresnel Coefficients Scan Range"))
        self.assertTrue(g_pol.isChecked(), "Polarimetry stays open in Maker Fringes (φ/ψ live there)")
        self.assertTrue(g_mk.isChecked())
        self.assertFalse(g_fr.isChecked(), "the Fresnel section is its own, separately-toggled group")
        _func_combo(ml).setCurrentText("Fresnel Coefficients")
        self.app.processEvents()
        self.assertFalse(g_mk.isChecked())
        self.assertTrue(g_fr.isChecked())
        self.assertFalse(g_pol.isChecked(), "Fresnel is linear s/p — polarimetry is not used")
        _func_combo(ml).setCurrentText("SHG Simulation")
        self.app.processEvents()


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
