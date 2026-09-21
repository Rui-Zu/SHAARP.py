"""The wavelength-sweep controls, driven through the real Update path on both tabs.

WHY THIS EXISTS. The coverage registry demands that every interactive widget be driven by a test
that shows it CAUSES something; a control that exists but changes no output is the defect class
that registry was built to catch. So each of the four new widgets is driven here and checked
against the computed output rather than against its own state.

Three behaviours are specific to this feature and are easy to regress:

  * Turning the sweep on PINS the geometry. A spectrum needs one scalar per wavelength, so the
    polarizer, analyzer and sample rotation are forced to their fixed settings. This mirrors what
    the original already does when the sample rotates, and it must release when the sweep is
    turned off.
  * A material whose data does not disperse still computes, and WARNS. Every single-interface
    palette case is in that position today, so the note is the only thing standing between a user
    and a flat line they might read as physics.
  * A wavelength where eps(2w) is unphysical REFUSES, with a message that names the material and
    the range that is usable. That message must reach the user intact rather than being wrapped in
    the generic "check your tensors" hint.
"""
from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from PySide6 import QtWidgets  # noqa: E402

SI, ML = 0, 1
DISPERSIVE_ML = "    Quartz z-cut · 800 nm"
POLED_ML = "    KTP x-cut · 1550 nm"
CONSTANT_SI = "GaAs (111)"


class SpectralSweepCausality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _page(self, win, index):
        return win.findChild(QtWidgets.QTabWidget).widget(index)

    def _controls(self, page):
        from shaarp.desktop_app import TOOLTIPS

        check = next(c for c in page.findChildren(QtWidgets.QCheckBox)
                     if "sweep the wavelength" in c.text())
        spins = [s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                 if s.toolTip() == TOOLTIPS["lambda_range"]]
        self.assertEqual(len(spins), 3, "expected lambda min/max/step")
        return check, spins

    def _update(self, page):
        button = next(b for b in page.findChildren(QtWidgets.QPushButton)
                      if b.text().strip().lower().startswith("update"))
        button.click()
        for _ in range(8):
            self.app.processEvents()

    def _select(self, page, text):
        combo = next(c for c in page.findChildren(QtWidgets.QComboBox) if c.findText(text) >= 0)
        combo.setCurrentIndex(combo.findText(text))
        return combo

    def _canvas(self, page, tag):
        return page.findChild(QtWidgets.QWidget, f"spectrum_canvas_{tag}")

    def _note(self, page, tag):
        return page.findChild(QtWidgets.QLabel, f"wl_note_{tag}")

    # -- the spins reach the result -------------------------------------------------------

    def test_the_lambda_spins_set_the_computed_grid_on_both_tabs(self):
        from shaarp.desktop_app import build_main_window

        for index, tag, case in ((SI, "si", CONSTANT_SI), (ML, "ml", DISPERSIVE_ML)):
            win = build_main_window()
            page = self._page(win, index)
            self._select(page, case)
            self._select(page, "SHG Simulation")
            check, spins = self._controls(page)
            check.setChecked(True)
            for spin, value in zip(spins, (0.7, 1.1, 0.1)):
                spin.setValue(value)
            self._update(page)
            axes = self._canvas(page, tag).figure.axes
            self.assertTrue(axes, f"{tag}: no spectrum drawn")
            xdata = axes[0].lines[0].get_xdata()
            np.testing.assert_allclose(xdata, [0.7, 0.8, 0.9, 1.0, 1.1], atol=1e-9,
                                       err_msg=f"{tag}: the spins did not set the grid")

            # ... and changing a spin changes the grid, which is what makes it a live control
            spins[1].setValue(0.9)
            self._update(page)
            np.testing.assert_allclose(self._canvas(page, tag).figure.axes[0].lines[0].get_xdata(),
                                       [0.7, 0.8, 0.9], atol=1e-9)

    def test_turning_the_sweep_on_switches_the_output_to_the_spectrum_tab(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        page = self._page(win, ML)
        self._select(page, DISPERSIVE_ML)
        self._select(page, "SHG Simulation")
        output = next(t for t in page.findChildren(QtWidgets.QTabWidget))
        check, spins = self._controls(page)

        self._update(page)
        self.assertNotEqual(output.tabText(output.currentIndex()), "Spectrum",
                            "the Spectrum tab must not be shown for a single-wavelength run")
        check.setChecked(True)
        for spin, value in zip(spins, (0.7, 1.1, 0.1)):
            spin.setValue(value)
        self._update(page)
        self.assertEqual(output.tabText(output.currentIndex()), "Spectrum")

    # -- the switch pins the geometry ------------------------------------------------------

    def test_the_switch_pins_the_rotating_geometry_and_releases_it(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        page = self._page(win, ML)
        self._select(page, "SHG Simulation")
        polarizer = next(c for c in page.findChildren(QtWidgets.QComboBox)
                         if c.findText("Rotate Polarizer") >= 0)
        analyzer = next(c for c in page.findChildren(QtWidgets.QComboBox)
                        if c.findText("Rotate Analyzer") >= 0)
        sample = next(c for c in page.findChildren(QtWidgets.QComboBox)
                      if c.findText("Rotate Sample") >= 0)
        for combo, rotating in ((polarizer, "Rotate Polarizer"), (analyzer, "Rotate Analyzer"),
                                (sample, "Rotate Sample")):
            combo.setCurrentText(rotating)
        check, _ = self._controls(page)
        before = {combo: combo.currentText() for combo in (polarizer, analyzer, sample)}

        check.setChecked(True)
        self.assertTrue(polarizer.currentText().startswith("Fix"))
        self.assertTrue(analyzer.currentText().startswith("Fix"))
        self.assertTrue(sample.currentText().startswith("Fix"))
        for combo in (polarizer, analyzer, sample):
            self.assertFalse(combo.isEnabled(), "a pinned control must not stay editable")

        check.setChecked(False)
        for combo in (polarizer, analyzer, sample):
            self.assertTrue(combo.isEnabled(), "the pin must release when the sweep is turned off")
            # ...and hand back what the user had: they were left on "Fix", so the next ordinary
            # Update drew one fixed-polarizer curve instead of the lobes set up before the sweep
            self.assertEqual(combo.currentText(), before[combo])

    def test_the_rotate_modes_come_back_on_the_si_tab_too(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        page = self._page(win, SI)
        self._select(page, "SHG Simulation")
        polarizer = next(c for c in page.findChildren(QtWidgets.QComboBox)
                         if c.findText("Rotate Polarizer") >= 0)
        polarizer.setCurrentText("Rotate Polarizer")
        check, _ = self._controls(page)
        check.setChecked(True)
        self.assertTrue(polarizer.currentText().startswith("Fix"))
        check.setChecked(False)
        self.assertEqual(polarizer.currentText(), "Rotate Polarizer")

    def test_the_switch_is_dead_in_the_analytical_modes_and_says_so(self):
        """The analytical modes return an expression, not a curve, and each wavelength would give a
        different one. The compute path has no spectrum for them, so leaving the switch live there
        accepted a tick and silently ignored it -- the dead-control class the coverage registry
        exists to catch."""
        from shaarp.desktop_app import build_main_window

        cases = ((SI, "si", ("Partial Analytical Expressions", "Full Analytical Expressions")),
                 (ML, "ml", ("Partial Analytical Expressions",)))
        for index, tag, analytical_modes in cases:
            win = build_main_window()
            page = self._page(win, index)
            check, _ = self._controls(page)
            note = page.findChild(QtWidgets.QLabel, f"spec_note_{tag}")
            self.assertIsNotNone(note, f"spec_note_{tag} missing")
            check.setChecked(True)

            for mode in analytical_modes:
                self._select(page, mode)
                self.assertFalse(check.isEnabled(), f"{tag}/{mode}: switch must grey out")
                self.assertFalse(page._spectral_state()["on"],
                                 f"{tag}/{mode}: a disabled switch must not report as on")
                self.assertTrue(note.isVisibleTo(page), f"{tag}/{mode}: no explanation shown")
                self.assertIn("SHG Simulation", note.text())

            self._select(page, "SHG Simulation")
            self.assertTrue(check.isEnabled(), f"{tag}: the switch must come back")
            self.assertTrue(page._spectral_state()["on"], f"{tag}: the tick must survive")
            self.assertFalse(note.isVisibleTo(page))

    # -- honesty about what the data supports ----------------------------------------------

    def test_a_material_that_does_not_disperse_computes_but_says_so(self):
        """Every single-interface palette case is in this position today."""
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        page = self._page(win, SI)
        self._select(page, CONSTANT_SI)
        self._select(page, "SHG Simulation")
        check, spins = self._controls(page)
        check.setChecked(True)
        for spin, value in zip(spins, (0.6, 1.4, 0.2)):
            spin.setValue(value)
        self._update(page)

        note = self._note(page, "si")
        self.assertTrue(note.isVisibleTo(page), "a flat spectrum must be called out")
        self.assertIn("does not move with wavelength", note.text())
        ydata = self._canvas(page, "si").figure.axes[0].lines[0].get_ydata()
        self.assertTrue(np.allclose(ydata, ydata[0]), "this case really is flat")

    def test_an_exported_spectrum_carries_its_constant_d_assumption(self):
        """Rui's ask was to document the constant-d assumption properly, and a saved file is where
        a result outlives the screen it was drawn on. The Python result carried it in
        stages["assumptions"]; the app's Export data dropped the stages, so a spectrum saved from
        the app arrived without it."""
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        page = self._page(win, ML)
        self._select(page, DISPERSIVE_ML)
        self._select(page, "SHG Simulation")
        check, spins = self._controls(page)
        check.setChecked(True)
        for spin, value in zip(spins, (0.8, 1.2, 0.2)):
            spin.setValue(value)
        self._update(page)
        payload = page._build_export_payload()
        self.assertIn("assumptions", payload)
        self.assertIn("wavelength-independent", payload["assumptions"]["d_voigt"])

    def test_a_dispersive_material_computes_quietly_and_moves(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        page = self._page(win, ML)
        self._select(page, DISPERSIVE_ML)
        self._select(page, "SHG Simulation")
        check, spins = self._controls(page)
        check.setChecked(True)
        # 0.01 um: the default film is 1 um of quartz, whose interference fringes sit about 0.06 um
        # apart at 0.6 um (lambda^2 / 4nh), so any step above half that is genuinely undersampled
        # and the app says so. This test is about a dispersive material raising no note of its own.
        for spin, value in zip(spins, (0.6, 1.4, 0.01)):
            spin.setValue(value)
        self._update(page)

        self.assertFalse(self._note(page, "ml").isVisibleTo(page),
                         "a genuinely dispersive, adequately sampled sweep must not raise a note")
        ydata = self._canvas(page, "ml").figure.axes[0].lines[0].get_ydata()
        self.assertFalse(np.allclose(ydata, ydata[0]), "a dispersive sweep must move")

    def test_every_shipped_dispersive_row_computes_on_both_tabs(self):
        """EVERY row, on BOTH tabs -- the one above drives a single hand-picked case.

        WHY THIS EXISTS. The ML combo learned the dispersive rows before the ML RUN PATH did, so
        all five were selectable and every Update raised "system_preset must be one of ...". The
        SI tab was fine, which is exactly why one hand-picked case is not enough: the two tabs
        resolve a selection through different code. The GUI matrix sweep did catch it, but that
        sweep costs an hour; this costs seconds and fails in the same place.

        The geometry is deliberately generic. At normal incidence, and at phi = 0 for KTP (100),
        the SHG channel is forbidden by symmetry and the curve is a true zero -- flat, correct, and
        indistinguishable here from a material whose index never reached the solver. So this drives
        an oblique angle and an off-axis polarizer, and asks whether ANY plotted channel moves."""
        from shaarp.desktop_app import TOOLTIPS, build_main_window
        from shaarp.dispersion import dispersive_material_names, load_shipped_table

        names = dispersive_material_names()
        self.assertTrue(names, "no dispersive materials ship")
        for index, tag in ((SI, "si"), (ML, "ml")):
            win = build_main_window()
            win._gui_smoke_errors = []
            page = self._page(win, index)
            for name in names:
                with self.subTest(tab=tag, material=name):
                    combo = next((c for c in page.findChildren(QtWidgets.QComboBox)
                                  if c.findText(name) >= 0), None)
                    self.assertIsNotNone(combo, f"{tag}: {name!r} is not offered")
                    combo.setCurrentIndex(combo.findText(name))
                    self._select(page, "SHG Simulation")
                    for key, value in (("theta", 45.0), ("polarizer", 30.0)):
                        for spin in page.findChildren(QtWidgets.QDoubleSpinBox):
                            if spin.toolTip() == TOOLTIPS[key]:
                                spin.setValue(value)
                                break
                    check, spins = self._controls(page)
                    check.setChecked(True)
                    # a band the table answers at BOTH harmonics: eps(2w) is read at lambda/2, so
                    # the blue end of a table runs out twice as fast as the fundamental suggests.
                    low, high = load_shipped_table(name).range_um
                    start = min(2.0 * low, high)
                    for spin, value in zip(spins, (start, high, (high - start) / 4.0)):
                        spin.setValue(float(value))
                    del win._gui_smoke_errors[:]
                    self._update(page)
                    self.assertFalse(win._gui_smoke_errors,
                                     f"{tag}: {name!r} errored on Update: "
                                     f"{win._gui_smoke_errors[:1]}")
                    lines = [line for axis in self._canvas(page, tag).figure.axes
                             for line in axis.lines]
                    self.assertTrue(lines, f"{tag}: {name!r} drew no spectrum at all")
                    moved = any(not np.allclose(y := np.asarray(line.get_ydata(), dtype=float),
                                                y[0]) for line in lines)
                    self.assertTrue(moved,
                                    f"{tag}: every plotted channel for {name!r} is flat, so its "
                                    f"published index curve is not reaching the solver")

    def test_an_unphysical_wavelength_range_refuses_with_its_own_message(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        page = self._page(win, ML)
        self._select(page, POLED_ML)
        self._select(page, "SHG Simulation")
        check, spins = self._controls(page)
        check.setChecked(True)
        for spin, value in zip(spins, (0.42, 1.0, 0.1)):
            spin.setValue(value)
        self._update(page)

        message = win.statusBar().currentMessage()
        self.assertIn("KTP x-cut", message)
        self.assertIn("0.54", message, "the message must name the range that IS usable")
        self.assertNotIn("Check the material tensors", message,
                         "a self-describing message must not be buried under the generic hint")

    # -- the map path ----------------------------------------------------------------------

    def test_fresnel_uses_its_own_scan_range_not_the_maker_one(self):
        """Each sweep mode owns its own angle range without the wavelength sweep, and that has to
        stay true with it. The first draft read the Maker spins in both modes, so the Fresnel map
        silently came out over the Maker range."""
        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        page = self._page(win, ML)
        self._select(page, DISPERSIVE_ML)
        self._select(page, "Fresnel Coefficients")
        maker = [s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                 if s.toolTip() == TOOLTIPS["theta_range"]]
        fresnel = [s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                   if s.toolTip() == TOOLTIPS["fresnel_range"]]
        self.assertEqual(len(maker), 3)
        self.assertEqual(len(fresnel), 3)
        # deliberately different ranges, so whichever one is read is visible in the output
        for spin, value in zip(maker, (0.0, 10.0, 5.0)):
            spin.setValue(value)
        for spin, value in zip(fresnel, (0.0, 60.0, 20.0)):
            spin.setValue(value)
        check, spins = self._controls(page)
        check.setChecked(True)
        for spin, value in zip(spins, (0.8, 1.2, 0.2)):
            spin.setValue(value)
        self._update(page)

        axis = self._canvas(page, "ml").figure.axes[0]
        # the Fresnel range reaches 60 deg; the Maker range would have stopped at 10
        self.assertGreater(axis.get_xlim()[1], 30.0,
                           "the Fresnel map was computed over the Maker scan range")

    def test_a_fresnel_map_shows_all_four_coefficients(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        page = self._page(win, ML)
        self._select(page, DISPERSIVE_ML)
        self._select(page, "Fresnel Coefficients")
        check, spins = self._controls(page)
        check.setChecked(True)
        for spin, value in zip(spins, (0.8, 1.2, 0.2)):
            spin.setValue(value)
        self._update(page)

        titles = [ax.get_title() for ax in self._canvas(page, "ml").figure.axes if ax.get_title()]
        for expected in ("$R_p$", "$R_s$", "$T_p$", "$T_s$"):
            self.assertIn(expected, titles)

    def test_maker_fringes_with_the_sweep_on_draws_a_wavelength_by_angle_map(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        page = self._page(win, ML)
        self._select(page, DISPERSIVE_ML)
        self._select(page, "Maker Fringes")
        check, spins = self._controls(page)
        check.setChecked(True)
        for spin, value in zip(spins, (0.8, 1.2, 0.2)):
            spin.setValue(value)
        self._update(page)

        figure = self._canvas(page, "ml").figure
        # both analyzer channels, each a heat map with its own colorbar
        titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
        self.assertEqual(len(titles), 2, f"expected both Maker channels, got {titles}")
        self.assertIn("wavelength", figure._suptitle.get_text().lower())

    # -- the sweep honours the same physics controls as the single-wavelength run -------------

    def _tipped_spin(self, page, key):
        from shaarp.desktop_app import TOOLTIPS

        return next(s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                    if s.toolTip() == TOOLTIPS[key])

    def _tipped_combo(self, page, key):
        from shaarp.desktop_app import TOOLTIPS

        return next(c for c in page.findChildren(QtWidgets.QComboBox)
                    if c.toolTip() == TOOLTIPS[key])

    def test_the_ml_spectrum_uses_the_model_the_assumptions_panel_names(self):
        """At one wavelength the spectrum must equal the single-wavelength polar value, under EVERY
        Assumptions setting. The sweep passed no assumption at all, so it always computed full
        multiple reflections with every wave: with Jerphagnon & Kurtz selected the polar plot said
        1.86 and the spectrum 55.5, under a schematic captioned with the panel's choice."""
        from shaarp.desktop_app import build_main_window
        from shaarp.layer_stack import build_system_from_stack
        from shaarp.shaarp_gui import FMR_SUBMODES, ML_ASSUMPTIONS, ml_polarimetry_curve
        from tests.gui_harness import ml_case_combo

        win = build_main_window()
        win._gui_smoke_errors = []
        page = self._page(win, ML)
        combo = ml_case_combo(page)
        combo.setCurrentIndex(combo.findText("LiNbO3 (dispersive) 0.40-5.00 um"))
        self._select(page, "SHG Simulation")
        self._tipped_spin(page, "theta").setValue(45.0)
        check, spins = self._controls(page)
        check.setChecked(True)                         # pins Fix Polarizer / Fix Analyzer
        self._tipped_spin(page, "polarizer").setValue(30.0)
        self._tipped_spin(page, "analyzer").setValue(0.0)
        for spin, value in zip(spins, (1.0, 1.0, 0.1)):
            spin.setValue(value)
        assumption = self._tipped_combo(page, "assumptions")
        submode = self._tipped_combo(page, "fmr_submode")

        got_by_model = {}
        for name, code in ML_ASSUMPTIONS.items():
            for sub in (FMR_SUBMODES if code == 0 else [None]):
                with self.subTest(assumption=name, submode=sub):
                    assumption.setCurrentText(name)
                    if sub:
                        submode.setCurrentText(sub)
                    del win._gui_smoke_errors[:]
                    self._update(page)
                    self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors[:1])
                    got = float(self._canvas(page, "ml").figure.axes[0].lines[0].get_ydata()[0])
                    system = build_system_from_stack(page._ml_stack_payload()["stack"],
                                                     wavelength_um=1.0, theta_deg=45.0)
                    ref = float(ml_polarimetry_curve(
                        system, theta_deg=45.0, fixed_phi_deg=30.0, mrassumption=code,
                        inhomogeneous_source_policy=(FMR_SUBMODES[sub] if code == 0 else "all"),
                    )["intensity_reflected"][0])       # psi = 0, the analyzer the spectrum used
                    self.assertAlmostEqual(got, ref, delta=1e-9 * max(1.0, abs(ref)))
                    got_by_model[(name, sub)] = got
        self.assertGreater(len({round(v, 6) for v in got_by_model.values()}), 2,
                           "the models must give different answers here, or this test is blind")

    def test_a_maker_map_row_equals_the_maker_run_at_that_wavelength(self):
        """A map at one wavelength IS a Maker scan, so it must equal the single-wavelength Maker run
        with the same controls. Three of them never reached the map: the sample azimuth, the Maker
        scan's own ellipticity, and the Assumptions panel. All three are set away from their
        defaults here, so dropping any one of them again fails this."""
        from shaarp.desktop_app import TOOLTIPS, build_main_window
        from shaarp.layer_stack import build_system_from_stack
        from shaarp.shaarp_gui import compute_ml_gui_result
        from tests.gui_harness import ml_case_combo

        jk = "Jerphagnon & Kurtz Assumption (No MR)"
        win = build_main_window()
        win._gui_smoke_errors = []
        page = self._page(win, ML)
        combo = ml_case_combo(page)
        combo.setCurrentIndex(combo.findText("LiNbO3 (dispersive) 0.40-5.00 um"))
        self._select(page, "Maker Fringes")
        theta = sorted((s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                        if s.toolTip() == TOOLTIPS["theta_range"]), key=lambda s: s.value())
        th_min, th_step, th_max = theta              # defaults 0.0, 0.05, 45.0
        th_min.setValue(10.0)
        th_max.setValue(30.0)
        th_step.setValue(10.0)
        check, spins = self._controls(page)
        check.setChecked(True)
        self._tipped_spin(page, "polarizer").setValue(45.0)
        self._tipped_spin(page, "analyzer").setValue(0.0)
        self._tipped_spin(page, "maker_ellipticity").setValue(60.0)
        self._tipped_spin(page, "sample_azimuth").setValue(60.0)
        self._tipped_combo(page, "assumptions").setCurrentText(jk)
        for spin, value in zip(spins, (1.0, 1.0, 0.1)):
            spin.setValue(value)
        self._update(page)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors[:1])
        got = np.asarray(page._build_export_payload()["numeric"]["parallel_intensity"], float)

        ref = compute_ml_gui_result(
            "Maker Fringes",
            system=build_system_from_stack(page._ml_stack_payload()["stack"], wavelength_um=1.0,
                                           theta_deg=20.0),
            theta_min_deg=10.0, theta_max_deg=30.0, theta_step_deg=10.0, assumption=jk,
            sample_azimuth_deg=60.0,
            sample_rotation_ccw=bool(page._sample_rotation_state().get("ccw", True)),
            fixed_phi_deg=45.0, analyzer_psi_deg=0.0, ellipticity_deg=60.0)
        want = np.asarray(ref.numeric["parallel_intensity"], float)
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-15)

    def test_a_fresnel_map_row_equals_the_fresnel_run_at_that_wavelength(self):
        """The Maker test above missed Fresnel, and so did the fix: every Fresnel map was computed
        with full multiple reflections whatever the Assumptions panel said (a JK row was off by up
        to 0.25 in T_s). JK and a non-zero azimuth are set here so either being dropped fails."""
        from shaarp.desktop_app import build_main_window
        from shaarp.layer_stack import build_system_from_stack
        from shaarp.shaarp_gui import compute_ml_gui_result
        from tests.gui_harness import ml_case_combo

        jk = "Jerphagnon & Kurtz Assumption (No MR)"
        win = build_main_window()
        win._gui_smoke_errors = []
        page = self._page(win, ML)
        combo = ml_case_combo(page)
        combo.setCurrentIndex(combo.findText("LiNbO3 (dispersive) 0.40-5.00 um"))
        self._select(page, "Fresnel Coefficients")
        from shaarp.desktop_app import TOOLTIPS

        fresnel = sorted((s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                          if s.toolTip() == TOOLTIPS["fresnel_range"]), key=lambda s: s.value())
        fr_min, fr_step, fr_max = fresnel                     # defaults 0.0, 0.05, 89.9
        fr_min.setValue(10.0)
        fr_max.setValue(70.0)
        fr_step.setValue(20.0)
        check, spins = self._controls(page)
        check.setChecked(True)
        self._tipped_spin(page, "sample_azimuth").setValue(60.0)
        self._tipped_combo(page, "assumptions").setCurrentText(jk)
        for spin, value in zip(spins, (1.0, 1.0, 0.1)):
            spin.setValue(value)
        self._update(page)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors[:1])
        numeric = page._build_export_payload()["numeric"]

        ref = compute_ml_gui_result(
            "Fresnel Coefficients",
            system=build_system_from_stack(page._ml_stack_payload()["stack"], wavelength_um=1.0,
                                           theta_deg=20.0),
            fresnel_min_deg=fr_min.value(), fresnel_max_deg=fr_max.value(),
            fresnel_step_deg=fr_step.value(), assumption=jk, sample_azimuth_deg=60.0,
            sample_rotation_ccw=bool(page._sample_rotation_state().get("ccw", True)))
        for name in ("rp", "rs", "tp", "ts"):
            with self.subTest(channel=name):
                np.testing.assert_allclose(np.asarray(numeric[name], float),
                                           np.asarray(ref.numeric[name], float),
                                           rtol=1e-9, atol=1e-12)

    def test_a_custom_film_sweeps_in_every_mode(self):
        """The sweep built its stack from the SAVED form of the editor state, where a Custom layer's
        complex tensors are {re, im} dicts, so every sweep over "Custom film (use fields)" failed
        with "must be real number, not dict" -- a blocker the sweep's own tests never reached."""
        from shaarp.desktop_app import TOOLTIPS, build_main_window
        from tests.gui_harness import ml_case_combo

        for mode in ("SHG Simulation", "Maker Fringes", "Fresnel Coefficients"):
            with self.subTest(mode=mode):
                win = build_main_window()
                win._gui_smoke_errors = []
                page = self._page(win, ML)
                combo = ml_case_combo(page)
                combo.setCurrentIndex(combo.findText("Custom film (use fields)"))
                self._select(page, mode)
                theta = sorted((s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                                if s.toolTip() == TOOLTIPS["theta_range"]), key=lambda s: s.value())
                for spin, value in zip((theta[0], theta[2], theta[1]), (0.0, 10.0, 5.0)):
                    spin.setValue(value)
                page._confirm_long_run = lambda *_a: True
                check, spins = self._controls(page)
                check.setChecked(True)
                for spin, value in zip(spins, (0.8, 1.0, 0.1)):
                    spin.setValue(value)
                self._update(page)
                self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors[:1])
                self.assertTrue(self._canvas(page, "ml").figure.axes, f"{mode}: nothing drawn")

    # -- the scan range fits the table that was chosen ---------------------------------------

    def _case_combo(self, page, index):
        from tests.gui_harness import combo_with_item, ml_case_combo

        return ml_case_combo(page) if index == ML else combo_with_item(page, "Custom (use fields)")

    def test_choosing_a_dispersive_material_fits_the_scan_range_to_its_table(self):
        """A table covering 0.40-5.00 um answers a sweep from 0.80 um only: eps(2w) is the table
        read at lambda / 2. The former default 0.55-1.60 um used to clamp four of the five shipped
        materials while their names said it would not; the range is set here explicitly so the
        fence does not depend on what the fields open at."""
        from shaarp.desktop_app import build_main_window

        name = "LiNbO3 (dispersive) 0.40-5.00 um"
        for index in (SI, ML):
            with self.subTest(tab=index):
                win = build_main_window()
                page = self._page(win, index)
                _check, spins = self._controls(page)
                spins[0].setValue(0.55)
                spins[1].setValue(1.60)
                combo = self._case_combo(page, index)
                combo.setCurrentIndex(combo.findText(name))
                combo.textActivated.emit(name)          # what a user's own pick fires
                self.assertAlmostEqual(spins[0].value(), 0.8, places=9)
                self.assertAlmostEqual(spins[1].value(), 1.6, places=9)
                self.assertIn("half the wavelength", win.statusBar().currentMessage(),
                              "a range that moves on its own has to say why")

    def test_a_range_that_already_fits_the_table_is_left_alone(self):
        from shaarp.desktop_app import build_main_window

        name = "KTP (dispersive) 0.43-3.54 um"
        win = build_main_window()
        page = self._page(win, SI)
        _check, spins = self._controls(page)
        spins[0].setValue(1.0)
        spins[1].setValue(1.2)
        combo = self._case_combo(page, SI)
        combo.setCurrentIndex(combo.findText(name))
        combo.textActivated.emit(name)
        self.assertEqual((spins[0].value(), spins[1].value()), (1.0, 1.2))

    def test_a_restored_selection_does_not_move_a_saved_range(self):
        """Session restore sets the combo programmatically after the scan spins. Snapping on that
        would overwrite the range the session saved, so only a USER's pick snaps."""
        from shaarp.desktop_app import build_main_window

        name = "LiNbO3 (dispersive) 0.40-5.00 um"
        win = build_main_window()
        page = self._page(win, SI)
        _check, spins = self._controls(page)
        spins[0].setValue(0.55)
        spins[1].setValue(1.60)
        combo = self._case_combo(page, SI)
        combo.setCurrentIndex(combo.findText(name))      # programmatic: no textActivated
        self.assertEqual((spins[0].value(), spins[1].value()), (0.55, 1.6))

    # -- a large map asks first --------------------------------------------------------------

    def test_a_long_run_asks_before_computing_and_a_short_one_does_not(self):
        """The 2026-09-19 default grids made a Maker map of 43 x 901 points: measured 21 min on a
        1 um z-cut quartz film and 48 min on the Quartz + Au preset. The defaults are now a
        seconds-long map (26 wavelengths, and ticking the sweep coarsens the theta step to 10 deg),
        so the long grid is set explicitly below. The question is a page hook so it can be
        observed here; in an offscreen run the real one passes straight through, because a modal
        dialog once hung a headless run for ten hours.

        It asks on estimated TIME, not a point count: a Maker map just under a 5,000-point
        threshold ran about eight minutes without a word, and a 1-D spectrum was never asked about
        at all. The estimate times one wavelength of the job itself, so the cases below are long by
        a wide margin on any machine rather than just over the line."""
        from shaarp.desktop_app import LONG_RUN_SECONDS, TOOLTIPS, build_main_window

        win = build_main_window()
        page = self._page(win, ML)
        self._select(page, DISPERSIVE_ML)
        self._select(page, "Maker Fringes")
        check, spins = self._controls(page)
        check.setChecked(True)
        asked = []
        page._confirm_long_run = (lambda n_lam, n_th, est:
                                  asked.append((n_lam, n_th, est)) or False)

        # long: 9 wavelengths x 0-45 deg at the fine 0.05 deg step (ticking the sweep set the
        # step to 10 deg; a user narrowing it back is exactly who meets the question)
        for spin, value in zip(spins, (0.8, 1.2, 0.05)):
            spin.setValue(value)
        theta = sorted((s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                        if s.toolTip() == TOOLTIPS["theta_range"]), key=lambda s: s.value())
        _th_min, th_step, _th_max = theta
        th_step.setValue(0.05)
        self._update(page)
        self.assertEqual(len(asked), 1, "a long map must ask first")
        self.assertGreater(asked[0][2], LONG_RUN_SECONDS)
        self.assertFalse(self._canvas(page, "ml").figure.axes,
                         "declining must leave the map uncomputed")
        self.assertIn("Map not computed", self._stale_text(page),
                      "a declined run must not leave the previous result looking current")
        bar = win.findChild(QtWidgets.QProgressBar)
        self.assertEqual(bar.format(), "Not computed",
                         "the progress bar said '100% Completed' for a run that never happened")

        # a 1-D spectrum with a tiny step is long too, and was never asked about. The Quartz + Au
        # preset costs about 26 ms a wavelength here (2026-09-19), so 10,001 wavelengths (the
        # step's minimum) is about four minutes. 5,001 was about two, and its estimate read 70-112 s
        # -- too close to the one-minute line for a test on an unknown machine.
        from tests.gui_harness import ml_case_combo

        preset = ml_case_combo(page)
        preset.setCurrentIndex(preset.findText("Quartz + Au (Fig 4, 800 nm)"))
        self._select(page, "SHG Simulation")
        for spin, value in zip(spins, (0.6, 1.6, 0.0001)):   # 10,001 wavelengths
            spin.setValue(value)
        self._update(page)
        self.assertEqual(len(asked), 2, "a long 1-D spectrum must ask as well")
        self.assertEqual(asked[1][:2], (10001, 1))
        self._select(page, "Maker Fringes")

        # small: shrink the angle grid to a handful of points; nothing to ask about
        theta = sorted((s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                        if s.toolTip() == TOOLTIPS["theta_range"]), key=lambda s: s.value())
        th_min, th_step, th_max = theta              # defaults 0.0, 0.05, 45.0
        th_min.setValue(0.0)
        th_max.setValue(10.0)
        th_step.setValue(5.0)
        for spin, value in zip(spins, (0.8, 1.2, 0.2)):
            spin.setValue(value)
        self._update(page)
        self.assertEqual(len(asked), 2, "a short map must not ask")
        self.assertTrue(self._canvas(page, "ml").figure.axes, "a short map computes")
        self.assertEqual(self._stale_text(page), "", "a completed run clears the stale marker")
        self.assertEqual(bar.format(), "100% Completed")

    def _stale_text(self, page):
        """The stale banner's text when it is showing, else ''."""
        for label in page.findChildren(QtWidgets.QLabel):
            if label.text().startswith("●") and label.isVisibleTo(page):
                return label.text()
        return ""

    # -- every note is shown, not the first --------------------------------------------------

    def test_a_stack_with_two_findings_shows_both(self):
        """The shipped Quartz + Au preset, swept past the quartz grid: the Au coating stays at one
        index (a partly-frozen stack) AND the quartz clamps at the red end. The note used to show
        notes[0] only, so whichever finding came second was never seen."""
        from shaarp.desktop_app import build_main_window
        from tests.gui_harness import ml_case_combo

        win = build_main_window()
        page = self._page(win, ML)
        combo = ml_case_combo(page)
        combo.setCurrentIndex(combo.findText("Quartz + Au (Fig 4, 800 nm)"))
        self._select(page, "SHG Simulation")
        check, spins = self._controls(page)
        check.setChecked(True)
        for spin, value in zip(spins, (1.6, 2.4, 0.4)):
            spin.setValue(value)
        win._gui_smoke_errors = []
        self._update(page)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors[:1])
        text = self._note(page, "ml").text()
        self.assertIn("part of this stack disperses", text)
        self.assertIn("Au", text)
        self.assertIn("tabulated grid", text, "the quartz clamp must be shown alongside")

        # A DECLINED run leaves this plot on screen, so its notes stay with it (they were cleared
        # at the start of every sweep, leaving a plot without the notes that qualify it)
        page._confirm_long_run = lambda *_a: False
        for spin, value in zip(spins, (0.6, 1.6, 0.0001)):
            spin.setValue(value)
        self._update(page)
        self.assertIn("not computed", self._stale_text(page))
        self.assertTrue(self._note(page, "ml").isVisibleTo(page))
        self.assertEqual(self._note(page, "ml").text(), text)

    def test_the_long_run_question_reads_naturally(self):
        from shaarp.desktop_app import _long_run_text

        spectrum = _long_run_text(10001, 1, 240.0)
        self.assertIn("This is 10,001 wavelengths and will take about 4 minutes", spectrum)
        self.assertNotIn("points", spectrum, "a spectrum said '10,001 points (10,001 wavelengths)'")
        self.assertNotIn("θ step", spectrum)
        a_map = _long_run_text(43, 901, 21 * 60.0)
        self.assertIn("38,743 points (43 wavelengths × 901 angles)", a_map)
        self.assertIn("about 21 minutes", a_map)
        self.assertIn("or the θ step", a_map)
        self.assertIn("about 34 hours", _long_run_text(43, 901, 2058 * 60.0),
                      "'about 2,058 minutes' is not a length of time anyone reads")
        self.assertIn("about a minute", _long_run_text(5, 10, 70.0))

    # -- what stays on screen after a run that did not complete ----------------------------------

    def test_the_single_wavelength_note_stands_down_while_sweeping(self):
        """With the sweep ticked the single wavelength is not computed, so its note described a
        run that was not going to happen."""
        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        page = self._page(win, SI)
        combo = self._case_combo(page, SI)
        combo.setCurrentIndex(combo.findText("LiNbO3 (dispersive) 0.40-5.00 um"))
        wavelength = next(s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                          if s.toolTip() == TOOLTIPS["wavelength"])
        wavelength.setValue(6.0)
        note = self._note(page, "si")
        self.assertTrue(note.isVisibleTo(page), "6 um is past the table: the note must show")
        self.assertIn("0.40–5.00", note.text(), "the table's span as the material's name prints it")
        check, _spins = self._controls(page)
        check.setChecked(True)
        self.assertFalse(note.isVisibleTo(page))
        # ...and the field itself stands down, as theta does in map mode: left live it changed
        # the schematic header and the banner, and no result
        self.assertFalse(wavelength.isEnabled(), "the sweep owns lambda while it is ticked")
        check.setChecked(False)
        self.assertTrue(note.isVisibleTo(page), "unticking brings the single wavelength's note back")
        self.assertTrue(wavelength.isEnabled())

    def test_a_failed_update_leaves_the_previous_plots_and_a_good_one_clears_the_error(self):
        """A failed Update used to empty the Spectrum tab under a banner saying the plots still
        show the previous result, and its red status line outlived the next successful run."""
        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        page = self._page(win, ML)
        self._select(page, DISPERSIVE_ML)
        self._select(page, "SHG Simulation")
        check, spins = self._controls(page)
        check.setChecked(True)
        for spin, value in zip(spins, (0.8, 0.9, 0.1)):
            spin.setValue(value)
        self._update(page)
        self.assertTrue(self._canvas(page, "ml").figure.axes, "the sweep drew")

        check.setChecked(False)
        self._select(page, "Maker Fringes")
        theta = sorted((s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                        if s.toolTip() == TOOLTIPS["theta_range"]), key=lambda s: s.value())
        th_min, th_step, th_max = theta
        th_min.setValue(20.0)
        th_max.setValue(10.0)
        self._update(page)
        self.assertIn("did not complete", self._stale_text(page))
        self.assertTrue(self._canvas(page, "ml").figure.axes,
                        "a failed Update must leave the previous plot where the banner says it is")
        red = [lab for lab in page.findChildren(QtWidgets.QLabel) if "b00020" in lab.styleSheet()]
        self.assertEqual(len(red), 1, "the failure is shown in red")
        status = red[0]
        # min > max, not only min = max: the check used to run after the compute, which had
        # already failed on the reversed grid with the generic "check the material tensors" text
        self.assertIn("Fix θ min / θ max in the Maker Fringes Scan Range", status.text())
        self.assertEqual(win.findChild(QtWidgets.QProgressBar).format(), "Did not complete")

        th_min.setValue(0.0)
        th_max.setValue(10.0)
        th_step.setValue(5.0)
        self._update(page)
        self.assertEqual(self._stale_text(page), "")
        self.assertEqual(status.styleSheet(), "", "a good run must not stay red")
        self.assertNotIn("scan range", status.toolTip())
        self.assertFalse(self._canvas(page, "ml").figure.axes,
                         "a completed non-sweep run empties the Spectrum tab")


if __name__ == "__main__":
    unittest.main()
