"""What the app shows on a first look, before any click (Rui's cold-eyes review, 2026-09-20).

Three findings from opening the app as a new user, each fenced here:

1. The single-interface tab opened on "Custom (use fields)" with the -43m / z-cut / normal-
   incidence defaults, whose reflected SHG is forbidden by symmetry -- so the first Update a new
   user pressed drew "SHG ≈ 0 (symmetry-forbidden)". It now opens on the README's own hero case.
2. The wavelength-sweep defaults (0.55-1.60 um at 0.025 um) were finer and wider than a first look
   needs, and with the fine single-wavelength theta step they made the app's default map a
   tens-of-minutes job. Ticking the sweep now coarsens the theta step to 10 deg -- unless the
   user already chose a step, which is theirs.
3. A box you type into and a chip you click looked the same: both white rounded rectangles with
   the same grey border, told apart only by the focus ring AFTER a click.

These are read off the LIVE widgets of a pristine window, as the matrix sweep builds it.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

import numpy as np
from PySide6 import QtWidgets  # noqa: E402

from tests import gui_harness as gh  # noqa: E402

SI, ML = 0, 1


def _tabs(win):
    top = next(t for t in win.findChildren(QtWidgets.QTabWidget)
               if t.count() >= 2 and "SHAARP" in t.tabText(0))
    return top.widget(SI), top.widget(ML)


def _sweep_switch(page):
    return next(c for c in page.findChildren(QtWidgets.QCheckBox)
                if "sweep the wavelength" in c.text())


def _step_spin(page, tip_key):
    """The theta-step spin of one ML scan-range group: the middle value of (min, step, max)."""
    spins = sorted(gh.spins_with_tip(page, tip_key), key=lambda s: s.value())
    return spins[1]


class TheFirstLookIsAComputableCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_the_si_tab_opens_on_the_documented_hero_case_with_its_panel_mirrored(self):
        from shaarp.desktop_app import STARTUP_SI_CASE, build_main_window

        win = build_main_window()
        si, _ml = _tabs(win)
        case = gh.combo_with_item(si, "Custom (use fields)")
        self.assertEqual(case.currentText(), STARTUP_SI_CASE)
        self.assertEqual(STARTUP_SI_CASE, "GaAs (111)",
                         "the README and first_run.md walk a new user to GaAs (111)")
        point_group = next(c for c in si.findChildren(QtWidgets.QComboBox)
                           if c.findText("-43m") >= 0)
        self.assertEqual(point_group.currentText(), "-43m", "the panel mirrors the case")
        orient = next(c for c in si.findChildren(QtWidgets.QComboBox)
                      if c.findText("z-cut (identity)") >= 0)
        self.assertNotEqual(orient.currentText(), "z-cut (identity)",
                            "GaAs (111) is an oriented cut, not the identity")

    def test_the_first_update_draws_lobes_not_a_symmetry_forbidden_zero(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        si, _ml = _tabs(win)
        gh.click_update(si, self.app)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
        curves = gh.snapshot_curves(si)
        self.assertTrue(curves, "no polar curves after the first Update")
        peak = max(float(np.nanmax(np.abs(c))) for c in curves if c.size)
        self.assertGreater(peak, 1e-6, "the opening case must not be symmetry-forbidden")


class TheSweepDefaultsSuitAFirstLook(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_lambda_defaults_on_both_tabs(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        for page in _tabs(win):
            lam = gh.spins_with_tip(page, "lambda_range")
            self.assertEqual([s.value() for s in lam], [0.55, 0.8, 0.01])
            # each default has a quick-pick chip, so the row shows the active value lit
            chips = [b for b in page.findChildren(QtWidgets.QPushButton)
                     if b.isCheckable() and b.isChecked() and b.text() in ("0.55", "0.8", "0.01")]
            self.assertEqual(sorted(b.text() for b in chips), ["0.01", "0.55", "0.8"])

    def test_ticking_the_sweep_coarsens_the_theta_steps_and_unticking_restores_them(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        _si, ml = _tabs(win)
        th_step, fr_step = _step_spin(ml, "theta_range"), _step_spin(ml, "fresnel_range")
        fine_th, fine_fr = th_step.value(), fr_step.value()
        self.assertLess(fine_th, 1.0, "the single-wavelength Maker step stays fine")
        switch = _sweep_switch(ml)
        switch.setChecked(True)
        self.app.processEvents()
        self.assertEqual((th_step.value(), fr_step.value()), (10.0, 10.0))
        self.assertIn("θ step set to 10°", win.statusBar().currentMessage(),
                      "a step that moves on its own has to say why")
        switch.setChecked(False)
        self.app.processEvents()
        self.assertEqual((th_step.value(), fr_step.value()), (fine_th, fine_fr))

    def test_a_step_the_user_chose_is_not_overridden_by_the_sweep(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        _si, ml = _tabs(win)
        th_step = _step_spin(ml, "theta_range")
        th_step.setValue(2.0)
        switch = _sweep_switch(ml)
        switch.setChecked(True)
        self.app.processEvents()
        self.assertEqual(th_step.value(), 2.0, "the user's step is theirs")
        switch.setChecked(False)
        self.app.processEvents()
        self.assertEqual(th_step.value(), 2.0)

    def test_the_single_wavelength_maker_fence_is_untouched_by_the_sweep_default(self):
        """The 10 deg applies to the SWEPT map only; the fine step that resolves the shipped
        preset's fringes at one wavelength (tests/test_maker_default_resolves_fringes.py) is what
        a pristine window still carries."""
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        _si, ml = _tabs(win)
        self.assertFalse(_sweep_switch(ml).isChecked())
        self.assertLessEqual(_step_spin(ml, "theta_range").value(), 0.1)


class AFieldAChipAndADropdownLookDifferent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_idle_chips_are_tinted_and_fields_carry_an_inset_edge(self):
        from shaarp.desktop_app import MODERN_QSS, build_main_window

        win = build_main_window()
        si, _ml = _tabs(win)
        chip = next(b for b in si.findChildren(QtWidgets.QPushButton)
                    if b.isCheckable() and b.text() == "15")
        sheet = chip.styleSheet()
        idle = sheet.split("QPushButton:hover")[0]
        self.assertIn("background-color", idle, "an idle chip must not fall back to the white button")
        self.assertNotIn("#ffffff", idle.lower())
        self.assertIn("border-radius: 10px", idle, "a chip is a pill")
        field_rule = MODERN_QSS.split("QLineEdit, QAbstractSpinBox")[1].split("}")[0]
        self.assertIn("border-bottom: 2px", field_rule, "a field carries a heavier bottom edge")
        self.assertIn("border-radius: 3px", field_rule, "a field is square-cornered, a chip round")

    def test_the_dropdown_arrow_is_a_bundled_image(self):
        from shaarp.desktop_app import MODERN_QSS, _asset_file_url

        url = _asset_file_url("combo_arrow.png")
        self.assertTrue(url, "shaarp/assets/combo_arrow.png must ship (package-data assets/*.png)")
        self.assertIn(f'url("{url}")', MODERN_QSS)
        self.assertNotIn("__COMBO_ARROW__", MODERN_QSS)


if __name__ == "__main__":
    unittest.main()
