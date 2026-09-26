"""Round 1 of the cold-eyes GUI audit (2026-09-20): what a first-time user met, now fenced.

The walk drove the app as a newcomer and judged the pictures. These are the defects it found
that a green suite had not: values cut off inside their own cells, a picture drawn from a
made-up number, a window that could not fit a laptop screen, a help page taller than any
screen, radial labels printed over the data, a warning far from the plot it warns about.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from tests import gui_harness as gh  # noqa: E402

SI, ML = 0, 1


def _tabs(win):
    top = next(t for t in win.findChildren(QtWidgets.QTabWidget)
               if t.count() >= 2 and "SHAARP" in t.tabText(0))
    return top, top.widget(SI), top.widget(ML)


def _canvases(page):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    return page.findChildren(FigureCanvasQTAgg)


def _figure_texts(page):
    out = []
    for cv in _canvases(page):
        for ax in cv.figure.axes:
            out.extend(t.get_text() for t in ax.texts)
            out.append(ax.get_title())
    return out


class ValuesAreReadableInTheirCells(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_a_complex_permittivity_fits_its_cell(self):
        """GaAs (111) writes 13.4361+0.432588j; at 62 px the cell showed "432588j"."""
        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        win.resize(1480, 940)
        win.show()
        self.app.processEvents()
        _top, si, _ml = _tabs(win)
        cells = [e for e in si.findChildren(QtWidgets.QLineEdit)
                 if e.toolTip() == TOOLTIPS["full_tensor"]]
        eps = [e for e in cells if "j" in e.text()]
        self.assertTrue(eps, "the startup case should fill complex permittivities")
        for e in eps:
            needed = e.fontMetrics().horizontalAdvance(e.text()) + 12
            self.assertGreaterEqual(e.width(), needed,
                                    f"cell {e.width()} px cannot show {e.text()!r} ({needed} px)")


class TheWindowFitsALaptop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_minimum_height_fits_768(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win.show()
        self.app.processEvents()
        self.assertLessEqual(win.minimumSizeHint().height(), 760,
                             "a 1366x768 laptop must show the Update strip and the status bar")
        win.resize(1366, 768)
        self.app.processEvents()
        self.assertEqual((win.width(), win.height()), (1366, 768))


class TheHelpPageScrolls(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_user_guide_is_a_scrollable_dialog_of_screen_height(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win.show()
        seen = {}

        def _inspect():
            d = QtWidgets.QApplication.activeModalWidget()
            if d is not None:
                seen["height"] = d.height()
                seen["browser"] = d.findChild(QtWidgets.QTextBrowser) is not None
                d.accept()

        QtCore.QTimer.singleShot(500, _inspect)
        act = next(a for a in win.findChildren(QtGui.QAction) if a.text() == "User Guide")
        act.trigger()
        self.app.processEvents()
        self.assertTrue(seen.get("browser"), "the guide must live in a scrolling text browser")
        self.assertLessEqual(seen["height"], 700, "the dialog must fit a laptop screen")


class ACollapsedGroupIsATitleLine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_collapsing_leaves_no_empty_card(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win.resize(1480, 940)
        win.show()
        self.app.processEvents()
        _top, si, _ml = _tabs(win)
        g = next(gb for gb in si.findChildren(QtWidgets.QGroupBox)
                 if gb.title().startswith("Crystal Structure"))
        open_h = g.height()
        g.setChecked(False)
        self.app.processEvents()
        self.app.processEvents()
        self.assertLess(g.height(), 45, f"collapsed group still {g.height()} px tall")
        g.setChecked(True)
        self.app.processEvents()
        self.assertGreater(g.height(), open_h - 4)


class PicturesTellTheTruth(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_polar_radial_labels_are_few_and_short(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        _top, si, _ml = _tabs(win)
        gh.click_update(si, self.app)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
        polar = [ax for cv in _canvases(si) for ax in cv.figure.axes
                 if ax.name == "polar" and ax.lines]
        self.assertTrue(polar, "no polar panel after the SI Update")
        for ax in polar:
            labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
            self.assertLessEqual(len(labels), 4, labels)
            for lbl in labels:
                self.assertLessEqual(len(lbl.replace("−", "-")), 7, f"radial label {lbl!r} too long")
            self.assertFalse(ax.yaxis.get_offset_text().get_visible(),
                             "an offset text such as 1e-11 collides with the 90 deg label")

    def test_the_multilayer_schematic_says_isotropic_not_a_curie_symbol(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        top, _si, ml = _tabs(win)
        top.setCurrentIndex(ML)
        gh.click_update(ml, self.app)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
        texts = " | ".join(_figure_texts(ml))
        self.assertIn("isotropic", texts)
        self.assertNotIn("∞∞m", texts)

    def test_the_launch_banner_shows_the_loaded_stack_and_case(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        _top, si, ml = _tabs(win)
        ml_texts = " | ".join(_figure_texts(ml))
        self.assertIn("quartz", ml_texts.lower(), "the Fig 4 preset is loaded, so draw it")
        si_texts = " | ".join(_figure_texts(si))
        self.assertIn("-43m", si_texts, "the case's point group, not a bare 'crystal'")

    def test_the_si_schematic_refracts_with_the_case_index(self):
        """GaAs at theta_i = 30 deg: n = 3.67 puts the pump at 7.8 deg inside the crystal, not the
        ~20 deg of an illustrative n = 1.5."""
        import math

        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        _top, si, _ml = _tabs(win)
        theta = next(s for s in si.findChildren(QtWidgets.QDoubleSpinBox)
                     if s.toolTip() == TOOLTIPS["theta"])
        theta.setValue(30.0)
        gh.click_update(si, self.app)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
        sch = next(cv for cv in _canvases(si)
                   if cv.figure.axes and cv.figure.axes[0].get_title().startswith("Optical setup"))
        ax = sch.figure.axes[0]
        # rays are arrow annotations (tail = xyann, tip = xy); the transmitted ones point DOWN.
        # With the illustrative n = 1.5 both transmitted rays sit near 19 deg; with the case's
        # own indices (3.67 / 4.19) they sit near 8 deg.
        from matplotlib.text import Annotation

        angles = []
        for a in ax.texts:
            if not isinstance(a, Annotation) or a.get_text():
                continue
            (x1, y1), (x0, y0) = a.xy, a.xyann
            dx, dy = float(x1 - x0), float(y1 - y0)
            if dy < 0 and abs(dx) > 1e-9:
                angles.append(math.degrees(math.atan2(abs(dx), abs(dy))))
        self.assertTrue(angles, "no transmitted ray drawn")
        self.assertLess(min(angles), 15.0, f"pump drawn at {min(angles):.1f} deg inside GaAs")


class TheSweepNoteSitsUnderThePlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_an_undersampled_map_says_so_on_the_spectrum_tab(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        top, _si, ml = _tabs(win)
        top.setCurrentIndex(ML)
        next(c for c in ml.findChildren(QtWidgets.QComboBox)
             if c.findText("Fresnel Coefficients") >= 0).setCurrentText("Fresnel Coefficients")
        switch = next(c for c in ml.findChildren(QtWidgets.QCheckBox)
                      if "sweep the wavelength" in c.text())
        switch.setChecked(True)
        self.app.processEvents()
        gh.click_update(ml, self.app)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
        note = ml.findChild(QtWidgets.QLabel, "spec_result_note_ml")
        self.assertIsNotNone(note)
        self.assertTrue(note.isVisibleTo(ml), "the note must show beside the map it warns about")
        self.assertIn("undersampled", note.text())
        # a run that is not a sweep empties the Spectrum tab and its note with it
        switch.setChecked(False)
        self.app.processEvents()
        gh.click_update(ml, self.app)
        self.assertFalse(note.isVisibleTo(ml))


if __name__ == "__main__":
    unittest.main()


class ButtonsSayWhenTheyCannotWork(unittest.TestCase):
    """A control that is live but does nothing is worse than one that is greyed with a reason."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _copy_buttons(self, page):
        return [b for b in page.findChildren(QtWidgets.QPushButton)
                if b.text().startswith("Copy closed form")]

    def test_copy_closed_form_is_greyed_until_there_is_one(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        _top, si, _ml = _tabs(win)
        buttons = self._copy_buttons(si)
        self.assertEqual(len(buttons), 2, "Python/SymPy and Mathematica")
        self.assertEqual([b.isEnabled() for b in buttons], [False, False],
                         "nothing has been computed yet")
        for b in buttons:
            # the reason lives on the STATUS tip: a widget's audit/session identity is keyed on
            # its tooltip (shaarp/gui_introspect.py), so a state-dependent tooltip renames the
            # widget and drops it out of the coverage registry and out of saved sessions
            self.assertIn("after", b.statusTip().lower(), "say when it becomes available")
            self.assertTrue(b.toolTip().strip(), "the tooltip stays the control's own description")
        gh.click_update(si, self.app)  # SHG Simulation: numbers, no closed form
        self.assertEqual([b.isEnabled() for b in buttons], [False, False],
                         "a numeric run produces no closed form")
        combo = next(c for c in si.findChildren(QtWidgets.QComboBox)
                     if c.findText("Partial Analytical Expressions") >= 0)
        combo.setCurrentText("Partial Analytical Expressions")
        gh.click_update(si, self.app)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
        self.assertEqual([b.isEnabled() for b in buttons], [True, True])
        clip = QtWidgets.QApplication.clipboard()
        clip.clear()
        buttons[0].click()
        self.app.processEvents()
        self.assertTrue(clip.text().strip(), "Copy put nothing on the clipboard")

    def test_adding_a_layer_says_what_the_new_row_holds(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        _top, _si, ml = _tabs(win)
        spin = next(s for s in ml.findChildren(QtWidgets.QSpinBox))
        spin.setValue(spin.value() + 1)
        self.app.processEvents()
        message = win.statusBar().currentMessage()
        self.assertIn("Added 1 layer", message)
        self.assertIn("LiNbO3", message, "name the material the new row silently carries")
