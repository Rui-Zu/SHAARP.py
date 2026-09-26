"""Round 2 of the cold-eyes GUI audit (2026-09-20): what still stopped a first-time user.

Round 1 fixed what a newcomer met on the first screen. This round walked the whole app -- every
Functionality, the layer editor, exports, sessions, error paths -- and found a different class of
defect: things the app SAYS or WRITES that are not true of the run in front of you.

  * a spectrum export dropped a byte-identical copy of the PREVIOUS analytical run's closed form
    beside it, named for this one;
  * "Export figure" before the first Update wrote a picture of the instruction page and reported
    success, while "Export data" in the same state correctly refused;
  * the closed form ran ~400,000 px off the right edge of an 800 px box, because the HTML set
    `white-space:pre` and defeated the box's own word wrap;
  * the undersampling note for a sweep was laid out below the bottom of its own tab;
  * the angle scan had no undersampling guard at all, so widening the Maker step off its slow
    default drew an aliased curve under a green "matches the original package" line;
  * the symbols line printed two numbers for one row ("2: layer 2: KTP x-cut");
  * a Full-Analytical run's symbol list overflowed the status label with nowhere to go.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

import numpy as np
from matplotlib.figure import Figure  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

from tests import gui_harness as gh  # noqa: E402

SI, ML = 0, 1


def _tabs(win):
    top = next(t for t in win.findChildren(QtWidgets.QTabWidget)
               if t.count() >= 2 and "SHAARP" in t.tabText(0))
    return top, top.widget(SI), top.widget(ML)


def _expr_box(page):
    return page.findChild(QtWidgets.QTextEdit, "expr_box")


def _select(page, text):
    combo = next(c for c in page.findChildren(QtWidgets.QComboBox) if c.findText(text) >= 0)
    combo.setCurrentText(text)
    return combo


def _line_fig(y):
    fig = Figure()
    fig.add_subplot(111).plot(np.arange(len(y)), y)
    return fig


class TheClosedFormIsReadable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_the_expression_wraps_to_the_viewport(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        win.resize(1480, 940)
        win.show()
        self.app.processEvents()
        _top, si, _ml = _tabs(win)
        _select(si, "Partial Analytical Expressions")
        gh.click_update(si, self.app)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
        box = _expr_box(si)
        self.assertTrue(box.toPlainText().strip(), "no expression to judge")
        viewport = box.viewport().width()
        self.assertLessEqual(box.document().idealWidth(), viewport + 4,
                             "the closed form must wrap, not run off the right edge")


class ExportsDescribeThisRun(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_a_sweep_clears_the_previous_runs_closed_form(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        _top, si, _ml = _tabs(win)
        _select(si, "Partial Analytical Expressions")
        gh.click_update(si, self.app)
        box = _expr_box(si)
        self.assertTrue(str(box.property("raw_text") or "").strip(), "expected a closed form")
        _select(si, "SHG Simulation")
        switch = next(c for c in si.findChildren(QtWidgets.QCheckBox)
                      if "sweep the wavelength" in c.text())
        switch.setChecked(True)
        self.app.processEvents()
        gh.click_update(si, self.app)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
        self.assertEqual(str(box.property("raw_text") or ""), "",
                         "a spectrum carries no closed form; a stale one gets exported beside it")
        buttons = [b for b in si.findChildren(QtWidgets.QPushButton)
                   if b.text().startswith("Copy closed form")]
        self.assertEqual([b.isEnabled() for b in buttons], [False, False])

    def test_export_figure_refuses_before_the_first_update(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        _top, si, ml = _tabs(win)
        for page in (si, ml):
            button = next(b for b in page.findChildren(QtWidgets.QPushButton)
                          if b.text() == "Export figure")
            button.click()
            self.app.processEvents()
            self.assertIn("Nothing to export", win.statusBar().currentMessage(),
                          "a premature click wrote a picture of the instruction page")


class WarningsReachTheEye(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_the_sweep_note_sits_above_the_plot_it_describes(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win.resize(1480, 940)
        win.show()
        self.app.processEvents()
        _top, si, _ml = _tabs(win)
        note = si.findChild(QtWidgets.QLabel, "spec_result_note_si")
        canvas = si.findChild(QtWidgets.QWidget, "spectrum_canvas_si")
        self.assertIsNotNone(note)
        self.assertIsNotNone(canvas)
        # compare LAYOUT ORDER, not coordinates: the note is hidden until a sweep raises one,
        # and a hidden widget shares the canvas's y
        layout = note.parentWidget().layout()
        order = [layout.itemAt(i).widget() for i in range(layout.count())]
        self.assertLess(order.index(note), order.index(canvas),
                        "below the canvas the note fell off the bottom of its own tab")

    def test_an_aliased_angle_scan_is_captioned_and_a_resolved_one_is_not(self):
        """Measured on the shipped Quartz + Au preset: 1 deg aliases (the slope reverses at ~45%
        of the steps), 0.1 deg resolves (~23%). See _undersampled_caption for the full table."""
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        top, _si, ml = _tabs(win)
        top.setCurrentIndex(ML)
        _select(ml, "Maker Fringes")
        step = sorted(gh.spins_with_tip(ml, "theta_range"), key=lambda s: s.value())[1]
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        canvas = ml.findChild(FigureCanvasQTAgg, "maker_canvas")
        for value, expected in ((1.0, True), (0.1, False)):
            step.setValue(value)
            gh.click_update(ml, self.app)
            self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
            captioned = any("undersampled" in t.get_text() for t in canvas.figure.texts)
            self.assertEqual(captioned, expected, f"theta step {value}")

    def test_the_detector_does_not_cry_wolf(self):
        from shaarp.desktop_app import _undersampled_caption

        self.assertIsNone(_undersampled_caption(_line_fig(np.linspace(0, 1, 200))), "a ramp")
        self.assertIsNone(_undersampled_caption(_line_fig(np.sin(np.linspace(0, 20, 400)))),
                          "a well-sampled oscillation")
        self.assertIsNone(_undersampled_caption(_line_fig(np.ones(50))), "a flat line")
        noise = np.random.default_rng(0).normal(0.0, 1e-30, 200)
        self.assertIsNone(_undersampled_caption(_line_fig(noise)),
                          "numerical noise is not an undersampled fringe train")
        self.assertIsNotNone(_undersampled_caption(_line_fig(np.sin(np.linspace(0, 600, 200)))),
                             "an aliased oscillation")

    def test_a_caption_does_not_overprint_the_axis_label(self):
        from shaarp.desktop_app import _caption_figure

        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = Figure(figsize=(7, 4), layout="constrained")
        FigureCanvasAgg(fig)  # a bare Figure has no renderer to measure against
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("Incident Angle")
        fig.canvas.draw()
        _caption_figure(fig, "⚠ a two-line warning that must sit clear of the axes below them")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        caption = fig.texts[-1]
        cap_top = caption.get_window_extent(renderer).y1
        label_bottom = ax.xaxis.get_label().get_window_extent(renderer).y0
        self.assertLessEqual(cap_top, label_bottom + 1,
                             "the caption overlaps the x-axis label")
        axes_top = ax.get_window_extent(renderer).y1
        self.assertLessEqual(axes_top, fig.bbox.y1 + 1, "the title was pushed off the top")


class OneNumberPerRow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_the_symbols_line_numbers_each_row_once(self):
        import re

        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        top, _si, ml = _tabs(win)
        top.setCurrentIndex(ML)
        _select(ml, "Partial Analytical Expressions")
        gh.click_update(ml, self.app)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
        text = _expr_box(ml).property("raw_text") or _expr_box(ml).toPlainText()
        match = re.search(r"layers = ([^\n]*)", text)
        self.assertIsNotNone(match, "no layers line in the closed form header")
        for entry in match.group(1).split(", "):
            self.assertIsNone(re.match(r"^\s*\d+:\s*layer\s*\d+:", entry),
                              f"two numbers for one row: {entry!r}")


class TheStatusLineFits(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_a_full_analytical_symbol_list_does_not_overflow_the_label(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        win.resize(1480, 940)
        win.show()
        self.app.processEvents()
        _top, si, _ml = _tabs(win)
        _select(si, "Full Analytical Expressions")
        gh.click_update(si, self.app)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
        label = next(lab for lab in si.findChildren(QtWidgets.QLabel)
                     if lab.text().startswith("Checked:"))
        self.assertLessEqual(label.heightForWidth(label.width()), label.height(),
                             "the status line is clipped with the status bar right below it")
        self.assertIn("symbols:", label.toolTip(), "the full list belongs in the tooltip")


if __name__ == "__main__":
    unittest.main()
