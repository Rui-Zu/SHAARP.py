"""WIDGET-GEOMETRY INVARIANTS -- the class of defect found by using the app.

He reported the stale-plot notification banner rendering enormous ("your notification page is too
large"): it measured 480 px tall for two lines of text. EVERY automated gate called that green,
because they all assert things about PLOT CANVASES (figure px-size matches its canvas, nothing
clipped, window min-width) -- and a 480 px LABEL is not a canvas.

The underlying pattern: those gates assert the ABSENCE OF FAILURE (no crash, no clip) and never the
PRESENCE OF QUALITY (sane sizes). This file asserts size budgets directly, on every visible widget,
and -- critically -- in the TRANSIENT STATES that only exist after you interact with the app. The
banner is invisible until an input changes, which is precisely why screenshot-based checking never
caught it: the clean default state does not contain the broken widget.

The budgets are deliberately loose (a wrong widget missed them by ~10x, not by 20%), so this fails on
real geometry blowups, not on font-metric noise:
  * a text LABEL may not exceed ~4 text lines of height (the banner is 2 lines by design);
  * no leaf (non-container, non-canvas) widget may exceed 25% of the window height;
  * nothing may exceed the window's own width.
Checked at 1366x768 (the laptop geometry the layout rules target) and 1920x1080.
"""

from __future__ import annotations

import os
import unittest

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

try:
    from PySide6 import QtWidgets
except Exception:  # pragma: no cover - PySide6 is an optional extra
    QtWidgets = None

SIZES = ((1366, 768), (1920, 1080))
MAX_LABEL_TEXT_LINES = 4          # the stale banner is 2 lines by design; the bug rendered ~20
MAX_LEAF_FRACTION_OF_WINDOW = 0.25


def _is_canvas(w) -> bool:
    """Matplotlib canvases are legitimately large -- they are the output, not chrome."""
    return type(w).__name__.startswith("FigureCanvas")


def _is_framework_chrome(w) -> bool:
    """Qt's own scaffolding -- scrollbars, scroll viewports, splitter handles.

    Their height legitimately tracks whatever they wrap, so a height budget is meaningless for
    them (they were the only false positives when this gate was first run). Excluded by CLASS and
    by Qt's internal ``qt_`` object-name prefix, not by a hand-listed set of instances."""
    return (isinstance(w, (QtWidgets.QScrollBar, QtWidgets.QSplitterHandle))
            or str(w.objectName()).startswith("qt_"))


def _is_container(w) -> bool:
    """Containers legitimately grow with their children; only LEAF widgets get a height budget."""
    if _is_canvas(w) or _is_framework_chrome(w):
        return True
    return isinstance(w, (QtWidgets.QMainWindow, QtWidgets.QTabWidget, QtWidgets.QScrollArea,
                          QtWidgets.QSplitter, QtWidgets.QGroupBox, QtWidgets.QStackedWidget,
                          QtWidgets.QDialog, QtWidgets.QMenuBar, QtWidgets.QStatusBar,
                          QtWidgets.QToolBar, QtWidgets.QFrame)) or bool(w.findChildren(QtWidgets.QWidget))


@unittest.skipIf(QtWidgets is None, "PySide6 not installed")
class GuiGeometrySanityTests(unittest.TestCase):
    """Every VISIBLE widget must have a sane size -- in the transient states too."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _pump(self, n=6):
        for _ in range(n):
            self.app.processEvents()

    def _build(self, size):
        from shaarp.desktop_app import build_main_window
        win = build_main_window()
        win.resize(*size)
        win.show()
        self._pump()
        return win

    def _offenders(self, win):
        """Return a list of human-readable violations across every visible widget."""
        bad = []
        win_h, win_w = win.height(), win.width()
        for w in win.findChildren(QtWidgets.QWidget):
            if not w.isVisible() or w.height() <= 0:
                continue
            label = f"{type(w).__name__}({(w.text()[:40] if isinstance(w, QtWidgets.QLabel) else w.objectName()) or '-'})"
            if w.width() > win_w + 2:
                bad.append(f"{label}: width {w.width()} > window {win_w}")
            if isinstance(w, QtWidgets.QLabel) and w.text():
                budget = MAX_LABEL_TEXT_LINES * w.fontMetrics().height() + 24
                if w.height() > budget:
                    bad.append(f"{label}: label height {w.height()} > {budget} "
                               f"({MAX_LABEL_TEXT_LINES} text lines)")
            elif not _is_container(w):
                if w.height() > MAX_LEAF_FRACTION_OF_WINDOW * win_h:
                    bad.append(f"{label}: leaf height {w.height()} > "
                               f"{MAX_LEAF_FRACTION_OF_WINDOW:.0%} of window ({win_h})")
        return bad

    # ---- states -------------------------------------------------------------------------

    @staticmethod
    def _si_page(win):
        return win.findChild(QtWidgets.QTabWidget).widget(0)

    def _show_stale_banner(self, page):
        """Reach the state that only exists AFTER an input changes -- where the bug lived.

        Nudging any tracked input marks the plot stale and reveals the banner; the banner is the
        widget that rendered 480 px tall, and it is invisible in the clean default state."""
        spins = page.findChildren(QtWidgets.QDoubleSpinBox)
        self.assertTrue(spins, "no spin boxes found on the SI page")
        spins[0].setValue(spins[0].value() + 1.0)
        self._pump()
        banners = [w for w in page.findChildren(QtWidgets.QLabel)
                   if w.isVisible() and "date" in w.text().lower() or
                   (w.isVisible() and "stale" in w.text().lower())]
        return banners

    def test_geometry_is_sane_in_the_default_state(self):
        for size in SIZES:
            with self.subTest(size=size):
                win = self._build(size)
                try:
                    self.assertEqual(self._offenders(win), [])
                finally:
                    win.close()

    def test_geometry_is_sane_while_the_stale_banner_is_shown(self):
        # THE REGRESSION FENCE for the 480 px banner: it is only visible in this transient state.
        for size in SIZES:
            with self.subTest(size=size):
                win = self._build(size)
                try:
                    page = self._si_page(win)
                    self._show_stale_banner(page)
                    self.assertEqual(self._offenders(win), [])
                finally:
                    win.close()

    def test_geometry_is_sane_after_a_run(self):
        for size in SIZES:
            with self.subTest(size=size):
                win = self._build(size)
                try:
                    page = self._si_page(win)
                    run = next(b for b in page.findChildren(QtWidgets.QPushButton)
                               if b.text() == "Update / Run")
                    run.click()
                    self._pump(10)
                    self.assertEqual(self._offenders(win), [])
                finally:
                    win.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
