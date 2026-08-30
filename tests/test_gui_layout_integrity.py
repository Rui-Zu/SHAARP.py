"""Layout-integrity harness for the merged SHAARP.py desktop GUI.

WHY THIS EXISTS: every other GUI test is pure-function or structural --
none drive the REAL window, click Update, and assert that nothing is clipped or
overflowing. So a whole class of bugs (figures not fitting, the 3D schematic clipped
after Update, single plots clipped-with-scroll, the window minimum width ballooning so
the geometry panel is cut on a normal laptop) was invisible to automation and only a
human, manually, could catch it -- which is exactly what happened, repeatedly.

This harness drives the actual window (build_main_window + MODERN_QSS, offscreen) across
a matrix of {SI, ML} x {every Functionality} x {laptop, maximized (+small for the
schematic-heavy cells)} x {before / after Update}, and asserts, from DATA (not eyeballed
thumbnails):

  1. window.minimumSizeHint().width() <= 1100 (no min-width bloat -> geometry panel fits laptops)
  2. the tab page is not wider than the window (no horizontal overflow)
  3. every visible output canvas lies within the window content rect (no widget clipped off-screen)
  4. single-plot tabs (Maker / Fresnel) FIT (vertical scrollbar.maximum()==0) -- never clipped-with-scroll;
     the polar grid must also fit when maximized
  5. 2D output figures (polar / Maker / Fresnel / 2D schematic) have a clean border (edge-ink fraction
     low) -- a validated proxy for "no title/label/axis clipped inside the canvas"
  6. the 3D schematic canvas stays NEAR-SQUARE (mplot3d clips its labels in a wide-short canvas; a
     near-square canvas + the within-window check is the reliable 3D guard)

Reliability notes from detector validation: edge-ink cleanly separates 2D clean (~0.01) from clipped
(~0.12); it does NOT separate the 3D cases (clean 0.023 vs clipped 0.037), and get_tightbbox misses 3D
tick-label clips entirely -- hence the structural near-square guard for 3D rather than a pixel/bbox check.
"""

from __future__ import annotations

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import numpy as np
    from PySide6 import QtCore, QtWidgets
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    HAVE_QT = True
except ImportError:  # pragma: no cover - environment without Qt
    HAVE_QT = False


LAPTOP = (1366, 768)
MAXIMIZED = (1920, 1080)
SMALL = (1280, 720)

# Compute-heavy modes (a real sweep/solve runs on Update) -- keep their size list short.
HEAVY = {"SHG Simulation", "Maker Fringes", "Fresnel Coefficients"}
# Single-plot output tabs that MUST fit the viewport (never clipped-with-scroll).
SINGLE_PLOT = {"Maker Fringes", "Fresnel Coefficients"}

EDGE_INK_MAX = 0.06       # 2D figure border-ink fraction (clean ~0.01, clipped ~0.12)
GEOM_TOL = 4              # px tolerance for widget-within-window
THREED_ASPECT_MAX = 1.7  # 3D canvas width / height upper bound (near-square so mplot3d renders fully)


@unittest.skipUnless(HAVE_QT, "PySide6 / matplotlib QtAgg not installed")
class GuiLayoutIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg", force=False)
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # ---- helpers -------------------------------------------------------------
    def _pump(self, n=8):
        for _ in range(n):
            self.app.processEvents()
            time.sleep(0.02)

    def _build(self, size):
        from shaarp.desktop_app import MODERN_QSS, build_main_window

        win = build_main_window()
        win.setStyleSheet(MODERN_QSS)
        win.resize(*size)
        win.show()
        self._pump()
        top = next(t for t in win.findChildren(QtWidgets.QTabWidget)
                   if t.count() >= 2 and "SHAARP" in t.tabText(0))
        return win, top

    @staticmethod
    def _functionality_combo(page):
        return next(c for c in page.findChildren(QtWidgets.QComboBox)
                    if c.findText("SHG Simulation") >= 0)

    @staticmethod
    def _case_combo(page):
        for c in page.findChildren(QtWidgets.QComboBox):
            if c.findText("Custom (use fields)") >= 0:
                return c
        return None

    def _set_case(self, page, name_substr):
        c = self._case_combo(page)
        if c is None:
            return
        for i in range(c.count()):
            if name_substr.lower() in c.itemText(i).lower():
                c.setCurrentIndex(i)
                self._pump()
                return

    def _click_update(self, page, heavy):
        btn = next(b for b in page.findChildren(QtWidgets.QPushButton) if "Update" in b.text())
        btn.click()
        budget = 18.0 if heavy else 6.0
        t0 = time.time()
        while time.time() - t0 < budget:
            self.app.processEvents()
            time.sleep(0.03)
            if time.time() - t0 > (10.0 if heavy else 2.0):
                break

    @staticmethod
    def _edge_ink_fraction(canvas, border=2, bg_thresh=250):
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        rgb = buf[..., :3]
        ink = rgb.min(axis=2) < bg_thresh
        frame = np.zeros(ink.shape, bool)
        frame[:border, :] = frame[-border:, :] = frame[:, :border] = frame[:, -border:] = True
        return float(ink[frame].mean())

    @staticmethod
    def _is_3d(canvas):
        return any(getattr(ax, "name", "") == "3d" for ax in canvas.figure.axes)

    @staticmethod
    def _enclosing_scrollarea(widget):
        p = widget.parentWidget()
        while p is not None:
            if isinstance(p, QtWidgets.QScrollArea):
                return p
            p = p.parentWidget()
        return None

    def _check_cell(self, win, top, tab_index, label):
        """Run every layout-integrity assertion for the current window state."""
        # (1) window minimum width stays laptop-friendly
        self.assertLessEqual(win.minimumSizeHint().width(), 1100,
                             f"{label}: window min width too large (geometry panel would clip)")
        page = top.widget(tab_index)
        # (2) no horizontal page overflow
        self.assertLessEqual(page.width(), win.width() + GEOM_TOL,
                             f"{label}: tab page wider than the window")

        canvases = [c for c in page.findChildren(FigureCanvasQTAgg)
                    if c.isVisible() and c.width() > 60 and c.height() > 60]
        self.assertTrue(canvases, f"{label}: no visible output canvas found")
        for c in canvases:
            # The rendered figure MUST be sized to its canvas. Swapping canvas.figure on Update does
            # not auto-resize the new figure, so a stale/oversized figsize renders clipped to the
            # canvas's top-left (the real cause of the 3D "doesn't rescale after Update" clip). Assert
            # figure pixel size == canvas size.
            fig = c.figure
            fpx_w = fig.get_size_inches()[0] * fig.get_dpi()
            fpx_h = fig.get_size_inches()[1] * fig.get_dpi()
            self.assertLess(abs(fpx_w - c.width()), 48,
                            f"{label}: figure px-width {fpx_w:.0f} != canvas {c.width()} (oversized -> clipped)")
            self.assertLess(abs(fpx_h - c.height()), 48,
                            f"{label}: figure px-height {fpx_h:.0f} != canvas {c.height()} (oversized -> clipped)")
            scroller = self._enclosing_scrollarea(c)
            if scroller is not None:
                # A canvas inside a QScrollArea is MEANT to exceed the viewport vertically (that's the
                # scroll). What must hold: it must not overflow HORIZONTALLY (width tracks the viewport),
                # and the scroll area itself must sit within the window. Vertical fit is the separate
                # scrollbar-intent check below.
                self.assertLessEqual(c.width(), scroller.viewport().width() + GEOM_TOL,
                                     f"{label}: a {'3D' if self._is_3d(c) else '2D'} canvas overflows its "
                                     f"scroll viewport horizontally ({c.width()} vs {scroller.viewport().width()})")
                stl = scroller.mapTo(win, QtCore.QPoint(0, 0))
                sbr = scroller.mapTo(win, QtCore.QPoint(scroller.width(), scroller.height()))
                self.assertTrue(
                    stl.x() >= -GEOM_TOL and stl.y() >= -GEOM_TOL
                    and sbr.x() <= win.width() + GEOM_TOL and sbr.y() <= win.height() + GEOM_TOL,
                    f"{label}: an output scroll area extends outside the window")
            else:
                tl = c.mapTo(win, QtCore.QPoint(0, 0))
                br = c.mapTo(win, QtCore.QPoint(c.width(), c.height()))
                # (3) a non-scrolled canvas (the schematics) must lie within the window content rect
                self.assertTrue(
                    tl.x() >= -GEOM_TOL and tl.y() >= -GEOM_TOL
                    and br.x() <= win.width() + GEOM_TOL and br.y() <= win.height() + GEOM_TOL,
                    f"{label}: a {'3D' if self._is_3d(c) else '2D'} canvas extends outside the window "
                    f"(x[{tl.x()},{br.x()}] y[{tl.y()},{br.y()}] vs {win.width()}x{win.height()})")
            if self._is_3d(c):
                # (6) 3D schematic must stay near-square so mplot3d renders the whole box + labels
                self.assertLessEqual(c.width(), THREED_ASPECT_MAX * c.height() + GEOM_TOL,
                                     f"{label}: 3D schematic canvas is wide-short ({c.width()}x{c.height()}) "
                                     "-> mplot3d will clip its labels")
            else:
                # (5) 2D figure must not have content jammed against / past the border
                frac = self._edge_ink_fraction(c)
                self.assertLessEqual(frac, EDGE_INK_MAX,
                                     f"{label}: 2D figure content clipped at the border (edge-ink {frac:.3f})")

        # (4) scrollbar intent on the active output tab
        out_tabs = [t for t in page.findChildren(QtWidgets.QTabWidget) if t is not top]
        if out_tabs:
            cur = out_tabs[0].currentWidget()
            if isinstance(cur, QtWidgets.QScrollArea):
                cur_label = out_tabs[0].tabText(out_tabs[0].currentIndex())
                fits = cur.verticalScrollBar().maximum() == 0
                if cur_label in ("Fresnel Coefficients", "Maker Fringes"):
                    self.assertTrue(fits, f"{label}: single-plot '{cur_label}' tab does not fit "
                                          "(clipped-with-scroll instead of shrinking)")
                elif cur_label == "Polar Plots" and win.height() >= 1000:
                    self.assertTrue(fits, f"{label}: polar grid does not fit even when maximized")

    # ---- the matrix ----------------------------------------------------------
    def _cells(self):
        """(tab_index, functionality, size, do_update, case_substr) tuples."""
        from shaarp.shaarp_gui import ML_FUNCTIONALITY_DISPLAY, SI_FUNCTIONALITY_DISPLAY

        tabs = [(0, list(SI_FUNCTIONALITY_DISPLAY)), (1, list(ML_FUNCTIONALITY_DISPLAY))]
        for tab_index, funcs in tabs:
            # startup (no Update) once per tab at laptop size
            yield (tab_index, "SHG Simulation", LAPTOP, False, None)
            for func in funcs:
                heavy = func in HEAVY
                sizes = [MAXIMIZED, LAPTOP]
                if func == "SHG Simulation":
                    sizes = [MAXIMIZED, LAPTOP, SMALL]
                for size in sizes:
                    yield (tab_index, func, size, True, None)
            # one real case study after Update (exercises populate + after-Update figsize)
            yield (tab_index, "SHG Simulation", MAXIMIZED, True, "LiNbO3")

    def test_layout_integrity_matrix(self):
        seen = 0
        for (tab_index, func, size, do_update, case) in self._cells():
            label = (f"{'SI' if tab_index == 0 else 'ML'} / {func} / {size[0]}x{size[1]} / "
                     f"{'after-update' if do_update else 'startup'}"
                     f"{(' / ' + case) if case else ''}")
            with self.subTest(cell=label):
                win, top = self._build(size)
                try:
                    top.setCurrentIndex(tab_index)
                    self._pump()
                    page = top.widget(tab_index)
                    combo = self._functionality_combo(page)
                    if combo.findText(func) >= 0:
                        combo.setCurrentIndex(combo.findText(func))
                        self._pump()
                    if case:
                        self._set_case(page, case)
                    if do_update:
                        self._click_update(page, heavy=(func in HEAVY))
                    else:
                        self._pump()
                    self._check_cell(win, top, tab_index, label)
                    seen += 1
                finally:
                    win.close()
                    win.deleteLater()
                    self._pump(2)
        self.assertGreaterEqual(seen, 20, "expected the full matrix to run")


if __name__ == "__main__":
    unittest.main()
