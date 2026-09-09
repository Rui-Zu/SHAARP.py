"""The input column must open wide enough for its own controls.

WHY THIS EXISTS. The column opened at a hardcoded 500 px while the single-interface controls need
579 and the multilayer ones 611. Every launch therefore came up with a permanent horizontal
scrollbar and the right-hand spin boxes cut off, at every window size including maximized -- and
that is the state captured in all three screenshots the README shows. It survived because the only
geometry fence asserted the window's minimum width stays small, which this does not violate: a
scroll area that scrolls is not, in itself, a layout error.

What drives the width is one QFormLayout. A form's minimum is the SUM of its two column maxima, so
Polarimetry Settings is wide because its widest label ("analyzer-polarizer offset (deg)") and its
widest field (the incidence row: spin plus slider plus quick-angle buttons) sit in DIFFERENT rows
and are added together.

The fix measures the content instead of guessing, and has to run after the pages are built: called
earlier, inside the page builder, minimumSizeHint still reports its pre-layout value of about 505,
which is exactly why a hardcoded 500 looked adequate for so long.

Below roughly 1280 px of window the scrollbar legitimately returns -- there is not enough room for
both columns -- so this asserts the sizes a real user sees, not every size.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

# Sizes that matter: the app's own default, and the two the doc screenshots are taken at.
REAL_WINDOW_SIZES = ((1480, 940), (1600, 1000), (1480, 920))


class InputColumnOpensWideEnough(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _pages(self, win):
        top = next(t for t in win.findChildren(QtWidgets.QTabWidget)
                   if t.count() >= 2 and "SHAARP" in t.tabText(0))
        return top, [(i, top.tabText(i)) for i in range(top.count())]

    def _input_scroll(self, page):
        scrolls = [s for s in page.findChildren(QtWidgets.QScrollArea) if s.widget() is not None]
        self.assertTrue(scrolls, "no scroll area on this page")
        return min(scrolls, key=lambda s: s.width())

    def test_no_horizontal_scrollbar_at_the_sizes_users_and_screenshots_use(self):
        from shaarp.desktop_app import MODERN_QSS, build_main_window

        win = build_main_window()
        win.setStyleSheet(MODERN_QSS)
        offenders = []
        for w, h in REAL_WINDOW_SIZES:
            win.resize(w, h)
            win.show()
            for _ in range(14):
                self.app.processEvents()
            top, tabs = self._pages(win)
            for idx, name in tabs:
                top.setCurrentIndex(idx)
                for _ in range(10):
                    self.app.processEvents()
                scroll = self._input_scroll(top.widget(idx))
                need = scroll.widget().minimumSizeHint().width()
                have = scroll.viewport().width()
                if have < need:
                    offenders.append(
                        f"{w}x{h} {name}: viewport {have} < content {need} "
                        f"(controls clipped, horizontal scrollbar shown)")
        self.assertEqual(offenders, [], "input column too narrow for its own controls:\n  "
                                        + "\n  ".join(offenders))

    def test_the_fit_is_measured_from_the_content_not_hardcoded(self):
        """Falsifiability: the helper must return a width above the old fixed 500."""
        from shaarp.desktop_app import _fit_input_columns, build_main_window

        win = build_main_window()
        widths = _fit_input_columns(win)
        self.assertTrue(widths, "no input columns were fitted")
        self.assertTrue(all(w > 500 for w in widths),
                        f"fit fell back to the old hardcoded width: {widths}")

    def test_the_column_stays_capped_so_the_plot_area_survives(self):
        """The fit must not hand the whole window to the inputs on a small screen."""
        from shaarp.desktop_app import _fit_input_columns, build_main_window

        win = build_main_window()
        widths = _fit_input_columns(win)
        self.assertTrue(all(w <= 700 for w in widths), f"input column too greedy: {widths}")
        # and the window itself must still fit a laptop -- the pre-existing geometry contract
        self.assertLessEqual(win.minimumSizeHint().width(), 1100)


if __name__ == "__main__":
    unittest.main()
