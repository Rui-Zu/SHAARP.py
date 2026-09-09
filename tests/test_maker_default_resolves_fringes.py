"""The default Maker scan step must resolve the default preset's fringes.

WHY THIS EXISTS. The step defaulted to 0.5 deg while the shipped default system preset
(Quartz + Au, the paper's Fig 4) has a fringe spacing of 0.560 deg over 30-40 deg. That is 1.12
samples per fringe, below the Nyquist limit of 2, so the app's out-of-the-box view of its own
documented example was aliased: the curve read as noise and its maxima sat at angles the physics
does not put them at. Nothing failed, because an aliased curve is a perfectly well-formed array.

The spacing is a property of the preset (121.2 um quartz at 800 nm), so it is pinned here as a
measured constant rather than recomputed on every run: deriving it needs a fine sweep, which is
slow, and if the preset ever changes this fence should be re-measured deliberately rather than
silently tracking the new value.

    measured 2026-09-09, 30-45 deg at 0.02 deg sampling: 22 maxima, median spacing 0.560 deg
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

FRINGE_SPACING_DEG = 0.560          # Quartz + Au (Fig 4, 800 nm), 30-45 deg
NYQUIST_SAMPLES_PER_FRINGE = 2.0
COMFORTABLE_SAMPLES_PER_FRINGE = 4.0


class MakerDefaultStepResolvesTheDefaultPreset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _default_step(self):
        """The theta-step spin on the multilayer tab's Maker Fringes Scan Range."""
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        top = next(t for t in win.findChildren(QtWidgets.QTabWidget)
                   if t.count() >= 2 and "SHAARP" in t.tabText(0))
        ml = top.widget(1)
        group = next(g for g in ml.findChildren(QtWidgets.QGroupBox)
                     if g.title().startswith("Maker Fringes Scan Range"))
        form = None
        for child in group.findChildren(QtWidgets.QWidget):
            if isinstance(child.layout(), QtWidgets.QFormLayout):
                form = child.layout()
                break
        self.assertIsNotNone(form, "Maker Fringes Scan Range lost its form layout")
        for row in range(form.rowCount()):
            label = form.itemAt(row, QtWidgets.QFormLayout.LabelRole)
            if label and isinstance(label.widget(), QtWidgets.QLabel) \
                    and "step" in label.widget().text().lower():
                field = form.itemAt(row, QtWidgets.QFormLayout.FieldRole)
                container = field.widget() or field.layout()
                spins = (container.findChildren(QtWidgets.QAbstractSpinBox)
                         if hasattr(container, "findChildren") else [])
                self.assertTrue(spins, "no spin box on the theta-step row")
                return float(spins[0].value())
        self.fail("no theta-step row in the Maker Fringes Scan Range group")

    def test_default_step_is_above_nyquist_for_the_default_preset(self):
        step = self._default_step()
        samples = FRINGE_SPACING_DEG / step
        self.assertGreater(
            samples, NYQUIST_SAMPLES_PER_FRINGE,
            f"default step {step} deg gives {samples:.2f} samples per fringe on the shipped "
            f"Quartz + Au preset (spacing {FRINGE_SPACING_DEG} deg) -- the default view aliases")
        self.assertGreaterEqual(
            samples, COMFORTABLE_SAMPLES_PER_FRINGE,
            f"default step {step} deg only just clears Nyquist ({samples:.2f} samples per fringe); "
            f"peak positions and heights are still distorted at that sampling")

    def test_the_fence_would_have_failed_on_the_old_default(self):
        """Falsifiability: 0.5 deg, the value that shipped, must be rejected."""
        self.assertLess(FRINGE_SPACING_DEG / 0.5, NYQUIST_SAMPLES_PER_FRINGE)


if __name__ == "__main__":
    unittest.main()
