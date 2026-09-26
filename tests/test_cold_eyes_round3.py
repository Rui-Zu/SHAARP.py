"""Round 3 of the cold-eyes GUI audit (2026-09-20): the remaining open items, closed.

These are the leftovers the first two rounds ranked below the blocking defects -- the ones that
waste a user's time rather than stop them outright, plus two long-standing register entries.

  * R19: the polar title named whatever row the layer EDITOR had selected, so picking the air
    half-space printed "point group inf inf m" over ZnO's own lobes.
  * R20: the schematic's colour key and assumption caption were spaced in DATA units, which shrink
    as the stack grows, so a five-layer stack overprinted them.
  * R21: the layer material list offered both "air" and the palette's "Air" case -- one medium,
    two rows.
  * A tensor-cell edit under a named case did not mark it edited, so re-selecting the case
    discarded the edit silently. Every other panel marked it.
  * An interior layer of 0 um ran for 12 s and returned numerical noise under "Run complete.";
    100,000 um is a 10 cm "thin film".
  * The wavelength note kept the previous material's text on a hidden label.
  * A semi-infinite row kept a disabled "0.0" thickness box beside its "no input" caption.
  * The layer-name placeholder promised an auto-label the app never writes.
  * Quick-pick chips said nothing about what they do.
  * The sweep group sat above Functionality and Case Study, which is where the README starts.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PySide6 import QtWidgets  # noqa: E402

from tests import gui_harness as gh  # noqa: E402

SI, ML = 0, 1


def _tabs(win):
    top = next(t for t in win.findChildren(QtWidgets.QTabWidget)
               if t.count() >= 2 and "SHAARP" in t.tabText(0))
    return top, top.widget(SI), top.widget(ML)


def _groups_in_order(page):
    area = page.findChild(QtWidgets.QScrollArea)
    body = area.widget()
    return [g.title().split("   —")[0].strip()
            for g in body.findChildren(QtWidgets.QGroupBox) if g.parent() is body]


def _layer_selector(page):
    return next(c for c in page.findChildren(QtWidgets.QComboBox)
                if c.count() and c.itemText(0).startswith("1:"))


class TheMaterialListOffersEachMediumOnce(unittest.TestCase):

    def test_air_appears_once(self):
        from shaarp.layer_stack import LAYER_MATERIAL_CHOICES

        airs = [c for c in LAYER_MATERIAL_CHOICES if c.strip().lower() == "air"]
        self.assertEqual(airs, ["air"], f"one medium, two rows: {airs}")

    def test_the_air_case_study_itself_still_exists(self):
        from shaarp.casestudy_materials import GUI_ML_CASES

        labels = [label for label, _key in GUI_ML_CASES]
        self.assertIn("Air", labels, "only the duplicate ROW was meant to go")


class ThePanelOrderFollowsTheDocumentedFlow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_functionality_and_case_study_come_before_the_sweep(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        _top, si, ml = _tabs(win)
        for page, tag in ((si, "si"), (ml, "ml")):
            order = _groups_in_order(page)
            self.assertEqual(order[0], "Wavelength Setting", f"{tag}: contract, first group")
            for name in ("Functionality", "Case Study and Examples", "Wavelength Scan Range"):
                self.assertIn(name, order, f"{tag}: {name} missing")
            self.assertLess(order.index("Functionality"), order.index("Wavelength Scan Range"), tag)
            self.assertLess(order.index("Case Study and Examples"),
                            order.index("Wavelength Scan Range"), tag)


class EditsAreNeverDiscardedSilently(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_a_tensor_edit_under_a_case_marks_it_edited(self):
        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        _top, si, _ml = _tabs(win)
        case = gh.combo_with_item(si, "Custom (use fields)")
        case.setCurrentText("KTP (100)")
        self.app.processEvents()
        group = next(g for g in si.findChildren(QtWidgets.QGroupBox)
                     if g.title().startswith("Case Study"))
        self.assertNotIn("edited", group.title(), "clean after selecting a case")
        cell = next(e for e in si.findChildren(QtWidgets.QLineEdit)
                    if e.toolTip() == TOOLTIPS["full_tensor"])
        cell.setText("9.5")
        cell.textEdited.emit("9.5")       # what typing fires; setText alone does not
        self.app.processEvents()
        self.assertIn("edited", group.title(),
                      "a tensor edit must mark the case, as every other panel does")


class DegenerateThicknessesAreRefused(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _run_with_thickness(self, value):
        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        top, _si, ml = _tabs(win)
        top.setCurrentIndex(ML)
        self.app.processEvents()
        spin = next(s for s in ml.findChildren(QtWidgets.QDoubleSpinBox)
                    if s.toolTip() == TOOLTIPS["thickness"])
        _layer_selector(ml).setCurrentIndex(1)
        self.app.processEvents()
        spin.setValue(value)
        self.app.processEvents()
        gh.click_update(ml, self.app)
        return win._gui_smoke_errors

    def test_zero_thickness_says_which_layer(self):
        errors = self._run_with_thickness(0.0)
        self.assertTrue(errors, "a zero-thickness film returned noise under 'Run complete.'")
        self.assertIn("thickness of 0", " ".join(errors))
        self.assertIn("Layer 2", " ".join(errors))

    def test_a_centimetre_slab_says_which_layer(self):
        errors = self._run_with_thickness(100000.0)
        self.assertTrue(errors, "a 10 cm 'thin film' ran without a word")
        self.assertIn("Layer 2", " ".join(errors))

    def test_a_real_thickness_still_runs(self):
        self.assertFalse(self._run_with_thickness(1.0), "a 1 um film must compute")


class ThePictureNamesWhatItDrew(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_the_polar_title_names_the_shg_source_not_the_selected_row(self):
        """R19. Fig 6 sources its SHG in ZnO (6mm); selecting the air half-space row must not
        retitle the plot with air's Curie group."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        win._gui_smoke_errors = []
        top, _si, ml = _tabs(win)
        top.setCurrentIndex(ML)
        gh.ml_case_combo(ml).setCurrentText("ZnO / Pt / Al2O3 (Fig 6, 1550 nm)")
        self.app.processEvents()
        selector = _layer_selector(ml)
        air_row = next(i for i in range(selector.count())
                       if "air" in selector.itemText(i).lower())
        selector.setCurrentIndex(air_row)
        self.app.processEvents()
        gh.click_update(ml, self.app)
        self.assertFalse(win._gui_smoke_errors, win._gui_smoke_errors)
        titles = [cv.figure._suptitle.get_text() for cv in ml.findChildren(FigureCanvasQTAgg)
                  if getattr(cv.figure, "_suptitle", None) is not None]
        joined = " | ".join(titles)
        self.assertIn("6mm", joined, "the SHG source's point group")
        self.assertNotIn("∞∞m", joined, "a passive half-space is not the source")

    def test_the_schematic_key_and_caption_do_not_overprint(self):
        """R20. Their spacing must not depend on how many layers the stack has."""
        from shaarp.shaarp_gui import build_schematic_figure

        for layers in ([("air", None), ("film", 1.0), ("air", None)],
                       [("air", None), ("a", 0.16), ("b", 0.2), ("c", 100.0), ("air", None)]):
            fig = build_schematic_figure(layers, theta_deg=0.0, assumption="Full Multiple Reflections")
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            FigureCanvasAgg(fig)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            key = [t for t in fig.axes[0].texts
                   if "fundamental" in t.get_text() or "SHG" in t.get_text()
                   or "bound source" in t.get_text()]
            caption = [t for t in fig.axes[0].texts
                       if "multiple reflections" in t.get_text().lower()]
            self.assertTrue(key and caption, "key and caption must both be drawn")
            key_bottom = min(t.get_window_extent(renderer).y0 for t in key)
            caption_top = max(t.get_window_extent(renderer).y1 for t in caption)
            self.assertLessEqual(caption_top, key_bottom + 1,
                                 f"{len(layers)}-layer stack: caption overprints the colour key")


class DeadAndMisleadingControlsAreGone(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_a_semi_infinite_row_hides_its_thickness_box(self):
        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        win.resize(1480, 940)
        win.show()
        self.app.processEvents()
        top, _si, ml = _tabs(win)
        top.setCurrentIndex(ML)
        self.app.processEvents()
        spin = next(s for s in ml.findChildren(QtWidgets.QDoubleSpinBox)
                    if s.toolTip() == TOOLTIPS["thickness"])
        selector = _layer_selector(ml)
        selector.setCurrentIndex(1)
        self.app.processEvents()
        self.assertTrue(spin.isVisibleTo(ml), "an interior layer has a thickness")
        selector.setCurrentIndex(0)
        self.app.processEvents()
        self.assertFalse(spin.isVisibleTo(ml),
                         "a semi-infinite medium kept a disabled 0.0 box with its label hidden")

    def test_the_layer_name_placeholder_matches_the_label_the_app_writes(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        _top, _si, ml = _tabs(win)
        name = next(e for e in ml.findChildren(QtWidgets.QLineEdit)
                    if (e.placeholderText() or "").startswith("(auto"))
        self.assertIn("film", name.placeholderText(),
                      "the old text promised a 'role: material' label that is never written")

    def test_quick_pick_chips_say_what_they_do(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        _top, si, _ml = _tabs(win)
        chips = [b for b in si.findChildren(QtWidgets.QPushButton)
                 if b.isCheckable() and b.text() == "45"]
        self.assertTrue(chips)
        for chip in chips:
            self.assertIn("Set the field", chip.statusTip())
            # NOT a tooltip: the audit/session identity of a chip falls back to its tooltip
            # before its text, so giving it one renames the widget
            self.assertEqual(chip.toolTip(), "", "a chip's tooltip is its identity; leave it empty")

    def test_the_wavelength_note_keeps_no_text_while_hidden(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        _top, si, _ml = _tabs(win)
        note = si.findChild(QtWidgets.QLabel, "wl_note_si")
        self.assertIsNotNone(note)
        case = gh.combo_with_item(si, "Custom (use fields)")
        for name in ("KTP (100)", "GaAs (dispersive) 0.21-12.40 um"):
            case.setCurrentText(name)
            case.textActivated.emit(name)
            self.app.processEvents()
            if not note.isVisible():
                self.assertEqual(note.text(), "",
                                 f"after {name}: a hidden note still held the previous text")


if __name__ == "__main__":
    unittest.main()
