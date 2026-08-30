"""PALETTE FIDELITY AS A GATE.

the author's rule for the case-study lists: *"strictly confined to examples either presented in the paper
or presented in the GUI examples with corresponding wavelength"*. CS-1 curated the lists to exactly
that (GUI_SI_GROUPS / GUI_ML_GROUPS / GUI_ML_CASES), and test_casestudy_materials already fences the
CONSTANTS' contents.

What was never fenced is the link between those constants and the WIDGETS the user actually sees:
nothing asserted that the live combos are built from the curated palettes, or that every selectable
row still resolves to a registry material. That gap is not hypothetical -- it is precisely the CS-1
defect class, where a combo carried the right display labels while a read site still passed the raw
LABEL to build_casestudy_material and every SI case Update failed silently.

So this file drives the BUILT WINDOW and asserts:
  * the SI case combo's selectable rows == GUI_SI_GROUPS, in order, with the group headers present
    and DISABLED (headers are titles, not choices);
  * the ML case combo's film rows == the curated ML palette;
  * EVERY selectable row resolves through resolve_case_label to a registry key that
    build_casestudy_material can actually build.
A future session cannot silently reintroduce a non-original example, or break the label->key
resolution, without this going red.
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


@unittest.skipIf(QtWidgets is None, "PySide6 not installed")
class GuiPaletteFidelityTests(unittest.TestCase):
    """The visible case lists must BE the curated original palettes."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from shaarp.desktop_app import build_main_window
        cls.win = build_main_window()
        cls.win.resize(1366, 768)
        cls.win.show()
        for _ in range(6):
            cls.app.processEvents()
        tabs = cls.win.findChild(QtWidgets.QTabWidget)
        cls.si_page, cls.ml_page = tabs.widget(0), tabs.widget(1)

    @classmethod
    def tearDownClass(cls):
        cls.win.close()

    @staticmethod
    def _rows(combo):
        """(text, enabled) for every row, so header rows can be told from choices."""
        model = combo.model()
        out = []
        for i in range(combo.count()):
            item = model.item(i) if hasattr(model, "item") else None
            enabled = True if item is None else bool(item.isEnabled())
            out.append((combo.itemText(i), enabled))
        return out

    def _si_case_combo(self):
        from tests.gui_harness import combo_with_item
        return combo_with_item(self.si_page, "Custom (use fields)")

    @staticmethod
    def _builtin_rows(rows):
        """The curated palette is everything BEFORE the user's 'My Materials' section."""
        from shaarp.user_materials import USER_SECTION_HEADER
        out = []
        for t, en in rows:
            if t == USER_SECTION_HEADER:
                break
            out.append((t, en))
        return out

    def test_si_combo_rows_are_exactly_the_curated_palette(self):
        from shaarp.casestudy_materials import GUI_SI_GROUPS
        rows = self._builtin_rows(self._rows(self._si_case_combo()))
        headers = [t for t, en in rows if not en]
        choices = [t.strip() for t, en in rows if en and t.strip() != "Custom (use fields)"]
        self.assertEqual(choices, [label for _hdr, entries in GUI_SI_GROUPS for label, _key in entries],
                         "SI case combo no longer matches the curated original palette")
        self.assertEqual([h.strip() for h in headers],
                         [hdr.strip() for hdr, _entries in GUI_SI_GROUPS],
                         "SI group headers missing/reordered (they must be present and disabled)")

    def test_ml_combo_film_rows_are_exactly_the_curated_palette(self):
        from shaarp.casestudy_materials import GUI_ML_CASES
        from tests.gui_harness import ml_film_labels
        self.assertEqual(ml_film_labels(self.ml_page), [label for label, _key in GUI_ML_CASES],
                         "ML film rows no longer match the curated original palette")

    def test_every_selectable_case_row_resolves_to_a_buildable_material(self):
        # THE CS-1 REGRESSION FENCE: a row whose label does not resolve to a registry key looks
        # perfectly fine in the combo and fails only when the user clicks Update.
        from shaarp.casestudy_materials import build_casestudy_material, resolve_case_label
        from tests.gui_harness import ml_film_labels
        si_rows = [t.strip() for t, en in self._builtin_rows(self._rows(self._si_case_combo()))
                   if en and t.strip() != "Custom (use fields)"]
        for label in si_rows + ml_film_labels(self.ml_page):
            with self.subTest(label=label):
                key = resolve_case_label(label)
                self.assertTrue(key, f"{label!r} does not resolve to a registry key")
                material = build_casestudy_material(key)
                self.assertTrue(getattr(material, "structure", None) is not None,
                                f"{label!r} -> {key!r} did not build a usable material")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
