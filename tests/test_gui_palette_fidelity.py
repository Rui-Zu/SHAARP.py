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
  * the SI case combo's curated rows == GUI_SI_GROUPS, in order, with the group headers present
    and DISABLED (headers are titles, not choices);
  * the ML case combo's film rows == the curated ML palette;
  * the ONE permitted extension -- the Dispersive group -- is exactly the shipped index tables,
    under its own disabled header, AFTER the curated palette;
  * EVERY selectable row resolves through the GUI's own resolution seam to a material that can
    actually be built.
A future session cannot silently reintroduce a non-original example, or break the label->key
resolution, without this going red.

Why the Dispersive group is fenced separately rather than folded into the curated lists: a
dispersive row is not a new example. It is a crystal ALREADY in the palette, with its linear optics
read from a published index table instead of one tabulated wavelength, so that a wavelength sweep
has something to move. Its membership is decided by shaarp.dispersion's shipped tables, not by an
editorial choice, and asserting it against that registry is the same fence the curated lists get --
a row cannot appear in it without a table (and therefore a citation) landing in the package first.
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

    @staticmethod
    def _dispersive_group():
        """(header text, [label]) for the shipped index tables -- the registry, not the widget."""
        from shaarp.casestudy_materials import gui_dispersive_group
        hdr, rows = gui_dispersive_group()
        return hdr.strip(), [label for label, _key in rows]

    @classmethod
    def _split_curated(cls, rows):
        """(rows before the Dispersive header, rows from it onward)."""
        hdr, _labels = cls._dispersive_group()
        for i, (text, _en) in enumerate(rows):
            if text.strip() == hdr:
                return rows[:i], rows[i:]
        return rows, []

    def test_si_combo_rows_are_exactly_the_curated_palette(self):
        from shaarp.casestudy_materials import GUI_SI_GROUPS
        rows, _disp = self._split_curated(self._builtin_rows(self._rows(self._si_case_combo())))
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

    def test_dispersive_rows_are_exactly_the_shipped_index_tables(self):
        """The one permitted extension, fenced as tightly as the palette it follows.

        A row may sit here only because a published index table (and therefore its citation) is in
        the package -- never because a session typed a name into a combo."""
        from tests.gui_harness import ml_case_combo
        hdr, labels = self._dispersive_group()
        self.assertTrue(labels, "no dispersive tables are shipped -- the group should be absent")
        tail_rows = {"N-layer stack (editor)", "Custom film (use fields)"}
        for tab, combo in (("SI", self._si_case_combo()), ("ML", ml_case_combo(self.ml_page))):
            with self.subTest(tab=tab):
                _curated, disp = self._split_curated(self._builtin_rows(self._rows(combo)))
                self.assertTrue(disp, f"{tab}: the Dispersive group is missing from the case combo")
                self.assertEqual(disp[0][0].strip(), hdr)
                self.assertFalse(disp[0][1],
                                 f"{tab}: the Dispersive header must be a title, not a choice")
                shown = [t.strip() for t, en in disp[1:] if en]
                self.assertEqual(shown[:len(labels)], labels,
                                 f"{tab}: the Dispersive rows are not the shipped index tables")
                self.assertEqual(set(shown[len(labels):]) - tail_rows, set(),
                                 f"{tab}: an unrecognised row follows the Dispersive group")

    def test_every_selectable_case_row_resolves_to_a_buildable_material(self):
        # THE CS-1 REGRESSION FENCE: a row whose label does not resolve to a registry key looks
        # perfectly fine in the combo and fails only when the user clicks Update.
        from shaarp.casestudy_materials import build_casestudy_material, resolve_case_label
        from tests.gui_harness import ml_film_labels
        curated, _disp = self._split_curated(self._builtin_rows(self._rows(self._si_case_combo())))
        si_rows = [t.strip() for t, en in curated
                   if en and t.strip() != "Custom (use fields)"]
        for label in si_rows + ml_film_labels(self.ml_page):
            with self.subTest(label=label):
                key = resolve_case_label(label)
                self.assertTrue(key, f"{label!r} does not resolve to a registry key")
                material = build_casestudy_material(key)
                self.assertTrue(getattr(material, "structure", None) is not None,
                                f"{label!r} -> {key!r} did not build a usable material")

    def test_every_dispersive_row_builds_through_the_gui_resolution_seam(self):
        """Same fence, other resolver: a dispersive label never reaches the registry by key -- it is
        resolved by material_for_label, which is the seam every GUI read site already goes through.
        A row whose table is missing or misnamed looks perfectly fine until the user clicks Update."""
        from shaarp.dispersion import load_shipped_table
        from shaarp.layer_stack import material_for_label
        _hdr, labels = self._dispersive_group()
        for label in labels:
            with self.subTest(label=label):
                low, high = load_shipped_table(label).range_um
                # a wavelength the table can answer at BOTH harmonics where it reaches that far:
                # eps(2w) is read at lam/2, so a table runs out at the blue end twice as fast.
                lam = 0.5 * (min(2.0 * low, high) + high)
                material = material_for_label(label, lam)
                self.assertTrue(getattr(material, "structure", None) is not None,
                                f"{label!r} did not build a usable material at {lam} um")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
