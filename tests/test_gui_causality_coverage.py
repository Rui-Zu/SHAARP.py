"""GUI CAUSALITY COVERAGE GATE.

the author's directive: "look through every functionality to make sure all GUI features work well" --
as an ENFORCED enumeration, not a hand-picked sample. This gate walks the LIVE widget tree of
both tabs plus the window chrome and asserts that every interactive widget matches exactly one
rule of the coverage registry (tests/gui_harness.py COVERAGE_RULES):

  covered -- a causality/correctness test drives it (note names the test/step)
  exempt -- deliberately not causality-driven, with the reason on record
  todo -- an enumerated gap; THE GATE FAILS while any remains (the ratchet that forces
             Phase C to closure and keeps future widgets from shipping untested)

A brand-new widget lands UNMATCHED and fails BY NAME. The F40 class -- a dead control passing
every gate -- cannot silently regrow.
"""

import os
import unittest
from collections import defaultdict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets
    HAVE_QT = True
except Exception:  # pragma: no cover
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class GuiCausalityCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from shaarp.desktop_app import build_main_window

        from . import gui_harness as gh

        cls.gh = gh
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = build_main_window()
        cls.walk = list(gh.walk_all(cls.win))

    def test_every_interactive_widget_is_covered_or_exempt(self):
        unmatched, todos = [], []
        for ident, _w in self.walk:
            status, note = self.gh.coverage_status(ident)
            if status == "UNMATCHED":
                unmatched.append(ident)
            elif status == "todo":
                todos.append(f"{ident}  ->  {note}")
        self.assertFalse(
            unmatched,
            "UNCOVERED WIDGETS (add a causality step or an exempt rule WITH a reason):\n  "
            + "\n  ".join(unmatched))
        self.assertFalse(
            todos,
            f"{len(todos)} ENUMERATED COVERAGE GAPS remain (the Phase-C worklist):\n  "
            + "\n  ".join(sorted(set(todos))))

    def test_no_dead_rules(self):
        """Registry hygiene: every rule must still match at least one live widget -- a rule
        matching nothing means the widget was removed/renamed and the registry is stale."""
        import re

        idents = [ident for ident, _w in self.walk]
        dead = []
        for pattern, _status, _note in self.gh.COVERAGE_RULES:
            if not any(re.match(pattern, ident) for ident in idents):
                dead.append(pattern)
        self.assertFalse(dead, "STALE REGISTRY RULES (match no live widget):\n  "
                         + "\n  ".join(dead))

    def test_walk_is_complete_enough(self):
        """Sanity floor: the walk must see the known scale of the GUI (233 page widgets at
        F42 time); a collapse here means the walker itself broke (e.g. the isinstance-vs-
        name regression that silently dropped every _NumBox spin)."""
        self.assertGreaterEqual(len(self.walk), 200,
                                f"widget walk collapsed to {len(self.walk)} entries")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
