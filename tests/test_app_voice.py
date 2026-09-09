"""The application must describe itself the way the documentation does.

WHY THIS EXISTS. A documentation re-voicing pass (2026-09-04) rewrote the README, the docs site,
the release notes and the notebooks, but not the GUI's own strings. For days afterwards the app
introduced itself as "a validated Python port of the Mathematica packages" in four places -- the
banner under the logo, the window title, the User Guide welcome page and the About box -- and
every README screenshot showed that banner. The landing page said "a new package" directly above
a picture saying "a port", and nothing failed, because the only fences on those strings asserted
that three unrelated words were still present.

So the fence is written the other way round: it enumerates the words the project has retired and
fails if any of them reaches a string a user can read. Scanning the module's string literals,
rather than a hand-listed set of constants, is deliberate -- the window title and the About dialog
body had NO fence at all, and a list of constants would have missed them again. Anything new that
someone adds to this file is covered the moment it is written.

Internal prose is exempt: docstrings and comments may say whatever is accurate about provenance.
This is about what the product says out loud.
"""
from __future__ import annotations

import ast
import os
import pathlib
import re
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "shaarp" / "desktop_app.py"

# Words retired from shipped prose. SHAARP.py is its own free package that carries both published
# SHAARP methods; it is not framed as a port, a reproduction, or a replacement of anything.
RETIRED = re.compile(
    r"\b("
    r"python port|validated python|a port of|this port|the port\b"
    r"|value-by-value|byte-exact|un-fakeable|claim by claim"
    r"|faithful reproduction|re-implementation"
    r")\b",
    re.IGNORECASE,
)


def _user_facing_string_literals(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every string constant in the module except docstrings.

    f-strings contribute their literal segments, which is what carries the prose in the About
    dialog and the window title.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            out.append((node.lineno, node.value))
    return out


class AppSpeaksInTheProjectsVoice(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # build_main_window needs a QApplication; without one Qt aborts the interpreter outright
        # rather than raising, which shows up as a dead run with no failure message.
        from PySide6 import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_no_retired_vocabulary_in_any_user_facing_string(self):
        offenders = []
        for lineno, text in _user_facing_string_literals(SOURCE):
            m = RETIRED.search(text)
            if m:
                snippet = text[max(0, m.start() - 50):m.start() + 60].replace("\n", " ")
                offenders.append(f"desktop_app.py:{lineno}: {m.group(0)!r} in ...{snippet}...")
        self.assertEqual(
            offenders, [],
            "the app must not describe itself with retired vocabulary:\n  " + "\n  ".join(offenders))

    def test_the_fence_actually_catches_the_wording_it_was_written_for(self):
        """Falsifiability: the exact banner text that shipped must be rejected."""
        shipped_regression = ("Validated Python port of SHAARP.si + SHAARP.ml "
                              "(Zu, Wang, Weber, Saha, Chen & Gopalan).")
        self.assertIsNotNone(RETIRED.search(shipped_regression))

    def test_window_title_names_the_package_and_both_methods_only(self):
        from shaarp.desktop_app import build_main_window

        title = build_main_window().windowTitle()
        self.assertIn("SHAARP.py", title)
        self.assertIn("SHAARP.si", title)
        self.assertIn("SHAARP.ml", title)
        self.assertIsNone(RETIRED.search(title), f"window title carries retired wording: {title!r}")

    def test_banner_and_user_guide_say_what_the_package_is(self):
        from shaarp.desktop_app import BRANDING_HTML, USER_GUIDE_HTML

        for name, html in (("BRANDING_HTML", BRANDING_HTML), ("USER_GUIDE_HTML", USER_GUIDE_HTML)):
            self.assertIsNone(RETIRED.search(html), f"{name} carries retired wording")
        # the banner still credits the authors and asks for acknowledgment
        for need in ("SHAARP.py", "Gopalan", "acknowledge"):
            self.assertIn(need, BRANDING_HTML)
        # the guide still points at both original repositories and both papers
        self.assertIn("github.com/Rui-Zu/SHAARP", USER_GUIDE_HTML)
        self.assertIn("github.com/bzw133/SHAARP.ml", USER_GUIDE_HTML)
        self.assertIn("npj Comput", USER_GUIDE_HTML)
        self.assertIn("DE-SC0020145", USER_GUIDE_HTML)

    def test_per_run_status_line_reports_a_check_not_a_provenance_claim(self):
        from shaarp.desktop_app import _friendly_validation_status

        for raw in ("staged_python_not_fully_mathematica_validated",
                    "mathematica_validated",
                    "reference_unavailable"):
            msg = _friendly_validation_status(raw)
            self.assertTrue(msg.startswith("Checked:"), f"{raw} -> {msg!r}")
            self.assertIsNone(RETIRED.search(msg))
            self.assertNotIn("_", msg, "the raw workflow tag belongs in the tooltip, not the label")


if __name__ == "__main__":
    unittest.main()
