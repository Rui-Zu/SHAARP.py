"""Every shipped Mathematica exporter resolves its paths from the checkout, not from one machine.

The exporters regenerate the reference JSON the suite compares against. They used to carry
hard-coded absolute paths into a specific user's directory tree, which made them unusable to
anyone else and leaked that layout into the published package. They now share one preamble
(``benchmarks/wolfram_paths.PREAMBLE``) defining ``SHAARPPaths`Ref`` / ``Repo`` / ``ML`` / ``SI``.

The preamble is duplicated into 100+ files because a Wolfram script has no import mechanism that
works before its own paths are known -- so these checks are what keep the copies identical and
prevent an absolute path creeping back in.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

from benchmarks.wolfram_paths import PREAMBLE

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"

# Absolute paths that are legitimately machine-global rather than personal.
ALLOWED_ABSOLUTE = ("C:/Program Files/Wolfram Research", "C:\\Program Files\\Wolfram Research")
_ABSOLUTE = re.compile(r'"([A-Za-z]:[\\/][^"]*)"')
_PATH_CALL = re.compile(r"SHAARPPaths`(?:Ref|Repo|ML|SI)\[|SHAARPPaths`(?:MLDir|SIDir)\[\]")


def shipped_wl_scripts() -> list[Path]:
    """The .wl scripts this repository actually publishes.

    Uses git so that scripts excluded from the package are not held to this contract; falls back
    to the reference directory when git is unavailable (a zip download, say).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, text=True, check=True, timeout=60).stdout
        paths = [ROOT / p for p in out.split("\0") if p.endswith(".wl")]
        if paths:
            return sorted(paths)
    except Exception:
        pass
    return sorted(REF_DIR.glob("export_*.wl"))


class WolframPathPreambleTests(unittest.TestCase):
    def setUp(self):
        self.scripts = shipped_wl_scripts()

    def test_the_script_set_is_plausible(self):
        """A broken enumerator would make every other check below vacuously true."""
        self.assertGreater(len(self.scripts), 90, "shipped .wl set unexpectedly small")
        names = {p.name for p in self.scripts}
        self.assertIn("export_maker_fringes_reference.wl", names)

    def test_every_shipped_script_carries_the_shared_preamble(self):
        offenders = [p.name for p in self.scripts
                     if PREAMBLE.strip() not in p.read_text(encoding="utf-8")]
        self.assertEqual(offenders, [],
                         "scripts whose preamble has drifted from benchmarks/wolfram_paths.py:\n  "
                         + "\n  ".join(offenders))

    def test_no_shipped_script_contains_a_personal_absolute_path(self):
        offenders = []
        for p in self.scripts:
            for hit in _ABSOLUTE.findall(p.read_text(encoding="utf-8")):
                if not hit.startswith(ALLOWED_ABSOLUTE):
                    offenders.append(f"{p.name}: {hit}")
        self.assertEqual(offenders, [],
                         "absolute paths in shipped Wolfram scripts:\n  " + "\n  ".join(offenders))

    def test_file_arguments_go_through_the_path_helpers(self):
        """Import/Export/Get/SetDirectory must take a helper call or a variable -- never a
        literal path."""
        offenders = []
        call = re.compile(r"\b(Import|Export|Get|SetDirectory)\[\s*([^\s,\]]+)")
        for p in self.scripts:
            for verb, arg in call.findall(p.read_text(encoding="utf-8")):
                if arg.startswith('"') and ":" in arg:
                    offenders.append(f"{p.name}: {verb}[{arg[:60]}")
        self.assertEqual(offenders, [],
                         "literal paths passed to file operations:\n  " + "\n  ".join(offenders))

    def test_detector_actually_detects(self):
        """Falsifiability: the absolute-path check must fire on a planted path and stay quiet on
        the allow-listed Wolfram install location."""
        planted = '"' + "D:/Goo" "gle Drive/x.json" + '"'
        self.assertTrue(
            [h for h in _ABSOLUTE.findall(planted) if not h.startswith(ALLOWED_ABSOLUTE)])
        allowed = '"C:/Program Files/Wolfram Research/Wolfram Engine/14.3/wolfram.exe"'
        self.assertFalse(
            [h for h in _ABSOLUTE.findall(allowed) if not h.startswith(ALLOWED_ABSOLUTE)])
        self.assertTrue(_PATH_CALL.search('Get[SHAARPPaths`Ref["x.wl"]]'))


if __name__ == "__main__":
    unittest.main()
