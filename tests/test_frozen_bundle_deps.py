"""The frozen bundle must collect the dependencies the frozen app actually imports.

WHY THIS EXISTS: a release build failed at the frozen ``--self-check`` with
``ModuleNotFoundError: No module named 'scipy._external.array_api_compat.numpy.fft'``. scipy vendors
``array_api_compat`` and MOVED it -- ``scipy._lib.array_api_compat`` up to 1.17,
``scipy._external.array_api_compat`` from 1.18 -- while PyInstaller 6.x's bundled scipy hook still
names the old path. Against a newer scipy the hook's hidden import silently resolves to nothing, so
the new location is never collected and the app dies the first time it touches scipy.

Every source-run test passed, and the build itself exited 0: PyInstaller downgrades an unresolvable
hidden import to a printed warning. The defect was reachable only by running the FROZEN app, which
happens in the release job -- so it was found by a failed release rather than by CI. The
development environment could not have caught it either; its scipy was still on the old layout
while the runners' unpinned scipy had moved ahead.

This module runs from source, in CI, against whatever scipy the runner resolved. If a future scipy
renames the vendored package again, it goes red here -- cheaply, before a release build -- instead
of in a packaging job twenty minutes deep.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_gui_bundle import scipy_compat_collect_args  # noqa: E402


class ScipyVendoredCompatTests(unittest.TestCase):
    def test_we_know_where_this_scipy_vendors_array_api_compat(self):
        """The detector must find a location. Empty means scipy moved somewhere we do not know."""
        args = scipy_compat_collect_args()
        self.assertTrue(
            args,
            "scipy_compat_collect_args() found no vendored array_api_compat. scipy has probably "
            "moved it again; add the new package path in scripts/build_gui_bundle.py, or the "
            "frozen app will raise ModuleNotFoundError on its first scipy call.")

    def test_the_args_are_well_formed_collect_pairs(self):
        args = scipy_compat_collect_args()
        self.assertEqual(len(args) % 2, 0, f"odd-length argument list: {args!r}")
        for flag, target in zip(args[::2], args[1::2]):
            with self.subTest(target=target):
                self.assertEqual(flag, "--collect-submodules")
                self.assertTrue(target.startswith("scipy."), f"unexpected target {target!r}")

    def test_every_collected_package_really_imports(self):
        """find_spec can succeed where import fails; PyInstaller collects what imports."""
        for target in scipy_compat_collect_args()[1::2]:
            with self.subTest(target=target):
                importlib.import_module(target)

    def test_the_submodule_that_broke_the_release_resolves(self):
        """``<vendored>.numpy.fft`` is the exact module whose absence killed the frozen app."""
        targets = scipy_compat_collect_args()[1::2]
        resolved = []
        for target in targets:
            try:
                importlib.import_module(f"{target}.numpy.fft")
                resolved.append(target)
            except ImportError:
                pass
        self.assertTrue(
            resolved,
            f"none of {targets!r} provides .numpy.fft -- the frozen app imports it through scipy, "
            f"so collecting a package that lacks it would not prevent the failure this fences.")


if __name__ == "__main__":
    unittest.main()
