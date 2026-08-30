"""The release archive must be named for the FULL version and the platform.

A release asset is a contract with the download table in README.md, and it is the one artefact no
test could previously see: the macOS packaging branch only runs on a macOS release build, so a
naming defect there ships silently and is discovered by users.

The defect this fences: the archive name was built with ``Path.with_suffix(".zip")``. A stem like
``SHAARP_py_v1.0.0_macos`` looks suffixed to pathlib -- it reads ``.0_macos`` as the extension and
replaces it -- so the asset came out as ``SHAARP_py_v1.0.zip``: patch digit gone, platform token
gone, and 1.0.1 would have collided with 1.0.0. The Windows branch was unaffected because
``shutil.make_archive`` appends rather than replaces, which is exactly why one platform passing
says nothing about the other.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_gui_bundle as bundle  # noqa: E402


def archive_name(version: str, token: str) -> str:
    """Reproduce the name ``package()`` builds, without running PyInstaller."""
    base = ROOT / "dist" / "release" / f"SHAARP_py_v{version}_{token}"
    return (base.parent / (base.name + ".zip")).name


class ReleaseArchiveNameTests(unittest.TestCase):
    VERSIONS = ("1.0.0", "1.0.1", "1.2.3", "10.20.30")
    TOKENS = ("win64", "macos")

    def test_the_name_keeps_the_full_version_and_the_platform(self):
        for version in self.VERSIONS:
            for token in self.TOKENS:
                name = archive_name(version, token)
                self.assertEqual(name, f"SHAARP_py_v{version}_{token}.zip")
                self.assertIn(version, name, "the full version must survive")
                self.assertIn(token, name, "the platform token must survive")

    def test_names_are_unique_across_versions_and_platforms(self):
        """1.0.0 and 1.0.1 assets must not overwrite each other on the same release."""
        names = [archive_name(v, t) for v in self.VERSIONS for t in self.TOKENS]
        self.assertEqual(len(names), len(set(names)))

    def test_with_suffix_would_have_been_wrong(self):
        """Pin the reason: this is why the construction is not ``Path.with_suffix``."""
        base = Path("dist/release/SHAARP_py_v1.0.0_macos")
        self.assertEqual(base.with_suffix(".zip").name, "SHAARP_py_v1.0.zip")
        self.assertNotEqual(base.with_suffix(".zip").name, archive_name("1.0.0", "macos"))

    def test_the_builder_still_constructs_the_name_this_way(self):
        """If package() changes shape, this fence must be revisited rather than silently bypassed."""
        source = (ROOT / "scripts" / "build_gui_bundle.py").read_text(encoding="utf-8")
        self.assertIn('base = out_dir / f"SHAARP_py_v{version}_{platform_token()}"', source)
        self.assertIn('base.parent / (base.name + ".zip")', source)
        self.assertNotIn('base.with_suffix(".zip")', source)

    def test_platform_token_is_one_per_operating_system(self):
        self.assertIn(bundle.platform_token(), ("win64", "macos", sys.platform))
        token_source = re.search(r"def platform_token.*?\n\n", source_of(bundle), re.S)
        self.assertIsNotNone(token_source)
        self.assertNotIn("arm64", token_source.group(0))
        self.assertNotIn("x86_64", token_source.group(0))


def source_of(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
