"""The release archive must be named for the FULL version and the platform.

A release asset is a contract with the download table in README.md, and it is the one artefact the
suite could not previously see: the macOS packaging branch runs only during a macOS release build,
so a naming defect there ships silently and is found by users.

The defect this fences: the name was built with ``Path.with_suffix(".zip")``. A stem like
``SHAARP_py_v1.0.0_macos`` looks suffixed to pathlib -- it reads ``.0_macos`` as the extension and
replaces it -- so the asset came out as ``SHAARP_py_v1.0.zip``: patch digit gone, platform token
gone, and 1.0.1 would have overwritten 1.0.0. The Windows branch was unaffected because
``shutil.make_archive`` appends rather than replaces, which is why one platform passing said
nothing about the other.

These tests call ``build_gui_bundle.archive_path`` -- the function ``package()`` itself uses. An
earlier version of this file re-implemented the naming expression locally, so four of its five
tests passed no matter what the builder did; only the source-text assertion could fail, and that
one broke whenever a comment was reflowed. A fence with its own copy of the logic is not a fence.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_gui_bundle import archive_path, platform_token  # noqa: E402

OUT = Path("dist") / "release"


class ReleaseArchiveNameTests(unittest.TestCase):
    VERSIONS = ("1.0.0", "1.0.1", "1.2.3", "1.10.0", "10.20.30")
    TOKENS = ("win64", "macos")

    def test_the_name_keeps_the_full_version_and_the_platform(self):
        for version in self.VERSIONS:
            for token in self.TOKENS:
                with self.subTest(version=version, token=token):
                    name = archive_path(OUT, version, token).name
                    self.assertEqual(name, f"SHAARP_py_v{version}_{token}.zip")

    def test_a_patch_release_does_not_collide_with_its_predecessor(self):
        """1.0.0 and 1.0.1 land on the same release page; identical names would overwrite."""
        for token in self.TOKENS:
            self.assertNotEqual(archive_path(OUT, "1.0.0", token).name,
                                archive_path(OUT, "1.0.1", token).name)

    def test_every_name_is_distinct(self):
        names = [archive_path(OUT, v, t).name for v in self.VERSIONS for t in self.TOKENS]
        self.assertEqual(len(names), len(set(names)))

    def test_the_archive_lands_in_the_directory_it_was_given(self):
        self.assertEqual(archive_path(OUT, "1.0.0", "macos").parent, OUT)

    def test_with_suffix_would_lose_the_version(self):
        """Pin the trap itself, so the reason for the construction survives a refactor."""
        stem = OUT / "SHAARP_py_v1.0.0_macos"
        self.assertEqual(stem.with_suffix(".zip").name, "SHAARP_py_v1.0.zip")
        self.assertNotEqual(stem.with_suffix(".zip").name,
                            archive_path(OUT, "1.0.0", "macos").name)

    def test_the_default_token_is_this_platform(self):
        self.assertEqual(archive_path(OUT, "1.0.0").name,
                         f"SHAARP_py_v1.0.0_{platform_token()}.zip")

    def test_one_bundle_per_operating_system(self):
        """Windows and macOS are the built targets; the tokens carry no CPU architecture."""
        for token in self.TOKENS:
            name = archive_path(OUT, "1.0.0", token).name
            self.assertNotIn("arm64", name)
            self.assertNotIn("x86_64", name)


if __name__ == "__main__":
    unittest.main()
