"""No shipped file carries a personal path or a note that only makes sense inside this project.

This is a package-wide extension of ``tests/test_notebook_hygiene.py``, which scans only the
notebooks. It exists because the leak it now prevents was real and large: 578 machine-specific
absolute paths across 256 shipped files, none of which any gate could see.

Two independent classes are checked:

``paths``
    Absolute paths naming a particular machine or user. A short allow-list covers genuinely
    machine-global install locations (a Wolfram kernel, the Windows font directory) and the
    deliberately synthetic paths used as test fixtures.
``process notes``
    Identifiers that only mean something inside the project's own development history --
    references to working documents that are not published, and tooling names.

Scanning discipline that matters:
  * text is scanned RAW and NORMALISED (``\\/``, ``\\\\``, ``%20`` and Wolfram line
    continuations all hide a path from a naive search);
  * in JSON, every string value is checked for paths, but only prose-valued keys are checked for
    process notes -- a frozen reference file contains symbolic coefficient names like ``F1x*d11``
    that look exactly like campaign identifiers;
  * ``image/*`` payloads in notebooks are skipped, since base64 contains any short pattern by
    chance.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".mx", ".pptx", ".whl",
                   ".zip", ".exe", ".pyc", ".pdf", ".ttf", ".otf", ".woff", ".woff2", ".nb"}
DERIVED_DIRS = {".git", "build", "dist", "outputs", "wheelhouse", "__pycache__", ".pytest_cache",
                ".ipynb_checkpoints", ".wolfram_tmp", "node_modules", ".venv", "Temp", "scratch",
                ".claude", ".mypy_cache"}

ALLOWED_ABSOLUTE = (
    "C:/Program Files/Wolfram Research", "C:\\Program Files\\Wolfram Research",
    "C:/Program Files (x86)/Wolfram Research", "C:\\Program Files (x86)\\Wolfram Research",
    "C:/Windows/Fonts", "C:\\Windows\\Fonts",
    # synthetic fixtures: the point of the tests that contain them
    "C:/Users/test", "C:\\Users\\test", "C:/Users/user", "C:\\Users\\user",
    "C:/temp", "C:\\temp", "C:/definitely_missing", "C:\\definitely_missing",
    "D:/tmp", "D:\\tmp",
)

# The detector files legitimately contain the strings they ban.
DETECTOR_FILES = {"tests/test_repo_hygiene.py", "tests/test_notebook_hygiene.py",
                  "tests/test_wolfram_path_preamble.py"}

PATH_PATTERNS = [
    (re.compile(r"Goo" "gle[ ]?Drive"), "a personal cloud-drive path"),
    (re.compile(r"PSU[\\/]+Research"), "a personal directory tree"),
    (re.compile(r"Users[\\/]+" + "51" "093"), "a developer username in a path"),
    (re.compile(r"anaconda3|[\\/]envs[\\/]"), "a local Python environment layout"),
]
# A real path has a component after the separator. Excluding whitespace there keeps prose ABOUT
# paths -- "C:\ strings parse as a single POSIX component" -- from being read as one.
_DRIVE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z]:[\\/](?![\\/\s])[^\"'<>|;\n\r]*)")

PROCESS_PATTERNS = [
    (re.compile(r"MASTER_HANDOFF|SURPRISE_LEDGER|FIDELITY_MATRIX|RELEASE_READINESS"
                r"|ANALYTICAL_COMPARISON_STRATEGY|MATHEMATICA_VALIDATION_(?:PLAN|SPRINT)"
                r"|SHAARP_FINISH_ROADMAP|SHAARP_PACKAGE_COMPLETION|SHAARP_DASHBOARD"
                r"|AGENTS_LOG|LOOP_STATUS"), "a working document that is not published"),
    (re.compile(r"gui-tester|fidelity-auditor|fidelity-judge|gate-runner|release-scribe"),
     "an internal tool name"),
]

PROSE_JSON_KEY = re.compile(
    r"^(?:source|status|notes?|description|comment|summary|[a-z_]*note[a-z_]*)$", re.I)


def _normalise(text: str) -> str:
    text = text.replace("\\/", "/").replace("\\\\", "\\").replace("%20", " ")
    text = re.sub(r"\\\r?\n", "", text)
    return text


def shipped_files() -> list[Path]:
    """Everything the repository publishes.

    git is the authority; the walk is the fallback for a source archive with no .git, where by
    construction only shipped files are present.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, text=True, check=True, timeout=60).stdout
        paths = [ROOT / p for p in out.split("\0") if p]
        if paths:
            return paths
    except Exception:
        pass
    return [p for p in ROOT.rglob("*")
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & DERIVED_DIRS)]


def _fragments(path: Path, text: str):
    """Yield (check_process, fragment) pairs for one file."""
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(text)
        except Exception:
            yield True, text
            return
        yield True, json.dumps({k: v for k, v in nb.items() if k != "cells"})
        for cell in nb.get("cells", []):
            yield True, "".join(cell.get("source", []))
            for out in cell.get("outputs", []) or []:
                if out.get("output_type") == "stream":
                    yield True, "".join(out.get("text", []))
                elif out.get("output_type") == "error":
                    yield True, "\n".join(out.get("traceback", []))
                else:
                    for mime, payload in (out.get("data") or {}).items():
                        if mime.startswith("image/"):
                            continue
                        if isinstance(payload, list):
                            payload = "".join(str(x) for x in payload)
                        yield True, str(payload)
        return

    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except Exception:
            yield False, text
            return

        def walk(node, key=None):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield from walk(v, k)
            elif isinstance(node, list):
                for v in node:
                    yield from walk(v, key)
            elif isinstance(node, str):
                yield bool(key and PROSE_JSON_KEY.match(key)), node

        yield from walk(data)
        return

    yield True, text


class RepoHygieneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = shipped_files()

    def _scan(self, *, process: bool):
        offenders = []
        for path in self.files:
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix.lower() in BINARY_SUFFIXES or rel in DETECTOR_FILES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for check_process, fragment in _fragments(path, text):
                for body in (fragment, _normalise(fragment)):
                    for pattern, label in PATH_PATTERNS:
                        m = pattern.search(body)
                        if m:
                            offenders.append(f"{rel}: {label} ({m.group(0)!r})")
                    for m in _DRIVE.finditer(body):
                        value = m.group(1).strip().rstrip(",)]}")
                        if not value.startswith(ALLOWED_ABSOLUTE):
                            offenders.append(f"{rel}: absolute path ({value[:70]!r})")
                    if process and check_process:
                        for pattern, label in PROCESS_PATTERNS:
                            m = pattern.search(body)
                            if m:
                                offenders.append(f"{rel}: {label} ({m.group(0)!r})")
        return sorted(set(offenders))

    def test_every_referenced_repo_path_is_actually_shipped(self):
        """A pointer to a file the package does not contain sends a reader nowhere.

        Checking existence ON DISK is not enough: a maintainer-only file is present in the working
        tree while being excluded from the package, so `(ROOT / ref).exists()` is True here and
        False for everyone else. Membership in the shipped set is the only question a clone can
        answer. Six such pointers survived the publication scrub -- to a release gate, a repo audit
        and a hang probe that no longer ship -- because the checks above look for banned
        identifiers and never resolve a path.
        """
        shipped = {p.relative_to(ROOT).as_posix() for p in self.files}
        # a backticked token that looks like a repo path: has a slash, ends in a known extension
        token = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|md|ipynb|json|toml|cff|yml|yaml|bat|wl))`")
        # this module documents the anchor FORMAT with invented examples ("path/like/this.py"),
        # which are illustrations rather than pointers
        skip = DETECTOR_FILES | {"tests/test_residual_risk_register_integrity.py"}
        offenders = []
        for path in self.files:
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix.lower() not in {".py", ".md"} or rel in skip:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for ref in set(token.findall(text)):
                if "/" not in ref or ref.startswith(("http", "..")):
                    continue
                if ref not in shipped and not (ROOT / ref).is_dir():
                    offenders.append(f"{rel}: points at {ref!r}, which the package does not ship")
        self.assertEqual(sorted(set(offenders)), [],
                         "references to files that are not in the shipped set:\n  "
                         + "\n  ".join(sorted(set(offenders))))

    def test_the_shipped_set_is_plausible(self):
        """A broken enumerator would make every check below vacuously green."""
        self.assertGreater(len(self.files), 800, "shipped set unexpectedly small")
        names = {p.relative_to(ROOT).as_posix() for p in self.files}
        for sentinel in ("README.md", "shaarp/api.py",
                         "benchmarks/multilayer_system_benchmarks_v1.json"):
            self.assertIn(sentinel, names)

    def test_no_shipped_file_contains_a_personal_path(self):
        offenders = self._scan(process=False)
        self.assertEqual(offenders, [],
                         "personal or machine-specific paths in shipped files:\n  "
                         + "\n  ".join(offenders))

    def test_no_shipped_file_references_an_unpublished_working_document(self):
        offenders = [o for o in self._scan(process=True)
                     if "working document" in o or "internal tool" in o]
        self.assertEqual(offenders, [],
                         "references to unpublished material in shipped files:\n  "
                         + "\n  ".join(offenders))

    def test_detector_actually_detects(self):
        """Falsifiability. The samples are assembled from split literals so this file does not
        contain the banned strings it is testing for."""
        leaks = [
            "D:/" + "Goo" "gle Drive" + "/Someone/project/file.json",
            "C:\\Users\\" + "51" "093" + "\\AppData",
            "/home/x/" + "anaconda3" + "/envs/thing",
        ]
        for sample in leaks:
            fired = any(p.search(sample) for p, _ in PATH_PATTERNS) or any(
                not m.group(1).startswith(ALLOWED_ABSOLUTE) for m in _DRIVE.finditer(sample))
            self.assertTrue(fired, f"no path pattern fires on {sample!r}")

        allowed = "C:/Program Files/Wolfram Research/Wolfram/14.3/WolframKernel.exe"
        self.assertFalse([m for m in _DRIVE.finditer(allowed)
                          if not m.group(1).startswith(ALLOWED_ABSOLUTE)],
                         "the Wolfram install path must not be flagged")

        doc = "MASTER" "_HANDOFF.md"
        self.assertTrue(any(p.search(doc) for p, _ in PROCESS_PATTERNS))

    def test_the_scan_visits_every_shipped_text_file(self):
        """A silently skipped extension could hide a leak."""
        visited = sum(1 for p in self.files
                      if p.suffix.lower() not in BINARY_SUFFIXES
                      and p.relative_to(ROOT).as_posix() not in DETECTOR_FILES)
        self.assertGreater(visited, 700, "too few files reached the scanner")

    def test_the_scan_is_fast_enough_to_keep(self):
        start = time.time()
        self._scan(process=True)
        self.assertLess(time.time() - start, 30.0, "hygiene scan has become too slow")


if __name__ == "__main__":
    unittest.main()
