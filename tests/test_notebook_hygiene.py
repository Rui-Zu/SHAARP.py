"""Notebook hygiene gate — no personal/local paths or stale metadata in SHIPPED notebooks.

WHY: the shipped notebooks embed their execution outputs, and three
leak classes actually occurred: (1) `build_paper_notebooks.py` printed the ABSOLUTE repo root into
cell-1 stdout (a one-off hand scrub was silently reverted by the next regeneration until the
builders were fixed to print `ROOT.name`); (2) a stored `%pip install` log carried the developer's
project path URL-ENCODED (`D:/Google%20Drive/...` — invisible to plain-text scans); (3) a stored
environment printout carried the local conda layout and a stale `shaarp-py-0.1.0` version. This
test is the permanent ratchet: it fails on every known class, in every shipped notebook, on every
machine (fresh clones included).

Scanning discipline: parse the ipynb JSON and scan cell SOURCES and TEXT outputs only — embedded
``image/*`` payloads are skipped because base64 can contain any short pattern by chance.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

try:  # stdlib on Python >= 3.11; the package floor is 3.10, so degrade to a skip there
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 only
    tomllib = None

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = (ROOT / "notebooks", ROOT / "docs" / "tutorials")

# Personal/local-path bans. The username is banned in path context; the project path is banned in
# BOTH plain and URL-encoded forms; any absolute Windows user-profile path is banned generically
# (covers redacted-but-machine-specific strings like ``C:\Users\user\anaconda3\...``).
BANNED = [
    # the username digits are split so this shipped file never contains the banned literal itself
    (re.compile(r"Users[\\/]+" + "51" "093"), "developer username in a path"),
    (re.compile(r"Google(?: |%20)Drive"), "local project path (Google Drive)"),
    (re.compile(r"EM(?: |%20)Waves"), "local project path (EM Waves)"),
    (re.compile(r"anaconda3"), "local environment layout (anaconda3)"),
    (re.compile(r"[A-Za-z]:[\\/]+Users"), "absolute Windows user-profile path"),
]
ALLOWED_KERNEL_DISPLAY_NAMES = {"Python 3", "Python 3 (ipykernel)"}
VERSION_PIN = re.compile(r"shaarp[-_]py[=-]=?(\d+\.\d+\.\d+)")


def _iter_text_fragments(nb: dict):
    """Yield (where, text) for every scannable text fragment of a parsed notebook.

    Includes the notebook-level METADATA. Until this walked ``cells`` only, so the
    whole ``metadata`` block -- kernelspec, language_info, and anything a tool writes there --
    was invisible to every ban below. The gate stayed green through a kernelspec edit it was
    structurally incapable of judging, and a local conda env name shipped in it.
    """
    yield "notebook metadata", json.dumps(
        {k: v for k, v in nb.items() if k != "cells"}, ensure_ascii=False)
    for i, cell in enumerate(nb.get("cells", [])):
        yield f"cell {i} source", "".join(cell.get("source", []))
        for j, out in enumerate(cell.get("outputs", []) or []):
            kind = out.get("output_type")
            if kind == "stream":
                yield f"cell {i} output {j} (stream)", "".join(out.get("text", []))
            elif kind == "error":
                yield f"cell {i} output {j} (error)", "\n".join(out.get("traceback", []))
            elif kind in ("execute_result", "display_data"):
                for mime, payload in (out.get("data") or {}).items():
                    if mime.startswith("image/"):
                        continue  # base64 — any substring can occur by chance
                    if isinstance(payload, list):
                        payload = "".join(str(p) for p in payload)
                    yield f"cell {i} output {j} ({mime})", str(payload)


class NotebookHygieneTests(unittest.TestCase):
    def _notebooks(self):
        nbs = sorted(p for d in NOTEBOOK_DIRS if d.is_dir() for p in d.glob("*.ipynb"))
        self.assertGreaterEqual(len(nbs), 8, "shipped notebook set unexpectedly small")
        return nbs

    def test_no_personal_or_local_paths_in_any_shipped_notebook(self):
        offenders = []
        for nb_path in self._notebooks():
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            for where, text in _iter_text_fragments(nb):
                for pattern, label in BANNED:
                    m = pattern.search(text)
                    if m:
                        offenders.append(f"{nb_path.name}: {where}: {label} ({m.group(0)!r})")
        self.assertEqual(offenders, [],
                         "personal/local paths in shipped notebooks:\n  " + "\n  ".join(offenders))

    def test_no_stale_shaarp_py_version_pins(self):
        if tomllib is None:
            self.skipTest("tomllib requires Python >= 3.11")
        current = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8")
                                )["project"]["version"]
        offenders = []
        for nb_path in self._notebooks():
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            for where, text in _iter_text_fragments(nb):
                for ver in VERSION_PIN.findall(text):
                    if ver != current:
                        offenders.append(f"{nb_path.name}: {where}: shaarp-py {ver} != {current}")
        self.assertEqual(offenders, [],
                         "stale version pins in shipped notebooks:\n  " + "\n  ".join(offenders))

    def test_kernelspec_is_present_and_machine_neutral(self):
        """Every shipped notebook opens without a kernel prompt, under a name that is not a
        local environment. Allowlist rather than a ban-list: the leak is the CLASS of names a
        machine happens to have (``py312``, ``base``, ``shaarp-dev``), which no ban can enumerate.
        """
        offenders = []
        for nb_path in self._notebooks():
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            ks = (nb.get("metadata") or {}).get("kernelspec")
            if not ks:
                offenders.append(f"{nb_path.name}: no kernelspec (Jupyter prompts on open)")
                continue
            if ks.get("name") != "python3":
                offenders.append(f"{nb_path.name}: kernel name {ks.get('name')!r} != 'python3'")
            if ks.get("display_name") not in ALLOWED_KERNEL_DISPLAY_NAMES:
                offenders.append(
                    f"{nb_path.name}: kernel display_name {ks.get('display_name')!r} looks "
                    f"machine-specific; expected one of {sorted(ALLOWED_KERNEL_DISPLAY_NAMES)}")
        self.assertEqual(offenders, [],
                         "kernelspec problems in shipped notebooks:\n  " + "\n  ".join(offenders))

    def test_detector_actually_detects(self):
        """Self-test: each banned pattern fires on a representative leaked string (falsifiability).
        The username sample is assembled from split literals so THIS shipped file never contains
        the banned string itself (it would otherwise trip repo-wide hygiene scans)."""
        username = "51" "093"
        # The samples are SYNTHETIC: they carry the tokens each pattern must catch, but not a
        # real directory hierarchy -- a detector should not be a copy of the thing it detects.
        drive = "Goo" "gle Drive"
        project = "EM" " Waves"
        samples = [
            "C:\\Users\\" + username + "\\AppData\\Local\\Temp",
            f"file:///D:/{drive.replace(' ', '%20')}/Example/{project.replace(' ', '%20')}/work",
            f"D:\\{drive}\\Example\\{project} - Copy",
            r"c:\Users\user\anaconda3\envs\some-env\python.exe",
        ]
        for sample in samples:
            self.assertTrue(any(p.search(sample) for p, _l in BANNED),
                            f"no banned pattern fires on {sample!r}")
        self.assertEqual(VERSION_PIN.findall("shaarp-py-0.1.0 shaarp_py==1.0.0"),
                         ["0.1.0", "1.0.0"])

        # and prove the scanner REACHES metadata -- the gap that let a kernelspec edit through
        leaky = {"cells": [], "metadata": {"kernelspec": {"display_name": "some-env"},
                                           "note": f"d:/{drive}/{project}"}}
        found = [t for where, t in _iter_text_fragments(leaky) if where == "notebook metadata"]
        self.assertEqual(len(found), 1, "metadata fragment not yielded by the scanner")
        self.assertTrue(any(p.search(found[0]) for p, _l in BANNED),
                        "a banned path inside metadata is not detected")


class NotebookCopyParityTests(unittest.TestCase):
    """The two shipped copies of a notebook must teach the same thing.

    Every tutorial ships twice: ``notebooks/`` for someone reading the repository, and
    ``docs/tutorials/`` for the Sphinx site (``docs/conf.py`` renders .ipynb through myst-nb with
    execution off, so the docs copy is what a visitor actually reads). Two builders keep their own
    pairs in step, but ``SHAARP_py_step_by_step.ipynb`` is hand-maintained -- and it drifted: the
    repo copy got the "nothing needs installing" rewrite while the published copy still told
    readers to run ``%pip install -e ".."``, a relative path that resolves from one directory only.

    Nothing caught it because ``tests/test_interactive.py`` asserts against ``notebooks/`` alone,
    leaving the published copy unexamined. This fence compares SOURCE cells only: outputs and
    execution counts legitimately differ between a re-executed copy and its sibling, and comparing
    those would go red on every regeneration -- a fence that cries wolf gets switched off.
    """

    def _pairs(self):
        for repo_copy in sorted((ROOT / "notebooks").glob("*.ipynb")):
            docs_copy = ROOT / "docs" / "tutorials" / repo_copy.name
            if docs_copy.exists():
                yield repo_copy, docs_copy

    @staticmethod
    def _sources(path: Path):
        nb = json.loads(path.read_text(encoding="utf-8"))
        return [(c.get("cell_type"), "".join(c.get("source", []))) for c in nb.get("cells", [])]

    def test_both_shipped_copies_carry_the_same_instructions(self):
        for repo_copy, docs_copy in self._pairs():
            with self.subTest(notebook=repo_copy.name):
                repo_cells, docs_cells = self._sources(repo_copy), self._sources(docs_copy)
                self.assertEqual(
                    len(repo_cells), len(docs_cells),
                    f"{repo_copy.name}: notebooks/ has {len(repo_cells)} cells, "
                    f"docs/tutorials/ has {len(docs_cells)} -- copy one over the other")
                for i, (repo_cell, docs_cell) in enumerate(zip(repo_cells, docs_cells)):
                    self.assertEqual(
                        repo_cell, docs_cell,
                        f"{repo_copy.name} cell {i} differs between notebooks/ and "
                        f"docs/tutorials/. The docs copy is the one rendered on the site, so a "
                        f"stale cell there is what visitors read.")

    def test_the_pairing_is_not_vacuous(self):
        """Without this, an empty docs/tutorials/ would make the fence above pass trivially."""
        pairs = list(self._pairs())
        self.assertGreaterEqual(len(pairs), 5, f"expected >=5 paired notebooks, found {len(pairs)}")
        self.assertIn("SHAARP_py_step_by_step.ipynb", {p.name for p, _ in pairs})


if __name__ == "__main__":
    unittest.main()
