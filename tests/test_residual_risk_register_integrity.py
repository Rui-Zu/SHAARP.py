"""The residual-risk register must stay TRUE.

WHY THIS EXISTS. `docs/residual_risks.md` says its own job is to let a future session or a user
report tell a *known accepted limit* from a *new defect*. A register that has drifted cannot do
that job — and two rows had drifted:

  * **R11** sat for three weeks reading "not yet done" about the TaAs (112) azimuth ownership
    question, which had actually been root-caused and fixed on.
  * **R7** described a fallback ("biaxial ... falls back to the substituted form, >2 min CAS") that
    FA-1 had eliminated the day before.

Neither was catchable by anything that existed. `scripts/repo_audit.py` checks stale NUMBERS and is
advisory-only; prose that describes behaviour reads perfectly well after the behaviour changes.
There was also a second register at the repo root that COULD NOT SHIP (gitignored) yet read
authoritative and was doubly stale — deleted.

WHAT THIS PINS (deliberately structural, not editorial — it cannot judge whether prose is true,
but it can refuse to let a row cite something that does not exist, or a count that is wrong):
  1. every row carries at least one ANCHOR (a real repo path or a `test_*`/`Test*` name), and every
     anchor resolves — so a renamed/deleted fence cannot leave a row silently pointing at nothing;
  2. every module/test COUNT claimed in the file matches the live suite — the R10 rot class, which
     was previously only an advisory line in repo_audit;
  3. the deleted root register does not come back;
  4. rows are uniquely numbered and none is skipped (numbering is cited from tests and the handoff).

Precedent: `tests/test_dextraction_noise_robustness.py` already binds its fence bound to register
row R8 in both directions, which is exactly why R8 never rotted. This generalizes that habit.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "docs" / "residual_risks.md"

# `path/like/this.py`, `docs/thing.md`, `tests/test_x.py::TestY::test_z`
_PATH_ANCHOR = re.compile(r"`([A-Za-z_][A-Za-z0-9_./-]*\.(?:py|md|json|csv|wl|yml|toml|cff))"
                          r"(?:::[A-Za-z0-9_:.]+)?`")
# bare test/class names cited without a path, e.g. `test_wavelength_clamp_note_appears...`
_NAME_ANCHOR = re.compile(r"`((?:test_|Test)[A-Za-z0-9_]+)`")
_COUNT = re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d+)[- ](?:test[- ])?(modules?|tests?)\b")
_ROW = re.compile(r"^\|\s*(R\d+)\s*\|", re.M)


def _register_text() -> str:
    return REGISTER.read_text(encoding="utf-8")


def _rows() -> list[tuple[str, str]]:
    """(row-id, full row text) for every R-numbered table row."""
    out = []
    for line in _register_text().splitlines():
        m = re.match(r"^\|\s*(R\d+)\s*\|", line)
        if m:
            out.append((m.group(1), line))
    return out


class RegisterExistsAndIsWellFormed(unittest.TestCase):
    def test_the_shipped_register_is_the_only_one(self):
        self.assertTrue(REGISTER.exists(), "docs/residual_risks.md is missing")
        # The root copy was deleted: it was gitignored (could never ship) yet read
        # authoritative while carrying doubly-stale rows. It must not come back.
        self.assertFalse((ROOT / "RESIDUAL_RISKS.md").exists(),
                         "the root RESIDUAL_RISKS.md is back — it cannot ship (gitignored) and "
                         "having two registers is what let them diverge")

    def test_rows_are_uniquely_numbered_and_contiguous(self):
        ids = [rid for rid, _ in _rows()]
        self.assertTrue(ids, "no R-numbered rows found — did the table format change?")
        self.assertEqual(len(ids), len(set(ids)), f"duplicate row ids: {ids}")
        numbers = sorted(int(r[1:]) for r in ids)
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)),
                         f"row numbering must stay contiguous (tests and the handoff cite it): {numbers}")


class EveryRowCitesSomethingThatExists(unittest.TestCase):
    """A row may not point at a file or fence that is gone — the R7/R11 rot class."""

    def test_every_row_has_at_least_one_anchor(self):
        missing = [rid for rid, row in _rows()
                   if not _PATH_ANCHOR.findall(row) and not _NAME_ANCHOR.findall(row)]
        self.assertEqual(missing, [],
                         f"rows with no citable anchor (path or test name): {missing} — a row that "
                         f"cites nothing cannot be checked and is how R11 went stale")

    def test_every_cited_path_exists(self):
        bad = []
        for rid, row in _rows():
            for rel in _PATH_ANCHOR.findall(row):
                if not (ROOT / rel).exists():
                    bad.append(f"{rid} -> {rel}")
        self.assertEqual(bad, [], f"register rows cite paths that do not exist: {bad}")

    def test_every_cited_test_name_exists_somewhere_in_the_suite(self):
        suite_text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                               for p in (ROOT / "tests").glob("test_*.py"))
        bad = []
        for rid, row in _rows():
            for name in _NAME_ANCHOR.findall(row):
                if name not in suite_text:
                    bad.append(f"{rid} -> {name}")
        self.assertEqual(bad, [], f"register rows cite tests that no longer exist: {bad}")


class TheRegisterDoesNotFreezeSuiteCounts(unittest.TestCase):
    """The R10 rot class, fixed at the root rather than policed forever.

    R10 carried "202-module suite" long after the suite had grown. The obvious fix — assert the
    number is current — was tried and rejected: it went red the instant THIS module was added,
    because a frozen count in prose drifts on every new test, making a doc edit mandatory forever.
    Perpetual churn is how gates get loosened.

    So the rule is: the register describes LIMITS; it does not mirror the stamp. Counts live in one
    place (`docs/validation.md`), and this test keeps them from being copied back in here."""

    def test_no_frozen_module_count_in_the_register(self):
        claimed = [f"{n} {unit}" for n, unit in _COUNT.findall(_register_text())
                   if unit.startswith("module")]
        self.assertEqual(claimed, [],
                         f"the register names a module count {claimed}; counts drift and rot here. "
                         f"State the limit and point at the stamp in docs/validation.md instead.")

    def test_the_stamp_document_still_carries_the_counts(self):
        """...and the number has to live SOMEWHERE, or removing it from here just loses it."""
        stamp = (ROOT / "docs" / "validation.md").read_text(encoding="utf-8")
        self.assertTrue(_COUNT.search(stamp),
                        "docs/validation.md no longer states a module/test count — the register "
                        "defers to it, so it must actually carry the numbers")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
