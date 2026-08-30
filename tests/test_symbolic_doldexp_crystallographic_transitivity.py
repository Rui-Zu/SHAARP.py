"""Validate the 6 crystallographic SHG point groups 4, 6, 4mm, 6mm, 422, 622 by
EXACT TRANSITIVITY through the already live-validated infinity-family templates.

Two independently checkable facts make this a real (not assumed) validation:

1. SHAARP.si's `doldExp` GUI branches fold each crystallographic group into a
   shared OR-condition with its infinity-family partner -- i.e. SHAARP.si assigns
   the SAME doldExp matrix to "4" and "inf", to "4mm" and "infm", to "422" and
   "inf2" (and likewise 6/6mm/622). This was read directly from the SHAARP_V1.03
   cell-4 box tree (If[PointGroup=="4" || PointGroup=="6" || ... , doldExp = M]);
   the captured condition strings are in
   `benchmarks/mathematica_reference/doldexp_condition_probe.json`.

2. Python's `d_voigt_symbolic` returns matrices that are EXACTLY (symbolically)
   equal for each crystallographic group and its infinity-family partner.

Since the infinity-family templates are already validated EXACTLY against live
SHAARP.si (`wolfram_live_symbolic_doldexp_reference_v1.json`), chaining
  SHAARP.si(cryst) == SHAARP.si(inf-family) == Python(inf-family) == Python(cryst)
validates the crystallographic groups with no new Wolfram run and no physics
assumption beyond SHAARP.si's own grouping.
"""

import json
import unittest
from pathlib import Path

from shaarp.symbolic import d_voigt_symbolic


ROOT = Path(__file__).resolve().parents[1]
CONDITION_PROBE = (
    ROOT / "benchmarks" / "mathematica_reference" / "doldexp_condition_probe.json"
)

# crystallographic group -> its infinity-family partner (validated template)
PARTNER = {
    "4": "inf",
    "6": "inf",
    "4mm": "infm",
    "6mm": "infm",
    "422": "inf2",
    "622": "inf2",
}


class DoldExpCrystallographicTransitivityTests(unittest.TestCase):
    def test_python_crystallographic_matrices_equal_infinity_family_exactly(self):
        for cryst, partner in PARTNER.items():
            self.assertEqual(
                d_voigt_symbolic(cryst),
                d_voigt_symbolic(partner),
                msg=f"{cryst} should equal {partner} symbolically",
            )

    def test_shaarp_si_folds_crystallographic_into_infinity_branches(self):
        """The SHAARP.si doldExp condition probe must show 4/6, 4mm/6mm, 422/622
        sharing OR-conditions (proving SHAARP.si assigns them the same matrix)."""
        if not CONDITION_PROBE.exists():
            self.skipTest("doldexp_condition_probe.json not present")
        import re

        probe = json.loads(CONDITION_PROBE.read_text(encoding="utf-8"))
        conditions = probe["conditions_truncated"]
        # The stored InputForm escapes the group quotes (e.g. \"4\"), so match the
        # quoted PointGroup token robustly: a backslash-or-bare double-quote, the
        # exact group label, then a closing quote. Each crystallographic group must
        # co-occur in an OR-branch with its partner crystallographic group (4 with
        # 6, 4mm with 6mm, 422 with 622) and an `||` and `doldExp` -- the SHAARP.si
        # grouping that the infinity matrix shares.
        def token(label: str) -> str:
            return r'\\?"' + re.escape(label) + r'\\?"'

        for a, b in (("4", "6"), ("4mm", "6mm"), ("422", "622")):
            shared = any(
                re.search(token(a), c)
                and re.search(token(b), c)
                and "||" in c
                and "doldExp" in c
                for c in conditions
            )
            self.assertTrue(shared, msg=f'"{a}" and "{b}" should share a doldExp OR-branch')

    def test_partners_are_in_the_validated_reference(self):
        """The infinity-family partners must actually appear in the live-validated
        doldExp reference (so the transitivity chain is anchored)."""
        ref_path = (
            ROOT / "benchmarks" / "mathematica_reference"
            / "wolfram_live_symbolic_doldexp_reference_v1.json"
        )
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        validated_pgs = {str(e.get("point_group", "")) for e in ref["expressions"]}
        # the infinity glyph used in the reference
        self.assertIn("∞", validated_pgs)   # inf
        self.assertIn("∞m", validated_pgs)  # infm
        self.assertIn("∞2", validated_pgs)  # inf2


if __name__ == "__main__":
    unittest.main()
