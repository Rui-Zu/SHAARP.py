"""The symbol a layer carries is the number on screen, and OFF means substituted.

Rui, driving the Fig-4 preset: "when analytical thickness or d, you should make sure
the symbol subscript match the layer materials, as indicated here, we are editing layer 2 but you
used h1" and "when I enable analytical d_SHG but turn off analytical h, I don't see SHG analytical
symbol but only see a symbol h, what is that?".

Both were real. F60 numbered the closed form's symbols with an INTERIOR-only counter while the
editor numbers every medium (air = 1, so the first film is 2), and a back-compat branch made an
UNFLAGGED layer symbolic with a bare ``h``. These fences pin the contract:

  * the number in a symbol == the layer's row number in the editor (`layer_stack.layer_number`);
  * a flag that is OFF substitutes -- no symbol survives for that layer;
  * the spelling does not depend on unrelated layers (pre-F61 one layer could be `d11`, `d11m1`
    or `d11m2` depending on how many OTHER layers were SHG-active);
  * a component declared KNOWN binds to the layer it was typed for, and to no other.
"""

import dataclasses
import re
import unittest

from shaarp.layer_stack import interior_layer_number, layer_number
from shaarp.shaarp_gui import compute_ml_gui_result, resolve_ml_system_preset

FIG4 = "Quartz + Au (Fig 4, 800 nm)"


def _flagged(system, index, *, analytic_h=False, analytic_d=False, shg_active=None):
    layers = list(system.layers)
    kw = {"analytic_h": analytic_h, "analytic_d": analytic_d}
    if shg_active is not None:
        kw["shg_active"] = shg_active
    layers[index] = dataclasses.replace(layers[index], **kw)
    return dataclasses.replace(system, layers=layers)


def _symbols(result, key="reflected_p_2omega"):
    text = str(result.stages.get(key))
    return {m for m in re.findall(r"\b[a-zA-Z]\w*\b", text)
            if m[0] in "hd" and not m.startswith("dtype")}


class NumberingHelpers(unittest.TestCase):
    def test_row_numbers_count_every_medium(self):
        """air is row 1, so the first FILM is row 2 -- the number the editor shows."""
        self.assertEqual(layer_number(0), 1)
        self.assertEqual(layer_number(1), 2)
        self.assertEqual(interior_layer_number(0), 2, "first interior layer is row 2")
        self.assertEqual(interior_layer_number(1), 3)


class SymbolsMatchTheEditedLayer(unittest.TestCase):
    """The Fig-4 stack is air / quartz / Au / air, so quartz is ROW 2 and Au is ROW 3."""

    @classmethod
    def setUpClass(cls):
        cls.base = resolve_ml_system_preset(FIG4)

    def _run(self, system, **kw):
        return compute_ml_gui_result("Partial Analytical", theta_deg=20.0, system=system, **kw)

    def test_thickness_symbol_carries_the_row_number(self):
        """RED before editing row 2 produced `h1`."""
        r = self._run(_flagged(self.base, 1, analytic_h=True))
        syms = _symbols(r)
        self.assertIn("h2", syms, f"row 2 must be h2, got {sorted(syms)}")
        self.assertNotIn("h1", syms, "h1 would be the interior counter, not the row on screen")

    def test_d_symbols_carry_the_row_number(self):
        r = self._run(_flagged(self.base, 1, analytic_d=True))
        syms = _symbols(r)
        self.assertTrue({"d11m2", "d14m2"} <= syms, f"row 2 d must be *m2, got {sorted(syms)}")
        self.assertNotIn("d11", syms - {"d11m2"}, "bare d11 has no layer indicator")

    def test_an_unflagged_layer_is_substituted(self):
        """RED before a 'stack default' branch made the first interior layer symbolic with
        a bare `h` even though nothing was flagged -- the author's "only see a symbol h"."""
        r = self._run(_flagged(self.base, 1, analytic_d=True, analytic_h=False))
        syms = _symbols(r)
        self.assertNotIn("h", syms, "an unflagged thickness must be substituted, not symbolic")
        self.assertNotIn("h2", syms)
        self.assertNotIn("h3", syms)
        self.assertIn("d11m2", syms, "the d flag must still produce its symbols")

    def test_nothing_flagged_leaves_no_symbol_at_all(self):
        r = self._run(self.base, analytical_d_symbolic=False)
        syms = _symbols(r)
        self.assertFalse({s for s in syms if s.startswith("h")},
                         f"no thickness symbol expected, got {sorted(syms)}")

    def test_spelling_does_not_depend_on_other_layers(self):
        """Pre-F61 the d suffix appeared only when 2+ layers were SHG-ACTIVE, so ONE layer could
        be spelled d11, d11m1 or d11m2 depending on unrelated layers. Making Au SHG-active must
        not rename quartz's symbols."""
        one = self._run(_flagged(self.base, 1, analytic_d=True))
        two = self._run(_flagged(_flagged(self.base, 1, analytic_d=True), 2, shg_active=True))
        self.assertTrue({"d11m2", "d14m2"} <= _symbols(one))
        self.assertTrue({"d11m2", "d14m2"} <= _symbols(two),
                        "quartz's symbols changed spelling because ANOTHER layer became active")

    def test_provenance_uses_the_same_numbers(self):
        r = self._run(_flagged(self.base, 1, analytic_h=True))
        syms = r.stages["symbols"]
        self.assertIn("h2 (symbolic)", syms["thickness"])
        self.assertIn("h3 =", syms["thickness"], "row 3 (Au) is substituted and named by its row")
        self.assertTrue(syms["layers"].startswith("2: "),
                        f"layers line must start at row 2, got {syms['layers']!r}")

    def test_known_d_binds_to_the_layer_it_was_typed_for(self):
        """A number typed into the grid describes the SELECTED layer. Pre-F61 the substitution
        sprayed the value over every layer's symbols."""
        sys_ = _flagged(_flagged(self.base, 1, analytic_d=True), 2,
                        analytic_d=True, shg_active=True)
        r = self._run(sys_, analytical_d_known={"d11m2": 2.5})
        syms = _symbols(r)
        self.assertNotIn("d11m2", syms, "the declared-known component must be substituted")
        self.assertIn("d14m2", syms, "the other components of that layer stay symbolic")
        self.assertTrue(any(s.endswith("m3") for s in syms),
                        f"row 3's symbols must survive a value declared for row 2: {sorted(syms)}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
