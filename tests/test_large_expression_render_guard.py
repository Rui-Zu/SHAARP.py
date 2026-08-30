"""RENDER GUARD for pathologically large symbolic expressions.

THE DEFECT. Clicking Update on the shipped case ML | GaAs (111) 800 nm | Partial Analytical froze
the GUI for ~8 minutes (471 s, reproduced independently; that material's sweep group took 1269.7 s
against 79-92 s for every other ML material). The matrix sweep had been passing it for the whole
life of the project, because a slow cell only prints `[SLOW ...]` -- it is not a failure.

THE CAUSE, profiled rather than guessed. Of 1089 s inside compute_ml_gui_result, **1065 s (98%)
was sympy STRING PRINTING** (sstr / _print_Add / _print_Mul / as_ordered_terms, incl. 2.1M
Expr.__complex__ calls while ordering terms). The physics solve is ~24 s. The reflected amplitude
is an Add with only **11 top-level terms** that renders to **21,167,857 characters (21 MB)**: it is
deep, not wide. The size comes from GaAs being ABSORBING at 800 nm (Im eps(2w) = 17.6, since 400 nm
is deep in its absorption), so every coefficient is complex. A dense rotated d was tested and
REFUTED as the explanation -- GaAs(111) has 8/18 nonzero lab d components, exactly like LiNbO3
z-cut, which renders ~20x faster.

WHAT THESE TESTS PIN. The guard must (1) leave every normal expression BYTE-IDENTICAL -- it may
never become a silent transformation of displayed physics; (2) summarize instead of rendering when
the tree is enormous; (3) cost O(limit), never O(tree) -- a guard that walks a multi-million-node
tree to decide not to print it has just moved the cost. An earlier version of this guard keyed on
len(expr.args) and never fired at all (11 <= any term threshold), which is exactly why (2) is
asserted on a genuinely deep expression rather than a wide one.
"""

from __future__ import annotations

import time
import unittest

import sympy as sp

from shaarp.api import _RENDER_NODE_LIMIT, _expr_node_count, _render_or_summarize


def _deep_expression(rounds: int):
    """A structurally deep expression: each round squares the node count."""
    x, y = sp.symbols("x y")
    e = x + y
    for i in range(rounds):
        e = sp.Add(sp.Mul(e, e + i, evaluate=False), i, evaluate=False)
    return e


class RenderGuardLeavesNormalExpressionsIdentical(unittest.TestCase):
    """(1) The guard must be invisible for everything the panel actually renders."""

    def test_small_expressions_are_byte_identical_to_str(self):
        phi, d14, h = sp.symbols("phi d14 h")
        for expr in (
            sp.Integer(0),
            d14 * sp.sin(2 * phi),
            (d14 * sp.sin(2 * phi) + 3 * sp.cos(phi) ** 2) / (1 + sp.sqrt(2)),
            sp.exp(sp.I * h) * d14 + sp.Rational(1, 3),
        ):
            with self.subTest(expr=str(expr)[:40]):
                self.assertEqual(_render_or_summarize(expr), str(expr))

    def test_a_realistically_sized_expression_is_still_rendered_in_full(self):
        # ~ the scale of a healthy ML/SI panel expression: fully rendered, not summarized.
        phi, h = sp.symbols("phi h")
        expr = sum(sp.Symbol(f"d{i}") * sp.sin(i * phi) * sp.exp(sp.I * i * h) for i in range(1, 40))
        out = _render_or_summarize(expr)
        self.assertEqual(out, str(expr))
        self.assertNotIn("too large to display", out)


class RenderGuardSummarizesPathologicalExpressions(unittest.TestCase):
    """(2) A tree too big to print must be summarized -- and the summary must say why."""

    def test_deep_expression_is_summarized_not_printed(self):
        expr = _deep_expression(22)          # deep, but only a handful of TOP-LEVEL args
        self.assertLessEqual(len(expr.args), 8,
                             "this fixture must be DEEP, not wide -- a term-count guard would miss it")
        out = _render_or_summarize(expr)
        self.assertIn("too large to display", out)
        self.assertIn("ABSORBING", out, "the summary must explain WHY this happens")
        self.assertLess(len(out), 2000, "the summary must be short; the point is not to build a huge string")


class RenderGuardIsCheap(unittest.TestCase):
    """(3) The guard must cost O(limit), not O(tree) -- else it becomes the problem it prevents."""

    def test_node_count_stops_at_the_limit(self):
        expr = _deep_expression(22)
        self.assertLessEqual(_expr_node_count(expr, 500), 501,
                             "the bounded walk kept counting past its limit")

    def test_guard_returns_quickly_on_a_pathological_expression(self):
        expr = _deep_expression(22)
        started = time.time()
        _render_or_summarize(expr)
        elapsed = time.time() - started
        # The real defect took 461 s to render. The bounded walk is ~200k node visits; 30 s is
        # enormous headroom on any machine while still failing loudly if the bound is removed.
        self.assertLess(elapsed, 30.0, f"render guard took {elapsed:.1f}s -- is the walk still bounded?")

    def test_limit_is_a_positive_int(self):
        self.assertIsInstance(_RENDER_NODE_LIMIT, int)
        self.assertGreater(_RENDER_NODE_LIMIT, 1000)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
