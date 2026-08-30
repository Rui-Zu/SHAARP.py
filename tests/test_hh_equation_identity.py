"""Fence: the archived analyticHH.mx expression IS the Herman & Hayden (1995) equation.

Wraps scripts/verify_hh_equation_identity.py: every literal constant in the ported expression must
reconstruct exactly from {L, lambda, n_w, n_2w, quartz d11}, and the from-physics reconstruction of
the full HH structure (sinc^2 coherence + free-wave Fabry-Perot MR + backward-wave cross term) must
equal the archived curve up to a single units prefactor. Guards against the circularity of only
comparing artifacts to each other (the requirement being to "get the correct HH
equation").
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


class HHEquationIdentityTests(unittest.TestCase):
    def test_archived_expression_is_the_herman_hayden_equation(self):
        import verify_hh_equation_identity as v

        # main() raises SystemExit on any failed identity; a clean return is the pass.
        try:
            v.main()
        except SystemExit as exc:  # pragma: no cover - failure path
            self.fail(f"HH equation identity failed: {exc}")


if __name__ == "__main__":
    unittest.main()
