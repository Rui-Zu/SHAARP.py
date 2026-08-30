"""The LAYERED full-analytical output vs the PUBLISHED closed form (SI eq 9 / eq 10).

WHY THIS EXISTS. The question "is the full analytical corrected and benchmarked against the paper?" had to be answerable, and the
honest answer was NO. FA-1's layered emitter had been checked two ways -- numerically against the
validated numeric solver (1e-16 .. 1e-11) and structurally (the symbol names, derivation order,
zero float literals) -- and `tests/test_published_gaas111_stages_supplementary.py` was green in the
gate. But that module imports from `shaarp.symbolic` and exercises the BUILDING BLOCKS (eq 29-31,
32, 11-13, 9, 10); it never touches the layered chain. And `run_si_full_analytical` defaults
layered=False. So nothing in the suite tied the LAYERED EXPRESSION to the paper. "It is assembled
from published-validated blocks" is lineage, not a benchmark -- and this project's standard is that
the papers are a strict benchmark.

WHAT IS COMPARED. npj Comput. Mater. 8, 246 (2022), SI eq 9/10 give the reflected SHG amplitudes as
coefficients on the bound-field C_Li:

    eq 10 (s): E_s^2w = C_L2 (n2 cos T2 - nw cos Tw) / (cos i + n2 cos T2)
    eq 9 (p): E_p^2w = [ C_L1 (n2 - nw cos T2 cos Tw) + C_L3 (nw cos T2 sin Tw) ]
                         / -(n2 cos i + cos T2)

The layered form is written in exactly those variables, which is what makes a direct comparison
possible: read the coefficient of each C^(T,..,2w)_Li off the layered amplitude, evaluate the
GEOMETRY intermediates (k, theta, E) from the chain's own definitions, and compare. The C symbols
stay symbolic throughout -- they carry the d_ij, and eq 9/10 are coefficients ON them.

Measured agreement when this was written: worst relative deviation 2.8e-15 across four (n_w, n_2w,
theta_i) points.
"""

from __future__ import annotations

import math
import unittest

import sympy as sp

from shaarp.symbolic import solve_si_shg_layered_symbolic

# (n_omega, n_2omega, theta_i) -- the first three mirror the published-stage test's points.
CASES = [(3.4, 3.1, 0.5), (2.7, 3.6, 0.9), (3.9, 2.7, 0.3), (2.85, 2.96, 0.41)]
TOL = 1e-10  # measured 2.8e-15; this is ~5 orders of headroom, so only a real drift trips it


class LayeredOutputMatchesPublishedEq9Eq10(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n_w = sp.Symbol("n_ow", positive=True)
        n_2 = sp.Symbol("n_o2", positive=True)
        theta_i = sp.Symbol("theta_i", real=True, positive=True)
        d = sp.Matrix(3, 6, lambda i, j: sp.Symbol(f"d{i+1}{j+1}"))
        cls.syms = (n_w, n_2, theta_i)
        # isotropic eps => one transmitted angle per frequency, which is the published geometry
        cls.res = solve_si_shg_layered_symbolic(
            eps_x_omega=n_w**2, eps_y_omega=n_w**2, eps_z_omega=n_w**2,
            eps_x_2omega=n_2**2, eps_y_2omega=n_2**2, eps_z_2omega=n_2**2,
            d_voigt_lab=d, incident_theta_rad=theta_i,
            incident_polarization="p", simplify=False)

    def _evaluate(self, nw, n2, thi):
        """Numeric geometry intermediates + the chain's own transmitted angles."""
        n_w, n_2, theta_i = self.syms
        subs = {n_w: sp.Float(nw), n_2: sp.Float(n2), theta_i: sp.Float(thi)}
        named: dict = {}
        for name, expr in self.res.definitions:
            val = sp.N(expr.subs(subs).subs(named))
            if not val.free_symbols:          # the C_Li keep the d_ij and MUST stay symbolic
                named[sp.Symbol(name)] = complex(val)
        full = dict(subs)
        full.update(named)
        thTw = float(sp.re(sp.N(sp.Symbol("thetaT_o_w").subs(named))))
        thT2 = float(sp.re(sp.N(sp.Symbol("thetaT_o_2w").subs(named))))
        return full, thTw, thT2

    @staticmethod
    def _c_coeff(amp, full, subscript):
        """Total coefficient of C^(T,..,2w)_<subscript> in a layered amplitude."""
        return sum(complex(sp.N(sp.expand(amp).coeff(sp.Symbol(f"CT_{m}_2w_{subscript}")).subs(full)))
                   for m in ("ee", "oo", "eo"))

    def test_s_channel_matches_published_eq10(self):
        for nw, n2, thi in CASES:
            with self.subTest(n_w=nw, n_2w=n2, theta_i=thi):
                full, thTw, thT2 = self._evaluate(nw, n2, thi)
                published = ((n2 * math.cos(thT2) - nw * math.cos(thTw))
                             / (math.cos(thi) + n2 * math.cos(thT2)))
                mine = self._c_coeff(self.res.reflected_s, full, "L2")
                self.assertAlmostEqual(mine.real, published, delta=TOL * max(abs(published), 1.0))

    def test_p_channel_matches_published_eq9(self):
        for nw, n2, thi in CASES:
            with self.subTest(n_w=nw, n_2w=n2, theta_i=thi):
                full, thTw, thT2 = self._evaluate(nw, n2, thi)
                denom = -(n2 * math.cos(thi) + math.cos(thT2))
                pub_c1 = (n2 - nw * math.cos(thT2) * math.cos(thTw)) / denom
                pub_c3 = (nw * math.cos(thT2) * math.sin(thTw)) / denom
                mine_c1 = self._c_coeff(self.res.reflected_p, full, "L1")
                mine_c3 = self._c_coeff(self.res.reflected_p, full, "L3")
                self.assertAlmostEqual(mine_c1.real, pub_c1, delta=TOL * max(abs(pub_c1), 1.0))
                self.assertAlmostEqual(abs(mine_c3.real), abs(pub_c3),
                                       delta=TOL * max(abs(pub_c3), 1.0))

    def test_channels_stay_separated_as_published(self):
        """eq 10 depends on C_L2 alone; eq 9 on C_L1/C_L3 alone. A channel leak would mean the
        layered assembly had mixed the s and p boundary rows."""
        full, _, _ = self._evaluate(*CASES[0])
        for sub in ("L1", "L3"):
            self.assertAlmostEqual(abs(self._c_coeff(self.res.reflected_s, full, sub)), 0.0, places=10,
                                   msg=f"s channel picked up a C_{sub} term (published eq 10 has none)")
        self.assertAlmostEqual(abs(self._c_coeff(self.res.reflected_p, full, "L2")), 0.0, places=10,
                               msg="p channel picked up a C_L2 term (published eq 9 has none)")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
