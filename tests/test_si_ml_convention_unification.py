"""Guard that the SI (single-interface) and ML (multilayer), numeric and symbolic
boundary solvers share ONE tangential-continuity convention.

Background (directive): SHAARP.si and SHAARP.ml were written separately and
historically handled some symbols/calculations slightly differently. In the Python
port the divergence was the tangential-vector ordering: the numeric solvers used the
SHAARP-faithful ``[Ex, Ey, Hx, Hy]`` (matching solveFresnelN's ``EE[[1;;2]]`` then
``getH[[1;;2]]``), while the symbolic single-interface solver used ``[Ex, Hx, Ey, Hy]``.
Both gave the same physics (a consistent row permutation), but the convention differed.

They are now converged onto the single shared ``shaarp.waves.tangential_eh`` helper.
This test pins that convergence so the SI/ML paths cannot silently drift apart again.
"""

import math
import unittest

import numpy as np
import sympy as sp

from shaarp.waves import make_isotropic_wave, tangential_eh
from shaarp.symbolic import make_isotropic_wave_symbolic, _tangential_sum_symbolic
from shaarp.boundary import _tangential_sum as _tangential_sum_numeric
from shaarp.multilayer_boundary import _tangential as _tangential_multilayer


class TangentialConventionUnificationTests(unittest.TestCase):
    def test_canonical_ordering_is_E_then_H(self):
        """SHAARP solveFresnelN orders each interface as the two tangential E
        components (x, y) THEN the two tangential H components."""
        e = [10.0, 20.0, 30.0]
        h = [40.0, 50.0, 60.0]
        self.assertEqual(tangential_eh(e, h), [10.0, 20.0, 40.0, 50.0])  # [Ex, Ey, Hx, Hy]

    def test_numeric_si_and_ml_tangential_identical(self):
        """boundary.py (SI) and multilayer_boundary.py (ML) numeric helpers must
        produce identical tangential vectors for the same wave."""
        w = make_isotropic_wave(n=2.3, theta_rad=0.4, polarization="p", omega=1.0)
        a = _tangential_sum_numeric([w], mu=1.0)
        b = _tangential_multilayer([w], mu=1.0)
        np.testing.assert_allclose(a, b, atol=0, rtol=0)

    def test_symbolic_matches_numeric_tangential_ordering(self):
        """The symbolic single-interface tangential vector must now equal the
        numeric one component-by-component (same [Ex,Ey,Hx,Hy] ordering)."""
        n, theta = 2.3, 0.4
        wn = make_isotropic_wave(n=n, theta_rad=theta, polarization="p", omega=1.0)
        ws = make_isotropic_wave_symbolic(n=n, theta_rad=theta, polarization="p", omega=1)
        num = _tangential_sum_numeric([wn], mu=1.0)
        sym = np.array([complex(x.evalf()) for x in _tangential_sum_symbolic([ws], mu=1)], dtype=complex)
        np.testing.assert_allclose(sym, num, atol=1e-12, rtol=1e-12)

    def test_symbolic_uses_E_then_H_not_interleaved(self):
        """Pin the exact ordering: index 0,1 are E (Ex,Ey); index 2,3 are H (Hx,Hy).
        Catches a regression back to the old interleaved [Ex,Hx,Ey,Hy]."""
        ws = make_isotropic_wave_symbolic(n=2.3, theta_rad=0.4, polarization="p", omega=1)
        e = sp.Matrix(ws.electric)
        h = ws.magnetic(mu=1)
        vec = _tangential_sum_symbolic([ws], mu=1)
        self.assertEqual(sp.simplify(vec[0] - e[0]), 0)  # Ex
        self.assertEqual(sp.simplify(vec[1] - e[1]), 0)  # Ey (NOT Hx)
        self.assertEqual(sp.simplify(vec[2] - h[0]), 0)  # Hx (NOT Ey)
        self.assertEqual(sp.simplify(vec[3] - h[1]), 0)  # Hy

    def test_pnl_factor2_cross_convention_shared(self):
        """The cross (eo / Fi-Bj) nonlinear source uses factor=2 in BOTH the SI and
        ML source builders -- already unified; pinned here as part of the convention
        contract."""
        from shaarp.nonlinear import compute_pnl_voigt
        from shaarp.symbolic import compute_pnl_voigt_symbolic

        d = np.zeros((3, 6))
        d[0, 3] = d[1, 4] = d[2, 5] = 0.7  # -43m-like
        e1 = np.array([0.3, -0.5, 0.8], dtype=complex)
        e2 = np.array([0.1, 0.9, -0.2], dtype=complex)
        num = compute_pnl_voigt(d, e1, e2, factor=2.0)
        s1 = sp.Matrix([complex(x) for x in e1])
        s2 = sp.Matrix([complex(x) for x in e2])
        sym = compute_pnl_voigt_symbolic(sp.Matrix(d.tolist()), s1, s2, factor=2)
        got = np.array([complex(x) for x in sym], dtype=complex)
        np.testing.assert_allclose(got, num, atol=1e-12, rtol=1e-12)


class NonlinearSourceUnificationTests(unittest.TestCase):
    """The pairwise 2-omega nonlinear-source generation was implemented three times
    with different symbols (SI ee/oo/eo over 2 modes; ML f1f1..f2b1 over 4 modes;
    symbolic SI ee/oo/eo). They are now all thin wrappers over ONE primitive
    (`pairwise_nonlinear_sources` / `pairwise_nonlinear_sources_symbolic`). These
    tests pin that convergence and the shared SHAARP factor rule
    (self pair -> factor 1, k=2k, omega=2omega; cross pair -> factor 2, k=k_i+k_j)."""

    def setUp(self):
        import numpy as np

        self.d = np.zeros((3, 6))
        self.d[0, 3] = self.d[1, 4] = self.d[2, 5] = 0.7  # -43m-like
        self.wa = make_isotropic_wave(n=2.1, theta_rad=0.3, polarization="p", omega=1.0)
        self.wb = make_isotropic_wave(n=2.1, theta_rad=0.3, polarization="s", omega=1.0)

    def test_si_sources_come_from_the_unified_primitive(self):
        import numpy as np
        from shaarp.nonlinear import single_interface_sources, pairwise_nonlinear_sources, _SI_SOURCE_PAIRS

        si = single_interface_sources(self.d, self.wa, self.wb)
        prim = pairwise_nonlinear_sources(self.d, [self.wa, self.wb], _SI_SOURCE_PAIRS)
        for label, src in (("ee", si.ee), ("oo", si.oo), ("eo", si.eo)):
            np.testing.assert_allclose(src.k, prim[label].k, atol=0, rtol=0)
            np.testing.assert_allclose(src.polarization, prim[label].polarization, atol=0, rtol=0)
            self.assertEqual(src.omega, prim[label].omega)

    def test_ml_sources_come_from_the_unified_primitive(self):
        import numpy as np
        from shaarp.nonlinear import multilayer_sources, pairwise_nonlinear_sources, _ML_SOURCE_PAIRS

        f1 = make_isotropic_wave(n=2.0, theta_rad=0.25, polarization="p", omega=1.0)
        f2 = make_isotropic_wave(n=2.0, theta_rad=0.25, polarization="s", omega=1.0)
        b1 = make_isotropic_wave(n=2.0, theta_rad=0.25, polarization="p", omega=1.0, direction_sign_z=-1)
        b2 = make_isotropic_wave(n=2.0, theta_rad=0.25, polarization="s", omega=1.0, direction_sign_z=-1)
        ml = multilayer_sources(self.d, f1, f2, b1, b2)
        prim = pairwise_nonlinear_sources(self.d, [f1, f2, b1, b2], _ML_SOURCE_PAIRS)
        for label in ("F1F1", "F2F2", "F1F2", "B1B1", "B2B2", "B1B2", "F1B1", "F2B2", "F1B2", "F2B1"):
            src = getattr(ml, label.lower())
            np.testing.assert_allclose(src.k, prim[label].k, atol=0, rtol=0)
            np.testing.assert_allclose(src.polarization, prim[label].polarization, atol=0, rtol=0)

    def test_same_wave_gives_identical_SI_and_ML_source_only_label_differs(self):
        """The decisive unification check: feed the SAME wave as an SI extraordinary
        mode (-> 'ee') and as an ML forward_1 mode. The resulting source
        (k, polarization, omega) must be IDENTICAL -- only the label differs. Same
        physics, one method, regardless of which notebook's naming you came in with."""
        import numpy as np
        from shaarp.nonlinear import single_interface_sources, multilayer_sources

        w = self.wa
        other = self.wb
        si = single_interface_sources(self.d, w, other)          # w is the 'extraordinary' -> ee = w*w
        ml = multilayer_sources(self.d, w, other, other, other)  # w is forward_1 -> F1F1 = w*w
        np.testing.assert_allclose(si.ee.k, ml.f1f1.k, atol=0, rtol=0)
        np.testing.assert_allclose(si.ee.polarization, ml.f1f1.polarization, atol=0, rtol=0)
        self.assertEqual(si.ee.omega, ml.f1f1.omega)
        self.assertEqual(si.ee.label, "ee")
        self.assertEqual(ml.f1f1.label, "F1F1")  # only the label differs

    def test_self_and_cross_factor_rule(self):
        import numpy as np
        from shaarp.nonlinear import pairwise_nonlinear_sources
        from shaarp.nonlinear import compute_pnl_voigt

        prim = pairwise_nonlinear_sources(self.d, [self.wa, self.wb], [(0, 0, "self"), (0, 1, "cross")])
        # self: factor 1 -> k = 2 k_a, omega = 2 omega_a
        np.testing.assert_allclose(prim["self"].k, 2 * self.wa.k, atol=0, rtol=0)
        self.assertEqual(prim["self"].omega, 2 * self.wa.omega)
        np.testing.assert_allclose(
            prim["self"].polarization, compute_pnl_voigt(self.d, self.wa.electric, self.wa.electric, factor=1), atol=0, rtol=0
        )
        # cross: factor 2 -> k = k_a + k_b
        np.testing.assert_allclose(prim["cross"].k, self.wa.k + self.wb.k, atol=0, rtol=0)
        np.testing.assert_allclose(
            prim["cross"].polarization, compute_pnl_voigt(self.d, self.wa.electric, self.wb.electric, factor=2), atol=0, rtol=0
        )

    def test_factor_intention_pairwise_sources_reconstruct_total_polarization(self):
        """INTENTION of the self=1 / cross=2 factor: it is the ab+ba degeneracy that
        makes the pairwise sources sum to the nonlinear polarization of the TOTAL
        (summed) fundamental field. If this invariant holds, the factor rule is
        semantically correct AND the pair list is a COMPLETE decomposition -- not just
        numerically matching the old code. This is the physical reason the merge is valid."""
        import numpy as np
        from shaarp.nonlinear import compute_pnl_voigt, _ML_SOURCE_PAIRS

        d = np.zeros((3, 6))
        d[0, 3] = d[1, 4] = d[2, 5] = 0.7
        d[0, 0] = 0.3
        d[1, 1] = 0.4  # general-ish d so the test is not accidentally trivial

        # 2-mode (SI): P(Ea+Eb) == ee + oo + eo
        Ea = np.array([0.3, -0.5, 0.8], dtype=complex)
        Eb = np.array([0.1, 0.9, -0.2], dtype=complex)
        total2 = compute_pnl_voigt(d, Ea + Eb, Ea + Eb, factor=1)
        recon2 = (
            compute_pnl_voigt(d, Ea, Ea, factor=1)
            + compute_pnl_voigt(d, Eb, Eb, factor=1)
            + compute_pnl_voigt(d, Ea, Eb, factor=2)
        )
        np.testing.assert_allclose(recon2, total2, atol=1e-14, rtol=0)

        # 4-mode (ML): P == sum of all 10 pairwise sources
        F = [
            np.array([0.3, -0.5, 0.8], dtype=complex),
            np.array([0.1, 0.9, -0.2], dtype=complex),
            np.array([-0.4, 0.2, 0.6], dtype=complex),
            np.array([0.7, -0.1, -0.3], dtype=complex),
        ]
        total4 = compute_pnl_voigt(d, sum(F), sum(F), factor=1)
        recon4 = np.zeros(3, dtype=complex)
        for i, j, _label in _ML_SOURCE_PAIRS:
            recon4 = recon4 + compute_pnl_voigt(d, F[i], F[j], factor=(1 if i == j else 2))
        np.testing.assert_allclose(recon4, total4, atol=1e-14, rtol=0)

    def test_symbolic_si_sources_use_symbolic_primitive(self):
        import sympy as sp
        from shaarp.symbolic import (
            single_interface_sources_symbolic,
            pairwise_nonlinear_sources_symbolic,
            make_isotropic_wave_symbolic,
        )

        d = sp.Matrix(self.d.tolist())
        e = make_isotropic_wave_symbolic(n=2.1, theta_rad=0.3, polarization="p", omega=1)
        o = make_isotropic_wave_symbolic(n=2.1, theta_rad=0.3, polarization="s", omega=1)
        si = single_interface_sources_symbolic(d, e, o)
        prim = pairwise_nonlinear_sources_symbolic(d, [e, o], [(0, 0, "ee"), (1, 1, "oo"), (0, 1, "eo")])
        for label, src in (("ee", si.ee), ("oo", si.oo), ("eo", si.eo)):
            self.assertEqual(sp.simplify(sp.Matrix(src.polarization) - sp.Matrix(prim[label].polarization)), sp.zeros(3, 1))


if __name__ == "__main__":
    unittest.main()
