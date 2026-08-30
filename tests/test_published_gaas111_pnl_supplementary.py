"""Validate the Python GaAs(111) nonlinear polarization (P_NL) stage against the
PUBLISHED supplementary of the SHAARP paper, form-vs-form.

Reference: R. Zu et al., "Analytical and numerical modeling of optical second
harmonic generation in anisotropic crystals using ##SHAARP package",
npj Comput. Mater. 8, 246 (2022), Supplementary Note 5
("Full Analytical SHG Polarimetry Expressions for GaAs (111)"),
file 41524_2022_930_MOESM1_ESM.pdf, equations 20-28.

This is the authoritative published ground truth for the P_NL stage of the
full-analytical SHG chain. The published expressions are the nonlinear source
P_{Li}^{T,(ee|oo|eo),2w} for GaAs (cubic -43m, d14=d25=d36) cut on the (111)
surface, in the lab frame L1-L2-L3 with the PoI in the L1-L3 plane.

The test does NOT compare messy symbolic forms. It (a) builds the Python lab-frame
SHG source symbolically and (b) compares it to the published polynomial by
exhaustive random numeric substitution over all free field symbols
(Schwartz-Zippel: agreement at enough generic points => identical polynomials),
to the rounding floor of the published 5-decimal coefficients.

Convention note (validated here): SHAARP's GaAs(111) lab frame has L1 along the
in-plane direction [1,-1,0]/sqrt(2); i.e. ``from_cubic_miller_surface([1,1,1],
in_plane_hint=(1,-1,0))``. This is the convention that reproduces eq 20-28.
"""

import unittest

import numpy as np
import sympy as sp

from shaarp.config import CrystalOrientation
from shaarp.nonlinear import compute_pnl_voigt
from shaarp.symbolic import compute_pnl_voigt_symbolic
from shaarp.tensors import rotate_d_voigt_crystal_to_lab


# Published GaAs(111) lab-frame in-plane convention (L1 along [1,-1,0]).
IN_PLANE_HINT = np.array([1.0, -1.0, 0.0])
# The published coefficients are rounded to 5 decimals -> agreement floor ~1e-5.
PUB_TOL = 1e-4
N_PROBE = 24


def _gaas111_lab_d_tensor():
    """Rotate the -43m d-tensor (d14=d25=d36=1) to the published GaAs(111) lab frame."""
    d = np.zeros((3, 6))
    d[0, 3] = d[1, 4] = d[2, 5] = 1.0  # d14 = d25 = d36 = 1
    ori = CrystalOrientation.from_cubic_miller_surface(
        np.array([1.0, 1.0, 1.0]), in_plane_hint=IN_PLANE_HINT
    )
    R = np.asarray(ori.rotation_matrix(), dtype=float)
    d_lab = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d, R), dtype=complex))
    return sp.Matrix(d_lab.tolist())


def _pnl(d_lab, field):
    return [sp.expand(x) for x in compute_pnl_voigt_symbolic(d_lab, sp.Matrix(field))]


def _max_residual(expr, symbols, rng):
    """Max |expr| over N_PROBE random real substitutions of `symbols`."""
    worst = 0.0
    for _ in range(N_PROBE):
        subs = {s: rng.uniform(-2.0, 2.0) for s in symbols}
        worst = max(worst, abs(complex(expr.subs(subs))))
    return worst


class PublishedGaAs111PNLTests(unittest.TestCase):
    """Form-vs-form agreement of the Python P_NL with published eq 20-28."""

    @classmethod
    def setUpClass(cls):
        cls.d_lab = _gaas111_lab_d_tensor()
        cls.rng = np.random.default_rng(20221122)  # paper year/issue, deterministic
        cls.e1, cls.e2, cls.e3 = sp.symbols("e1 e2 e3")
        cls.o1, cls.o2, cls.o3 = sp.symbols("o1 o2 o3")
        cls.Pee = _pnl(cls.d_lab, [cls.e1, cls.e2, cls.e3])
        cls.Poo = _pnl(cls.d_lab, [cls.o1, cls.o2, cls.o3])
        Psum = _pnl(cls.d_lab, [cls.e1 + cls.o1, cls.e2 + cls.o2, cls.e3 + cls.o3])
        # eo cross source = P(e+o) - P(e) - P(o) = 2 d_ijk E_j^e E_k^o
        cls.Peo = [sp.expand(Psum[i] - cls.Pee[i] - cls.Poo[i]) for i in range(3)]

    def test_ee_source_matches_published_eq20_22(self):
        e1, e2, e3 = self.e1, self.e2, self.e3
        pub = [
            1.63299 * e1 * e2 - 1.15470 * e1 * e3,                       # eq 20
            0.81650 * e1**2 - 0.81650 * e2**2 - 1.15470 * e2 * e3,       # eq 21
            -0.57735 * e1**2 - 0.57735 * e2**2 + 1.15470 * e3**2,        # eq 22
        ]
        for i in range(3):
            r = _max_residual(self.Pee[i] - pub[i], [e1, e2, e3], self.rng)
            self.assertLess(r, PUB_TOL, f"P_L{i+1}^ee deviates from published eq {20+i}: {r:.2e}")

    def test_oo_source_matches_published_eq23_25(self):
        o1, o2, o3 = self.o1, self.o2, self.o3
        pub = [
            1.63299 * o1 * o2 - 1.15470 * o1 * o3,                       # eq 23
            0.81650 * o1**2 - 0.81650 * o2**2 - 1.15470 * o2 * o3,       # eq 24
            -0.57735 * o1**2 - 0.57735 * o2**2 + 1.15470 * o3**2,        # eq 25
        ]
        for i in range(3):
            r = _max_residual(self.Poo[i] - pub[i], [o1, o2, o3], self.rng)
            self.assertLess(r, PUB_TOL, f"P_L{i+1}^oo deviates from published eq {23+i}: {r:.2e}")

    def test_eo_cross_source_matches_published_eq26_28(self):
        e1, e2, e3 = self.e1, self.e2, self.e3
        o1, o2, o3 = self.o1, self.o2, self.o3
        # Symmetric reconstruction of eq 26-28 (the PDF text dropped some field
        # factors; the symmetric eo cross source = 2 d_ijk E_j^e E_k^o has these
        # paired terms, with cross coefficients = 2x the ee/oo diagonal ones).
        pub = [
            1.63299 * (o1 * e2 + e1 * o2) - 1.15470 * (o1 * e3 + e1 * o3),   # eq 26
            1.63299 * e1 * o1 - 1.63299 * e2 * o2 - 1.15470 * (o2 * e3 + e2 * o3),  # eq 27
            -1.15470 * e1 * o1 - 1.15470 * e2 * o2 + 2.30940 * e3 * o3,      # eq 28
        ]
        syms = [e1, e2, e3, o1, o2, o3]
        for i in range(3):
            r = _max_residual(self.Peo[i] - pub[i], syms, self.rng)
            self.assertLess(r, PUB_TOL, f"P_L{i+1}^eo deviates from published eq {26+i}: {r:.2e}")

    def test_published_signature_coefficients_present(self):
        """Guard the published numeric fingerprints (2*sqrt(2/3)=1.63299 etc.)."""
        # P_L1^ee = 1.63299 e1 e2 - 1.15470 e1 e3: coefficient of e1*e2 must be 2*sqrt(2/3).
        c = sp.Poly(self.Pee[0], self.e1, self.e2, self.e3).coeff_monomial(self.e1 * self.e2)
        self.assertAlmostEqual(float(c), 2.0 * np.sqrt(2.0 / 3.0), places=4)
        self.assertAlmostEqual(float(c), 1.63299, places=4)
        # eo L3 e3 o3 coefficient must be 2.30940 = 4/sqrt(3).
        c2 = sp.Poly(self.Peo[2], self.e3, self.o3).coeff_monomial(self.e3 * self.o3)
        self.assertAlmostEqual(float(c2), 4.0 / np.sqrt(3.0), places=4)
        self.assertAlmostEqual(float(c2), 2.30940, places=4)

    def test_pnl_handles_complex_d14_and_complex_fields(self):
        """SHAARP targets absorbing/anisotropic media: d14 and the driving fields
        are COMPLEX. The published coefficients are real geometric factors that must
        multiply complex d14 and complex E correctly. Validate eq 20-22 over complex
        d14 and complex e1,e2,e3 (not just the real case used above)."""
        d14 = sp.symbols("d14")
        d = np.zeros((3, 6))
        d[0, 3] = d[1, 4] = d[2, 5] = 1.0
        ori = CrystalOrientation.from_cubic_miller_surface(
            np.array([1.0, 1.0, 1.0]), in_plane_hint=IN_PLANE_HINT
        )
        R = np.asarray(ori.rotation_matrix(), dtype=float)
        geom = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d, R), dtype=complex))
        M = sp.Matrix(geom.tolist()) * d14  # complex d14 times real geometric rotation
        e1, e2, e3 = self.e1, self.e2, self.e3
        pnl = _pnl(M, [e1, e2, e3])
        pub = [
            d14 * (1.63299 * e1 * e2 - 1.15470 * e1 * e3),
            d14 * (0.81650 * e1**2 - 0.81650 * e2**2 - 1.15470 * e2 * e3),
            d14 * (-0.57735 * e1**2 - 0.57735 * e2**2 + 1.15470 * e3**2),
        ]
        syms = [d14, e1, e2, e3]
        for i in range(3):
            worst = 0.0
            for _ in range(N_PROBE):
                subs = {s: complex(self.rng.uniform(-2, 2), self.rng.uniform(-2, 2)) for s in syms}
                worst = max(worst, abs(complex((pnl[i] - pub[i]).subs(subs))))
            self.assertLess(worst, PUB_TOL, f"complex P_L{i+1} deviates from eq {20+i}: {worst:.2e}")

    def test_in_plane_convention_is_required(self):
        """The published form requires L1 along [1,-1,0]; a wrong in-plane hint
        does NOT reproduce eq 20 (sanity that the convention is load-bearing)."""
        d = np.zeros((3, 6))
        d[0, 3] = d[1, 4] = d[2, 5] = 1.0
        ori = CrystalOrientation.from_cubic_miller_surface(
            np.array([1.0, 1.0, 1.0]), in_plane_hint=np.array([1.0, 1.0, -2.0])
        )
        R = np.asarray(ori.rotation_matrix(), dtype=float)
        d_lab = sp.Matrix(np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d, R), dtype=complex)).tolist())
        e1, e2, e3 = self.e1, self.e2, self.e3
        pnl = _pnl(d_lab, [e1, e2, e3])
        pub_L1 = 1.63299 * e1 * e2 - 1.15470 * e1 * e3
        r = _max_residual(pnl[0] - pub_L1, [e1, e2, e3], self.rng)
        self.assertGreater(r, PUB_TOL, "different in-plane hint unexpectedly matched eq 20")


class GaAs111ThreefoldSymmetryTests(unittest.TestCase):
    """Physics cross-check INDEPENDENT of the published equations: GaAs is -43m (Td),
    which has a 3-fold (C3) rotation axis along each <111> body diagonal. So the (111)
    surface has C3v symmetry, and rotating the sample about its [111] normal must make
    the SHG response EXACTLY 120-degree periodic. This validates the (111) lab-frame
    convention (L1 along [1,-1,0]) against the crystal's known symmetry, not against the
    paper -- a fully independent confirmation that the chain encodes the right crystal."""

    @staticmethod
    def _rz(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def setUp(self):
        d = np.zeros((3, 6))
        d[0, 3] = d[1, 4] = d[2, 5] = 1.0
        ori = CrystalOrientation.from_cubic_miller_surface(
            np.array([1.0, 1.0, 1.0]), in_plane_hint=IN_PLANE_HINT
        )
        self.d_lab0 = np.real(
            np.asarray(rotate_d_voigt_crystal_to_lab(d, np.asarray(ori.rotation_matrix(), dtype=float)), dtype=complex)
        )

    def _d_rotated(self, psi):
        return np.real(np.asarray(rotate_d_voigt_crystal_to_lab(self.d_lab0, self._rz(psi)), dtype=complex))

    def test_d_tensor_is_120deg_periodic_about_111_normal(self):
        worst = 0.0
        for p in np.linspace(0.0, 2 * np.pi, 37):
            worst = max(worst, np.max(np.abs(self._d_rotated(p) - self._d_rotated(p + np.radians(120)))))
        self.assertLess(worst, 1e-12, f"GaAs(111) d-tensor not 120-deg periodic about [111]: {worst:.2e}")

    def test_symmetry_is_genuinely_threefold_not_sixfold(self):
        # A 60-deg rotation must NOT be a symmetry (else we'd have mislabeled the axis).
        diff60 = max(
            np.max(np.abs(self._d_rotated(np.radians(p)) - self._d_rotated(np.radians(p + 60))))
            for p in (10.0, 40.0, 75.0)
        )
        self.assertGreater(diff60, 0.5, "rotation by 60 deg unexpectedly a symmetry (should be 3-fold, not 6-fold)")

    def test_shg_source_polarimetry_inherits_threefold_and_varies(self):
        import math

        th = math.radians(35.0)
        E = np.array([math.cos(th), 0.4, math.sin(th)], dtype=complex)  # arbitrary fixed fundamental

        def power(psi):
            P = compute_pnl_voigt(self._d_rotated(psi).astype(complex), E, E)
            return float(np.vdot(P, P).real)

        worst = max(abs(power(p) - power(p + np.radians(120))) for p in np.linspace(0.0, 2 * np.pi, 37))
        self.assertLess(worst, 1e-12, f"SHG |P_NL|^2 not 120-deg periodic: {worst:.2e}")
        samples = [power(p) for p in np.linspace(0.0, np.radians(120), 24)]
        self.assertGreater(max(samples) - min(samples), 0.1, "polarimetry is ~constant (no real 3-fold pattern)")


if __name__ == "__main__":
    unittest.main()
