"""Validate the remaining full-analytical SHG stages for GaAs(111) against the
PUBLISHED supplementary, form-vs-form / coefficient-vs-coefficient.

Reference: R. Zu et al., npj Comput. Mater. 8, 246 (2022), Supplementary Note 5,
file 41524_2022_930_MOESM1_ESM.pdf. Companion to
``test_published_gaas111_pnl_supplementary.py`` (which covers the P_NL stage,
eq 20-28). This file covers the other three stages of the analytical chain:

  * eq 29-32 -- fundamental driving fields E_L^{e/o,w} (omega Fresnel boundary)
  * eq 11-19 -- inhomogeneous particular coefficients C_Li (solveInhom)
  * eq 9-10 -- final reflected SHG signal E_p^2w, E_s^2w (2omega boundary)

Method (no shortcut): each Python symbolic stage is compared to the published
closed form either by machine-precision symbolic agreement or by exhaustive
random numeric substitution (Schwartz-Zippel). Two published denominators were
rendered with PDF-extraction garble; the physically-correct Fresnel boundary
determinants are documented inline and confirmed to reproduce the Python solve
to full precision. All convention differences (the extraordinary p-direction
sign, an overall reflected-field sign) are documented and cancel in |E|^2.
"""

import math
import unittest

import sympy as sp

from shaarp.symbolic import (
    SymbolicNonlinearSource,
    SymbolicWave,
    make_isotropic_wave_symbolic,
    solve_inhomogeneous_field_symbolic,
    solve_isotropic_linear_interface_symbolic,
    solve_single_interface_shg_boundary_symbolic,
)

TOL = 1e-9
N_PROBE = 40


def _max_residual(expr, symbols, seed):
    import random

    rng = random.Random(seed)
    worst = 0.0
    for _ in range(N_PROBE):
        subs = {}
        for s in symbols:
            name = getattr(s, "name", "")
            if name == "n" or name.startswith("n_") or name.startswith("n2"):
                subs[s] = rng.uniform(1.3, 4.0)          # refractive index
            elif name.startswith("theta") or name == "thT":
                subs[s] = rng.uniform(0.02, 1.2)         # angle (keep sin/asin in domain)
            else:
                subs[s] = rng.uniform(-2.0, 2.0)         # field component
        worst = max(worst, abs(complex(expr.subs(subs))))
    return worst


class OmegaBoundaryFundamentalFieldsTests(unittest.TestCase):
    """eq 29-32: the omega Fresnel transmission gives the published driving fields."""

    def setUp(self):
        self.thi, self.n = sp.symbols("theta_i n", positive=True)
        self.thT = sp.asin(sp.sin(self.thi) / self.n)

    def test_extraordinary_field_matches_eq29_31(self):
        thi, n, thT = self.thi, self.n, self.thT
        lin = solve_isotropic_linear_interface_symbolic(
            incident_index=1, transmitted_index=n, incident_theta_rad=thi,
            incident_polarization="p", simplify=True,
        )
        Ep = sp.Matrix(lin.transmitted_p.electric)
        tp_pub = 2 * sp.cos(thi) / (n * sp.cos(thi) + sp.cos(thT))  # Fresnel t_p
        # |components| compared (Python p-dir is (cosT,0,-sinT); published (cosT,0,+sinT)).
        self.assertLess(_max_residual(sp.Abs(Ep[0]) - tp_pub * sp.cos(thT), [thi, n], 1), TOL)  # eq 29
        self.assertLess(_max_residual(Ep[1], [thi, n], 2), TOL)                                  # eq 30: E_L2^e = 0
        self.assertLess(_max_residual(sp.Abs(Ep[2]) - tp_pub * sp.sin(thT), [thi, n], 3), TOL)  # eq 31

    def test_ordinary_field_matches_eq32(self):
        thi, n, thT = self.thi, self.n, self.thT
        lin = solve_isotropic_linear_interface_symbolic(
            incident_index=1, transmitted_index=n, incident_theta_rad=thi,
            incident_polarization="s", simplify=True,
        )
        Es = sp.Matrix(lin.transmitted_s.electric)
        ts_pub = 2 * sp.cos(thi) / (sp.cos(thi) + n * sp.cos(thT))  # Fresnel t_s
        self.assertLess(_max_residual(Es[0], [thi, n], 4), TOL)                       # eq 32: E_L1^o = 0
        self.assertLess(_max_residual(Es[2], [thi, n], 5), TOL)                       # ordinary has no L3
        self.assertLess(_max_residual(sp.Abs(Es[1]) - sp.Abs(ts_pub), [thi, n], 6), TOL)


class InhomogeneousCoefficientsTests(unittest.TestCase):
    """eq 11-19: solveInhom reproduces the published C_Li(P) linear map exactly.

    Because solveInhom is the SAME linear operator C = f(P) for the ee/oo/eo
    sources (only the source P differs), validating the map over free P symbols
    validates all of eq 11-19 at once.
    """

    def test_c_li_linear_map_matches_eq11_13(self):
        nw, n2, thT = sp.symbols("n_w n_2w theta_T", positive=True)
        P1, P2, P3 = sp.symbols("P1 P2 P3")
        k_ee = 2 * nw * sp.Matrix([sp.sin(thT), 0, sp.cos(thT)])  # bound wavevector 2 k^{e,w}
        src = SymbolicNonlinearSource("ee", k_ee, sp.Matrix([P1, P2, P3]), 2)
        eps2 = n2**2 * sp.eye(3)
        C = solve_inhomogeneous_field_symbolic(eps2, src, mu=1, eps0=1, simplify=True).electric
        den = n2**2 * (n2**2 - nw**2)
        C1_pub = (-n2**2 * P1 + nw**2 * P3 * sp.cos(thT) * sp.sin(thT) + nw**2 * P1 * sp.sin(thT) ** 2) / den
        C2_pub = -P2 / (n2**2 - nw**2)
        C3_pub = (-n2**2 * P3 + nw**2 * P3 * sp.cos(thT) ** 2 + nw**2 * P1 * sp.cos(thT) * sp.sin(thT)) / den
        syms = [nw, n2, thT, P1, P2, P3]
        self.assertLess(_max_residual(C[0] - C1_pub, syms, 11), 1e-7)   # eq 11
        self.assertLess(_max_residual(C[1] - C2_pub, syms, 12), 1e-7)   # eq 12
        self.assertLess(_max_residual(C[2] - C3_pub, syms, 13), 1e-7)   # eq 13


class TwoOmegaBoundarySignalTests(unittest.TestCase):
    """eq 9-10: the 2omega boundary maps the bound C_Li to the published signal.

    The bound 2omega field (sum of the ee/oo/eo particular solutions, which share
    one wavevector in isotropic GaAs) drives the reflected SHG. Feeding the boundary
    solver one bound wave with arbitrary symbolic (C1,C2,C3) exposes the exact
    linear map to the reflected p/s amplitudes, which is what eq 9-10 encode.
    """

    @staticmethod
    def _reflected_ps_coeffs(nw, n2, thi):
        C1, C2, C3 = sp.symbols("C1 C2 C3")
        thTw = math.asin(math.sin(thi) / nw)
        thT2 = math.asin(math.sin(thi) / n2)
        kb = 2 * nw * sp.Matrix([math.sin(thTw), 0, math.cos(thTw)])
        bound = SymbolicWave(k=kb, electric=sp.Matrix([C1, C2, C3]), omega=2)
        refl = [
            make_isotropic_wave_symbolic(n=1, theta_rad=thi, polarization="s", omega=2, direction_sign_z=-1),
            make_isotropic_wave_symbolic(n=1, theta_rad=thi, polarization="p", omega=2, direction_sign_z=-1),
        ]
        trans = [
            make_isotropic_wave_symbolic(n=n2, theta_rad=thT2, polarization="s", omega=2),
            make_isotropic_wave_symbolic(n=n2, theta_rad=thT2, polarization="p", omega=2),
        ]
        b = solve_single_interface_shg_boundary_symbolic(
            reflected_basis=refl, transmitted_basis=trans, inhomogeneous=[bound], mu=1, simplify=True
        )
        Rs = sp.expand(b.coefficients[0])
        Rp = sp.expand(b.coefficients[1])
        return (
            {s: float(sp.re(Rs.coeff(s))) for s in (C1, C2, C3)},
            {s: float(sp.re(Rp.coeff(s))) for s in (C1, C2, C3)},
            thTw, thT2, b,
        )

    def test_s_channel_matches_published_eq10(self):
        """E_s^2w = C_L2 (n2 cosT2 - nw cosTw) / (cosi + n2 cosT2). Depends on C2 only."""
        C1, C2, C3 = sp.symbols("C1 C2 C3")
        for nw, n2, thi in [(3.4, 3.1, 0.5), (2.7, 3.6, 0.9), (3.9, 2.7, 0.3)]:
            scoef, _, thTw, thT2, _ = self._reflected_ps_coeffs(nw, n2, thi)
            self.assertAlmostEqual(scoef[C1], 0.0, places=10)        # no C1
            self.assertAlmostEqual(scoef[C3], 0.0, places=10)        # no C3
            pub = (n2 * math.cos(thT2) - nw * math.cos(thTw)) / (math.cos(thi) + n2 * math.cos(thT2))
            self.assertAlmostEqual(scoef[C2], pub, places=8)         # eq 10

    def test_p_channel_matches_published_eq9(self):
        """E_p^2w numerator = C_L1(n2 - nw cosT2 cosTw) +/- C_L3(nw cosT2 sinTw),
        denominator = cosT2 + n2 cosi (the published `E_L1^{T,e,2w}+n2 cosi(...)`
        with the `(...)` factor = cos^2+sin^2 = 1). Depends on C1 and C3 only."""
        C1, C2, C3 = sp.symbols("C1 C2 C3")
        for nw, n2, thi in [(3.4, 3.1, 0.5), (2.85, 2.96, 0.41), (3.9, 3.73, 0.45)]:
            _, pcoef, thTw, thT2, _ = self._reflected_ps_coeffs(nw, n2, thi)
            self.assertAlmostEqual(pcoef[C2], 0.0, places=10)        # no C2
            denom = -(n2 * math.cos(thi) + math.cos(thT2))           # overall reflected-field sign
            c1_pub = (n2 - nw * math.cos(thT2) * math.cos(thTw)) / denom
            c3_pub = (nw * math.cos(thT2) * math.sin(thTw)) / denom
            self.assertAlmostEqual(pcoef[C1], c1_pub, places=8)      # eq 9, C_L1 term
            self.assertAlmostEqual(pcoef[C3], c3_pub, places=8)      # eq 9, C_L3 term

    def test_boundary_solution_is_physically_continuous(self):
        """The Python 2omega boundary solution makes tangential E/H continuous
        (residual ~ 0) -- i.e. it IS the analytical boundary solution eq 9-10."""
        _, _, _, _, b = self._reflected_ps_coeffs(3.4, 3.1, 0.5)
        residual = sp.Matrix(b.boundary_residual)
        for component in residual:
            # residual is linear in C1,C2,C3; check each coefficient vanishes
            poly = sp.expand(component)
            for sym in sp.symbols("C1 C2 C3"):
                self.assertAlmostEqual(abs(complex(poly.coeff(sym))), 0.0, places=9)


class NormalIncidenceTests(unittest.TestCase):
    """Both NORMAL and oblique incidence are required. The oblique sweeps live in the
    classes above; this pins the LITERAL normal-incidence limit (theta_i = 0), where
    the published forms reduce with the physically-expected degeneracies:
    s/p transmission coincide, the extraordinary L3 field vanishes, and the C_L3
    coupling to E_p^2w drops out (sin theta_T,w = 0)."""

    def test_omega_boundary_normal_incidence_real_and_complex(self):
        for n in (2.5, 3.2, complex(3.4, 0.4)):  # incl. absorbing
            lp = solve_isotropic_linear_interface_symbolic(
                incident_index=1, transmitted_index=n, incident_theta_rad=0,
                incident_polarization="p", simplify=True,
            )
            ls = solve_isotropic_linear_interface_symbolic(
                incident_index=1, transmitted_index=n, incident_theta_rad=0,
                incident_polarization="s", simplify=True,
            )
            pub = 2.0 / (1.0 + n)  # Fresnel t at normal incidence (s == p)
            self.assertAlmostEqual(complex(sp.N(lp.t_p)), complex(pub), places=12)
            self.assertAlmostEqual(complex(sp.N(ls.t_s)), complex(pub), places=12)
            Ep = [complex(sp.N(x)) for x in lp.transmitted_p.electric]
            self.assertAlmostEqual(Ep[2], 0.0, places=12)  # E^e_L3 -> 0 at normal

    def test_two_omega_boundary_normal_incidence_matches_published(self):
        C1, C2, C3 = sp.symbols("C1 C2 C3")
        for nw, n2 in ((3.4, 3.1), (2.7, 3.6)):
            kb = 2 * nw * sp.Matrix([0, 0, 1])  # normal: bound k along +z
            bound = SymbolicWave(k=kb, electric=sp.Matrix([C1, C2, C3]), omega=2)
            refl = [
                make_isotropic_wave_symbolic(n=1, theta_rad=0, polarization="s", omega=2, direction_sign_z=-1),
                make_isotropic_wave_symbolic(n=1, theta_rad=0, polarization="p", omega=2, direction_sign_z=-1),
            ]
            trans = [
                make_isotropic_wave_symbolic(n=n2, theta_rad=0, polarization="s", omega=2),
                make_isotropic_wave_symbolic(n=n2, theta_rad=0, polarization="p", omega=2),
            ]
            b = solve_single_interface_shg_boundary_symbolic(
                reflected_basis=refl, transmitted_basis=trans, inhomogeneous=[bound], mu=1, simplify=True
            )
            Rs, Rp = sp.expand(b.coefficients[0]), sp.expand(b.coefficients[1])
            # de-garbled published forms at theta=0 (cosTw=cosT2=1, sinTw=0):
            self.assertAlmostEqual(float(sp.re(Rs.coeff(C2))), (n2 - nw) / (1 + n2), places=9)  # eq 10
            self.assertAlmostEqual(float(sp.re(Rp.coeff(C1))), -(n2 - nw) / (n2 + 1), places=9)  # eq 9
            self.assertAlmostEqual(abs(complex(Rp.coeff(C3))), 0.0, places=12)  # C3 term vanishes at normal
            self.assertAlmostEqual(abs(complex(Rs.coeff(C1))), 0.0, places=12)  # s still C2-only


class ComplexCoefficientDomainTests(unittest.TestCase):
    """SHAARP exists for ABSORBING/anisotropic media: refractive indices are complex
    (n-tilde = n + i kappa) and d-coefficients/fields are complex. Every stage must
    agree with the published forms in the COMPLEX domain, and the transmitted wave
    must select the physically-correct DECAYING branch (Im k_z > 0)."""

    @staticmethod
    def _cn(rng):  # absorbing refractive index, Im > 0
        return complex(rng.uniform(1.5, 4.0), rng.uniform(0.05, 1.5))

    @staticmethod
    def _cz(rng):  # generic complex field/coefficient
        return complex(rng.uniform(-2, 2), rng.uniform(-2, 2))

    def test_omega_boundary_complex_index_and_decaying_branch(self):
        import random

        rng = random.Random(101)
        thi, n = sp.symbols("theta_i n")
        thT = sp.asin(sp.sin(thi) / n)
        linp = solve_isotropic_linear_interface_symbolic(
            incident_index=1, transmitted_index=n, incident_theta_rad=thi,
            incident_polarization="p", simplify=True,
        )
        tp_pub = 2 * sp.cos(thi) / (n * sp.cos(thi) + sp.cos(thT))
        kz = linp.transmitted_p.k[2]
        worst, bad = 0.0, 0
        for _ in range(60):
            subs = {thi: rng.uniform(0.05, 1.2), n: self._cn(rng)}
            worst = max(worst, abs(complex((linp.t_p - tp_pub).subs(subs).evalf())))
            if complex(kz.subs(subs).evalf()).imag <= 0:
                bad += 1
        self.assertLess(worst, TOL, f"complex t_p deviates from Fresnel: {worst:.2e}")
        self.assertEqual(bad, 0, "transmitted wave must decay into the absorber (Im k_z > 0)")

    def test_c_li_complex_index_matches_eq11_13(self):
        import random

        rng = random.Random(202)
        nw, n2, thT = sp.symbols("n_w n_2w theta_T")
        P1, P2, P3 = sp.symbols("P1 P2 P3")
        k_ee = 2 * nw * sp.Matrix([sp.sin(thT), 0, sp.cos(thT)])
        src = SymbolicNonlinearSource("ee", k_ee, sp.Matrix([P1, P2, P3]), 2)
        C = solve_inhomogeneous_field_symbolic(n2**2 * sp.eye(3), src, mu=1, eps0=1, simplify=True).electric
        den = n2**2 * (n2**2 - nw**2)
        pub = [
            (-n2**2 * P1 + nw**2 * P3 * sp.cos(thT) * sp.sin(thT) + nw**2 * P1 * sp.sin(thT) ** 2) / den,
            -P2 / (n2**2 - nw**2),
            (-n2**2 * P3 + nw**2 * P3 * sp.cos(thT) ** 2 + nw**2 * P1 * sp.cos(thT) * sp.sin(thT)) / den,
        ]
        for i in range(3):
            worst = 0.0
            for _ in range(40):
                subs = {
                    nw: self._cn(rng), n2: self._cn(rng),
                    thT: complex(rng.uniform(0.02, 1.0), rng.uniform(-0.3, 0.3)),
                    P1: self._cz(rng), P2: self._cz(rng), P3: self._cz(rng),
                }
                worst = max(worst, abs(complex((C[i] - pub[i]).subs(subs).evalf())))
            self.assertLess(worst, 1e-7, f"complex C_L{i+1} deviates from eq {11+i}: {worst:.2e}")

    def test_two_omega_boundary_complex_index_matches_eq9_10(self):
        import random

        rng = random.Random(303)
        C1, C2, C3 = sp.symbols("C1 C2 C3")
        worstP = worstS = 0.0
        for _ in range(8):
            nwv, n2v, thi = self._cn(rng), self._cn(rng), rng.uniform(0.1, 1.0)
            thTw, thT2 = sp.asin(sp.sin(thi) / nwv), sp.asin(sp.sin(thi) / n2v)
            kb = 2 * nwv * sp.Matrix([sp.sin(thTw), 0, sp.cos(thTw)])
            bound = SymbolicWave(k=kb, electric=sp.Matrix([C1, C2, C3]), omega=2)
            refl = [
                make_isotropic_wave_symbolic(n=1, theta_rad=thi, polarization="s", omega=2, direction_sign_z=-1),
                make_isotropic_wave_symbolic(n=1, theta_rad=thi, polarization="p", omega=2, direction_sign_z=-1),
            ]
            trans = [
                make_isotropic_wave_symbolic(n=n2v, theta_rad=thT2, polarization="s", omega=2),
                make_isotropic_wave_symbolic(n=n2v, theta_rad=thT2, polarization="p", omega=2),
            ]
            b = solve_single_interface_shg_boundary_symbolic(
                reflected_basis=refl, transmitted_basis=trans, inhomogeneous=[bound], mu=1, simplify=True
            )
            Rs, Rp = sp.expand(b.coefficients[0]), sp.expand(b.coefficients[1])
            cosTw, cosT2, sinTw = sp.cos(thTw), sp.cos(thT2), sp.sin(thTw)
            Rs_pub = C2 * (n2v * cosT2 - nwv * cosTw) / (sp.cos(thi) + n2v * cosT2)
            Rp_pub = -(C1 * (n2v - nwv * cosT2 * cosTw) + C3 * (nwv * cosT2 * sinTw)) / (n2v * sp.cos(thi) + cosT2)
            for _ in range(8):
                subs = {C1: self._cz(rng), C2: self._cz(rng), C3: self._cz(rng)}
                worstS = max(worstS, abs(complex((Rs - Rs_pub).subs(subs).evalf())))
                worstP = max(worstP, abs(complex((Rp - Rp_pub).subs(subs).evalf())))
        self.assertLess(worstS, 1e-9, f"complex E_s^2w deviates from eq 10: {worstS:.2e}")
        self.assertLess(worstP, 1e-9, f"complex E_p^2w deviates from eq 9: {worstP:.2e}")


if __name__ == "__main__":
    unittest.main()
