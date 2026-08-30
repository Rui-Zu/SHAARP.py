"""Form-validate the EIGENVALUE / eigenmode stage for the GENERAL BIAXIAL case against
the closed-form Fresnel equation of wave normals -- completing the eigenmode-stage
closed-form validation across ALL crystal optical classes (isotropic, uniaxial here
in test_uniaxial_eigenmode_closed_form.py, and biaxial here).

KEY RESULT: the two transverse-mode refractive indices of a biaxial crystal (principal
dielectric constants ex, ey, ez along the principal axes) for a wave normal
s = (sx, sy, sz) are NOT only available numerically -- they are the roots of the
Fresnel equation of wave normals,

    sx^2/(u - 1/ex) + sy^2/(u - 1/ey) + sz^2/(u - 1/ez) = 0, u = 1/n^2,

which clears to a QUADRATIC in u (the u^3 terms cancel because sx^2+sy^2+sz^2 = 1):

    u^2 + B u + C = 0,
    B = -(sx^2(ay+az) + sy^2(ax+az) + sz^2(ax+ay)), ai = 1/ei,
    C = sx^2 ay az + sy^2 ax az + sz^2 ax ay.

So n = 1/sqrt(u_pm) is a closed form via the quadratic formula. The numeric SHAARP
`modes_for_direction` (the solveSnell `L E = eta eps E` port) must reproduce these two
roots exactly, for REAL and for COMPLEX (absorbing) biaxial media, at general wave
normals (all of sx, sy, sz nonzero). This closes the eigenvalue-stage form-validation
for arbitrary anisotropic media; the residual `not_full` scope is now only the full
symbolic assembly of the downstream boundary/signal in the general-anisotropic facade,
not the eigenmode indices themselves.
"""

import cmath
import math
import unittest

import numpy as np

from shaarp.anisotropic import modes_for_direction


def _fresnel_quadratic_indices(ex, ey, ez, s):
    """Closed-form biaxial transverse indices from the Fresnel equation of wave normals."""
    ax, ay, az = 1.0 / ex, 1.0 / ey, 1.0 / ez
    sx, sy, sz = s
    B = -(sx**2 * (ay + az) + sy**2 * (ax + az) + sz**2 * (ax + ay))
    C = sx**2 * ay * az + sy**2 * ax * az + sz**2 * ax * ay
    disc = cmath.sqrt(B * B - 4.0 * C)
    u1, u2 = (-B + disc) / 2.0, (-B - disc) / 2.0
    return sorted([1.0 / cmath.sqrt(u1), 1.0 / cmath.sqrt(u2)], key=lambda z: z.real)


def _unit(theta, phi):
    return np.array(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
    )


DIRECTIONS_DEG = [(20, 30), (45, 60), (70, 25), (35, 80), (10, 10), (85, 85), (55, 5)]


class BiaxialEigenmodeClosedFormTests(unittest.TestCase):
    def test_real_biaxial_indices_match_fresnel_quadratic(self):
        ex, ey, ez = 2.0**2, 2.3**2, 2.6**2  # nx<ny<nz biaxial
        eps = np.diag([ex, ey, ez]).astype(complex)
        worst = 0.0
        for a, b in DIRECTIONS_DEG:
            s = _unit(math.radians(a), math.radians(b))
            s = s / np.linalg.norm(s)
            m1, m2 = modes_for_direction(eps, s)
            num = sorted([complex(m1.refractive_index), complex(m2.refractive_index)], key=lambda z: z.real)
            cf = _fresnel_quadratic_indices(ex, ey, ez, s)
            worst = max(worst, abs(num[0] - cf[0]), abs(num[1] - cf[1]))
        self.assertLess(worst, 1e-12, f"real biaxial indices deviate from Fresnel quadratic: {worst:.2e}")

    def test_complex_absorbing_biaxial_indices_match_fresnel_quadratic(self):
        ex = complex(2.0, 0.05) ** 2
        ey = complex(2.3, 0.10) ** 2
        ez = complex(2.6, 0.15) ** 2
        eps = np.diag([ex, ey, ez]).astype(complex)
        worst = 0.0
        for a, b in DIRECTIONS_DEG:
            s = _unit(math.radians(a), math.radians(b))
            s = s / np.linalg.norm(s)
            m1, m2 = modes_for_direction(eps, s)
            got = {complex(m1.refractive_index), complex(m2.refractive_index)}
            for target in _fresnel_quadratic_indices(ex, ey, ez, s):
                worst = max(worst, min(abs(n - target) for n in got))
        self.assertLess(worst, 1e-10, f"complex biaxial indices deviate from Fresnel quadratic: {worst:.2e}")

    def test_uniaxial_limit_of_biaxial_formula(self):
        """When ex == ey the biaxial Fresnel quadratic must collapse to the uniaxial
        ordinary/extraordinary pair -- a consistency check on the closed form itself."""
        no, ne = 2.2, 2.5
        ex = ey = no**2
        ez = ne**2
        eps = np.diag([ex, ey, ez]).astype(complex)
        for a in (15, 40, 75):
            th = math.radians(a)
            s = _unit(th, 0.0)  # in x-z plane
            m1, m2 = modes_for_direction(eps, s)
            num = sorted([complex(m1.refractive_index), complex(m2.refractive_index)], key=lambda z: z.real)
            ne_th = 1.0 / math.sqrt(math.cos(th) ** 2 / no**2 + math.sin(th) ** 2 / ne**2)
            closed = sorted([complex(no), complex(ne_th)], key=lambda z: z.real)
            self.assertAlmostEqual(num[0], closed[0], places=12)
            self.assertAlmostEqual(num[1], closed[1], places=12)

    def test_biaxial_field_directions_match_closed_form(self):
        """Field-DIRECTION (eigenvector) closed form, to complete the eigenmode stage.

        Intention of the formula: the displacement D is transverse to the wave normal
        (D. s = 0) and E = eps^{-1} D. For diagonal eps the Fresnel eigenvector is
        D_i ∝ s_i/(u - 1/e_i), u = 1/n^2, so E_i = D_i / e_i. The numeric eigenvector
        from modes_for_direction must align with this (up to global phase/sign)."""
        ex, ey, ez = 2.0**2, 2.3**2, 2.6**2
        eps = np.diag([ex, ey, ez]).astype(complex)
        a = [1.0 / ex, 1.0 / ey, 1.0 / ez]
        worst = 0.0
        for ad, bd in DIRECTIONS_DEG:
            s = _unit(math.radians(ad), math.radians(bd))
            s = s / np.linalg.norm(s)
            for m in modes_for_direction(eps, s):
                n = complex(m.refractive_index)
                u = 1.0 / n**2
                D = np.array([s[i] / (u - a[i]) for i in range(3)], dtype=complex)
                E_closed = np.array([D[i] * a[i] for i in range(3)], dtype=complex)  # E_i = D_i/e_i
                E_closed = E_closed / np.linalg.norm(E_closed)
                e_num = np.asarray(m.electric_direction, dtype=complex)
                e_num = e_num / np.linalg.norm(e_num)
                # direction equality up to global phase/sign: |<e_num, E_closed>| -> 1
                worst = max(worst, abs(1.0 - abs(np.vdot(e_num, E_closed))))
        self.assertLess(worst, 1e-10, f"biaxial field directions deviate from closed form: {worst:.2e}")

    def test_indices_are_directionally_dispersive_not_constant(self):
        """Sanity: a genuine biaxial crystal's indices actually vary with direction
        (so the closed-form match is non-trivial, not both-equal degeneracy)."""
        ex, ey, ez = 2.0**2, 2.3**2, 2.6**2
        eps = np.diag([ex, ey, ez]).astype(complex)
        highs = []
        for a, b in DIRECTIONS_DEG:
            s = _unit(math.radians(a), math.radians(b))
            s = s / np.linalg.norm(s)
            m1, m2 = modes_for_direction(eps, s)
            highs.append(max(complex(m1.refractive_index).real, complex(m2.refractive_index).real))
        self.assertGreater(max(highs) - min(highs), 0.05)


class SymbolicEigenmodeBuildingBlockTests(unittest.TestCase):
    """The SYMBOLIC closed-form eigenmode (`fresnel_quadratic_eigenmodes_symbolic`) is
    the building block the anisotropic full-analytical SI assembly needs. It must (a)
    build a genuine symbolic closed form and (b) reproduce the numeric solver when
    numeric values are substituted -- for real, complex (absorbing), and uniaxial media.
    Intention of the formula is documented on the function (Fresnel quadratic in u=1/n^2,
    D_i~s_i/(u-1/e_i), E=eps^{-1}D)."""

    def test_builds_symbolic_closed_form(self):
        import sympy as sp
        from shaarp.symbolic import fresnel_quadratic_eigenmodes_symbolic

        ex, ey, ez, a, b = sp.symbols("ex ey ez a b", positive=True)
        modes = fresnel_quadratic_eigenmodes_symbolic(
            [ex, ey, ez], [sp.sin(a) * sp.cos(b), sp.sin(a) * sp.sin(b), sp.cos(a)], simplify=False
        )
        self.assertEqual(len(modes), 2)
        # the index expression genuinely depends on the dielectric constants + direction
        free = modes[0][0].free_symbols
        for sym in (ex, ey, ez):
            self.assertIn(sym, free)

    def _check(self, eps_diag, s):
        import sympy as sp
        from shaarp.symbolic import fresnel_quadratic_eigenmodes_symbolic

        ex, ey, ez, a, b = sp.symbols("ex ey ez a b")
        modes_sym = fresnel_quadratic_eigenmodes_symbolic(
            [ex, ey, ez], [sp.sin(a) * sp.cos(b), sp.sin(a) * sp.sin(b), sp.cos(a)], simplify=False
        )
        sn = np.array(s, dtype=complex)
        sn = sn / np.linalg.norm(sn)
        subs = {ex: eps_diag[0], ey: eps_diag[1], ez: eps_diag[2],
                a: cmath.acos(sn[2]), b: cmath.atan(sn[1] / sn[0])}
        sym = sorted([complex(sp.N(m[0].subs(subs))) for m in modes_sym], key=lambda z: z.real)
        eps = np.diag(eps_diag).astype(complex)
        num = sorted([complex(m.refractive_index) for m in modes_for_direction(eps, np.real_if_close(sn))],
                     key=lambda z: z.real)
        return max(abs(sym[i] - num[i]) for i in range(2))

    def test_real_biaxial_symbolic_matches_numeric(self):
        d = self._check([2.0**2, 2.3**2, 2.6**2],
                        [math.sin(0.6) * math.cos(0.7), math.sin(0.6) * math.sin(0.7), math.cos(0.6)])
        self.assertLess(d, 1e-10)

    def test_complex_biaxial_symbolic_matches_numeric(self):
        d = self._check([complex(2.0, 0.05) ** 2, complex(2.3, 0.1) ** 2, complex(2.6, 0.15) ** 2],
                        [0.4, 0.5, 0.768])
        self.assertLess(d, 1e-10)

    def _field_align_worst(self, eps_diag, s):
        """Worst (1 - |<E_symbolic, E_numeric>|) over the two modes, matched by index.
        This is the check that catches the uniaxial ordinary-mode degeneracy bug
        (the D-formula is singular there; the function must special-case it)."""
        import sympy as sp
        from shaarp.symbolic import fresnel_quadratic_eigenmodes_symbolic

        modes = fresnel_quadratic_eigenmodes_symbolic([sp.Float(e) for e in eps_diag], list(s), simplify=True)
        sn = np.array(s, dtype=complex)
        sn = sn / np.linalg.norm(sn)
        num = list(modes_for_direction(np.diag(eps_diag).astype(complex), np.real_if_close(sn)))
        worst = 0.0
        for n, e in modes:
            nv = complex(sp.N(n))
            ev = np.array([complex(sp.N(c)) for c in e], dtype=complex)
            self.assertGreater(np.linalg.norm(ev), 1e-9, "degenerate mode returned a zero field vector")
            ev = ev / np.linalg.norm(ev)
            m = min(num, key=lambda mm: abs(complex(mm.refractive_index) - nv))
            en = np.asarray(m.electric_direction, dtype=complex)
            en = en / np.linalg.norm(en)
            worst = max(worst, abs(1.0 - abs(np.vdot(ev, en))))
        return worst

    def test_uniaxial_symbolic_field_directions_match_numeric(self):
        """Regression for the degeneracy bug: the UNIAXIAL ordinary mode (e_x=e_y) made
        the D-formula singular. Field directions must now match numeric for the optic
        axis along z, y, AND x (all three degenerate-pair placements)."""
        no2, ne2 = 2.2**2, 2.5**2
        s = [math.sin(0.8) * math.cos(0.3), math.sin(0.8) * math.sin(0.3), math.cos(0.8)]
        self.assertLess(self._field_align_worst([no2, no2, ne2], s), 1e-9)  # optic z
        self.assertLess(self._field_align_worst([no2, ne2, no2], [0.4, 0.5, 0.768]), 1e-9)  # optic y
        self.assertLess(self._field_align_worst([ne2, no2, no2], [0.5, 0.4, 0.768]), 1e-9)  # optic x

    def test_biaxial_symbolic_field_directions_match_numeric(self):
        self.assertLess(self._field_align_worst([2.0**2, 2.3**2, 2.6**2], [0.5, 0.4, 0.768]), 1e-9)

    def test_isotropic_symbolic_fields_are_perpendicular_to_k(self):
        """Isotropic: both modes share n and the field is undetermined in the plane
        perpendicular to s. So the test is that D = eps E is transverse to k (D.s=0),
        not a unique direction, and that no zero vector is returned."""
        import sympy as sp
        from shaarp.symbolic import fresnel_quadratic_eigenmodes_symbolic

        e0 = 2.3**2
        s = np.array([0.5, 0.4, 0.768]); s = s / np.linalg.norm(s)
        modes = fresnel_quadratic_eigenmodes_symbolic([sp.Float(e0)] * 3, list(s), simplify=True)
        for n, e in modes:
            ev = np.array([complex(sp.N(c)) for c in e], dtype=complex)
            self.assertGreater(np.linalg.norm(ev), 1e-9)
            D = e0 * ev  # isotropic: D = e0 E
            self.assertLess(abs(np.vdot(D, s)) / np.linalg.norm(D), 1e-9)  # D perp k

    def test_rejects_non_principal_axis_eps(self):
        import sympy as sp
        from shaarp.symbolic import fresnel_quadratic_eigenmodes_symbolic

        bad = sp.Matrix([[4, 0, 0.3], [0, 5, 0], [0.3, 0, 6]])  # non-diagonal -> not principal-axis
        with self.assertRaises(ValueError):
            fresnel_quadratic_eigenmodes_symbolic(bad, [0.3, 0.4, 0.866])


if __name__ == "__main__":
    unittest.main()
