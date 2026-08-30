"""Symbolic film-thickness (h) SHG coefficient: the reflected SHG amplitude of a single
nonlinear film as a closed form in h (the Maker fringe), validated to reduce to the
numeric ``solve_multilayer_shg_from_fundamental`` to machine precision over a thickness
sweep -- so it is the closed-form-in-h form of the validated numeric Maker-fringe SHG.
"""

import math
import unittest

import numpy as np
import sympy as sp

from shaarp.multilayer_basis import build_multilayer_omega_basis, build_multilayer_2omega_basis
from shaarp.multilayer_shg_boundary import solve_multilayer_shg_from_fundamental
from shaarp.multilayer_shg_symbolic import solve_single_film_shg_symbolic_thickness


def _d():
    d = np.zeros((3, 6))
    d[0, 3] = d[1, 4] = 0.7
    d[2, 2] = 0.4
    d[2, 0] = 0.2
    return d


def _bases(nf, ns, nf2, ns2, thi, pol):
    epsW = [np.diag([nf**2] * 3).astype(complex)]
    epsW_sub = np.diag([ns**2] * 3).astype(complex)
    eps2 = [np.diag([nf2**2] * 3).astype(complex)]
    eps2_sub = np.diag([ns2**2] * 3).astype(complex)
    bw = build_multilayer_omega_basis(
        incident_index=1.0, incident_theta_rad=thi, incident_polarization=pol,
        layer_epsilon_lab=epsW, substrate_epsilon_lab=epsW_sub, omega=1.0,
    )
    b2 = build_multilayer_2omega_basis(
        top_index_2omega=1.0, tangential_index_omega=1.0, incident_theta_rad=thi,
        layer_epsilon_2omega_lab=eps2, substrate_epsilon_2omega_lab=eps2_sub, omega_2=2.0,
    )
    return bw, b2, eps2


class SymbolicThicknessSHGTests(unittest.TestCase):
    def _worst_over_sweep(self, nf, ns, nf2, ns2, thetas=(0.3, 0.6, 0.9), thicknesses=(0.3, 0.7, 1.2, 2.0, 3.5)):
        d = _d()
        worst, self._max_out = 0.0, 0.0
        for thi in thetas:
            for pol in ("s", "p"):
                bw, b2, eps2 = _bases(nf, ns, nf2, ns2, thi, pol)
                h = sp.Symbol("h", positive=True)
                sym = solve_single_film_shg_symbolic_thickness(
                    omega_basis=bw, twoomega_basis=b2, d_voigt_lab=d, eps_2omega_lab=eps2[0],
                    thickness_symbol=h, mu=1.0, eps0=1.0,
                )
                for hv in thicknesses:
                    num = solve_multilayer_shg_from_fundamental(
                        incident_omega=bw.incident, reflected_basis_omega=bw.reflected_basis,
                        layer_basis_omega=bw.layer_basis, substrate_basis_omega=bw.substrate_basis,
                        layer_d_voigt_lab=[d], layer_epsilon_2omega_lab=eps2,
                        reflected_basis_2omega=b2.reflected_basis, layer_homogeneous_basis_2omega=b2.layer_basis,
                        substrate_basis_2omega=b2.substrate_basis, thicknesses=[hv], mu=1.0, eps0=1.0,
                    )
                    got = sym.at(hv)
                    worst = max(
                        worst,
                        abs(got[0] - complex(num.shg.coefficients[0])),
                        abs(got[1] - complex(num.shg.coefficients[1])),
                    )
                    self._max_out = max(
                        self._max_out, abs(complex(num.shg.coefficients[0])), abs(complex(num.shg.coefficients[1])),
                    )
        return worst

    def test_real_film_matches_numeric_over_thickness_sweep(self):
        worst = self._worst_over_sweep(2.1, 1.5, 2.25, 1.55)
        self.assertGreater(self._max_out, 1e-4, "near-zero output -> vacuous 0==0 check")
        self.assertLess(worst, 1e-12)

    def test_complex_absorbing_film_matches_numeric(self):
        worst = self._worst_over_sweep(
            complex(2.1, 0.05), complex(1.5, 0.02), complex(2.25, 0.08), complex(1.55, 0.03),
            thetas=(0.4,), thicknesses=(0.5, 1.5, 2.5),
        )
        self.assertGreater(self._max_out, 1e-4, "near-zero output -> vacuous 0==0 check")
        self.assertLess(worst, 1e-11)

    def test_shg_is_genuinely_thickness_dependent_maker_fringe(self):
        """The closed form must actually OSCILLATE with h (a real Maker fringe), not be
        a constant -- otherwise the symbolic-h dependence would be vacuous."""
        bw, b2, eps2 = _bases(2.1, 1.5, 2.25, 1.55, 0.5, "p")
        h = sp.Symbol("h", positive=True)
        sym = solve_single_film_shg_symbolic_thickness(
            omega_basis=bw, twoomega_basis=b2, d_voigt_lab=_d(), eps_2omega_lab=eps2[0],
            thickness_symbol=h, mu=1.0, eps0=1.0,
        )
        vals = [abs(sym.at(hv)[1]) for hv in np.linspace(0.1, 6.0, 25)]  # |r_p^2w| vs h
        self.assertGreater(max(vals) - min(vals), 0.1 * max(vals))  # meaningful modulation
        self.assertTrue(any(h == s for s in sym.reflected_p.free_symbols))  # genuinely a function of h


class SymbolicThicknessAndDSHGTests(unittest.TestCase):
    """The SHG as a closed form in BOTH thickness h AND the d_ijk coefficient tensor
    (solve_single_film_shg_symbolic_thickness_and_d). Validated vs the numeric ground
    truth with distinct nonzero d components -> NONZERO output (guarded explicitly, since
    a single d component at p-incidence can give a physically-zero output that would make
    the comparison a vacuous 0==0)."""

    POSITIONS = [(0, 3), (1, 4), (2, 2), (2, 0)]  # distinct d components -> nonzero SHG
    VALS = [0.7, 0.5, 0.4, 0.2]

    def _setup(self, thi, pol, nf=2.1, ns=1.5, nf2=2.25, ns2=1.55):
        epsW = [np.diag([nf**2] * 3).astype(complex)]
        eps2 = [np.diag([nf2**2] * 3).astype(complex)]
        bw = build_multilayer_omega_basis(
            incident_index=1.0, incident_theta_rad=thi, incident_polarization=pol,
            layer_epsilon_lab=epsW, substrate_epsilon_lab=np.diag([ns**2] * 3).astype(complex), omega=1.0,
        )
        b2 = build_multilayer_2omega_basis(
            top_index_2omega=1.0, tangential_index_omega=1.0, incident_theta_rad=thi,
            layer_epsilon_2omega_lab=eps2, substrate_epsilon_2omega_lab=np.diag([ns2**2] * 3).astype(complex), omega_2=2.0,
        )
        return bw, b2, eps2

    def _symbolic_d_matrix(self, symbols):
        d = sp.zeros(3, 6)
        for (m, ell), s in zip(self.POSITIONS, symbols):
            d[m, ell] = s
        return d

    def _numeric_d(self):
        d = np.zeros((3, 6))
        for (m, ell), v in zip(self.POSITIONS, self.VALS):
            d[m, ell] = v
        return d

    def test_matches_numeric_with_nonzero_output(self):
        from shaarp.multilayer_shg_symbolic import solve_single_film_shg_symbolic_thickness_and_d

        syms = sp.symbols("da db dc de")
        d_sym = self._symbolic_d_matrix(syms)
        worst, max_output = 0.0, 0.0
        for thi in (0.5, 0.8):
            bw, b2, eps2 = self._setup(thi, "p")
            h = sp.Symbol("h", positive=True)
            sol = solve_single_film_shg_symbolic_thickness_and_d(
                omega_basis=bw, twoomega_basis=b2, d_voigt_symbolic=d_sym,
                eps_2omega_lab=eps2[0], thickness_symbol=h, mu=1.0, eps0=1.0,
            )
            for hv in (0.7, 1.5, 2.5):
                num = solve_multilayer_shg_from_fundamental(
                    incident_omega=bw.incident, reflected_basis_omega=bw.reflected_basis,
                    layer_basis_omega=bw.layer_basis, substrate_basis_omega=bw.substrate_basis,
                    layer_d_voigt_lab=[self._numeric_d()], layer_epsilon_2omega_lab=eps2,
                    reflected_basis_2omega=b2.reflected_basis, layer_homogeneous_basis_2omega=b2.layer_basis,
                    substrate_basis_2omega=b2.substrate_basis, thicknesses=[hv], mu=1.0, eps0=1.0,
                )
                sub = {h: hv, **{syms[i]: self.VALS[i] for i in range(4)}}
                rs = complex(sol.coefficients[0].subs(sub).evalf())
                rp = complex(sol.coefficients[1].subs(sub).evalf())
                rsn, rpn = complex(num.shg.coefficients[0]), complex(num.shg.coefficients[1])
                worst = max(worst, abs(rs - rsn), abs(rp - rpn))
                max_output = max(max_output, abs(rpn), abs(rsn))
        self.assertGreater(max_output, 1e-4, "test setup gave ~zero SHG -> would be a vacuous 0==0 check")
        self.assertLess(worst, 1e-10, f"symbolic-(h,d) SHG deviates from numeric: {worst:.2e}")

    def test_is_genuinely_symbolic_and_linear_in_d(self):
        from shaarp.multilayer_shg_symbolic import solve_single_film_shg_symbolic_thickness_and_d

        syms = sp.symbols("da db dc de")
        d_sym = self._symbolic_d_matrix(syms)
        bw, b2, eps2 = self._setup(0.5, "p")
        h = sp.Symbol("h", positive=True)
        sol = solve_single_film_shg_symbolic_thickness_and_d(
            omega_basis=bw, twoomega_basis=b2, d_voigt_symbolic=d_sym,
            eps_2omega_lab=eps2[0], thickness_symbol=h, mu=1.0, eps0=1.0,
        )
        rp = sol.reflected_p
        # genuinely symbolic in h and at least one d component
        self.assertIn(h, rp.free_symbols)
        self.assertTrue(set(syms) & rp.free_symbols, "output not symbolic in any d component")
        # LINEAR in each d component: d(rp)/d(sym) is independent of that sym
        for s in syms:
            if s in rp.free_symbols:
                self.assertEqual(sp.simplify(sp.diff(rp, s, 2)), 0, f"SHG not linear in {s}")


if __name__ == "__main__":
    unittest.main()
