"""SI full-analytical single-interface SHG as a closed form in the SHG coefficient
tensor d_ijk (Voigt d_il). The SI assemblies are a fully symbolic chain, so they accept
a SYMBOLIC d_voigt_lab directly (no decomposition) -- this pins that, validating the
substituted value against the numeric solve_single_interface_shg and confirming the
output is genuinely symbolic in d and LINEAR in each d component.

CRITICAL conventions (lessons baked in):
  * symbolic-vs-numeric comparisons MUST pass identical mu, eps0 (here mu=1, eps0=1).
    Mixing eps0=1 with the numeric default EPS0 [MU0*EPS0=1/c^2] rescales the inhomogeneous
    field per polarization row and looks like a (spurious) bug.
  * a single d component at a given incidence can give a PHYSICALLY-ZERO SHG (selection
    rule), so every case asserts a NONZERO numeric output -> no vacuous 0==0 passes.
"""

import math
import unittest

import numpy as np
import sympy as sp

from shaarp.shg import solve_single_interface_shg
from shaarp.symbolic import (
    solve_biaxial_single_interface_shg_symbolic,
    solve_uniaxial_single_interface_shg_symbolic,
)

# d positions that give a NONZERO reflected SHG for the geometries below.
_POSITIONS = [(0, 0), (0, 3), (1, 4), (2, 0), (2, 2)]
_VALS = [0.2, 0.8, 0.6, 0.3, 0.5]


def _sym_d(symbols):
    d = sp.zeros(3, 6)
    for (m, ell), s in zip(_POSITIONS, symbols):
        d[m, ell] = s
    return d


def _num_d():
    d = np.zeros((3, 6))
    for (m, ell), v in zip(_POSITIONS, _VALS):
        d[m, ell] = v
    return d


def _numeric_shg(epsW_diag, eps2_diag, thi, pol):
    num = solve_single_interface_shg(
        np.diag(epsW_diag).astype(complex), np.diag(eps2_diag).astype(complex), _num_d(),
        incident_index_omega=1.0, incident_index_2omega=1.0,
        incident_theta_rad=thi, incident_polarization=pol, omega=1.0, mu=1.0, eps0=1.0,
    )
    nc = np.asarray(num.coefficients)
    return complex(nc[0]), complex(nc[1])


class SISymbolicDTests(unittest.TestCase):
    def test_biaxial_symbolic_d_matches_numeric_nonzero(self):
        syms = sp.symbols("d_a d_b d_c d_d d_e")
        d_sym = _sym_d(syms)
        epsW, eps2 = [2.0**2, 2.3**2, 2.6**2], [2.2**2, 2.5**2, 2.8**2]
        worst, max_out = 0.0, 0.0
        for thi in (0.4, 0.7):
            for pol in ("s", "p"):
                res = solve_biaxial_single_interface_shg_symbolic(
                    eps_x_omega=epsW[0], eps_y_omega=epsW[1], eps_z_omega=epsW[2],
                    eps_x_2omega=eps2[0], eps_y_2omega=eps2[1], eps_z_2omega=eps2[2],
                    d_voigt_lab=d_sym, incident_theta_rad=thi, incident_polarization=pol,
                    omega=1, mu=1, eps0=1, simplify=False,
                )
                sub = {syms[i]: _VALS[i] for i in range(len(syms))}
                rs = complex(sp.N(res.boundary.coefficients[0].subs(sub)))
                rp = complex(sp.N(res.boundary.coefficients[1].subs(sub)))
                nrs, nrp = _numeric_shg(epsW, eps2, thi, pol)
                worst = max(worst, abs(rs - nrs), abs(rp - nrp))
                max_out = max(max_out, abs(nrs), abs(nrp))
        self.assertGreater(max_out, 1e-4, "near-zero output -> would be a vacuous 0==0 check")
        self.assertLess(worst, 1e-10, f"biaxial symbolic-d SHG deviates from numeric: {worst:.2e}")

    def test_uniaxial_symbolic_d_matches_numeric_nonzero(self):
        syms = sp.symbols("d_a d_b d_c d_d d_e")
        d_sym = _sym_d(syms)
        no, ne, no2, ne2 = 2.2, 2.5, 2.4, 2.7
        epsW, eps2 = [no**2, no**2, ne**2], [no2**2, no2**2, ne2**2]
        worst, max_out = 0.0, 0.0
        for thi in (0.4, 0.7):
            for pol in ("s", "p"):
                res = solve_uniaxial_single_interface_shg_symbolic(
                    eps_ordinary_omega=no**2, eps_extraordinary_omega=ne**2,
                    eps_ordinary_2omega=no2**2, eps_extraordinary_2omega=ne2**2,
                    d_voigt_lab=d_sym, incident_theta_rad=thi, incident_polarization=pol,
                    omega=1, mu=1, eps0=1, simplify=False,
                )
                sub = {syms[i]: _VALS[i] for i in range(len(syms))}
                rs = complex(sp.N(res.boundary.coefficients[0].subs(sub)))
                rp = complex(sp.N(res.boundary.coefficients[1].subs(sub)))
                nrs, nrp = _numeric_shg(epsW, eps2, thi, pol)
                worst = max(worst, abs(rs - nrs), abs(rp - nrp))
                max_out = max(max_out, abs(nrs), abs(nrp))
        self.assertGreater(max_out, 1e-4, "near-zero output -> would be a vacuous 0==0 check")
        self.assertLess(worst, 1e-10, f"uniaxial symbolic-d SHG deviates from numeric: {worst:.2e}")

    def test_point_group_shared_symbol_d(self):
        """A point-group d with a SHARED symbol across Voigt positions (e.g. d31=d32)
        must flow through and still match numeric when substituted."""
        d31 = sp.Symbol("d31")
        d33 = sp.Symbol("d33")
        d_sym = sp.zeros(3, 6)
        d_sym[2, 0] = d31  # d31
        d_sym[2, 1] = d31  # d32 = d31 (shared)
        d_sym[2, 2] = d33  # d33
        epsW, eps2 = [2.2**2, 2.2**2, 2.5**2], [2.4**2, 2.4**2, 2.7**2]
        res = solve_biaxial_single_interface_shg_symbolic(
            eps_x_omega=epsW[0], eps_y_omega=epsW[1], eps_z_omega=epsW[2],
            eps_x_2omega=eps2[0], eps_y_2omega=eps2[1], eps_z_2omega=eps2[2],
            d_voigt_lab=d_sym, incident_theta_rad=0.6, incident_polarization="p",
            omega=1, mu=1, eps0=1, simplify=False,
        )
        self.assertTrue({d31, d33} & res.boundary.coefficients[1].free_symbols)
        sub = {d31: 0.3, d33: 0.5}
        rp = complex(sp.N(res.boundary.coefficients[1].subs(sub)))
        d_num = np.zeros((3, 6))
        d_num[2, 0] = d_num[2, 1] = 0.3
        d_num[2, 2] = 0.5
        num = solve_single_interface_shg(
            np.diag(epsW).astype(complex), np.diag(eps2).astype(complex), d_num,
            incident_index_omega=1.0, incident_index_2omega=1.0,
            incident_theta_rad=0.6, incident_polarization="p", omega=1.0, mu=1.0, eps0=1.0,
        )
        nrp = complex(np.asarray(num.coefficients)[1])
        self.assertGreater(abs(nrp), 1e-4)
        self.assertLess(abs(rp - nrp), 1e-10)

    def test_complex_absorbing_symbolic_d(self):
        syms = sp.symbols("d_a d_b d_c d_d d_e")
        d_sym = _sym_d(syms)
        epsW = [complex(2.0, 0.03) ** 2, complex(2.3, 0.06) ** 2, complex(2.6, 0.09) ** 2]
        eps2 = [complex(2.2, 0.05) ** 2, complex(2.5, 0.08) ** 2, complex(2.8, 0.12) ** 2]
        res = solve_biaxial_single_interface_shg_symbolic(
            eps_x_omega=epsW[0], eps_y_omega=epsW[1], eps_z_omega=epsW[2],
            eps_x_2omega=eps2[0], eps_y_2omega=eps2[1], eps_z_2omega=eps2[2],
            d_voigt_lab=d_sym, incident_theta_rad=0.5, incident_polarization="p",
            omega=1, mu=1, eps0=1, simplify=False,
        )
        sub = {syms[i]: _VALS[i] for i in range(len(syms))}
        rp = complex(sp.N(res.boundary.coefficients[1].subs(sub)))
        nrs, nrp = _numeric_shg(epsW, eps2, 0.5, "p")
        self.assertGreater(abs(nrp), 1e-4)
        self.assertLess(abs(rp - nrp), 1e-9)

    def test_output_is_symbolic_and_linear_in_d(self):
        syms = sp.symbols("d_a d_b d_c d_d d_e")
        d_sym = _sym_d(syms)
        res = solve_biaxial_single_interface_shg_symbolic(
            eps_x_omega=2.0**2, eps_y_omega=2.3**2, eps_z_omega=2.6**2,
            eps_x_2omega=2.2**2, eps_y_2omega=2.5**2, eps_z_2omega=2.8**2,
            d_voigt_lab=d_sym, incident_theta_rad=0.5, incident_polarization="p",
            omega=1, mu=1, eps0=1, simplify=False,
        )
        rp = res.boundary.coefficients[1]
        self.assertTrue(set(syms) & rp.free_symbols, "SHG not symbolic in any d component")
        for s in syms:
            if s in rp.free_symbols:
                self.assertEqual(sp.simplify(sp.diff(rp, s, 2)), 0, f"SHG not linear in {s}")


if __name__ == "__main__":
    unittest.main()
