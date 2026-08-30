"""PARTIAL ANALYTICAL SHG POLARIMETRY (SHAARP.ml): the reflected/transmitted SHG of a
nonlinear film as a closed form in the INPUT POLARIZATION angle phi and the SHG coefficient
d_ijk (and the film thickness h), while the eigenmodes / refraction angles / refractive
indices stay NUMERIC -- the d-extraction expression for SHAARP.ml.

That numeric-eigenmode / symbolic-(phi, d, h) split is exactly what makes SHAARP.ml "partial
analytical" (symbolic only in input polarization, the SHG coefficient d, and sometimes
thickness). ``solve_single_film_shg_symbolic_polarimetry`` builds it by combining the
incidence-LINEAR omega layer amplitudes as c = J_s c^s + J_p c^p (Jones), so the SHG is
bilinear in (J_s,J_p) = the polarimetry curve, and linear in d (d-linearity decomposition).

Validated WITHOUT any vacuous (0==0) shortcut, against the INDEPENDENT numeric arbitrary-Jones
multilayer workflow ``solve_multilayer_shg_from_tensors_jones`` (a different assembly than the
symbolic-h factorization), over a (phi, h) sweep, with nonzero-output and phi-varying guards.
"""

import math
import unittest

import numpy as np
import sympy as sp

from shaarp.multilayer_basis import build_multilayer_omega_basis, build_multilayer_2omega_basis
from shaarp.multilayer_shg_boundary import solve_multilayer_shg_from_tensors_jones
from shaarp.multilayer_shg_symbolic import solve_single_film_shg_symbolic_polarimetry

# distinct nonzero d positions -> nonzero SHG (avoid a single-component selection-rule zero)
POSITIONS = [(0, 3), (1, 4), (2, 2), (2, 0)]
VALUES = [0.7, 0.5, 0.4, 0.2]


def _omega_bases(nf, ns, thi):
    epsW = [np.diag([nf**2] * 3).astype(complex)]
    epsW_sub = np.diag([ns**2] * 3).astype(complex)
    common = dict(incident_index=1.0, incident_theta_rad=thi, layer_epsilon_lab=epsW,
                  substrate_epsilon_lab=epsW_sub, omega=1.0)
    bs = build_multilayer_omega_basis(incident_polarization="s", **common)
    bp = build_multilayer_omega_basis(incident_polarization="p", **common)
    return bs, bp, epsW, epsW_sub


def _twoomega_basis(nf2, ns2, thi):
    eps2 = [np.diag([nf2**2] * 3).astype(complex)]
    eps2_sub = np.diag([ns2**2] * 3).astype(complex)
    b2 = build_multilayer_2omega_basis(
        top_index_2omega=1.0, tangential_index_omega=1.0, incident_theta_rad=thi,
        layer_epsilon_2omega_lab=eps2, substrate_epsilon_2omega_lab=eps2_sub, omega_2=2.0,
    )
    return b2, eps2, eps2_sub


def _symbolic_d(symbols):
    d = sp.zeros(3, 6)
    for (m, ell), s in zip(POSITIONS, symbols):
        d[m, ell] = s
    return d


def _numeric_d():
    d = np.zeros((3, 6))
    for (m, ell), v in zip(POSITIONS, VALUES):
        d[m, ell] = v
    return d


class MLPolarimetryVsNumericJonesTests(unittest.TestCase):
    """The symbolic-(phi,d,h) polarimetry reproduces the numeric arbitrary-Jones workflow."""

    def _build_symbolic(self, nf, ns, nf2, ns2, thi, symbols):
        bs, bp, _, _ = _omega_bases(nf, ns, thi)
        b2, eps2, _ = _twoomega_basis(nf2, ns2, thi)
        h = sp.Symbol("h", positive=True)
        phi = sp.Symbol("phi", real=True)
        sol = solve_single_film_shg_symbolic_polarimetry(
            omega_basis_s=bs, omega_basis_p=bp, twoomega_basis=b2,
            d_voigt_symbolic=_symbolic_d(symbols), eps_2omega_lab=eps2[0],
            thickness_symbol=h, phi_symbol=phi, mu=1.0, eps0=1.0,
        )
        return sol, h, phi

    def _numeric(self, nf, ns, nf2, ns2, thi, phi_val, hv):
        epsW = [np.diag([nf**2] * 3).astype(complex)]
        eps2 = [np.diag([nf2**2] * 3).astype(complex)]
        ref = solve_multilayer_shg_from_tensors_jones(
            incident_index_omega=1.0, top_index_2omega=1.0, incident_theta_rad=thi,
            incident_jones_sp=(math.sin(phi_val), math.cos(phi_val)),
            layer_epsilon_omega_lab=epsW, substrate_epsilon_omega_lab=np.diag([ns**2] * 3).astype(complex),
            layer_d_voigt_lab=[_numeric_d()], layer_epsilon_2omega_lab=eps2,
            substrate_epsilon_2omega_lab=np.diag([ns2**2] * 3).astype(complex),
            thicknesses=[hv], omega=1.0, mu=1.0, eps0=1.0,
        )
        return complex(np.asarray(ref.shg.coefficients)[0]), complex(np.asarray(ref.shg.coefficients)[1])

    def test_real_film_phi_h_sweep_matches_numeric(self):
        syms = sp.symbols("da db dc dd")
        sol, h, phi = self._build_symbolic(2.1, 1.5, 2.25, 1.55, 0.5, syms)
        sub_d = {syms[i]: VALUES[i] for i in range(4)}
        worst, max_out, curve_p = 0.0, 0.0, []
        for hv in (0.7, 2.0, 3.3):
            for v in np.linspace(0.1, math.pi - 0.1, 11):
                rsn, rpn = self._numeric(2.1, 1.5, 2.25, 1.55, 0.5, v, hv)
                sub = {h: hv, phi: v, **sub_d}
                rs = complex(sol.coefficients[0].subs(sub).evalf())
                rp = complex(sol.coefficients[1].subs(sub).evalf())
                worst = max(worst, abs(rs - rsn), abs(rp - rpn))
                max_out = max(max_out, abs(rsn), abs(rpn))
                if hv == 0.7:
                    curve_p.append(abs(rp) ** 2)
        self.assertGreater(max_out, 1e-4, "near-zero output -> vacuous 0==0 check")
        self.assertGreater(max(curve_p) - min(curve_p), 0.3 * max(curve_p), "polarimetry barely varies with phi")
        self.assertLess(worst, 1e-10, f"symbolic-(phi,d,h) ML SHG deviates from numeric Jones: {worst:.2e}")

    def test_transmitted_substrate_channel_matches_numeric(self):
        """The transmitted (Maker) channel: the ML substrate SHG amplitudes (substrate_s/p =
        coefficients[6]/[7], used by extract_ml_film_d_voigt(observable='transmitted')) match
        the numeric workflow's substrate coefficients over a (phi, h) sweep."""
        syms = sp.symbols("da db dc dd")
        sol, h, phi = self._build_symbolic(2.1, 1.5, 2.25, 1.55, 0.5, syms)
        sub_d = {syms[i]: VALUES[i] for i in range(4)}
        worst, max_out = 0.0, 0.0
        for hv in (0.7, 2.0):
            for v in np.linspace(0.2, math.pi - 0.2, 7):
                ref = solve_multilayer_shg_from_tensors_jones(
                    incident_index_omega=1.0, top_index_2omega=1.0, incident_theta_rad=0.5,
                    incident_jones_sp=(math.sin(v), math.cos(v)),
                    layer_epsilon_omega_lab=[np.diag([2.1**2] * 3).astype(complex)],
                    substrate_epsilon_omega_lab=np.diag([1.5**2] * 3).astype(complex),
                    layer_d_voigt_lab=[_numeric_d()], layer_epsilon_2omega_lab=[np.diag([2.25**2] * 3).astype(complex)],
                    substrate_epsilon_2omega_lab=np.diag([1.55**2] * 3).astype(complex),
                    thicknesses=[hv], omega=1.0, mu=1.0, eps0=1.0,
                )
                tsn = complex(np.asarray(ref.shg.coefficients)[6])
                tpn = complex(np.asarray(ref.shg.coefficients)[7])
                sub = {h: hv, phi: v, **sub_d}
                ts = complex(sol.substrate_s.subs(sub).evalf())
                tp = complex(sol.substrate_p.subs(sub).evalf())
                worst = max(worst, abs(ts - tsn), abs(tp - tpn))
                max_out = max(max_out, abs(tsn), abs(tpn))
        self.assertGreater(max_out, 1e-4, "near-zero transmitted output -> vacuous")
        self.assertLess(worst, 1e-10, f"ML transmitted (substrate) channel deviates from numeric: {worst:.2e}")

    def test_complex_absorbing_film_matches_numeric(self):
        syms = sp.symbols("da db dc dd")
        nf, ns, nf2, ns2 = complex(2.1, 0.05), complex(1.5, 0.02), complex(2.25, 0.08), complex(1.55, 0.03)
        sol, h, phi = self._build_symbolic(nf, ns, nf2, ns2, 0.4, syms)
        sub_d = {syms[i]: VALUES[i] for i in range(4)}
        worst, max_out = 0.0, 0.0
        for hv in (0.6, 1.8):
            for v in (0.2, 0.9, 1.4, 2.3):
                rsn, rpn = self._numeric(nf, ns, nf2, ns2, 0.4, v, hv)
                sub = {h: hv, phi: v, **sub_d}
                rs = complex(sol.coefficients[0].subs(sub).evalf())
                rp = complex(sol.coefficients[1].subs(sub).evalf())
                worst = max(worst, abs(rs - rsn), abs(rp - rpn))
                max_out = max(max_out, abs(rsn), abs(rpn))
        self.assertGreater(max_out, 1e-4, "near-zero output -> vacuous 0==0 check")
        self.assertLess(worst, 1e-10, f"complex ML polarimetry deviates from numeric Jones: {worst:.2e}")


class MLPolarimetrySymbolicStructureTests(unittest.TestCase):
    """The output is genuinely symbolic in phi, d, h -- and linear in each d component."""

    def test_symbolic_in_phi_d_h_and_linear_in_d(self):
        syms = sp.symbols("da db dc dd")
        bs, bp, _, _ = _omega_bases(2.1, 1.5, 0.5)
        b2, eps2, _ = _twoomega_basis(2.25, 1.55, 0.5)
        h = sp.Symbol("h", positive=True)
        phi = sp.Symbol("phi", real=True)
        sol = solve_single_film_shg_symbolic_polarimetry(
            omega_basis_s=bs, omega_basis_p=bp, twoomega_basis=b2,
            d_voigt_symbolic=_symbolic_d(syms), eps_2omega_lab=eps2[0],
            thickness_symbol=h, phi_symbol=phi, mu=1.0, eps0=1.0,
        )
        rp = sol.reflected_p
        self.assertIn(h, rp.free_symbols)
        self.assertIn(phi, rp.free_symbols)
        self.assertTrue(set(syms) & rp.free_symbols, "output not symbolic in any d component")
        # LINEAR in each present d component: d^2(rp)/d(sym)^2 == 0
        for s in syms:
            if s in rp.free_symbols:
                self.assertEqual(sp.simplify(sp.diff(rp, s, 2)), 0, f"SHG not linear in {s}")

    def test_phi_endpoints_reduce_to_fixed_polarization(self):
        """phi = 0 -> pure p incidence; phi = 90 deg -> pure s incidence. Cross-check the
        polarimetry endpoints against the fixed-polarization symbolic-(d,h) solver."""
        from shaarp.multilayer_shg_symbolic import solve_single_film_shg_symbolic_thickness_and_d

        syms = sp.symbols("da db dc dd")
        d_sym = _symbolic_d(syms)
        bs, bp, _, _ = _omega_bases(2.1, 1.5, 0.5)
        b2, eps2, _ = _twoomega_basis(2.25, 1.55, 0.5)
        h = sp.Symbol("h", positive=True)
        phi = sp.Symbol("phi", real=True)
        pol = solve_single_film_shg_symbolic_polarimetry(
            omega_basis_s=bs, omega_basis_p=bp, twoomega_basis=b2, d_voigt_symbolic=d_sym,
            eps_2omega_lab=eps2[0], thickness_symbol=h, phi_symbol=phi, mu=1.0, eps0=1.0,
        )
        fixed = {
            "p": solve_single_film_shg_symbolic_thickness_and_d(
                omega_basis=bp, twoomega_basis=b2, d_voigt_symbolic=d_sym,
                eps_2omega_lab=eps2[0], thickness_symbol=h, mu=1.0, eps0=1.0),
            "s": solve_single_film_shg_symbolic_thickness_and_d(
                omega_basis=bs, twoomega_basis=b2, d_voigt_symbolic=d_sym,
                eps_2omega_lab=eps2[0], thickness_symbol=h, mu=1.0, eps0=1.0),
        }
        sub_d = {syms[i]: VALUES[i] for i in range(4)}
        hv = 1.3
        max_out = 0.0
        for phi_val, key in ((0.0, "p"), (math.pi / 2, "s")):
            for idx in (0, 1):
                got = complex(pol.coefficients[idx].subs({h: hv, phi: phi_val, **sub_d}).evalf())
                ref = complex(fixed[key].coefficients[idx].subs({h: hv, **sub_d}).evalf())
                max_out = max(max_out, abs(ref))
                self.assertLess(abs(got - ref), 1e-11, f"phi={phi_val} channel {idx} != fixed {key}")
        self.assertGreater(max_out, 1e-4, "near-zero output -> vacuous 0==0 check")


if __name__ == "__main__":
    unittest.main()
