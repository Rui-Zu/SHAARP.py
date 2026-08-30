"""N-LAYER symbolic thickness through the full 2-omega SHG stage -- parity with the
original, which lets ANY SUBSET of interior layers carry a symbolic thickness.

The original's contract (read from the source): ``SHAARP.ml.nb:5135-5146``
gives every interior layer its own "analytical h" button storing a distinct symbol h1,
h2, ...; ``setup.nb:11011-11017`` gathers ``thickness = Key[h] /@ mAll[[2;;-2]]`` as a
plain LIST consumed element-wise, so numeric and symbolic entries mix freely. Before
the port raised "This symbolic-thickness SHG solve is for a single nonlinear film." for
anything but one film -- these fences are the parity proof.

EVERY fence substitutes concrete thicknesses into the symbolic result and compares against
the VALIDATED NUMERIC solver ``solve_multilayer_shg_from_fundamental`` for the same stack,
with a vacuity guard (a near-zero output would make the comparison a meaningless 0 == 0).
"""

import unittest

import numpy as np
import sympy as sp

from shaarp.multilayer_basis import build_multilayer_2omega_basis, build_multilayer_omega_basis
from shaarp.multilayer_shg_boundary import solve_multilayer_shg_from_fundamental
from shaarp.multilayer_shg_symbolic import (
    SymbolicThicknessBudget,
    SymbolicThicknessBudgetExceeded,
    solve_multilayer_shg_symbolic_thickness,
    solve_single_film_shg_symbolic_thickness,
)


def _d(scale=1.0):
    d = np.zeros((3, 6))
    d[0, 3] = d[1, 4] = 0.7 * scale
    d[2, 2] = 0.4 * scale
    return d


def _build(n_layers, thi=0.5, pol="p", absorbing=False):
    """A stack of ``n_layers`` distinct interior layers between air and a substrate."""
    ki = 0.05 if absorbing else 0.0
    nf = [complex(2.1 + 0.13 * k, ki) for k in range(n_layers)]
    nf2 = [complex(2.25 + 0.11 * k, 1.6 * ki) for k in range(n_layers)]
    ns, ns2 = complex(1.5, 0.4 * ki), complex(1.55, 0.6 * ki)
    eps_w = [np.diag([v ** 2] * 3).astype(complex) for v in nf]
    eps_2 = [np.diag([v ** 2] * 3).astype(complex) for v in nf2]
    bw = build_multilayer_omega_basis(
        incident_index=1.0, incident_theta_rad=thi, incident_polarization=pol,
        layer_epsilon_lab=eps_w, substrate_epsilon_lab=np.diag([ns ** 2] * 3).astype(complex),
        omega=1.0)
    b2 = build_multilayer_2omega_basis(
        top_index_2omega=1.0, tangential_index_omega=1.0, incident_theta_rad=thi,
        layer_epsilon_2omega_lab=eps_2,
        substrate_epsilon_2omega_lab=np.diag([ns2 ** 2] * 3).astype(complex), omega_2=2.0)
    return bw, b2, eps_2


def _numeric(bw, b2, eps_2, d_layers, thicknesses):
    return solve_multilayer_shg_from_fundamental(
        incident_omega=bw.incident, reflected_basis_omega=bw.reflected_basis,
        layer_basis_omega=bw.layer_basis, substrate_basis_omega=bw.substrate_basis,
        layer_d_voigt_lab=d_layers, layer_epsilon_2omega_lab=eps_2,
        reflected_basis_2omega=b2.reflected_basis,
        layer_homogeneous_basis_2omega=b2.layer_basis,
        substrate_basis_2omega=b2.substrate_basis, thicknesses=thicknesses, mu=1.0, eps0=1.0)


class NLayerSymbolicThicknessParity(unittest.TestCase):
    """Substituting numeric thicknesses into the N-layer closed form must reproduce the
    numeric multilayer SHG."""

    def _compare(self, *, n_layers, symbolic_flags, thicknesses, d_layers=None,
                 absorbing=False, thi=0.5, pol="p"):
        bw, b2, eps_2 = _build(n_layers, thi=thi, pol=pol, absorbing=absorbing)
        d_layers = d_layers if d_layers is not None else [_d() for _ in range(n_layers)]
        syms = [sp.Symbol("h%d" % (k + 1), positive=True) for k in range(n_layers)]
        entries = [syms[k] if symbolic_flags[k] else sp.Float(thicknesses[k])
                   for k in range(n_layers)]
        sol = solve_multilayer_shg_symbolic_thickness(
            omega_basis=bw, twoomega_basis=b2, layer_d_voigt_lab=d_layers,
            layer_epsilon_2omega_lab=eps_2, thickness_symbols=entries, mu=1.0, eps0=1.0)
        # every FLAGGED layer's symbol must genuinely survive into the closed form
        free = set().union(*[e.free_symbols for e in sol.coefficients])
        for k, flagged in enumerate(symbolic_flags):
            if flagged:
                self.assertIn(syms[k], free,
                              f"h{k + 1} vanished from the closed form -> layer {k + 1} is not "
                              f"actually symbolic (vacuous fence)")
        got = sol.at(*[thicknesses[k] for k in range(n_layers) if symbolic_flags[k]])
        ref = [complex(c) for c in _numeric(bw, b2, eps_2, d_layers, list(thicknesses)).shg.coefficients]
        self.assertEqual(len(got), len(ref))
        mag = max(abs(v) for v in ref)
        self.assertGreater(mag, 1e-4, "near-zero output -> vacuous 0 == 0 comparison")
        return max(abs(a - b) for a, b in zip(got, ref)), mag

    def test_all_numeric_entries_match_the_numeric_solver(self):
        """- when EVERY thickness is numeric the boundary solve takes the
        numpy path (sympy's LU over a+b*I Floats expanded every product into four terms: GaAs
        (111) Partial Analytical took 105 s numeric vs 14.5 s symbolic for the SAME closed
        form, and the gate's module budget was blown). The numeric path must reproduce the
        validated numeric solver to machine precision, for lossless and absorbing stacks."""
        for absorbing in (False, True):
            for n in (1, 2, 3):
                worst, mag = self._compare(n_layers=n, symbolic_flags=[False] * n,
                                           thicknesses=[0.4, 0.7, 0.55][:n],
                                           absorbing=absorbing)
                self.assertLess(worst / mag, 1e-12,
                                f"all-numeric {n}-layer absorbing={absorbing}: rel {worst / mag:.2e}")

    def test_two_layers_both_symbolic(self):
        """- the headline parity case. RED before ValueError('This symbolic-thickness
        SHG solve is for a single nonlinear film.')."""
        worst, _ = self._compare(n_layers=2, symbolic_flags=[True, True],
                                 thicknesses=[0.4, 0.7])
        self.assertLess(worst, 1e-11)

    def test_two_layers_both_symbolic_s_incidence(self):
        """s-incidence needs a d with a yy (Voigt column 1) component: with E along y only,
        a d whose nonzero columns are yz/xz/zz drives NO polarization at all, so the SHG is
        a selection-rule zero -- physics, not a defect. (The vacuity guard in _compare
        caught exactly that when this fence first used the p-incidence tensor.)"""
        d_s = np.zeros((3, 6))
        d_s[2, 1] = 0.6   # P_z from Ey^2
        d_s[1, 1] = 0.3   # P_y from Ey^2
        worst, _ = self._compare(n_layers=2, symbolic_flags=[True, True],
                                 thicknesses=[0.55, 1.1], thi=0.3, pol="s",
                                 d_layers=[d_s, d_s])
        self.assertLess(worst, 1e-11)

    def test_three_layers_mixed_symbolic_and_numeric(self):
        """- the original's actual capability: layer 1 and 3 symbolic, layer 2 a plain
        number (``setup.nb:11011`` mixes them freely)."""
        worst, _ = self._compare(n_layers=3, symbolic_flags=[True, False, True],
                                 thicknesses=[0.4, 0.35, 0.9])
        self.assertLess(worst, 1e-11)

    def test_three_layers_middle_layer_shg_inactive(self):
        """- a passive interior layer contributes NO source but still carries phase.
        The symbolic path skips its (zero) source; the numeric reference runs it with d=0."""
        d_layers = [_d(), np.zeros((3, 6)), _d(0.8)]
        worst, _ = self._compare(n_layers=3, symbolic_flags=[True, True, True],
                                 thicknesses=[0.4, 0.6, 0.9], d_layers=d_layers)
        self.assertLess(worst, 1e-11)

    def test_a_passive_layer_still_changes_the_answer(self):
        """The companion guard for 'contributes no source' must NOT mean 'is invisible'
        -- a passive spacer still propagates both omega and 2 omega, so removing it entirely
        must change the result. Without this, F4 could pass with the layer dropped."""
        bw3, b23, eps23 = _build(3)
        d3 = [_d(), np.zeros((3, 6)), _d(0.8)]
        with_spacer = [complex(c) for c in
                       _numeric(bw3, b23, eps23, d3, [0.4, 0.6, 0.9]).shg.coefficients]
        bw2, b22, eps22 = _build(2)
        without = [complex(c) for c in
                   _numeric(bw2, b22, eps22, [_d(), _d(0.8)], [0.4, 0.9]).shg.coefficients]
        scale = max(abs(v) for v in with_spacer)
        delta = max(abs(a - b) for a, b in zip(with_spacer[:2], without[:2]))
        self.assertGreater(delta / scale, 1e-3,
                           "dropping the passive spacer changed nothing -> the F4 comparison "
                           "would not detect a dropped layer")

    def test_two_absorbing_layers(self):
        """- complex eps at omega and 2 omega (the Au-like class)."""
        worst, _ = self._compare(n_layers=2, symbolic_flags=[True, True],
                                 thicknesses=[0.3, 0.5], absorbing=True, thi=0.4)
        self.assertLess(worst, 1e-10)


class NLayerApiContracts(unittest.TestCase):
    def test_at_requires_one_value_per_symbolic_layer(self):
        """- numeric entries were already substituted at assembly, so ``at`` takes values
        only for the flagged layers, and says which ones when the count is wrong."""
        bw, b2, eps_2 = _build(2)
        h2 = sp.Symbol("h2", positive=True)
        sol = solve_multilayer_shg_symbolic_thickness(
            omega_basis=bw, twoomega_basis=b2, layer_d_voigt_lab=[_d(), _d()],
            layer_epsilon_2omega_lab=eps_2, thickness_symbols=[sp.Float(0.4), h2],
            mu=1.0, eps0=1.0)
        self.assertEqual([str(s) for s in sol.symbolic_symbols], ["h2"])
        self.assertEqual(len(sol.at(0.7)), len(sol.coefficients))
        with self.assertRaises(ValueError) as ctx:
            sol.at(0.7, 0.9)
        self.assertIn("h2", str(ctx.exception))

    def test_substrate_accessors_are_indexed_from_the_end(self):
        """Defect fix: ``coefficients[6]/[7]`` was the substrate only for ONE film; at 2+
        layers it silently returned layer 2's F1/F2 amplitudes."""
        bw, b2, eps_2 = _build(2)
        h1, h2 = sp.symbols("h1 h2", positive=True)
        sol = solve_multilayer_shg_symbolic_thickness(
            omega_basis=bw, twoomega_basis=b2, layer_d_voigt_lab=[_d(), _d()],
            layer_epsilon_2omega_lab=eps_2, thickness_symbols=[h1, h2], mu=1.0, eps0=1.0)
        self.assertEqual(len(sol.coefficients), 2 + 4 * 2 + 2)
        self.assertIs(sol.substrate_s, sol.coefficients[-2])
        self.assertIs(sol.substrate_p, sol.coefficients[-1])
        self.assertEqual(len(sol.layer_amplitudes(1)), 4)
        # and the 2-layer substrate is NOT what the old fixed index would have returned
        self.assertIsNot(sol.substrate_s, sol.coefficients[6])

    def test_single_film_wrapper_matches_the_nlayer_entry_point(self):
        """- the back-compat wrapper is exactly the N=1 case."""
        bw, b2, eps_2 = _build(1)
        h = sp.Symbol("h", positive=True)
        old = solve_single_film_shg_symbolic_thickness(
            omega_basis=bw, twoomega_basis=b2, d_voigt_lab=_d(), eps_2omega_lab=eps_2[0],
            thickness_symbol=h, mu=1.0, eps0=1.0)
        new = solve_multilayer_shg_symbolic_thickness(
            omega_basis=bw, twoomega_basis=b2, layer_d_voigt_lab=[_d()],
            layer_epsilon_2omega_lab=eps_2, thickness_symbols=[h], mu=1.0, eps0=1.0)
        self.assertEqual(old.thickness_symbol, h)
        for a, b in zip(old.at(0.8), new.at(0.8)):
            self.assertAlmostEqual(abs(a - b), 0.0, places=14)

    def test_numeric_entry_matches_a_substituted_symbol(self):
        """- a numeric thickness entry (collapsed to a constant at assembly) must equal
        the same layer kept symbolic and substituted afterwards."""
        bw, b2, eps_2 = _build(2)
        h1, h2 = sp.symbols("h1 h2", positive=True)
        both = solve_multilayer_shg_symbolic_thickness(
            omega_basis=bw, twoomega_basis=b2, layer_d_voigt_lab=[_d(), _d()],
            layer_epsilon_2omega_lab=eps_2, thickness_symbols=[h1, h2], mu=1.0, eps0=1.0)
        mixed = solve_multilayer_shg_symbolic_thickness(
            omega_basis=bw, twoomega_basis=b2, layer_d_voigt_lab=[_d(), _d()],
            layer_epsilon_2omega_lab=eps_2, thickness_symbols=[h1, sp.Float(0.7)],
            mu=1.0, eps0=1.0)
        for a, b in zip(both.at(0.4, 0.7), mixed.at(0.4)):
            self.assertLess(abs(a - b), 1e-13)

    def test_all_numeric_result_carries_no_symbols_and_equals_the_substituted_symbolic(self):
        """- the numpy path and the (untouched) symbolic path agree: a fully numeric
        2-layer stack vs both layers symbolic with the numbers substituted afterwards."""
        bw, b2, eps_2 = _build(2)
        h1, h2 = sp.symbols("h1 h2", positive=True)
        both = solve_multilayer_shg_symbolic_thickness(
            omega_basis=bw, twoomega_basis=b2, layer_d_voigt_lab=[_d(), _d()],
            layer_epsilon_2omega_lab=eps_2, thickness_symbols=[h1, h2], mu=1.0, eps0=1.0)
        num = solve_multilayer_shg_symbolic_thickness(
            omega_basis=bw, twoomega_basis=b2, layer_d_voigt_lab=[_d(), _d()],
            layer_epsilon_2omega_lab=eps_2, thickness_symbols=[sp.Float(0.4), sp.Float(0.7)],
            mu=1.0, eps0=1.0)
        self.assertFalse(any(c.free_symbols for c in num.coefficients))
        for a, b in zip(both.at(0.4, 0.7), num.at()):
            self.assertLess(abs(a - b), 1e-12)

    def test_budget_refuses_before_starting_and_names_the_way_forward(self):
        """- a refusal is a stated RESULT, not a hang: the pre-flight cap fires before
        any CAS work."""
        bw, b2, eps_2 = _build(2)
        h1, h2 = sp.symbols("h1 h2", positive=True)
        budget = SymbolicThicknessBudget(max_symbolic_layers=1)
        with self.assertRaises(SymbolicThicknessBudgetExceeded) as ctx:
            solve_multilayer_shg_symbolic_thickness(
                omega_basis=bw, twoomega_basis=b2, layer_d_voigt_lab=[_d(), _d()],
                layer_epsilon_2omega_lab=eps_2, thickness_symbols=[h1, h2],
                mu=1.0, eps0=1.0, budget=budget)
        msg = str(ctx.exception)
        self.assertIn("analytical h", msg)
        self.assertIn("numeric", msg, "a refusal must name the way forward")

    def test_every_layer_zero_d_is_refused_by_name(self):
        bw, b2, eps_2 = _build(2)
        h1, h2 = sp.symbols("h1 h2", positive=True)
        with self.assertRaises(ValueError) as ctx:
            solve_multilayer_shg_symbolic_thickness(
                omega_basis=bw, twoomega_basis=b2,
                layer_d_voigt_lab=[np.zeros((3, 6)), np.zeros((3, 6))],
                layer_epsilon_2omega_lab=eps_2, thickness_symbols=[h1, h2], mu=1.0, eps0=1.0)
        self.assertIn("SHG-active", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
