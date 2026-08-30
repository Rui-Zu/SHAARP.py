"""LAYERED full-analytical SHAARP.si: the derivation as a CHAIN of named symbolic intermediates.

Rui: *"you are not actually showing full analytical expression but just partial
analytical. If you look at my full analytical Mathematica, everything are symbolic; I want you to
do exactly what I did in Mathematica SHAARP.si full analytical calculation."* His printed output is
not one expanded blob -- it is a sequence of DEFINED quantities, each with its own equation:

    k^(T,o,w), theta^(T,o,w), E_(Li)^(T,o,w) ... then the 2w bases, then C_(Li)^(T,ee,2w),
    and finally E_(s,p)^(2w) written IN TERMS OF those names.

That structure is not cosmetic. Substituting each eigenmode's explicit radical inline is what made
a BIAXIAL eps intractable in the earlier formulation (measured on the shipped SI palette: KTP (100)
and LBO both exceeded a 420 s budget); with the intermediates kept as NAMED SYMBOLS the same cases
assemble in ~0.4 s. So this file fences BOTH properties at once -- his form, and the scope it buys.

Validated without any vacuous (0 == 0) shortcut:

  1. STRUCTURE -- the chain emits his symbol names, in derivation order, and the final amplitudes
     are written in those names (they reference the bound-field C_(Li) symbols, not expanded
     radicals). Nonzero-guarded.
  2. NUMERICS -- evaluating the chain the way his notebook reads it (substitute the physical values
     into each named intermediate IN ORDER, then into the final amplitude) reproduces the validated
     numeric ``solve_single_interface_shg`` to ~1e-12, for a UNIAXIAL and a BIAXIAL crystal, with a
     nonzero-magnitude guard on the reference so the agreement cannot be trivial.
  3. SCOPE -- a principal-aligned BIAXIAL eps (KTP) assembles fully symbolically inside a generous
     wall-clock budget. This is the regression fence for the >1000x formulation difference: if a
     future change reintroduces inline radical substitution, this test times out rather than
     silently falling back to substituted numbers.
  4. NO SILENT FALLBACK -- the GUI's Full Analytical path returns a genuinely symbolic result
     (symbolic theta_i, zero float literals) for both a uniaxial and a biaxial case study.
"""

from __future__ import annotations

import time
import unittest

import numpy as np
import sympy as sp

from shaarp.casestudy_materials import build_casestudy_material
from shaarp.shaarp_gui import _epsilon_lab_of, _snap_degenerate_triple, compute_si_gui_result
from shaarp.shg import solve_single_interface_shg
from shaarp.symbolic import (
    d_voigt_symbolic,
    rotate_d_voigt_symbolic,
    solve_si_shg_full_analytical_symbolic,
    solve_si_shg_layered_symbolic,
)
from shaarp.tensors import rotate_d_voigt_crystal_to_lab

THETA = 0.4  # rad -- a generic off-normal angle (no accidental symmetry zero)


def _principal_triples(material):
    """The material's LAB-frame principal eps triples, plus the max off-diagonal magnitude.

    A principal-aligned closed form only applies when the lab eps is diagonal; the off-diagonal
    magnitude is returned so a test can assert it is using a case where that holds."""
    ew = _epsilon_lab_of(material, omega=True)
    e2 = _epsilon_lab_of(material, omega=False)
    off = max(abs(complex(m[i, j])) for m in (ew, e2) for i in range(3) for j in range(3) if i != j)
    tw = _snap_degenerate_triple(complex(ew[i, i]) for i in range(3))
    t2 = _snap_degenerate_triple(complex(e2[i, i]) for i in range(3))
    return tw, t2, off


def _symbolic_triple(triple, tag):
    """Symbols for a principal triple + the substitution map back to its physical values."""
    a, b, c = triple
    eq = lambda u, v: abs(u - v) <= 1e-9 * max(abs(u), 1.0)  # noqa: E731
    n_o = sp.Symbol("n_o" + tag, positive=True)
    n_e = sp.Symbol("n_e" + tag, positive=True)
    if eq(a, b) and eq(b, c):
        return (n_o**2,) * 3, {n_o: sp.sqrt(a)}, "isotropic"
    if eq(a, b):
        return (n_o**2, n_o**2, n_e**2), {n_o: sp.sqrt(a), n_e: sp.sqrt(c)}, "axis||z"
    if eq(b, c):
        return (n_e**2, n_o**2, n_o**2), {n_e: sp.sqrt(a), n_o: sp.sqrt(b)}, "axis||x"
    if eq(a, c):
        return (n_o**2, n_e**2, n_o**2), {n_o: sp.sqrt(a), n_e: sp.sqrt(b)}, "axis||y"
    nx, ny, nz = (sp.Symbol(f"n_{s}{tag}", positive=True) for s in "xyz")
    return (nx**2, ny**2, nz**2), {nx: sp.sqrt(a), ny: sp.sqrt(b), nz: sp.sqrt(c)}, "biaxial"


def _layered_for(case_key):
    """Run the layered chain for a shipped case study, with everything symbolic."""
    material = build_casestudy_material(case_key)
    tw, t2, off = _principal_triples(material)
    pw, vals_w, kind = _symbolic_triple(tw, "w")
    p2, vals_2, _ = _symbolic_triple(t2, "2")
    rot = np.asarray(material.orientation.rotation_matrix(), dtype=float)
    rot_exact = sp.Matrix(rot.T.tolist()).applyfunc(
        lambda v: sp.nsimplify(v, rational=False, tolerance=1e-12))
    d_sym = d_voigt_symbolic(material.structure.point_group)
    d_lab_sym = rotate_d_voigt_symbolic(d_sym, rot_exact)
    theta = sp.Symbol("theta_i", real=True, positive=True)
    result = solve_si_shg_layered_symbolic(
        eps_x_omega=pw[0], eps_y_omega=pw[1], eps_z_omega=pw[2],
        eps_x_2omega=p2[0], eps_y_2omega=p2[1], eps_z_2omega=p2[2],
        d_voigt_lab=d_lab_sym, incident_theta_rad=theta,
        incident_polarization="p", simplify=False)
    d_cry = np.asarray(material.d_voigt_pm_v, dtype=complex)
    d_map = {}
    for s in d_sym.free_symbols:
        name = str(s)  # d<i><j>, 1-based Voigt
        d_map[s] = complex(d_cry[int(name[1]) - 1, int(name[2]) - 1])
    subs = {theta: sp.Float(THETA)}
    subs.update(vals_w)
    subs.update(vals_2)
    subs.update({s: sp.nsimplify(v.real) + sp.I * sp.nsimplify(v.imag) for s, v in d_map.items()})
    reference = solve_single_interface_shg(
        np.diag([complex(x) for x in tw]), np.diag([complex(x) for x in t2]),
        np.asarray(rotate_d_voigt_crystal_to_lab(d_cry, rot), dtype=complex),
        incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=THETA,
        incident_polarization="p", omega=1.0, mu=1.0, eps0=1.0)
    return result, subs, reference, kind, off


def _evaluate_chain(result, subs):
    """Evaluate the chain as his notebook reads it: each named intermediate IN ORDER, then the
    final amplitudes in terms of the values just computed."""
    named: dict = {}
    for name, expr in result.definitions:
        named[sp.Symbol(name)] = complex(sp.N(expr.subs(subs).subs(named)))
    full = dict(subs)
    full.update(named)
    return complex(sp.N(result.reflected_s.subs(full))), complex(sp.N(result.reflected_p.subs(full)))


def _polarimetry_chain(case_key):
    """The DELIVERABLE form: the layered chain under a symbolic input polarization phi.

    Structure is asserted here rather than on a fixed s/p input because a pure p-input drives the
    ordinary mode to exactly zero, so its (correctly) omitted E^(T,o,w) components would make the
    name list look incomplete. Under symbolic phi both eigenmodes are populated -- which is also
    the case the panel actually shows."""
    material = build_casestudy_material(case_key)
    tw, t2, _off = _principal_triples(material)
    pw, _vw, _kind = _symbolic_triple(tw, "w")
    p2, _v2, _ = _symbolic_triple(t2, "2")
    rot = np.asarray(material.orientation.rotation_matrix(), dtype=float)
    rot_exact = sp.Matrix(rot.T.tolist()).applyfunc(
        lambda v: sp.nsimplify(v, rational=False, tolerance=1e-12))
    d_lab_sym = rotate_d_voigt_symbolic(
        d_voigt_symbolic(material.structure.point_group), rot_exact)
    capture: dict = {}
    solve_si_shg_full_analytical_symbolic(
        eps_x_omega=pw[0], eps_y_omega=pw[1], eps_z_omega=pw[2],
        eps_x_2omega=p2[0], eps_y_2omega=p2[1], eps_z_2omega=p2[2],
        d_voigt_lab=d_lab_sym,
        incident_theta_rad=sp.Symbol("theta_i", real=True, positive=True),
        layered=True, capture_stages=capture, simplify=False)
    return capture


class LayeredFullAnalyticalStructureTest(unittest.TestCase):
    """(1) His printed structure: named intermediates, in derivation order."""

    def test_chain_uses_his_symbol_names_in_derivation_order(self):
        capture = _polarimetry_chain("LiNbO3 (11-20) MTI X-cut")
        names = [n for n, _ in capture["layered_definitions"]]
        self.assertTrue(names, "the layered chain emitted no definitions")
        # his quantities: transmitted/reflected, mode, frequency -- k, theta, E at omega; C at 2omega.
        # (Stored as sympify-safe identifiers; the panel typesets kT_o_w_z as k^(T,o,ω)_z.)
        for expected in ("kT_o_w_z", "thetaT_o_w", "ET_o_w_L2",
                         "kT_e_w_z", "ET_e_w_L1",
                         "ET_e_2w_L1", "ER_p_2w_L1", "CT_ee_2w_L1"):
            self.assertIn(expected, names, f"missing his intermediate {expected}")
        # derivation ORDER: the omega eigenmodes are defined before the bound 2omega fields
        self.assertLess(names.index("ET_o_w_L2"), names.index("CT_ee_2w_L1"))
        # every definition is a real expression, not a placeholder
        for name, expr in capture["layered_definitions"]:
            self.assertNotEqual(sp.simplify(expr), 0, f"{name} is identically zero")

    def test_final_amplitudes_are_written_in_the_named_intermediates(self):
        result, _subs, _ref, _kind, _off = _layered_for("KBBF")
        free = {str(s) for s in result.reflected_p.free_symbols}
        self.assertTrue(any(s.startswith("CT_") for s in free),
                        f"final amplitude does not reference the bound-field C_(Li) names: {sorted(free)}")
        self.assertTrue(any(s.startswith("kT_") for s in free),
                        f"final amplitude does not reference the eigenmode wavevector names: {sorted(free)}")
        # SYMPIFY ROUND-TRIP: the copyable expression is re-parsed by the Mathematica export, so
        # every name in it must survive sympify. Decorated labels like "k^(T,o,w)_z" raise
        # SympifyError -- that regression would break Copy-as-Mathematica for every layered case.
        self.assertEqual(sp.sympify(str(result.reflected_p)).free_symbols,
                         result.reflected_p.free_symbols)
        # ... and it is genuinely symbolic: no substituted numeric literals in the amplitude
        import re
        self.assertEqual(re.findall(r"\d+\.\d{3,}", str(result.reflected_p)), [],
                         "float literals survived in the final amplitude (not fully symbolic)")


class LayeredFullAnalyticalNumericTest(unittest.TestCase):
    """(2) Un-fakeable: the chain must EVALUATE to the validated numeric solver."""

    def _assert_matches_numeric(self, case_key, expected_kind, tol):
        result, subs, reference, kind, off = _layered_for(case_key)
        self.assertEqual(kind, expected_kind)
        self.assertLess(off, 1e-9, "case is not principal-aligned; a diagonal treatment is invalid")
        got_s, got_p = _evaluate_chain(result, subs)
        ref_s = complex(reference.coefficients[0])
        ref_p = complex(reference.coefficients[1])
        scale = max(abs(ref_s), abs(ref_p))
        self.assertGreater(scale, 1e-6, "reference SHG is ~0; the comparison would be vacuous")
        self.assertLess(abs(got_s - ref_s) / scale, tol)
        self.assertLess(abs(got_p - ref_p) / scale, tol)

    def test_uniaxial_chain_matches_validated_numeric(self):
        # measured: 4.3e-14 (and 1e-15 / 3e-13 for the isotropic and axis||z cases)
        self._assert_matches_numeric("LiNbO3 (11-20) MTI X-cut", "axis||x", 1e-12)

    def test_biaxial_chain_matches_validated_numeric(self):
        # BIAXIAL FLOOR, measured and attributed -- 1.1e-11 at theta = 0.4 rad,
        # rising to ~6e-11 by 0.7 rad, where every other optical class sits at 1e-15..3e-13.
        # It is NOT arithmetic: the chain gives the same value at 15, 30 and 50 digits, the inputs
        # are bit-identical on both sides (LBO's orientation is the identity, its eps are exact
        # 2-decimals, and the symbolic-vs-numeric rotated d agree to 0.0), the numeric solver's own
        # boundary residual is 4e-16, and the two sides' OMEGA eigenmode k_z agree to 1e-16.
        # What remains is the 2omega BIAXIAL MODE BASIS: this chain uses the analytic closed-form
        # roots, while the numeric reference builds a numerically-derived 'fast_slow_unclassified'
        # basis for a biaxial eps. OPEN ITEM: pin that difference exactly (same class as the
        # near-degenerate biaxial basis work) rather than leaving it merely explained.
        self._assert_matches_numeric("LiB3O5 (LBO)", "biaxial", 1e-10)


class LayeredFullAnalyticalScopeTest(unittest.TestCase):
    """(3) The scope the layered form buys: BIAXIAL is tractable, symbolically."""

    def test_biaxial_assembles_symbolically_within_budget(self):
        # REGRESSION FENCE. The expanded formulation exceeded 420 s on this exact case; the layered
        # one takes ~0.4 s. The budget is deliberately generous (100x the measured cost) so this
        # fails only on a genuine return to inline radical substitution, not on machine noise.
        started = time.time()
        result, subs, reference, kind, _off = _layered_for("KTP (100)")
        elapsed = time.time() - started
        self.assertEqual(kind, "biaxial")
        self.assertLess(elapsed, 40.0, f"biaxial layered assembly took {elapsed:.1f} s")
        got_s, _got_p = _evaluate_chain(result, subs)
        ref_s = complex(reference.coefficients[0])
        self.assertGreater(abs(ref_s), 1e-6, "reference s-channel is ~0; comparison would be vacuous")
        self.assertLess(abs(got_s - ref_s) / abs(ref_s), 1e-10)  # biaxial floor, see the note above


class LayeredFullAnalyticalGuiTest(unittest.TestCase):
    """(4) Through the GUI's own entry point: no silent fallback to substituted numbers."""

    def _assert_gui_full_symbolic(self, case_key):
        import re
        material = build_casestudy_material(case_key)
        result = compute_si_gui_result("Full Analytical",
                                       point_group=material.structure.point_group,
                                       theta_deg=45.0, material=material)
        symbols = result.stages.get("symbols", {}) or {}
        self.assertIn("symbolic", str(symbols.get("incidence", "")),
                      f"{case_key}: theta was substituted -- Full Analytical fell back")
        expression = str(result.stages.get("reflected_p_2omega", ""))
        self.assertTrue(expression, "no reflected p amplitude returned")
        self.assertEqual(re.findall(r"\d+\.\d{3,}", expression), [],
                         f"{case_key}: float literals in the Full Analytical amplitude")
        self.assertTrue(str(result.stages.get("deriv_0_definitions", "")).strip(),
                        f"{case_key}: the layered definition chain was not emitted")

    def test_uniaxial_case_study_is_fully_symbolic(self):
        self._assert_gui_full_symbolic("LiNbO3 (11-20) MTI X-cut")

    def test_biaxial_case_study_is_fully_symbolic(self):
        self._assert_gui_full_symbolic("KTP (100)")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
