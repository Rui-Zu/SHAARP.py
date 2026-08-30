"""Partial Analytical must respect the crystal's OWN point-group symmetry.

Rui: "if the same derivation, they should exactly match". Rotating a crystal by one of
its own symmetry operations is the sharpest possible version of that test — it needs no second
implementation to compare against, because the answer must be *identical to itself*.

It was not. PA broke the invariant by 1.1e-2 (MoS2, -6m2, 120 deg) and 3.4e-1 (ZnO, 6mm, 120 deg)
while the numeric path held it to 2.4e-15. Root cause: `_run_ml_partial_analytical_polarimetry`
passed SI `mu=MU0, eps0=EPS0` into the symbolic solve, where every other path (and the validated
fence tests/test_ml_shg_polarimetry_symbolic.py, on BOTH sides of its comparison) uses the
SHAARP.ml normalization mu=1, eps0=1. The SHG inhomogeneous operator is
`curl_curl(k) - omega^2 mu eps0 eps`; in SI its two terms differ by ~1e16, so the operator is
numerically singular — measured condition 7.2e14 .. 7.7e17, with all 30 solves already setting
`ill_conditioned=True` and silently falling back to `solve_ill_conditioned`. Under the
normalization the SAME physics conditions at 9.68, exactly what the numeric path reports.

This module fences BOTH the invariant and the conditioning, because the invariant alone would pass
for any material whose symmetry operation happens not to excite the ill-conditioned modes.

NOTE ON PICKING CASES: the rotation must be a symmetry of the LAB d tensor, not merely of the
point group. "Quartz x-cut" is 32 (3-fold) but its 3-fold axis lies IN PLANE, so a rotation about
the surface normal is NOT a symmetry there (the solver's d genuinely changes by 1.7); likewise
-43m's 3-fold axes are along <111>, not z. Both were rejected as cases for exactly this reason.
"""
from __future__ import annotations

import unittest

import numpy as np
import sympy as sp

from shaarp.casestudy_materials import build_casestudy_ml_system
from shaarp.config import with_sample_azimuth_deg
from shaarp.shaarp_gui import compute_ml_gui_result

# (case label, wavelength um, point group, rotations that are symmetries about the SURFACE NORMAL)
CASES = [
    ("MoS2", 1.064, "-6m2", (120.0, 240.0)),
    ("ZnO (001)", 1.064, "6mm", (60.0, 120.0, 180.0)),
]


def _pa_value(system, theta_deg=20.0):
    result = compute_ml_gui_result("Partial Analytical", system=system, theta_deg=theta_deg)
    expr = sp.sympify(result.stages["reflected_p_2omega"])
    subs = {s: (0.4363 if str(s) == "phi" else 1.0) for s in expr.free_symbols}
    return complex(expr.subs(subs).evalf())


class PartialAnalyticalRespectsCrystalSymmetry(unittest.TestCase):
    def test_rotating_by_a_symmetry_operation_leaves_the_answer_unchanged(self):
        for label, wl, pg, rotations in CASES:
            base = build_casestudy_ml_system(label, thickness_um=0.5, wavelength_um=wl)
            v0 = _pa_value(base)
            self.assertGreater(abs(v0), 1e-12, f"{label}: near-zero output -> vacuous check")
            for ang in rotations:
                rotated = with_sample_azimuth_deg(base, ang, rotate_top=False,
                                                  rotate_substrate=False)
                v = _pa_value(rotated)
                rel = abs(v - v0) / abs(v0)
                self.assertLess(
                    rel, 1e-9,
                    f"{label} ({pg}) rotated {ang:g} deg about its own symmetry axis changed the "
                    f"Partial Analytical answer by {rel:.2e} (was 1.1e-2 before F72)")

    def test_the_rotation_really_is_a_symmetry_of_the_solver_d(self):
        """Guards the fence itself: if a case's d were NOT invariant, the test above would be
        vacuous-by-construction in the other direction (it would fail for correct physics)."""
        import copy

        import shaarp.multilayer_shg_symbolic as MS

        for label, wl, pg, rotations in CASES:
            seen = []
            real = MS.solve_multilayer_shg_symbolic_polarimetry

            def spy(*a, **k):
                seen.append(copy.deepcopy(k))
                return real(*a, **k)

            MS.solve_multilayer_shg_symbolic_polarimetry = spy
            try:
                base = build_casestudy_ml_system(label, thickness_um=0.5, wavelength_um=wl)
                compute_ml_gui_result("Partial Analytical", system=base, theta_deg=20.0)
                compute_ml_gui_result(
                    "Partial Analytical", theta_deg=20.0,
                    system=with_sample_azimuth_deg(base, rotations[0], rotate_top=False,
                                                   rotate_substrate=False))
            finally:
                MS.solve_multilayer_shg_symbolic_polarimetry = real
            a = sp.Matrix(seen[0]["layer_d_voigt_symbolic"][0])
            b = sp.Matrix(seen[1]["layer_d_voigt_symbolic"][0])
            syms = sorted(set(list(a.free_symbols) + list(b.free_symbols)), key=str)
            probe = {s: 1.0 + 0.13 * i for i, s in enumerate(syms)}
            an = np.array(a.subs(probe).evalf().tolist(), dtype=complex)
            bn = np.array(b.subs(probe).evalf().tolist(), dtype=complex)
            self.assertLess(np.max(np.abs(an - bn)), 1e-12,
                            f"{label}: {rotations[0]:g} deg is not a symmetry of the lab d — "
                            f"this case is invalid for the symmetry fence")


class PartialAnalyticalOperatorIsWellConditioned(unittest.TestCase):
    """The direct fence on the root cause: the SHG inhomogeneous operator must be built in the
    SHAARP.ml normalization, where it conditions like the numeric path (~10), not like SI (~1e17)."""

    def _max_condition(self, run):
        import shaarp.nonlinear as NL

        conds = []
        real = NL.solve_inhomogeneous_field

        def patched(eps, source, **kw):
            out = real(eps, source, **kw)
            value = getattr(out, "operator_condition", None)
            if value is not None:
                conds.append(float(value))
            return out

        NL.solve_inhomogeneous_field = patched
        try:
            run()
        finally:
            NL.solve_inhomogeneous_field = real
        self.assertTrue(conds, "no inhomogeneous solves observed -> vacuous check")
        return float(np.nanmax(conds)), len(conds)

    def test_partial_analytical_matches_the_numeric_conditioning(self):
        from dataclasses import replace

        from shaarp.api import run_sample_rotation

        system = build_casestudy_ml_system("MoS2", thickness_um=0.5, wavelength_um=1.064)

        pa_cond, n_pa = self._max_condition(
            lambda: compute_ml_gui_result("Partial Analytical", system=system, theta_deg=20.0))

        def numeric():
            pol = replace(system.polarimetry, theta_deg=20.0, phi_deg=25.0, psi_deg=0.0)
            run_sample_rotation(replace(system, polarimetry=pol), [0.0])

        num_cond, _ = self._max_condition(numeric)

        self.assertLess(pa_cond, 1e12,
                        f"Partial Analytical's SHG inhomogeneous operator is ill-conditioned "
                        f"({pa_cond:.2e} over {n_pa} solves) — the SI-units regression (F72)")
        self.assertLess(pa_cond, 1e3 * max(num_cond, 1.0),
                        f"PA conditioning ({pa_cond:.2e}) is far worse than the numeric path's "
                        f"({num_cond:.2e}) on identical physics")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
