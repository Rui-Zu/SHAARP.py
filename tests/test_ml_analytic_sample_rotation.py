"""The SAMPLE-AZIMUTH CLOSED FORM: Partial Analytical carries the sample rotation as a
symbolic psi_s instead of re-solving at every azimuth.

The requirement, stated directly: "Can you make it analytically such that [it] does not
depend on sampling different rotational angles[?]". It works because the SHG is linear in every d
component and `solve_multilayer_shg_symbolic_polarimetry` carries each one as the scalar weight of
a precomputed unit-d source column — so a d that is a trig polynomial in psi_s costs ZERO extra
linear solves.

TIER-1 (the fence that matters): substituting a numeric psi into the closed form must reproduce
re-solving the numerically-rotated sample — the validated route. It does, to ~4e-13.

This fence was RED for three campaigns at 3.2e-01, and the cause was not here: the api passed SI
mu/eps0 into the symbolic solve, making the SHG inhomogeneous operator numerically singular and
corrupting the reference itself (tests/test_pa_crystal_symmetry.py). Keep this
module: it is the fence that would have caught an actual sign/convention error in the azimuth
threading, and it is what re-qualified the feature once the reference was trustworthy.

CONVENTION: the GUI's azimuth is CCW looking at the sample from the beam side, while the solver's
internal +azimuth is a right-hand rotation about the into-sample normal (CW from that view), so the
GUI negates. `case["sample_azimuth_symbol"] = -psi_s` for CCW. The SI closed form
(symbolic.solve_si_shg_full_analytical_symbolic) uses the MIRRORED convention -- deliberately left
alone and filed for review; do not "harmonise" one to the other without deciding which is right.
"""
from __future__ import annotations

import math
import unittest

import sympy as sp

from shaarp.casestudy_materials import build_casestudy_ml_system
from shaarp.config import with_sample_azimuth_deg
import shaarp.shaarp_gui as G

PHI = 25.0
THETA = 20.0
ANGLES = (0.0, 37.0, 113.0, 250.0)


def _value(result, subs):
    expr = sp.sympify(result.stages["reflected_p_2omega"])
    return complex(expr.subs({s: subs.get(str(s), 1.0)
                              for s in expr.free_symbols}).evalf())


class Tier1ClosedFormMatchesTheValidatedRoute(unittest.TestCase):
    def setUp(self):
        self.system = build_casestudy_ml_system("MoS2", thickness_um=0.5, wavelength_um=1.064)

    def _closed_form(self, ccw=True):
        return G.compute_ml_gui_result(
            "Partial Analytical", system=self.system, theta_deg=THETA,
            sample_rotation=True, sample_rotation_ccw=ccw, fixed_phi_deg=PHI)

    def test_substituting_psi_reproduces_the_rotated_solve(self):
        closed = self._closed_form()
        self.assertIn("psi_s", closed.stages["reflected_p_2omega"],
                      "the closed form must carry the azimuth symbol")
        worst = 0.0
        for ang in ANGLES:
            got = _value(closed, {"psi_s": math.radians(ang)})
            reference = G.compute_ml_gui_result(
                "Partial Analytical", theta_deg=THETA, fixed_phi_deg=PHI,
                system=with_sample_azimuth_deg(self.system, -ang, rotate_top=False,
                                               rotate_substrate=False))
            want = _value(reference, {"phi": math.radians(PHI)})
            self.assertGreater(abs(want), 1e-12, "near-zero reference -> vacuous check")
            worst = max(worst, abs(got - want) / abs(want))
        self.assertLess(worst, 1e-9,
                        f"closed form deviates from the numerically-rotated solve: {worst:.2e}")

    def test_the_sign_convention_is_pinned(self):
        """Proves the fence can fail: the CW closed form must NOT match a CCW reference (this is what
        a flipped azimuth sign would look like, and it is the mirror the SI path still uses)."""
        cw = self._closed_form(ccw=False)
        ang = 37.0
        got = _value(cw, {"psi_s": math.radians(ang)})
        ccw_reference = G.compute_ml_gui_result(
            "Partial Analytical", theta_deg=THETA, fixed_phi_deg=PHI,
            system=with_sample_azimuth_deg(self.system, -ang, rotate_top=False,
                                           rotate_substrate=False))
        want = _value(ccw_reference, {"phi": math.radians(PHI)})
        self.assertGreater(abs(got - want) / abs(want), 1e-6,
                           "CW and CCW closed forms agree -> the direction control is inert")

    def test_phi_is_substituted_not_symbolic(self):
        """Under the sample-rotation pin the polarizer is fixed, so the closed form must be a
        function of the azimuth ALONE -- otherwise it would not match the numeric sweep."""
        expr = sp.sympify(self._closed_form().stages["reflected_p_2omega"])
        names = {str(s) for s in expr.free_symbols}
        self.assertIn("psi_s", names, "the azimuth must be the symbolic variable")
        self.assertNotIn("phi", names,
                         f"phi must be substituted under the sample-rotation pin; got {names}")


class FallbackWhenTheClosedFormIsNotPhysical(unittest.TestCase):
    """The gate: a closed form in the azimuth exists only where the rotation leaves eps alone.
    Rotating a biaxial film turns it into a ROTATED biaxial whose k_z solve is the full Booker
    quartic — no closed form — so those fall back to the numeric sweep WITH A STATED REASON."""

    def test_biaxial_film_falls_back_with_a_reason(self):
        system = build_casestudy_ml_system("LiNbO3 x-cut (1550 nm)", thickness_um=0.5,
                                           wavelength_um=1.55)
        result = G.compute_ml_gui_result(
            "Partial Analytical", system=system, theta_deg=THETA, sample_rotation=True,
            sample_rotation_step_deg=60.0, fixed_phi_deg=PHI)
        self.assertEqual(result.kind, "sample_rotation", "must fall back to the numeric sweep")
        reason = result.stages.get("analytic_azimuth_fallback_reason", "")
        self.assertIn("not invariant", reason, f"fallback must say why; got {reason!r}")

    def test_invariant_film_takes_the_closed_form(self):
        result = G.compute_ml_gui_result(
            "Partial Analytical",
            system=build_casestudy_ml_system("MoS2", thickness_um=0.5, wavelength_um=1.064),
            theta_deg=THETA, sample_rotation=True, fixed_phi_deg=PHI)
        self.assertEqual(result.kind, "ml_partial_analytical_polarimetry")
        self.assertNotIn("analytic_azimuth_fallback_reason", result.stages)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
