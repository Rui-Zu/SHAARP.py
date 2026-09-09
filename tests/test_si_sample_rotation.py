"""Sample rotation on the single-interface side, in every one of its modes.

The single-interface method needs no new physics for this: every quantity its solvers use is
derived from the crystal orientation, so turning the orientation turns the permittivity and the
nonlinear tensor together, which is what a physical sample rotation does. What these fences pin is
that the azimuth actually REACHES each mode, and that the displayed expression and the plotted
curve describe the same rotated crystal rather than drifting apart.

The original single-interface package has no sample rotation at all, so this is a new capability
rather than a restored one. It uses the same sense for a positive azimuth as the multilayer side;
that shared sense is fenced in tests/test_sample_rotation_sign_convention.py.
"""

import unittest

import numpy as np

from shaarp.casestudy_materials import build_casestudy_material
from shaarp.shaarp_gui import compute_si_gui_result, si_material_at_azimuth

# z-cut: a rotation about the surface normal leaves the permittivity principal-aligned, so the
# closed forms stay in scope and the azimuth is carried by the nonlinear tensor.
MATERIAL = "LiNbO3 z-cut (1064 nm)"
AZ = 37.0
THETA = 45.0


def _material():
    return build_casestudy_material(MATERIAL)


def _run(functionality, azimuth):
    return compute_si_gui_result(functionality, material=_material(), theta_deg=THETA,
                                 sample_azimuth_deg=azimuth)


class AzimuthReachesEveryMode(unittest.TestCase):
    def test_numeric_mode(self):
        a = _run("SHG Simulation", 0.0).numeric["reflected_intensity"]
        b = _run("SHG Simulation", AZ).numeric["reflected_intensity"]
        self.assertGreater(abs(a - b) / max(abs(a), abs(b)), 1e-6,
                           "the sample azimuth does not reach the numeric single-interface mode")

    def test_partial_analytical_expression(self):
        a = str(_run("Partial Analytical", 0.0).stages["reflected_p_2omega"])
        b = str(_run("Partial Analytical", AZ).stages["reflected_p_2omega"])
        self.assertNotEqual(a, b, "the azimuth does not reach the partial-analytical expression")

    def test_full_analytical_derivation(self):
        """Full Analytical states its answer through named intermediates, so the rotation shows up
        in the derivation stages rather than in the final compact line."""
        a, b = _run("Full Analytical", 0.0), _run("Full Analytical", AZ)
        changed = [k for k in a.stages
                   if str(a.stages.get(k)) != str(b.stages.get(k))]
        self.assertTrue(changed, "the azimuth does not reach the full-analytical derivation")
        self.assertIn("deriv_2_pnl", changed,
                      f"the nonlinear polarization should carry the rotation; changed: {changed}")


class TheRotationIsPhysical(unittest.TestCase):
    def test_it_turns_the_crystal_not_just_the_tensor(self):
        """A sample rotation must move the orientation itself, so permittivity and nonlinear tensor
        turn together. Rotating only one of them is the classic wrong answer."""
        base = _material()
        turned = si_material_at_azimuth(base, AZ)
        a = np.asarray(base.orientation.rotation_matrix(), dtype=float)
        b = np.asarray(turned.orientation.rotation_matrix(), dtype=float)
        self.assertGreater(float(np.max(np.abs(a - b))), 1e-6, "the orientation did not move")

    def test_a_full_turn_is_the_identity(self):
        base = _material()
        a = np.asarray(base.orientation.rotation_matrix(), dtype=float)
        b = np.asarray(si_material_at_azimuth(base, 360.0).orientation.rotation_matrix(),
                       dtype=float)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_zero_azimuth_is_a_no_op(self):
        base = _material()
        self.assertIs(si_material_at_azimuth(base, 0.0), base)

    def test_direction_mirrors_the_rotation(self):
        """Clockwise and counter-clockwise must not agree, or the direction control is inert."""
        base = _material()
        ccw = si_material_at_azimuth(base, AZ, ccw=True).orientation.rotation_matrix()
        cw = si_material_at_azimuth(base, AZ, ccw=False).orientation.rotation_matrix()
        self.assertGreater(float(np.max(np.abs(np.asarray(ccw) - np.asarray(cw)))), 1e-6)


class TheAzimuthCanStaySymbolic(unittest.TestCase):
    """Parity with the multilayer side: the closed forms carry the azimuth as a symbol, so one
    solve describes every angle instead of one solve per angle."""

    def _closed(self, functionality, ccw=True):
        return compute_si_gui_result(functionality, material=_material(), theta_deg=THETA,
                                     sample_rotation=True, sample_rotation_ccw=ccw)

    def test_partial_analytical_is_symbolic_in_the_azimuth(self):
        txt = str(self._closed("Partial Analytical").stages["reflected_p_2omega"])
        self.assertIn("psi_s", txt, "the closed form must carry the azimuth symbol")

    def test_full_analytical_carries_it_through_the_derivation(self):
        """The layered form states its answer through named intermediates, so the symbol lives in
        those definitions rather than in the final compact line."""
        st = self._closed("Full Analytical").stages
        carriers = [k for k, v in st.items() if "psi_s" in str(v)]
        self.assertIn("deriv_2_pnl", carriers,
                      f"the nonlinear polarization should carry the azimuth; got {carriers}")

    def test_substituting_the_symbol_reproduces_turning_the_crystal(self):
        """THE fence that matters: the symbolic route and the physically turned crystal are the
        same answer. Without this, a wrong sign or a wrong contraction would still look plausible."""
        import math
        import sympy as sp

        expr = sp.sympify(str(self._closed("Partial Analytical").stages["reflected_p_2omega"]))
        self.assertIn("psi_s", {str(x) for x in expr.free_symbols})
        got = complex(expr.subs({x: (math.radians(AZ) if str(x) == "psi_s" else 1.0)
                                 for x in expr.free_symbols}).evalf())
        ref = sp.sympify(str(_run("Partial Analytical", AZ).stages["reflected_p_2omega"]))
        want = complex(ref.subs({x: 1.0 for x in ref.free_symbols}).evalf())
        self.assertGreater(abs(want), 1e-30, "near-zero reference makes this vacuous")
        self.assertLess(abs(got - want) / abs(want), 1e-9,
                        "the symbolic azimuth disagrees with physically turning the crystal")

    def test_the_direction_is_not_ignored(self):
        """Proves the fence can fail: the two directions must not produce the same expression."""
        self.assertNotEqual(str(self._closed("Partial Analytical", ccw=True).stages["reflected_p_2omega"]),
                            str(self._closed("Partial Analytical", ccw=False).stages["reflected_p_2omega"]))

    def test_a_turning_permittivity_declines_instead_of_guessing(self):
        """A crystal whose permittivity turns with the sample has no closed form in the angle. The
        result must say so rather than emit a confidently wrong expression."""
        biaxial = build_casestudy_material("KTP x-cut")
        res = compute_si_gui_result("Partial Analytical", material=biaxial, theta_deg=THETA,
                                    sample_rotation=True)
        note = str(res.stages.get("symbols", {}))
        self.assertNotIn("psi_s (radians", note,
                         "a rotated-biaxial case must not be offered as a closed form in the angle")


if __name__ == "__main__":
    unittest.main()
