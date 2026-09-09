"""The multilayer closed form must equal the numeric route in ABSOLUTE magnitude AND phase.

This is the fence that did not exist. Every prior symbolic-vs-numeric test built BOTH sides at
omega = 1, so a frequency that did not match the units the thickness is expressed in was invisible
to all of them: the closed form was describing a film of optical thickness h*lambda/(2 pi), a
different point on the Maker fringe, and the disagreement was a complex factor that moved with
angle, thickness and wavelength while staying constant in azimuth.

Frequency reaches the answer through exactly one quantity, the propagation phase exp(i k_z h). The
magnetic field is frequency-invariant and the inhomogeneous solve scales its matrix and its
right-hand side together, so nothing else in the chain carries omega. That is why the wrong
frequency looked like a plausible answer at every angle.

The negative control is load-bearing: it asserts the reduced-unit build does NOT match, so a
regression that reinstates a hard-coded frequency fails here instead of passing vacuously.
"""

import unittest
from dataclasses import replace

import numpy as np
import sympy as sp

from shaarp.api import run_ml_partial_analytical
from shaarp.multilayer_shg_boundary import (
    _d_voigt_lab,
    _epsilon_lab,
    solve_multilayer_shg_from_system_polarimetry,
)
from shaarp.shaarp_gui import resolve_ml_system_preset

PRESET = "Quartz + Au (Fig 4, 800 nm)"
PHI = 0.0


def _case(system, theta_deg, policy, wavelength_um):
    interior = list(system.layers)[1:-1]
    top, substrate = system.layers[0].material, system.layers[-1].material
    return {
        "point_group": interior[0].material.structure.point_group,
        "incident_theta_rad": float(np.deg2rad(theta_deg)),
        "layer_epsilon_omega_lab": [_epsilon_lab(L.material, omega=True) for L in interior],
        "layer_epsilon_2omega_lab": [_epsilon_lab(L.material, omega=False) for L in interior],
        "layer_d_voigt_symbolic": [
            sp.Matrix(_d_voigt_lab(L.material) if L.shg_active else np.zeros((3, 6), dtype=complex))
            for L in interior],
        "thickness_symbols": [sp.Float(float(L.thickness_um or 0.0)) for L in interior],
        "substrate_epsilon_omega_lab": _epsilon_lab(substrate, omega=True),
        "substrate_epsilon_2omega_lab": _epsilon_lab(substrate, omega=False),
        "ambient_index_omega": complex(np.sqrt(np.trace(top.eps_w()) / 3)),
        "ambient_index_2omega": complex(np.sqrt(np.trace(top.eps_2w()) / 3)),
        "phi_symbol": sp.Float(float(np.deg2rad(PHI))),
        "wavelength_um": wavelength_um,
        "inhomogeneous_source_policy": policy,
    }


def _closed(system, theta_deg, policy, wavelength_um):
    res = run_ml_partial_analytical(_case(system, theta_deg, policy, wavelength_um),
                                    {"workflow": "polarimetry"})
    expr = sp.sympify(res.stages["reflected_p_2omega"])
    return complex(expr.evalf()), res


def _numeric(system, theta_deg, policy):
    pol = replace(system.polarimetry, theta_deg=float(theta_deg), phi_deg=PHI, psi_deg=0.0,
                  ellipticity_deg=0.0)
    r = solve_multilayer_shg_from_system_polarimetry(
        replace(system, polarimetry=pol), inhomogeneous_source_policy=policy,
        inhomogeneous_solution_policy="solve")
    return complex(np.asarray(r.shg.coefficients)[1])


class AbsoluteScale(unittest.TestCase):
    TOL = 1e-9   # measured worst deviation over this grid: 7.2e-12

    def setUp(self):
        self.base = resolve_ml_system_preset(PRESET)

    def _ratio(self, system, theta, policy, wavelength_um):
        got, _ = _closed(system, theta, policy, wavelength_um)
        want = _numeric(system, theta, policy)
        self.assertGreater(abs(want), 1e-20, "near-zero reference makes the check vacuous")
        return got / want

    def test_matches_across_angle_and_wavelength(self):
        for theta in (20.0, 45.0, 65.0):
            for lam in (0.4, 0.8, 1.6):
                system = replace(self.base, wavelength_um=lam)
                r = self._ratio(system, theta, "forward_only", lam)
                self.assertLess(abs(r - 1.0), self.TOL,
                                f"theta={theta}, lambda={lam}: ratio {r!r}")

    def test_matches_across_thickness(self):
        for h in (60.6, 121.2, 242.4):
            layers = list(self.base.layers)
            layers[1] = replace(layers[1], thickness_um=h)
            system = replace(self.base, layers=tuple(layers))
            r = self._ratio(system, 45.0, "forward_only", system.wavelength_um)
            self.assertLess(abs(r - 1.0), self.TOL, f"h={h} um: ratio {r!r}")

    def test_every_source_policy_matches_its_own_counterpart(self):
        for policy in ("forward_only", "forward_backward", "all"):
            r = self._ratio(self.base, 45.0, policy, self.base.wavelength_um)
            self.assertLess(abs(r - 1.0), self.TOL, f"policy={policy}: ratio {r!r}")

    def test_the_policies_are_not_all_the_same(self):
        """Otherwise the policy parameter would be inert and the test above vacuous."""
        vals = [_closed(self.base, 45.0, p, self.base.wavelength_um)[0]
                for p in ("forward_only", "forward_backward", "all")]
        scale = max(abs(v) for v in vals)
        self.assertGreater(abs(vals[0] - vals[2]) / scale, 1e-6,
                           "the source policy does not reach the closed form")

    def test_negative_control_reduced_units_do_not_match(self):
        """A regression that reinstates a hard-coded frequency must FAIL, not pass vacuously."""
        r = self._ratio(self.base, 45.0, "forward_only", None)
        self.assertGreater(abs(r - 1.0), 1e-3,
                           "reduced units agreed with a micron thickness -- the fence above cannot "
                           "be distinguishing the two builds")

    def test_the_expression_says_which_units_it_used(self):
        _, res = _closed(self.base, 45.0, "forward_only", 0.8)
        syms = res.stages["symbols"]
        self.assertIn("um", syms["wavelength"])
        self.assertEqual(syms["source_waves"], "forward_only")
        _, reduced = _closed(self.base, 45.0, "forward_only", None)
        self.assertIn("reduced units", reduced.stages["symbols"]["wavelength"])


class AssumptionsReachTheClosedForm(unittest.TestCase):
    """The Assumptions panel used to change nothing in Partial Analytical: the closed form was
    always built with every bound wave in the source while the plotted numeric curve used the
    panel's choice, so the expression and the curve described different physics. Dead-control
    class, same as the sample-rotation sweep before it was wired."""

    def _result(self, assumption, submode="Forward waves only"):
        from shaarp.shaarp_gui import compute_ml_gui_result

        return compute_ml_gui_result(
            "Partial Analytical", system_preset=PRESET, theta_deg=45.0, fixed_phi_deg=0.0,
            assumption=assumption, fmr_submode=submode)

    def _expression(self, assumption, submode="Forward waves only"):
        return self._result(assumption, submode).stages["reflected_p_2omega"]

    def test_the_fmr_submode_changes_the_expression(self):
        a = self._expression("Full Multiple Reflections (FMR)", "Forward waves only")
        b = self._expression("Full Multiple Reflections (FMR)",
                             "Forward + Backward + Standing waves")
        self.assertNotEqual(a, b, "the FMR sub-mode does not reach the closed form")

    def test_single_pass_assumptions_are_declared_not_faked(self):
        """JK and HH drop multiply-reflected waves. The closed form has no single-pass parameter,
        so it cannot express them; it must SAY that rather than return the full-reflection
        expression under a JK/HH label."""
        for label in ("Jerphagnon & Kurtz Assumption (No MR)",
                      "Herman & Hayden Assumption (MR only for 2ω Homo Waves)"):
            note = self._result(label).stages["symbols"]["assumption"]
            self.assertIn("not represented", note, f"{label} is silently faked")
        fmr = self._result("Full Multiple Reflections (FMR)").stages["symbols"]["assumption"]
        self.assertNotIn("not represented", fmr)

    def test_the_gui_supplies_a_physical_wavelength(self):
        """Without this the expression is built in reduced units while its thicknesses are in
        microns, which is the defect this module exists for."""
        from shaarp.shaarp_gui import compute_ml_gui_result

        res = compute_ml_gui_result("Partial Analytical", system_preset=PRESET, theta_deg=45.0,
                                    fixed_phi_deg=0.0)
        self.assertIn("um", res.stages["symbols"]["wavelength"])
        self.assertNotIn("reduced units", res.stages["symbols"]["wavelength"])


if __name__ == "__main__":
    unittest.main()
