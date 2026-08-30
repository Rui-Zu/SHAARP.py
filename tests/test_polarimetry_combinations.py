"""Self-consistency fences for the rotate/fix x 3 polarimetry.

Rui: polarizer, analyzer, and sample each carry an independent rotate/fix choice; any of the 8
combinations is legal; every rotating element follows ONE common scan angle t through its rotation
cos(t)/sin(t); and "you need to make sure all those are self-consistent ... with any combination".

The equivalence fences here encode that self-consistency as physics:

* FRAME EQUIVALENCE (normal incidence): rotating the SAMPLE by t (CCW, looking at the sample from
  the beam side) is the same measurement as fixing the sample and rotating the polarizer AND
  analyzer together by +t. This holds exactly only at normal incidence -- at oblique incidence the
  fixed plane of incidence breaks it (measured: 4.9e-10 at theta=0, 1.5e-4 already at 0.5 deg,
  2.4e-3 at 2 deg) -- which is itself the physics, so the fence pins theta=0.
* CO-ROTATION FLATNESS (normal incidence): with ALL THREE rotating, the measurement co-rotates
  with the crystal frame, so on a film whose eps is in-plane isotropic the curve is constant.
  Channel choice matters: for MoS2 (-6m2) the PARALLEL channel is symmetry-dark in this
  configuration (~1e-19 -- pure noise), so the fence reads the PERPENDICULAR channel.
* DIRECTION: the CW curve is the reversed CCW curve (carried from).
* OFFSET GATE: the analyzer-polarizer offset participates only when BOTH polarizer and analyzer
  rotate (the original's `Dynamic[If[RotateAnalyzer, If[RotatePolarizer, offset, ""]]]`).

Plus the Maker Fringes / Fresnel Coefficients compute fences: the Polarimetry panel's phi /
psi / ellipticity genuinely drive the Maker sweep (phi and psi via the parallel channel; the
ellipticity via the PERPENDICULAR channel at phi=45 -- on MoS2 the parallel channel is
|Ex*Ey|-like and Delta-delta-independent by symmetry), and the Fresnel scan range is its own
(min, max, step) with None-defaults reproducing the original 0-89.9 convention.
"""
from __future__ import annotations

import unittest

import numpy as np

from shaarp.casestudy_materials import build_casestudy_ml_system
from shaarp.shaarp_gui import compute_ml_gui_result, ml_sample_rotation_result


def _system():
    return build_casestudy_ml_system("MoS2", thickness_um=0.5, wavelength_um=1.064)


class RotateFixCombinationFences(unittest.TestCase):
    PHI0, PSI0 = 25.0, 10.0

    def test_frame_equivalence_at_normal_incidence(self):
        """rotate-sample-only(t) == fix-sample + co-rotating polarizer AND analyzer at +t.

        NOTE: this drives the NUMERIC routes (ml_sample_rotation_result / run_sample_rotation).
        The analytic azimuth closed form is now ENABLED and is fenced separately, against the
        numerically-rotated solve, in tests/test_ml_analytic_sample_rotation.py (~4e-13) -- do not
        read this test as covering it."""
        from dataclasses import replace

        from shaarp.api import run_sample_rotation

        sysm = _system()
        A = ml_sample_rotation_result(sysm, theta_deg=0.0, fixed_phi_deg=self.PHI0,
                                      analyzer_psi_deg=self.PSI0, step_deg=60.0, ccw=True)
        user = A.numeric["sample_azimuth_deg_user"]
        IA = np.asarray(A.numeric["reflected_parallel_intensity"], dtype=float)
        IB = []
        for t in user:
            pol = replace(sysm.polarimetry, theta_deg=0.0,
                          phi_deg=self.PHI0 + t, psi_deg=self.PSI0 + t)
            r = run_sample_rotation(replace(sysm, polarimetry=pol), [0.0])
            IB.append(float(np.asarray(r.numeric["reflected_parallel_intensity"]).ravel()[0]))
        rel = np.max(np.abs(IA - np.asarray(IB))) / max(1e-30, np.max(np.abs(IA)))
        self.assertLess(rel, 1e-8, f"frame equivalence broken: rel={rel:.3e}")

    def test_direction_mirrors_the_curve(self):
        sysm = _system()
        kw = dict(theta_deg=20.0, fixed_phi_deg=self.PHI0, analyzer_psi_deg=self.PSI0,
                  step_deg=45.0)
        A = ml_sample_rotation_result(sysm, ccw=True, **kw)
        C = ml_sample_rotation_result(sysm, ccw=False, **kw)
        IA = np.asarray(A.numeric["reflected_parallel_intensity"], dtype=float)
        IC = np.asarray(C.numeric["reflected_parallel_intensity"], dtype=float)
        self.assertLess(np.max(np.abs(IC - IA[::-1])), 1e-12 * max(1.0, IA.max()))

    def test_all_three_rotating_is_flat_at_normal_incidence(self):
        sysm = _system()
        D = ml_sample_rotation_result(sysm, theta_deg=0.0, fixed_phi_deg=0.0,
                                      analyzer_psi_deg=0.0, step_deg=60.0, ccw=True,
                                      rotate_polarizer=True, rotate_analyzer=True)
        IP = np.asarray(D.numeric["reflected_perpendicular_intensity"], dtype=float)
        self.assertGreater(IP.mean(), 1e-6, "fence must read a channel that carries signal")
        flat = (IP.max() - IP.min()) / max(1e-30, IP.max())
        self.assertLess(flat, 1e-8, f"co-rotating measurement must be flat: {flat:.3e}")

    def test_offset_participates_only_when_both_rotate(self):
        sysm = _system()
        kw = dict(theta_deg=20.0, fixed_phi_deg=0.0, analyzer_psi_deg=0.0, step_deg=90.0,
                  ccw=True)
        # analyzer rotates alone: a nonzero offset must be inert (the control is greyed)
        a0 = ml_sample_rotation_result(sysm, rotate_analyzer=True, analyzer_offset_deg=0.0, **kw)
        a1 = ml_sample_rotation_result(sysm, rotate_analyzer=True, analyzer_offset_deg=50.0, **kw)
        np.testing.assert_array_equal(
            np.asarray(a0.numeric["reflected_parallel_intensity"]),
            np.asarray(a1.numeric["reflected_parallel_intensity"]),
            err_msg="offset leaked into an analyzer-only rotation")
        # both rotate: the offset must move the curve
        b0 = ml_sample_rotation_result(sysm, rotate_polarizer=True, rotate_analyzer=True,
                                       analyzer_offset_deg=0.0, **kw)
        b1 = ml_sample_rotation_result(sysm, rotate_polarizer=True, rotate_analyzer=True,
                                       analyzer_offset_deg=50.0, **kw)
        d = np.max(np.abs(np.asarray(b0.numeric["reflected_parallel_intensity"])
                          - np.asarray(b1.numeric["reflected_parallel_intensity"])))
        self.assertGreater(d, 1e-12, "offset must move the curve when both elements rotate")

    def test_rotating_polarizer_ignores_fixed_phi(self):
        sysm = _system()
        kw = dict(theta_deg=20.0, analyzer_psi_deg=0.0, step_deg=90.0, ccw=True,
                  rotate_polarizer=True)
        r0 = ml_sample_rotation_result(sysm, fixed_phi_deg=0.0, **kw)
        r1 = ml_sample_rotation_result(sysm, fixed_phi_deg=77.0, **kw)
        np.testing.assert_array_equal(
            np.asarray(r0.numeric["reflected_parallel_intensity"]),
            np.asarray(r1.numeric["reflected_parallel_intensity"]),
            err_msg="fixed φ leaked into a rotating polarizer (dead-control breach)")

    def test_shg_simulation_headless_dispatch(self):
        """compute_ml_gui_result('SHG Simulation', sample_rotation=True) mirrors the GUI path."""
        res = compute_ml_gui_result("SHG Simulation", system=_system(), theta_deg=20.0,
                                    sample_rotation=True, sample_rotation_step_deg=90.0,
                                    fixed_phi_deg=self.PHI0, analyzer_psi_deg=self.PSI0)
        self.assertEqual(res.kind, "sample_rotation")
        st = res.stages["sample_rotation"]
        self.assertEqual(st["fixed_phi_deg"], self.PHI0)
        self.assertFalse(st["rotate_polarizer"])


class MakerFringesPolarizationFences(unittest.TestCase):
    """The Maker Fringes sweep takes its input polarization (φ, Δδ) and detection
    polarization (ψ) from Polarimetry Settings -- all three were dead in Maker mode before."""

    KW = dict(theta_min_deg=0.0, theta_max_deg=30.0, theta_step_deg=5.0)

    def _run(self, **overrides):
        return compute_ml_gui_result("Maker Fringes", system=_system(), **{**self.KW, **overrides})

    def test_phi_moves_the_curve(self):
        base = np.asarray(self._run().numeric["parallel_intensity"])
        moved = np.asarray(self._run(fixed_phi_deg=37.0).numeric["parallel_intensity"])
        self.assertGreater(np.max(np.abs(moved - base)), 0.0, "input polarization φ is dead")

    def test_psi_moves_the_curve(self):
        base = np.asarray(self._run().numeric["parallel_intensity"])
        moved = np.asarray(self._run(analyzer_psi_deg=25.0).numeric["parallel_intensity"])
        self.assertGreater(np.max(np.abs(moved - base)), 0.0, "detection polarization ψ is dead")

    def test_ellipticity_moves_the_perpendicular_channel(self):
        # at φ=45 the s-component carries e^{iΔδ}; MoS2's parallel channel is |Ex·Ey|-like and
        # Δδ-independent by symmetry, so the fence reads the perpendicular channel.
        base = np.asarray(self._run(fixed_phi_deg=45.0).numeric["perpendicular_intensity"])
        moved = np.asarray(self._run(fixed_phi_deg=45.0,
                                     ellipticity_deg=30.0).numeric["perpendicular_intensity"])
        rel = np.max(np.abs(moved - base)) / max(1e-30, np.max(np.abs(base)))
        self.assertGreater(rel, 1e-3, "ellipticity Δδ is dead in the Maker sweep")

    def test_maker_specific_ellipticity_drives_the_maker_sweep(self):
        """The original .ml gives Maker Fringes its OWN delta-delta (-90..90) alongside the
        general polarimetry one. The GUI routes the Maker control into this sweep, so the value
        must reach it -- read on the PERPENDICULAR channel at phi=45, because MoS2's parallel
        channel is Delta-delta-independent by symmetry (see the module docstring)."""
        base = np.asarray(self._run(fixed_phi_deg=45.0).numeric["perpendicular_intensity"])
        moved = np.asarray(self._run(fixed_phi_deg=45.0,
                                     ellipticity_deg=-60.0).numeric["perpendicular_intensity"])
        rel = np.max(np.abs(moved - base)) / max(1e-30, np.max(np.abs(base)))
        self.assertGreater(rel, 1e-3,
                           "the Maker-specific ellipticity never reached the Maker sweep")

    def test_omitted_polarimetry_keeps_system_values(self):
        """None-defaults: a pre-F70 headless call is byte-identical."""
        a = np.asarray(self._run().numeric["parallel_intensity"])
        b = np.asarray(self._run().numeric["parallel_intensity"])
        np.testing.assert_array_equal(a, b)


class FresnelScanRangeFences(unittest.TestCase):
    def test_custom_range_and_step(self):
        res = compute_ml_gui_result("Fresnel Coefficients", system=_system(),
                                    fresnel_min_deg=10.0, fresnel_max_deg=40.0,
                                    fresnel_step_deg=0.1)
        key = next(k for k in res.numeric if "theta" in k)
        grid = np.asarray(res.numeric[key], dtype=float)
        self.assertAlmostEqual(grid[0], 10.0)
        self.assertAlmostEqual(grid[-1], 40.0)
        self.assertEqual(len(grid), 301)

    def test_defaults_reproduce_the_original_convention(self):
        """No Fresnel args -> the original's fixed 0-89.9 grid at theta_step_deg."""
        res = compute_ml_gui_result("Fresnel Coefficients", system=_system(), theta_step_deg=0.5)
        key = next(k for k in res.numeric if "theta" in k)
        grid = np.asarray(res.numeric[key], dtype=float)
        self.assertAlmostEqual(grid[0], 0.0)
        self.assertLessEqual(grid[-1], 89.9 + 1e-9)
        self.assertAlmostEqual(grid[1] - grid[0], 0.5)

    def test_max_clamped_below_90(self):
        res = compute_ml_gui_result("Fresnel Coefficients", system=_system(),
                                    fresnel_min_deg=80.0, fresnel_max_deg=95.0,
                                    fresnel_step_deg=1.0)
        key = next(k for k in res.numeric if "theta" in k)
        self.assertLessEqual(float(np.asarray(res.numeric[key]).max()), 89.9 + 1e-9,
                             "the solver's open interval excludes 90 deg")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
