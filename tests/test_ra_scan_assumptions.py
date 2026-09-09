"""SAMPLE ROTATION x ASSUMPTIONS — the Assumptions panel must reach the azimuth sweep.

The gap these fences close: `ml_sample_rotation_result` called `run_sample_rotation(ra_sys, solver)`
with NO options, and `run_sample_rotation` pinned `mrassumption=0` / `forward_only`. So Rotate
Sample always ran FMR/forward-only whatever the panel said, while the schematic caption showed the
panel's value. The original's `SampleRotate` branches on its `assumption` argument (setup.nb:
`If[assumption==0/1/2, ... f4NL[...]]`, with JK/HH setting flagBackward = flagStandingWave = False),
and the solver has been Mathematica-validated for 0/1/2 (test_jkhh_samplerotate_agreement.py).

Same failure class the T1 input-sensitivity gate caught on the 4-polar branch: a silently dropped
input reads as "working" because the plot still renders.
"""

import unittest

from dataclasses import replace

import numpy as np

from shaarp.api import run_sample_rotation
from shaarp.casestudy_materials import build_casestudy_ml_system
from shaarp.multilayer_shg_boundary import solve_multilayer_shg_sample_azimuth_sweep
from shaarp.shaarp_gui import (
    ml_sample_rotation_assumption_options,
    ml_sample_rotation_grid,
    ml_sample_rotation_result,
)

KW = dict(theta_deg=45.0, fixed_phi_deg=0.0, analyzer_psi_deg=0.0, step_deg=90.0)


def _system():
    """A film thick enough that multiple reflections matter, so FMR/JK/HH really differ."""
    return build_casestudy_ml_system("MoS2", thickness_um=0.5, wavelength_um=1.064)


def _refl(res):
    return np.asarray(res.numeric["reflected_parallel_intensity"], dtype=float)


class AssumptionReachesTheSweep(unittest.TestCase):
    def test_jk_and_hh_change_the_result(self):
        """Input-sensitivity fence: if the assumption is dropped again, these go equal."""
        sysm = _system()
        full = _refl(ml_sample_rotation_result(sysm, mrassumption=0, **KW))
        scale = max(1e-30, float(np.max(np.abs(full))))
        for code, name in ((1, "JK"), (2, "HH")):
            other = _refl(ml_sample_rotation_result(sysm, mrassumption=code, **KW))
            rel = float(np.max(np.abs(other - full))) / scale
            self.assertGreater(rel, 1e-6, f"{name} is indistinguishable from FMR (rel={rel:.3e}) "
                                          "-- the assumption is not reaching the azimuth sweep")

    def test_fmr_submode_changes_the_result(self):
        """The FMR sub-mode (backward / standing waves) must reach it too."""
        sysm = _system()
        fwd = _refl(ml_sample_rotation_result(
            sysm, mrassumption=0, inhomogeneous_source_policy="forward_only", **KW))
        allw = _refl(ml_sample_rotation_result(
            sysm, mrassumption=0, inhomogeneous_source_policy="all", **KW))
        rel = float(np.max(np.abs(allw - fwd))) / max(1e-30, float(np.max(np.abs(fwd))))
        self.assertGreater(rel, 1e-6, f"FMR sub-mode does not reach the sweep (rel={rel:.3e})")

    def test_matches_the_validated_solver_call(self):
        """Equality fence: the GUI helper == the Mathematica-validated solver entry point."""
        sysm = _system()
        for code in (0, 1, 2):
            got = _refl(ml_sample_rotation_result(sysm, mrassumption=code, **KW))
            _user, solver = ml_sample_rotation_grid(KW["step_deg"], True)
            gui_sys = ml_sample_rotation_result(
                sysm, mrassumption=code, **KW).stages["sample_rotation"]["system"]
            raw = solve_multilayer_shg_sample_azimuth_sweep(
                gui_sys, sample_azimuth_deg=solver, rotate_top=False, rotate_substrate=False,
                inhomogeneous_source_policy="forward_only", inhomogeneous_solution_policy="solve",
                mrassumption=code)
            want = np.asarray(raw.reflected_parallel_intensity, dtype=float)
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-30,
                                       err_msg=f"mrassumption={code} diverges from the solver")

    def test_library_default_is_the_per_point_loop(self):
        """run_sample_rotation's own contract is unchanged: no options == the honest per-point
        loop, one solve per azimuth point. Only the GUI helper opts into the fast path."""
        sysm = _system()
        grid = ml_sample_rotation_grid(KW["step_deg"], True)[1]
        gui_sys = ml_sample_rotation_result(sysm, **KW).stages["sample_rotation"]["system"]
        res = run_sample_rotation(gui_sys, grid)
        self.assertEqual(len(res.stages["results"]), len(grid),
                         "the library default must still solve every point")
        self.assertFalse(res.stages["fast_linear_d_used"])

    def test_stages_record_what_ran(self):
        res = ml_sample_rotation_result(_system(), mrassumption=2,
                                        inhomogeneous_source_policy="forward_only", **KW)
        self.assertEqual(res.stages["sample_rotation"]["mrassumption"], 2)
        self.assertEqual(res.stages["mrassumption"], 2)


class DLinearityFastPath(unittest.TestCase):
    """The azimuth sweep as a LINEAR COMBINATION of one solve per touched d component.

    Physics: where every rotating layer's lab eps is invariant about the surface normal, the whole
    linear problem is azimuth-independent and only the nonlinear source turns, so the sweep is a
    linear map of d(psi_s). Cost goes from one solve per point to one per component (6 vs 181 on
    quartz + Au, ~25x wall clock), using the SAME validated solver.

    TOLERANCE, measured not guessed: the per-point loop is only self-consistent to 1.2e-12 of peak
    (comparing azimuth g against g+360, physically identical, different float path), and the fast
    path agrees with it to 3.3e-11 of peak. 1e-9 sits above both and far below anything physical.
    """

    TOL = 1e-9

    def _sweep(self, **kw):
        sysm = _system()
        pol = replace(sysm.polarimetry, theta_deg=45.0, phi_deg=0.0, psi_deg=0.0,
                      ellipticity_deg=0.0)
        grid = ml_sample_rotation_grid(10.0, True)[1]
        return solve_multilayer_shg_sample_azimuth_sweep(
            replace(sysm, polarimetry=pol), sample_azimuth_deg=grid, **kw)

    def test_fast_matches_the_per_point_loop(self):
        for mr, pol in ((0, "forward_only"), (0, "all"), (1, "forward_only"), (2, "forward_only")):
            slow = self._sweep(mrassumption=mr, inhomogeneous_source_policy=pol)
            fast = self._sweep(mrassumption=mr, inhomogeneous_source_policy=pol,
                               fast_linear_d=True)
            for key in ("reflected_parallel_intensity", "reflected_perpendicular_intensity",
                        "transmitted_parallel_intensity", "transmitted_perpendicular_intensity"):
                a = np.asarray(getattr(slow, key), dtype=float)
                b = np.asarray(getattr(fast, key), dtype=float)
                peak = max(1e-30, float(np.max(np.abs(a))))
                dev = float(np.max(np.abs(a - b))) / peak
                self.assertLess(dev, self.TOL,
                                f"fast path deviates on {key} (mr={mr}, {pol}): {dev:.2e}")

    def test_it_really_took_the_shortcut(self):
        """Proves the speedup is real and that the fence above is not comparing loop with loop."""
        slow = self._sweep()
        fast = self._sweep(fast_linear_d=True)
        n_points = np.asarray(slow.sample_azimuth_deg).size
        self.assertEqual(len(slow.results), n_points)
        self.assertLess(len(fast.results), n_points,
                        "fast_linear_d did not reduce the solve count")

    def test_it_falls_back_when_eps_is_not_invariant(self):
        """A rotating layer whose lab eps turns with the sample (a rotated biaxial) has no shared
        basis. The fast path must decline and give the loop's answer, not a wrong cheap one."""
        sysm = _system()
        lay = list(sysm.layers)
        mat = lay[1].material
        biaxial = np.diag([2.1 + 0j, 2.6 + 0j, 3.4 + 0j])          # eps_x != eps_y -> turns with psi_s
        lay[1] = replace(lay[1], material=replace(mat, epsilon_omega=biaxial,
                                                  epsilon_2omega=biaxial))
        pol = replace(sysm.polarimetry, theta_deg=45.0, phi_deg=0.0, psi_deg=0.0,
                      ellipticity_deg=0.0)
        sysb = replace(sysm, layers=tuple(lay), polarimetry=pol)
        grid = ml_sample_rotation_grid(30.0, True)[1]
        slow = solve_multilayer_shg_sample_azimuth_sweep(sysb, sample_azimuth_deg=grid)
        fast = solve_multilayer_shg_sample_azimuth_sweep(sysb, sample_azimuth_deg=grid,
                                                         fast_linear_d=True)
        self.assertEqual(len(fast.results), len(slow.results), "it should have fallen back")
        np.testing.assert_allclose(
            np.asarray(fast.reflected_parallel_intensity, dtype=float),
            np.asarray(slow.reflected_parallel_intensity, dtype=float), rtol=0, atol=0,
            err_msg="the fallback must be the loop itself, bit-for-bit")

    def test_gui_helper_uses_it_and_says_so(self):
        """At a real step (37 points) the shortcut engages and the result says it did."""
        kw = dict(KW, step_deg=10.0)
        res = ml_sample_rotation_result(_system(), **kw)
        self.assertTrue(res.stages["sample_rotation"]["fast_linear_d"])
        self.assertTrue(res.stages["sample_rotation"]["fast_linear_d_used"],
                        "the GUI RA sweep should take the fast path on this stack")

    def test_it_declines_when_there_is_nothing_to_gain(self):
        """A coarse grid with fewer points than basis components must NOT pay for the basis: the
        guard keeps the honest loop. (KW's 90 deg step is 5 points.)"""
        res = ml_sample_rotation_result(_system(), **KW)
        self.assertFalse(res.stages["sample_rotation"]["fast_linear_d_used"])


class PanelMapping(unittest.TestCase):
    """FMR keeps the panel sub-mode; JK/HH force forward-only -- the original's own branches
    (flagBackward = flagStandingWave = False), and the setting the JK/HH SampleRotate references
    were validated under."""

    def test_fmr_keeps_the_submode(self):
        got = ml_sample_rotation_assumption_options(
            "Full Multiple Reflections (FMR)", "Forward + Backward + Standing waves")
        self.assertEqual(got, {"mrassumption": 0, "inhomogeneous_source_policy": "all"})

    def test_jk_and_hh_force_forward_only(self):
        for label, code in (("Jerphagnon & Kurtz Assumption (No MR)", 1),
                            ("Herman & Hayden Assumption (MR only for 2ω Homo Waves)", 2)):
            got = ml_sample_rotation_assumption_options(
                label, "Forward + Backward + Standing waves")
            self.assertEqual(got, {"mrassumption": code,
                                   "inhomogeneous_source_policy": "forward_only"})


class TheAssumptionReachesTheFundamentalStage(unittest.TestCase):
    """The multiple-reflection assumption governs the FUNDAMENTAL solve too, not only the
    second-harmonic stage: Jerphagnon-Kurtz and Herman-Hayden take a single pass at omega. So the
    reflected and transmitted fundamental beams depend on it -- their Fresnel coefficients and
    their polarization ellipses both. Each of these was silently pinned to full reflections."""

    def _system(self):
        from shaarp.shaarp_gui import resolve_ml_system_preset
        return resolve_ml_system_preset("Quartz + Au (Fig 4, 800 nm)")

    def test_the_beam_ellipses_move_with_it(self):
        from shaarp.shaarp_gui import ml_beam_ellipses

        def amps(code):
            e = ml_beam_ellipses(self._system(), theta_deg=45.0, phi_deg=0.0, mrassumption=code)
            return np.array([abs(e["reflected"][1]), abs(e["transmitted"][1])])

        full, single = amps(0), amps(1)
        scale = max(1e-30, float(np.max(np.abs(full))))
        self.assertGreater(float(np.max(np.abs(full - single))) / scale, 1e-6,
                           "the fundamental ellipses ignore the assumption")

    def test_the_fresnel_coefficients_move_with_it(self):
        from shaarp.shaarp_gui import compute_ml_gui_result

        def curve(assumption):
            r = compute_ml_gui_result("Fresnel Coefficients", system=self._system(),
                                      theta_deg=45.0, assumption=assumption,
                                      fresnel_min_deg=0.0, fresnel_max_deg=40.0,
                                      fresnel_step_deg=10.0)
            return np.concatenate([np.abs(np.asarray(r.numeric[k])).ravel()
                                   for k in ("rp", "rs", "tp", "ts")])

        full = curve("Full Multiple Reflections (FMR)")
        jk = curve("Jerphagnon & Kurtz Assumption (No MR)")
        scale = max(1e-30, float(np.max(np.abs(full))))
        self.assertGreater(float(np.max(np.abs(full - jk))) / scale, 1e-6,
                           "the Fresnel coefficients ignore the assumption")


if __name__ == "__main__":
    unittest.main()
