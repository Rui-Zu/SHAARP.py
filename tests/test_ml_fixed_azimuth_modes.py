"""A FIXED sample azimuth for the multilayer modes that scan something else.

Maker fringes scan the incidence angle and Fresnel coefficients are linear optics, so neither
sweeps the azimuth. Both can still be evaluated with the sample turned, and the rotation is applied
once before the sweep so the angle-independent setup stays hoisted out of the per-angle loop.

Two things here are easy to get wrong and are fenced deliberately:

- ``with_sample_azimuth_deg`` defaults to rotating the substrate, while the rotation sweep pins it
  off. Taking the default would make a Maker curve at an azimuth disagree with the sweep's point at
  that same azimuth, with nothing on screen to explain it.
- For a stack whose permittivity is unchanged by a rotation about the surface normal, the Fresnel
  coefficients genuinely do not depend on the azimuth. That is physics rather than a dead control,
  so the result has to SAY it rather than the app quietly hiding the setting.
"""

import unittest
from dataclasses import replace

import numpy as np

from shaarp.shaarp_gui import compute_ml_gui_result, resolve_ml_system_preset

ISOTROPIC_STACK = "Quartz + Au (Fig 4, 800 nm)"   # z-cut quartz: eps invariant about the normal
AZ = 37.0


def _maker(system=None, preset=None, azimuth=0.0):
    kw = dict(theta_deg=20.0, fixed_phi_deg=0.0, analyzer_psi_deg=0.0, ellipticity_deg=0.0,
              theta_min_deg=0.0, theta_max_deg=20.0, theta_step_deg=5.0,
              sample_azimuth_deg=azimuth)
    if system is not None:
        return compute_ml_gui_result("Maker Fringes", system=system, **kw)
    return compute_ml_gui_result("Maker Fringes", system_preset=preset, **kw)


def _anisotropic_stack():
    """Give the film a permittivity that turns with the sample, so the azimuth must matter."""
    base = resolve_ml_system_preset(ISOTROPIC_STACK)
    layers = list(base.layers)
    mat = layers[1].material
    biaxial = np.diag([2.10 + 0j, 2.75 + 0j, 2.40 + 0j])
    layers[1] = replace(layers[1], material=replace(mat, epsilon_omega=biaxial,
                                                    epsilon_2omega=biaxial))
    return replace(base, layers=tuple(layers))


def _curve(res):
    """The scanned quantity, whichever mode produced it: Maker returns intensities, Fresnel
    returns the four amplitude coefficients."""
    n = res.numeric
    keys = [k for k in ("parallel_intensity", "perpendicular_intensity", "rp", "rs", "tp", "ts")
            if k in n]
    if not keys:
        raise AssertionError(f"no scanned quantity in {sorted(n)}")
    return np.concatenate([np.abs(np.asarray(n[k])).astype(float).ravel() for k in keys])


class MakerAtAFixedAzimuth(unittest.TestCase):
    def test_the_azimuth_changes_the_maker_curve(self):
        system = _anisotropic_stack()
        a = _curve(_maker(system=system, azimuth=0.0))
        b = _curve(_maker(system=system, azimuth=AZ))
        scale = max(1e-30, float(np.max(np.abs(a))))
        self.assertGreater(float(np.max(np.abs(a - b))) / scale, 1e-6,
                           "the fixed azimuth does not reach the Maker sweep")

    def test_it_agrees_with_the_rotation_sweep_at_the_same_azimuth(self):
        """The footgun fence. Both routes must turn the SAME layers; the helper's substrate
        default differs from the sweep's, so a mismatch here means one of them took it."""
        from shaarp.shaarp_gui import _with_fixed_sample_azimuth, azimuth_user_to_solver
        from shaarp.config import with_sample_azimuth_deg

        base = _anisotropic_stack()
        mine = _with_fixed_sample_azimuth(base, AZ, True)
        sweep_style = with_sample_azimuth_deg(base, azimuth_user_to_solver(AZ, True),
                                              rotate_top=False, rotate_substrate=False)
        for a, b in zip(mine.layers, sweep_style.layers):
            np.testing.assert_allclose(
                np.asarray(a.material.orientation.rotation_matrix(), dtype=float),
                np.asarray(b.material.orientation.rotation_matrix(), dtype=float),
                rtol=0, atol=0,
                err_msg="the fixed-azimuth rotation turns different layers than the sweep does")

    def test_the_setup_is_still_hoisted_out_of_the_angle_loop(self):
        """A fixed azimuth must not cost an eigen-decomposition per angle. If this count starts
        scaling with the number of angles, the hoist is broken and the mode got expensive."""
        import shaarp.multilayer_shg_boundary as B

        calls = {"n": 0}
        original = B._system_setup

        def counting(system):
            calls["n"] += 1
            return original(system)

        B._system_setup = counting
        try:
            _maker(preset=ISOTROPIC_STACK, azimuth=AZ)
        finally:
            B._system_setup = original
        self.assertLessEqual(calls["n"], 2,
                             f"setup rebuilt {calls['n']} times: the angle-independent hoist broke")


class FresnelAtAFixedAzimuth(unittest.TestCase):
    def _fresnel(self, system, azimuth):
        return compute_ml_gui_result(
            "Fresnel Coefficients", system=system, theta_deg=20.0,
            fresnel_min_deg=0.0, fresnel_max_deg=40.0, fresnel_step_deg=10.0,
            sample_azimuth_deg=azimuth)

    def test_it_matters_for_an_anisotropic_stack(self):
        system = _anisotropic_stack()
        a = _curve(self._fresnel(system, 0.0))
        b = _curve(self._fresnel(system, AZ))
        scale = max(1e-30, float(np.max(np.abs(a))))
        self.assertGreater(float(np.max(np.abs(a - b))) / scale, 1e-9,
                           "an anisotropic layer's coefficients should depend on the azimuth")
        self.assertIn("changes these coefficients",
                      self._fresnel(system, AZ).stages["sample_azimuth_effect"])

    def test_it_is_inert_for_an_isotropic_stack_and_says_so(self):
        system = resolve_ml_system_preset(ISOTROPIC_STACK)
        a = _curve(self._fresnel(system, 0.0))
        b = _curve(self._fresnel(system, AZ))
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14,
                                   err_msg="azimuth changed a rotationally invariant stack")
        self.assertIn("no effect", self._fresnel(system, AZ).stages["sample_azimuth_effect"])


if __name__ == "__main__":
    unittest.main()
