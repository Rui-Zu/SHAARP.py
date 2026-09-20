"""Multilayer SHG across a wavelength grid.

WHY THIS EXISTS. On a multilayer the wavelength reaches the answer through TWO channels -- the
permittivity, and the propagation phase exp(i k_z h) -- and that second channel makes the
multilayer sweep much easier to get subtly wrong than the single-interface one.

The sharpest available fence is an exact identity rather than a reference point. At frozen
permittivity a multilayer answer depends on omega only through the products omega*h, so scaling
every layer thickness in step with the wavelength must leave the result completely unchanged. It
does, to 3e-14 across a 2.8x span of wavelength, while the same sweep without the thickness
scaling moves by more than the mean. That is a claim about every wavelength, it needs no external
reference, and it fails immediately if the wavelength is ever wired to the permittivity but not to
omega, or the reverse.

The other reason this file exists is the SILENT case. A multilayer sweep at frozen permittivity
still produces a curve that MOVES, because the propagation phase responds to omega on its own. So
unlike the single-interface path, where a frozen permittivity gives an obviously wrong flat line,
here it gives a plausible-looking spectrum that is really a thickness sweep. The support check has
to catch that from the permittivity alone -- never from whether the output varies -- and this file
pins that behaviour.
"""
from __future__ import annotations

import unittest
import warnings
from dataclasses import replace

import numpy as np

from shaarp.casestudy_materials import build_casestudy_ml_system
from shaarp.spectral import (
    casestudy_ml_spectrum,
    solve_ml_spectral_sweep,
    wavelength_grid,
)

DISPERSIVE = "Quartz z-cut (800 nm)"
CONSTANT = "GaAs (111) (800 nm)"
POLED = "KTP x-cut"
LAMBDA0, THICKNESS0 = 0.8, 5.0


def _base_system(name=DISPERSIVE, thickness_um=THICKNESS0, wavelength_um=LAMBDA0):
    return build_casestudy_ml_system(name, thickness_um=thickness_um, wavelength_um=wavelength_um)


class TheOmegaTimesThicknessIdentity(unittest.TestCase):
    """At frozen permittivity the answer depends on omega only through omega*h."""

    def test_scaling_every_thickness_with_the_wavelength_changes_nothing(self):
        base = _base_system()

        def scaled(lam):
            layers = list(base.layers)
            layers[1] = replace(layers[1], thickness_um=THICKNESS0 * float(lam) / LAMBDA0)
            return replace(base, wavelength_um=float(lam), layers=layers)

        grid = [0.6, 0.7, 0.8, 1.0, 1.3, 1.7]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sweep = solve_ml_spectral_sweep(scaled, wavelength_um=grid, theta_deg=20.0,
                                            phi_deg=30.0, allow_constant_dispersion=True)
        deviation = np.abs(sweep.intensity - sweep.intensity[0]) / abs(sweep.intensity[0])
        self.assertLess(deviation.max(), 1e-11,
                        f"omega*h invariance broken by {deviation.max():.2e}: the wavelength is "
                        "reaching the multilayer answer through something other than the "
                        "permittivity and the propagation phase")

    def test_without_the_thickness_scaling_the_answer_really_does_move(self):
        """Guards the test above: an identity that holds trivially fences nothing."""
        base = _base_system()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sweep = solve_ml_spectral_sweep(lambda lam: replace(base, wavelength_um=float(lam)),
                                            wavelength_um=[0.6, 0.7, 0.8, 1.0, 1.3, 1.7],
                                            theta_deg=20.0, phi_deg=30.0,
                                            allow_constant_dispersion=True)
        spread = (sweep.intensity.max() - sweep.intensity.min()) / sweep.intensity.mean()
        self.assertGreater(spread, 0.5, "the control must move, or the identity proves nothing")


class TheSweepItself(unittest.TestCase):

    def test_a_dispersive_film_produces_a_moving_spectrum_with_tiny_residuals(self):
        sweep = solve_ml_spectral_sweep(
            casestudy_ml_spectrum(DISPERSIVE, thickness_um=THICKNESS0),
            lambda_min_um=0.6, lambda_max_um=1.6, lambda_step_um=0.1,
            theta_deg=20.0, phi_deg=30.0)
        self.assertEqual(sweep.wavelength_um.size, 11)
        self.assertGreater(sweep.intensity.min(), 0.0)
        self.assertFalse(np.allclose(sweep.intensity, sweep.intensity[0]))
        self.assertLess(sweep.fundamental_residual_norm.max(), 1e-10)
        self.assertLess(sweep.shg_residual_norm.max(), 1e-10)

    def test_omega_is_carried_and_matches_the_grid(self):
        sweep = solve_ml_spectral_sweep(casestudy_ml_spectrum(DISPERSIVE),
                                        wavelength_um=[0.8, 1.0], theta_deg=20.0)
        np.testing.assert_allclose(sweep.omega, 2.0 * np.pi / np.array([0.8, 1.0]))

    def test_both_channels_are_available_and_differ(self):
        spectrum = casestudy_ml_spectrum(DISPERSIVE, thickness_um=THICKNESS0)
        common = dict(wavelength_um=[0.7, 0.9, 1.1], theta_deg=20.0, phi_deg=30.0)
        reflected = solve_ml_spectral_sweep(spectrum, channel="reflected", **common)
        transmitted = solve_ml_spectral_sweep(spectrum, channel="transmitted", **common)
        self.assertEqual(reflected.channel, "reflected")
        self.assertEqual(transmitted.channel, "transmitted")
        self.assertFalse(np.allclose(reflected.intensity, transmitted.intensity))

    def test_the_geometry_is_reported_as_used(self):
        sweep = solve_ml_spectral_sweep(casestudy_ml_spectrum(DISPERSIVE),
                                        wavelength_um=[0.8], theta_deg=33.0, phi_deg=12.0,
                                        psi_deg=44.0, ellipticity_deg=5.0)
        self.assertEqual((sweep.theta_deg, sweep.phi_deg, sweep.psi_deg, sweep.ellipticity_deg),
                         (33.0, 12.0, 44.0, 5.0))

    def test_the_multiple_reflection_assumptions_are_distinguishable(self):
        spectrum = casestudy_ml_spectrum(DISPERSIVE, thickness_um=THICKNESS0)
        common = dict(wavelength_um=[0.7, 0.9, 1.1], theta_deg=20.0, phi_deg=30.0)
        curves = [solve_ml_spectral_sweep(spectrum, mrassumption=m, **common).intensity
                  for m in (0, 1, 2)]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(np.allclose(curves[a], curves[b]),
                             f"assumptions {a} and {b} gave the same spectrum")

    def test_a_one_point_grid_works(self):
        sweep = solve_ml_spectral_sweep(casestudy_ml_spectrum(DISPERSIVE),
                                        wavelength_um=wavelength_grid(0.8, 0.8, 0.1),
                                        theta_deg=20.0)
        self.assertEqual(sweep.intensity.size, 1)
        self.assertIsNone(sweep.support.varies)

    def test_copy_payload_is_wavelength_value_pairs(self):
        sweep = solve_ml_spectral_sweep(casestudy_ml_spectrum(DISPERSIVE),
                                        wavelength_um=[0.8, 1.0], theta_deg=20.0)
        block = sweep.list_spectrum[0]
        self.assertEqual(block.shape, (2, 2))
        np.testing.assert_allclose(block[:, 0], [0.8, 1.0])


class TheGuards(unittest.TestCase):

    def test_a_frozen_permittivity_warns_even_though_the_curve_moves(self):
        """THE case this module exists for. On a multilayer the propagation phase alone makes the
        curve move, so a frozen permittivity yields something that LOOKS like a spectrum."""
        base = _base_system()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sweep = solve_ml_spectral_sweep(lambda lam: replace(base, wavelength_um=float(lam)),
                                            wavelength_um=[0.7, 0.9, 1.1], theta_deg=20.0)
        messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(messages, "a frozen-permittivity multilayer sweep must warn")
        self.assertIn("does not move with wavelength", messages[0])
        self.assertIn("thickness sweep", messages[0])
        self.assertFalse(np.allclose(sweep.intensity, sweep.intensity[0]),
                         "the curve moving is precisely why the warning is needed")

    def test_a_constant_material_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve_ml_spectral_sweep(casestudy_ml_spectrum(CONSTANT),
                                    wavelength_um=[0.7, 0.9, 1.1], theta_deg=20.0)
        messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(messages)
        self.assertIn(CONSTANT, messages[0])

    def test_the_pole_region_raises(self):
        with self.assertRaises(ValueError) as ctx:
            solve_ml_spectral_sweep(casestudy_ml_spectrum(POLED),
                                    lambda_min_um=0.42, lambda_max_um=1.0, lambda_step_um=0.1,
                                    theta_deg=20.0)
        self.assertIn(POLED, str(ctx.exception))

    def test_bad_arguments_raise(self):
        spectrum = casestudy_ml_spectrum(DISPERSIVE)
        with self.assertRaises(ValueError):
            solve_ml_spectral_sweep(spectrum, wavelength_um=[0.8], channel="sideways")
        with self.assertRaises(ValueError):
            solve_ml_spectral_sweep(spectrum, wavelength_um=[0.8], mrassumption=7)
        with self.assertRaises(ValueError):
            solve_ml_spectral_sweep(spectrum, wavelength_um=[])


class TheMixedStack(unittest.TestCase):
    """One layer disperses, another stands still -- the case the flatness check cannot see.

    The flatness check asks whether the STACK's permittivity moves, and one dispersive layer is
    enough to say yes. A frozen metal coating inside it then went unmentioned, and the curve read as
    the whole stack's spectrum. Found by the docs review on the shipped Quartz + Au preset."""

    DISPERSIVE_FILM = "LiNbO3 (dispersive) 0.40-5.00 um"
    FROZEN_COATING = "Au coating (800 nm)"

    def _spectrum(self, film, coating):
        from shaarp.layer_stack import build_system_from_stack, default_layer_spec

        stack = [default_layer_spec("air", 0.0), default_layer_spec(film, 5.0),
                 default_layer_spec(coating, 0.1), default_layer_spec("air", 0.0)]

        def _at(lam):
            return build_system_from_stack(stack, wavelength_um=float(lam), theta_deg=30.0)

        _at.material_names = (film, coating)
        return _at

    def _warnings(self, spectrum):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve_ml_spectral_sweep(spectrum, wavelength_um=[0.9, 1.1, 1.3])
        return [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]

    def test_a_frozen_layer_inside_a_dispersive_stack_is_named(self):
        messages = self._warnings(self._spectrum(self.DISPERSIVE_FILM, self.FROZEN_COATING))
        partial = [m for m in messages if "part of this stack disperses" in m]
        self.assertEqual(len(partial), 1, messages)
        self.assertIn("Au", partial[0], "the note has to name the layer that is standing still")
        # something DOES disperse, so the whole-stack flatness warning must not also fire
        self.assertFalse([m for m in messages if "does not move with wavelength" in m], messages)

    def test_a_stack_where_every_layer_disperses_says_nothing_of_the_kind(self):
        messages = self._warnings(self._spectrum(self.DISPERSIVE_FILM, self.DISPERSIVE_FILM))
        self.assertFalse([m for m in messages if "part of this stack" in m], messages)


def _runtime_warnings(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(*args, **kwargs)
    return [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


class TheFringes(unittest.TestCase):
    """A thick layer puts interference fringes far closer together than the default step.

    The estimate is lambda^2 / (4 n h), n the larger index at the two harmonics. Measured by FFT of
    fine sweeps: quartz 121.2 um 0.0009 um (estimate 0.00085), LiNbO3 1 um 0.153 (0.131), GaAs
    1 um 0.068 (0.065), TaAs 1 um no fringes at all (opaque at both harmonics). A first estimate
    from the fundamental alone, lambda^2 / (2 n h), was twice too coarse and missed GaAs."""

    def _thick(self, h):
        return casestudy_ml_spectrum(DISPERSIVE, thickness_um=h)

    def test_a_thick_layer_at_the_default_step_warns(self):
        messages = _runtime_warnings(solve_ml_spectral_sweep, self._thick(121.2),
                                     lambda_min_um=0.8, lambda_max_um=1.2, lambda_step_um=0.025,
                                     theta_deg=20.0)
        alias = [m for m in messages if "undersampled" in m]
        self.assertEqual(len(alias), 1, messages)
        self.assertIn("121.2", alias[0])

    def test_a_fine_enough_step_does_not(self):
        messages = _runtime_warnings(solve_ml_spectral_sweep, self._thick(121.2),
                                     lambda_min_um=0.8, lambda_max_um=0.801,
                                     lambda_step_um=0.0002, theta_deg=20.0)
        self.assertFalse([m for m in messages if "undersampled" in m], messages)

    def test_a_thin_film_at_the_default_step_does_not(self):
        messages = _runtime_warnings(solve_ml_spectral_sweep, self._thick(1.0),
                                     lambda_min_um=0.8, lambda_max_um=1.2, lambda_step_um=0.025,
                                     theta_deg=20.0)
        self.assertFalse([m for m in messages if "undersampled" in m], messages)

    def test_a_high_index_film_is_caught(self):
        """1 um of GaAs fringes every ~0.068 um near 1 um (31% deep); a 0.04 um step draws about
        one sample per fringe. The fundamental-only estimate (0.15 um) let this pass silently."""
        spectrum = casestudy_ml_spectrum("GaAs (dispersive)", thickness_um=1.0)
        messages = _runtime_warnings(solve_ml_spectral_sweep, spectrum, lambda_min_um=0.9,
                                     lambda_max_um=1.3, lambda_step_um=0.04, theta_deg=20.0,
                                     phi_deg=30.0)
        self.assertTrue([m for m in messages if "undersampled" in m], messages)

    def test_a_fresnel_map_is_held_to_the_fundamentals_own_fringes(self):
        """Fresnel coefficients are linear optics at the fundamental, whose Fabry-Perot period is
        lambda^2 / (2 n_w h) -- twice the second-harmonic estimate. Measured at 40 deg: 1 um quartz
        film maxima 0.25-0.33 um apart, the 121.2 um plate 0.0019 um. The second-harmonic estimate
        warned a Fresnel map about fringes twice as dense as any it has. A Maker map, whose rows do
        fringe that densely (0.139 um on this film), keeps the warning at the same step."""
        from shaarp.spectral import solve_spectral_angle_map

        grid = dict(lambda_min_um=0.6, lambda_max_um=1.4, theta_deg=[40.0])
        film = self._thick(1.0)
        fresnel = _runtime_warnings(solve_spectral_angle_map, film, kind="fresnel",
                                    lambda_step_um=0.04, **grid)
        self.assertFalse([m for m in fresnel if "undersampled" in m], fresnel)
        maker = _runtime_warnings(solve_spectral_angle_map, film, kind="maker",
                                  lambda_step_um=0.04, **grid)
        self.assertTrue([m for m in maker if "undersampled" in m], maker)
        coarse = _runtime_warnings(solve_spectral_angle_map, film, kind="fresnel",
                                   lambda_step_um=0.1, **grid)
        self.assertTrue([m for m in coarse if "undersampled" in m],
                        "a Fresnel map must still warn when it really is undersampled")

    def test_an_opaque_film_does_not_warn(self):
        """1 um of TaAs absorbs both harmonics on a round trip, and a fine sweep shows no fringes at
        all -- the old estimate warned about fringes that are not there."""
        spectrum = casestudy_ml_spectrum("TaAs (dispersive)", thickness_um=1.0)
        messages = _runtime_warnings(solve_ml_spectral_sweep, spectrum, lambda_min_um=0.6,
                                     lambda_max_um=1.0, lambda_step_um=0.04, theta_deg=20.0)
        self.assertFalse([m for m in messages if "undersampled" in m], messages)

    def test_the_apps_suggested_step_fits_the_step_field(self):
        """The field holds four decimals. Suggesting 0.00057 rounded UP to 0.0006 in the field --
        still too coarse -- so in the app the suggestion is rounded down to what the field holds."""
        import re

        from shaarp.spectral import speaking_to

        with speaking_to("app"):
            messages = _runtime_warnings(solve_ml_spectral_sweep, self._thick(20.0),
                                         lambda_min_um=0.8, lambda_max_um=1.2,
                                         lambda_step_um=0.025, theta_deg=20.0)
        alias = [m for m in messages if "undersampled" in m]
        self.assertEqual(len(alias), 1, messages)
        step = re.search(r"Use a step of ([0-9.]+) µm", alias[0])
        self.assertIsNotNone(step, alias[0])
        self.assertEqual(round(float(step.group(1)), 4), float(step.group(1)))


class TheFresnelMapIsLinearOptics(unittest.TestCase):
    """A Fresnel map reads eps(w) only, so the eps(2w) rules must not reach it: the ultraviolet pole
    at lambda/2 refused a KTP Fresnel sweep its linear optics answer, and a table's clamp was
    reported from twice its low end."""

    def test_the_pole_refuses_a_maker_map_but_not_a_fresnel_map(self):
        from shaarp.spectral import solve_spectral_angle_map

        spectrum = casestudy_ml_spectrum(POLED, thickness_um=1.0)
        with self.assertRaises(ValueError):
            solve_spectral_angle_map(spectrum, wavelength_um=[0.45, 1.0], theta_deg=[10.0],
                                     kind="maker")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = solve_spectral_angle_map(spectrum, wavelength_um=[0.45, 1.0],
                                              theta_deg=[10.0], kind="fresnel")
        self.assertEqual(result.shape, (2, 1))

    def test_a_fresnel_map_reports_a_table_from_its_own_low_end(self):
        from shaarp.spectral import solve_spectral_angle_map

        spectrum = casestudy_ml_spectrum("LiB3O5 / LBO (dispersive)", thickness_um=1.0)
        messages = _runtime_warnings(solve_spectral_angle_map, spectrum, wavelength_um=[0.4, 0.6],
                                     theta_deg=[10.0], kind="fresnel")
        # 0.4 um is inside LBO's 0.29-1.06 um data: nothing to report for linear optics
        self.assertFalse([m for m in messages if "index data covers" in m], messages)
        maker = _runtime_warnings(solve_spectral_angle_map, spectrum, wavelength_um=[0.4, 0.6],
                                  theta_deg=[10.0], kind="maker")
        self.assertTrue([m for m in maker if "half the wavelength" in m], maker)

    def test_the_ml_factory_takes_the_short_name_too(self):
        short = casestudy_ml_spectrum("LiB3O5 / LBO (dispersive)")(0.8)
        full = casestudy_ml_spectrum("LiB3O5 / LBO (dispersive) 0.29-1.06 um")(0.8)
        np.testing.assert_allclose(short.layers[1].material.epsilon_omega,
                                   full.layers[1].material.epsilon_omega)
        with self.assertRaises(ValueError):
            casestudy_ml_spectrum("no such crystal")


if __name__ == "__main__":
    unittest.main()
