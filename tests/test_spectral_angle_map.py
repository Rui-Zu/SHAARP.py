"""Maker fringes and Fresnel coefficients over a wavelength-by-incidence-angle grid.

WHY THIS EXISTS. Two things about a two-dimensional result are easy to get wrong and invisible
once wrong.

The first is the LAYOUT. The ordinary CSV exporter requires every array in a result to have the
same flattened length, which a 2-D block beside two 1-D axes does not satisfy. Storing the map in
long format -- both axes expanded to the full length -- keeps that exporter working unchanged, but
only if the two axes are expanded the RIGHT way round. A repeat/tile swap produces a map that is
the correct size, plots without complaint, and is transposed. So the axes are checked against the
per-point values rather than against each other.

The second is the DEGENERATE case. A user is entitled to set the start and end wavelength equal,
and then the map is an ordinary incidence-angle scan; the same with one angle makes it an ordinary
spectrum. Both have to survive and be recognisable, because the renderer picks its plot from the
shape.

The map deliberately reuses the existing angle sweep underneath each wavelength rather than
reimplementing it, so this file also checks that a one-wavelength map reproduces that sweep
exactly -- if it does, the map cannot have drifted from the validated angle path.
"""
from __future__ import annotations

import unittest
import warnings

import numpy as np

from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep
from shaarp.spectral import (
    MAP_CHANNELS,
    casestudy_ml_spectrum,
    inclusive_grid,
    solve_spectral_angle_map,
)

DISPERSIVE = "Quartz z-cut (800 nm)"
POLED = "KTP x-cut"
THICKNESS = 5.0


def _spectrum():
    return casestudy_ml_spectrum(DISPERSIVE, thickness_um=THICKNESS)


class TheLongFormatLayout(unittest.TestCase):

    def setUp(self):
        self.map = solve_spectral_angle_map(
            _spectrum(), wavelength_um=[0.7, 0.9, 1.1, 1.3], theta_deg=[0.0, 10.0, 20.0],
            kind="maker", phi_deg=30.0)

    def test_every_array_has_the_same_length_so_csv_export_stays_valid(self):
        expected = 4 * 3
        self.assertEqual(self.map.shape, (4, 3))
        self.assertEqual(self.map.wavelength_um.size, expected)
        self.assertEqual(self.map.theta_deg.size, expected)
        for name, values in self.map.channels.items():
            self.assertEqual(np.asarray(values).size, expected, name)

    def test_the_axes_are_expanded_the_right_way_round(self):
        """A repeat/tile swap gives a correctly sized, silently transposed map."""
        # wavelength is the SLOW axis: it repeats in blocks
        np.testing.assert_allclose(self.map.wavelength_um[:3], [0.7, 0.7, 0.7])
        # theta is the FAST axis: it tiles
        np.testing.assert_allclose(self.map.theta_deg[:3], [0.0, 10.0, 20.0])
        np.testing.assert_allclose(self.map.wavelength_axis, [0.7, 0.9, 1.1, 1.3])
        np.testing.assert_allclose(self.map.theta_axis, [0.0, 10.0, 20.0])

    def test_grid_reshapes_consistently_with_the_long_arrays(self):
        grid = self.map.grid("parallel_intensity")
        self.assertEqual(grid.shape, (4, 3))
        flat = np.asarray(self.map.channels["parallel_intensity"])
        for row in range(4):
            for col in range(3):
                self.assertEqual(grid[row, col], flat[row * 3 + col])

    def test_the_map_varies_along_both_axes(self):
        """Guards every layout assertion above: a constant map cannot expose a transpose."""
        grid = self.map.grid("parallel_intensity")
        self.assertFalse(np.allclose(grid[0, :], grid[0, 0]), "no variation along angle")
        self.assertFalse(np.allclose(grid[:, 0], grid[0, 0]), "no variation along wavelength")


class ItReusesTheValidatedAngleSweep(unittest.TestCase):

    def test_one_wavelength_reproduces_the_plain_angle_sweep_exactly(self):
        theta = inclusive_grid(0.0, 20.0, 5.0, what="Angle")
        spectrum = _spectrum()
        mapped = solve_spectral_angle_map(spectrum, wavelength_um=[0.8], theta_deg=theta,
                                          kind="maker")
        direct = solve_multilayer_maker_fringes_sweep(spectrum(0.8), theta_deg=theta, mu=1.0,
                                                      eps0=1.0, mrassumption=0,
                                                      inhomogeneous_source_policy="forward_only")
        np.testing.assert_array_equal(np.asarray(mapped.channels["parallel_intensity"]),
                                      np.asarray(direct.parallel_intensity, dtype=float))
        np.testing.assert_array_equal(np.asarray(mapped.channels["perpendicular_intensity"]),
                                      np.asarray(direct.perpendicular_intensity, dtype=float))


class TheDegenerateAxes(unittest.TestCase):

    def test_a_single_wavelength_is_an_angle_scan(self):
        result = solve_spectral_angle_map(_spectrum(), wavelength_um=[0.8],
                                          theta_deg=[0.0, 10.0, 20.0])
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(result.is_angle_scan_only)
        self.assertFalse(result.is_spectrum_only)

    def test_a_single_angle_is_a_spectrum(self):
        result = solve_spectral_angle_map(_spectrum(), wavelength_um=[0.7, 0.9, 1.1],
                                          theta_deg=[20.0])
        self.assertEqual(result.shape, (3, 1))
        self.assertTrue(result.is_spectrum_only)
        self.assertFalse(result.is_angle_scan_only)

    def test_equal_start_and_end_wavelength_is_one_point_not_an_error(self):
        result = solve_spectral_angle_map(_spectrum(), lambda_min_um=0.8, lambda_max_um=0.8,
                                          lambda_step_um=0.05, theta_deg=[10.0, 20.0])
        self.assertEqual(result.shape, (1, 2))

    def test_both_axes_collapsed_is_a_single_point(self):
        result = solve_spectral_angle_map(_spectrum(), wavelength_um=[0.8], theta_deg=[20.0])
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(np.asarray(result.channels["parallel_intensity"]).size, 1)


class TheFresnelMap(unittest.TestCase):

    def test_it_carries_the_four_fresnel_channels(self):
        result = solve_spectral_angle_map(_spectrum(), wavelength_um=[0.7, 1.1],
                                          theta_deg=[0.0, 20.0, 40.0], kind="fresnel")
        self.assertEqual(tuple(result.channels), MAP_CHANNELS["fresnel"])
        self.assertEqual(result.shape, (2, 3))
        for name in MAP_CHANNELS["fresnel"]:
            values = np.asarray(result.channels[name])
            self.assertEqual(values.size, 6, name)
            self.assertTrue(np.all(values >= -1e-12), f"{name} has a negative power ratio")
            self.assertTrue(np.all(values <= 1.0 + 1e-9), f"{name} exceeds unity")


class TheFigureShowsEveryChannel(unittest.TestCase):
    """The single-wavelength figures plot all of their channels; the map must too.

    The first draft drew only the first channel, which silently dropped three of the four Fresnel
    curves and half of Maker. A map with the right axes and one curve looks entirely plausible."""

    @staticmethod
    def _figure(**kwargs):
        from shaarp.shaarp_gui import build_spectral_map_figure

        return build_spectral_map_figure(solve_spectral_angle_map(_spectrum(), **kwargs))

    def test_a_fresnel_map_draws_all_four_panels(self):
        figure = self._figure(wavelength_um=[0.7, 0.9, 1.1], theta_deg=[0.0, 20.0, 40.0],
                              kind="fresnel")
        titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
        self.assertEqual(len(titles), 4, f"expected four panels, got {titles}")
        for expected in ("$R_p$", "$R_s$", "$T_p$", "$T_s$"):
            self.assertIn(expected, titles)

    def test_a_collapsed_fresnel_map_draws_all_four_curves(self):
        figure = self._figure(wavelength_um=[0.8], theta_deg=[0.0, 20.0, 40.0, 60.0],
                              kind="fresnel")
        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(len(figure.axes[0].lines), 4)

    def test_a_maker_map_draws_both_channels(self):
        figure = self._figure(wavelength_um=[0.7, 0.9], theta_deg=[0.0, 20.0], kind="maker")
        titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
        self.assertEqual(len(titles), 2, f"expected two panels, got {titles}")

    def test_the_fresnel_figure_does_not_claim_an_shg_assumption(self):
        """Fresnel coefficients are linear optics -- no SHG tensor takes part, so the constant-d
        caveat would describe an assumption this figure does not make."""
        figure = self._figure(wavelength_um=[0.7, 0.9], theta_deg=[0.0, 20.0], kind="fresnel")
        text = (figure._suptitle.get_text() if figure._suptitle else "")
        self.assertNotIn("SHG tensor", text)
        self.assertIn("dielectric", text)

    def test_the_maker_figure_still_carries_the_constant_d_caveat(self):
        figure = self._figure(wavelength_um=[0.7, 0.9], theta_deg=[0.0, 20.0], kind="maker")
        self.assertIn("SHG tensor held constant", figure._suptitle.get_text())

    def test_one_channel_can_still_be_isolated(self):
        from shaarp.shaarp_gui import build_spectral_map_figure

        result = solve_spectral_angle_map(_spectrum(), wavelength_um=[0.7, 0.9],
                                          theta_deg=[0.0, 20.0], kind="fresnel")
        figure = build_spectral_map_figure(result, channel="ts")
        titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
        self.assertEqual(titles, ["$T_s$"])

    def test_a_one_point_spectrum_names_its_wavelength_instead_of_the_sweep_caveat(self):
        """One wavelength sweeps nothing, so "only the dielectric tensors disperse" under a single
        dot described nothing on the plot -- the reason the caveat left one-wavelength maps."""
        from shaarp.shaarp_gui import build_spectrum_figure
        from shaarp.spectral import solve_ml_spectral_sweep

        one = solve_ml_spectral_sweep(_spectrum(), lambda_min_um=0.8, lambda_max_um=0.8,
                                      lambda_step_um=0.05, theta_deg=20.0, phi_deg=30.0)
        title = build_spectrum_figure(one).axes[0].get_title()
        self.assertNotIn("SHG tensor held constant", title)
        self.assertIn("0.8", title)
        many = solve_ml_spectral_sweep(_spectrum(), lambda_min_um=0.8, lambda_max_um=0.9,
                                       lambda_step_um=0.05, theta_deg=20.0, phi_deg=30.0)
        self.assertIn("SHG tensor held constant", build_spectrum_figure(many).axes[0].get_title())

    def test_a_single_wavelength_names_its_wavelength_once(self):
        """The title already says "at lambda = 0.8 um"; a caption repeating it printed it twice."""
        figure = self._figure(wavelength_um=[0.8], theta_deg=[0.0, 20.0, 40.0], kind="maker")
        title = figure.axes[0].get_title()
        self.assertEqual(title.count("0.8"), 1, title)

    def test_a_channel_that_vanishes_reads_as_zero(self):
        """p-in on z-cut quartz leaves the perpendicular channel at rounding noise (~1e-31). On
        its own colour bar that noise was drawn as full-contrast structure; on the parallel
        channel's scale it reads as the zero it is, and the panel title says so."""
        result = solve_spectral_angle_map(_spectrum(), wavelength_um=[0.7, 0.9],
                                          theta_deg=[0.0, 20.0], kind="maker", phi_deg=0.0)
        self.assertLess(float(np.max(result.grid("perpendicular_intensity"))), 1e-20)
        from shaarp.shaarp_gui import build_spectral_map_figure

        figure = build_spectral_map_figure(result)
        panels = {ax.get_title(): ax for ax in figure.axes if ax.get_title()}
        zero = next(t for t in panels if "perp" in t)
        self.assertIn("approx", zero)
        low, high = panels[zero].collections[0].get_clim()
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 1.0, "the vanishing channel must share the other one's scale")
        live = next(t for t in panels if "parallel" in t)
        self.assertNotIn("approx", live)

        # a weak channel that is NOT zero keeps its own scale and no mark
        figure = build_spectral_map_figure(solve_spectral_angle_map(
            _spectrum(), wavelength_um=[0.7, 0.9], theta_deg=[0.0, 20.0], kind="maker",
            phi_deg=30.0))
        self.assertFalse([ax.get_title() for ax in figure.axes if "approx" in ax.get_title()])


class TheFresnelPhysics(unittest.TestCase):

    def test_a_lossless_stack_conserves_energy_at_every_wavelength(self):
        """R + T = 1 per polarization -- the check that says the linear stage is right, and it has
        to hold across the whole wavelength axis, not just at the native one."""
        result = solve_spectral_angle_map(_spectrum(), wavelength_um=[0.7, 1.0, 1.3],
                                          theta_deg=[0.0, 20.0, 40.0, 60.0, 80.0], kind="fresnel")
        for reflect, transmit in (("rp", "tp"), ("rs", "ts")):
            total = result.grid(reflect) + result.grid(transmit)
            np.testing.assert_allclose(total, 1.0, atol=1e-9,
                                       err_msg=f"{reflect} + {transmit} != 1")


class TheGuards(unittest.TestCase):

    def test_an_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            solve_spectral_angle_map(_spectrum(), wavelength_um=[0.8], theta_deg=[0.0],
                                     kind="polarimetry")

    def test_empty_axes_raise(self):
        with self.assertRaises(ValueError):
            solve_spectral_angle_map(_spectrum(), wavelength_um=[], theta_deg=[0.0])
        with self.assertRaises(ValueError):
            solve_spectral_angle_map(_spectrum(), wavelength_um=[0.8], theta_deg=[])

    def test_the_pole_region_raises_on_a_map_too(self):
        with self.assertRaises(ValueError):
            solve_spectral_angle_map(casestudy_ml_spectrum(POLED, thickness_um=THICKNESS),
                                     lambda_min_um=0.42, lambda_max_um=1.0, lambda_step_um=0.2,
                                     theta_deg=[20.0])

    def test_an_angle_grid_may_start_at_zero(self):
        """The wavelength grid forbids a non-positive start; the angle grid must not inherit it."""
        grid = inclusive_grid(0.0, 3.0, 1.0, what="Angle")
        np.testing.assert_allclose(grid, [0.0, 1.0, 2.0, 3.0])

    def test_the_assumptions_travel_with_the_map(self):
        result = solve_spectral_angle_map(_spectrum(), wavelength_um=[0.8, 1.0],
                                          theta_deg=[20.0])
        self.assertIn("wavelength-independent", result.assumptions["d_voigt"])

    def test_a_fresnel_map_states_no_shg_assumption(self):
        """Fresnel coefficients are linear optics at the fundamental; neither d nor eps(2w) enters.
        The figure already left the constant-d caption off -- the exported assumptions said it
        anyway, so a saved Fresnel map claimed an assumption its result never made."""
        result = solve_spectral_angle_map(_spectrum(), wavelength_um=[0.8, 1.0],
                                          theta_deg=[20.0], kind="fresnel")
        self.assertNotIn("d_voigt", result.assumptions)
        self.assertNotIn("epsilon_2omega", result.assumptions)
        self.assertIn("linear", result.assumptions["optics"])


if __name__ == "__main__":
    unittest.main()
