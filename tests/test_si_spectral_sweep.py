"""Single-interface SHG across a wavelength grid.

WHY THIS EXISTS. A wavelength sweep is the first feature whose correctness cannot be checked
against a reference point, because no live-Mathematica export exists at any wavelength other than
each material's own native one -- that gap is named in the residual-risk register. So the sweep is
fenced by things that must hold at EVERY wavelength rather than at one:

  * The scale identity. A single interface contains no length, so with mu = eps0 = 1 the driven
    wave equation is homogeneous of degree 2 in omega and the frequency cancels exactly. The
    wavelength therefore reaches the answer through the permittivity and through nothing else.
    This is what licenses sweeping the MATERIAL and leaving the solver's omega at its dimensionless
    1 and 2, and it is the claim that breaks first if anyone ever wires a physical omega into the
    single-interface path -- where, unlike the multilayer path, there is no mu*eps0 == 1 guard.
  * Cross-path agreement. The closed form and the numeric solver are independent implementations.
    They were known to agree at ONE wavelength; running the comparison over the whole grid turns
    that into a claim about the wavelength axis, for free.
  * Endpoint anchoring. A grid containing a material's native wavelength must reproduce the
    existing single-wavelength result there exactly. One assertion, and it catches any off-by-one
    in the grid or the per-point material rebuild.

The channel-order check is deliberate. The generic solver returns the reflected s amplitude in
coefficients[0] and p in [1]; the compat workflow's order is the OPPOSITE. Swapping them is silent
-- both channels are plausible positive intensities -- so the cross-path comparison is done on s
and p SEPARATELY, and on this material they differ by about 4.5x, which is what gives that
comparison the power to see a swap.
"""
from __future__ import annotations

import math
import unittest
import warnings

import numpy as np

from shaarp.casestudy_materials import build_casestudy_material
from shaarp.shaarp_gui import si_polarimetry_curve
from shaarp.shg import solve_single_interface_shg
from shaarp.spectral import (
    SPECTRAL_ASSUMPTIONS,
    _lab_tensors,
    casestudy_spectrum,
    constant_spectrum,
    solve_si_spectral_sweep,
    wavelength_grid,
)

DISPERSIVE = "Quartz z-cut (800 nm)"    # identity orientation, point group 32, genuinely dispersive
CONSTANT = "GaAs (111) (800 nm)"        # full grid, but constant by source
POLED = "KTP x-cut"                     # physical only from 0.54 um


class TheScaleIdentity(unittest.TestCase):
    """A single interface has no length in it, so the frequency must cancel."""

    def test_the_answer_is_invariant_under_the_omega_scale(self):
        material = build_casestudy_material("LiNbO3 (11-20) MTI X-cut")
        eps_w, eps_2w, d_lab = _lab_tensors(material)

        def reflected(omega):
            r = solve_single_interface_shg(
                eps_w, eps_2w, d_lab, incident_index_omega=1.0, incident_index_2omega=1.0,
                incident_theta_rad=math.radians(45.0), incident_jones=(0.4, 0.9),
                omega=omega, mu=1.0, eps0=1.0)
            return np.asarray(r.coefficients, dtype=complex)

        reference = reflected(1.0)
        scale = float(np.abs(reference).max())
        for omega in (2 * math.pi / 0.8, 2 * math.pi / 1.55, 17.0):
            deviation = float(np.abs(reflected(omega) - reference).max()) / scale
            self.assertLess(deviation, 1e-12,
                            f"omega = {omega:g} changed the answer by {deviation:.2e}; the "
                            "single-interface problem is supposed to be scale-free")

    def test_a_material_that_does_not_disperse_gives_a_flat_spectrum(self):
        """The same identity from the other side: hold eps fixed and the wavelength does nothing."""
        material = build_casestudy_material(DISPERSIVE, wavelength_um=0.8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sweep = solve_si_spectral_sweep(constant_spectrum(material),
                                            wavelength_um=[0.6, 0.9, 1.4, 1.9],
                                            allow_constant_dispersion=True)
        self.assertTrue(np.allclose(sweep.intensity_p, sweep.intensity_p[0], rtol=0, atol=0),
                        "frozen permittivity must give a bit-identical flat spectrum")


class AgreementAcrossTheGrid(unittest.TestCase):

    def test_the_closed_form_agrees_with_the_numeric_sweep_at_every_wavelength(self):
        """Two independent implementations, compared per channel over the whole grid."""
        phi_deg, theta_deg, n_phi = 30.0, 45.0, 181
        grid = np.array([0.60, 0.80, 1.00, 1.30, 1.60])
        sweep = solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE), wavelength_um=grid,
                                        theta_deg=theta_deg, phi_deg=phi_deg)
        index = int(round(phi_deg / 360.0 * (n_phi - 1)))
        self.assertAlmostEqual(np.linspace(0.0, 360.0, n_phi)[index], phi_deg, places=9)

        for i, lam in enumerate(grid):
            material = build_casestudy_material(DISPERSIVE, wavelength_um=float(lam))
            eps_w, eps_2w, d_lab = _lab_tensors(material)
            curve = si_polarimetry_curve(
                material.structure.point_group, theta_deg=theta_deg, n_phi=n_phi,
                eps_omega_principal=tuple(np.diag(eps_w)),
                eps_2omega_principal=tuple(np.diag(eps_2w)), d_voigt_lab_full=d_lab)
            for channel, ours in (("intensity_s", sweep.intensity_s[i]),
                                  ("intensity_p", sweep.intensity_p[i])):
                theirs = float(curve[channel][index])
                self.assertAlmostEqual(ours / theirs, 1.0, delta=1e-9,
                                       msg=f"{channel} disagrees at {lam:g} um")

    def test_the_analyzer_follows_the_apps_own_convention(self):
        """0 deg detects p and 90 deg detects s, as the polarimetry panel states. Inverting this
        is silent -- both are positive intensities of the right order."""
        grid = [0.7, 1.0, 1.3]
        for psi, channel in ((0.0, "intensity_p"), (90.0, "intensity_s")):
            sweep = solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE), wavelength_um=grid,
                                            theta_deg=45.0, phi_deg=30.0, psi_deg=psi)
            np.testing.assert_allclose(sweep.intensity_analyzed, getattr(sweep, channel),
                                       rtol=1e-12,
                                       err_msg=f"psi={psi} should detect {channel}")

    def test_the_two_channels_are_far_enough_apart_to_expose_a_swap(self):
        """Guards the test above: if s and p were nearly equal here, it could not see a swap."""
        sweep = solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE),
                                        wavelength_um=[0.8], theta_deg=45.0, phi_deg=30.0)
        ratio = float(sweep.intensity_s[0] / sweep.intensity_p[0])
        self.assertGreater(ratio, 2.0, "the channels must differ enough for a swap to be visible")

    def test_the_native_wavelength_reproduces_the_single_wavelength_result_exactly(self):
        """Endpoint anchoring: on the grid point a validated result already exists for, the sweep
        must BE that result, not merely agree with it."""
        material = build_casestudy_material(DISPERSIVE)          # native wavelength
        eps_w, eps_2w, d_lab = _lab_tensors(material)
        direct = solve_single_interface_shg(
            eps_w, eps_2w, d_lab, incident_index_omega=1.0, incident_index_2omega=1.0,
            incident_theta_rad=math.radians(45.0),
            incident_jones=(math.sin(math.radians(30.0)), math.cos(math.radians(30.0))),
            omega=1.0, mu=1.0, eps0=1.0)
        sweep = solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE), wavelength_um=[0.8],
                                        theta_deg=45.0, phi_deg=30.0)
        self.assertEqual(complex(sweep.reflected_s[0]), complex(np.asarray(direct.coefficients)[0]))
        self.assertEqual(complex(sweep.reflected_p[0]), complex(np.asarray(direct.coefficients)[1]))

    def test_a_dispersive_material_actually_produces_a_spectrum(self):
        sweep = solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE), phi_deg=30.0,
                                        lambda_min_um=0.6, lambda_max_um=1.6, lambda_step_um=0.1)
        self.assertGreater(sweep.intensity_p.min(), 0.0)
        self.assertFalse(np.allclose(sweep.intensity_p, sweep.intensity_p[0]))
        self.assertTrue(sweep.support.varies)
        self.assertLess(sweep.boundary_residual_norm.max(), 1e-10)

    def test_a_one_point_grid_does_not_claim_the_spectrum_is_flat(self):
        """Flatness is not a question a single wavelength can answer, and warning there would fire
        on every collapsed range -- which is a case the app has to support."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sweep = solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE), wavelength_um=[0.8])
        self.assertIsNone(sweep.support.varies)
        self.assertFalse([w for w in caught if "does not move with wavelength" in str(w.message)],
                         "a one-point grid must not warn about flatness")


class TheGuards(unittest.TestCase):

    def test_a_constant_material_warns_and_still_computes(self):
        """Rui's call: warn and say how to fix it, rather than refuse."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sweep = solve_si_spectral_sweep(casestudy_spectrum(CONSTANT),
                                            wavelength_um=[0.6, 0.9, 1.2])
        messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(messages, "a flat spectrum must warn")
        self.assertIn("does not move with wavelength", messages[0])
        self.assertIn(CONSTANT, messages[0])
        # the warning has to say how to ADDRESS it, not just that it happened: GaAs has a
        # table-backed variant, so that is what it names
        self.assertIn("GaAs (dispersive)", messages[0])
        self.assertIn("allow_constant_dispersion", messages[0])
        # a material whose crystal has no index table gets the general route instead
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve_si_spectral_sweep(casestudy_spectrum("Au coating (800 nm)"),
                                    wavelength_um=[0.6, 0.9, 1.2])
        general = [str(w.message) for w in caught if "does not move with wavelength" in str(w.message)]
        self.assertEqual(len(general), 1)
        self.assertIn("Dispersive group", general[0])
        self.assertIn("index table", general[0])
        # the sweep has already computed by the time this is read; the flag only silences it.
        # The old text told users to pass it "to compute it anyway", which was never true.
        self.assertNotIn("compute it anyway", messages[0])
        # and it must not claim flatness for every path: on a stack the curve still moves
        self.assertIn("thickness sweep", messages[0])
        self.assertTrue(np.allclose(sweep.intensity_p, sweep.intensity_p[0]))

    def test_allow_constant_dispersion_silences_the_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve_si_spectral_sweep(casestudy_spectrum(CONSTANT), wavelength_um=[0.6, 1.2],
                                    allow_constant_dispersion=True)
        flat = [w for w in caught if "does not move with wavelength" in str(w.message)]
        self.assertFalse(flat, "the opt-in must silence the flatness warning")

    def test_running_past_the_red_end_of_the_grid_clamps_rather_than_refusing(self):
        """A pole and a grid end are different things. The exported models pole at the BLUE end,
        inside the tabulated span; past the RED end the data simply stops and the tensors clamp.
        Treating 'outside the usable window' as 'in the pole region' refused this ordinary sweep."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sweep = solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE),
                                            lambda_min_um=1.8, lambda_max_um=2.6,
                                            lambda_step_um=0.4)
        self.assertEqual(sweep.wavelength_um.size, 3)
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("clamp" in m for m in messages), messages)
        self.assertFalse(any("ultraviolet pole" in m for m in messages), messages)

    def test_a_range_entirely_below_the_grid_clamps_too(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve_si_spectral_sweep(casestudy_spectrum(POLED), lambda_min_um=0.20,
                                    lambda_max_um=0.35, lambda_step_um=0.05)
        self.assertTrue(any("clamp" in str(w.message) for w in caught))

    def test_the_pole_region_raises_rather_than_returning_a_number(self):
        with self.assertRaises(ValueError) as ctx:
            solve_si_spectral_sweep(casestudy_spectrum(POLED),
                                    lambda_min_um=0.42, lambda_max_um=1.0, lambda_step_um=0.1)
        message = str(ctx.exception)
        self.assertIn(POLED, message)
        self.assertIn("0.54", message)

    def test_the_same_material_is_fine_inside_its_usable_window(self):
        """phi = 30 deg on purpose. KTP x-cut's p channel is IDENTICALLY zero at phi = 0 -- a
        symmetry selection rule, not a defect -- and an all-zero channel satisfies a
        does-it-vary assertion vacuously."""
        sweep = solve_si_spectral_sweep(casestudy_spectrum(POLED),
                                        lambda_min_um=0.60, lambda_max_um=1.60,
                                        lambda_step_um=0.10, phi_deg=30.0)
        self.assertEqual(sweep.wavelength_um.size, 11)
        for channel in (sweep.intensity_s, sweep.intensity_p):
            self.assertGreater(channel.min(), 0.0, "channel is zero; the check would be vacuous")
            self.assertFalse(np.allclose(channel, channel[0]))


class TheGrid(unittest.TestCase):

    def test_endpoint_is_included(self):
        np.testing.assert_allclose(wavelength_grid(0.55, 0.70, 0.05), [0.55, 0.60, 0.65, 0.70])

    def test_start_equal_to_stop_is_one_point_not_an_error(self):
        """A sweep collapsed to a single wavelength has to render as the ordinary result."""
        grid = wavelength_grid(0.8, 0.8, 0.1)
        self.assertEqual(grid.tolist(), [0.8])
        sweep = solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE), wavelength_um=grid)
        self.assertEqual(sweep.wavelength_um.size, 1)
        self.assertEqual(sweep.intensity_p.size, 1)

    def test_bad_grids_raise(self):
        for args in ((0.5, 1.0, 0.0), (0.5, 1.0, -0.1), (1.0, 0.5, 0.1), (0.0, 1.0, 0.1)):
            with self.assertRaises(ValueError, msg=f"{args} should have raised"):
                wavelength_grid(*args)

    def test_an_empty_grid_raises(self):
        with self.assertRaises(ValueError):
            solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE), wavelength_um=[])


class TheResultCarriesItsAssumptions(unittest.TestCase):

    def test_the_constant_d_assumption_travels_with_the_data(self):
        sweep = solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE), wavelength_um=[0.8, 1.0])
        self.assertEqual(sweep.assumptions, dict(SPECTRAL_ASSUMPTIONS))
        self.assertIn("wavelength-independent", sweep.assumptions["d_voigt"])

    def test_d_really_is_held_constant_across_the_sweep(self):
        """The assumption is only honest if the code actually does it."""
        first = _lab_tensors(build_casestudy_material(DISPERSIVE, wavelength_um=0.6))[2]
        last = _lab_tensors(build_casestudy_material(DISPERSIVE, wavelength_um=1.6))[2]
        np.testing.assert_array_equal(first, last)

    def test_copy_payloads_are_wavelength_value_pairs(self):
        sweep = solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE), wavelength_um=[0.8, 1.0])
        for block in sweep.list_spectrum:
            self.assertEqual(block.shape, (2, 2))
            np.testing.assert_allclose(block[:, 0], [0.8, 1.0])

    def test_per_point_diagnostics_are_carried(self):
        sweep = solve_si_spectral_sweep(casestudy_spectrum(DISPERSIVE), wavelength_um=[0.8, 1.0])
        for field in ("boundary_residual_norm", "operator_condition", "ill_conditioned"):
            self.assertEqual(getattr(sweep, field).shape, (2,), field)
        self.assertEqual(len(sweep.results), 2)


def _messages(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(*args, **kwargs)
    return [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


class TheNotesSayOnlyWhatIsTrue(unittest.TestCase):
    """Each case here is a note a GUI review found saying something false or contradictory."""

    def test_a_constant_material_gets_the_flat_note_and_no_clamp_note(self):
        """Holding a constant at its end value changes nothing; saying the curve "flattens at that
        end" right after saying it is flat everywhere contradicted the note above it."""
        messages = _messages(solve_si_spectral_sweep, casestudy_spectrum(CONSTANT),
                             wavelength_um=[1.0, 2.5, 3.0])          # past the 2.0 um grid end
        self.assertTrue([m for m in messages if "does not move with wavelength" in m], messages)
        self.assertFalse([m for m in messages if "tabulated grid" in m], messages)

    def test_the_opt_in_leaves_a_constant_material_silent(self):
        """allow_constant_dispersion=True used to silence the flat note but leave a
        "(0.8-0.8 um) grid ... flattens" line behind for a single-wavelength material."""
        messages = _messages(solve_si_spectral_sweep, casestudy_spectrum("KTP (100)"),
                             wavelength_um=[0.9, 1.2, 1.5], allow_constant_dispersion=True)
        self.assertEqual(messages, [])

    def test_the_advice_names_the_dispersive_twin_when_there_is_one(self):
        """KTP (100) has a table-backed twin, so the note names it. Quartz has none, so the note
        must not send anyone to look for "the same crystal" in the Dispersive group."""
        ktp = _messages(solve_si_spectral_sweep, casestudy_spectrum("KTP (100)"),
                        wavelength_um=[0.9, 1.2, 1.5])
        self.assertTrue([m for m in ktp if "KTP (dispersive) 0.43-3.54 um" in m], ktp)
        self.assertFalse([m for m in ktp if "same crystal from the Dispersive group" in m], ktp)

    def test_the_same_crystal_in_another_cut_is_offered_and_its_cut_named(self):
        """GaAs (111) (800 nm) -- the SI palette's GaAs, and the first sweep a new user tries -- was
        told only "a crystal from the Dispersive group" although GaAs (dispersive) exists. It is
        offered now, with the cut it is built on, so nobody swaps orientation without knowing."""
        messages = _messages(solve_si_spectral_sweep, casestudy_spectrum(CONSTANT),
                             wavelength_um=[0.9, 1.2, 1.5])
        flat = [m for m in messages if "does not move with wavelength" in m]
        self.assertEqual(len(flat), 1, messages)
        self.assertIn("GaAs (dispersive) 0.21-12.40 um (cut as GaAs (111) (1064 nm))", flat[0])

    def test_the_app_wording_has_no_python_in_it(self):
        """The desktop app shows these verbatim. Its users cannot pass a keyword argument or
        "give it an index table", and the app writes µm, not um."""
        from shaarp.spectral import speaking_to

        with speaking_to("app"):
            messages = _messages(solve_si_spectral_sweep, casestudy_spectrum("KTP (100)"),
                                 wavelength_um=[0.9, 1.2, 1.5])
        from shaarp.dispersion import dispersive_material_names

        flat = [m for m in messages if "does not move with wavelength" in m]
        self.assertEqual(len(flat), 1, messages)
        self.assertIn("KTP (dispersive) 0.43-3.54 um", flat[0])   # a combo label, left as it is
        prose = flat[0]
        for label in dispersive_material_names():                 # names keep their own "um"
            prose = prose.replace(label, "")
        for python_only in ("allow_constant_dispersion", "index table", "shaarp.", " um"):
            self.assertNotIn(python_only, prose, python_only)
        self.assertIn("µm", prose)

    def test_a_refusal_speaks_the_apps_units_in_the_app(self):
        from shaarp.spectral import speaking_to

        with speaking_to("app"):
            with self.assertRaises(ValueError) as ctx:
                solve_si_spectral_sweep(casestudy_spectrum(POLED), wavelength_um=[0.44, 0.9])
        self.assertIn("ε(2ω)", str(ctx.exception))
        self.assertIn("µm", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            solve_si_spectral_sweep(casestudy_spectrum(POLED), wavelength_um=[0.44, 0.9])
        self.assertIn("eps(2w)", str(ctx.exception), "Python keeps its ASCII wording")

    def test_a_stack_row_that_is_not_a_material_is_never_named_as_one(self):
        """A stack's rows include the half-spaces. "'isotropic n (set below)' is not a Case Study
        material" reached the app's note because every row label was treated as a material."""
        from shaarp.spectral import check_spectral_support

        def spectrum(lam):
            return build_casestudy_material(CONSTANT, wavelength_um=lam)

        spectrum.material_names = ("isotropic n (set below)", "air", CONSTANT, "Custom (fields)")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_spectral_support(spectrum, [0.9, 1.2, 1.5])
        text = " ".join(str(w.message) for w in caught)
        self.assertIn(CONSTANT, text)
        for not_a_material in ("isotropic n", "not a Case Study material", "Custom (fields)"):
            self.assertNotIn(not_a_material, text)


if __name__ == "__main__":
    unittest.main()
