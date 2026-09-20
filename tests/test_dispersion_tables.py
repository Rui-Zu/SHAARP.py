"""Refractive-index tables: the format, the half-wavelength rule, and the announced clamp.

WHY THIS EXISTS. Most case-study materials carry a constant index, so a wavelength sweep over them
is flat. An index table is how a material gets real dispersion, and three things about it are easy
to get wrong in ways that produce a plausible-looking spectrum:

  * THE HALF-WAVELENGTH RULE. The permittivity at the second harmonic is the table read at
    lambda/2, so supporting a sweep over [a, b] needs data from a/2. A table that looks like it
    covers the sweep can still be half a band short, and the shortfall lands entirely in the
    second-harmonic term where nobody is looking.
  * CLAMPING. numpy's interpolation clamps at the ends rather than failing, silently. That is
    exactly how the exported registry came to report physical-looking permittivities outside the
    range its model was valid over. A table still clamps -- stopping a sweep dead at a table edge
    helps nobody -- but it SAYS SO, because a spectrum that goes quietly flat at one end cannot be
    told apart from physics.
  * THE SIGN CONVENTION. eps = (n + i k)**2 has to come out with a POSITIVE imaginary part, which
    is what the rest of the package means by an absorbing medium. Getting it backwards makes an
    absorbing crystal amplify.

The strongest check here is the cross-check: a table built from the package's own quartz Sellmeier
coefficients reproduces the independently exported registry permittivity for the same material.
Two routes that never share code arrive at the same numbers.
"""
from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from shaarp.casestudy_materials import build_casestudy_material
from shaarp.dispersion import IndexTable, load_index_table, table_spectrum
from shaarp.quartz_au_docs_case import quartz_sellmeier_index


def _write(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    return Path(handle.name)


BIAXIAL = """wavelength_um,nx,ny,nz,kx,ky,kz
0.30,1.80,1.85,1.90,0.00,0.00,0.00
0.80,1.70,1.75,1.80,0.01,0.02,0.03
1.60,1.65,1.70,1.75,0.00,0.00,0.00
"""

UNIAXIAL = """wavelength_um,no,ne
0.30,1.60,1.62
0.80,1.54,1.55
1.60,1.53,1.54
"""

ISOTROPIC = """wavelength_um,n
0.30,1.50
0.80,1.45
1.60,1.44
"""


class TheColumnLayouts(unittest.TestCase):

    def test_biaxial_columns_map_to_the_three_principal_axes(self):
        table = load_index_table(_write(BIAXIAL))
        np.testing.assert_allclose(table.n[1], [1.70, 1.75, 1.80])
        np.testing.assert_allclose(table.k[1], [0.01, 0.02, 0.03])

    def test_uniaxial_columns_repeat_the_ordinary_index_on_x_and_y(self):
        table = load_index_table(_write(UNIAXIAL))
        np.testing.assert_allclose(table.n[1], [1.54, 1.54, 1.55])
        np.testing.assert_allclose(table.k, 0.0)

    def test_isotropic_columns_fill_all_three_axes(self):
        table = load_index_table(_write(ISOTROPIC))
        np.testing.assert_allclose(table.n[1], [1.45, 1.45, 1.45])

    def test_extinction_is_optional_and_defaults_to_transparent(self):
        table = load_index_table(_write(UNIAXIAL))
        self.assertTrue(np.all(table.k == 0.0))
        self.assertTrue(np.all(np.imag(table.epsilon(0.8)) == 0.0))

    def test_rows_out_of_order_are_sorted(self):
        table = load_index_table(_write(
            "wavelength_um,n\n1.60,1.44\n0.30,1.50\n0.80,1.45\n"))
        np.testing.assert_allclose(table.wavelength_um, [0.30, 0.80, 1.60])
        np.testing.assert_allclose(table.n[:, 0], [1.50, 1.45, 1.44])

    def test_comments_and_a_byte_order_mark_are_tolerated(self):
        table = load_index_table(_write(
            "﻿# quartz, from somewhere\nwavelength_um,n\n0.3,1.5\n0.8,1.45\n"))
        self.assertEqual(table.wavelength_um.size, 2)


class TheHalfWavelengthRule(unittest.TestCase):

    def setUp(self):
        self.table = load_index_table(_write(UNIAXIAL))      # 0.30 to 1.60 um

    def test_a_sweep_needs_data_from_half_its_lowest_wavelength(self):
        self.assertTrue(self.table.covers(0.60, 1.60), "0.60 needs 0.30, which the table has")
        self.assertFalse(self.table.covers(0.59, 1.60), "0.59 needs 0.295, which it does not")

    def test_the_upper_end_is_the_fundamental_not_its_harmonic(self):
        self.assertTrue(self.table.covers(0.60, 1.60))
        self.assertFalse(self.table.covers(0.60, 1.61))

    def test_reading_outside_the_table_warns_and_holds_the_index_constant(self):
        """Rui's call: warn and carry on, rather than stop the sweep dead. The thing being avoided
        is the SILENT version -- a spectrum that goes quietly flat at one end cannot be told apart
        from physics."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = self.table.principal(0.20)
        messages = [str(w.message) for w in caught]
        self.assertTrue(messages, "reading past the table must say so")
        self.assertIn("held constant", messages[0])
        self.assertIn("half the fundamental", messages[0])
        # held at the blue end, which for this table is the 0.30 um row
        self.assertAlmostEqual(value[0].real, 1.60, places=9)

    def test_inside_the_table_it_stays_quiet(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.table.principal(0.80)
        self.assertEqual([str(w.message) for w in caught], [])


class ThePermittivity(unittest.TestCase):

    def test_eps_is_n_plus_ik_squared(self):
        table = load_index_table(_write(BIAXIAL))
        eps = table.epsilon(0.8)
        expected = np.diag((np.array([1.70, 1.75, 1.80]) + 1j * np.array([0.01, 0.02, 0.03])) ** 2)
        np.testing.assert_allclose(eps, expected, rtol=1e-12)

    def test_an_absorbing_medium_gets_a_positive_imaginary_permittivity(self):
        """The package's own convention -- the registry's absorbing materials all carry Im eps > 0.
        Getting this backwards turns an absorber into a gain medium."""
        table = load_index_table(_write(BIAXIAL))
        imaginary = np.imag(np.diag(table.epsilon(0.8)))
        self.assertTrue(np.all(imaginary > 0), imaginary)

    def test_it_is_diagonal_in_the_principal_frame(self):
        table = load_index_table(_write(BIAXIAL))
        eps = table.epsilon(0.8)
        off_diagonal = eps - np.diag(np.diag(eps))
        self.assertTrue(np.all(off_diagonal == 0))


class TheCrossCheckAgainstTheRegistry(unittest.TestCase):
    """Two independent routes to the same permittivity."""

    def test_a_table_from_the_package_sellmeier_reproduces_the_exported_registry(self):
        grid = np.round(np.arange(0.25, 2.001, 0.01), 4)
        lines = ["wavelength_um,no,ne"]
        lines += [f"{lam},{quartz_sellmeier_index(lam, extraordinary=False):.8f},"
                  f"{quartz_sellmeier_index(lam, extraordinary=True):.8f}" for lam in grid]
        table = load_index_table(_write("\n".join(lines)), name="quartz")
        registry = build_casestudy_material("Quartz z-cut (800 nm)")

        # the registry stores eps(2w) keyed on the FUNDAMENTAL, so its 2w tensor at 0.8 um is the
        # table read at 0.4 um -- which is the half-wavelength rule from the other direction
        np.testing.assert_allclose(np.diag(table.epsilon(0.8)).real,
                                   np.diag(registry.eps_w()).real, atol=2e-5)
        np.testing.assert_allclose(np.diag(table.epsilon(0.4)).real,
                                   np.diag(registry.eps_2w()).real, atol=2e-5)


class TheSpectrumFactory(unittest.TestCase):

    def setUp(self):
        self.table = load_index_table(_write(UNIAXIAL), name="demo")
        self.template = build_casestudy_material("Quartz z-cut (800 nm)")
        self.spectrum = table_spectrum(self.table, self.template)

    def test_the_table_supplies_the_linear_optics(self):
        material = self.spectrum(0.8)
        np.testing.assert_allclose(material.eps_w(), self.table.epsilon(0.8))
        np.testing.assert_allclose(material.eps_2w(), self.table.epsilon(0.4))

    def test_everything_else_comes_from_the_template(self):
        material = self.spectrum(0.8)
        np.testing.assert_array_equal(np.asarray(material.d_voigt_pm_v),
                                      np.asarray(self.template.d_voigt_pm_v, dtype=complex))
        self.assertEqual(material.structure.point_group, self.template.structure.point_group)
        np.testing.assert_allclose(material.orientation.rotation_matrix(),
                                   self.template.orientation.rotation_matrix())

    def test_it_returns_a_fresh_material_that_actually_moves(self):
        first, second = self.spectrum(0.7), self.spectrum(1.2)
        self.assertIsNot(first, second)
        self.assertFalse(np.allclose(first.eps_w(), second.eps_w()))

    def test_it_drives_a_spectral_sweep(self):
        from shaarp.spectral import solve_si_spectral_sweep

        sweep = solve_si_spectral_sweep(self.spectrum, lambda_min_um=0.7, lambda_max_um=1.5,
                                        lambda_step_um=0.2, theta_deg=45.0, phi_deg=30.0)
        self.assertEqual(sweep.wavelength_um.size, 5)
        self.assertTrue(sweep.support.varies, "a real table must not read as constant")
        self.assertLess(sweep.boundary_residual_norm.max(), 1e-10)


class MalformedTables(unittest.TestCase):

    def _expect(self, text, fragment):
        with self.assertRaises(ValueError) as ctx:
            load_index_table(_write(text))
        self.assertIn(fragment, str(ctx.exception).lower())

    def test_a_missing_wavelength_column(self):
        self._expect("n,k\n1.5,0\n1.4,0\n", "wavelength")

    def test_no_recognised_index_columns(self):
        self._expect("wavelength_um,epsilon\n0.8,2.25\n1.0,2.2\n", "index columns")

    def test_a_repeated_wavelength(self):
        self._expect("wavelength_um,n\n0.8,1.5\n0.8,1.6\n", "repeated wavelengths")

    def test_a_non_positive_index(self):
        self._expect("wavelength_um,n\n0.8,1.5\n1.0,0.0\n", "non-positive")

    def test_a_negative_extinction(self):
        self._expect("wavelength_um,n,k\n0.8,1.5,0.0\n1.0,1.5,-0.1\n", "negative extinction")

    def test_a_header_only_file(self):
        self._expect("wavelength_um,n\n", "data row")

    def test_a_non_numeric_cell_names_its_line(self):
        with self.assertRaises(ValueError) as ctx:
            load_index_table(_write("wavelength_um,n\n0.8,1.5\n1.0,abc\n"))
        self.assertIn("line 3", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TheShippedTables(unittest.TestCase):
    """The four dispersive materials that ship with the package.

    These carry published optical constants into the repository, so what matters is not that the
    loader works -- covered above -- but that each table is what it claims to be: the right
    crystal, over the range its name advertises, agreeing with the literature it cites."""

    # Literature values at 1.064 um, from the sources each table names. Pinned as constants: a
    # test that recomputes them from the same table it is checking asserts nothing.
    AT_1064 = {
        "linbo3_zelmon_1997": ([2.2321, 2.2321, 2.1555], 1e-3),
        "ktp_kato_2002": ([1.7379, 1.7455, 1.8297], 1e-3),
        "lbo_chen_1989": ([1.5655, 1.5906, 1.6056], 1e-3),
    }

    # TaAs is checked against the EXPORTED REGISTRY instead, because its source tabulates no
    # index -- see TheTaAsInfraredPoleSign below for why that check is load-bearing.
    TAAS_VS_REGISTRY_TOLERANCE = 0.03

    def test_every_registered_table_loads(self):
        from shaarp.dispersion import load_shipped_table, shipped_tables

        registry = shipped_tables()
        self.assertTrue(registry, "no dispersive materials are registered")
        for slug in registry:
            table = load_shipped_table(slug)
            self.assertGreater(table.wavelength_um.size, 10, slug)
            self.assertTrue(np.all(table.n > 0), slug)
            self.assertTrue(np.all(table.k >= 0), slug)

    def test_each_display_name_carries_its_wavelength_range(self):
        """The range is the first thing that decides whether a material can answer a given sweep,
        so it belongs in the name rather than only in a tooltip."""
        from shaarp.dispersion import dispersive_material_names, shipped_tables

        names = dispersive_material_names()
        self.assertEqual(len(names), len(shipped_tables()))
        for name, meta in zip(names, shipped_tables().values()):
            low, high = meta["range_um"]
            self.assertIn(f"{low:.2f}-{high:.2f} um", name, name)

    def test_the_tables_agree_with_the_literature_they_cite(self):
        from shaarp.dispersion import load_shipped_table

        for slug, (expected, tolerance) in self.AT_1064.items():
            got = load_shipped_table(slug).principal(1.064).real
            np.testing.assert_allclose(got, expected, atol=tolerance,
                                       err_msg=f"{slug} disagrees with its own source")

    def test_the_advertised_range_matches_the_data(self):
        from shaarp.dispersion import load_shipped_table, shipped_tables

        for slug, meta in shipped_tables().items():
            low, high = meta["range_um"]
            table = load_shipped_table(slug)
            self.assertAlmostEqual(table.range_um[0], low, places=4, msg=slug)
            self.assertAlmostEqual(table.range_um[1], high, places=4, msg=slug)

    def test_every_table_cites_a_primary_source(self):
        """Published optical constants in a repository without a citation are unverifiable."""
        from shaarp.dispersion import _data_path, shipped_tables

        for slug, meta in shipped_tables().items():
            self.assertTrue(meta.get("reference"), f"{slug} has no reference")
            self.assertRegex(meta.get("doi", ""), r"^10\.\d{4,}/", f"{slug} has no DOI")
            header = _data_path(f"{slug}.csv").read_text(encoding="utf-8").splitlines()[:8]
            banner = "\n".join(header)
            self.assertIn("source:", banner, slug)
            self.assertIn(meta["doi"], banner, slug)

    def test_gaas_absorbs_above_its_band_edge_and_is_clear_below(self):
        """A sanity check with physics in it: GaAs's gap is near 0.87 um, so its second harmonic
        at 0.532 um must absorb while the 1.064 um fundamental passes."""
        from shaarp.dispersion import load_shipped_table

        table = load_shipped_table("gaas_rakic_1996")
        self.assertGreater(table.principal(0.532)[0].imag, 0.1)
        self.assertLess(table.principal(1.064)[0].imag, 0.05)

    def test_a_dispersive_material_resolves_through_the_ordinary_seam(self):
        """The GUI reaches every material through material_for_label; these must arrive that way
        too, or they exist only in the spectral path and not in an ordinary single-wavelength run."""
        from shaarp.dispersion import dispersive_material_names
        from shaarp.layer_stack import material_for_label

        for name in dispersive_material_names():
            material = material_for_label(name, 1.064)
            self.assertEqual(material.name, name)
            self.assertTrue(np.all(np.isfinite(material.eps_w())))

    def test_a_dispersive_material_actually_sweeps(self):
        from shaarp.dispersion import dispersive_spectrum
        from shaarp.spectral import solve_si_spectral_sweep

        sweep = solve_si_spectral_sweep(dispersive_spectrum("KTP (dispersive) 0.43-3.54 um"),
                                        lambda_min_um=0.9, lambda_max_um=2.0, lambda_step_um=0.1,
                                        theta_deg=45.0, phi_deg=30.0)
        self.assertTrue(sweep.support.varies, "a published index table must not read as constant")
        self.assertFalse(np.allclose(sweep.intensity_s, sweep.intensity_s[0]))
        self.assertLess(sweep.boundary_residual_norm.max(), 1e-10)

    def test_the_tables_are_declared_as_package_data(self):
        """A source checkout finds them by path whether or not they are declared; an INSTALLED
        wheel only ships what package-data names, and the top-level "*.json" glob does not reach
        into a subdirectory. Without this the dispersive materials work here and vanish for a user
        who pip-installs."""
        import tomllib

        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
        patterns = config["tool"]["setuptools"]["package-data"]["shaarp"]
        for needed in ("dispersion_data/*.csv", "dispersion_data/*.json"):
            self.assertIn(needed, patterns,
                          f"{needed} is not declared, so it would not ship in a wheel")

    def test_a_sweep_reports_a_table_clamp_once_with_the_range_it_can_answer(self):
        """The half-wavelength rule, stated where it bites.

        A name says 0.40-5.00 um and a sweep from 0.55 um still clamps, because eps(2w) is the
        table read at lambda / 2. The table used to warn once PER GRID POINT, with the point's own
        wavelength in each line, and in the app that stream pushed the more important notes out of
        sight. A sweep now says it once, naming the range the table can actually answer."""
        from shaarp.dispersion import dispersive_spectrum
        from shaarp.spectral import solve_si_spectral_sweep

        name = "LiNbO3 (dispersive) 0.40-5.00 um"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve_si_spectral_sweep(dispersive_spectrum(name), lambda_min_um=0.55,
                                    lambda_max_um=1.6, lambda_step_um=0.05)
        messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        summary = [m for m in messages if "can answer a sweep over" in m]
        self.assertEqual(len(summary), 1, messages)
        self.assertIn("0.8-5 um", summary[0], "the answerable range is 2x the low end to the top")
        self.assertIn("half the wavelength", summary[0])
        self.assertFalse([m for m in messages if "has index data over" in m],
                         "the per-point form must not come back inside a sweep")

    def test_in_the_app_the_summary_prints_the_table_as_its_name_does(self):
        """LBO's table runs 0.2894-1.064 um and its name says 0.29-1.06; the note printed the raw
        ends, so it read like a second table. In the app it prints two decimals with an en dash,
        and the answerable start is rounded UP (0.5788 -> 0.58) so the number stated is inside."""
        from shaarp.dispersion import dispersive_spectrum
        from shaarp.spectral import solve_si_spectral_sweep, speaking_to

        name = "LiB3O5 / LBO (dispersive) 0.29-1.06 um"
        with warnings.catch_warnings(record=True) as caught, speaking_to("app"):
            warnings.simplefilter("always")
            solve_si_spectral_sweep(dispersive_spectrum(name), lambda_min_um=0.5,
                                    lambda_max_um=1.0, lambda_step_um=0.05)
        summary = [str(w.message) for w in caught if "can answer a sweep over" in str(w.message)]
        self.assertEqual(len(summary), 1, [str(w.message) for w in caught])
        self.assertIn("covers 0.29–1.06 µm", summary[0])
        self.assertIn("sweep over 0.58–1.06 µm", summary[0])
        self.assertNotIn("0.2894", summary[0])

    def test_the_short_name_the_docs_print_reaches_the_material(self):
        """docs/usage.md lists "LiNbO3 (dispersive)"; a reader copies that. It used to fail with an
        "unknown Case Study material" error whose list of choices left out every dispersive name,
        so the reader had nowhere to go from it."""
        from shaarp.spectral import casestudy_spectrum

        short = casestudy_spectrum("LiNbO3 (dispersive)")(1.064)
        full = casestudy_spectrum("LiNbO3 (dispersive) 0.40-5.00 um")(1.064)
        np.testing.assert_allclose(short.epsilon_omega, full.epsilon_omega)
        np.testing.assert_allclose(short.epsilon_2omega, full.epsilon_2omega)

        # the table in docs/usage.md sets formulae with subscript digits; a name copied from it
        # used to fail for exactly the two crystals whose formulae have digits
        for pretty in ("LiNbO₃ (dispersive)", "LiB₃O₅ / LBO (dispersive)"):
            with self.subTest(name=pretty):
                self.assertIsNotNone(casestudy_spectrum(pretty)(1.0).epsilon_omega)

        with self.assertRaises(ValueError) as ctx:
            casestudy_spectrum("LiNbO3 (dispersive, typo)")
        self.assertIn("LiNbO3 (dispersive) 0.40-5.00 um", str(ctx.exception),
                      "the error must offer the dispersive materials, not only the registry")

    def test_a_single_wavelength_read_keeps_its_own_clamp_warning(self):
        """Outside a sweep there is nothing to summarize, so the table still speaks for itself."""
        from shaarp.dispersion import dispersive_spectrum

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dispersive_spectrum("TaAs (dispersive) 0.21-1.03 um")(2.0)
        self.assertTrue([w for w in caught if "has index data over" in str(w.message)])

    def test_the_documented_table_lists_every_shipped_material(self):
        """docs/usage.md prints the materials WITH THEIR CITATIONS, which is how a reader decides
        whether a curve is trustworthy -- so a material that ships without a row there arrives
        uncited. Adding TaAs did exactly that: the table still said "Four crystals" and listed
        four. Matched on the registry rather than on prose so the doc cannot drift again."""
        from shaarp.dispersion import dispersive_material_names

        root = Path(__file__).resolve().parents[1]
        usage = (root / "docs" / "usage.md").read_text(encoding="utf-8")
        block = usage.split("### Materials that already carry dispersion", 1)
        self.assertEqual(len(block), 2, "docs/usage.md no longer documents the dispersive materials")
        table = block[1].split("###", 1)[0]
        rows = [line for line in table.splitlines()
                if line.startswith("|") and "---" not in line and "| Material |" not in line]
        names = dispersive_material_names()
        self.assertEqual(len(rows), len(names),
                         f"docs/usage.md lists {len(rows)} dispersive materials; {len(names)} ship")
        # the doc sets formulae with subscript digits (LiNbO₃); the registry name is plain ASCII.
        subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        plain_table = table.translate(subscripts)
        for name in names:
            # the display name is "<crystal> (dispersive) <lo>-<hi> um" -- match on the crystal,
            # since the doc renders the range with an en dash and a typographic micro sign.
            crystal = name.split(" (dispersive)")[0]
            self.assertIn(crystal, plain_table,
                          f"{crystal} ships but is not documented in docs/usage.md")


class TheTaAsInfraredPoleSign(unittest.TestCase):
    """TaAs is reconstructed from a Lorentz model whose printed equation has a sign slip.

    Zu et al., Phys. Rev. B 103, 165137 (2021) writes Eq. 1's infrared-pole term with a PLUS. A
    plus puts the ordinary eps_1 at 14.5 at 1.55 eV; the paper's own Fig. 1(a) reads about 6 there,
    and the exported case-study registry -- built independently from the author's setup.nb -- holds
    6.2591. The minus is also what the pole function A/(E_pole^2 - E^2) gives as E_pole goes to
    zero, which is the form the same equation writes its ultraviolet pole in.

    So this fence is not decoration: it is the only thing standing between the shipped table and a
    silent factor-of-two error in the real permittivity of a published material. It compares
    against the registry, which never passed through this module."""

    def setUp(self):
        from shaarp.casestudy_materials import build_casestudy_material
        from shaarp.dispersion import load_shipped_table

        self.table = load_shipped_table("taas_zu_2021")
        self.registry = build_casestudy_material("TaAs (112)")

    def test_the_table_reproduces_the_registry_at_both_harmonics(self):
        tolerance = TheShippedTables.TAAS_VS_REGISTRY_TOLERANCE
        for lam, reference in ((0.8, self.registry.eps_w()), (0.4, self.registry.eps_2w())):
            eps = self.table.epsilon(lam)
            for axis, label in ((0, "ordinary"), (2, "extraordinary")):
                got, want = eps[axis, axis], reference[axis, axis]
                self.assertLess(abs(got - want) / abs(want), tolerance,
                                f"{label} at {lam} um: {got:.4f} vs registry {want:.4f}")

    def test_a_positive_infrared_pole_would_fail_that_check(self):
        """Red-proof. Without this the test above passes for the wrong reason if someone 'fixes'
        the sign back to what the paper prints."""
        energy = 1.23984193 / 0.8
        ordinary_ir_amplitude = 9.9950
        correct = self.table.epsilon(0.8)[0, 0]
        flipped = correct + 2 * ordinary_ir_amplitude / energy**2      # minus -> plus
        want = self.registry.eps_w()[0, 0]
        self.assertGreater(abs(flipped - want) / abs(want),
                           10 * TheShippedTables.TAAS_VS_REGISTRY_TOLERANCE,
                           "a flipped pole sign must be far outside tolerance, or this fence is "
                           "not actually pinning the sign")

    def test_it_absorbs_everywhere_as_a_semimetal_must(self):
        """TaAs has no transparent window in this range; a zero extinction anywhere would mean the
        model had been mistaken for a Sellmeier fit."""
        self.assertTrue(np.all(self.table.k > 0.1), "a Weyl semimetal does not stop absorbing")
