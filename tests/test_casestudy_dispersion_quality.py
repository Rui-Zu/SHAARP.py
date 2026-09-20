"""A material's tabulated grid says where its data EXISTS, not where it is still physical.

WHY THIS EXISTS. Five shipped materials return an unphysical eps(2w) well inside their own
advertised 0.40-2.00 um grid, because the exported Sellmeier models run into their ultraviolet
pole at the HALF wavelength. At lambda = 0.44 um, KTP's eps(2w) principal diagonal reaches
[60.8, 436.1, 7039.9] and LiNbO3 (1550 nm) goes to -0.33 with a ZERO imaginary part -- a lossless
negative permittivity, which a transparent medium cannot have. `casestudy_lambda_range` reports
(0.40, 2.00) for all of them, so the R2 clamp note stayed silent: grid endpoints cannot catch a
pole that lies between them. Anyone computing at a blue-end wavelength on the two materials most
worth picking to SEE dispersion got a confident wrong answer.

Two details were wrong in the first draft of the screen:

  * Taking min/max over the good points spans straight ACROSS an interior pole and re-admits the
    values being excluded -- LiNbO3 (1550 nm) has a good point on the far blue side of its own
    pole, so min/max returns 0.40-2.00 where the contiguous run returns 0.54-2.00. The window is
    the CONTIGUOUS run containing the native wavelength, and that is fenced at registry level.
  * Screening eps(w) as well as eps(2w) rejects data that is physically fine, because an
    ABSORBING medium legitimately has eps(w) far above eps(2w). On today's registry this changes
    nothing -- every absorbing material is constant-by-source and returns before the screen is
    consulted, and the six dispersive ones are transparent dielectrics whose window is identical
    either way (measured). So it is fenced on the PREDICATE with a synthetic absorbing tensor
    rather than through the registry, where it would be a test that cannot fail.

The reason strings are asserted ASCII because they are carried into RuntimeWarning messages, and a
plain Windows console is cp1252: a stray omega there raises UnicodeEncodeError from inside the
warning rather than from the physics.
"""
from __future__ import annotations

import unittest

import numpy as np

from shaarp.casestudy_materials import (
    CASE_STUDY_ORDER,
    _point_is_physical,
    build_casestudy_material,
    casestudy_lambda_range,
    casestudy_spectral_support,
    casestudy_usable_lambda_range,
)

# MEASURED from the shipped registry, pinned as constants rather than recomputed: a test that
# recomputes the thing it is checking passes no matter what the screen does.
POLED = {
    "LiNbO3 z-cut (1550 nm)": 0.54,
    "LiNbO3 x-cut (1550 nm)": 0.54,
    "KTP x-cut": 0.54,
    "KTP y-cut": 0.54,
    "ZnO (001)": 0.42,
}
# Materials the screen must NOT touch: two real metals (negative eps, but LOSSY), a layered
# semiconductor with a small eps(2w), and a single-point .si case.
UNTOUCHED = ("Pt (111) (1550 nm)", "Au coating (800 nm)", "MoS2", "TaAs (112)",
             "Quartz z-cut (800 nm)", "GaAs (111) (800 nm)")


class SpectralSupportClassification(unittest.TestCase):

    def test_every_registry_material_classifies(self):
        for name in CASE_STUDY_ORDER:
            support = casestudy_spectral_support(name)
            self.assertIn(support.kind, {"tabulated", "constant", "single_point"}, name)
            self.assertTrue(support.reason, name)
            self.assertEqual(support.varies, support.kind == "tabulated", name)

    def test_the_three_kinds_partition_the_registry_as_measured(self):
        counts: dict[str, int] = {}
        for name in CASE_STUDY_ORDER:
            kind = casestudy_spectral_support(name).kind
            counts[kind] = counts.get(kind, 0) + 1
        # 6 genuinely dispersive, 11 constant-by-source, 7 single-point .si cases.
        self.assertEqual(counts, {"tabulated": 6, "constant": 11, "single_point": 7})

    def test_a_flat_material_is_never_reported_as_dispersive(self):
        for name in CASE_STUDY_ORDER:
            support = casestudy_spectral_support(name)
            if not support.varies:
                continue
            lo, hi = support.usable_um
            a = build_casestudy_material(name, wavelength_um=lo)
            b = build_casestudy_material(name, wavelength_um=hi)
            self.assertFalse(np.allclose(a.eps_w(), b.eps_w())
                             and np.allclose(a.eps_2w(), b.eps_2w()),
                             f"{name} is classified dispersive but its tensors do not move")

    def test_unknown_material_does_not_raise(self):
        support = casestudy_spectral_support("no such material")
        self.assertIsNone(support.usable_um)
        self.assertIsNone(casestudy_usable_lambda_range("no such material"))


class ThePoleRegionIsExcluded(unittest.TestCase):

    def test_the_five_poled_materials_are_narrowed_to_their_measured_floor(self):
        for name, floor in POLED.items():
            usable = casestudy_usable_lambda_range(name)
            grid = casestudy_lambda_range(name)
            self.assertAlmostEqual(usable[0], floor, places=6, msg=name)
            self.assertAlmostEqual(usable[1], grid[1], places=6, msg=name)
            self.assertGreater(usable[0], grid[0],
                               f"{name} must be narrower than its grid, not equal to it")

    def test_the_screen_leaves_metals_and_flat_materials_alone(self):
        for name in UNTOUCHED:
            self.assertEqual(casestudy_usable_lambda_range(name), casestudy_lambda_range(name),
                             f"{name} must keep its full grid span")

    def test_the_excluded_wavelengths_really_are_unphysical(self):
        """Red-proof: the detector must reject what it excludes AND accept what it keeps."""
        for name, floor in POLED.items():
            below = build_casestudy_material(name, wavelength_um=floor - 0.04)
            self.assertFalse(_point_is_physical(below.eps_w(), below.eps_2w()),
                             f"{name} below its floor should have been rejected")
            inside = build_casestudy_material(name, wavelength_um=floor + 0.02)
            self.assertTrue(_point_is_physical(inside.eps_w(), inside.eps_2w()),
                            f"{name} above its floor should have been accepted")

    def test_the_named_symptoms_are_still_present_in_the_shipped_data(self):
        """If a future re-export fixes the data, this fails and the screen can be retired."""
        ktp = build_casestudy_material("KTP x-cut", wavelength_um=0.44)
        self.assertGreater(max(abs(ktp.eps_2w()[i, i].real) for i in range(3)), 1e3)
        lno = build_casestudy_material("LiNbO3 z-cut (1550 nm)", wavelength_um=0.44)
        lossless_negative = [lno.eps_2w()[i, i] for i in range(3)
                             if lno.eps_2w()[i, i].real < 0 and abs(lno.eps_2w()[i, i].imag) <= 1e-12]
        self.assertTrue(lossless_negative, "LiNbO3 (1550 nm) at 0.44 um should be lossless-negative")

    def test_the_window_does_not_span_across_an_interior_pole(self):
        """min/max over the good points would re-admit the pole; the run must be contiguous."""
        for name in POLED:
            lo, hi = casestudy_usable_lambda_range(name)
            grid = np.asarray(
                [lo + k * (hi - lo) / 40.0 for k in range(41)], dtype=float)
            for lam in grid:
                mat = build_casestudy_material(name, wavelength_um=float(lam))
                self.assertTrue(_point_is_physical(mat.eps_w(), mat.eps_2w()),
                                f"{name} at {lam:g} um is inside the usable window but unphysical")


class ThePredicateItself(unittest.TestCase):
    """Fences the screen's two rules directly, including the absorbing case the shipped registry
    does not currently exercise."""

    @staticmethod
    def _diag(*values):
        return np.diag(np.asarray(values, dtype=complex))

    def test_a_lossless_negative_eps_2w_is_rejected(self):
        self.assertFalse(_point_is_physical(self._diag(5, 5, 5), self._diag(-0.33, -0.33, 4.57)))

    def test_a_lossy_negative_eps_2w_is_accepted(self):
        """A real metal has a negative eps WITH loss -- Pt sits near -60 + 20j. Rejecting that
        would throw out every metal in the registry."""
        self.assertTrue(_point_is_physical(self._diag(-60 + 20j, -60 + 20j, -60 + 20j),
                                           self._diag(-60 + 20j, -60 + 20j, -60 + 20j)))

    def test_an_eps_2w_far_above_eps_w_is_rejected_as_a_pole(self):
        self.assertFalse(_point_is_physical(self._diag(11.1, 11.5, 14.9),
                                            self._diag(60.8, 436.1, 7039.9)))

    def test_an_absorbing_medium_with_eps_w_far_above_eps_2w_is_ACCEPTED(self):
        """The rule must not be applied symmetrically. MoS2-like: a large eps(w) over a small
        eps(2w) is ordinary absorption, not a pole. Screening eps(w) too would reject this."""
        self.assertTrue(_point_is_physical(self._diag(25 + 12j, 25 + 12j, 8 + 3j),
                                           self._diag(0.25, 0.25, 1.94)))

    def test_an_ordinary_transparent_crystal_is_accepted(self):
        self.assertTrue(_point_is_physical(self._diag(2.43, 2.43, 2.46),
                                           self._diag(2.50, 2.50, 2.53)))


class ReasonsAreSafeToPrint(unittest.TestCase):

    def test_reasons_are_ascii_so_a_cp1252_console_can_print_a_warning(self):
        for name in CASE_STUDY_ORDER:
            reason = casestudy_spectral_support(name).reason
            reason.encode("ascii")   # raises UnicodeEncodeError if a stray omega creeps back in
            reason.encode("cp1252")

    def test_each_reason_names_its_material(self):
        for name in CASE_STUDY_ORDER:
            self.assertIn(name, casestudy_spectral_support(name).reason, name)


if __name__ == "__main__":
    unittest.main()
