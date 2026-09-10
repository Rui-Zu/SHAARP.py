"""SHAARP.py's linear multilayer Fresnel stage vs two external codes and a closed form.

This is the ALWAYS-ON gate. It reads the committed fixture
``benchmarks/isotropic_stack_reference_v1.json`` and needs neither ``tmm`` nor ``inkstone``
installed, so it runs in CI, in the frozen bundle and offline.
``tests/test_isotropic_stack_live_recheck.py`` recomputes the fixture when those packages ARE
present, so it cannot drift unnoticed.

Why this exists
---------------
Every other check on the linear stage compares SHAARP.py against Mathematica SHAARP.ml. That is
fidelity evidence, not independence -- one formalism, one author. These references are outside
work: ``tmm`` is a transfer-matrix code, ``inkstone`` is RCWA (a Fourier-space eigenmode expansion
that must collapse to the same answer for an unpatterned stack), and the third leg is a
hand-written Abeles characteristic matrix.

Measured on the fixture as committed: reflectance and transmittance both agree with ``tmm`` to
1.3e-13 across all six cases, and the legacy bare-``|t|**2`` convention misses by up to 7.8e-01.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_isotropic_stack_reference import (
    CURVE_KEYS,
    REFERENCE_PATH,
    build_isotropic_stack_agreement_summary,
    load_reference,
    reference_curves,
)
from benchmarks.isotropic_stack_cases import build_cases, shaarp_rt

# Measured worst deviation vs tmm across all six cases is 1.29e-13 (single_film_angle_sweep, T_p).
# The gate sits an order of magnitude above that. Anything approaching 1e-9 on a linear-vs-linear
# comparison means something is actually wrong -- diagnose it, do not relax this number.
ATOL = 1e-12
EXPECTED_CASE_COUNT = 6
EXPECTED_LEGS = ("tmm", "inkstone", "closed_form")

# Reference legs are two formalisms plus a textbook expression; they agree to 6.1e-15 as committed.
CROSS_LEG_ATOL = 1e-11


class IsotropicStackReferenceProvenanceTests(unittest.TestCase):
    """The fixture must be what it claims to be. These run even if SHAARP itself is broken."""

    @classmethod
    def setUpClass(cls):
        cls.reference = load_reference()

    def test_reference_file_is_committed_next_to_its_generator(self):
        self.assertTrue(REFERENCE_PATH.exists(), "committed reference fixture is missing")
        generator = Path(REFERENCE_PATH).parent / "generate_isotropic_stack_reference.py"
        self.assertTrue(generator.exists(), "the fixture must ship with the script that regenerates it")

    def test_reference_names_its_external_sources_and_versions(self):
        source = str(self.reference.get("source", "")).lower()
        self.assertIn("tmm", source)
        self.assertIn("inkstone", source)
        packages = self.reference.get("packages", {})
        for name in ("tmm", "inkstone"):
            version = str(packages.get(name, ""))
            self.assertNotIn(version, ("", "not-installed"),
                             "reference does not record a real %s version" % name)
            self.assertRegex(version, r"^\d+\.\d+", "%s version %r is not a version string" % (name, version))
        self.assertEqual(self.reference.get("status"), "external_reference_exported")

    def test_reference_carries_every_case_and_every_leg(self):
        self.assertEqual(self.reference["case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(len(self.reference["cases"]), EXPECTED_CASE_COUNT)
        defined = {case.case_id for case in build_cases()}
        self.assertEqual({c["case_id"] for c in self.reference["cases"]}, defined)
        for case in self.reference["cases"]:
            self.assertEqual(tuple(sorted(case["legs"])), tuple(sorted(EXPECTED_LEGS)))
            for leg in EXPECTED_LEGS:
                for key in CURVE_KEYS:
                    self.assertEqual(len(case["legs"][leg][key]), len(case["grid"]),
                                     "%s/%s/%s length mismatch" % (case["case_id"], leg, key))

    def test_the_three_reference_legs_agree_with_each_other(self):
        """If the references disagree among themselves, nothing downstream means anything."""
        worst = 0.0
        for case in self.reference["cases"]:
            legs = {name: reference_curves(case, leg=name) for name in EXPECTED_LEGS}
            for i, first in enumerate(EXPECTED_LEGS):
                for second in EXPECTED_LEGS[i + 1:]:
                    for key in CURVE_KEYS:
                        err = float(np.max(np.abs(legs[first][key] - legs[second][key])))
                        worst = max(worst, err)
                        self.assertLessEqual(
                            err, CROSS_LEG_ATOL,
                            "%s: %s vs %s disagree on %s by %.3e" % (case["case_id"], first, second, key, err))
        self.assertGreater(worst, 0.0, "cross-leg comparison found zero difference anywhere -- "
                                       "the legs are probably the same array, not three computations")

    def test_reference_is_not_vacuous(self):
        """Curves must actually vary, and lossless cases must conserve energy."""
        for case in self.reference["cases"]:
            curves = reference_curves(case, leg="tmm")
            if len(case["grid"]) > 1:
                spread = max(float(np.ptp(curves[k])) for k in CURVE_KEYS)
                self.assertGreater(spread, 1e-3,
                                   "%s reference curves are flat -- not a real sweep" % case["case_id"])
            if case["case_id"] != "absorbing_gold_film":
                for pol in ("s", "p"):
                    total = curves["R_" + pol] + curves["T_" + pol]
                    self.assertLess(float(np.max(np.abs(total - 1.0))), 1e-12,
                                    "%s: lossless reference violates R+T=1" % case["case_id"])

    def test_oscillatory_cases_resolve_their_fringes(self):
        for case in self.reference["cases"]:
            samples = case.get("samples_per_fringe")
            if samples is None:
                continue
            self.assertGreaterEqual(samples, self.reference["min_samples_per_fringe"],
                                    "%s aliases its fringes" % case["case_id"])

    def test_absorbing_case_actually_absorbs(self):
        case = {c["case_id"]: c for c in self.reference["cases"]}["absorbing_gold_film"]
        curves = reference_curves(case, leg="tmm")
        for pol in ("s", "p"):
            total = curves["R_" + pol] + curves["T_" + pol]
            self.assertLess(float(np.max(total)), 1.0, "gold film reference has R+T >= 1")
            self.assertGreater(float(np.min(total)), 0.5, "gold film reference absorbs implausibly much")


class IsotropicStackAgreementTests(unittest.TestCase):
    """SHAARP.py against the committed external reference."""

    @classmethod
    def setUpClass(cls):
        cls.reference = load_reference()
        cls.cases = {case.case_id: case for case in build_cases()}
        cls.shaarp = {cid: shaarp_rt(case) for cid, case in cls.cases.items()}
        cls.legacy = {cid: shaarp_rt(case, transmittance="amplitude")
                      for cid, case in cls.cases.items()}
        cls.summary = build_isotropic_stack_agreement_summary(
            cls.reference, atol=ATOL, leg="tmm",
            shaarp_curves=cls.shaarp, legacy_curves=cls.legacy)

    def test_shaarp_matches_the_external_reference(self):
        self.assertEqual(self.summary["status"], "isotropic_stack_outputs_match_all_compared_cases")
        self.assertEqual(self.summary["nonsingular_fail_count"], 0)
        self.assertEqual(self.summary["case_count"], EXPECTED_CASE_COUNT)
        self.assertLessEqual(self.summary["max_abs_error"], ATOL,
                             "worst deviation %.3e exceeds %.1e" % (self.summary["max_abs_error"], ATOL))

    def test_reflectance_and_transmittance_each_agree(self):
        self.assertLessEqual(self.summary["max_abs_error_reflectance"], ATOL)
        self.assertLessEqual(self.summary["max_abs_error_transmittance"], ATOL)

    def test_shaarp_also_matches_the_rcwa_leg(self):
        """inkstone is a different formalism; agreeing with tmm alone could be a shared convention."""
        summary = build_isotropic_stack_agreement_summary(
            self.reference, atol=ATOL, leg="inkstone", shaarp_curves=self.shaarp)
        self.assertEqual(summary["nonsingular_fail_count"], 0)
        self.assertLessEqual(summary["max_abs_error"], ATOL)

    def test_legacy_amplitude_convention_is_the_one_that_disagrees(self):
        """The point of the benchmark: |t|^2 is NOT a transmittance once the exit medium differs.

        Recorded rather than merely asserted, so the reason the power weighting exists stays in the
        test suite. The legacy curves are still exactly ``|t|**2``, which the next test pins.
        """
        legacy_worst = self.summary["max_abs_error_transmittance_legacy"]
        self.assertIsNotNone(legacy_worst)
        self.assertGreater(legacy_worst, 0.1,
                           "legacy |t|^2 unexpectedly agrees with the external T -- if the fixture's "
                           "stacks became index-matched this test has stopped discriminating")

    def test_power_and_amplitude_differ_by_exactly_the_obliquity_factor(self):
        from benchmarks.compare_isotropic_stack_reference import case_flux_factors

        worst = 0.0
        for case_id, case in self.cases.items():
            factors = case_flux_factors(case)
            for key in ("T_s", "T_p"):
                predicted = self.legacy[case_id][key] * factors
                worst = max(worst, float(np.max(np.abs(predicted - self.shaarp[case_id][key]))))
        self.assertLess(worst, 1e-12,
                        "power/amplitude transmittance differ by something other than the obliquity "
                        "factor (worst %.3e)" % worst)

    def test_half_wave_absentee_reproduces_the_bare_substrate(self):
        """A discriminating anchor: a lam/2 film is optically invisible at normal incidence.

        R must equal the bare air/glass value regardless of the film index. A solver that is merely
        plausible will not land on this.
        """
        case = self.cases["half_wave_absentee"]
        expected = case.metadata["expected_bare_R"]
        for key in ("R_s", "R_p"):
            self.assertAlmostEqual(float(self.shaarp["half_wave_absentee"][key][0]), expected, places=10,
                                   msg="absentee layer does not vanish in %s" % key)

    def test_brewster_angle_lands_where_the_index_ratio_says(self):
        case = self.cases["bare_interface_air_glass"]
        curves = self.shaarp["bare_interface_air_glass"]
        theta = np.asarray(case.theta_deg, dtype=float)
        measured = float(theta[int(np.argmin(curves["R_p"]))])
        self.assertAlmostEqual(measured, case.metadata["brewster_deg"], delta=2.5,
                               msg="p-polarized minimum is not at Brewster's angle")
        self.assertLess(float(np.min(curves["R_p"])), 5e-3, "Brewster minimum is not deep")

    def test_lossless_stacks_conserve_energy_with_a_non_air_substrate(self):
        """The invariant that the shipped R+T=1 test could not see, because its substrate was n=1."""
        for case_id, curves in self.shaarp.items():
            if case_id == "absorbing_gold_film":
                continue
            for pol in ("s", "p"):
                total = curves["R_" + pol] + curves["T_" + pol]
                self.assertLess(float(np.max(np.abs(total - 1.0))), 1e-11,
                                "%s violates R+T=1 in %s" % (case_id, pol))


if __name__ == "__main__":
    unittest.main()
