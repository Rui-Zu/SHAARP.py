"""Gated agreement test at GRAZING INCIDENCE (sigraze) — the R1 angle band, closed.

WHY THIS EXISTS. Residual risk R1 says live-Mathematica agreement is certified only AT the exported
reference points. Enumerating the incidence angles those points actually cover found a specific,
reachable hole:

    certified theta (deg), all pre-existing SI mappings:
      v1 0.0 8.0 17.5 29.0 41.0 55.0
      ext1 4.0 12.0 20.5 35.0 48.0 62.0 70.0
      sipg1 8.0 12.0 20.5 29.0 35.0 48.0
      sipg2 6.5 9.0 14.0 19.0 27.5 33.0 41.0 52.0
    -> MAX = 70.0

Above 70 deg nothing was certified, yet the GUI exposes the full range and its own matrix sweep
drives theta = 89 deg. That band is where sin(theta) -> 1, the transmitted wave approaches grazing,
cos(theta_T) gets small and the boundary determinants are worst-conditioned — i.e. where a geometry
or convention divergence between the port and the original would show up first if one existed.

RESULT (live Wolfram 14.3 export): it does not. All six cases agree to
max_abs_err = 3.56e-15 / max_rel_err = 5.46e-13, nine orders inside the 2e-6 gate, with the worst
angle (89.5 deg) the TIGHTEST case rather than the loosest. The certified theta ceiling moves
70 -> 89.5 deg and the register's angle-coverage hole is closed by measurement, not by re-wording.

The cases are built by the SAME validated helpers as sipg1/sipg2 (`_eps_axes`,
`_pointgroup_d_voigt_crystal`, `run_case`, `orientation_matrix`, `complex_symmetric_epsilon`), so
the ONLY thing that differs from an already-validated set is the incidence angle — which is what
makes agreement here attributable to the angle. eps is complex by construction (imag_base
0.025/0.035), so these cases are absorbing AS WELL AS grazing.

Skipped until the live reference JSON is present, matching the sipg1/sipg2/ext1 pattern.
"""

import json
import unittest
from pathlib import Path

from benchmarks.compare_shaarp_si_compat_signal_agreement_ext import build_agreement


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
SIGNAL_REFERENCE_PATH = REF_DIR / "shaarp_si_signal_stage_reference_sigraze.json"
MAPPING_PATH = REF_DIR / "shaarp_si_case_input_mapping_sigraze.json"
SUMMARY_PATH = REF_DIR / "shaarp_si_compat_signal_agreement_sigraze.json"

EXPECTED_CASE_COUNT = 6
ATOL = 2e-6
RTOL = 0.0

# The whole point of the set: every case sits ABOVE the previous certified ceiling.
PREVIOUS_CERTIFIED_THETA_CEILING_DEG = 70.0
EXPECTED_THETA_DEG = [75.0, 80.0, 85.0, 88.0, 89.0, 89.5]


@unittest.skipUnless(
    SIGNAL_REFERENCE_PATH.exists(),
    f"Wolfram SHAARP.si sigraze ER2w stage reference not found at {SIGNAL_REFERENCE_PATH}; "
    "export it with export_shaarp_si_signal_stage_reference_sigraze.wl before running this test.",
)
class SHAARPSICompatSignalAgreementGrazingTests(unittest.TestCase):
    def test_si_compat_solver_matches_live_shaarp_si_at_grazing_incidence(self):
        summary = build_agreement(tag="sigraze", atol=ATOL, rtol=RTOL)

        self.assertEqual(
            summary["status"],
            "si_compat_reflected_signal_matches_extension_shaarp_si_stage_reference",
        )
        self.assertEqual(summary["tag"], "sigraze")
        self.assertEqual(summary["case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["passing_er2w_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["failing_er2w_case_count"], 0)
        self.assertLessEqual(summary["max_er2w_abs_error"], ATOL)
        self.assertEqual(summary["normal_incidence_2omega_branch_policy"], "shaarp_reference_like")

    def test_agreement_is_machine_precision_not_merely_inside_the_gate(self):
        """A 2e-6 pass would be uninformative here — the measured agreement is ~1e-15.

        Pinning the achieved level (with headroom) is what makes this a REGRESSION fence: if a
        future change quietly degrades grazing accuracy to, say, 1e-9, the loose gate above would
        still pass and the degradation would ship silently.
        """
        summary = build_agreement(tag="sigraze", atol=ATOL, rtol=RTOL)
        self.assertLess(
            summary["max_er2w_abs_error"], 1e-12,
            f"grazing agreement degraded: max_abs_err={summary['max_er2w_abs_error']:.3e} "
            f"(measured 3.56e-15 on)")
        self.assertLess(
            summary["max_er2w_rel_error"], 1e-10,
            f"grazing relative agreement degraded: "
            f"max_rel_err={summary['max_er2w_rel_error']:.3e} (measured 5.46e-13)")

    def test_every_case_is_above_the_previously_certified_angle_ceiling(self):
        """Guards the REASON the set exists: if someone re-tuned these angles down into the band
        ext1 already covers, the module would still pass while certifying nothing new."""
        mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
        self.assertEqual(mapping["tag"], "sigraze")
        self.assertEqual(len(mapping["cases"]), EXPECTED_CASE_COUNT)

        thetas = [case["theta_deg"] for case in mapping["cases"]]
        self.assertEqual(thetas, EXPECTED_THETA_DEG)
        for theta in thetas:
            self.assertGreater(
                theta, PREVIOUS_CERTIFIED_THETA_CEILING_DEG,
                f"theta={theta} is at or below the pre-existing certified ceiling "
                f"({PREVIOUS_CERTIFIED_THETA_CEILING_DEG} deg from ext1) — this set exists to "
                f"certify ABOVE it")

    def test_every_case_is_point_group_constrained_and_absorbing(self):
        """Same structural checks sipg1/sipg2 apply, plus the absorbing-eps property that makes
        these cases doubly stressful (grazing AND complex eps)."""
        mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
        for case in mapping["cases"]:
            self.assertIn("point_group", case)
            crystal_nz = case["crystal_nonzero_d"]
            self.assertGreater(
                crystal_nz, 0,
                f"{case['id']} ({case['point_group']}) has an all-zero crystal d-tensor")
            self.assertLess(
                crystal_nz, 18,
                f"{case['id']} ({case['point_group']}) crystal d-tensor is not symmetry-constrained "
                f"(crystal_nonzero_d={crystal_nz} is not < 18)")

    def test_saved_compat_signal_agreement_sigraze_matches_generator(self):
        if not SUMMARY_PATH.exists():
            self.skipTest(f"Saved sigraze agreement summary not found at {SUMMARY_PATH}")
        saved = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_agreement(tag="sigraze", atol=ATOL, rtol=RTOL))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
