"""Gated agreement test for the POINT-GROUP single-interface SHAARP.si family (sipg1).

Mirrors ``test_shaarp_si_compat_signal_agreement_ext.py`` but for the ``sipg1``
tag, whose active crystals carry SYMMETRY-CONSTRAINED point-group d-tensors
(3m/-42m/mm2/4mm/32/-6m2) rotated crystal->lab, rather than the generic
18/18-nonzero lab d shared by every prior SI case.

The test is skipped until the live Wolfram SHAARP.si ER2w stage reference
(``shaarp_si_signal_stage_reference_sipg1.json``) has been exported into the
benchmarks/mathematica_reference directory. Once present, it asserts that the
Python ``solve_shaarp_si_reflected_shg`` ER2w matches the live SHAARP.si ER2w for
every case, and independently confirms (from the input mapping) that every case
is genuinely point-group-constrained (crystal_nonzero_d < 18).
"""

import json
import unittest
from pathlib import Path

from benchmarks.compare_shaarp_si_compat_signal_agreement_ext import build_agreement


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
SIGNAL_REFERENCE_PATH = REF_DIR / "shaarp_si_signal_stage_reference_sipg1.json"
MAPPING_PATH = REF_DIR / "shaarp_si_case_input_mapping_sipg1.json"
SUMMARY_PATH = REF_DIR / "shaarp_si_compat_signal_agreement_sipg1.json"

EXPECTED_CASE_COUNT = 6
ATOL = 2e-6
RTOL = 0.0


@unittest.skipUnless(
    SIGNAL_REFERENCE_PATH.exists(),
    f"Wolfram SHAARP.si sipg1 ER2w stage reference not found at {SIGNAL_REFERENCE_PATH}; "
    "export it with export_shaarp_si_signal_stage_reference_sipg1.wl before running this test.",
)
class SHAARPSICompatSignalAgreementSIPG1Tests(unittest.TestCase):
    def test_si_compat_solver_matches_all_exported_sipg1_signal_stage_er2w_values(self):
        summary = build_agreement(tag="sipg1", atol=ATOL, rtol=RTOL)

        self.assertEqual(
            summary["status"],
            "si_compat_reflected_signal_matches_extension_shaarp_si_stage_reference",
        )
        self.assertEqual(summary["tag"], "sipg1")
        self.assertEqual(summary["case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["passing_er2w_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["failing_er2w_case_count"], 0)
        self.assertLessEqual(summary["max_er2w_abs_error"], ATOL)
        self.assertEqual(summary["normal_incidence_2omega_branch_policy"], "shaarp_reference_like")

    def test_every_sipg1_case_is_point_group_constrained(self):
        mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
        self.assertEqual(mapping["tag"], "sipg1")
        self.assertEqual(len(mapping["cases"]), EXPECTED_CASE_COUNT)
        for case in mapping["cases"]:
            crystal_nz = case["crystal_nonzero_d"]
            self.assertIn("point_group", case)
            self.assertGreater(
                crystal_nz, 0, f"{case['id']} ({case['point_group']}) has an all-zero crystal d-tensor"
            )
            self.assertLess(
                crystal_nz,
                18,
                f"{case['id']} ({case['point_group']}) crystal d-tensor is not symmetry-constrained "
                f"(crystal_nonzero_d={crystal_nz} is not < 18)",
            )

    def test_saved_compat_signal_agreement_sipg1_matches_generator(self):
        if not SUMMARY_PATH.exists():
            self.skipTest(f"Saved sipg1 agreement summary not found at {SUMMARY_PATH}")
        saved = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_agreement(tag="sipg1", atol=ATOL, rtol=RTOL))


if __name__ == "__main__":
    unittest.main()
