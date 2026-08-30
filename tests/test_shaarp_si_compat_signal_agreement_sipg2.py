"""Gated agreement test for the POINT-GROUP single-interface SHAARP.si family (sipg2).

Companion to ``test_shaarp_si_compat_signal_agreement_sipg1.py``. Where sipg1
validated 6 crystal classes (3m/-42m/mm2/4mm/32/-6m2), sipg2 validates the
remaining 8 (2/m/-4/422/3/-6/4/6) so that the SI reflected-SHG path is validated
at FULL PARITY with the transmitted/Maker pipeline: every distinct buildable
symmetry-constrained d-pattern is now exercised in BOTH geometries.

The test is skipped until the live Wolfram SHAARP.si ER2w stage reference
(``shaarp_si_signal_stage_reference_sipg2.json``) has been exported into the
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
SIGNAL_REFERENCE_PATH = REF_DIR / "shaarp_si_signal_stage_reference_sipg2.json"
MAPPING_PATH = REF_DIR / "shaarp_si_case_input_mapping_sipg2.json"
SUMMARY_PATH = REF_DIR / "shaarp_si_compat_signal_agreement_sipg2.json"

EXPECTED_CASE_COUNT = 8
ATOL = 2e-6
RTOL = 0.0


@unittest.skipUnless(
    SIGNAL_REFERENCE_PATH.exists(),
    f"Wolfram SHAARP.si sipg2 ER2w stage reference not found at {SIGNAL_REFERENCE_PATH}; "
    "export it with export_shaarp_si_signal_stage_reference_sipg2.wl before running this test.",
)
class SHAARPSICompatSignalAgreementSIPG2Tests(unittest.TestCase):
    def test_si_compat_solver_matches_all_exported_sipg2_signal_stage_er2w_values(self):
        summary = build_agreement(tag="sipg2", atol=ATOL, rtol=RTOL)

        self.assertEqual(
            summary["status"],
            "si_compat_reflected_signal_matches_extension_shaarp_si_stage_reference",
        )
        self.assertEqual(summary["tag"], "sipg2")
        self.assertEqual(summary["case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["passing_er2w_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["failing_er2w_case_count"], 0)
        self.assertLessEqual(summary["max_er2w_abs_error"], ATOL)
        self.assertEqual(summary["normal_incidence_2omega_branch_policy"], "shaarp_reference_like")

    def test_every_sipg2_case_is_point_group_constrained(self):
        mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
        self.assertEqual(mapping["tag"], "sipg2")
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

    def test_saved_compat_signal_agreement_sipg2_matches_generator(self):
        if not SUMMARY_PATH.exists():
            self.skipTest(f"Saved sipg2 agreement summary not found at {SUMMARY_PATH}")
        saved = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_agreement(tag="sipg2", atol=ATOL, rtol=RTOL))


if __name__ == "__main__":
    unittest.main()
