import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
REFERENCE_PATH = REF_DIR / "shaarp_si_principal_axis_stage_reference_v1.json"


class SHAARPSIPrincipalAxisStageReferenceTests(unittest.TestCase):
    def test_principal_axis_stage_reference_is_exported_but_limited(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_stage_reference_exported")
        self.assertEqual(payload["validation_claim"], "principal_axis_stage_only_not_full_shaarp_agreement")
        self.assertEqual(payload["region_id"], "principal_axis_and_index_setup")
        self.assertEqual(payload["case_count"], 36)

    def test_principal_axis_stage_reconstructs_lab_epsilon_values(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertLess(payload["max_abs_elabwNum_minus_expected_lab"], 1e-12)
        self.assertLess(payload["max_abs_elab2wNum_minus_expected_lab"], 1e-12)
        for case in payload["cases"]:
            self.assertEqual(case["parse_status"], "parsed")
            self.assertLess(case["checks"]["max_abs_elabwNum_minus_expected_lab"], 1e-12)
            self.assertLess(case["checks"]["max_abs_elab2wNum_minus_expected_lab"], 1e-12)

    def test_principal_axis_stage_exports_expected_intermediates(self):
        first = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))["cases"][0]
        outputs = first["outputs"]

        for key in (
            "a",
            "aCryToPrinw",
            "aCryToPrin2w",
            "Qw",
            "Q2w",
            "nwPrinciple",
            "n2wPrinciple",
            "elabwNum",
            "elab2wNum",
        ):
            self.assertIn(key, outputs)


if __name__ == "__main__":
    unittest.main()
