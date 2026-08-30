import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
REFERENCE_PATH = REF_DIR / "shaarp_si_two_omega_setup_stage_reference_v1.json"


def _walk_values(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)


def _assert_complex_record(testcase, value):
    testcase.assertEqual(set(value), {"real", "imag"})
    testcase.assertTrue(math.isfinite(value["real"]))
    testcase.assertTrue(math.isfinite(value["imag"]))


class SHAARPSITwoOmegaSetupStageReferenceTests(unittest.TestCase):
    def test_two_omega_setup_reference_is_exported_but_limited(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_stage_reference_exported")
        self.assertEqual(payload["validation_claim"], "two_omega_setup_stage_only_not_full_shaarp_agreement")
        self.assertEqual(payload["case_count"], 36)
        self.assertEqual(payload["numeric_boundary_residual_case_count"], 36)
        self.assertEqual(payload["numeric_pnl_case_count"], 36)
        self.assertEqual(payload["numeric_croots_residual_case_count"], 36)
        self.assertEqual(payload["numeric_two_omega_setup_case_count"], 36)
        self.assertIn("ER2wp", payload["policy"]["boundary_unknown_policy"])

    def test_two_omega_setup_stage_residuals_remain_numeric(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertLess(payload["max_linear_boundary_abs_residual"], 1e-12)
        self.assertLess(payload["max_croots_abs_residual"], 1e-11)
        self.assertGreater(payload["max_abs_pnl_component"], 0.0)
        self.assertGreater(payload["max_abs_two_omega_setup_component"], 0.0)
        for case in payload["cases"]:
            self.assertEqual(case["parse_status"], "parsed")
            self.assertLess(case["checks"]["max_linear_boundary_abs_residual"], 1e-12)
            self.assertLess(case["checks"]["max_croots_abs_residual"], 1e-11)
            self.assertGreater(case["checks"]["max_abs_two_omega_setup_component"], 0.0)

    def test_two_omega_setup_exports_numeric_source_fields(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        vector_keys = (
            "DT2w2eo",
            "DT2w2o",
            "DT2weoo",
            "Vkt2w2eo",
            "Vkt2w2o",
            "Vkt2weoo",
            "HT2w2eo",
            "HT2w2o",
            "HT2weoo",
        )
        for case in payload["cases"]:
            rc = case["outputs"]["RC"]
            self.assertEqual(len(rc), 3)
            for row in rc:
                self.assertEqual(len(row), 3)
                for value in row:
                    _assert_complex_record(self, value)

            for key in vector_keys:
                vector = case["outputs"][key]
                self.assertEqual(len(vector), 3, key)
                for value in vector:
                    _assert_complex_record(self, value)

    def test_two_omega_setup_json_has_no_symbolic_fallbacks(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        forbidden = {"input_form", 'Missing["NonNumeric"]', "Indeterminate", "ComplexInfinity"}
        for value in _walk_values(payload):
            if isinstance(value, dict):
                self.assertNotIn("input_form", value)
            elif isinstance(value, str):
                self.assertNotIn(value, forbidden)


if __name__ == "__main__":
    unittest.main()
