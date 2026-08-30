import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
REFERENCE_PATH = REF_DIR / "shaarp_si_signal_stage_reference_v1.json"


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


class SHAARPSISignalStageReferenceTests(unittest.TestCase):
    def test_signal_stage_reference_is_exported_but_limited(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_stage_reference_exported")
        self.assertEqual(payload["validation_claim"], "signal_stage_only_not_full_shaarp_agreement")
        self.assertEqual(payload["case_count"], 36)
        self.assertEqual(payload["numeric_two_omega_boundary_residual_case_count"], 36)
        self.assertEqual(payload["numeric_two_omega_signal_case_count"], 36)
        self.assertEqual(payload["numeric_intensity_case_count"], 36)
        self.assertIn("RotateAnalyzer=False", payload["policy"]["rotate_analyzer_policy"])

    def test_signal_stage_keeps_boundary_residuals_small(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertLess(payload["max_two_omega_boundary_abs_residual"], 1e-12)
        self.assertGreater(payload["max_abs_two_omega_signal_component"], 0.0)
        self.assertGreater(payload["max_abs_intensity_component"], 0.0)
        for case in payload["cases"]:
            self.assertEqual(case["parse_status"], "parsed")
            self.assertLess(case["checks"]["max_two_omega_boundary_abs_residual"], 1e-12)
            self.assertGreater(case["checks"]["max_abs_intensity_component"], 0.0)

    def test_signal_stage_exports_nonnegative_real_intensities(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        intensity_keys = (
            "I2wx_at_phi_0",
            "I2wy_at_phi_0",
            "I2wParallel_at_phi_0",
            "I2wPerpdicular_at_phi_0",
            "MaxI2wx",
            "MaxI2wy",
            "MaxI2w",
        )
        for case in payload["cases"]:
            crystal = case["outputs"]["RSignalCrystal"]
            self.assertEqual(len(crystal), 2)
            for value in crystal:
                _assert_complex_record(self, value)

            values = {}
            for key in intensity_keys:
                value = case["outputs"][key]
                _assert_complex_record(self, value)
                self.assertGreaterEqual(value["real"], -1e-14, key)
                self.assertAlmostEqual(value["imag"], 0.0, delta=1e-14)
                values[key] = value["real"]

            expected_max = 1.1 * max(values["MaxI2wx"], values["MaxI2wy"])
            self.assertAlmostEqual(values["MaxI2w"], expected_max, delta=1e-12)

    def test_signal_stage_json_has_no_symbolic_fallbacks(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        forbidden = {"input_form", 'Missing["NonNumeric"]', "Indeterminate", "ComplexInfinity"}
        for value in _walk_values(payload):
            if isinstance(value, dict):
                self.assertNotIn("input_form", value)
            elif isinstance(value, str):
                self.assertNotIn(value, forbidden)


if __name__ == "__main__":
    unittest.main()
