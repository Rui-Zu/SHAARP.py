import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
REFERENCE_PATH = REF_DIR / "shaarp_si_two_omega_boundary_stage_reference_v1.json"


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


class SHAARPSITwoOmegaBoundaryStageReferenceTests(unittest.TestCase):
    def test_two_omega_boundary_reference_is_exported_but_limited(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_stage_reference_exported")
        self.assertEqual(payload["validation_claim"], "two_omega_boundary_stage_only_not_full_shaarp_agreement")
        self.assertEqual(payload["case_count"], 36)
        self.assertEqual(payload["numeric_two_omega_setup_case_count"], 36)
        self.assertEqual(payload["numeric_two_omega_boundary_residual_case_count"], 36)
        self.assertEqual(payload["numeric_two_omega_signal_case_count"], 36)
        self.assertIn("explicit names", payload["policy"]["boundary_unknown_policy"])

    def test_two_omega_boundary_stage_residuals_are_small(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertLess(payload["max_linear_boundary_abs_residual"], 1e-12)
        self.assertLess(payload["max_croots_abs_residual"], 1e-11)
        self.assertLess(payload["max_two_omega_boundary_abs_residual"], 1e-12)
        self.assertGreater(payload["max_abs_two_omega_signal_component"], 0.0)
        for case in payload["cases"]:
            self.assertEqual(case["parse_status"], "parsed")
            self.assertLess(case["checks"]["max_two_omega_boundary_abs_residual"], 1e-12)
            self.assertGreater(case["checks"]["max_abs_two_omega_signal_component"], 0.0)

    def test_two_omega_boundary_exports_named_coefficients_and_signal(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        expected_symbols = ["ER2wp", "ER2ws", "ET2weoA", "ET2woA"]
        for case in payload["cases"]:
            rules = case["outputs"]["E2wroots"]
            self.assertEqual([rule["symbol"] for rule in rules], expected_symbols)
            for rule in rules:
                _assert_complex_record(self, rule["value"])

            signal = case["outputs"]["RSignal"]
            self.assertEqual(len(signal), 2)
            self.assertEqual(signal, [rules[0]["value"], rules[1]["value"]])

            for key in ("ER2w", "HR2w", "ET2wNumExt", "ET2wNumOrd", "HT2wNumExt", "HT2wNumOrd"):
                vector = case["outputs"][key]
                self.assertEqual(len(vector), 3, key)
                for value in vector:
                    _assert_complex_record(self, value)

    def test_two_omega_boundary_json_has_no_symbolic_fallbacks(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        forbidden = {"input_form", 'Missing["NonNumeric"]', "Indeterminate", "ComplexInfinity"}
        for value in _walk_values(payload):
            if isinstance(value, dict):
                self.assertNotIn("input_form", value)
            elif isinstance(value, str):
                self.assertNotIn(value, forbidden)


if __name__ == "__main__":
    unittest.main()
