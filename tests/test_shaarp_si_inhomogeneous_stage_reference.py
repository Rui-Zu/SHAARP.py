import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
REFERENCE_PATH = REF_DIR / "shaarp_si_inhomogeneous_stage_reference_v1.json"


def _walk_values(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)


class SHAARPSIInhomogeneousStageReferenceTests(unittest.TestCase):
    def test_inhomogeneous_stage_reference_is_exported_but_limited(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_stage_reference_exported")
        self.assertEqual(payload["validation_claim"], "inhomogeneous_stage_only_not_full_shaarp_agreement")
        self.assertEqual(payload["case_count"], 36)
        self.assertEqual(payload["numeric_boundary_residual_case_count"], 36)
        self.assertEqual(payload["numeric_pnl_case_count"], 36)
        self.assertEqual(payload["numeric_croots_residual_case_count"], 36)
        self.assertIn("line-850-to-856", payload["policy"]["wave_constant_policy"])

    def test_inhomogeneous_stage_keeps_upstream_numeric_stages(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertLess(payload["max_linear_boundary_abs_residual"], 1e-12)
        self.assertLess(payload["max_croots_abs_residual"], 1e-11)
        self.assertGreater(payload["max_abs_pnl_component"], 0.0)
        for case in payload["cases"]:
            self.assertEqual(case["parse_status"], "parsed")
            self.assertLess(case["checks"]["max_linear_boundary_abs_residual"], 1e-12)
            self.assertLess(case["checks"]["max_croots_abs_residual"], 1e-11)

    def test_inhomogeneous_stage_exports_numeric_croots_rules(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        expected_symbols = ["C11", "C12", "C13", "C21", "C22", "C23", "C31", "C32", "C33"]
        for case in payload["cases"]:
            rules = case["outputs"]["Croots"]
            self.assertEqual([rule["symbol"] for rule in rules], expected_symbols)
            for rule in rules:
                value = rule["value"]
                self.assertEqual(set(value), {"real", "imag"})
                self.assertTrue(math.isfinite(value["real"]))
                self.assertTrue(math.isfinite(value["imag"]))

    def test_inhomogeneous_stage_json_has_no_symbolic_fallbacks(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        forbidden = {"input_form", 'Missing["NonNumeric"]', "Indeterminate", "ComplexInfinity"}
        for value in _walk_values(payload):
            if isinstance(value, dict):
                self.assertNotIn("input_form", value)
            elif isinstance(value, str):
                self.assertNotIn(value, forbidden)


if __name__ == "__main__":
    unittest.main()
