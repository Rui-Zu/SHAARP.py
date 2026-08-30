import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
REFERENCE_PATH = REF_DIR / "shaarp_si_pnl_stage_reference_v1.json"


class SHAARPSIPNLStageReferenceTests(unittest.TestCase):
    def test_pnl_stage_reference_is_exported_but_limited(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_stage_reference_exported")
        self.assertEqual(payload["validation_claim"], "pnl_stage_only_not_full_shaarp_agreement")
        self.assertEqual(payload["case_count"], 36)
        self.assertEqual(payload["numeric_boundary_residual_case_count"], 36)
        self.assertEqual(payload["numeric_pnl_case_count"], 36)
        self.assertIn("dold_policy", payload["policy"])
        self.assertIn("line-850-to-856", payload["policy"]["wave_constant_policy"])

    def test_pnl_stage_keeps_fresnel_boundary_solution_numeric(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertLess(payload["max_linear_boundary_abs_residual"], 1e-12)
        for case in payload["cases"]:
            self.assertEqual(case["parse_status"], "parsed")
            self.assertLess(case["checks"]["max_linear_boundary_abs_residual"], 1e-12)

    def test_pnl_stage_exports_d_tensor_and_rotated_dshg_shapes(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        for case in payload["cases"]:
            outputs = case["outputs"]
            for key in ("dold", "dSHG"):
                self.assertIn(key, outputs)
                self.assertEqual(len(outputs[key]), 3)
                for row in outputs[key]:
                    self.assertEqual(len(row), 6)
                    for component in row:
                        self.assertEqual(set(component), {"real", "imag"})
                        self.assertTrue(math.isfinite(component["real"]))
                        self.assertTrue(math.isfinite(component["imag"]))

    def test_pnl_stage_exports_all_three_complex_nonlinear_sources(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertGreater(payload["max_abs_pnl_component"], 0.0)
        for case in payload["cases"]:
            outputs = case["outputs"]
            for key in ("P2eo", "P2o", "Peoo"):
                self.assertIn(key, outputs)
                self.assertEqual(len(outputs[key]), 3)
                for component in outputs[key]:
                    self.assertEqual(set(component), {"real", "imag"})
                    self.assertTrue(math.isfinite(component["real"]))
                    self.assertTrue(math.isfinite(component["imag"]))

    def test_pnl_stage_preserves_fresnel_solve_rule_names(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        for case in payload["cases"]:
            rules = case["outputs"]["EwrootsNum"]
            self.assertEqual([rule["symbol"] for rule in rules], ["ERpNum", "ERsNum", "twe", "two"])


if __name__ == "__main__":
    unittest.main()
