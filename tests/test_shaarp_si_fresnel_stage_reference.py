import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
REFERENCE_PATH = REF_DIR / "shaarp_si_fresnel_stage_reference_v1.json"


class SHAARPSIFresnelStageReferenceTests(unittest.TestCase):
    def test_fresnel_stage_reference_is_exported_but_limited(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_stage_reference_exported")
        self.assertEqual(payload["validation_claim"], "fresnel_stage_only_not_full_shaarp_agreement")
        self.assertEqual(payload["case_count"], 36)
        self.assertEqual(payload["numeric_residual_case_count"], 36)
        self.assertIn("wave_constant_policy", payload["policy"])
        self.assertIn("line-850-to-856", payload["policy"]["wave_constant_policy"])
        self.assertIn("Biaxial=True", payload["policy"]["anisotropy_branch_policy"])
        self.assertEqual(
            payload["region_ids"],
            [
                "principal_axis_and_index_setup",
                "active_wave_equation_setup",
                "active_backward_wave_equation_setup",
                "omega_angle_root_assignments_active_path",
                "two_omega_angle_root_assignments_active_path",
                "omega_angle_roots_active_numerical_path",
                "omega_transmitted_field_directions_active_path",
                "omega_incident_controls_active_path",
                "omega_fresnel_field_setup_active_path",
                "omega_fresnel_equations_active_path",
                "omega_fresnel_substitution_active_path",
            ],
        )

    def test_fresnel_stage_satisfies_linear_boundary_equations(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertLess(payload["max_linear_boundary_abs_residual"], 1e-12)
        for case in payload["cases"]:
            self.assertEqual(case["parse_status"], "parsed")
            self.assertLess(case["checks"]["max_linear_boundary_abs_residual"], 1e-12)

    def test_fresnel_solve_rules_preserve_symbol_names(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        for case in payload["cases"]:
            rules = case["outputs"]["EwrootsNum"]
            self.assertEqual([rule["symbol"] for rule in rules], ["ERpNum", "ERsNum", "twe", "two"])
            for rule in rules:
                value = rule["value"]
                self.assertEqual(set(value), {"real", "imag"})
                self.assertTrue(math.isfinite(value["real"]))
                self.assertTrue(math.isfinite(value["imag"]))

    def test_fresnel_stage_exports_expected_complex_vectors(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        vector_outputs = (
            "ETweo",
            "ETwo",
            "Vktweo",
            "Vktwo",
            "EIwNum",
            "ERwNum",
            "HIwNum",
            "HRwNum",
            "ETwNumExt",
            "ETwNumOrd",
            "HTwNumExt",
            "HTwNumOrd",
        )

        for case in payload["cases"]:
            outputs = case["outputs"]
            for key in vector_outputs:
                self.assertIn(key, outputs)
                self.assertEqual(len(outputs[key]), 3)
                for component in outputs[key]:
                    self.assertEqual(set(component), {"real", "imag"})
                    self.assertTrue(math.isfinite(component["real"]))
                    self.assertTrue(math.isfinite(component["imag"]))


if __name__ == "__main__":
    unittest.main()
