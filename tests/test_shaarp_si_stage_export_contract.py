import json
import unittest
from pathlib import Path

from benchmarks.generate_shaarp_si_stage_export_contract import build_contract


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
CONTRACT_PATH = REF_DIR / "shaarp_si_stage_export_contract_v1.json"
TEMPLATE_OUTPUT_PATH = REF_DIR / "shaarp_si_stage_reference_values_v1.json"
WOLFRAM_TEMPLATE_PATH = REF_DIR / "export_shaarp_si_stage_reference_template.wl"


class SHAARPSIStageExportContractTests(unittest.TestCase):
    def test_saved_contract_matches_generator(self):
        expected = build_contract()
        actual = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(actual, expected)

    def test_contract_is_not_validation_evidence(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(contract["status"], "export_contract_not_validation_evidence")
        self.assertFalse(contract["validation_claim"])
        self.assertEqual(
            contract["dependency_audit_file"],
            "benchmarks/mathematica_reference/shaarp_si_active_route_dependency_audit_v1.json",
        )
        self.assertEqual(
            contract["case_input_mapping_file"],
            "benchmarks/mathematica_reference/shaarp_si_case_input_mapping_v1.json",
        )
        self.assertEqual(
            contract["case_input_mapping_preflight_file"],
            "benchmarks/mathematica_reference/shaarp_si_case_input_mapping_preflight_v1.json",
        )
        self.assertFalse(contract["export_status"]["mathematica_values_exported"])
        self.assertFalse(contract["export_status"]["safe_to_claim_shaarp_agreement"])

    def test_contract_separates_active_fresnel_pnl_and_unresolved_symbols(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        active = contract["exact_active_requested_symbols"]
        unresolved = {item["symbol"]: item for item in contract["unresolved_exact_requested_symbols"]}

        self.assertIn("omega_angle_root_assignments_active_path", contract["active_numeric_region_order"])
        self.assertIn("active_wave_equation_setup", contract["active_numeric_region_order"])
        self.assertIn("two_omega_angle_root_assignments_active_path", contract["active_numeric_region_order"])
        self.assertIn("shg_tensor_transformation_active_path", contract["active_numeric_region_order"])
        self.assertIn("fresnel_sweep_table_rows", contract["fresnel_sweep_region_order"])
        self.assertIn("ThetaExt", active)
        self.assertIn("Fresnel", active)
        self.assertEqual(set(unresolved), {"P2e", "Peo", "I2"})
        self.assertEqual(unresolved["P2e"]["policy"], "do_not_alias_to_P2eo_or_Peoo_without_notebook_evidence")
        self.assertEqual(unresolved["I2"]["policy"], "export_specific_intensity_functions_until_exact_I2_assignment_is_found")

    def test_wolfram_template_output_is_clearly_not_reference_values(self):
        self.assertTrue(WOLFRAM_TEMPLATE_PATH.exists())
        payload = json.loads(TEMPLATE_OUTPUT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_reference_export_template_not_values")
        self.assertFalse(payload["validation_claim"])
        self.assertEqual(payload["case_count"], 36)
        self.assertEqual(payload["cases"], [])
        self.assertEqual(payload["unresolved_symbols"], ["P2e", "Peo", "I2"])


if __name__ == "__main__":
    unittest.main()
