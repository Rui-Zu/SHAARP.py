import json
import unittest
from pathlib import Path

from benchmarks.generate_shaarp_si_symbol_gap_audit import build_payload as build_gap_audit
from benchmarks.generate_shaarp_si_stage_symbol_sources import build_payload


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
SYMBOL_SOURCE_PATH = REF_DIR / "shaarp_si_stage_symbol_sources_v1.json"
GAP_AUDIT_PATH = REF_DIR / "shaarp_si_symbol_gap_audit_v1.json"


class SHAARPSIStageSymbolSourceTests(unittest.TestCase):
    def test_generated_symbol_source_file_matches_builder(self):
        expected = build_payload()
        actual = json.loads(SYMBOL_SOURCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(actual, expected)

    def test_symbol_source_map_tracks_active_and_commented_pnl_sources(self):
        payload = json.loads(SYMBOL_SOURCE_PATH.read_text(encoding="utf-8"))
        sources = payload["symbol_sources"]

        for symbol in ("P2eo", "P2o", "Peoo"):
            symbol_sources = sources[symbol]
            self.assertEqual(len(symbol_sources), 2, symbol)
            self.assertEqual(symbol_sources[0]["region_active_status"], "commented_reference_only")
            self.assertEqual(symbol_sources[1]["region_active_status"], "active_or_setup_code")
            self.assertEqual(symbol_sources[1]["region_id"], "active_complex_field_pnl_block")

    def test_symbol_source_map_tracks_active_fresnel_and_signal_assignments(self):
        payload = json.loads(SYMBOL_SOURCE_PATH.read_text(encoding="utf-8"))
        sources = payload["symbol_sources"]

        self.assertEqual(sources["EwrootsNum"][0]["region_id"], "omega_fresnel_equations_active_path")
        self.assertEqual(sources["ERpNum"][0]["region_id"], "omega_fresnel_substitution_active_path")
        self.assertEqual(sources["ERsNum"][0]["region_id"], "omega_fresnel_substitution_active_path")
        self.assertEqual(sources["Croots"][0]["region_id"], "inhomogeneous_particular_solution_coefficients")
        self.assertEqual(sources["E2wroots"][0]["region_id"], "two_omega_boundary_solution")
        self.assertEqual(sources["ThetaExt"][-1]["region_id"], "omega_angle_root_assignments_active_path")
        self.assertEqual(sources["ThetaOrd"][-1]["region_id"], "omega_angle_root_assignments_active_path")
        self.assertIn("fresnel_sweep_table_rows", {item["region_id"] for item in sources["Fresnel"]})
        self.assertEqual(len(sources["RSignalCrystal"]), 2)
        self.assertTrue(all(item["region_id"] == "reflected_signal_and_basic_intensity" for item in sources["RSignalCrystal"]))

    def test_missing_requested_symbols_are_exact_name_gaps_not_silent_assumptions(self):
        payload = json.loads(SYMBOL_SOURCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["missing_requested_symbols"],
            ["P2e", "Peo", "I2"],
        )
        self.assertNotIn("If", payload["symbol_sources"])
        all_assignments = {assignment["symbol"] for region in payload["regions"] for assignment in region["assignments"]}
        self.assertIn("dold", all_assignments)
        self.assertIn("dSHG", all_assignments)
        self.assertIn("I2wx", all_assignments)
        self.assertIn("I2wParallel", all_assignments)

    def test_gap_audit_file_matches_builder_and_classifies_remaining_gaps(self):
        expected = build_gap_audit()
        actual = json.loads(GAP_AUDIT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(actual, expected)
        records = {item["symbol"]: item for item in actual["records"]}
        self.assertEqual(set(records), {"P2e", "Peo", "I2"})
        self.assertEqual(records["P2e"]["classification"], "substring_only_of_active_pnl_symbols")
        self.assertEqual(records["Peo"]["classification"], "substring_only_of_active_pnl_symbols")
        self.assertEqual(records["I2"]["classification"], "represented_by_specific_intensity_functions_not_exact_symbol")
        self.assertEqual(records["P2e"]["exact_assignment_count"], 0)
        self.assertEqual(records["I2"]["exact_assignment_count"], 0)


if __name__ == "__main__":
    unittest.main()
