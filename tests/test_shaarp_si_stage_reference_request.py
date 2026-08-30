import json
import unittest
from pathlib import Path

from benchmarks.generate_shaarp_si_stage_reference_request import build_request


ROOT = Path(__file__).resolve().parents[1]
REQUEST_PATH = ROOT / "benchmarks" / "mathematica_reference" / "shaarp_si_stage_reference_request_v1.json"


class SHAARPSIStageReferenceRequestTests(unittest.TestCase):
    def test_saved_request_matches_generator(self):
        payload = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload, build_request())

    def test_request_is_explicitly_not_validation_evidence(self):
        payload = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "reference_values_requested_not_exported")
        self.assertFalse(payload["validation_claim"])
        self.assertIn("Agreement requires value-by-value comparison", " ".join(payload["notes"]))

    def test_request_covers_complex_nondiagonal_angle_orientation_batch(self):
        payload = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
        coverage = payload["coverage_summary"]

        self.assertEqual(payload["case_count"], 36)
        self.assertTrue(coverage["has_normal_incidence"])
        self.assertGreaterEqual(coverage["max_angle_deg"], 55.0)
        self.assertGreaterEqual(coverage["orientation_count"], 6)
        self.assertTrue(coverage["has_s_polarization"])
        self.assertTrue(coverage["has_p_polarization"])
        self.assertTrue(coverage["all_cases_have_complex_epsilon"])
        self.assertTrue(coverage["all_cases_have_complex_2omega_epsilon"])
        self.assertTrue(coverage["all_cases_have_nondiagonal_epsilon"])
        self.assertTrue(coverage["all_cases_have_complex_d"])

    def test_request_asks_for_stage_symbols_from_si_notebook_audit(self):
        payload = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
        symbols = set(payload["requested_mathematica_symbols"])

        for symbol in (
            "ThetaExt",
            "ThetaOrd",
            "EwrootsNum",
            "E2wroots",
            "ERpNum",
            "ERsNum",
            "P2e",
            "P2o",
            "Peo",
            "RSignalCrystal",
            "I2",
            "Fresnel",
        ):
            self.assertIn(symbol, symbols)

    def test_request_links_refined_region_and_symbol_source_evidence(self):
        payload = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
        evidence = set(payload["extraction_evidence"])

        for path in (
            "benchmarks/mathematica_reference/shaarp_si_refined_extraction_map_v1.json",
            "benchmarks/mathematica_reference/shaarp_si_refined_region_texts_v1.json",
            "benchmarks/mathematica_reference/shaarp_si_stage_symbol_sources_v1.json",
            "benchmarks/mathematica_reference/shaarp_si_symbol_gap_audit_v1.json",
        ):
            self.assertIn(path, evidence)

    def test_request_records_exact_symbol_resolution(self):
        payload = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
        resolution = payload["requested_symbol_resolution"]

        self.assertEqual(resolution["ThetaExt"]["status"], "mapped_to_refined_region_sources")
        self.assertIn("omega_angle_root_assignments_active_path", resolution["ThetaExt"]["regions"])
        self.assertEqual(resolution["Fresnel"]["status"], "mapped_to_refined_region_sources")
        self.assertIn("fresnel_sweep_table_rows", resolution["Fresnel"]["regions"])
        self.assertEqual(resolution["P2e"]["status"], "not_mapped_to_exact_refined_assignment")
        self.assertEqual(resolution["P2e"]["classification"], "substring_only_of_active_pnl_symbols")
        self.assertEqual(resolution["Peo"]["classification"], "substring_only_of_active_pnl_symbols")
        self.assertEqual(resolution["I2"]["classification"], "represented_by_specific_intensity_functions_not_exact_symbol")

    def test_request_keeps_follow_up_case_families_for_uncovered_extremes(self):
        payload = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
        family_ids = {item["family_id"] for item in payload["required_follow_up_case_families"]}

        for family_id in (
            "trivial_real_isotropic_normal_incidence",
            "simple_uniaxial_ordinary_extraordinary_identity",
            "phase_matching_like_epsilon_omega_equals_epsilon_2omega",
            "biaxial_branch_ambiguity",
            "nontrivial_reciprocal_miller_all_indices_nonzero",
        ):
            self.assertIn(family_id, family_ids)


if __name__ == "__main__":
    unittest.main()
