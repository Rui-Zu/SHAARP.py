import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"


class SHAARPSILoaderProbeTests(unittest.TestCase):
    def test_import_element_probe_identifies_available_notebook_exports(self):
        payload = json.loads((REF_DIR / "shaarp_si_import_elements.json").read_text(encoding="utf-8"))

        self.assertIn("SHAARP.si", payload["source"])
        self.assertIn("Plaintext", payload["elements"])

        probes = {item["element"]: item for item in payload["candidate_probes"]}
        self.assertEqual(probes["NB"]["head"], "Notebook")
        self.assertEqual(probes["Notebook"]["head"], "Notebook")
        self.assertEqual(probes["Plaintext"]["head"], "String")
        self.assertEqual(probes["InputText"]["head"], "$Failed")
        self.assertEqual(probes["Input"]["head"], "$Failed")

    def test_boxdata_smoke_loader_is_not_full_si_ingestion(self):
        payload = json.loads((REF_DIR / "load_shaarp_si_smoke_output.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_smoke_loaded")
        self.assertEqual(payload["inputCellCount"], 8)
        self.assertEqual(payload["matchedNames"], ["FullAnalyticalFlag"])
        self.assertNotIn("I2", payload["downValueCounts"])
        self.assertNotIn("E2", payload["downValueCounts"])

    def test_plaintext_parse_probe_records_current_failure(self):
        payload = json.loads((REF_DIR / "shaarp_si_plaintext_parse_probe.json").read_text(encoding="utf-8"))

        self.assertIn("SHAARP.si", payload["source"])
        self.assertGreater(payload["plain_byte_count"], 300_000)
        self.assertEqual(payload["parse_status"], "failed")
        self.assertGreater(payload["message_count"], 0)

    def test_initialization_element_probe_records_dead_end(self):
        diagnostics = (REF_DIR / "shaarp_si_initialization_element_diagnostics.txt").read_text(encoding="utf-8")

        self.assertIn("starting initialization import probe", diagnostics)
        self.assertFalse((REF_DIR / "shaarp_si_initialization_element.json").exists())

    def test_frontend_text_conversion_succeeds_but_is_not_parseable_yet(self):
        payload = json.loads((REF_DIR / "shaarp_si_cell4_frontend_text_probe.json").read_text(encoding="utf-8"))

        self.assertIn("FrontEndExecute", payload["source"])
        self.assertEqual(payload["inputCellCount"], 8)
        results = {item["format"]: item for item in payload["results"]}
        self.assertEqual(results["InputText"]["head"], "List")
        self.assertEqual(results["PlainText"]["head"], "List")
        self.assertGreater(results["InputText"]["byte_count"], 450_000)
        self.assertGreater(results["PlainText"]["byte_count"], 300_000)

        parse_payload = json.loads(
            (REF_DIR / "shaarp_si_cell4_frontend_inputtext_parse.json").read_text(encoding="utf-8")
        )
        parse_results = {item["format"]: item for item in parse_payload["results"]}
        self.assertEqual(parse_results["InputText"]["parse_status"], "failed")
        self.assertEqual(parse_results["PlainText"]["parse_status"], "failed")
        self.assertIn("k^(R", " ".join(parse_results["PlainText"]["messages_preview"]))

    def test_sanitized_frontend_text_probe_moves_past_notation_block_only(self):
        payload = json.loads(
            (REF_DIR / "shaarp_si_cell4_sanitized_frontend_text_parse.json").read_text(encoding="utf-8")
        )

        self.assertEqual(payload["sanitization"], "remove_initial_Symbolize_declaration_block_before_LogoExists")
        results = {item["format"]: item for item in payload["results"]}
        self.assertGreater(results["PlainText"]["removedByteCount"], 4_000)
        self.assertEqual(results["PlainText"]["parse_status"], "failed")
        self.assertIn("kROmega", " ".join(results["PlainText"]["messages_preview"]))

    def test_frontend_sections_capture_si_numerical_and_analytical_landmarks(self):
        payload = json.loads((REF_DIR / "shaarp_si_frontend_sections.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["cell_index"], 4)
        summaries = {item["format"]: item for item in payload["summaries"]}
        plain_counts = summaries["PlainText"]["pattern_counts"]
        for pattern in (
            "SHG Simulations",
            "Partial Analytical Expressions",
            "Full Analytical Expressions",
            "Use Numerical Solution with Fresnel",
            "RSignalCrystal",
            "NSolve",
            "FindRoot",
            "Eigenvectors",
            'Export["Full Analytical Expressions.nb"',
        ):
            self.assertGreater(plain_counts[pattern], 0, pattern)

        records = payload["records"]
        self.assertTrue(any(item["format"] == "PlainText" and item["pattern"] == "SHG Simulations" for item in records))
        self.assertTrue(
            any(item["format"] == "PlainText" and item["pattern"] == 'Export["Full Analytical Expressions.nb"' for item in records)
        )

    def test_extraction_plan_is_candidate_only_with_bounded_numerical_ranges(self):
        payload = json.loads((REF_DIR / "shaarp_si_extraction_plan_v1.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "candidate_extraction_plan_not_validation_evidence")
        self.assertIn("must not be treated", payload["rule"])

        plain_regions = {item["region_id"]: item for item in payload["formats"]["PlainText"]["candidate_regions"]}
        numerical = plain_regions["si_numerical_shg_compute_candidate"]
        partial = plain_regions["si_partial_analytical_compute_candidate"]
        full = plain_regions["si_full_analytical_export_candidate"]

        self.assertEqual(numerical["status"], "candidate_not_validated")
        self.assertLess(numerical["start"], numerical["end"])
        self.assertGreater(numerical["byte_count"], 50_000)
        self.assertLess(partial["start"], partial["end"])
        self.assertIsNotNone(full["start"])
        self.assertIsNone(full["end"])

    def test_candidate_region_parse_probe_records_non_standalone_regions(self):
        payload = json.loads((REF_DIR / "shaarp_si_candidate_region_parse.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "parse_probe_not_validation_evidence")
        records = {(item["format"], item["region_id"]): item for item in payload["records"]}

        numerical = records[("PlainText", "si_numerical_shg_compute_candidate")]
        partial = records[("PlainText", "si_partial_analytical_compute_candidate")]
        full = records[("PlainText", "si_full_analytical_export_candidate")]

        self.assertEqual(numerical["parse_status"], "failed")
        self.assertIn("SHG Simulations", " ".join(numerical["messages_preview"]))
        self.assertEqual(partial["parse_status"], "failed")
        self.assertEqual(full["parse_status"], "skipped")

    def test_candidate_region_symbol_audit_locates_core_si_stage_terms(self):
        payload = json.loads((REF_DIR / "shaarp_si_candidate_region_symbol_audit.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "symbol_audit_not_validation_evidence")
        records = {(item["format"], item["region_id"]): item for item in payload["records"]}
        numerical = records[("PlainText", "si_numerical_shg_compute_candidate")]["pattern_records"]
        partial = records[("PlainText", "si_partial_analytical_compute_candidate")]["pattern_records"]

        for pattern in ("EwrootsNum", "E2wroots", "ERpNum", "ERsNum", "P2e", "P2o", "Peo", "RSignalCrystal", "Fresnel"):
            self.assertGreater(numerical[pattern]["count"], 0, pattern)
        self.assertGreater(numerical["NSolve"]["count"], 0)
        self.assertGreater(numerical["FindRoot"]["count"], 0)
        self.assertGreater(numerical["Eigenvectors"]["count"], 0)

        for pattern in ("E2wroots", "P2e", "P2o", "Peo", "RSignalCrystal"):
            self.assertGreater(partial[pattern]["count"], 0, pattern)

    def test_numerical_stage_snippets_cover_core_export_targets(self):
        payload = json.loads((REF_DIR / "shaarp_si_numerical_stage_snippets.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "targeting_snippets_not_validation_evidence")
        self.assertEqual(payload["region"]["region_id"], "si_numerical_shg_compute_candidate")
        patterns = {item["pattern"] for item in payload["records"]}
        for pattern in (
            "EwrootsNum",
            "ThetaExt",
            "ThetaOrd",
            "ERpNum",
            "ERsNum",
            "P2e",
            "P2o",
            "Peo",
            "E2wroots",
            "ER2w",
            "RSignalCrystal",
            "I2",
            "Use Numerical Solution with Fresnel",
        ):
            self.assertIn(pattern, patterns)

    def test_line_audit_and_refined_map_use_normalized_coordinates(self):
        audit = json.loads((REF_DIR / "shaarp_si_candidate_region_line_audit.json").read_text(encoding="utf-8"))
        refined = json.loads((REF_DIR / "shaarp_si_refined_extraction_map_v1.json").read_text(encoding="utf-8"))

        self.assertEqual(audit["coordinate_system"], "PlainText normalized to LF newlines")
        self.assertEqual(refined["coordinate_system"], "PlainText normalized to LF newlines")
        numerical = next(item for item in audit["records"] if item["region_id"] == "si_numerical_shg_compute_candidate")
        self.assertGreater(numerical["line_count"], 1000)
        self.assertGreater(len(numerical["target_window_lines"]), 700)

        regions = {item["region_id"]: item for item in refined["regions"]}
        self.assertEqual(len(regions), 23)
        self.assertEqual(regions["shg_tensor_input_matrix"]["line_start"], 309)
        self.assertEqual(regions["active_complex_field_pnl_block"]["line_start"], 790)
        self.assertEqual(regions["active_complex_field_pnl_block"]["line_end"], 810)
        self.assertEqual(regions["fresnel_sweep_table_initialization"]["line_start"], 318)
        self.assertEqual(regions["fresnel_sweep_table_rows"]["line_start"], 449)
        self.assertEqual(regions["omega_angle_root_assignments_active_path"]["line_start"], 508)
        self.assertEqual(regions["omega_transmitted_field_directions_active_path"]["line_start"], 576)
        self.assertEqual(regions["omega_incident_controls_active_path"]["line_start"], 697)
        self.assertEqual(regions["omega_fresnel_field_setup_active_path"]["line_start"], 720)
        self.assertEqual(regions["omega_fresnel_equations_active_path"]["line_start"], 732)
        self.assertEqual(regions["omega_fresnel_substitution_active_path"]["line_start"], 740)
        self.assertEqual(regions["shg_tensor_transformation_active_path"]["line_start"], 759)
        self.assertIn("comment", " ".join(refined["notes"]).lower())

    def test_refined_region_text_snippets_cover_all_mapped_regions(self):
        refined = json.loads((REF_DIR / "shaarp_si_refined_extraction_map_v1.json").read_text(encoding="utf-8"))
        snippets = json.loads((REF_DIR / "shaarp_si_refined_region_texts_v1.json").read_text(encoding="utf-8"))

        self.assertEqual(snippets["status"], "snippet_export_not_validation_evidence")
        self.assertEqual(snippets["coordinate_system"], "PlainText normalized to LF newlines")
        mapped_ids = {item["region_id"] for item in refined["regions"]}
        snippet_records = {item["region_id"]: item for item in snippets["records"]}
        self.assertEqual(set(snippet_records), mapped_ids)
        self.assertEqual(len(snippet_records), 23)
        self.assertIn("Complex Field of E", snippet_records["active_complex_field_pnl_block"]["text"])
        self.assertIn("dSHG", snippet_records["shg_tensor_transformation_active_path"]["text"])
        self.assertIn("Fresnel", snippet_records["fresnel_sweep_table_rows"]["text"])
        self.assertIn("ThetaOrd", snippet_records["omega_angle_root_assignments_active_path"]["text"])
        self.assertIn("ETwNumExt", snippet_records["omega_transmitted_field_directions_active_path"]["text"])
        self.assertIn("RHTilt", snippet_records["omega_incident_controls_active_path"]["text"])
        self.assertIn("AwNum", snippet_records["active_wave_equation_setup"]["text"])
        self.assertIn("ThetaExt2w", snippet_records["two_omega_angle_root_assignments_active_path"]["text"])
        for record in snippet_records.values():
            self.assertEqual(record["status"], "snippet_ready", record["region_id"])
            self.assertFalse(record["missing_lines"], record["region_id"])
            self.assertGreater(record["byte_count"], 0, record["region_id"])

    def test_refined_region_parse_probe_identifies_parseable_and_unparseable_blocks(self):
        payload = json.loads((REF_DIR / "shaarp_si_refined_region_parse.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "parse_probe_not_validation_evidence")
        self.assertEqual(
            payload["parser_route"],
            "parse audited region snippets; no notebook front-end conversion in this script",
        )
        records = {item["region_id"]: item for item in payload["records"]}
        self.assertEqual(len(records), 23)
        for region_id in (
            "principal_axis_and_index_setup",
            "shg_tensor_input_matrix",
            "fresnel_sweep_table_initialization",
            "fresnel_sweep_angle_root_assignments",
            "omega_angle_roots_initial_path",
            "fresnel_sweep_table_rows",
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
            "shg_tensor_transformation_active_path",
            "commented_real_field_pnl_block",
            "active_complex_field_pnl_block",
            "inhomogeneous_particular_solution_coefficients",
            "numerical_coefficients_and_two_omega_field_setup",
            "two_omega_boundary_solution",
            "reflected_signal_and_basic_intensity",
        ):
            self.assertEqual(records[region_id]["parse_status"], "parsed", region_id)


if __name__ == "__main__":
    unittest.main()
