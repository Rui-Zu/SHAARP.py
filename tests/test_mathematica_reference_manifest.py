import json
import unittest
from pathlib import Path

from benchmarks.generate_mathematica_reference_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmarks" / "mathematica_reference" / "reference_manifest_v1.json"


class MathematicaReferenceManifestTests(unittest.TestCase):
    def test_saved_manifest_matches_current_benchmarks(self):
        saved = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_manifest())

    def test_manifest_covers_all_current_benchmark_families(self):
        manifest = build_manifest()
        self.assertEqual(manifest["status"], "mathematica_reference_values_requested_not_yet_available")
        self.assertEqual(manifest["suite_count"], 10)
        self.assertEqual(
            {suite["suite_id"] for suite in manifest["suites"]},
            {
                "reduced_numeric_si_ml",
                "single_interface_solver_stage",
                "shaarp_si_stage_reference_request",
                "complex_inhomogeneous_field",
                "reflected_complex_snell",
                "biaxial_branch_sweeps",
                "multilayer_boundary_system",
                "multilayer_system_workflow",
                "multilayer_polarimetry_workflow",
                "maker_fringes_workflow",
            },
        )
        self.assertGreaterEqual(manifest["total_reference_items"], 176)
        self.assertEqual(manifest["coverage_snapshot"]["benchmark_counts"]["solver_stage_cases"], 36)

    def test_manifest_suite_ids_match_source_files(self):
        manifest = build_manifest()
        for suite in manifest["suites"]:
            source_path = ROOT / "benchmarks" / suite["source_benchmark_file"]
            source = json.loads(source_path.read_text(encoding="utf-8"))
            source_items = source[suite["source_key"]]
            self.assertEqual(suite["item_count"], len(source_items))
            self.assertEqual(suite["item_ids"], [item["id"] for item in source_items])
            self.assertGreater(len(suite["requested_outputs"]), 0)

    def test_manifest_requests_intermediate_solver_outputs_not_only_intensities(self):
        manifest = build_manifest()
        solver = next(suite for suite in manifest["suites"] if suite["suite_id"] == "single_interface_solver_stage")
        required = {
            "omega_transmitted_wavevectors",
            "omega_transmitted_electric_fields",
            "P2eo",
            "P2o",
            "Peoo",
            "inhomogeneous_ee_field",
            "two_omega_boundary_coefficients",
        }
        self.assertTrue(required.issubset(set(solver["requested_outputs"])))
        si_request = next(suite for suite in manifest["suites"] if suite["suite_id"] == "shaarp_si_stage_reference_request")
        self.assertIn("EwrootsNum", si_request["requested_outputs"])
        self.assertIn("E2wroots", si_request["requested_outputs"])
        self.assertIn("RSignalCrystal", si_request["requested_outputs"])
        self.assertIn("I2", si_request["requested_outputs"])
        multilayer = next(suite for suite in manifest["suites"] if suite["suite_id"] == "multilayer_system_workflow")
        self.assertIn("ten_source_polarizations", multilayer["requested_outputs"])
        self.assertIn("two_omega_boundary_coefficients", multilayer["requested_outputs"])
        boundary_system = next(suite for suite in manifest["suites"] if suite["suite_id"] == "multilayer_boundary_system")
        self.assertIn("solveFresnelN_matrix", boundary_system["requested_outputs"])
        self.assertIn("solveFresnelN_rhs", boundary_system["requested_outputs"])
        self.assertIn("equation_metadata", boundary_system["requested_outputs"])
        self.assertIn("unknown_labels", boundary_system["requested_outputs"])
        self.assertIn("unknown_metadata", boundary_system["requested_outputs"])
        self.assertIn("unknown_basis_waves", boundary_system["requested_outputs"])
        self.assertIn("known_waves", boundary_system["requested_outputs"])
        polarimetry = next(suite for suite in manifest["suites"] if suite["suite_id"] == "multilayer_polarimetry_workflow")
        self.assertIn("incident_jones_sp", polarimetry["requested_outputs"])
        self.assertIn("analyzer_jones_sp", polarimetry["requested_outputs"])
        self.assertIn("sample_azimuth_deg", polarimetry["requested_outputs"])
        self.assertIn("reflected_two_omega_jones_sp", polarimetry["requested_outputs"])
        self.assertIn("analyzer_amplitude", polarimetry["requested_outputs"])
        maker = next(suite for suite in manifest["suites"] if suite["suite_id"] == "maker_fringes_workflow")
        self.assertIn("MFList", maker["requested_outputs"])
        self.assertIn("listMFpara", maker["requested_outputs"])
        self.assertIn("listMFperp", maker["requested_outputs"])
        self.assertIn("transmitted_two_omega_jones_sp", maker["requested_outputs"])

    def test_manifest_lists_live_validated_reference_artifacts(self):
        manifest = build_manifest()
        artifacts = {item["path"]: item for item in manifest["validated_artifacts"]}

        expected = {
            "mathematica_reference/fresnel_sweep_reference_v1.json",
            "mathematica_reference/maker_fine_reference_v1.json",
            "mathematica_reference/sample_rotation_reference_v1.json",
            "mathematica_reference/solve_fresnel_n_reference_v1.json",
            "mathematica_reference/wolfram_live_symbolic_compute_pnl_reference_v1.json",
            "mathematica_reference/wolfram_live_symbolic_ml_partial_pnl_reference_v1.json",
        }
        self.assertTrue(expected.issubset(set(artifacts)))
        self.assertEqual(
            artifacts["mathematica_reference/wolfram_live_symbolic_ml_partial_pnl_reference_v1.json"][
                "validation_level"
            ],
            "symbolic_residual_and_complex_substitution_grid",
        )
        self.assertIn(
            "tests/test_wolfram_live_symbolic_ml_partial_pnl_reference.py",
            artifacts["mathematica_reference/wolfram_live_symbolic_ml_partial_pnl_reference_v1.json"]["tests"],
        )

    def test_manifest_lists_diagnostic_artifacts_separately_from_validated_references(self):
        manifest = build_manifest()
        diagnostics = {item["path"]: item for item in manifest["diagnostic_artifacts"]}
        self.assertEqual(
            diagnostics["mathematica_reference/wolfram_live_symbolic_point_group_d_probe_v1.json"][
                "validation_level"
            ],
            "diagnostic_only_not_point_group_tensor_validation",
        )

    def test_manifest_lists_source_reference_artifacts(self):
        manifest = build_manifest()
        source_references = {item["path"]: item for item in manifest["source_reference_artifacts"]}
        self.assertEqual(
            source_references["mathematica_reference/point_group_tensor_patterns_from_si_source_v1.json"][
                "validation_level"
            ],
            "source_text_pattern_extraction_not_live_symbolic_execution",
        )
        self.assertIn(
            "tests/test_point_group_tensor_source_reference.py",
            source_references["mathematica_reference/point_group_tensor_patterns_from_si_source_v1.json"]["tests"],
        )


if __name__ == "__main__":
    unittest.main()
