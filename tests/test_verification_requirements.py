import json
import unittest
from pathlib import Path

from benchmarks.generate_verification_requirements import build_manifest


MANIFEST_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "verification_requirements_v1.json"


class VerificationRequirementsTests(unittest.TestCase):
    def test_saved_manifest_matches_generator(self):
        saved = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_manifest())

    def test_manifest_keeps_user_acceptance_standard_explicit(self):
        manifest = build_manifest()
        standard = manifest["acceptance_standard"]
        self.assertFalse(manifest["full_end_to_end_agreement_proven"])
        self.assertTrue(standard["compare_values_not_patterns"])
        self.assertTrue(standard["strict_mathematica_source_required_for_validation_claims"])
        self.assertTrue(standard["document_discrepancies_instead_of_guessing"])
        self.assertTrue(standard["do_not_skip_complex_symbolic_cases_due_to_runtime"])

    def test_manifest_covers_requested_case_families(self):
        manifest = build_manifest()
        requirement_ids = {item["requirement_id"] for item in manifest["requirements"]}
        expected = {
            "value_level_mathematica_comparison",
            "at_least_20_complex_numerical_cases",
            "incidence_angle_sweeps",
            "fresnel_coefficient_sweep_plots",
            "maker_fringes_multilayer_angle_scan",
            "phase_matching_and_singular_limits",
            "all_complex_properties",
            "nondiagonal_dielectric_permittivity",
            "biaxial_and_branch_identity",
            "miller_indices_and_lattice_orientation",
            "multilayer_solve_fresnel_n_boundary",
            "symbolic_si_full_and_ml_partial",
            "shaarp_ml_gui_feature_parity",
        }
        self.assertEqual(requirement_ids, expected)

    def test_manifest_distinguishes_python_regression_from_mathematica_validation(self):
        manifest = build_manifest()
        requirements = {item["requirement_id"]: item for item in manifest["requirements"]}
        self.assertEqual(
            requirements["value_level_mathematica_comparison"]["status"],
            "partially_met_stage_level_only",
        )
        self.assertEqual(
            requirements["biaxial_and_branch_identity"]["validation_level"],
            "python_residual_and_regression",
        )
        self.assertEqual(
            requirements["miller_indices_and_lattice_orientation"]["validation_level"],
            "python_regression",
        )
        self.assertEqual(
            requirements["symbolic_si_full_and_ml_partial"]["validation_level"],
            "symbolic_comparator_scaffold",
        )

    def test_manifest_records_existing_complex_and_orientation_counts(self):
        manifest = build_manifest()
        requirements = {item["requirement_id"]: item for item in manifest["requirements"]}
        complex_cases = requirements["at_least_20_complex_numerical_cases"]["measured"]
        self.assertGreaterEqual(complex_cases["numeric_reduced_model_cases"], 20)
        self.assertGreaterEqual(complex_cases["solver_stage_cases"], 20)
        self.assertEqual(complex_cases["solve_inhom_mathematica_cases"], 20)
        self.assertTrue(complex_cases["complex_d_covered"])
        self.assertTrue(complex_cases["complex_epsilon_covered"])
        self.assertTrue(complex_cases["nondiagonal_epsilon_covered"])

        orientation = requirements["miller_indices_and_lattice_orientation"]["measured"]
        self.assertTrue(orientation["has_nonzero_cubic_miller_orientation"])
        self.assertTrue(orientation["has_noncubic_reciprocal_miller_orientation"])
        self.assertTrue(orientation["complex_orientation_matrix_rejected"])

    def test_manifest_keeps_pending_full_validation_next_actions(self):
        manifest = build_manifest()
        requirements = {item["requirement_id"]: item for item in manifest["requirements"]}
        self.assertIn("full SHAARP.si", requirements["value_level_mathematica_comparison"]["next_action"])
        self.assertIn("full SHAARP.ml", requirements["value_level_mathematica_comparison"]["next_action"])
        self.assertIn("orientation_reference_v1.json", requirements["miller_indices_and_lattice_orientation"]["next_action"])
        self.assertIn("solveFresnelN", requirements["multilayer_solve_fresnel_n_boundary"]["next_action"])
        self.assertIn("InputForm", requirements["symbolic_si_full_and_ml_partial"]["next_action"])
        self.assertIn("Fresnel", requirements["fresnel_coefficient_sweep_plots"]["next_action"])
        self.assertIn("Maker fringes", requirements["maker_fringes_multilayer_angle_scan"]["next_action"])
        self.assertIn("GUI audit", requirements["shaarp_ml_gui_feature_parity"]["next_action"])

    def test_manifest_tracks_fresnel_sweep_and_maker_fringes_validation_state(self):
        manifest = build_manifest()
        requirements = {item["requirement_id"]: item for item in manifest["requirements"]}
        self.assertEqual(
            requirements["fresnel_coefficient_sweep_plots"]["status"],
            "mathematica_value_validated_for_exported_gui_list_fresnel_cases",
        )
        self.assertEqual(
            requirements["fresnel_coefficient_sweep_plots"]["validation_level"],
            "mathematica_gui_list_fresnel_value_validated_with_transmitted_wave_policy_caveat",
        )
        self.assertEqual(
            requirements["maker_fringes_multilayer_angle_scan"]["validation_level"],
            "mathematica_mf_list_value_validated_with_phase_matching_diagnostic",
        )
        self.assertEqual(
            requirements["maker_fringes_multilayer_angle_scan"]["status"],
            "partially_mathematica_value_validated_not_full_ml_end_to_end",
        )
        self.assertTrue(
            requirements["fresnel_coefficient_sweep_plots"]["measured"][
                "first_class_fresnel_coefficient_plot_workflow_present"
            ]
        )
        self.assertTrue(
            requirements["fresnel_coefficient_sweep_plots"]["measured"][
                "fresnel_coefficient_sweep_api_present"
            ]
        )
        self.assertTrue(
            requirements["fresnel_coefficient_sweep_plots"]["measured"][
                "fresnel_list_mathematica_export_present"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_reference_export_present"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_sweep_api_present"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_copy_lists_present"
            ]
        )
        self.assertEqual(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_saved_python_case_count"
            ],
            3,
        )
        self.assertEqual(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_saved_python_total_angles"
            ],
            10,
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_has_negative_angle"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_has_normal_incidence"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_has_phase_matching_like_case"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_slow_path_small_grid"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_fine_sampling_plan_present"
            ]
        )
        self.assertGreaterEqual(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_fine_sampling_total_planned_angles"
            ],
            800,
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_fine_sampling_requires_confirm_slow"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_fine_sampling_not_routine_suite"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_agreement_requires_mathematica_mf_values"
            ]
        )
        self.assertEqual(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_required_mathematica_outputs"
            ],
            ["MFList", "listMFpara", "listMFperp"],
        )
        self.assertFalse(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_python_only_regression_is_not_agreement"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_comparator_present"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_manual_reference_builder_present"
            ]
        )
        self.assertTrue(
            requirements["maker_fringes_multilayer_angle_scan"]["measured"][
                "maker_fringes_fine_to_coarse_comparator_present"
            ]
        )

    def test_manifest_tracks_gui_feature_parity_audit(self):
        manifest = build_manifest()
        requirements = {item["requirement_id"]: item for item in manifest["requirements"]}
        gui = requirements["shaarp_ml_gui_feature_parity"]
        self.assertEqual(gui["status"], "merged_gui_implemented_complete")
        self.assertTrue(gui["measured"]["gui_audit_present"])
        self.assertEqual(gui["measured"]["gui_capabilities_audited"], 16)
        self.assertFalse(gui["measured"]["full_gui_feature_parity_claimed"])


if __name__ == "__main__":
    unittest.main()
