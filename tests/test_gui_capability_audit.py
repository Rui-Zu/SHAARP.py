import json
import unittest
from pathlib import Path

from benchmarks.generate_gui_capability_audit import build_audit


AUDIT_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "gui_capability_audit_v1.json"


class GUICapabilityAuditTests(unittest.TestCase):
    def test_saved_audit_matches_generator(self):
        saved = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_audit())

    def test_audit_does_not_claim_full_gui_feature_parity(self):
        audit = build_audit()
        self.assertEqual(audit["status"], "gui_capability_audit_not_full_port_claim")
        self.assertFalse(audit["full_gui_feature_parity_claimed"])
        self.assertIn("Github/SHAARP.ml/SHAARP.ml.nb", audit["source_notebook"])

    def test_audit_covers_major_gui_capabilities_from_notebook_and_docs(self):
        audit = build_audit()
        capability_ids = {item["capability_id"] for item in audit["capabilities"]}
        expected = {
            "gui_user_guide",
            "gui_set_material_properties",
            "gui_material_selection_blank_linear_nonlinear",
            "gui_layer_thickness_symbolic_h",
            "gui_crystal_structure_orientation",
            "gui_dielectric_tensor_inputs",
            "gui_shg_tensor_inputs_and_analytical_dij",
            "gui_layer_property_presets",
            "gui_shg_simulation_polarimetry",
            "gui_calculation_assumptions_full_jk_hh",
            "gui_fresnel_coefficient_plot",
            "gui_maker_fringes_plot",
            "gui_sample_rotation_scan",
            "gui_2d_3d_schematics",
            "gui_partial_analytical_expressions",
            "gui_copy_export_outputs",
        }
        self.assertEqual(capability_ids, expected)

    def test_audit_marks_gui_only_features_as_missing_or_partial(self):
        audit = build_audit()
        capabilities = {item["capability_id"]: item for item in audit["capabilities"]}
        # gui_fresnel_coefficient_plot: the plot now routes a MultilayerSystem through the
        # Mathematica-listFresnel-validated selected policy (see FresnelPlotMathematicaValidationTests).
        self.assertEqual(
            capabilities["gui_fresnel_coefficient_plot"]["python_status"],
            "implemented_multilayer_listfresnel_mathematica_validated",
        )
        self.assertEqual(
            capabilities["gui_fresnel_coefficient_plot"]["validation_status"],
            "multilayer_listfresnel_mathematica_validated_to_1e-10_with_transmitted_sum_caveat",
        )
        # gui_maker_fringes_plot: the plot now defaults to the SHAARP.ml unit normalization, so
        # plotted listMFpara/listMFperp match Mathematica MFList (see MakerPlotMathematicaValidationTests).
        self.assertEqual(
            capabilities["gui_maker_fringes_plot"]["python_status"],
            "implemented_mflist_mathematica_validated",
        )
        self.assertEqual(
            capabilities["gui_maker_fringes_plot"]["validation_status"],
            "plot_series_match_mathematica_mflist_to_1e-15_nonsingular_with_phase_matching_diagnostic",
        )
        # gui_2d_3d_schematics: 2D + 3D schematic figures are rendered by the merged GUI on each Run.
        self.assertEqual(
            capabilities["gui_2d_3d_schematics"]["python_status"],
            "implemented_2d_and_3d_figures_rendered_in_merged_gui",
        )
        # gui_layer_property_presets: headless registry + the merged GUI Preset 1-4 save/recall slots
        self.assertEqual(capabilities["gui_layer_property_presets"]["python_status"], "implemented_headless_plus_persistent_user_material_store")
        # symbolic-h is now implemented + validated through the FULL 2omega SHG stage
        # (solve_single_film_shg_symbolic_thickness: SHG coefficient as a closed form in h).
        self.assertEqual(
            capabilities["gui_layer_thickness_symbolic_h"]["python_status"],
            "per_layer_symbolic_h_implemented_and_validated_through_full_2omega_shg_stage",
        )

    def test_audit_tracks_critical_assumption_controls(self):
        audit = build_audit()
        capabilities = {item["capability_id"]: item for item in audit["capabilities"]}
        assumption = capabilities["gui_calculation_assumptions_full_jk_hh"]
        # JK/HH (mrassumption) now implemented + live-validated; was the critical physics gap.
        self.assertEqual(assumption["validation_status"], "live_wolfram_validated")
        self.assertIn("mrassumption", assumption["next_action"])
        self.assertIn("winhAssumption", assumption["next_action"])

    def test_audit_tracks_copy_export_payloads(self):
        audit = build_audit()
        capabilities = {item["capability_id"]: item for item in audit["capabilities"]}
        copy_export = capabilities["gui_copy_export_outputs"]
        self.assertIn("benchmarks/convert_wolfram_copied_output.py", copy_export["current_python_evidence"])
        self.assertIn("Copy/Export", copy_export["description"])
        self.assertIn("every GUI Copy/Export payload", copy_export["next_action"])


if __name__ == "__main__":
    unittest.main()
