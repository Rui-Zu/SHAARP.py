"""Lock in that the public API facades advertise the CURRENT live-Wolfram
multilayer validated scope (ext1 + ml1/ml2/ml3), not just the original
single-film v1 reference. This is the modular_user_api 'promote validated
workflows' guarantee added."""

import unittest

import shaarp.api as api


class ApiMultilayerValidationMetadataTests(unittest.TestCase):
    def test_maker_artifacts_include_all_multilayer_breadth_summaries(self):
        artifacts = api._load_maker_validation_artifacts()
        breadth = artifacts.get("multilayer_breadth_live_wolfram_summaries", {})
        for key in (
            "single_film_incidence_ext1",
            "multilayer_4layer_ml1",
            "multilayer_5layer_anisotropic_ml2",
            "multilayer_two_active_layer_ml3",
        ):
            self.assertIn(key, breadth, msg=f"missing breadth summary {key}")
            entry = breadth[key]
            self.assertEqual(entry["status"], "maker_outputs_match_all_compared_cases")
            self.assertTrue(entry["full_shaarp_ml_agreement_claimed"])
            self.assertEqual(entry["nonsingular_fail_count"], 0)

    def test_sample_rotation_artifacts_include_extended_azimuth_grid(self):
        artifacts = api._load_sample_rotation_validation_artifacts()
        ext1 = artifacts.get("extended_azimuth_grid_ext1")
        self.assertIsNotNone(ext1, "extended_azimuth_grid_ext1 summary missing")
        self.assertEqual(ext1["status"], "sample_rotation_outputs_match")
        self.assertTrue(ext1["full_sample_rotation_agreement_claimed"])
        self.assertEqual(ext1["nonsingular_case_count"], 12)
        self.assertEqual(ext1["nonsingular_fail_count"], 0)

    def test_optional_loader_returns_none_for_missing_file(self):
        self.assertIsNone(api._load_project_json_optional("benchmarks/does_not_exist_xyz.json"))


if __name__ == "__main__":
    unittest.main()
