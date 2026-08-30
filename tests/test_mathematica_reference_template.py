import json
import unittest
from pathlib import Path

from benchmarks.generate_mathematica_reference_template import build_template


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmarks" / "mathematica_reference" / "reference_manifest_v1.json"
TEMPLATE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "reference_template_v1.json"


class MathematicaReferenceTemplateTests(unittest.TestCase):
    def test_saved_template_matches_manifest(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        saved = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_template(manifest))

    def test_template_covers_all_manifest_suites_and_items(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        template = build_template(manifest)
        self.assertEqual(template["suite_count"], manifest["suite_count"])
        self.assertEqual(template["total_items"], manifest["total_reference_items"])
        manifest_suites = {suite["suite_id"]: suite for suite in manifest["suites"]}
        for suite in template["suites"]:
            manifest_suite = manifest_suites[suite["suite_id"]]
            self.assertEqual(suite["item_count"], manifest_suite["item_count"])
            self.assertEqual(suite["requested_outputs"], manifest_suite["requested_outputs"])
            self.assertEqual([item["id"] for item in suite["items"]], manifest_suite["item_ids"])

    def test_template_outputs_are_placeholders_with_requested_keys(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        template = build_template(manifest)
        for suite in template["suites"]:
            requested = set(suite["requested_outputs"])
            for item in suite["items"]:
                self.assertEqual(set(item["mathematica_outputs"]), requested)
                self.assertTrue(all(value is None for value in item["mathematica_outputs"].values()))
                self.assertIn("inputs", item)

    def test_template_includes_solver_stage_intermediate_placeholders(self):
        template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        solver = next(suite for suite in template["suites"] if suite["suite_id"] == "single_interface_solver_stage")
        first = solver["items"][0]
        self.assertIn("epsilon_omega_lab", first["inputs"])
        self.assertIn("P2eo", first["mathematica_outputs"])
        self.assertIn("inhomogeneous_ee_field", first["mathematica_outputs"])
        self.assertIn("two_omega_boundary_coefficients", first["mathematica_outputs"])


if __name__ == "__main__":
    unittest.main()
