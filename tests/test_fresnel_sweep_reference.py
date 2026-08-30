import json
import unittest
from pathlib import Path


REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "mathematica_reference"
    / "fresnel_sweep_reference_v1.json"
)


class FresnelSweepReferenceTests(unittest.TestCase):
    def test_fresnel_sweep_reference_is_mathematica_gui_shaped(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertIn("Mathematica", payload["source"])
        self.assertEqual(payload["status"], "mathematica_reference_exported")
        self.assertEqual(payload["input_manifest_status"], "python_input_manifest_not_mathematica_validation")
        self.assertEqual(payload["case_count"], 3)
        self.assertGreaterEqual(payload["functionDownValues"]["Fresnel"], 1)
        suite = payload["suites"][0]
        self.assertEqual(suite["suite_id"], "fresnel_sweep_gui_workflow")
        self.assertEqual(len(suite["items"]), 3)

        for case in suite["items"]:
            with self.subTest(case=case["id"]):
                outputs = case["mathematica_outputs"]
                self.assertEqual(outputs["listFresnel_labels"], ["Rp", "Rs", "Tp", "Ts"])
                self.assertEqual(len(outputs["listFresnel"]), 4)
                self.assertEqual(len(outputs["listFresnel"][0]), len(case["inputs"]["theta_deg"]))
                self.assertEqual(outputs["listFresnel"][0][0][0]["real"], case["inputs"]["theta_deg"][0])


if __name__ == "__main__":
    unittest.main()
