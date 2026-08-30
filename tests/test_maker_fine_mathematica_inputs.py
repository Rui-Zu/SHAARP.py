import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.generate_maker_fine_mathematica_inputs import build_fine_input_payload, write_fine_input_payload


class MakerFineMathematicaInputsTests(unittest.TestCase):
    def test_fine_input_payload_uses_plan_case_ids_and_dense_theta_grid(self):
        payload = build_fine_input_payload(case_ids=["fine_maker_negative_normal_positive"])

        self.assertEqual(payload["status"], "python_fine_sampling_input_manifest_not_mathematica_validation")
        self.assertEqual(payload["case_count"], 1)
        case = payload["cases"][0]
        self.assertEqual(case["id"], "fine_maker_negative_normal_positive")
        self.assertEqual(case["source_case_id"], "maker_fringes_negative_normal_positive")
        self.assertEqual(len(case["theta_deg"]), 97)
        self.assertIn(0.0, case["theta_deg"])

    def test_fine_input_writer_creates_machine_readable_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "fine_inputs.json"

            payload = write_fine_input_payload(output, case_ids=["fine_maker_high_oblique"])

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertEqual(payload["cases"][0]["id"], "fine_maker_high_oblique")


if __name__ == "__main__":
    unittest.main()
