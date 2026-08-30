import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "benchmarks" / "wolfram_batch_status_v1.json"


class WolframBatchStatusTests(unittest.TestCase):
    def test_wolfram_batch_status_is_machine_readable_and_file_export_confirmed(self):
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))

        self.assertEqual(status["schema_version"], 1)
        self.assertEqual(status["status"], "batch_payload_confirmed_by_file_export")
        self.assertTrue(status["batch_execution_confirmed"])
        self.assertTrue(status["usable_kernel_found"])
        self.assertGreaterEqual(len(status["kernel_candidates"]), 1)
        self.assertGreaterEqual(len(status["attempts"]), 1)
        self.assertIn("result", status["proof_payload"])
        self.assertEqual(status["proof_payload"]["result"], 4)
        self.assertIn("Use the script strategy", status["next_action"])


if __name__ == "__main__":
    unittest.main()
