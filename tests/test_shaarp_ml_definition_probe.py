import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "shaarp_ml_definition_probe_v1.json"
SCRIPT_PATH = ROOT / "benchmarks" / "mathematica_reference" / "export_shaarp_ml_definition_probe.wl"


class SHAARPMLDefinitionProbeTests(unittest.TestCase):
    def test_definition_probe_preserves_input_cell_count_after_setup_clearall(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "definition_probe_exported")
        self.assertEqual(payload["input_cell_count"], 49)
        self.assertIn("solveSnell[", payload["solveSnell_definition"])
        self.assertIn("solveFresnelN[", payload["solveFresnelN_definition"])

    def test_export_script_stores_cell_count_outside_global_context(self):
        text = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("SHAARPReferenceLoader`inputCellCount = Length[cells]", text)
        self.assertIn('"input_cell_count" -> SHAARPReferenceLoader`inputCellCount', text)


if __name__ == "__main__":
    unittest.main()
