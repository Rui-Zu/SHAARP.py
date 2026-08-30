import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "benchmarks" / "mathematica_reference" / "shaarp_si_cell_inventory.json"
SNIPPETS_PATH = ROOT / "benchmarks" / "mathematica_reference" / "shaarp_si_snippets.json"


class SHAARPSINotebookInventoryTests(unittest.TestCase):
    def test_si_cell_inventory_identifies_large_initialization_cell(self):
        payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))

        self.assertIn("SHAARP.si", payload["source"])
        self.assertEqual(payload["inputCellCount"], 8)
        matched = {item["index"]: item for item in payload["matchedCells"]}
        self.assertIn(4, matched)
        self.assertGreater(matched[4]["byte_count"], 1_000_000)
        for pattern in ("Analytical", "Full", "Partial", "SHG", "doldExp", "dnewExp", "P2e", "P2o", "Peo"):
            self.assertIn(pattern, matched[4]["patterns"])

    def test_si_snippets_include_symbolic_and_numeric_landmarks(self):
        payload = json.loads(SNIPPETS_PATH.read_text(encoding="utf-8"))

        self.assertIn("SHAARP.si", payload["source"])
        self.assertEqual(payload["cell_index"], 4)
        self.assertGreater(payload["cell_byte_count"], 1_000_000)
        patterns = {item["pattern"] for item in payload["snippets"]}
        for pattern in ("doldExp", "dnewExp", "P2e", "P2o", "Peo", "E2", "I2", "NSolve", "FindRoot", "Eigenvectors"):
            self.assertIn(pattern, patterns)


if __name__ == "__main__":
    unittest.main()
