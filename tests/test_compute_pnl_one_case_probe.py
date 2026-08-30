import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.nonlinear import compute_pnl_voigt


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "compute_pnl_one_case_probe_v1.json"
SCRIPT_PATH = ROOT / "benchmarks" / "mathematica_reference" / "export_compute_pnl_one_case_probe.wl"


class ComputePNLOneCaseProbeTests(unittest.TestCase):
    def test_probe_script_preserves_input_cell_count_outside_global_context(self):
        text = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("SHAARPReferenceLoader`inputCellCount = Length[cells]", text)
        self.assertIn('"inputCellCount" -> SHAARPReferenceLoader`inputCellCount', text)
        # The export target resolves from whichever checkout runs the script, rather than being
        # pinned to one machine's directory layout.
        self.assertIn(
            'Export[\n  SHAARPPaths`Ref["compute_pnl_one_case_probe_v1.json"]',
            text,
        )
        self.assertNotIn("outputPath", text)

    def test_live_one_case_compute_pnl_probe_matches_python(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_reference_exported")
        self.assertEqual(payload["batch_execution"], "file_export_confirmed")
        self.assertEqual(payload["inputCellCount"], 49)
        self.assertEqual(payload["functionDownValues"]["computePNL"], 1)
        self.assertEqual(len(payload["cases"]), 1)

        case = payload["cases"][0]
        inputs = case["inputs"]
        d = numeric_array(inputs["d_voigt_lab"])
        e1 = numeric_array(inputs["e1"])
        e2 = numeric_array(inputs["e2"])
        factor = numeric_array(inputs["factor"]).item()
        expected = numeric_array(case["outputs"]["pnl"])

        actual = compute_pnl_voigt(d, e1, e2, factor=factor)

        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
