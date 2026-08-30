import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_sample_rotation_reference import (
    build_sample_rotation_summary,
    compare_sample_rotation_payloads,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "sample_rotation_reference_ml4.json"
PYTHON_PATH = ROOT / "benchmarks" / "sample_rotation_benchmarks_ml4.json"

ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 4


@unittest.skipUnless(REFERENCE_PATH.exists(), "ml4 live Wolfram reference not present")
class SampleRotationMultilayerML4ReferenceComparisonTests(unittest.TestCase):
    """VALIDATED: azimuth-rotated MULTILAYER SHG. A 4-layer stack
    (air / active-film / isotropic-interlayer / substrate) swept over SAMPLE AZIMUTH
    via the SampleRotate path now matches live Wolfram SHAARP.ml to ~1e-15 for every
    rotation step in all 4 cases.

    Previously OPEN: the live Wolfram SampleRotate export returned EXACTLY ZERO for
    every rotated step (az 22.5/45 deg) on multilayer stacks while Python returned
    finite, continuous values. ROOT CAUSE (proven by an isolating Wolfram
    experiment): the export harness rotated EVERY internal layer and re-ran extMater
    on each; re-extending a rotated PASSIVE ISOTROPIC interlayer produces a
    degenerate eigenmode basis -> singular f4NL boundary solve -> the whole rotated
    multilayer stack zeroed. The single-layer case (ext1) was unaffected because it
    only rotates the active film. A fresh-rebuild that still rotated the interlayer
    also zeroed; rotating ONLY SHG-active layers (rotating an isotropic layer is
    physically a no-op) makes the live SHAARP.ml reference reproduce Python to
    ~1e-15. The fix is in export_sample_rotation_reference_ml4.wl (rotate-active-only
    guard); Python was correct all along and was NOT changed.
    See the project notes ' ml4 RESOLVED'."""

    def setUp(self):
        self.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

    def test_reference_is_live_wolfram_sourced_4_layer(self):
        source = str(self.reference.get("source", "")).lower()
        self.assertTrue("mathematica" in source or "wolfram" in source)
        self.assertEqual(self.reference.get("case_count"), EXPECTED_CASE_COUNT)
        self.assertEqual(self.reference["functionDownValues"]["SampleRotate"], 1)
        self.assertEqual(self.reference["functionDownValues"]["f4NL"], 1)
        for case in self.python["cases"]:
            self.assertEqual(case["layer_count"], 4)

    def test_rotated_multilayer_matches_live_mathematica_all_steps(self):
        """Every azimuth step (including the rotated 22.5/45 deg steps) now agrees
        with live Wolfram SHAARP.ml across all 4 multilayer cases."""
        results = compare_sample_rotation_payloads(self.reference, self.python, atol=ATOL, rtol=RTOL)
        summary = build_sample_rotation_summary(results, self.reference, atol=ATOL, rtol=RTOL)
        self.assertEqual(summary["status"], "sample_rotation_outputs_match")
        self.assertTrue(summary["full_sample_rotation_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["nonsingular_fail_count"], 0)

    def test_rotated_steps_are_nonzero_finite(self):
        """Regression guard against the old export-harness bug: the rotated-step
        rows must be finite and non-zero (they were exactly 0.0 before the fix)."""
        items = {it["id"]: it for s in self.reference["suites"] for it in s["items"]}
        for cid in (c["id"] for c in self.python["cases"]):
            ref_rows = items[cid]["mathematica_outputs"]["sampleRotateList"]
            for row in ref_rows[1:]:  # skip az=0
                vals = np.array(
                    [(x["real"] if isinstance(x, dict) else x) for x in row[1:]], dtype=float
                )
                self.assertTrue(np.all(np.isfinite(vals)), f"{cid} rotated-step row must be finite")
                self.assertFalse(
                    np.allclose(vals, 0.0),
                    f"{cid} rotated-step row is all-zero (the pre-fix export-harness bug)",
                )


if __name__ == "__main__":
    unittest.main()
