"""Lock in the analytical-facade live-Wolfram PROVENANCE notes (modular_user_api
'expose analytical provenance' item). The facades remain honestly labeled
not-full-validated, but now cite the live comparison summaries that DO validate
their underlying symbolic building blocks -- and those summary files must exist."""

import json
import unittest
from pathlib import Path

from shaarp.api import run_ml_partial_analytical, run_si_full_analytical

ROOT = Path(__file__).resolve().parents[1]


class AnalyticalFacadeProvenanceTests(unittest.TestCase):
    def test_si_full_analytical_notes_cite_live_provenance_but_stay_unvalidated(self):
        result = run_si_full_analytical({})
        vs = result.validation
        # Still honestly NOT a full-validation claim:
        self.assertFalse(vs.is_mathematica_validated)
        self.assertIn("not_full_mathematica_validated", vs.status)
        notes = " ".join(vs.notes).lower()
        # ...but provenance of the validated building blocks is now surfaced:
        self.assertIn("live-wolfram", notes)
        self.assertIn("doldexp", notes)
        self.assertIn("computepnl", notes)

    def test_ml_partial_analytical_notes_cite_live_pnl_provenance(self):
        result = run_ml_partial_analytical({})
        vs = result.validation
        self.assertFalse(vs.is_mathematica_validated)
        notes = " ".join(vs.notes).lower()
        self.assertIn("live-wolfram", notes)
        self.assertIn("si_pnl_stage", notes)

    def test_cited_provenance_summary_files_exist_and_are_passing(self):
        """The provenance must point at REAL passing summaries, not vapor."""
        pnl = ROOT / "benchmarks" / "mathematica_reference" / "symbolic_pnl_stage_live_comparison_summary_v1.json"
        self.assertTrue(pnl.exists(), "cited symbolic PNL summary must exist")
        payload = json.loads(pnl.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "symbolic_pnl_matches_live_wolfram")
        self.assertTrue(payload["full_symbolic_pnl_agreement_claimed"])
        self.assertEqual(payload["failing_case_count"], 0)


if __name__ == "__main__":
    unittest.main()
