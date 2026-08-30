import unittest

from benchmarks.compare_symbolic_pnl_stage_live import build_summary


class SymbolicPnlStageLiveComparisonTests(unittest.TestCase):
    """Python's symbolic Voigt computePNL is validated against LIVE Wolfram
    SHAARP.si PNL-stage source polarizations (P2o same-mode, P2eo same-mode,
    Peoo mixed factor-2) over all 36 biaxial-anisotropic cases of
    shaarp_si_pnl_stage_reference_v1.json. Broader than the existing partial-PNL
    symbolic test (which used a few synthetic cases)."""

    def test_symbolic_pnl_matches_live_wolfram_all_cases(self):
        summary = build_summary(atol=1e-9, rtol=1e-9)
        self.assertEqual(summary["status"], "symbolic_pnl_matches_live_wolfram")
        self.assertTrue(summary["full_symbolic_pnl_agreement_claimed"])
        self.assertEqual(summary["case_count"], 36)
        self.assertEqual(summary["passing_case_count"], 36)
        self.assertEqual(summary["failing_case_count"], 0)
        self.assertLessEqual(summary["max_abs_error_over_all_cases"], 1e-9)

    def test_reference_is_live_wolfram_sourced_and_covers_full_trio(self):
        summary = build_summary(atol=1e-9, rtol=1e-9)
        source = str(summary.get("source", "")).lower()
        self.assertTrue("mathematica" in source or "wolfram" in source)
        for result in summary["results"]:
            self.assertEqual(set(result["components"]), {"P2o", "P2eo", "Peoo"})
            self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
